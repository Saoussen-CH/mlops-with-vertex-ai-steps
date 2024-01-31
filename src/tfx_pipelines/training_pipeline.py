# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX training pipeline definition."""


import sys
import os
import json
import numpy as np
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils
import logging

from google.cloud import aiplatform

from src.common import features
from src.model_training import data
from src.tfx_pipelines import components

from tfx.proto import example_gen_pb2, transform_pb2, pusher_pb2

from tfx.v1.types.standard_artifacts import Model, ModelBlessing, Schema
from tfx.v1.dsl import Pipeline, Importer, Resolver, Channel
from tfx.v1.dsl.experimental import LatestBlessedModelStrategy, LatestArtifactStrategy

from tfx.v1.components import (
    StatisticsGen,
    SchemaGen,
    ImportSchemaGen,
    ExampleValidator,
    Transform,
    Evaluator,
    Pusher,
)

logging.getLogger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

print("TFX Version:", tfx.__version__)
print("Tensorflow Version:", tf.__version__)

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))
 
import os
import json
import numpy as np
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils
import logging
import sys
from google.cloud import aiplatform


from src.model_training import data


from tfx.proto import example_gen_pb2, transform_pb2, pusher_pb2

from tfx.v1.types.standard_artifacts import Model, ModelBlessing, Schema
from tfx.v1.dsl import Pipeline, Importer, Resolver, Channel
from tfx.v1.dsl.experimental import LatestBlessedModelStrategy, LatestArtifactStrategy

from tfx.v1.components import (
    StatisticsGen,
    SchemaGen,
    ImportSchemaGen,
    ExampleValidator,
    Trainer,
    Transform,
    Evaluator,
    Pusher,
)


from tfx.v1.extensions.google_cloud_ai_platform import Trainer as VertexTrainer 
from tfx.v1.extensions.google_cloud_ai_platform import Pusher as VertexPrediction 

from tfx.orchestration import pipeline, data_types
from src.tfx_pipelines import config, components
from src.common import features, datasource_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

RAW_SCHEMA_DIR = "src/tfx_model_training/raw_schema"
TRANSFORM_MODULE_FILE = "src/preprocessing/transformations.py"
TRAIN_MODULE_FILE = "src/tfx_model_training/model_runner.py"

def _create_pipeline(
    pipeline_root: str,
    num_epochs: data_types.RuntimeParameter,
    batch_size: data_types.RuntimeParameter,
    learning_rate: data_types.RuntimeParameter,
    metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,
)-> tfx.dsl.Pipeline:

    # Hyperparameter generation.
    hyperparams_gen = components.hyperparameters_gen(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    ).with_id("HyperparamsGen")
    
    beam_pipeline_args = config.BEAM_DIRECT_PIPELINE_ARGS
    if config.BEAM_RUNNER == "DataflowRunner":
        beam_pipeline_args = config.BEAM_DATAFLOW_PIPELINE_ARGS

    # Get train source query.
    train_sql_query = datasource_utils.get_training_source_query(
        project=config.PROJECT,
        region=config.REGION,
        dataset_display_name=config.DATASET_DISPLAY_NAME,
        ml_use="UNASSIGNED",
        limit=int(config.TRAIN_LIMIT),
    )

    train_output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=int(config.NUM_TRAIN_SPLITS)
                ),
                example_gen_pb2.SplitConfig.Split(
                    name="eval", hash_buckets=int(config.NUM_EVAL_SPLITS)
                ),
            ]
        )
    )

    # Train example generation
    
    train_example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=train_sql_query,
        output_config=train_output_config,
        custom_config=json.dumps({})
    ).with_beam_pipeline_args(beam_pipeline_args).with_id("TrainDataGen")

    # Get test source query.
    test_sql_query = datasource_utils.get_training_source_query(
        config.PROJECT,
        config.REGION,
        config.DATASET_DISPLAY_NAME,
        ml_use="TEST",
        limit=int(config.TEST_LIMIT),
    )

    test_output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="test", hash_buckets=1),
            ]
        )
    )

    # Test example generation.
    test_example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=test_sql_query,
        output_config=test_output_config,
        custom_config=json.dumps({})
    ).with_beam_pipeline_args(beam_pipeline_args).with_id("TestDataGen")

    # Schema importer.
    schema_importer = Importer(
        source_uri=RAW_SCHEMA_DIR,
        artifact_type=Schema,
    ).with_id("SchemaImporter")


    # Statistics generation.
    statistics_gen = StatisticsGen(examples=train_example_gen.outputs["examples"]).with_id(
        "StatisticsGen"
    )

    # Example validation.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_importer.outputs["result"],
    ).with_id("ExampleValidator")

    # Data transformation.
    transform = Transform(
        examples=train_example_gen.outputs["examples"],
        schema=schema_importer.outputs["result"],
        module_file=TRANSFORM_MODULE_FILE,
        # This is a temporary workaround to run on Dataflow.
        force_tf_compat_v1=config.BEAM_RUNNER == "DataflowRunner",
        splits_config=transform_pb2.SplitsConfig(
            analyze=["train"], transform=["train", "eval"]
        ),
    ).with_id("DataTransformer")

    # Add dependency from example_validator to transform.
    transform.add_upstream_node(example_validator)

    # Get the latest model to warmstart
    warmstart_model_resolver = Resolver(
        strategy_class=LatestArtifactStrategy,
        model=Channel(type=Model),
        ).with_id("WarmstartModelResolver")
    
    # Add the ImportSchemaGen so that it will add it to the GCS bucket
    schema_gen = ImportSchemaGen(schema_file='src/tfx_model_training/raw_schema/schema.pbtxt')
    
    # Model training.
    if config.TRAINING_RUNNER == "vertex":
        trainer = VertexTrainer(
            module_file=TRAIN_MODULE_FILE,
            examples=transform.outputs["transformed_examples"],
            schema=schema_gen.outputs['schema'],
            base_model=warmstart_model_resolver.outputs["model"],
            transform_graph=transform.outputs["transform_graph"],
            train_args=tfx.proto.TrainArgs(num_steps=10),
            eval_args=tfx.proto.EvalArgs(num_steps=5),
            hyperparameters=hyperparams_gen.outputs["hyperparameters"],
            custom_config=config.VERTEX_TRAINING_CONFIG
        ).with_id("ModelTrainer")
    else :
        trainer = Trainer(
            module_file=TRAIN_MODULE_FILE,
            examples=transform.outputs["transformed_examples"],
            schema=schema_importer.outputs["result"],
            base_model=warmstart_model_resolver.outputs["model"],
            train_args=tfx.proto.TrainArgs(num_steps=10),
            eval_args=tfx.proto.EvalArgs(num_steps=5),
            transform_graph=transform.outputs["transform_graph"],
            hyperparameters=hyperparams_gen.outputs["hyperparameters"],
        ).with_id("ModelTrainer")
        

    # Get the latest blessed model (baseline) for model validation.
    baseline_model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id("BaselineModelResolver")

    # Prepare evaluation config.
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_tf_example",
                label_key=features.TARGET_FEATURE_NAME,
                prediction_key="probabilities",
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": float(config.ACCURACY_THRESHOLD)}
                            ),
                            # Change threshold will be ignored if there is no
                            # baseline model resolved from MLMD (first run).
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )

    # Model evaluation.
    evaluator = Evaluator(
        examples=test_example_gen.outputs["examples"],
        example_splits=["test"],
        model=trainer.outputs["model"],
        baseline_model=baseline_model_resolver.outputs["model"],
        eval_config=eval_config,
        schema=schema_gen.outputs['schema'],
    ).with_id("ModelEvaluator")

    ########################## Use this code to push the model to Model Registery #########
    #exported_model_location = os.path.join(
    #    config.MODEL_REGISTRY_URI, config.MODEL_DISPLAY_NAME
    #)
    #push_destination = pusher_pb2.PushDestination(
    #    filesystem=pusher_pb2.PushDestination.Filesystem(
    #        base_directory=exported_model_location
    #    )
    #)

    # Push custom model to model registry.
    #pusher = Pusher(
    #    model=trainer.outputs["model"],
    #    model_blessing=evaluator.outputs["blessing"],
    #    push_destination=push_destination,
    #).with_id("ModelPusher")
    
    # Upload custom trained model to Vertex AI.
    #labels = {
    #    "dataset_name": config.DATASET_DISPLAY_NAME,
    #    "pipeline_name": config.PIPELINE_NAME,
    #    "pipeline_root": pipeline_root
    #}
    #labels = json.dumps(labels)
    
    #explanation_config = json.dumps(features.generate_explanation_config())
    
    #vertex_model_uploader = custom_components.vertex_model_uploader(
    #   project=config.PROJECT,
    #    region=config.REGION,
    #    model_display_name=config.MODEL_DISPLAY_NAME,
    #    pushed_model_location=exported_model_location,
    #    serving_image_uri=config.SERVING_IMAGE_URI,
    #    model_blessing=evaluator.outputs["blessing"],
    #    explanation_config=explanation_config,
    #    labels=labels
    #).with_id("VertexUploader")

    pusher = VertexPrediction(
        model=trainer.outputs['model'],
        model_blessing = evaluator.outputs['blessing'],
        custom_config=config.VERTEX_PREDICTION_CONFIG)
   
    pipeline_components = [
        hyperparams_gen,
        train_example_gen,
        test_example_gen,
        statistics_gen,
        schema_importer,
        example_validator,
        transform,
        warmstart_model_resolver,
        schema_gen,
        trainer,
        baseline_model_resolver,
        evaluator,
        pusher,
    ]

    #if int(config.UPLOAD_MODEL):
    #    pipeline_components.append(vertex_model_uploader)
    #    # Add dependency from pusher to aip_model_uploader.
    #    vertex_model_uploader.add_upstream_node(pusher)

    logging.info(
        f"Pipeline components: {[component.id for component in pipeline_components]}"
    )



    logging.info(f"Beam pipeline args: {beam_pipeline_args}")

    return Pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config,
        enable_cache=True,
    )
