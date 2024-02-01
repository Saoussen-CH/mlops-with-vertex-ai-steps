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
"""Define KubeflowV2DagRunner to run the training pipeline using Managed Pipelines."""


import os
from tfx import v1 as tfx
from tfx.orchestration import data_types
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs


from src.tfx_pipelines import config, training_pipeline, prediction_pipeline
from src.tfx_model_training import defaults


def compile_training_pipeline():

    managed_pipeline =  training_pipeline._create_pipeline(
           pipeline_root=config.PIPELINE_ROOT,
            num_epochs=data_types.RuntimeParameter(
                name="num_epochs",
                default=defaults.NUM_EPOCHS,
                ptype=int,
            ),
            batch_size=data_types.RuntimeParameter(
                name="batch_size",
                default=defaults.BATCH_SIZE,
                ptype=int,
            ),
            learning_rate=data_types.RuntimeParameter(
                name="learning_rate",
                default=defaults.LEARNING_RATE,
                ptype=float,
            ),

        )

    # Path to various pipeline artifact.
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(default_image=config.TFX_IMAGE_URI),
        output_filename=config.PIPELINE_DEFINITION_FILE)

    return runner.run(managed_pipeline)


def submit_pipeline():
    
    aiplatform.init(project=config.PROJECT, location=config.REGION)
    
    # Create a job to submit the pipeline
    job = pipeline_jobs.PipelineJob(template_path=config.PIPELINE_DEFINITION_FILE,
                                    display_name=config.PIPELINE_NAME,
                                    parameter_values={
                                        'learning_rate': 0.003,
                                        'batch_size': 512,
                                        'num_epochs': 30,
                                    })
    job.submit()