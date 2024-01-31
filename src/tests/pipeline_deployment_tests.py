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
"""Test training pipeline using local runner."""

import sys
import os

from tfx.v1.orchestration import LocalDagRunner

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
import logging

from src.tfx_pipelines import config
from src.tfx_pipelines import training_pipeline

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

MLMD_SQLLITE = "mlmd.sqllite"
NUM_EPOCHS = 1
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN_UNITS = "128,128"

test_instance = {
    "dropoff_grid": ["POINT(-87.6 41.9)"],
    "euclidean": [2064.2696],
    "loc_cross": [""],
    "payment_type": ["Credit Card"],
    "pickup_grid": ["POINT(-87.6 41.9)"],
    "trip_miles": [1.37],
    "trip_day": [12],
    "trip_hour": [16],
    "trip_month": [2],
    "trip_day_of_week": [4],
    "trip_seconds": [555],
}

SERVING_DEFAULT_SIGNATURE_NAME = "serving_default"

from google.cloud import aiplatform as vertex_ai

def test_e2e_pipeline():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    dataset_display_name = os.getenv("DATASET_DISPLAY_NAME")
    gcs_location = os.getenv("GCS_LOCATION")
    model_registry = os.getenv("MODEL_REGISTRY_URI")
    upload_model = os.getenv("UPLOAD_MODEL")
    endpoint_display_name = os.getenv("ENDPOINT_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert dataset_display_name, "Environment variable DATASET_DISPLAY_NAME is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"
    assert gcs_location, "Environment variable GCS_LOCATION is None!"
    assert model_registry, "Environment variable MODEL_REGISTRY_URI is None!"
    assert endpoint_display_name, "Environment variable ENDPOINT_DISPLAY_NAME is None!"
    
    logging.info(f"upload_model: {upload_model}")
    if tf.io.gfile.exists(gcs_location):
        tf.io.gfile.rmtree(gcs_location)
    logging.info(f"Pipeline e2e test artifacts stored in: {gcs_location}")

    if tf.io.gfile.exists(MLMD_SQLLITE):
        tf.io.gfile.remove(MLMD_SQLLITE)
    
    metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    metadata_connection_config.sqlite.filename_uri = MLMD_SQLLITE
    metadata_connection_config.sqlite.connection_mode = 3
    logging.info("ML metadata store is ready.")

    runner = LocalDagRunner()

    pipeline = training_pipeline._create_pipeline(
        pipeline_root=config.PIPELINE_ROOT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        metadata_connection_config=metadata_connection_config,
        
    )

    runner.run(pipeline)

    #logging.info(f"Model output: {os.path.join(model_registry, model_display_name)}")
    #assert tf.io.gfile.exists(os.path.join(model_registry, model_display_name))
    

    endpoints = vertex_ai.Endpoint.list(
        filter=f'display_name={endpoint_display_name}',
        order_by="update_time"
    )
    assert (
        endpoints
    ), f"Endpoint with display name {endpoint_display_name} does not exist! in region {region}"

    endpoint = endpoints[-1]
    logging.info(f"Calling endpoint: {endpoint}.")

    prediction = endpoint.predict([test_instance]).predictions[0]

    keys = ["classes", "scores"]
    for key in keys:
        assert key in prediction, f"{key} in prediction outputs!"

    assert (
        len(prediction["classes"]) == 2
    ), f"Invalid number of output classes: {len(prediction['classes'])}!"
    assert (
        len(prediction["scores"]) == 2
    ), f"Invalid number output scores: {len(prediction['scores'])}!"

    logging.info(f"Prediction output: {prediction}")

