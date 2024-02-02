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
"""Test an uploaded model to Vertex AI."""

import os
import logging
import tensorflow as tf

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


def test_model_endpoint():

    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    endpoint_display_name = os.getenv("ENDPOINT_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert region, "Environment variable REGION is None!"
    assert model_display_name, "Environment variable MODEL_DISPLAY_NAME is None!"
    assert endpoint_display_name, "Environment variable ENDPOINT_DISPLAY_NAME is None!"

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
