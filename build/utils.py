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
"""Utilities for deploying pipelines and models to Vertex AI."""


import argparse
import os
import sys
import logging
import json

from google.cloud import aiplatform as vertex_ai
from src.tfx_pipelines import runner

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

SERVING_SPEC_FILEPATH = 'build/serving_resources_spec.json'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        type=str,
    )

    parser.add_argument(
        '--project',  
        type=str,
    )
    
    parser.add_argument(
        '--region',  
        type=str,
    )
    
    parser.add_argument(
        '--endpoint-display-name', 
        type=str,
    )

    parser.add_argument(
        '--model-display-name', 
        type=str,
    )
    
    parser.add_argument(
        '--pipeline-name', 
        type=str,
    )
    
    parser.add_argument(
        '--pipelines-store', 
        type=str,
    )

    return parser.parse_args()



def compile_pipeline():
    return runner.compile_training_pipeline()

def run_pipeline():
    return runner.submit_pipeline()


def main():
    args = get_args()
    
    if args.mode == 'compile-pipeline':
        result = compile_pipeline()
    elif args.mode == 'run-pipeline':
        result = run_pipeline()
    else:
        raise ValueError(f"Invalid mode {args.mode}.")
        
    logging.info(result)
        
    
if __name__ == "__main__":
    main()
    