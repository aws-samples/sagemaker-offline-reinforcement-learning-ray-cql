#https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deployment.html
#https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html
import os
import json

import boto3
import random
import string
from datetime import datetime


import sagemaker
# from sagemaker.estimator import EstimatorBase
# from sagemaker.model import FrameworkModel
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def lambda_handler(event, context):
    """Deploys a serverless rllib model.

    Parameters
    ----------
    event: dict, required
        Input event to the Lambda function

    context: object, required
        Lambda Context runtime methods and attributes

    Returns
    ------
        None
    """
    
    print('event')
    print(json.dumps(event, indent = 4).replace("\n", "\r")) #This prints as one event in cloudwatch
    
    # print('environmental vars:')
    # print(json.dumps(dict(os.environ), indent = 2).replace("\n", "\r"))
    
    MODEL_DATA = event.get('DescribeTrainingJob').get('ModelArtifacts').get('S3ModelArtifacts')
    
    # uniqueId = "".join(random.choices(string.ascii_lowercase, k=5))
    uniqueId ="-".join(MODEL_DATA.split("/")[-3].split("-")[-3:])
    
    MODEL_DEPLOY_ROLE = os.environ['SM_MODEL_DEPLOY_ROLE']
    ENDPOINT_NAME = f'{uniqueId}'
    
    print(f'Deploing model from {MODEL_DATA} using role {MODEL_DEPLOY_ROLE}')
    
    #https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#create-an-estimator
    model = PyTorchModel(
        entry_point="inference.py",
        source_dir="src",
        role=MODEL_DEPLOY_ROLE,
        model_data=MODEL_DATA,
        framework_version="1.13.1",
        py_version="py39",
        name = ENDPOINT_NAME
    )
    
    # If model.bucket is blank, SM looks first in the default sagemaker bucket, which this function dosen't have access to.
    model.bucket = os.environ['MODEL_BUCKET']
    print(f'Deploying model from bucket {model.bucket}')
    
    if 'DEPLOY_INSTANCE_TYPE' in os.environ:
        # for debugging
        print('Deploying endpoint to instance.')
        predictor = model.deploy(
            initial_instance_count=1, 
            instance_type=os.environ['DEPLOY_INSTANCE_TYPE'], 
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
    
    predictor = model.deploy(
        serverless_inference_config=sagemaker.serverless.serverless_inference_config.ServerlessInferenceConfig(
            memory_size_in_mb=2048, max_concurrency=5),
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        wait = False,
        tags=[
                {
                    'Key':'training_job_model_data',
                    'Value':MODEL_DATA
                }
            ]
    )
    
    return f'Deployed model with endpoint name {ENDPOINT_NAME}'
    