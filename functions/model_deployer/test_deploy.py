import os

from model_deployer_app import lambda_handler

import json
from pprint import pprint

dirname = os.path.dirname(__file__)
os.chdir(dirname)

with open('event.json') as f:
    event = json.load(f)

with open('environmental_vars.json') as f:
    env_vars = json.load(f)

for key, value in env_vars.get('Parameters').items():
    os.environ[key] = value

#Comment this line out to use an instince in the cloud
# os.environ['DEPLOY_INSTANCE_TYPE'] = 'local'
# os.environ['DEPLOY_INSTANCE_TYPE'] = 'ml.m6g.large'

print('Running the lambda function')
pprint(lambda_handler(event,{}))