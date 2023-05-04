import sys, os
import json
from pprint import pprint

from hyperparameter_tuning_launcher_app import lambda_handler

dirname = os.path.dirname(__file__)
os.chdir(dirname)

with open('event.json') as f:
    event = json.load(f)

with open('environmental_vars.json') as f:
    env_vars = json.load(f)

pprint(env_vars)

for key, value in env_vars.get('Parameters').items():
    os.environ[key] = value

### Comment this line out for training in the cloud
os.environ['TRAIN_INSTANCE_TYPE'] = 'local'

print('Running the lambda function')
pprint(lambda_handler(event,{}))

