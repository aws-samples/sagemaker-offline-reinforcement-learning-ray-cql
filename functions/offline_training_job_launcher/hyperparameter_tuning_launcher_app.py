import sys, os, json
from pprint import pprint
from datetime import datetime
import logging

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.session import Session
from sagemaker.local import LocalSession
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, CollectionConfig

from sagemaker.pytorch import PyTorch

from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

from sagemaker.tuner import ContinuousParameter, IntegerParameter, HyperparameterTuner

import boto3

from pprint import pprint

def lambda_handler(event, context):
    """Sample Lambda function which mocks the operation of buying a random number
    of shares for a stock.

    For demonstration purposes, this Lambda function does not actually perform any 
    actual transactions. It simply returns a mocked result.

    Parameters
    ----------
    event: dict, required
        Input event to the Lambda function
        - max_tuning_jobs
        
    context: object, required
        Lambda Context runtime methods and attributes

    Returns
    ------
        dict: Object containing details of the tuning job
    """
    
    print('event')
    print(json.dumps(event, indent = 4).replace("\n", "\r")) #This prints as one event in cloudwatch
    
    # create a descriptive job name
    job_name_prefix = f"offline-rl-{os.environ['TRAINING_ITERATIONS']}-iter"
    
    
    if 'input_model_uri' in event:
        input_model_uri = event['input_model_uri']
    else:
        input_model_uri = None
        logging.info('No input_model_uri in the input event')
        
    
    train_instance_type = os.environ["TRAIN_INSTANCE_TYPE"]
    training_bucket = os.environ["SAGEMAKER_TRAINING_BUCKET"]
    s3_output_path = f"s3://{training_bucket}/training/"
    s3_checkpoint_path = f"s3://{training_bucket}/checkpoints/"
    # tuner_objective_metric = 'sum_actor_critic_loss'
    # tuner_objective_metric = 'critic_loss'
    tuner_objective_metric='td_mse'
    
    hyperparameter_ranges = {
        # "actor_learning_rate": ContinuousParameter(1e-5, 1e-3), #3e-4 is the default
        "critic_learning_rate": ContinuousParameter(1e-3, 1e-1),
        # "number_of_layers": IntegerParameter(1,8),
        # "unused_dummy_var": ContinuousParameter(1e-5, 1e-1), 
    }
    
    if train_instance_type == 'local':
        sm_session = LocalSession()
    else:
        sm_session = Session()
    
    sm_role = os.environ['SAGEMAKER_TRAINING_ROLE']
    
    print(f'Using role: {sm_role}')

    metric_names=[
        'training_iteration',
        'iterations_since_restore',
        # 'total_training_iterations',
        'timesteps_total',
        'time_total_s',
        'num_grad_updates_lifetime',
        'actor_loss',
        'critic_loss',
        'cql_loss',
        'td_error',
        'td_mse',
        'sum_actor_critic_loss',
        'validation_mean_q',
        'validation_critic_loss',
        'validation_actor_loss',
        'validation_cql_loss',
        'validation_td_mse',
        'min_q',
        'mean_q',
        'max_q'
        ]
    
    metric_definitions = [{'Name': name, 'Regex': '"' + name + '":(.*?)[,}]'} for name in metric_names] # This regular expression will catch JSON values assocated with the "name" key
    
    
    print('Metric Definitions')
    print(json.dumps(metric_definitions, indent = 2).replace("\n", "\r")) #This prints as one event in cloudwatch
    
    offline_training_estimator_parameters = {
        # "entry_point":"offline_train_and_export_cql_old.py",
        "entry_point":"train_and_export_cql.py",
        "source_dir":"offline_training",
        "base_job_name": job_name_prefix,
        
        "framework_version":"1.12.1",
        "py_version":"py38",
        # "image_uri": os.environ['TRAINING_IMAGE_URI'],

        "role":sm_role,
        
        "instance_type":train_instance_type,
        "instance_count":1,
        "output_path":s3_output_path,
        "metric_definitions":metric_definitions,
        "hyperparameters":{
            "algorithm": "CQL",
            "training_iterations": int(os.environ['TRAINING_ITERATIONS']),
            "actor_learning_rate": 3e-3, #3e-4 is the default
            # "critic_learning_rate": 3e-4,
            "number_of_layers": 2,
            "seed": 1,
            "number_of_state_variables": len(os.environ['STATES'].split(",")),
            "number_of_actions": len(os.environ['ACTIONS'].split(","))
        },
        
        #Sagemaker Info
        "model_uri": input_model_uri,
        
        "sagemaker_session":sm_session,
        "tags": [
            {
                'Key': 'Info', 
                'Value': 'Initial Tests' 
            }
        ],
        
        #Debugging
        # "debugger_hook_config":False,
        # "rules": [ProfilerRule.sagemaker(rule_configs.ProfilerReport())],
        # "keep_alive_period_in_seconds": 300, # This is just for debugging. It lets you spin up subsiquent training jobs faster when deploying to the cloud.
    }
    
    if train_instance_type != 'local':
        offline_training_estimator_parameters.update({
            "checkpoint_s3_uri": s3_checkpoint_path,
            # "use_spot_instances":True,
            # "max_wait": 24 * 60 * 60 * 2, #Seconds
        })
    
    print('Estimator Parameters')
    print(json.dumps(
        offline_training_estimator_parameters, 
        indent = 4,
        default = lambda obj: f'<{str(obj)}, type:{type(obj)} not serializable>'
        ).replace("\n", "\r"))
    
    offline_rl_estimator = PyTorch(**offline_training_estimator_parameters)
    
    inputs = {
        # "train": event['s3_data_loc'],
        "train": os.environ['DATA_LOCATION'],
    }
    
    # inputs = TrainingInput(s3_data=event['s3_data_loc'])
    
    print(f'Training on inputs: {inputs}')
    
    # https://github.com/aws/amazon-sagemaker-examples/blob/main/hyperparameter_tuning/tensorflow2_mnist/hpo_tensorflow2_mnist.ipynb
    
    if train_instance_type == 'local':
        #Locally train the estimator as a test. This will never run outside Dev.
        offline_rl_estimator.fit(inputs = inputs)
        return "Successfully tested the estimator"
        
    # ## Test out a training job
    # offline_rl_estimator.fit(inputs = inputs)
    # return 'done testing'
    
    tuner = HyperparameterTuner(
        offline_rl_estimator,
        objective_metric_name=tuner_objective_metric,
        hyperparameter_ranges = hyperparameter_ranges,
        metric_definitions = metric_definitions,
        max_jobs= int(os.environ['MAX_TUNING_JOBS']),
        max_parallel_jobs=4,
        objective_type="Minimize",
        base_tuning_job_name=job_name_prefix,
        # early_stopping_type = 'Auto'
    )
    
    
    tuner.fit(
        inputs = inputs,
        wait=False
        )
    
    tuning_job_name = tuner.latest_tuning_job.name
    
    print(f'Tuning Job Name: {tuning_job_name}')
    
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            'TuningJobName':tuning_job_name
        }),
    }
    