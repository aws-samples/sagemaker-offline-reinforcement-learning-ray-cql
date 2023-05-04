import sys, os
import math
import json
import torch
# import numpy as np
import logging
import gymnasium as gym


def list_files(startpath):
    for dirpath, _, files in os.walk(startpath):
        for x in files:
            logging.info(os.path.join(dirpath, x))

logging.info('listing files at opt/ml/model')
list_files('opt/ml/model')

def model_fn(model_dir):
    logging.info(f'model_dir: {model_dir}')
    logging.info('model_dir contents')
    list_files(model_dir)
    
    filename = os.path.join(model_dir,"model","model.pt")
    model = torch.load(filename)
    return model


def input_fn(request_body, request_content_type):
    logging.info('Request Body: ')
    logging.info(request_body)
    
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = {"obs":  torch.tensor(data, dtype=torch.float32)}
    return data


def predict_fn(input_object, model):
    logging.info(f'Making predictions on: {input_object}')
    with torch.no_grad():
        model_out, _ = model(input_object)
        action_output, _ = model.get_action_model_outputs(input_object)
        
        logging.info(f'action_output: {action_output}')
        
        ## Use this if the action_dist is TorchSquashedGausian https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_action_dist.py
        logging.info('Using TorchSquashedGausian Action Distribution')
        action_rec_mean = torch.atanh(torch.clip(action_output[:,0:1],min=-1, max=1)).tolist()
        action_rec_std = torch.exp(action_output[:,1:2]).tolist()
        
        ## Use this if the action_dist is TorchDiagGaussian https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_action_dist.py
        # logging.info('Using Torch Diag Gaussian Action Distribution')
        # action_rec_mean = action_output[:,0:1].tolist()
        # action_rec_std = torch.exp(action_output[:,1:2]).tolist()
        
        q_val, _ = model.get_q_values(
                        model_out, torch.tensor(action_rec_mean)
                    )
    
    return {
        "action_recommendation_mean": action_rec_mean,
        "action_recommendation_std": action_rec_std,
        "conservative_action_value":  q_val.tolist()
    }
    
    
def output_fn(predictions, content_type):
    logging.info(f'outputing predictions: {predictions}')
    
    return json.dumps(predictions, default = str)
