import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import base64
import json
import os
import numpy as np
import boto3
from datetime import datetime

from cart_pole_continuous import ContinuousCartPoleEnv
from physics_controller import PhysicsController

GOAL_CHANGE_PERIOD = 600 #Timesteps for goal high and hoal low cycle
FIREHOSE_UPLOAD_BATCH_SIZE=100 #How many records to upload to the firehose at one time

firehose_client = boto3.client('firehose')
sm_client = boto3.client('sagemaker-runtime')

# def choose_action(env,obs,goal_location,event,controller, epsilon, time_step):
#   if 'inference_endpoint' in event:
#     pass
#   else:
#     action = controller.apply_state_controller(obs[:4] - (goal_location,0,0,0)).astype(np.float32)/env.force_mag
  
  
#   action = np.clip(action, env.action_space.low, env.action_space.high)
  
#   # act randomly sometimes
#   if time_step%int(1./epsilon) == 0:
#     action = env.action_space.sample()

def lambda_handler(event, context):
  
  logger.info('event')
  logger.info(json.dumps(event, indent = 4).replace("\n", "\r")) #This prints as one event in cloudwatch
  
  # logger.info('environmental vars:')
  # logger.info(json.dumps(dict(os.environ), indent = 4).replace("\n", "\r"))
  EPSILON = event['random_action_fraction'] if 'random_action_fraction' in event else 0.5 #Fraction of the time to act randomly
  
  output_dict = {"steps":[]}
  
  env = ContinuousCartPoleEnv()
  env.theta_threshold_radians = 90 * np.pi / 180
  
  controller = PhysicsController(mk = env.masscart, mp = env.masspole, lp = env.length)
  
  logger.info(f'Action Space: {env.action_space}')
  
  # Initilize simulation variables
  obs = env.reset()
  goal_location = 0
  # action = env.action_space.sample()
  action = np.array([0.0], dtype=np.float32) # TO_DO: use the initial obs to find the initial aciton.
  reward = 0
  done = False
  info = {'First Timestep'}
  
  # This is over-written if a Value based RL model is used.
  inference_action_value = 0
  
  # logger.info(f'Action: {action}, type: {type(action)}')
  
  episode_id = datetime.now().isoformat()
  
  firehose_record_batch = []
  
  #The default action source is a linear quadratic regulator
  if 'inference_endpoint_name' in event:
    action_source = event['inference_endpoint_name']
  else:
    action_source = 'LQR'
  
  action_source += f'_epsilon={EPSILON}'
  
  for i in range(1000):
    # First choose an action
    if 'inference_endpoint_name' in event:
      data = json.dumps({"inputs": [obs.tolist()]})
      endpoint_response = sm_client.invoke_endpoint(
        EndpointName=event['inference_endpoint_name'],
        Body = data,
        ContentType='application/json',
        )
      model_output = json.loads(endpoint_response["Body"].read())
      action = np.array(model_output['action_recommendation_mean'][0], dtype = np.float32)
      inference_action_value = model_output['conservative_action_value'][0][0]
    else:
      action = controller.apply_state_controller(obs[:4] - (goal_location,0,0,0)).astype(np.float32)/env.force_mag
    
    
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    # act randomly sometimes
    if EPSILON != 0:
      if i%int(1./EPSILON) == 0:
        action = env.action_space.sample()
    
    # Then record the step info.
    #### x, x_dot, theta, theta_dot = obs
    step_data = {
        "cart_position": obs[0],
        "cart_velocity": obs[1],
        "pole_angle": obs[2],
        "pole_angular_velocity": obs[3],
        "goal_position": obs[4],
        "external_force": action[0].item(),
        "reward": reward,
        "done": done,
        "info": info,
        "device_id": 'lambda_function',
        "episode_id": episode_id,
        "epoch_time": datetime.now().timestamp(),
        "time_step": i,
        "action_source": action_source,
        "action_value": inference_action_value
      }
    
    output_dict["steps"].append(step_data)
    firehose_record_batch.append({
      'Data': json.dumps(step_data, default = str)
    })
    
    # Add records to the firehose delivery stream when a certain number of steps have been recorded.
    if i%FIREHOSE_UPLOAD_BATCH_SIZE == 0 and len(firehose_record_batch) > 0:
      firehose_put_response = firehose_client.put_record_batch(
        DeliveryStreamName=os.environ['DELIVERY_STREAM_NAME'],
        Records=firehose_record_batch
      )
      
      if firehose_put_response['FailedPutCount'] > 0:
        logger.warn('Some records failed to put')
        logger.warn(f'{firehose_put_response=}')
        logger.warn(json.dumps(firehose_put_response, default = str))
      
      firehose_record_batch = [] # reset the batch catcher
    
    
    # Change the goal location periodically
    if i%600 < 300: goal_location = 0.5
    else: goal_location = -0.5
    env.x_goal = goal_location
    
    # Take that action in the environment
    try:
      obs, reward, done, info = env.step(action)
    except Exception as e:
      print('Step Failed')
      print(f'obs: {obs}')
      print(f'action: {action}')
      raise e
    
    if done:
      break
  
  ## Put the remaining records in the firehose stream.
  if len(firehose_record_batch) > 0:
    firehose_client.put_record_batch(
          DeliveryStreamName=os.environ['DELIVERY_STREAM_NAME'],
          Records=firehose_record_batch
        )
  
  reward_sum = sum([step['reward'] for step in output_dict["steps"]])
  
  print(f'Successfully Completed Simulation. {len(output_dict["steps"])} total timesteps. {reward_sum:.2f} total reward. {reward_sum/len(output_dict["steps"]):.2f} average reward per timestep.')
  
  output_dict.update({"statusCode": 200})
  return json.loads(json.dumps(output_dict,default = str))
  
  
