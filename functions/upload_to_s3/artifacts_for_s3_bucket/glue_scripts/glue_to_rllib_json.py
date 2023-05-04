# # https://docs.ray.io/en/latest/rllib/rllib-offline.html#example-converting-external-experiences-to-batch-format
# # https://docs.ray.io/en/latest/rllib/rllib-offline.html#example-converting-external-experiences-to-batch-format
# import warnings
# # warnings.filterwarnings(action='once')
# warnings.filterwarnings('ignore')

import os
import sys
import random
import string
from datetime import datetime, timedelta
import tempfile
import argparse
import pandas as pd
import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import boto3
import awswrangler as wr
from awsglue.utils import getResolvedOptions
import gym
from tqdm import tqdm

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 50)

# STATES = ['SlugVolume10sAfterClose']
# ACTIONS = ['CsMinusLn']
# REWARD_COLUMN = ['AvgGasRate']

OUTPUT_DIR = os.path.join(tempfile.gettempdir(),'ray','ray_offline_data')

# https://docs.ray.io/en/latest/rllib/rllib-offline.html#example-converting-external-experiences-to-batch-format
def df_to_ray_json(episode_id: int, df: pd.DataFrame, state_columns: list, action_columns: list, reward_column: str, output_dir: str) -> None:
    """
    Input a pandas df with one row per timestep, and columns recording the state, action, and reward data.
    Save a json file in rllib format with the episode data
    """
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(output_dir)
    
    prev_actions = np.zeros_like(range(len(action_columns)))
    prev_reward = 0
    
    ### gym.spaces.Box type inputs have no pre-processor https://github.com/ray-project/ray/blob/master/rllib/models/preprocessors.py
    # This observation space is only used to form prep. The high and low values do not effect data creation.
    # observation_space = gym.spaces.Box(low = -100, high = 100, shape = (len(state_columns),))
    # observation_space = gym.spaces.Box(low = np.array([-4.8000002e+00,-3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
    #                                   high = np.array([4.8000002, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]))
    # prep = get_preprocessor(observation_space)(observation_space)
    
    first_row = df.iterrows()
    second_row = df.iterrows()
    _ = next(second_row) # move the second_row iterator to the second row in the df
    for (t, row), (_, next_row) in zip(first_row, second_row):
        #Mark done if on the second to last row of the df
        # We can't process the last row because we don't have reward info from taking the last action
        if t == df.shape[0]-2:
            terminated = True
        elif t == df.shape[0]-1:
            break
        else:
            terminated = False
        
        # print(f't = {t}')
        
        #Record the state information
        obs = np.array(row[state_columns].tolist())
        actions = np.array(row[action_columns].tolist())
        new_obs = np.array(next_row[state_columns].tolist())
        rew = np.float32(next_row[reward_column])
        
        truncated = False
        # new_obs, rew, terminated, truncated, info = env.step(action)
        batch_builder.add_values(
            t=t,
            eps_id=episode_id,
            agent_index=0,
            # obs=prep.transform(obs),
            obs=obs,
            actions=actions,
            action_prob=1.0,  # put the true action probability here
            action_logp=0.0,
            rewards=rew,
            prev_actions=prev_actions,
            prev_rewards=prev_reward,
            # terminateds=terminated,
            # truncateds=truncated,
            dones=terminated,
            infos={},
            # new_obs=prep.transform(new_obs),
            new_obs=new_obs
        )
        
        if t < 5:
            print(f"{t} current batch builder timestep")
            print(f"{row['time_step']} current row timestep")
            print(f"{next_row['time_step']} next row timestep")
            print(batch_builder.buffers)
        
        prev_actions = actions
        prev_reward = rew
        
        #Offline RL in ray requires each time step to be in it's own row. 
        writer.write(batch_builder.build_and_reset())
    
    #Sync record with s3 and clear tmp directory
    for file in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR,file)
        
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
                             
        #Upload the rllib data file to s3
        wr.s3.upload(local_file=file_path, path=f's3://{OUTPUT_BUCKET}/ray_offline_data/{random_string}-{file}')
        
        #Delete the file after it's uploaded to S3
        os.remove(file_path)

#########################
### Main starts here ####
#########################
if __name__ == '__main__':


    # https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python-calling.html?icmpid=docs_glue_studio_helppanel#aws-glue-programming-python-calling-parameters
    args = getResolvedOptions(sys.argv,
                              ['states',
                               'actions',
                               'reward_col',
                               'glue_db',
                               'glue_table',
                               'output_bucket',
                               'records_per_batch'
                               ])
    
    print('Input Args')
    print(args)
    
    STATES = args['states'].split()
    ACTIONS = args['actions'].split()
    REWARD_COLUMN = args['reward_col']
    GLUE_DB = args['glue_db']
    GLUE_TABLE = args['glue_table']
    OUTPUT_BUCKET = args['output_bucket']
    NUM_RECORDS_PER_BATCH = int(args['records_per_batch'])
    
    # For testing purposes
    # STATES = ['cart_position','cart_velocity','pole_angle','pole_angular_velocity']
    # ACTIONS = ['external_force']
    # REWARD_COLUMN = 'reward'
    # GLUE_DB = 'ol-rl-v0-0-0_historian_db'
    # GLUE_TABLE = 'measurements_table'
    # OUTPUT_BUCKET = 'ol-rl-v0-0-0-databucket-cgsox0x4oglk'
    # NUM_RECORDS_PER_BATCH = 1e3
    
    glue_client = boto3.client('glue')
    table_def = glue_client.get_table(
        DatabaseName= GLUE_DB,
        Name= GLUE_TABLE
    )
    
    num_columns = len(table_def.get('Table').get('StorageDescriptor').get('Columns'))
    
    # Get the unique episodes fore each device.
    unique_episodes_query = f"""
    SELECT device_id, episode_id
    FROM "{GLUE_DB}"."{GLUE_TABLE}" 
    GROUP BY device_id, episode_id
    """
    unique_episodes_db = wr.athena.read_sql_query(
        sql=unique_episodes_query,
        database = GLUE_DB
    )
    
    print(f'About to iterate through {unique_episodes_db.shape[0]} episodes')
    
    
    for i, (device_id, episode_id) in tqdm(unique_episodes_db.iterrows(), total=unique_episodes_db.shape[0]):
        print(device_id)
        print(episode_id)
        
        device_query = f"""
        SELECT *
        FROM "{GLUE_DB}"."{GLUE_TABLE}" 
        WHERE device_id='{device_id}' AND episode_id='{episode_id}'
        ORDER BY epoch_time ASC
        """
        try:
            device_query_db_iter = wr.athena.read_sql_query(
                sql=device_query,
                database = GLUE_DB,
                chunksize = NUM_RECORDS_PER_BATCH
            )
        except:
            print('Query failed to run')
            print(device_query)
            continue
        
        
        for device_query_db in device_query_db_iter:
            
            if device_query_db.shape[0] == 0:
                # print('Device query returned an empty df')
                # print(device_query)
                continue
            
            print('Head of timestep dataframe')
            print(device_query_db.head())
            
            df_to_ray_json(
                episode_id = episode_id,
                df = device_query_db,
                state_columns = STATES,
                action_columns = ACTIONS,
                reward_column = REWARD_COLUMN,
                output_dir = OUTPUT_DIR
            )
            
            
            
    print('Completed iterations. Ending job now.')