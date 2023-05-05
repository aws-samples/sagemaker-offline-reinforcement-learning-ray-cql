import cfnresponse

import shutil
import json, os
from pprint import pprint
import traceback

import boto3

s3_client = boto3.client('s3')

def lambda_handler(event, context):
  """This lambda function uploads files in the "artifacts_fo_s3_bucket" folder to s3

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
  
  if 'RequestType' not in event: event['RequestType'] ='Create' # Default to create
  
  result = cfnresponse.SUCCESS
  
  try:
    if event['RequestType']=='Create':
      # bucket = event['ResourceProperties']['BucketName']
      
      print('local files')
      print(os.listdir())
      
      # os.chdir('artifacts_for_s3_bucket')
      
      root_path = './artifacts_for_s3_bucket/'
      for root, dirs, files in os.walk(root_path):
          prefix = root[len(root_path):]
          for file in files:
              print(f'file: {os.path.join(root,file)}')
              
              key = os.path.join(prefix,file)
              
              print(f'key: {key}')
              
              s3_client.upload_file(os.path.join(root,file), os.environ['BUCKET'], key)
      
      
  except Exception as err:
    result = cfnresponse.FAILED
    print('Failed to upload')
    print(err)
    print(traceback.format_exc())

  try:
    cfnresponse.send(event, context, result, {})
  except Exception as e:
    print(f'Could not sent cfn response due to : {e}')
  
  return "function successfully exited"
  