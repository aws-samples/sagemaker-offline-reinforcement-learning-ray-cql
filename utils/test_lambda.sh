set -e -o pipefail
source sam_functions.sh

ResourceId=RunPhysicsSimulationFunction
event='{"random_action_fraction": 0.0, "inference_endpoint_name": "2359-003-bebc2abb-2023-05-03-22-04-38-560"}'
# event='{"random_action_fraction": 0.5}'

# ResourceId=S3UploadHandler

# ResourceId=TuningJobLauncherFunction
# event='{"input_model_uri": "s3://ol-rl-v0-0-0-assetsbucket-ngagfxwcupoc/training/offline-rl-2000-iter-230412-1210-012-6e2b2ed5/output/model.tar.gz"}'

# ResourceId=ModelDeployerFunction
# event=$(cat << EndOfMessage
# {
#   "DescribeTrainingJob": {
#     "ModelArtifacts": {
#       "S3ModelArtifacts": "s3://ol-rl-v0-0-0-assetsbucket-ngagfxwcupoc/training/offline-rl-1000-iter-230502-2359-003-bebc2abb/output/model.tar.gz"
#     }
#   }
# }
# EndOfMessage
# )

sam_config_dir=$(realpath '../samconfig.toml')

echo "Parsing config dir"

#This makes the sam config variables available as delcared variables
parse_config_vars $sam_config_dir

echo "Stack Name: '$stack_name'"

stack_resources=$(aws cloudformation describe-stack-resources --stack-name $stack_name)

lambda_name=$(echo $stack_resources | jq -r ".StackResources[] | select(.LogicalResourceId==\"$ResourceId\") | .PhysicalResourceId")

echo "Lambda Arn: $lambda_name"

### Invoke and view logs
aws lambda invoke --function-name $lambda_name --payload "$event" ../data/simulation.json  --log-type Tail --query 'LogResult' --output text |  base64 -d

# # Invoke with event
# lambda_response=$(aws lambda invoke --function-name $lambda_name --payload "$event" ../data/simulation.json )
# echo "lambda response"
# # echo $lambda_response | jq . 
# cat ../data/simulation.json | jq .

# ## Call the lambda x number of times
# for i in {1..5}
# do
#   # The command below outputs to /dev/stdin because the output is not meant to be recorded or visualized.
#   aws lambda invoke --function-name $lambda_name --payload "$event" /dev/stdin & #/dev/stdin, /dev/stdout and /dev/stderr
#   sleep 0.1
# done
