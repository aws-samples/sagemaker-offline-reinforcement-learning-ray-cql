set -e -o pipefail
source sam_functions.sh

ResourceId=RunPhysicsSimulationFunction
event='{"random_action_fraction": 0.0, "inference_endpoint_name": "1859-004-dee358c6-2023-04-21-13-50-00-154"}'

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
