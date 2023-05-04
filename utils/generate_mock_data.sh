set -e -o pipefail
source sam_functions.sh

ResourceId=RunPhysicsSimulationFunction
event='{"random_action_fraction": 0.5}'

sam_config_dir=$(realpath '../samconfig.toml')

echo "Parsing config dir"

#This makes the sam config variables available as delcared variables
parse_config_vars $sam_config_dir

echo "Stack Name: '$stack_name'"

stack_resources=$(aws cloudformation describe-stack-resources --stack-name $stack_name)

lambda_name=$(echo $stack_resources | jq -r ".StackResources[] | select(.LogicalResourceId==\"$ResourceId\") | .PhysicalResourceId")

echo "Lambda Arn: $lambda_name"

## Call the lambda x number of times
for i in {1..2000}
do
  # The command below outputs to /dev/stdin because the output is not meant to be recorded or visualized.
  aws lambda invoke --function-name $lambda_name --payload "$event" /dev/stdin & #/dev/stdin, /dev/stdout and /dev/stderr
  sleep 0.01
done
