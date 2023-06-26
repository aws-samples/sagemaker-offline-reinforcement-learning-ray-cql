set -e -o pipefail
source sam_functions.sh

ResourceId=TuningJobLauncherFunction

sam_config_dir=$(realpath '../samconfig.toml')

echo "Parsing config dir"

#This makes the sam config variables available as delcared variables
parse_config_vars $sam_config_dir

echo "Stack Name: '$stack_name'"

stack_resources=$(aws cloudformation describe-stack-resources --stack-name $stack_name)

lambda_name=$(echo $stack_resources | jq -r ".StackResources[] | select(.LogicalResourceId==\"$ResourceId\") | .PhysicalResourceId")

echo "Lambda Arn: $lambda_name"

# ### Invoke and view logs
aws lambda invoke --function-name $lambda_name --payload "$event" /dev/null  --log-type Tail --query 'LogResult' --output text |  base64 -d