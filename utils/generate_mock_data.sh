set -e -o pipefail
source ./sam_functions.sh

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
  echo -ne "$i iterations\r"
  # The command below outputs to /dev/null because the output is not meant to be recorded or visualized.
  aws lambda invoke --function-name $lambda_name --invocation-type Event --payload "$event" /dev/null > /dev/null &
  sleep 0.2
  
  if (($i % 10 == 0));
  then
    currently_active_jobs=$(jobs -r | wc -l | xargs)
    if (($currently_active_jobs > 100));
    then
      echo "$currently_active_jobs currently active jobs"
      sleep 10
    fi
  fi
  
done
