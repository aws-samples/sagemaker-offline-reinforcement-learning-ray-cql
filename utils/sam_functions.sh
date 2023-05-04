# echo "Hello from functions.sh"

parse_config_vars(){
    [[ -f $1 ]] || { echo "$1 is not a file." >&2;return 1;}
    local line key value entry_regex
    entry_regex="^[[:blank:]]*([[:alnum:]_-]+)[[:blank:]]*=[[:blank:]]*('[^']+'|\"[^\"]+\"|[^#]+)"
    while read -r line
    do
        [[ -n $line ]] || continue
        [[ $line =~ $entry_regex ]] || continue
        key=${BASH_REMATCH[1]}
        value=${BASH_REMATCH[2]#[\'\"]} # strip quotes
        value=${value%[\'\"]}
        value=${value%${value##*[![:blank:]]}} # strip trailing spaces
        declare -g "${key}"="${value}"
    done < "$1"
}

get_resource_arn(){
    # Call like this
    # $get_resource_arn logical_id stack_name
    
    # First get the correct S3 bucket
    logical_id_filter=$( jq -n \
      --arg logical_id "$1" \
      '{Key: "aws:cloudformation:logical-id", Values: [$logical_id]}' )
    
    stack_name_filter=$( jq -n \
      --arg stack_name "$2" \
      '{Key: "aws:cloudformation:stack-name", Values: [$stack_name]}' )
    
    
    tag_response=$(aws resourcegroupstaggingapi get-resources \
        --tag-filters "[$logical_id_filter, $stack_name_filter]" \
        )
    
    
    num_resources=$(echo "$tag_response" | jq '.ResourceTagMappingList | length')
    
    # echo "Found $num_resources resource(s) satisfing the tagging filters"
    
    resource_arn=$(echo "$tag_response" | jq -r '.ResourceTagMappingList[0].ResourceARN')
    
    echo "$resource_arn"
}

# echo "Done importing functions.sh"