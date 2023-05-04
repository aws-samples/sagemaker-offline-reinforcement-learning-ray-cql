


Use this repo like this:

```
  WriteToFirehoseRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: 2012-10-17
        Statement:
          - Effect: Allow
            Action: sts:AssumeRole
            Principal:
              Service:
                - iot.amazonaws.com
      Policies: 
          - PolicyName: FirehoseWrite
            PolicyDocument:
              Version: '2012-10-17'
              Statement:
                - Sid: AllowFirehoseWrite
                  Effect: Allow
                  Action:
                    - "firehose:PutRecord"
                  Resource: 
                    - !GetAtt MeasurementsFirehose.Outputs.DeliveryStreamArn
                    - !GetAtt ActionsFirehose.Outputs.DeliveryStreamArn


  MeasurementsFirehose:
    Type: AWS::Serverless::Application
    Properties:
      Location: nested_stacks/firehose_to_s3/template.yaml
      Parameters:
        OutputBucketArn: !GetAtt DataBucket.Arn
        MetadataExtractionQueryString: >
          {
          device_id:  .DeviceId,
          year:       .EpochTime| strftime("%Y"),
          month:      .EpochTime| strftime("%m"),
          day:        .EpochTime| strftime("%d"),
          hour:       .EpochTime| strftime("%H")
          }
        FirehoseOutputS3Prefix: > 
          device/!{partitionKeyFromQuery:device_id}/
          type/measurement/
          year/!{partitionKeyFromQuery:year}/
          month/!{partitionKeyFromQuery:month}/
          day/!{partitionKeyFromQuery:day}/
          hour/!{partitionKeyFromQuery:hour}/


  StoreMeasurments:
    Type: AWS::IoT::TopicRule
    Properties: 
      TopicRulePayload: 
          Actions: 
            - Firehose:
                DeliveryStreamName: !GetAtt MeasurementsFirehose.Outputs.DeliveryStreamName
                RoleArn: !GetAtt WriteToFirehoseRole.Arn
                Separator: "\n"

```


