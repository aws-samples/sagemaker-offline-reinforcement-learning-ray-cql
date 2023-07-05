# Optimize systems using historic data and machine learning

Efficient control policies enable industrial companies to increase their profitability by maximizing productivity while reducing unscheduled downtime and energy consumption. Finding optimal control policies is a complex task because physical systems, such as chemical reactors and wind turbines, are often hard to model and because drift in process dynamics can cause performance to deteriorate over time. Offline reinforcement learning is a control strategy that allows industrial companies to build control policies using only historical data, without the need for an explicit process model. This approach does not require interaction with the process directly in an exploration stage, which has prevented the adoption of reinforcement learning in safety-critical applications. This repo is an example end-to-end solution to find optimal control policies using only historical data on Amazon SageMaker using Ray’s RLlib library.

This code sample shows how to train an optimized control policy from historic data using the [Conservative Q Learning](https://sites.google.com/view/cql-offline-rl) (CQL) reinforcement learning method.

CQL finds optimized actions based on data you already have, and conservativly estimates it's performance after taking that action.

You can compare CQL's performance estimate with historic system performance for an indication of if CQL will perform better than the existing control system.


## Architecture
![Diagram](assets/offline_rl_architecture.svg)

## Example simulated training data and trained model episodes
To show how this process works, this repo contains a sample system to optimize using historic data. The sample system is a cart-pole environment which contains a pole balanced on top of a cart. The goal in this system is to move the cart to the green box, while keeping the pole upright. The historic data was generated using 50% expert actions and 50% random actions. The CQL algorithm learns which actions tend to produce higher rewards, and then recommends those actions. The animation on the left shows an example episode from the historic data. On the right we see an episode where the CQL model's action recommnedations were followed. The CQL model learned a control policy that moves the cart quickly to the goal position, while maintaining stability.

![training-data](assets/cartpole_training_data.gif "episode from historic data") ![trained-model-episode](assets/cartpole_trained_model.gif "episode using trained model")


## Steps to use this project
1. Deploy the template using the [Serverless Application Module](https://aws.amazon.com/serverless/sam/). Use the command `sam build --use-container` followed by `sam deploy --guided` from this project's root directory.
2. Generate mock historic data by running the 'generate_mock_data.sh' script: `sh ./assets/generate_mock_data.sh`. This script runs a simulation 2000 times using an AWS Lambda function. The results of those simulation runs are put into a Kinesis Firehose which delivers the data to S3. The template contains a glue table for this data, which allows the AWS Glue ETL job to process the data.
3. You can now view the measurement data using Amazon Athena. Look for the Glue database named `<CloudFormation stack name>_historian_db`.
4. Transform data from your measurement table into JSON format. Look in the outputs seciton of your AWS CloudFormation stack for the query named "AthenaQueryToCreateJsonFormatedData". Run this query with Amazon Athena to create the formatted data.
5. Now the S3 bucket `<CloudFormation stack name>-databucket-<uniquie id>` will have a `ray_offline_data` prefix with the pre-processed data files. These data files contain json objects with information about each time step's observtions (measurement values), actions, and rewards.
6. Train a ML model by invoking the lambda function named `<CloudFormation stack name>-TuningJobLauncherFunction-<uniquie id>`.
7. View the results of the tuning jobs in Sagemaker Studio. Select a tuning run based on criteria in [this paper](https://arxiv.org/abs/2109.10813). Look for a training run where the `critic_loss` stays low and the `mean_q` is high.
8. Find the "S3 model artifact" for the favored training job. Supply it to the Lambda function named `<CloudFormation stack name>-ModelDeployerFunction-<uniquie id>`. This will create a serverless SageMaker endpoint to serve the trained model.
9. Now call the Lambda function named `<CloudFormation stack name>-RunPhysicsSimulationFunction-<uniquie id>` with the endpoint name in the event. This will run the physic simulation using the model you just trained.
10. The results of this Lambda invocation will also be stored in the S3 bucket `<CloudFormation stack name>-databucket-<uniquie id>`. You can compare the performance of the trained model with the performance in the training data using the sample Athena query below.


### Sample Athena Query
``` sql
WITH 
    sum_reward_by_episode AS (
        SELECT SUM(reward) as sum_reward, COUNT(1) as timesteps, m_temp.episode_id, m_temp.action_source
        FROM "<CloudFormation stack name>_historian_db"."measurements_table" m_temp
        GROUP BY m_temp.episode_id, m_temp.action_source
        )

SELECT sre.action_source, AVG(sre.sum_reward) as avg_total_reward_per_episode, AVG(timesteps) AS avg_timesteps_per_episode, SUM(CAST(timesteps=1000 AS DOUBLE))/COUNT(1) AS p_max_time, SUM(timesteps) AS total_timesteps, COUNT(1) as total_episodes
FROM  sum_reward_by_episode sre
GROUP BY sre.action_source
ORDER BY avg_total_reward_per_episode DESC
```

## Detailed Walkthrough
This project first generates mock historic data by running a simulation in a lambda function. The simulated environment is cart-pole, but with force control instead of the usual binary velocity control. This environment was chosen because: 1/ The observations and actions are continuous values, 2/ A gif can communicate the performance of an actor on the system, 3/ The system has an unstable equlibrium point. Code for the simulated environment is stored in a lambda layer (source code at `functions/simulation_layer/`). When the lambda function `RunPhysicsSimulationFunction` is called, every x time steps the time step data is put into a Kinesis Firehose which stores it in S3. The template contains a glue table `GlueMeasurementsTable` which lets Amazon Athena query the data. To generate data for the offline rl workflow, call the lambda function a number of times. The script `./assets/generate_mock_data.sh` automates the process of calling the simulation lambda funciton 2000 times. After calling the `generate_mock_data.sh` file, wait 1 minute, and then you can query the simulation timestep data using Athena. 

Companies with data lakes may already have a Glue table with measurement data. If a company has historic process data outside of a data lake, check out [AWS Lake Formation](https://aws.amazon.com/lake-formation/) to move data into a data lake.

Now that historic measurement data is stored in S3, we can use an [Amazon Athena Unload Operation](https://docs.aws.amazon.com/athena/latest/ug/unload.html) to transform the data into a format approprate for traing a reinforcement learning model. In this sample the Ray rllib library is used. For the rllib library, you store timestep data as json objects. The object below shows a few example timesteps.
```json
{"type":"SampleBatch","episode_id":["2023-04-11T21:01:17.786965"],"unroll_id":[911876],"obs":[[0.4212790641025132,1.033157927079702,0.0016930879401219566,-0.09199378805520442,0.5]],"new_obs":[[0.44194222264410726,1.1364661110881928,-1.4678782098213167E-4,-0.24645807434810413,0.5]],"actions":[[0.17652631]],"rewards":[11.982922835931035],"dones":[false]}
{"type":"SampleBatch","episode_id":["2023-04-11T21:01:17.786965"],"unroll_id":[911848],"obs":[[0.44194222264410726,1.1364661110881928,-1.4678782098213167E-4,-0.24645807434810413,0.5]],"new_obs":[[0.46467154486587114,1.2977569804286362,-0.005075949307944214,-0.4884375313715229,0.5]],"actions":[[0.275535]],"rewards":[11.870076775558045],"dones":[false]}
{"type":"SampleBatch","episode_id":["2023-04-11T21:01:17.786965"],"unroll_id":[911889],"obs":[[0.46467154486587114,1.2977569804286362,-0.005075949307944214,-0.4884375313715229,0.5]],"new_obs":[[0.49062668447444385,1.337493026285129,-0.014844699935374673,-0.5495331549888742,0.5]],"actions":[[0.0677602]],"rewards":[11.21375939348018],"dones":[false]}
```

The CloudFormation stack output named "AthenaQueryToCreateJsonFormatedData" contains an Amazon Athena query which will perform this transformation. Copy this query into the Amazon Athena query editor to run the transformation. The query stores the trasformed data in the S3 bucket `<CloudFormation stack name>-databucket-<uniquie id>` under the prefix `json_offline_data`. 

Now the data is prepaired for the hyperparameter tuning job. You can launch the hyperparameter tuning job by invoking the lambda function `<CloudFormation stack name>-TuningJobLauncherFunction-<uniquie id>`.
When you launch this function, the flow below executes:
1. The lambda function creates an estimator to execute training runs.
2. The lambda function invokes a hyperparameter tuning job with this estimator. Options for which hyperparameters to tune include learning rates for the actor and critic, and model capacity for actor / critic. This creates 16 training jobs (with concurrency of 4), each with a unique set of hyper parameters.
3. Each training job executes the following steps:
   1. A [Conservative Q Learning](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#cql) algorithm is built.
   2. Settings from the hyperparameter tuning job are applied to the algorithm.
   3. A stopper is configured to end the tuning job when the "Critic Loss" stops decreasing.
   4. The data is split into train / validation groups.
   5. The algorithm begins training. Each epoch validation loss metrics are saved.
   6. When training completes, the cql policy is exported into PyTorch format. An ONNX format policy model is also saved.
4. After each training run, SageMaker stores the trained model data in S3.
5. The tuning job continues until the maximum number of experiments has been reached.

Now that the tuning job has finished, take a look at the metrics emitted during training. You can use CloudWatch, SageMaker Studio, or the `utils/investigate_training.py` file to view these metrics. Use the methodology from [this paper](https://arxiv.org/abs/2109.10813) to chose the model to deploy. Generally speaking, look for a training job with low `td_mse` and a high `q_mean` value. Look for this job in the sagemaker training jobs console page, and record the "S3 model artifact".

Now call the lambda function `<CloudFormation stack name>-ModelDeployerFunction-<uniquie id>` with an event of this form:

```json
{
  "DescribeTrainingJob": {
    "ModelArtifacts": {
      "S3ModelArtifacts": "s3://<CloudFormation stack name>-assetsbucket-<unique id>/training/<training job name>/output/model.tar.gz"
    }
  }
}
```
This will deploy the trained model to a serverless SageMaker endpoint. You can get acton recommendations from this endpoint by sending a message body of the form:
```json
{"inputs": ["Measurement1", "Measurement2", "..."]}
```

To test trained model, call the `RunPhysicsSimulationFunction` lambda function with an event of the form:
```json
{"random_action_fraction": 0.0, "inference_endpoint_name": "<sagemaker inference endpoint name>"}
```
This will run the simulation, using the new trained model as the action recommender. The results of this sumulation will be stored in S3 along with the data generated earlier.

You can view the performance of the new model by using Athena. Try the "Sample Athena Query" above to see performance metrics for each action source. One action source will be called `LQR_epsilon=0.5` and the other will be called `<sagemaker inference endpoint name>_epsilon=0.0`. The LQR action source was used to generate the training data. It uses a Linear Quadratic Regulator to find the optimal aciton, but actions randomly 50% of the time.

## Usage Guidance
The sample code; software libraries; command line tools; proofs of concept; templates; or other related technology (including any of the foregoing that are provided by our personnel) is provided to you as AWS Content under the AWS Customer Agreement, or the relevant written agreement between you and AWS (whichever applies). You should not use this AWS Content in your production accounts, or on production or other critical data. You are responsible for testing, securing, and optimizing the AWS Content, such as sample code, as appropriate for production grade use based on your specific quality control practices and standards. Deploying AWS Content may incur AWS charges for creating or using AWS chargeable resources, such as running Amazon EC2 instances or using Amazon S3 storage.

## Contributing
Please create a new GitHub issue for any feature requests, bugs, or documentation improvements.

Where possible, please also submit a pull request for the change.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.