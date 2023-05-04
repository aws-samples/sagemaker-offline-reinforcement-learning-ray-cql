
import os, traceback
import json
import random
from glob import glob
import logging
from pprint import pprint
import tarfile
import argparse

import gymnasium as gym
import numpy as np
import jq


import torch

import ray
from ray import air, tune
from ray.train.rl import RLTrainer
from ray.tune.stopper.stopper import Stopper
from ray.rllib.algorithms import cql
from ray.rllib.algorithms.cql.cql_torch_policy import cql_loss
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def list_files(startpath):
    for dirpath, _, files in os.walk(startpath):
        for x in files:
            print(os.path.join(dirpath, x))
            
def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="CQL", help="The RLlib-registered algorithm to use.")
    
    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--training_iterations", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--actor_learning_rate", type=float, default=3-2)
    parser.add_argument("--critic_learning_rate", type=float, default=3e-4)
    parser.add_argument("--number_of_layers", type=int, default=2)
    
    parser.add_argument("--seed", type=int, default=1)
    
    return parser.parse_known_args()


# https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py

class NestedMetricNotDecreasingStopper(Stopper):
    """Early stop the experiment when a metric plateaued across trials.
    Stops the entire experiment when the metric has plateaued
    for more than the given amount of iterations specified in
    the patience parameter.
    Args:
        list_of_metric_keys: A list of keys to a dictionary 
            which returns the metric.
            Ex: result = {"Key1":{"Key2":{"Key3": metric}}},
                list_of_metric_keys = ["Key1","Key2","Key3"]
        std: The minimal standard deviation after which
            the tuning process has to stop.
        top: The number of best models to consider.
        mode: The mode to select the top results.
            Can either be "min" or "max".
        patience: Number of epochs to wait for
            a change in the top models.
    Raises:
        ValueError: If the mode parameter is not "min" nor "max".
        ValueError: If the top parameter is not an integer
            greater than 1.
        ValueError: If the standard deviation parameter is not
            a strictly positive float.
        ValueError: If the patience parameter is not
            a strictly positive integer.
    """

    def __init__(
        self,
        list_of_metric_keys: list,
        top: int = 10,
        mode: str = "min",
        patience: int = 0,
        iterations_before_evaluate = 100,
    ):
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be either min or max.")
        if not isinstance(top, int) or top <= 1:
            raise ValueError(
                "Top results to consider must be"
                " a positive integer greater than one."
            )
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be a strictly positive integer.")
        self._mode = mode
        self._list_of_metric_keys = list_of_metric_keys
        self._patience = patience
        self._patence_iterations = 0
        self._iterations = 0
        self._top = top
        self._recent_values = [float('inf')]*top
        self._stopper_triggered = False
        self._iterations_before_evaluate = iterations_before_evaluate
    
    def _get_nested_dictionary_key(self, d: dict, key_list: list):
        next_value = d[key_list[0]]
        if isinstance(next_value, dict):
            next_value = self._get_nested_dictionary_key(next_value, key_list[1:])
        return next_value
        
    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        self._iterations += 1
        metric_value = self._get_nested_dictionary_key(result, self._list_of_metric_keys)
        
        if metric_value > max(self._recent_values):
            self._stopper_triggered = True
            self._patence_iterations += 1
            print('Recent Stopper Values')
            print(self._recent_values)
            print('Metric Value')
            print(metric_value)
        else:
            self._stopper_triggered = False
            self._patence_iterations = 0
        
        self._recent_values = self._recent_values[1:]
        self._recent_values.append(metric_value)

        # and then call the method that re-executes
        # the checks, including the iterations.
        return self.stop_all()

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        if self._stopper_triggered:
            print(f'Metric not decreasing stopper is triggered with {self._patence_iterations} iterations and patence {self._patience}')
            
        return self._stopper_triggered and self._patence_iterations >= self._patience and self._iterations > self._iterations_before_evaluate

def batch_generator(items, batch_size):
    count = 1
    chunk = []
    
    for item in items:
        if count % batch_size:
            chunk.append(item)
        else:
            chunk.append(item)
            yield chunk
            chunk = []
        count += 1
    
    if len(chunk):
        yield chunk

def custom_eval_function(algorithm, eval_workers, batch_size = 2**10):
    """This function runs the cql loss function against evaluation data
    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.
    Returns:
        metrics: Evaluation metrics dict.
    """
    
    reader = JsonReader(algorithm.evaluation_config.to_dict()['input'])
    trajectory_batches = reader.read_all_files()
    policy = algorithm.get_policy()

    sum_actor_loss = sum_critic_loss = sum_td_mse = sum_cql_loss = 0.
    
    for batch_number, batch_list in enumerate(batch_generator(trajectory_batches, batch_size)):
        for i, batch in enumerate(batch_list):
            for col in ['rewards','obs','new_obs','actions','prev_rewards','terminateds']:
                batch[col]=torch.tensor(batch[col])
            if i == 0:
                batch_catcher = batch
            else:
                batch_catcher.concat(batch)
        
        with torch.no_grad():
            model = next(iter(policy.target_models))
            # print(f'Initial model tower stats: {model.tower_stats.get("td_error",None)}')
            actor_loss, critic_loss, alpha_loss, alpha_prime_loss = cql_loss(
                policy = policy, 
                model = model, 
                dist_class = algorithm.get_policy().dist_class,
                train_batch = batch_catcher
                )
            sum_actor_loss += actor_loss.tolist()
            sum_critic_loss += np.average(model.tower_stats.get("critic_loss")).tolist()#critic_loss.tolist()
            sum_cql_loss += np.average(model.tower_stats.get("cql_loss")).tolist()
            # sum_td_mae += model.tower_stats.get("td_error").tolist()[0]
            sum_td_mse += np.average(model.tower_stats.get("critic_loss")).tolist()-np.average(model.tower_stats.get("cql_loss")).tolist()#td_error is mae
    
    return {
        "evaluation_actor_loss":  sum_actor_loss / (batch_number+1),
        "evaluation_critic_loss": sum_critic_loss / (batch_number+1),
        "evaluation_cql_loss": sum_cql_loss / (batch_number+1),
        "evaluation_td_mse": sum_td_mse/(batch_number+1)
    }

#https://github.com/ray-project/ray/blob/master/rllib/examples/vizdoom_with_attention_net.py
if __name__ == "__main__":
    # #Put ray into the output model loc b/c I can't seem to get dependencies to work in the PyTorchModel function.
    # shutil.copytree(ray.__file__, os.path.join(os.environ['SM_MODEL_DIR'], "dependencies"))
    print('Hello from the training script')
    print(f'Ray version: {ray.__version__}')
    
    args, _ = parse_args()
    print('args: ')
    # print(json.dumps(args, indent = 4))
    print(args)
    
    ALGORITHM = args.algorithm
    TRAINING_DATA_DIR = os.environ.get("SM_CHANNEL_TRAIN")
    # CHECKPOINT_DIR = os.path.join(os.environ['SM_OUTPUT_INTERMEDIATE_DIR'],'tuner_checkpoint')
    CHECKPOINT_DIR = os.path.join('/opt/ml/checkpoints','tuner_checkpoint')
    MODEL_DIR = os.environ['SM_MODEL_DIR']
    TRAINING_ITERATIONS = args.training_iterations
    ACTOR_LEARNING_RATE = args.actor_learning_rate
    CRITIC_LEARNING_RATE = args.critic_learning_rate
    INITIAL_Q_VALUE = 0.
    ACTION_SPACE_LENGTH = 1
    STATE_SPACE_LENGTH = 5
    NUMBER_LAYERS = args.number_of_layers
    VALIDATION_DATA_FRAC = 0.1
    SEED = args.seed
    
    random.seed(SEED)
    
    input_data_file_list = [os.path.join(TRAINING_DATA_DIR, file_name) for file_name in os.listdir(TRAINING_DATA_DIR)]
    random.shuffle(input_data_file_list)
    last_validation_file_index = int(VALIDATION_DATA_FRAC*len(input_data_file_list))
    validation_data_file_list = input_data_file_list[:last_validation_file_index]
    train_data_file_list = input_data_file_list[last_validation_file_index:]
    
    print(f'Training on {len(train_data_file_list)} files, validating on {len(validation_data_file_list)}, out of a total {len(input_data_file_list)} files from path: {TRAINING_DATA_DIR}')
    print('First 5 training files')
    pprint(train_data_file_list[:5])
    
    # # load information from previous trials to restore.
    # # These are default values to be replaced if there is a checkpoint to load.
    # restored_policy_weights = None
    # previous_iterations = 1
    # try:
    #     print(f'Attempting to load previous checkpoint. CHECKPOINT_DIR ({CHECKPOINT_DIR}) content:')
    #     list_files(CHECKPOINT_DIR)
    #     #First get model weights. This way of finding the experiment_dirs guarentees that there is a checkpoint in the dir.
    #     experiment_dirs = [os.path.join(*exp_dir.split("/")[:-1]) for exp_dir in glob(f"{CHECKPOINT_DIR}/CQL/*/checkpoint*", recursive = True)]
    #     input_model_path = os.path.join(
    #                 os.environ['SM_INPUT_DIR'],
    #                 'data',
    #                 'model'
    #             )
    #     if len(experiment_dirs) > 0:
    #         # If there is a previous experiment in the intermediate directory, load the weights
    #         print('Extracting previous checkpoint from experiment directory')
    #         pprint(experiment_dirs)
    #         experiment_dir_dttms = [x[-19:] for x in experiment_dirs]
    #         index_of_last_experiment = experiment_dir_dttms.index(max(experiment_dir_dttms))
    #         experiment_dir = experiment_dirs[index_of_last_experiment]
    #         print(f'Best Experiment Dir: {experiment_dir}')
    #         last_checkpoint_dir = glob(f"/{experiment_dir}/*/", recursive = True)[-1]
    #     elif os.path.exists(input_model_path):
    #         # If an input model was supplied, load it as the inital model
    #         print(f'Extracting files in input_model_path: {input_model_path}')
    #         with tarfile.open(os.path.join(input_model_path,'model.tar.gz')) as file:
    #             # extract files
    #             file.extractall(input_model_path)
    #         print('Finished extracting model files')
    #         list_files(input_model_path)
    #         last_checkpoint_dir = os.path.join(input_model_path,'checkpoint')
    #     else:
    #         print('Did not find checkpoints in either the checkpoint dir or the input_model_path')
        
    #     print(f'Last checkpoint dir: {last_checkpoint_dir}')
        
    #     last_checkpoint_policy_dir = os.path.join(last_checkpoint_dir,'policies','default_policy')
    #     restored_policy = Policy.from_checkpoint( last_checkpoint_policy_dir)
    #     restored_policy_weights = restored_policy.get_weights()
        
    #     print('Successfully restored policy weights')
        
    #     # Now get the training iteration
    #     previous_iterations = 0
    #     experiment_state_paths = glob(f"{CHECKPOINT_DIR}/CQL/experiment_state*", recursive = True)
    #     print('Experiment State Paths:')
    #     print(experiment_state_paths)
    #     for experiment_state_path in experiment_state_paths:
    #         print(f'Opening experiment: {experiment_state_path}')
    #         with open(experiment_state_path) as jsonfile:
    #             experiment_state = json.load(jsonfile)
    #         last_training_iteration = json.loads(experiment_state.get('checkpoints')[0]).get('_last_result').get('training_iteration')
    #         previous_iterations += last_training_iteration
    #     print(f'Number of previous iterations: {previous_iterations}')
        
    #     print('Successfully loaded model weights and found previous iterations.')
    #     # print(restored_policy_weights)
    # except Exception as e:
    #     print(f'Unable to load previous checkpoint due to {e}')
    #     print(traceback.format_exc())

    
    
    class MyCallback(DefaultCallbacks):
        
        # def on_algorithm_init(self, algorithm, **kwargs):
            
        #     # Load algorithm weights from previous training
        #     try:
        #         #This is how you resorte weights
        #         algorithm.set_weights({"default_policy": restored_policy_weights})
        #         print('Successfully set model weights')
        #     except Exception as e:
        #         print(f'Unable to resore algorithm weights due to {e}')
        #         print(traceback.format_exc())
        #         print('End of handled error related to loading model weights.')
            
        #     #Set the initial q_net bias to be the lowest q value in the behavioral data set
        #     policy = algorithm.get_policy()
            
        #     #Set both the q_net and twin_q_net for compatability
        #     policy.model.q_net._logits._model[0].bias = torch.nn.Parameter(torch.tensor([INITIAL_Q_VALUE]))
            
        #     if policy.model.twin_q_net:
        #         policy.model.twin_q_net._logits._model[0].bias = torch.nn.Parameter(torch.tensor([INITIAL_Q_VALUE]))
        
        def on_train_result(self, algorithm, result, **kwargs):
            "Calculate objective metric. Here it's a sum of actor, critic, and cql losses. Expose them and all results at the end of each training cycle."
            policy = algorithm.get_policy()
            for tower in policy.model_gpu_towers:
                print('on_train_result Tower Keys: ',', '.join(tower.tower_stats.keys()))
                
            reloaded_results = json.loads(json.dumps(result, default = str)) # This is required so that jq can read np values w/o custom SerDe
            
            # See the definition of critic_loss: https://github.com/ray-project/ray/blob/master/rllib/algorithms/cql/cql_torch_policy.py
            td_mse = float(jq.compile('.info.learner.default_policy.learner_stats.critic_loss').input(reloaded_results).first()) - float(jq.compile('.info.learner.default_policy.learner_stats.cql_loss').input(reloaded_results).first())
            
            loss_keys = [
                '.info.learner.default_policy.learner_stats.actor_loss',
                '.info.learner.default_policy.learner_stats.critic_loss',
                # '.info.learner.default_policy.learner_stats.cql_loss'
                ]
            metric_value = 0
            for loc in loss_keys:
                metric_value += float(jq.compile(loc).input(reloaded_results).first())
            #add additional metrics
            result.update({
                "sum_actor_critic_loss": metric_value,
                "td_mse": td_mse,
                # "total_training_iterations": result.get('training_iteration') + previous_iterations
            })
            
            result.pop('config') #This is printed earlier in the log.
            
            # This will be picked up in the logs.
            print(json.dumps(result, default = str))
            # print(f'CHECKPOINT_DIR ({CHECKPOINT_DIR}) content:')
            # list_files(CHECKPOINT_DIR)
    
    action_space = gym.spaces.Box(low = np.array([-1.]), high = np.array([1.]*ACTION_SPACE_LENGTH))
    observation_space = gym.spaces.Box(low = np.array([-2.]*STATE_SPACE_LENGTH), high = np.array([2.]*STATE_SPACE_LENGTH))
    
    #https://github.com/ray-project/ray/blob/master/rllib/examples/offline_rl.py 
    config = (
        cql.CQLConfig()
        .framework(framework='torch')
        .environment(observation_space = observation_space, action_space = action_space)
        .offline_data(
            input_= train_data_file_list, #TRAINING_DATA_DIR,
            actions_in_input_normalized=False, # If true the actor to use the "TorchSquashedGaussian" distribution. Be sure to update the inference code if this changes. 
        )
        .training(
            # clip_actions=False,
            # normalize_actions = True,
            twin_q=True,
            
            # q_model_config = {
            #     "fcnet_hiddens": [256]*NUMBER_LAYERS,
            #     "fcnet_activation": "relu",
            # },
            
            # policy_model_config = {
            #     "fcnet_hiddens": [64],#[256]*NUMBER_LAYERS,
            #     "fcnet_activation": "relu",
            # },
            
            train_batch_size = 2**7,
            optimization_config = {
                "actor_learning_rate": ACTOR_LEARNING_RATE,
                "critic_learning_rate": CRITIC_LEARNING_RATE,
                "entropy_learning_rate": 3e-4,
            },
            # num_steps_sampled_before_learning_starts=2**8,
            bc_iters = 0#2**10
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=2**13,
            evaluation_num_workers=1,
            evaluation_duration_unit="timesteps",
            evaluation_parallel_to_training=False,
            evaluation_config={"input": validation_data_file_list},
            custom_evaluation_function=custom_eval_function,
        )
        .reporting(
            min_train_timesteps_per_iteration = 2**13 # epoch size. How many time steps to sample each epoch.
        )
        .callbacks(
            callbacks_class=MyCallback
        )
        .debugging(seed=SEED)
        # .resources(
        #     num_trainer_workers=os.cpu_count()-2 # Docs say this should be num_learner_workers, but help() says otherwise.
        # )
    )
    
    trainer = RLTrainer(
        algorithm=ALGORITHM,
        config=config.to_dict(),
        
    )
    trainable = trainer.as_trainable()
    
    stop = [
        tune.stopper.MaximumIterationStopper(max_iter = TRAINING_ITERATIONS),
        tune.stopper.TimeoutStopper(timeout = 60*60*20), #Seconds
        NestedMetricNotDecreasingStopper(
            list_of_metric_keys = ["info","learner","default_policy","learner_stats","critic_loss"], 
            top = 100, 
            mode = 'min', 
            patience = 2,
            iterations_before_evaluate = 10
            )
    ]
    
    print('Algorithm Config:')
    print(json.dumps(config.to_dict(), default = str, indent = 2, sort_keys=True))
        
    tuner = tune.Tuner(
        trainable,
        run_config=air.RunConfig(
            name=ALGORITHM,
            stop=stop,
            verbose=2,
            # sync_config=sync_config,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=2,
                checkpoint_at_end=True,
                # checkpoint_score_attribute="info/learner/default_policy/learner_stats/critic_loss",
                # checkpoint_score_order="min",
                # num_to_keep=5
            ),
            local_dir=CHECKPOINT_DIR,
        ),
    )
    
    try:
        # tuner.restore(os.path.join(CHECKPOINT_DIR,ALGORITHM))
        tuner = tune.Tuner.restore(
            os.path.join(CHECKPOINT_DIR,ALGORITHM),
            trainable=trainable,
            resume_errored=True,
        )
        print('Successfully restored tuner job')
    except Exception as e:
        print('Could not restore checkpoint due to ', e)
    
    results = tuner.fit()
    
    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="info/learner/default_policy/learner_stats/mean_q", mode="max")
    
    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    
    best_checkpoint.to_directory(os.path.join(MODEL_DIR,'checkpoint'))
    
    # Save the pytorch model too
    algorithm = config.build()
    algorithm.restore(os.path.join(MODEL_DIR,'checkpoint'))
    
    #Export in pytorch format
    algorithm.export_policy_model(os.path.join(MODEL_DIR,'model'))
    
    #Export in onnx format
    algorithm.export_policy_model(os.path.join(MODEL_DIR,'onnx_model'), onnx = 13)
    
    #Print Action Distribution Type
    print(f'Action Model Distribution Class: {algorithm.get_policy().dist_class}')
    
    print('Done training!')