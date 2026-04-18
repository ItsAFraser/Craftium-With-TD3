"""
This file runs training for all methods (TD3, PPO, A2C) and puts the results all in one specified directory.
Very similar to sb3_train_td3.py, but pulls the util methods from utils.py to keep this file short so it is easier for me (Ben) to read.  
Also makes a lot of variables I don't plan on changing constants instead of command line args. 
"""

from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from gymnasium import spaces
from argparse import ArgumentParser
from uuid import uuid4
import os
import craftium # This import is used even though the VSCode says it isn't!

from td3_gumbel_policy import GumbelMapperTD3Policy
from utils import make_env

# General Constants
NUMBER_OF_ENVS = 1
FRAMESKIP = 4
SYNC_MODE = False
FPS_MAX = 200
FRAME_STACK = 2
METHODS = [
    "td3-gumbel",
    "ppo",
    "a2c"
]
# Hardcoded action spaces for TD3. Taken from craftium docs (mouse control is always 4 here so it is not specified)
ACTION_SPACES = {
    "Craftium/ChopTree-v0": ["forward", "jump", "dig"],
    "Craftium/Speleo-v0": ["forward", "jump"]
}

# TD3 Constants
TD3_ACTION_THRESHOLD = 0.2
TD3_LEARNING_RATE = 1e-4
TD3_LEARNING_STARTS = 5_000
TD3_BUFFER_SIZE = 100_000
TD3_BATCH_SIZE = 128
TD3_TRAIN_FREQ = 80
TD3_GRADIENT_STEPS = 8

def parse_args():
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--runs-dir", type=str, default="./run-logs/",
        help="Name of the directory where run's data is stored. Defaults to './run-logs/'")
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0",
        help="Name (registered) of the environment.")
    parser.add_argument("--total-timesteps", type=int, default=50_000,
        help="Number of timesteps to train for.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # get actions for this environment
    td3_action_names = ACTION_SPACES.get(args.env_id, None)
    if td3_action_names is None:
        raise ValueError(f"Unsupported env_id {args.env_id}. Supported envs are: {list(ACTION_SPACES.keys())}")
    
    # turn into Box
    td3_action_names = [a.strip() for a in td3_action_names if a.strip()]
    if not td3_action_names:
        raise ValueError("td3_action_names must include at least one action")

    for method in METHODS:
        # Generate a unique directory name if not provided
        runs_dir = args.runs_dir if args.runs_dir != "./run-logs/" else f"./run-logs/{uuid4()}"

        # configure SB3 logger
        log_path = os.path.join(runs_dir, method) # save logs in runs_dir/run_name
        new_logger = logger.configure(log_path, ["stdout", "csv"])  # log to both console and CSV file for later analysis

        # Create a vectorized environment with the specified number of parallel environments, each initialized with the appropriate wrappers based on the method.
        print(f"Using env {args.env_id} with method {method} and {args.total_timesteps} timesteps")
        envs = DummyVecEnv([
            make_env(
                args.env_id,
                method,
                td3_action_names=td3_action_names,
                td3_action_threshold=TD3_ACTION_THRESHOLD,
                frameskip=FRAMESKIP,
                sync_mode=SYNC_MODE,
                fps_max=FPS_MAX,
            )
            for _ in range(NUMBER_OF_ENVS)
        ]) 
        envs = VecFrameStack(envs, FRAME_STACK) # Stack a short history for motion cues while keeping memory/compute manageable.
        envs = VecMonitor(envs) # Monitor wrapper to track episode rewards, lengths, and other metrics across the vectorized environments. 

        #if PPO or A2C, initialize with the standard CNN policy. if TD3, initialize with the custom JointGaussianMapperTD3Policy
        #that maps CNN features to continuous action scores, which are then converted to discrete actions by the ContinuousToDiscreteActionWrapper.
        if method == "ppo":
            model = PPO("CnnPolicy", envs, verbose=1)
        elif method == "a2c":
            model = A2C("CnnPolicy", envs, verbose=1)
        else:  # TD3
            # TD3 actor deterministically maps CNN features through a trainable NN to action
            # scores. Wrapper fires actions above a configurable threshold + mouse passthrough.
            if not isinstance(envs.action_space, spaces.Box):
                raise TypeError(f"TD3 requires a Box action space, got {type(envs.action_space)}")

            # TD3 hyperparameters! Mostly found in td3.py in the SB3 repo, but I decreased some in a somewhat vain
            # attempt to help speed up training.
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py
            model = TD3(
                GumbelMapperTD3Policy,
                envs,
                learning_rate=TD3_LEARNING_RATE,
                verbose = 1, # Print TD3's own debug info (e.g. actor/critic losses) to console. 1 is SB3's default; 0 would disable, 2 would be more verbose.
                learning_starts = TD3_LEARNING_STARTS,
                buffer_size = TD3_BUFFER_SIZE,
                batch_size = TD3_BATCH_SIZE,
                train_freq = TD3_TRAIN_FREQ,
                gradient_steps = TD3_GRADIENT_STEPS,
            )

        model.set_logger(new_logger)
        model.learn(total_timesteps=args.total_timesteps)
        envs.close()

if __name__ == "__main__":
    main()