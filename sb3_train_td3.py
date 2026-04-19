from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common import logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from argparse import ArgumentParser
from uuid import uuid4
import os
import craftium # This import is used even though the VSCode says it isn't!
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch device:", device)
print("cuda available:", torch.cuda.is_available())

def parse_args():
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--run-name", type=str, default=None,
        help="Unique name for the run. Defaults to a random uuid.")
    parser.add_argument("--runs-dir", type=str, default="./run-logs/",
        help="Name of the directory where run's data is stored. Defaults to './run-logs/'")

    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0",
        help="Name (registered) of the environment.")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="Number of timesteps to train for.")
    parser.add_argument("--num-envs", type=int, default=4,
        help="Number of environments to use.")
    # Adding td3 as an option here
    parser.add_argument("--method", type=str, default="td3", choices=["ppo", "a2c", "td3"],
        help="RL method to use to optimize the agent.")
    # fmt: on

    return parser.parse_args()

'''
This gym wrapper is in part derived from this tutorial/learning resource:
https://alexandervandekleut.github.io/gym-wrappers/ 
'''
class ObersavtionSaverWrapper(gym.Wrapper): # TODO: Is gym.ObservationWrapper better?
    def __init__(self, env, env_index):
        super().__init__(env)
        self.steps = 0 # Used in step() below
        self.env_index = env_index
        os.makedirs("results", exist_ok=True)

    # This was autofilled by Intellisense, and it works :]
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.save(obs)
        return obs, info

    def step(self, action):
        # This step & return logic seems to be necessary for Gym wrappers, even though I'm not using most of it here
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        # This controls how often observations/frames are saved. I was worried the saving would slow down the training
        # too much, but it seemed to have very little impact, so I'm just saving every frame. However, be warned that
        # these frames quickly add up in space (6000 frames was ~120MB!)
        if self.steps % 1 == 0:
            self.save(obs)

        return obs, reward, terminated, truncated, info

    def save(self, obs):
        plt.clf() # Some of the observations seemed to be overlaying each other without this, so I'm force-clearing here
        plt.imshow(obs)
        plt.axis("off")
        os.makedirs(f"results/env{self.env_index}", exist_ok=True)
        plt.savefig(f"results/env{self.env_index}/{self.env_index}_observation_{self.steps}.png")

'''
This gym wrapper is in part derived from this tutorial/learning resource:
https://alexandervandekleut.github.io/gym-wrappers/ 
'''
class RoomActionSpaceConversionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # The idea here is to take the continuous TD3 output and convert it into a discrete 'range' of actions
        # from -1 to 1. The dimension/shape works as 1 here since we're just getting one scalar as output.
        # See here for more detail on Spaces: https://gymnasium.farama.org/api/spaces/
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        z = float(action[0])
        # Basically discretizing the direct output of TD3. TODO: This is not using the mean/variance idea that Alex
        # was talking about, but I can't figure out how to hijack the SB3 implementation to do more than this. Talk
        # to Alex and Ben about that!
        if z < -0.75:
            return 0
        elif z < -0.5:
            return 1
        elif z < -0.25:
            return 2
        elif z < 0:
            return 3
        elif z < 0.25:
            return 4
        elif z < 0.5:
            return 5
        elif z < 0.75:
            return 6
        else:
            return 7

def make_env(env_id, method, env_index):
    def _init():
        # set up the environment
        craftium_kwargs = dict(
            frameskip=3,
            rgb_observations=True,
            gray_scale_keepdim=True,
        )
        
        env = gym.make(env_id, **craftium_kwargs)
        # Uncomment this for saving observations to file! Be warned it takes up quite a bit of space!
        env = ObersavtionSaverWrapper(env, env_index)
        # Maybe a bit hacky, but this specially handles TD3 since it needs a special wrapper.
        # TODO: Will need a smarter way to do this for the various Discrete action space sizes
        if method == "td3":
            env = RoomActionSpaceConversionWrapper(env)
        env.reset()

        return env
    return _init

if __name__ == "__main__":
    args = parse_args()

    if args.run_name is None:
        run_name = f"{args.env_id.replace('/', '-')}-{args.method}--{str(uuid4())}"
    else:
        run_name = args.run_name

    # configure SB3 logger
    log_path = os.path.join(args.runs_dir, run_name)
    print(f"** Storing run's data in {log_path}")
    new_logger = logger.configure(log_path, ["stdout", "csv"])

    envs = DummyVecEnv([make_env(args.env_id, args.method, i) for i in range(args.num_envs)])
    envs = VecFrameStack(envs, 3)
    envs = VecMonitor(envs)

    if args.method == "ppo":
        model = PPO("CnnPolicy", envs, verbose=1)
    elif args.method == "a2c":
        model = A2C("CnnPolicy", envs, verbose=1)
    else:
        # Derived from the example on this SB3 documentation page:
        # https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
        n_actions = envs.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )

        # TD3 hyperparameters! Mostly found in td3.py in the SB3 repo, but I decreased some in a somewhat vain
        # attempt to help speed up training.
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py
        model = TD3(
            "CnnPolicy",
            envs,
            action_noise = action_noise,
            verbose = 1,
            learning_starts = 1_000,
            buffer_size = 100_000,
            batch_size = 64,
            train_freq = 1,
            gradient_steps = 1,
            device="cuda",
        )
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps)

    envs.close()