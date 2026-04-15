from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common import logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from typing import Any
from argparse import ArgumentParser
from uuid import uuid4
import os
import craftium # This import is used even though the VSCode says it isn't!
import matplotlib.pyplot as plt

from td3_joint_policy import JointGaussianMapperTD3Policy

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
    parser.add_argument("--method", type=str, default="a2c", choices=["ppo", "a2c", "td3"],
        help="RL method to use to optimize the agent.")
    # fmt: on

    return parser.parse_args()

'''
This gym wrapper is in part derived from this tutorial/learning resource:
https://alexandervandekleut.github.io/gym-wrappers/ 
'''
class ObservationSaverWrapper(gym.Wrapper): # TODO: Is gym.ObservationWrapper better?
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0 # Used in step() below
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

        # Saves every frame. Change to e.g. `if self.steps % 100 == 0:` to reduce disk usage.
        # Warning: frames add up fast — 6000 frames was ~120MB!
        self.save(obs)

        return obs, reward, terminated, truncated, info

    def save(self, obs):
        plt.clf() # Some of the observations seemed to be overlaying each other without this, so I'm force-clearing here
        plt.imshow(obs)
        plt.axis("off")
        plt.savefig(f"results/observation_{self.steps}.png")




'''
Craftium declares its observation space as (W, H, C) but actual frames arrive as (H, W, C).
This wrapper corrects the declared observation_space to match reality so SB3's VecEnv
buffers are allocated with the correct shape.
'''
class FixObsSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Craftium swaps width/height in the space declaration; swap them back.
        w, h, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def observation(self, obs):
        # Frames are already (H, W, C); no data transformation needed.
        return obs


'''
This gym wrapper converts TD3's continuous action scores to Craftium's discrete action space.
Any action whose score exceeds 0.0 fires, so the agent can press multiple buttons simultaneously
(e.g. forward + dig at the same time), which is required for ChopTree.
'''
class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """Converts TD3 mapped action scores to Craftium actions via per-action threshold."""

    def __init__(self, env, num_actions=18):
        super().__init__(env)
        # Number of discrete actions (e.g., 18 for OpenWorld). Any action whose score exceeds
        # 0.0 fires, allowing simultaneous button presses. The remaining 2 dimensions are
        # reserved for continuous mouse control, which passes through without discretization.
        self.num_actions = num_actions

        # Action vector layout (output by JointGaussianMapperActor):
        # [0:num_actions] -> mapped action scores in [-1, 1]
        # [num_actions:num_actions+2] -> mouse x,y (continuous pass-through)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_actions + 2,),
            dtype=np.float32
        )

        # OpenWorld-focused 18-action set (mouse handled separately as continuous)
        self.action_names = [
            "forward", "backward", "left", "right", "jump", "sneak",
            "dig", "place", "inventory",
            "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
            "slot_6", "slot_7", "slot_8", "slot_9",
        ]
        assert len(self.action_names) == self.num_actions
    
    # The action method converts the continuous action vector from TD3 into a discrete action dictionary that Craftium can use. 
    def action(self, action):
        """Convert TD3 mapped action-score vector into Craftium action dict."""
        # Sanity check on input shape. Should be (num_actions + 2,) where last 2 are mouse control.
        assert action.shape == (self.num_actions + 2,), \
            f"Expected shape {(self.num_actions + 2,)}, got {action.shape}"

        # Split the input action vector into discrete action scores and mouse control values.
        action_scores = action[:self.num_actions]
        mouse_vals = action[self.num_actions:self.num_actions + 2]

        # Fire every action whose score exceeds the threshold. Unlike argmax (which picks
        # exactly one action), this allows simultaneous presses, critical for tasks that
        # require holding forward and dig at the same time.
        action_dict = {}
        for i, score in enumerate(action_scores):
            if score > 0.0:
                action_dict[self.action_names[i]] = 1

        # Keep mouse control continuous as requested.
        action_dict["mouse"] = np.array(mouse_vals, dtype=np.float32)
        return action_dict


#the Make_env function creates a factory function that initializes the environment
#with the appropriate wrappers based on the specified method (PPO, A2C, or TD3).
#For TD3, it layers the FixObsSpaceWrapper and ContinuousToDiscreteActionWrapper on 
#top of the base Craftium environment to ensure compatibility with TD3's requirements.
def make_env(env_id, method, num_actions=18):
    '''
    Factory function to create environment initializer with appropriate wrappers based on method (PPO, A2C, or TD3).
    '''
    def _init():
        # set up the environment
        craftium_kwargs: dict[str, Any] = dict(
            frameskip=3,
            rgb_observations=False,  # Grayscale: 3x smaller obs, critical for replay buffer memory
            gray_scale_keepdim=True,  # Keeps the channel dim so shape is (H, W, 1) not (H, W)
            sync_mode=True,          # Sync Luanti steps to agent steps — prevents wasted simulation
            fps_max=30,              # Cap Luanti at 30fps; no point rendering faster than frameskip needs
        )
        
        env = gym.make(env_id, **craftium_kwargs)
        # Uncomment this for saving observations to file! Be warned it takes up quite a bit of space!
        # env = ObservationSaverWrapper(env)
        # For TD3, bypass the DiscreteActionWrapper that Craftium bakes into its registered envs.
        # That wrapper expects an integer, but our ContinuousToDiscreteActionWrapper outputs a dict.
        # Unwrapping gives us CraftiumEnv directly (accepts dict actions), then we layer our
        # wrappers on top.
        if method == "td3":
            base_env = env.unwrapped  # strip DiscreteActionWrapper and any other registered wrappers
            env = FixObsSpaceWrapper(base_env)
            env = ContinuousToDiscreteActionWrapper(env, num_actions=num_actions)
        else:
            env = FixObsSpaceWrapper(env)
        # Note: do NOT call env.reset() here — DummyVecEnv calls reset() on each env itself.

        return env
    return _init

#Main training Loop. Configures the logger, creates the vectorized environment with the appropriate wrappers
#initializes the model based on a specific method (PPO., A2C, or TD3), and starts the learning process for a given number ot timesteps.
if __name__ == "__main__":
    args = parse_args()

    # Generate a unique run name if not provided, and configure the SB3 logger to save logs in the specified directory.
    if args.run_name is None:
        run_name = f"{args.env_id.replace('/', '-')}-{args.method}--{str(uuid4())}"
    else:
        run_name = args.run_name

    # configure SB3 logger
    log_path = os.path.join(args.runs_dir, run_name) # save logs in runs_dir/run_name
    print(f"** Storing run's data in {log_path}")
    new_logger = logger.configure(log_path, ["stdout", "csv"]) # log to both console and CSV file for later analysis

    envs = DummyVecEnv([make_env(args.env_id, args.method, num_actions=18) for _ in range(args.num_envs)]) # Create a vectorized environment with the specified number of parallel environments, each initialized with the appropriate wrappers based on the method.
    envs = VecFrameStack(envs, 3) #stack the last 3 frames together to give agent temporal context. helps with tasks that require understanding of motion or change over time.
    envs = VecMonitor(envs) # Monitor wrapper to track episode rewards, lengths, and other metrics across the vectorized environments. 


    #if PPO or A2C, initialize with the standard CNN policy. if TD3, initialize with the custom JointGaussianMapperTD3Policy
    #that maps CNN features to continuous action scores, which are then converted to discrete actions by the ContinuousToDiscreteActionWrapper.
    if args.method == "ppo":
        model = PPO("CnnPolicy", envs, verbose=1)
    elif args.method == "a2c":
        model = A2C("CnnPolicy", envs, verbose=1)
    else:  # TD3
        # TD3 actor deterministically maps CNN features through a trainable NN to action
        # scores. Wrapper fires all actions above 0.0 threshold + mouse passthrough.
        if not isinstance(envs.action_space, spaces.Box):
            raise TypeError(f"TD3 requires a Box action space, got {type(envs.action_space)}")
        n_actions = envs.action_space.shape[0]  # Should be 20 (18 actions + 2 mouse)
        # Sigma reduced to 0.05 because the actor is now deterministic — exploration comes
        # from this noise only (no internal Gaussian sampling adding on top).
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.05 * np.ones(n_actions),
        )

        # TD3 hyperparameters! Mostly found in td3.py in the SB3 repo, but I decreased some in a somewhat vain
        # attempt to help speed up training.
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py
        model = TD3(
            JointGaussianMapperTD3Policy,
            envs,
            action_noise = action_noise,
            verbose = 1, # Print TD3's own debug info (e.g. actor/critic losses) to console. 1 is SB3's default; 0 would disable, 2 would be more verbose.
            learning_starts = 1_000, # Number of steps to collect transitions with the untrained policy before starting to update the networks. 
            buffer_size = 100_000,  # Size of the Replay Buffer. 
            batch_size = 64, # Number of samples per batch for each training step.
            train_freq = 1, # Frequency of training steps (in environment steps).
            gradient_steps = 1, # Number of gradient steps per training step.
        )
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps)

    envs.close()