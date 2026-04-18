import os
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, env, action_names, action_threshold=0.2):
        super().__init__(env)
        # Number of discrete actions controlled by TD3. The remaining 4 dimensions are
        # reserved for continuous mouse control, which passes through without discretization.
        self.action_names = list(action_names)
        self.num_actions = len(self.action_names)
        self.action_threshold = action_threshold
        if self.num_actions == 0:
            raise ValueError("action_names must contain at least one action")

        # Action vector layout (output by JointGaussianMapperActor):
        # [0:num_actions] -> mapped action scores in [-1, 1]
        # [num_actions:num_actions+4] -> mouse x,y (continuous pass-through)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions + 4,),
            dtype=np.float32
        )
    
    # The action method converts the continuous action vector from TD3 into a discrete action dictionary that Craftium can use. 
    def action(self, action):
        """Convert TD3 mapped action-score vector into Craftium action dict."""
        # Sanity check on input shape. Should be (num_actions + 4,) where last 4 are mouse control.
        assert action.shape == (self.num_actions + 4,), \
            f"Expected shape {(self.num_actions + 4,)}, got {action.shape}"

        # Split the input action vector into discrete action scores and mouse control values.
        action_scores = action[:self.num_actions]
        mouse_vals = action[self.num_actions:self.num_actions + 4]

        # Fire every action whose score exceeds the threshold. Unlike argmax (which picks
        # exactly one action), this allows simultaneous presses, critical for tasks that
        # require holding forward and dig at the same time.
        action_dict = {}
        for i, score in enumerate(action_scores):
            if score > self.action_threshold:
                action_dict[self.action_names[i]] = 1

        # Keep mouse control continuous as requested.
        action_dict["mouse"] = np.array(mouse_vals, dtype=np.float32)
        return action_dict

def make_env(
    env_id,
    method,
    td3_action_names,
    td3_action_threshold,
    frameskip,
    sync_mode,
    fps_max,
):
    '''
    Factory function to create environment initializer with appropriate wrappers based on method (PPO, A2C, or TD3).
    '''
    def _init():
        # set up the environment
        craftium_kwargs: dict[str, Any] = dict(
            frameskip=frameskip,
            rgb_observations=False,  # Grayscale: 3x smaller obs, critical for replay buffer memory
            gray_scale_keepdim=True,  # Keeps the channel dim so shape is (H, W, 1) not (H, W)
            sync_mode=sync_mode,
            fps_max=fps_max,
        )
        
        env = gym.make(env_id, **craftium_kwargs)
        # Uncomment this for saving observations to file! Be warned it takes up quite a bit of space!
        # env = ObservationSaverWrapper(env)
        # For TD3, bypass the DiscreteActionWrapper that Craftium bakes into its registered envs.
        # That wrapper expects an integer, but our ContinuousToDiscreteActionWrapper outputs a dict.
        # Unwrapping gives us CraftiumEnv directly (accepts dict actions), then we layer our
        # wrappers on top.
        if method.startswith("td3"):
            base_env = env.unwrapped  # strip DiscreteActionWrapper and any other registered wrappers
            env = FixObsSpaceWrapper(base_env)
            env = ContinuousToDiscreteActionWrapper(
                env,
                action_names=td3_action_names,
                action_threshold=td3_action_threshold,
            )
        else:
            env = FixObsSpaceWrapper(env)
        # Note: do NOT call env.reset() here — DummyVecEnv calls reset() on each env itself.

        return env
    return _init