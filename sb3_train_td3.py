from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common import logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any
from argparse import ArgumentParser
from uuid import uuid4
import os
import craftium # This import is used even though the VSCode says it isn't!
import matplotlib.pyplot as plt

from td3_gumbel_policy import GumbelMapperTD3Policy
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
    parser.add_argument("--frameskip", type=int, default=4,
        help="Craftium frameskip. Higher values increase speed but reduce control granularity.")
    parser.add_argument("--sync-mode", action="store_true",
        help="Enable Craftium sync mode. Off by default for faster training.")
    parser.add_argument("--fps-max", type=int, default=200,
        help="Craftium fps cap. Use a high value for faster simulation throughput.")
    parser.add_argument("--frame-stack", type=int, default=2,
        help="Number of stacked frames for temporal context.")
    # Adding td3 as an option here
    parser.add_argument("--method", type=str, default="a2c", choices=["ppo", "a2c", "td3"],
        help="RL method to use to optimize the agent.")
    parser.add_argument("--td3-actions", type=str, default="forward,left,right,jump,dig",
        help="Comma-separated discrete actions TD3 can press. Keep small for sparse tasks like ChopTree.")
    parser.add_argument("--td3-action-threshold", type=float, default=0.2,
        help="Press a discrete action when its score is above this threshold.")
    parser.add_argument("--td3-noise-sigma", type=float, default=0.10,
        help="Stddev for TD3 exploration noise.")
    parser.add_argument("--td3-learning-rate", type=float, default=1e-4,
        help="TD3 optimizer learning rate.")
    parser.add_argument("--td3-learning-starts", type=int, default=5_000,
        help="Steps to collect before TD3 updates start.")
    parser.add_argument("--td3-buffer-size", type=int, default=100_000,
        help="Replay buffer size for TD3.")
    parser.add_argument("--td3-batch-size", type=int, default=128,
        help="Minibatch size for TD3 updates.")
    parser.add_argument("--td3-train-freq", type=int, default=80,
        help="TD3 training frequency in env steps.")
    parser.add_argument("--td3-gradient-steps", type=int, default=8,
        help="Number of gradient steps per TD3 training phase.")
    parser.add_argument("--bc-demo-steps", type=int, default=0,
        help="Number of heuristic demo steps to collect for TD3 behavior cloning warm-start. 0 disables BC.")
    parser.add_argument("--bc-pretrain-gradient-steps", type=int, default=0,
        help="Number of supervised gradient steps to pretrain TD3 actor on demos. 0 disables BC.")
    parser.add_argument("--bc-batch-size", type=int, default=256,
        help="Batch size for TD3 behavior cloning warm-start.")
    parser.add_argument("--bc-log-interval", type=int, default=200,
        help="How often to print BC pretraining loss.")
    parser.add_argument("--bc-demo-file", type=str, default=None,
        help="Optional .npz file with demo_obs and demo_actions arrays for BC warm-start.")
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

    def __init__(self, env, action_names, action_threshold=0.2):
        super().__init__(env)
        # Number of discrete actions controlled by TD3. The remaining 2 dimensions are
        # reserved for continuous mouse control, which passes through without discretization.
        self.action_names = list(action_names)
        self.num_actions = len(self.action_names)
        self.action_threshold = action_threshold
        if self.num_actions == 0:
            raise ValueError("action_names must contain at least one action")

        # Action vector layout (output by JointGaussianMapperActor):
        # [0:num_actions] -> mapped action scores in [-1, 1]
        # [num_actions:num_actions+2] -> mouse x,y (continuous pass-through)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions + 2,),
            dtype=np.float32
        )
    
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
            if score > self.action_threshold:
                action_dict[self.action_names[i]] = 1

        # Keep mouse control continuous as requested.
        action_dict["mouse"] = np.array(mouse_vals, dtype=np.float32)
        return action_dict


#the Make_env function creates a factory function that initializes the environment
#with the appropriate wrappers based on the specified method (PPO, A2C, or TD3).
#For TD3, it layers the FixObsSpaceWrapper and ContinuousToDiscreteActionWrapper on 
#top of the base Craftium environment to ensure compatibility with TD3's requirements.
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
        if method == "td3":
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

# The JointGaussianMapperActor is defined in td3_joint_policy.py, which is imported at the top.
# It implements a deterministic actor network for TD3 that maps CNN features to action scores, which are then converted to discrete actions by the ContinuousToDiscreteActionWrapper. The JointGaussianMapperTD3Policy class specifies that this actor should be used as the policy's actor network.
def td3_heuristic_action(step_idx: int, action_names: list[str]) -> np.ndarray:
    """Heuristic action used to collect coarse demonstrations for BC warm-start."""
    action = -1.0 * np.ones((len(action_names) + 2,), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(action_names)}

    # This heuristic is designed to have a decent chance of hitting tree trunks while sweeping the camera around, which is critical for getting any positive reward signal in ChopTree. It's not meant to be a strong policy, just a way to get some initial TD3 learning signal via behavior cloning.
    if "forward" in name_to_idx:
        action[name_to_idx["forward"]] = 1.0
    if "dig" in name_to_idx:
        action[name_to_idx["dig"]] = 1.0
    if "jump" in name_to_idx and (step_idx % 18 == 0):
        action[name_to_idx["jump"]] = 1.0

    # Sweep camera left and right to increase chances of hitting tree trunks. The sweep is a simple sine wave pattern that oscillates over time, creating a back-and-forth motion. The thresholds for left and right are set to 0.5 and -0.5 respectively, which means the heuristic will trigger left or right actions when the sine wave exceeds those values.
    sweep = np.sin(step_idx * 0.08)
    if "left" in name_to_idx and sweep > 0.5:
        action[name_to_idx["left"]] = 1.0
    if "right" in name_to_idx and sweep < -0.5:
        action[name_to_idx["right"]] = 1.0

    # Mouse control is continuous and passed through directly, so we set it based on the same sweep pattern to create a coordinated camera motion. The mouse x value oscillates with the sweep, while the mouse y value is set to a constant negative value to keep the camera angled downward towards the ground where the tree trunks are.
    action[-2] = np.float32(0.85 * sweep)
    action[-1] = np.float32(-0.18)
    return action

# This function collects demonstration observation-action pairs from a simple scripted policy defined by td3_heuristic_action.
# It runs the specified number of steps in the environment, applying the heuristic policy to generate actions, and stores the corresponding observations and actions in batches.
# Finally, it concatenates the batches into single arrays for observations and actions, which can be used for behavior cloning pretraining of the TD3 actor.
def collect_td3_heuristic_demos(envs, action_names: list[str], total_steps: int):
    """Collect observation-action pairs from a simple scripted policy."""
    obs = envs.reset()
    n_envs = envs.num_envs
    action_dim = len(action_names) + 2

    obs_batches = []
    action_batches = []

    # Loop through the specified number of steps, applying the heuristic policy to generate actions for each environment in the vectorized setup.
    for t in range(total_steps):
        # Generate actions for each environment based on the heuristic policy, which depends on the current step index and the defined action names.
        actions = np.zeros((n_envs, action_dim), dtype=np.float32)
        for env_idx in range(n_envs):
            actions[env_idx] = td3_heuristic_action(t + 17 * env_idx, action_names)
        # Store the current observations and actions in batches. We make copies to ensure that we don't accidentally overwrite data due to mutable references.
        obs_batches.append(np.array(obs, copy=True))
        action_batches.append(np.array(actions, copy=True))
        # Step the environments with the generated actions to get the next set of observations. We ignore rewards, terminations, and truncations here since we're only interested in collecting demo data.
        obs, _, _, _ = envs.step(actions)
    # After collecting the specified number of steps, we concatenate the batches of observations and actions into single arrays that can be used for behavior cloning pretraining.
    demo_obs = np.concatenate(obs_batches, axis=0)
    demo_actions = np.concatenate(action_batches, axis=0)
    return demo_obs, demo_actions

# This function performs supervised pretraining of the TD3 actor using demonstration observation-action pairs. 
# It samples batches from the demo data, computes the mean squared error loss between the actor's predicted actions and the demo actions,
# and updates the actor's parameters accordingly. After pretraining, it synchronizes the target actor with the updated actor weights to ensure 
# consistency when TD3 training starts.
def pretrain_td3_actor_with_bc(
    model: TD3,
    demo_obs: np.ndarray,
    demo_actions: np.ndarray,
    gradient_steps: int,
    batch_size: int,
    log_interval: int,
):
    """Supervised warm-start for the TD3 actor using demo observation-action pairs."""
    n_samples = demo_obs.shape[0]
    if n_samples == 0 or gradient_steps <= 0:
        return
# Set actor to training mode for BC pretraining. TD3's learn() will switch it back to eval mode when actual training starts.
    model.actor.set_training_mode(True)

# Perform supervised gradient steps on the actor using the demo data. 
# We sample random batches of demo transitions,
# compute the MSE loss between the actor's predicted actions and the demo actions,
# and update the actor's parameters using backpropagation.
# We also log the BC loss at regular intervals for monitoring.
    for step in range(1, gradient_steps + 1):
        # Sample a random batch of demo transitions
        idx = np.random.randint(0, n_samples, size=min(batch_size, n_samples))
        obs_batch = demo_obs[idx]
        action_batch = demo_actions[idx]

        # Convert the batch of observations to a tensor and get the actor's predicted actions.
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        pred_actions = model.actor(obs_tensor)
        target_actions = th.as_tensor(action_batch, device=model.device, dtype=pred_actions.dtype)
       
        # Compute the MSE loss between the predicted actions and the demo actions.
        bc_loss = F.mse_loss(pred_actions, target_actions)
       
        # Backpropagate the loss and update the actor's parameters.
        model.actor.optimizer.zero_grad()
        bc_loss.backward()
        th.nn.utils.clip_grad_norm_(model.actor.parameters(), max_norm=1.0)
        model.actor.optimizer.step()
        
        # Log the BC loss at regular intervals.
        if step % max(1, log_interval) == 0 or step == gradient_steps:
            print(f"[BC] step={step}/{gradient_steps} loss={bc_loss.item():.6f}")

    # Keep target actor in sync after warm-start.
    model.actor_target.load_state_dict(model.actor.state_dict())


def load_bc_demo_file(demo_file: str):
    """Load demonstration arrays from an .npz file."""
    data = np.load(demo_file)
    if "demo_obs" not in data or "demo_actions" not in data:
        raise ValueError(f"{demo_file} must contain demo_obs and demo_actions arrays")
    demo_obs = data["demo_obs"]
    demo_actions = data["demo_actions"]
    if demo_obs.shape[0] != demo_actions.shape[0]:
        raise ValueError(
            f"Mismatched demo lengths in {demo_file}: "
            f"demo_obs={demo_obs.shape[0]} demo_actions={demo_actions.shape[0]}"
        )
    return demo_obs, demo_actions

#Main training Loop. Configures the logger, creates the vectorized environment with the appropriate wrappers
#initializes the model based on a specific method (PPO., A2C, or TD3), and starts the learning process for a given number ot timesteps.
if __name__ == "__main__":
    args = parse_args()
    td3_action_names = [a.strip() for a in args.td3_actions.split(",") if a.strip()]
    if not td3_action_names:
        raise ValueError("--td3-actions must include at least one action")

    # Generate a unique run name if not provided, and configure the SB3 logger to save logs in the specified directory.
    if args.run_name is None:
        run_name = f"{args.env_id.replace('/', '-')}-{args.method}--{str(uuid4())}"
    else:
        run_name = args.run_name

    # configure SB3 logger
    log_path = os.path.join(args.runs_dir, run_name) # save logs in runs_dir/run_name
    print(f"** Storing run's data in {log_path}")
    new_logger = logger.configure(log_path, ["stdout", "csv"])  # log to both console and CSV file for later analysis

    print(f"Using method {args.method} with {args.total_timesteps} timesteps")
    envs = DummyVecEnv([
        make_env(
            args.env_id,
            args.method,
            td3_action_names=td3_action_names,
            td3_action_threshold=args.td3_action_threshold,
            frameskip=args.frameskip,
            sync_mode=args.sync_mode,
            fps_max=args.fps_max,
        )
        for _ in range(args.num_envs)
    ]) # Create a vectorized environment with the specified number of parallel environments, each initialized with the appropriate wrappers based on the method.
    envs = VecFrameStack(envs, args.frame_stack) # Stack a short history for motion cues while keeping memory/compute manageable.
    envs = VecMonitor(envs) # Monitor wrapper to track episode rewards, lengths, and other metrics across the vectorized environments. 


    #if PPO or A2C, initialize with the standard CNN policy. if TD3, initialize with the custom JointGaussianMapperTD3Policy
    #that maps CNN features to continuous action scores, which are then converted to discrete actions by the ContinuousToDiscreteActionWrapper.
    if args.method == "ppo":
        model = PPO("CnnPolicy", envs, verbose=1)
    elif args.method == "a2c":
        model = A2C("CnnPolicy", envs, verbose=1)
    else:  # TD3
        # TD3 actor deterministically maps CNN features through a trainable NN to action
        # scores. Wrapper fires actions above a configurable threshold + mouse passthrough.
        if not isinstance(envs.action_space, spaces.Box):
            raise TypeError(f"TD3 requires a Box action space, got {type(envs.action_space)}")
        n_actions = envs.action_space.shape[0]
        # Actor is deterministic, so exploration comes
        # from this noise only (no internal Gaussian sampling adding on top).
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.td3_noise_sigma * np.ones(n_actions),
        )

        # TD3 hyperparameters! Mostly found in td3.py in the SB3 repo, but I decreased some in a somewhat vain
        # attempt to help speed up training.
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py
        model = TD3(
            GumbelMapperTD3Policy,
            envs,
            learning_rate=args.td3_learning_rate,
            #action_noise = action_noise, # remove noise since gumbel does this already
            verbose = 1, # Print TD3's own debug info (e.g. actor/critic losses) to console. 1 is SB3's default; 0 would disable, 2 would be more verbose.
            learning_starts = args.td3_learning_starts,
            buffer_size = args.td3_buffer_size,
            batch_size = args.td3_batch_size,
            train_freq = args.td3_train_freq,
            gradient_steps = args.td3_gradient_steps,
        )

        if args.bc_pretrain_gradient_steps > 0:
            demo_obs, demo_actions = None, None
            if args.bc_demo_file is not None:
                print(f"[BC] Loading demos from {args.bc_demo_file}...")
                demo_obs, demo_actions = load_bc_demo_file(args.bc_demo_file)
                print(f"[BC] Loaded {demo_obs.shape[0]} demo transitions from file.")
            elif args.bc_demo_steps > 0:
                print(f"[BC] Collecting {args.bc_demo_steps} heuristic demo steps...")
                demo_obs, demo_actions = collect_td3_heuristic_demos(
                    envs=envs,
                    action_names=td3_action_names,
                    total_steps=args.bc_demo_steps,
                )
                print(f"[BC] Collected {demo_obs.shape[0]} demo transitions.")

            if demo_obs is not None:
                print("[BC] Starting actor pretraining...")
                pretrain_td3_actor_with_bc(
                    model=model,
                    demo_obs=demo_obs,
                    demo_actions=demo_actions,
                    gradient_steps=args.bc_pretrain_gradient_steps,
                    batch_size=args.bc_batch_size,
                    log_interval=args.bc_log_interval,
                )
                print("[BC] Warm-start complete.")
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.total_timesteps)

    envs.close()