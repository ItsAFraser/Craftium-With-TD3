"""Quick sanity checks for reward reachability in Craftium ChopTree.

This script runs two short probes:
1) TD3-style dict actions on the unwrapped env (forward/dig/jump + continuous mouse sweep).
2) Random actions on the registered discrete env.

If both probes get zero positive rewards, reward is likely too sparse for current exploration
settings or there may be an environment setup issue.
"""

from __future__ import annotations

import argparse
import math
import time

import gymnasium as gym
import numpy as np
import craftium  # noqa: F401  # Needed for env registration side effects.

# The TD3-style probe uses the same dict action format and a simple heuristic policy that continuously digs while moving forward and sweeping the camera left and right.
# This is designed to have a decent chance of hitting tree trunks, which is critical for getting any positive reward signal in ChopTree.
# The random probe samples from the discrete action space registered by default, which is a more naive approach that may or may not get lucky enough to hit rewards within the short probe duration.
def run_td3_style_probe(steps: int, frameskip: int, sync_mode: bool, fps_max: int) -> tuple[int, int, float]:
    """Probe reward reachability via the same dict-action path TD3 uses."""
    env = gym.make(
        "Craftium/ChopTree-v0",
        frameskip=frameskip,
        rgb_observations=False,
        gray_scale_keepdim=True,
        sync_mode=sync_mode,
        fps_max=fps_max,
    ).unwrapped

    obs, info = env.reset()
    positive_hits = 0
    episodes = 0

    for t in range(steps):
        # Keep pressure on digging while moving and sweeping camera.
        # This approximates an exploratory policy that can actually hit tree trunks.
        sweep = math.sin(t * 0.08)
        # The action dict is structured according to the expected input of the ContinuousToDiscreteActionWrapper used in the TD3 training setup. The discrete actions are set based on the heuristic policy, while the continuous mouse control is set to create a sweeping motion that increases the chances of hitting tree trunks.
        action = {
            "forward": 1,
            "dig": 1,
            "jump": 1 if (t % 18 == 0) else 0,
            "left": 1 if (sweep > 0.6) else 0,
            "right": 1 if (sweep < -0.6) else 0,
            "mouse": np.array([0.85 * sweep, -0.18], dtype=np.float32),
        }
        # Step the environment with the heuristic action and check for positive rewards. We also track episode terminations to get a sense of how many distinct episodes we went through during the probe.
        obs, reward, terminated, truncated, info = env.step(action)
        if reward > 0:
            positive_hits += 1

        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()

    env.close()
    hit_rate = positive_hits / max(1, steps)
    return positive_hits, episodes, hit_rate


# The random probe samples actions from the discrete action space registered by default, which is a more naive approach that may or may not get lucky enough to hit rewards within the short probe duration. 
# This serves as a baseline to compare against the TD3-style probe, which uses a simple heuristic policy designed to have a better chance of hitting tree trunks and getting positive rewards.
def run_discrete_random_probe(steps: int) -> tuple[int, int, float]:
    """Probe reward reachability in the default registered discrete action env."""
    env = gym.make("Craftium/ChopTree-v0")
    obs, info = env.reset()

    positive_hits = 0
    episodes = 0
    # In this probe, we simply sample random actions from the discrete action space and step through the environment, 
    # counting how many times we get positive rewards and how many episodes we go through. This is a more naive approach compared to the TD3-style probe, but it can still provide insight into whether rewards are reachable at all with random exploration.
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if reward > 0:
            positive_hits += 1
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()

    env.close()
    hit_rate = positive_hits / max(1, steps)
    return positive_hits, episodes, hit_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500, help="Probe steps per test.")
    parser.add_argument("--frameskip", type=int, default=3, help="Frameskip for TD3-style probe.")
    parser.add_argument("--sync-mode", action="store_true", help="Use sync_mode for TD3-style probe.")
    parser.add_argument("--fps-max", type=int, default=30, help="fps_max for TD3-style probe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Running TD3-style probe...")
    t0 = time.time()
    td3_hits, td3_episodes, td3_rate = run_td3_style_probe(
        steps=args.steps,
        frameskip=args.frameskip,
        sync_mode=args.sync_mode,
        fps_max=args.fps_max,
    )
    t1 = time.time()

    print(
        f"td3_style: steps={args.steps} positive_reward_steps={td3_hits} "
        f"episodes={td3_episodes} hit_rate={td3_rate:.6f} elapsed_s={t1 - t0:.2f}"
    )

    print("Running discrete-random probe...")
    t2 = time.time()
    disc_hits, disc_episodes, disc_rate = run_discrete_random_probe(steps=args.steps)
    t3 = time.time()

    print(
        f"discrete_random: steps={args.steps} positive_reward_steps={disc_hits} "
        f"episodes={disc_episodes} hit_rate={disc_rate:.6f} elapsed_s={t3 - t2:.2f}"
    )

    if td3_hits == 0 and disc_hits == 0:
        print("RESULT: zero rewards in both probes -> focus on stronger exploration/curriculum first.")
    elif td3_hits == 0 and disc_hits > 0:
        print("RESULT: discrete gets reward but td3-style does not -> action mapping/exploration is the bottleneck.")
    else:
        print("RESULT: reward is reachable via td3-style interaction -> tune exploration and training schedule.")


if __name__ == "__main__":
    main()
