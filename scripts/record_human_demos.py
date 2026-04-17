# Doesnt work yet, since it needs to run outside of the docker container to capture keyboard input.
# This is meant to be a simple script for recording human demonstrations that can be used to warm-start TD3 training via behavior cloning. 
# It uses matplotlib to display the environment observations and capture keyboard input, which is then converted into the same action format used by the TD3 training setup.
# The recorded observations and actions are saved in a compressed .npz file that can be loaded during TD3 training for BC warm-starting.


"""Record human demonstrations for TD3 BC warm-start.

Controls:
- Movement: W/A/S/D
- Jump: Space
- Sneak: Left Shift
- Dig: F
- Place: E
- Slots: 1..5
- Mouse-look via keys: J/L (x- / x+), I/K (y+ / y-)
- Quit: Q or close window

This script displays environment observations with matplotlib and stores arrays in an .npz file
that can be consumed by sb3_train_td3.py via --bc-demo-file.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import craftium  # noqa: F401  # Needed for env registration side effects.


@dataclass
class KeyState:
    pressed: set[str] = field(default_factory=set)

    def is_down(self, key: str) -> bool:
        return key in self.pressed

# The JointGaussianMapperActor is defined in td3_joint_policy.py, which is imported at the top.
# It implements a deterministic actor network for TD3 that maps CNN features to action scores, which is then converted to discrete actions by the ContinuousToDiscreteActionWrapper. The JointGaussianMapperTD3Policy class specifies that this actor should be used as the policy's actor network.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Craftium/ChopTree-v0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--outfile", type=str, default="results/human_demos_choptree.npz")
    parser.add_argument("--frameskip", type=int, default=3)
    parser.add_argument("--sync-mode", action="store_true")
    parser.add_argument("--fps-max", type=int, default=30)
    parser.add_argument(
        "--td3-actions",
        type=str,
        default="forward,left,right,jump,dig",
        help="Comma-separated discrete actions represented in demo action vectors.",
    )
    parser.add_argument(
        "--mouse-step",
        type=float,
        default=0.8,
        help="Absolute value used for mouse x/y command when look keys are pressed.",
    )
    parser.add_argument(
        "--action-off",
        type=float,
        default=-1.0,
        help="Value for non-pressed discrete actions in the TD3 action vector.",
    )
    parser.add_argument(
        "--action-on",
        type=float,
        default=1.0,
        help="Value for pressed discrete actions in the TD3 action vector.",
    )
    return parser.parse_args()


def normalize_key(event_key: str | None) -> str | None:
    if event_key is None:
        return None
    key = event_key.lower()
    if key == " ":
        return "space"
    if key in {"shift", "shift_l", "shift_r"}:
        return "shift"
    return key


def build_td3_action_vector(
    key_state: KeyState,
    action_names: list[str],
    action_on: float,
    action_off: float,
    mouse_step: float,
) -> np.ndarray:
    key_map = {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "jump": "space",
        "sneak": "shift",
        "dig": "f",
        "place": "e",
        "slot_1": "1",
        "slot_2": "2",
        "slot_3": "3",
        "slot_4": "4",
        "slot_5": "5",
        "slot_6": "6",
        "slot_7": "7",
        "slot_8": "8",
        "slot_9": "9",
    }

    vec = np.full((len(action_names) + 2,), action_off, dtype=np.float32)
    for i, name in enumerate(action_names):
        key = key_map.get(name)
        if key is not None and key_state.is_down(key):
            vec[i] = np.float32(action_on)

    mouse_x = 0.0
    mouse_y = 0.0
    if key_state.is_down("j"):
        mouse_x -= mouse_step
    if key_state.is_down("l"):
        mouse_x += mouse_step
    if key_state.is_down("i"):
        mouse_y += mouse_step
    if key_state.is_down("k"):
        mouse_y -= mouse_step

    vec[-2] = np.float32(np.clip(mouse_x, -1.0, 1.0))
    vec[-1] = np.float32(np.clip(mouse_y, -1.0, 1.0))
    return vec


def main() -> None:
    args = parse_args()
    action_names = [a.strip() for a in args.td3_actions.split(",") if a.strip()]
    if not action_names:
        raise ValueError("--td3-actions must include at least one action")

    env = gym.make(
        args.env_id,
        frameskip=args.frameskip,
        rgb_observations=False,
        gray_scale_keepdim=True,
        sync_mode=args.sync_mode,
        fps_max=args.fps_max,
    ).unwrapped

    obs, info = env.reset()
    key_state = KeyState()
    should_quit = {"value": False}

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(np.squeeze(obs), cmap="gray")
    ax.set_title("Human demo recording (Q to quit)")
    ax.axis("off")

    def on_key_press(event):
        key = normalize_key(event.key)
        if key is None:
            return
        if key == "q":
            should_quit["value"] = True
            return
        key_state.pressed.add(key)

    def on_key_release(event):
        key = normalize_key(event.key)
        if key is None:
            return
        key_state.pressed.discard(key)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    demo_obs = []
    demo_actions = []

    print("Recording started. Focus the plot window and use keyboard controls.")
    print("Press Q in the plot window to stop early.")

    for t in range(args.steps):
        if should_quit["value"]:
            break

        action_vec = build_td3_action_vector(
            key_state=key_state,
            action_names=action_names,
            action_on=args.action_on,
            action_off=args.action_off,
            mouse_step=args.mouse_step,
        )

        demo_obs.append(np.array(obs, copy=True))
        demo_actions.append(np.array(action_vec, copy=True))

        obs, reward, terminated, truncated, info = env.step(action_vec)

        frame = np.squeeze(obs)
        img.set_data(frame)
        ax.set_title(f"step={t+1} reward={reward:.3f}")
        plt.pause(0.001)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    plt.close(fig)

    if len(demo_obs) == 0:
        print("No data recorded. Nothing saved.")
        return

    demo_obs_arr = np.stack(demo_obs, axis=0)
    demo_actions_arr = np.stack(demo_actions, axis=0)

    np.savez_compressed(
        args.outfile,
        demo_obs=demo_obs_arr,
        demo_actions=demo_actions_arr,
        action_names=np.array(action_names, dtype=object),
    )
    print(f"Saved {demo_obs_arr.shape[0]} demonstrations to {args.outfile}")


if __name__ == "__main__":
    main()
