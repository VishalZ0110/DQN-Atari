"""
Multi-environment evaluation with grid video output and average rewards.

Usage:
    python -m dqn_atari.evaluate --config configs/mario_bros.yaml --checkpoint checkpoints/model.pt [--device cuda]
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym

from dqn_atari.model import DQN
from dqn_atari.buffers import MultiEnvFrameBuffer
from dqn_atari.env import make_env
from dqn_atari.utils import load_config, get_device, env_short_name, save_grid_video


def evaluate(cfg, checkpoint_path, device):
    env_cfg = cfg["env"]
    eval_cfg = cfg["eval_full"]
    paths_cfg = cfg["paths"]

    num_envs = eval_cfg["num_envs"]
    grid_cols = eval_cfg["grid_cols"]
    fps = eval_cfg["video_fps"]
    short_name = env_short_name(env_cfg["name"])

    os.makedirs(paths_cfg["video_dir"], exist_ok=True)

    model = DQN(num_actions=env_cfg["num_actions"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(env_cfg["name"], env_cfg["frameskip"], seed=s, render_mode="rgb_array")
            for s in range(num_envs)
        ]
    )

    fb = MultiEnvFrameBuffer(num_envs)
    states, _ = envs.reset()
    for _ in range(4):
        fb.add(states)

    # Collect per-env frames and rewards
    all_env_frames = [[] for _ in range(num_envs)]
    env_rewards = np.zeros(num_envs)
    dones = np.zeros(num_envs, dtype=bool)

    # Store initial rendered frames
    rendered = envs.call("render")
    for i in range(num_envs):
        all_env_frames[i].append(rendered[i])

    while not dones.all():
        with torch.no_grad():
            stack = fb.get_stack().float().div_(255.0).to(device)
            actions = model(stack).argmax(dim=1).cpu().numpy()

        # For envs already done, send a no-op (action 0) to avoid errors
        final_actions = np.where(dones, 0, actions)
        next_states, rewards, terminateds, truncateds, _ = envs.step(final_actions)

        env_rewards += rewards * (~dones)
        step_dones = np.logical_or(terminateds, truncateds)

        rendered = envs.call("render")
        for i in range(num_envs):
            if not dones[i]:
                all_env_frames[i].append(rendered[i])

        dones = np.logical_or(dones, step_dones)
        fb.add(next_states)

    envs.close()

    avg_reward = float(env_rewards.mean())
    print(f"Average reward across {num_envs} environments: {avg_reward:.2f}")
    for i, r in enumerate(env_rewards):
        print(f"  Env {i}: reward = {r:.2f} | frames = {len(all_env_frames[i])}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(paths_cfg["video_dir"], f"eval_{short_name}_{timestamp}.mp4")
    save_grid_video(all_env_frames, video_path, fps=fps, grid_cols=grid_cols)
    print(f"Grid video saved to {video_path}")

    return avg_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN on Atari (multi-env grid video)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/mps/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"Using device: {device}")

    evaluate(cfg, args.checkpoint, device)


if __name__ == "__main__":
    main()
