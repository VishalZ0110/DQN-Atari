"""
Single-environment demo with live playback or video save.

Usage:
    python -m dqn_atari.demo --config configs/mario_bros.yaml --checkpoint checkpoints/model.pt [--save-video] [--device cuda]
"""

import argparse
import torch
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from dqn_atari.model import DQN
from dqn_atari.buffers import FrameBuffer
from dqn_atari.utils import load_config, get_device, save_video



def demo(cfg, checkpoint_path, device, save_video_flag=False):
    env_cfg = cfg["env"]
    paths_cfg = cfg["paths"]

    render_mode = "rgb_array" if save_video_flag else "human"
    env = gym.make(env_cfg["name"], frameskip=env_cfg["frameskip"], render_mode=render_mode)

    model = DQN(num_actions=env_cfg["num_actions"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    fb = FrameBuffer()
    state, _ = env.reset()
    for _ in range(4):
        fb.add(state)

    frames = []
    total_reward = 0.0

    while True:
        with torch.no_grad():
            s = fb.get_stack().unsqueeze(0).to(device)
            action = model(s).argmax(dim=1).cpu().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        fb.add(next_state)

        if save_video_flag:
            frames.append(env.render())

        if terminated or truncated:
            break

    env.close()
    print(f"Total reward: {total_reward:.2f}")

    if save_video_flag and frames:
        import os
        os.makedirs(paths_cfg["video_dir"], exist_ok=True)
        video_path = os.path.join(paths_cfg["video_dir"], "demo.mp4")
        save_video(frames, video_path, fps=cfg.get("eval", {}).get("video_fps", 30))
        print(f"Demo video saved to {video_path}")

    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Demo DQN on Atari (single environment)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/mps/cpu)")
    parser.add_argument("--save-video", action="store_true", help="Save gameplay video instead of live playback")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"Using device: {device}")

    demo(cfg, args.checkpoint, device, save_video_flag=args.save_video)


if __name__ == "__main__":
    main()
