"""
Config-driven DQN training script.

Usage:
    python -m dqn_atari.train --config configs/mario_bros.yaml [--device cuda]
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gymnasium as gym

from dqn_atari.model import DQN
from dqn_atari.buffers import FrameBuffer, MultiEnvFrameBuffer, MultiEnvReplayBuffer
from dqn_atari.scheduler import EpsilonScheduler
from dqn_atari.env import make_env
from dqn_atari.utils import (
    load_config,
    get_device,
    env_short_name,
    epsilon_greedy,
    sync_weights,
    save_video,
)


def eval_single_env_video(model, env_name, frameskip, device, video_path, fps=30):
    """Run one episode in a single env and save the gameplay video."""
    eval_env = gym.make(env_name, frameskip=frameskip, render_mode="rgb_array")
    fb = FrameBuffer()
    state, _ = eval_env.reset()
    for _ in range(4):
        fb.add(state)

    frames = []
    model.eval()
    total_reward = 0.0

    while True:
        with torch.no_grad():
            s = fb.get_stack().unsqueeze(0).to(device)
            action = model(s).argmax(dim=1).cpu().item()
        next_state, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        frames.append(eval_env.render() if eval_env.render_mode == "rgb_array" else next_state)
        fb.add(next_state)
        if terminated or truncated:
            break

    eval_env.close()
    save_video(frames, video_path, fps=fps)
    model.train()
    return total_reward


def eval_batch_reward(model, env_name, frameskip, num_envs, device):
    """Run one episode per env in a vectorised batch, return mean reward (no video)."""
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_name, frameskip, seed=s) for s in range(num_envs)]
    )
    fb = MultiEnvFrameBuffer(num_envs)
    states, _ = envs.reset()
    for _ in range(4):
        fb.add(states)

    model.eval()
    env_rewards = np.zeros(num_envs)
    dones = np.zeros(num_envs, dtype=bool)

    while not dones.all():
        with torch.no_grad():
            stack = fb.get_stack().float().div_(255.0).to(device)
            actions = model(stack).argmax(dim=1).cpu().numpy()
        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        env_rewards += rewards * (~dones)
        step_dones = np.logical_or(terminateds, truncateds)
        dones = np.logical_or(dones, step_dones)
        fb.add(next_states)

    envs.close()
    model.train()
    return float(env_rewards.mean())


def train(cfg, device):
    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    eps_cfg = cfg["epsilon"]
    eval_cfg = cfg["eval"]
    paths_cfg = cfg["paths"]

    short_name = env_short_name(env_cfg["name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{short_name}_{timestamp}"

    run_checkpoint_dir = os.path.join(paths_cfg["checkpoint_dir"], run_name)
    run_video_dir = os.path.join(paths_cfg["video_dir"], run_name)
    train_eval_dir = os.path.join(run_video_dir, "train_eval")

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(train_eval_dir, exist_ok=True)
    print(f"Run: {run_name}")
    print(f"  Checkpoints -> {run_checkpoint_dir}")
    print(f"  Videos      -> {run_video_dir}")

    num_envs = env_cfg["num_envs"]
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_cfg["name"], env_cfg["frameskip"], seed=s) for s in range(num_envs)]
    )
    frame_buffer = MultiEnvFrameBuffer(num_envs)
    replay_buffer = MultiEnvReplayBuffer(
        num_envs,
        max_size=train_cfg["buffer_size"],
        batch_size=train_cfg["batch_size"],
        device=device,
    )

    q_online = DQN(num_actions=env_cfg["num_actions"]).to(device)
    q_target = DQN(num_actions=env_cfg["num_actions"]).to(device)
    optimizer = torch.optim.Adam(q_online.parameters(), lr=train_cfg["learning_rate"])
    q_online, q_target = sync_weights(q_online, q_target)

    eps_scheduler = EpsilonScheduler(
        min_eps=eps_cfg["min"],
        max_eps=eps_cfg["max"],
        total_steps=train_cfg["total_steps"],
        exploration_frac=eps_cfg["exploration_frac"],
    )

    discount = train_cfg["discount_rate"]
    reward_scale = train_cfg["reward_scale"]
    life_loss_penalty = train_cfg["life_loss_penalty"]
    update_freq = train_cfg["target_update_freq"]
    eval_freq = train_cfg["eval_freq"]

    episode_rewards = []
    episode_losses = []

    done = False
    episode_loss = 0.0
    episode_reward = 0.0
    episode_step = 0
    num_episodes = 0
    training = False
    current_lives = np.zeros(num_envs, dtype=np.int32)

    prog_bar = tqdm(range(train_cfg["total_steps"]))
    for global_step in prog_bar:

        if done or global_step == 0:
            states, infos = envs.reset()
            current_lives = infos["lives"]

            for _ in range(4):
                frame_buffer.add(states)

            if global_step > 0:
                episode_losses.append(episode_loss / max(1, episode_step))
                episode_rewards.append(episode_reward)
                episode_loss, episode_reward, episode_step = 0.0, 0.0, 0

            num_episodes += 1
            done = False

        eps = eps_scheduler(global_step)

        current_stack = frame_buffer.get_stack()
        actions = epsilon_greedy(q_online, envs, current_stack, eps, device)

        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        rewards = rewards / reward_scale

        mask = infos["lives"] < current_lives
        rewards[mask] += life_loss_penalty
        current_lives[mask] = infos["lives"][mask]

        frame_buffer.add(next_states)
        next_stack = frame_buffer.get_stack()
        dones = np.logical_or(terminateds, truncateds)

        replay_buffer.add(current_stack, actions, rewards, next_stack, dones)

        episode_step += 1
        episode_reward += np.mean(rewards)
        done = np.any(terminateds) or np.any(truncateds)

        # --- DQN update (Double DQN) ---
        if len(replay_buffer) >= train_cfg["buffer_size"]:
            training = True
            batch = replay_buffer.sample()
            b_states = batch["state"]
            b_actions = batch["action"]
            b_rewards = batch["reward"]
            b_next_states = batch["next_state"]
            b_dones = batch["done"]

            predicted_q = q_online(b_states).gather(1, b_actions.unsqueeze(1)).flatten()
            best_actions = q_online(b_next_states).argmax(dim=1)
            next_q = q_target(b_next_states).gather(1, best_actions.unsqueeze(1)).flatten()
            target_q = b_rewards + discount * (1 - b_dones) * next_q

            loss = F.smooth_l1_loss(predicted_q, target_q)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            episode_loss += loss.item()

        # --- Target network sync ---
        if global_step % update_freq == 0:
            q_online, q_target = sync_weights(q_online, q_target)

        # --- Mid-training evaluation (only after training has started) ---
        if training and global_step % eval_freq == 0 and global_step > 0:
            ckpt_path = os.path.join(
                run_checkpoint_dir, f"step_{global_step}.pt"
            )
            torch.save(q_online.state_dict(), ckpt_path)

            if eval_cfg["save_video"]:
                vid_path = os.path.join(
                    train_eval_dir, f"step_{global_step}.mp4"
                )
                eval_reward = eval_single_env_video(
                    q_online,
                    env_cfg["name"],
                    env_cfg["frameskip"],
                    device,
                    vid_path,
                    fps=eval_cfg["video_fps"],
                )
                tqdm.write(f"[eval step {global_step}] video saved | reward: {eval_reward:.1f}")
            else:
                avg_r = eval_batch_reward(
                    q_online, env_cfg["name"], env_cfg["frameskip"], num_envs, device
                )
                tqdm.write(f"[eval step {global_step}] avg reward: {avg_r:.1f}")

        prog_bar.set_description(
            f"{training=} ep={num_episodes} | "
            f"loss: {episode_loss / max(1, episode_step):.3f} | "
            f"reward: {episode_reward:.2f} | "
            f"buf: {len(replay_buffer)} | eps: {eps:.3f}"
        )

    # --- Cleanup ---
    envs.close()

    final_ckpt = os.path.join(run_checkpoint_dir, "final.pt")
    torch.save(q_online.state_dict(), final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(episode_rewards)
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[1].plot(episode_losses)
    axes[1].set_title("Episode Losses")
    axes[1].set_xlabel("Episode")
    plt.tight_layout()
    plot_path = os.path.join(run_video_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Atari")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/mps/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train(cfg, device)


if __name__ == "__main__":
    main()
