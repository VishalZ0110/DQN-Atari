import os
import math
import numpy as np
import cv2
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(override=None):
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def env_short_name(env_name):
    """'ALE/MarioBros-v5' -> 'MarioBros'"""
    name = env_name.split("/")[-1]
    return name.split("-")[0]


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame / 255.0


def epsilon_greedy(q_online, envs, state, eps, device):
    if eps > np.random.uniform():
        return envs.action_space.sample()
    with torch.no_grad():
        state = state.float().div_(255.0).to(device)
        q_values = q_online(state)
        return q_values.argmax(dim=1).cpu().numpy()


def sync_weights(q_online, q_target):
    q_target.load_state_dict(q_online.state_dict())
    return q_online, q_target


def save_video(frames, path, fps=30):
    """Write a list of RGB numpy frames to an mp4 file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def save_grid_video(all_env_frames, path, fps=30, grid_cols=2):
    """
    Write a grid video from multiple environments.

    Parameters
    ----------
    all_env_frames : list[list[np.ndarray]]
        One list of RGB frames per environment.
    path : str
        Output mp4 path.
    fps : int
        Frames per second.
    grid_cols : int
        Number of columns in the grid.
    """
    num_envs = len(all_env_frames)
    grid_rows = math.ceil(num_envs / grid_cols)

    # Pad shorter episodes by repeating the last frame
    max_len = max(len(f) for f in all_env_frames)
    for i in range(num_envs):
        while len(all_env_frames[i]) < max_len:
            all_env_frames[i].append(all_env_frames[i][-1])

    h, w = all_env_frames[0][0].shape[:2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w * grid_cols, h * grid_rows))

    blank = np.zeros((h, w, 3), dtype=np.uint8)

    for t in range(max_len):
        rows = []
        for r in range(grid_rows):
            row_frames = []
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx < num_envs:
                    row_frames.append(all_env_frames[idx][t])
                else:
                    row_frames.append(blank)
            rows.append(np.concatenate(row_frames, axis=1))
        grid = np.concatenate(rows, axis=0)
        out.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    out.release()
