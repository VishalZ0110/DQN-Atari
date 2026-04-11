import numpy as np
import torch
from collections import deque
import cv2

class FrameBuffer:
    def __init__(self, frame_limit=4):
        self.stack = deque(maxlen=frame_limit)

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.

    def add(self, frame):
        frame = self.preprocess_frame(frame)
        self.stack.append(frame)

    def get_stack(self):
        stack = np.stack(self.stack)
        return torch.from_numpy(stack).float()

class MultiEnvFrameBuffer:
    def __init__(self, num_envs, frame_limit=4):
        self.num_envs = num_envs
        self.frames = {env_id: deque(maxlen=frame_limit) for env_id in range(self.num_envs)}

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.

    def add(self, env_frames):
        for env_id in range(self.num_envs):
            frame = self.preprocess(env_frames[env_id])
            self.frames[env_id].append(frame)

    def get_stack(self):
        stack = np.stack([np.stack(frames) for _, frames in self.frames.items()])
        return torch.from_numpy(stack).float()

class MultiEnvReplayBuffer:
    def __init__(self, num_envs, state_shape=(4, 84, 84), max_size=1000000, batch_size=64, device="cpu"):
        self.num_envs = num_envs
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory with PyTorch tensors
        self.states = torch.zeros((max_size, *state_shape), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((max_size,), dtype=torch.uint8)
        self.rewards = torch.zeros((max_size,), dtype=torch.float32)
        self.dones = torch.zeros((max_size,), dtype=torch.bool)

    def add(self, state, action, reward, next_state, done):
        """Adds experiences efficiently using preallocated tensors."""
        idx = np.arange(self.ptr, self.ptr + self.num_envs) % self.max_size

        self.states[idx] = state
        self.actions[idx] = torch.from_numpy(action).to(dtype=torch.uint8)
        self.rewards[idx] = torch.from_numpy(reward).float()
        self.next_states[idx] = next_state
        self.dones[idx] = torch.from_numpy(done).to(dtype=torch.bool)

        self.ptr = (self.ptr + self.num_envs) % self.max_size
        self.size = min(self.size + self.num_envs, self.max_size)

    def sample(self):
        """Samples a batch and moves it to the device."""
        idxs = torch.randint(0, self.size, (self.batch_size,), dtype=torch.long)

        # Use non_blocking only for CUDA/MPS
        non_blocking = self.device in ['cuda', 'mps']

        return {
            "state": self.states[idxs].to(self.device, non_blocking=non_blocking),
            "action": self.actions[idxs].long().to(self.device, non_blocking=non_blocking),
            "reward": self.rewards[idxs].to(self.device, non_blocking=non_blocking),
            "next_state": self.next_states[idxs].to(self.device, non_blocking=non_blocking),
            "done": self.dones[idxs].float().to(self.device, non_blocking=non_blocking),
        }

    def is_ready(self):
        return self.size >= self.batch_size

    def __len__(self):
        return self.size
