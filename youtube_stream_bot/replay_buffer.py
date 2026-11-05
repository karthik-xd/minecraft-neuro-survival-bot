"""
replay_buffer.py - Fixed-size 50,000 capacity in-memory circular replay buffer.
Breaks temporal correlations across streaming episodes by drawing uniform random batches.
"""
import random
import torch
import config


class InactiveReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_CAPACITY):
        self.capacity = capacity
        # Stored in CPU memory to avoid VRAM exhaustion
        self.states = []         # Tensor: [12, 128, 128], float32
        self.buttons = []        # Tensor: [9], float32
        self.camera_deltas = []  # Tensor: [2], float32
        self.position = 0

    def push(self, state: torch.Tensor, buttons: torch.Tensor, camera_delta: torch.Tensor):
        """
        Stores state transition directly in RAM without writing to disk.
        state:        CPU float32 Tensor [12, 128, 128]
        buttons:      CPU float32 Tensor [9] (0.0 or 1.0)
        camera_delta: CPU float32 Tensor [2] (normalized [-1.0, 1.0])
        """
        if len(self.states) < self.capacity:
            self.states.append(state.cpu())
            self.buttons.append(buttons.cpu())
            self.camera_deltas.append(camera_delta.cpu())
        else:
            self.states[self.position] = state.cpu()
            self.buttons[self.position] = buttons.cpu()
            self.camera_deltas[self.position] = camera_delta.cpu()

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = config.BATCH_SIZE):
        """
        Uniform random non-consecutive batch sampling across the entire memory buffer.
        Returns Tensors moved to GPU / CPU target device:
          batch_states:  [B, 12, 128, 128], float32
          batch_buttons: [B, 9], float32
          batch_camera:  [B, 2], float32
        """
        indices = random.sample(range(len(self.states)), batch_size)

        batch_states = torch.stack([self.states[i] for i in indices]).to(config.DEVICE)
        batch_buttons = torch.stack([self.buttons[i] for i in indices]).to(config.DEVICE)
        batch_camera = torch.stack([self.camera_deltas[i] for i in indices]).to(config.DEVICE)

        return batch_states, batch_buttons, batch_camera

    def __len__(self):
        return len(self.states)
