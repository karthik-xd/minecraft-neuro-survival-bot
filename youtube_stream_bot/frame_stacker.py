"""
frame_stacker.py - Preprocesses individual frames and stacks the last 4 frames.
Output Tensor: [12, 128, 128], float32 (normalized to [0.0, 1.0]).
"""
import collections
import numpy as np
import torch
import config


class FrameStacker:
    def __init__(self, stack_size: int = config.STACK_SIZE):
        self.stack_size = stack_size
        self.buffer = collections.deque(maxlen=stack_size)

    def reset(self):
        """Clears the frame buffer on stream transitions."""
        self.buffer.clear()

    def add_frame(self, frame_np: np.ndarray) -> bool:
        """
        Receives raw numpy frame [H, W, C] = [128, 128, 3], uint8.
        Converts to PyTorch tensor [3, 128, 128], float32 in [0.0, 1.0].
        Returns True when the 4-frame stack is fully populated.
        """
        tensor_frame = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0  # [3, 128, 128]
        self.buffer.append(tensor_frame)
        return len(self.buffer) == self.stack_size

    def get_stacked_state(self) -> torch.Tensor:
        """
        Concatenates the 4 buffered frames along channel dimension (dim 0).
        Output Shape: [12, 128, 128] (4 frames * 3 channels = 12 channels)
        """
        if len(self.buffer) < self.stack_size:
            raise RuntimeError(f"Buffer has {len(self.buffer)} frames, requires {self.stack_size}.")
        return torch.cat(list(self.buffer), dim=0)
