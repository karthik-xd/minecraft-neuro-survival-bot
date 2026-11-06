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
        # Stored in CPU memory as uint8 (0-255) to conserve 75% RAM and prevent OOM
        self.states = []         # Tensor: [12, 128, 128], uint8
        self.buttons = []        # Tensor: [9], float32
        self.camera_deltas = []  # Tensor: [2], float32
        self.position = 0

    def push(self, state: torch.Tensor, buttons: torch.Tensor, camera_delta: torch.Tensor):
        """
        Stores state transition directly in RAM without writing to disk.
        Converts state to compact uint8 Tensor [12, 128, 128] (192 KB per transition).
        buttons:      CPU float32 Tensor [9] (0.0 or 1.0)
        camera_delta: CPU float32 Tensor [2] (normalized [-1.0, 1.0])
        """
        if state.dtype == torch.uint8:
            state_uint8 = state.cpu()
        else:
            state_uint8 = (state.cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        btn_cpu = buttons.cpu().float()
        cam_cpu = camera_delta.cpu().float()

        if len(self.states) < self.capacity:
            self.states.append(state_uint8)
            self.buttons.append(btn_cpu)
            self.camera_deltas.append(cam_cpu)
        else:
            self.states[self.position] = state_uint8
            self.buttons[self.position] = btn_cpu
            self.camera_deltas[self.position] = cam_cpu

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = config.BATCH_SIZE):
        """
        Uniform random non-consecutive batch sampling across the entire memory buffer.
        Returns float32 Tensors moved to GPU / CPU target device in range [0.0, 1.0]:
          batch_states:  [B, 12, 128, 128], float32
          batch_buttons: [B, 9], float32
          batch_camera:  [B, 2], float32
        """
        indices = random.sample(range(len(self.states)), batch_size)

        raw_stack = torch.stack([self.states[i] for i in indices])
        batch_states = (raw_stack.float() / 255.0).to(config.DEVICE)
        batch_buttons = torch.stack([self.buttons[i] for i in indices]).to(config.DEVICE)
        batch_camera = torch.stack([self.camera_deltas[i] for i in indices]).to(config.DEVICE)

        return batch_states, batch_buttons, batch_camera

    def sample_sequences(self, batch_size: int = config.BATCH_SIZE, seq_len: int = 8):
        """
        Samples contiguous sub-sequences of length `seq_len` for Recurrent BPTT.
        Trains GRU recurrent gates to maintain persistent hidden memory h_t across
        camera rotations and off-screen entity tracking, eliminating short-horizon amnesia.
        Returns:
          batch_states:  [B, seq_len, 12, 128, 128], float32
          batch_buttons: [B, seq_len, 9], float32
          batch_camera:  [B, seq_len, 2], float32
        """
        n = len(self.states)
        if n <= seq_len:
            states, buttons, camera = self.sample(batch_size)
            return states.unsqueeze(1), buttons.unsqueeze(1), camera.unsqueeze(1)

        valid_starts = []
        head = self.position if n == self.capacity else n

        for _ in range(batch_size * 3):
            idx = random.randint(0, n - seq_len)
            # Avoid the circular write boundary between old and new episodes
            if n == self.capacity and (idx <= head < idx + seq_len):
                continue
            valid_starts.append(idx)
            if len(valid_starts) >= batch_size:
                break

        while len(valid_starts) < batch_size:
            valid_starts.append(random.randint(0, max(0, n - seq_len)))

        seq_states = []
        seq_buttons = []
        seq_cameras = []

        for start_idx in valid_starts:
            s_chunk = torch.stack([self.states[(start_idx + t) % n] for t in range(seq_len)])
            b_chunk = torch.stack([self.buttons[(start_idx + t) % n] for t in range(seq_len)])
            c_chunk = torch.stack([self.camera_deltas[(start_idx + t) % n] for t in range(seq_len)])

            seq_states.append(s_chunk)
            seq_buttons.append(b_chunk)
            seq_cameras.append(c_chunk)

        batch_states = (torch.stack(seq_states).float() / 255.0).to(config.DEVICE)
        batch_buttons = torch.stack(seq_buttons).to(config.DEVICE)
        batch_camera = torch.stack(seq_cameras).to(config.DEVICE)

        return batch_states, batch_buttons, batch_camera

    def __len__(self):
        return len(self.states)
