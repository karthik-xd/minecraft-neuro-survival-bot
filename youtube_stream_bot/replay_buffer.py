"""
replay_buffer.py - Fixed-capacity in-memory circular replay buffer.
Updated for OpenAI VPT Parity:
  - Thread-safe operations (threading.Lock).
  - Supports 21-button multi-binary actions.
  - Supports 121-class categorical camera labels (with automatic conversion from continuous dx/dy).
  - Mathematically sound circular write boundary handling in sample_sequences()
    that strictly prevents splicing across the newest-to-oldest circular head boundary.
  - Safe sampling with replacement when buffer has fewer elements than batch_size or seq_len.
  - Supports sequence length T=16 and T=32 for recurrent BPTT.
"""
import random
import threading
import torch
import config


class InactiveReplayBuffer:
    def __init__(self, capacity: int = config.BUFFER_CAPACITY):
        self.capacity = capacity
        # Stored in CPU memory as uint8 (0-255) to conserve RAM and prevent OOM
        self.states = []           # Tensor: [12, 128, 128], uint8
        self.buttons = []          # Tensor: [21], float32
        self.camera_classes = []   # Tensor: scalar long [0, 120]
        self.position = 0
        self.lock = threading.Lock()

    def __len__(self):
        with self.lock:
            return len(self.states)

    def push(self, state: torch.Tensor, buttons: torch.Tensor, camera):
        """
        Stores state transition directly in RAM without writing to disk.
        state:   [12, 128, 128], float32 in [0, 1] or uint8 [0, 255]
        buttons: [21] or [9] float32 (automatically padded to config.NUM_BUTTONS)
        camera:  int / long scalar [0..120] OR continuous [2] float32 [yaw_dx, pitch_dy]
        """
        if state.dtype == torch.uint8:
            state_uint8 = state.cpu()
        else:
            state_uint8 = (state.cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        # Standardize button dimension to config.NUM_BUTTONS (21)
        btn_cpu = buttons.cpu().float()
        if btn_cpu.dim() == 1 and btn_cpu.shape[0] < config.NUM_BUTTONS:
            padded = torch.zeros(config.NUM_BUTTONS, dtype=torch.float32)
            padded[:btn_cpu.shape[0]] = btn_cpu
            btn_cpu = padded
        elif btn_cpu.dim() == 0:
            btn_cpu = btn_cpu.unsqueeze(0)

        # Standardize camera to categorical class [0..120]
        if isinstance(camera, (int, torch.LongTensor)) or (isinstance(camera, torch.Tensor) and camera.dtype in (torch.int64, torch.int32, torch.long) and camera.numel() == 1):
            cam_class = torch.as_tensor(camera, dtype=torch.long).cpu().view(())
        elif isinstance(camera, torch.Tensor) and camera.numel() == 2:
            cam_class = config.camera_to_bin(camera.cpu().float()).view(())
        elif isinstance(camera, (list, tuple)) and len(camera) == 2:
            cam_class = config.camera_to_bin(torch.tensor(camera, dtype=torch.float32)).view(())
        else:
            cam_class = torch.as_tensor(camera, dtype=torch.long).cpu().view(())

        with self.lock:
            if len(self.states) < self.capacity:
                self.states.append(state_uint8)
                self.buttons.append(btn_cpu)
                self.camera_classes.append(cam_class)
            else:
                self.states[self.position] = state_uint8
                self.buttons[self.position] = btn_cpu
                self.camera_classes[self.position] = cam_class

            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = config.BATCH_SIZE):
        """
        Random batch sampling across the entire memory buffer.
        Falls back to sampling with replacement if buffer has fewer items than batch_size.
        Returns:
          batch_states:  [B, 12, 128, 128], float32
          batch_buttons: [B, 21], float32
          batch_camera:  [B], long (categorical class in [0, 120])
        """
        with self.lock:
            n = len(self.states)
            if n == 0:
                raise ValueError("Cannot sample from an empty replay buffer.")

            if n >= batch_size:
                indices = random.sample(range(n), batch_size)
            else:
                indices = random.choices(range(n), k=batch_size)

            raw_stack = torch.stack([self.states[i] for i in indices])
            batch_states = (raw_stack.float() / 255.0).to(config.DEVICE)
            batch_buttons = torch.stack([self.buttons[i] for i in indices]).to(config.DEVICE)
            batch_camera = torch.stack([self.camera_classes[i] for i in indices]).to(config.DEVICE)

            return batch_states, batch_buttons, batch_camera

    def sample_sequences(self, batch_size: int = config.BATCH_SIZE, seq_len: int = config.DEFAULT_SEQ_LEN):
        """
        Samples contiguous sub-sequences of length `seq_len` for Recurrent BPTT.
        Guarantees that no sequence crosses the circular write head boundary
        (which separates the newest frame from the oldest frame).
        Always returns tensors with first dimension equal to `batch_size`.
        Returns:
          batch_states:  [B, seq_len, 12, 128, 128], float32
          batch_buttons: [B, seq_len, 21], float32
          batch_camera:  [B, seq_len], long
        """
        with self.lock:
            n = len(self.states)
            if n == 0:
                dummy_s = torch.zeros(batch_size, seq_len, 12, config.FRAME_HEIGHT, config.FRAME_WIDTH, device=config.DEVICE)
                dummy_b = torch.zeros(batch_size, seq_len, config.NUM_BUTTONS, device=config.DEVICE)
                dummy_c = torch.full((batch_size, seq_len), config.NEUTRAL_CAMERA_CLASS, dtype=torch.long, device=config.DEVICE)
                return dummy_s, dummy_b, dummy_c

            if n < seq_len:
                # Buffer has fewer items than requested sequence length: sample with replacement and tile
                indices = random.choices(range(n), k=batch_size)
                s_sample = torch.stack([self.states[i] for i in indices]).float() / 255.0
                b_sample = torch.stack([self.buttons[i] for i in indices])
                c_sample = torch.stack([self.camera_classes[i] for i in indices])
                return (
                    s_sample.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).to(config.DEVICE),
                    b_sample.unsqueeze(1).repeat(1, seq_len, 1).to(config.DEVICE),
                    c_sample.unsqueeze(1).repeat(1, seq_len).to(config.DEVICE)
                )

            valid_starts = []
            if n < self.capacity:
                # Buffer has not wrapped yet. Contiguous indices are strictly 0 .. n - seq_len
                max_start = n - seq_len
                for _ in range(batch_size):
                    valid_starts.append(random.randint(0, max_start))
            else:
                # Buffer is full circular buffer.
                # Chronological order runs from oldest (self.position) to newest ((self.position - 1) % capacity).
                # Any offset in [0, capacity - seq_len] is mathematically guaranteed not to cross the write head!
                max_offset = self.capacity - seq_len
                for _ in range(batch_size):
                    offset = random.randint(0, max_offset)
                    valid_starts.append((self.position + offset) % self.capacity)

            seq_states = []
            seq_buttons = []
            seq_cameras = []

            for start_idx in valid_starts:
                s_chunk = torch.stack([self.states[(start_idx + t) % n] for t in range(seq_len)])
                b_chunk = torch.stack([self.buttons[(start_idx + t) % n] for t in range(seq_len)])
                c_chunk = torch.stack([self.camera_classes[(start_idx + t) % n] for t in range(seq_len)])

                seq_states.append(s_chunk)
                seq_buttons.append(b_chunk)
                seq_cameras.append(c_chunk)

            batch_states = (torch.stack(seq_states).float() / 255.0).to(config.DEVICE)
            batch_buttons = torch.stack(seq_buttons).to(config.DEVICE)
            batch_camera = torch.stack(seq_cameras).to(config.DEVICE)

            return batch_states, batch_buttons, batch_camera
