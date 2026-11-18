"""
models.py - Neural Network Architectures with explicit tensor shape annotations.
Updated for OpenAI VPT Parity:
1. CNNBackbone: 4-stage convolutional visual feature extractor.
2. MinecraftPolicy: CNN + GRU model with 121-class Cross-Entropy camera head and 21-button discrete actions.
3. InverseDynamicsModel (IDM): Infers action a_t (21 buttons + 121 camera classes) given (s_t, s_{t+1}).
"""
import os
import torch
import torch.nn as nn
import config


class CNNBackbone(nn.Module):
    """
    Lightweight 4-stage CNN backbone for 128x128 resolution.
    Input Tensor Shape:  [B, In_Channels, 128, 128]
    Output Tensor Shape: [B, 256]
    """
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.conv = nn.Sequential(
            # Stage 1: [B, in_channels, 128, 128] -> [B, 32, 64, 64]
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Stage 2: [B, 32, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Stage 3: [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Stage 4: [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
            nn.Flatten()                   # [B, 256]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MinecraftPolicy(nn.Module):
    """
    Recurrent Behavioral Cloning Policy with dynamic GRU temporal memory.
    Supports Net2Net Progressive Neurogenesis.
    Outputs:
      - button_logits: 21 discrete binary heads [B, 21] or [B, T, 21]
      - camera_logits: 121 categorical classes [B, 121] or [B, T, 121] (eliminates mode averaging drift)
    """
    def __init__(self, in_channels: int = 12, gru_hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.backbone = CNNBackbone(in_channels=in_channels)
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, batch_first=True)

        # Output Head 1: 121-class categorical camera classification (11x11 pitch/yaw grid)
        self.camera_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.NUM_CAMERA_CLASSES)
        )

        # Output Head 2: 21-binary button classification
        self.button_head = nn.Linear(gru_hidden_dim, config.NUM_BUTTONS)

        self._init_balanced_weights()

    def _init_balanced_weights(self):
        """Initializes weights to prevent mode collapse into back+left and static camera drift."""
        with torch.no_grad():
            # Buttons: balanced slight negative bias (~ -2.0 gives ~12% sigmoid firing rate)
            nn.init.orthogonal_(self.button_head.weight, gain=0.01)
            self.button_head.bias.fill_(-2.0)

            # Camera: initialize with slight prior on neutral look (class 60: yaw=0, pitch=0)
            nn.init.orthogonal_(self.camera_head[2].weight, gain=0.01)
            self.camera_head[2].bias.zero_()
            self.camera_head[2].bias[config.NEUTRAL_CAMERA_CLASS] = 1.0

    def grow_gru_neurons(self, add_neurons: int = 32) -> int:
        """
        Progressive Neurogenesis (Net2Net):
        Sprouts `add_neurons` fresh recurrent memory units in GRU while preserving
        100% of previously learned weights and maintaining identical functional output.
        """
        old_h = self.gru_hidden_dim
        new_h = old_h + add_neurons
        device = next(self.parameters()).device

        # 1. Create expanded GRU
        new_gru = nn.GRU(input_size=256, hidden_size=new_h, batch_first=True).to(device)

        with torch.no_grad():
            for g in range(3):
                # Input-hidden weights
                new_gru.weight_ih_l0[g*new_h : g*new_h + old_h, :] = self.gru.weight_ih_l0[g*old_h : (g+1)*old_h, :]
                nn.init.orthogonal_(new_gru.weight_ih_l0[g*new_h + old_h : (g+1)*new_h, :], gain=0.02)

                # Recurrent hidden-hidden weights
                new_gru.weight_hh_l0[g*new_h : g*new_h + old_h, :old_h] = self.gru.weight_hh_l0[g*old_h : (g+1)*old_h, :]
                nn.init.orthogonal_(new_gru.weight_hh_l0[g*new_h + old_h : (g+1)*new_h, :], gain=0.02)
                new_gru.weight_hh_l0[g*new_h : g*new_h + old_h, old_h:] = 0.0

                # Biases
                new_gru.bias_ih_l0[g*new_h : g*new_h + old_h] = self.gru.bias_ih_l0[g*old_h : (g+1)*old_h]
                new_gru.bias_hh_l0[g*new_h : g*new_h + old_h] = self.gru.bias_hh_l0[g*old_h : (g+1)*old_h]
                new_gru.bias_ih_l0[g*new_h + old_h : (g+1)*new_h] = 0.0
                new_gru.bias_hh_l0[g*new_h + old_h : (g+1)*new_h] = 0.0

        # 2. Expand Button Head
        new_button_head = nn.Linear(new_h, config.NUM_BUTTONS).to(device)
        with torch.no_grad():
            new_button_head.weight[:, :old_h] = self.button_head.weight
            new_button_head.weight[:, old_h:] = 0.0
            new_button_head.bias.copy_(self.button_head.bias)

        # 3. Expand Camera Head first layer
        old_cam_linear = self.camera_head[0]
        new_cam_linear = nn.Linear(new_h, 128).to(device)
        with torch.no_grad():
            new_cam_linear.weight[:, :old_h] = old_cam_linear.weight
            new_cam_linear.weight[:, old_h:] = 0.0
            new_cam_linear.bias.copy_(old_cam_linear.bias)

        new_camera_head = nn.Sequential(
            new_cam_linear,
            nn.ReLU(inplace=True),
            self.camera_head[2]
        ).to(device)

        # 4. Swap modules
        self.gru = new_gru
        self.button_head = new_button_head
        self.camera_head = new_camera_head
        self.gru_hidden_dim = new_h

        return new_h

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        """
        x: [B, 12, 128, 128] OR sequence [B, T, 12, 128, 128], float32
        h: [1, B, gru_hidden_dim], float32 (Optional GRU recurrent memory state)
        Returns:
            button_logits: [B, 21] or [B, T, 21] (Multi-binary key probabilities before sigmoid)
            camera_logits: [B, 121] or [B, T, 121] (Categorical camera classification logits)
            next_h:        [1, B, gru_hidden_dim] (Updated recurrent memory state across time)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.dim() == 5:
            # Multi-step Contiguous Sequence Training (BPTT across T timesteps)
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            features_flat = self.backbone(x_flat)        # [B*T, 256]
            features_seq = features_flat.view(B, T, 256) # [B, T, 256]

            gru_out, next_h = self.gru(features_seq, h)  # [B, T, gru_hidden_dim]

            camera_logits = self.camera_head(gru_out)    # [B, T, 121]
            button_logits = self.button_head(gru_out)    # [B, T, 21]

            return button_logits, camera_logits, next_h
        else:
            # Single-step inference [B, 12, 128, 128]
            features = self.backbone(x)           # [B, 256]
            features_seq = features.unsqueeze(1)  # [B, 1, 256]

            gru_out, next_h = self.gru(features_seq, h)  # gru_out: [B, 1, gru_hidden_dim]
            gru_out = gru_out.squeeze(1)                 # [B, gru_hidden_dim]

            camera_logits = self.camera_head(gru_out)    # [B, 121]
            button_logits = self.button_head(gru_out)    # [B, 21]

            return button_logits, camera_logits, next_h


class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model (IDM).
    Takes a visual transition (s_t, s_{t+1}) and predicts the action a_t that caused it.
    Input Channels: 3 (current frame) + 3 (next frame) = 6 channels.
    Outputs:
      - button_logits: 21 discrete binary heads
      - camera_logits: 121 categorical classes (11x11 pitch/yaw grid)
    """
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone(in_channels=6)

        # Multi-binary button prediction (21 buttons)
        self.button_head = nn.Linear(256, config.NUM_BUTTONS)

        # Categorical camera prediction (121 classes)
        self.camera_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.NUM_CAMERA_CLASSES)
        )

        self._init_balanced_weights()

    def _init_balanced_weights(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.button_head.weight, gain=0.01)
            self.button_head.bias.fill_(-2.0)

            nn.init.orthogonal_(self.camera_head[2].weight, gain=0.01)
            self.camera_head[2].bias.zero_()
            self.camera_head[2].bias[config.NEUTRAL_CAMERA_CLASS] = 1.0

    def forward(self, current_frame: torch.Tensor, next_frame: torch.Tensor):
        """
        current_frame: [B, 3, 128, 128] or [3, 128, 128], float32
        next_frame:    [B, 3, 128, 128] or [3, 128, 128], float32
        Returns:
            button_logits: [B, 21], float32
            camera_logits: [B, 121], float32
        """
        if current_frame.dim() == 3:
            current_frame = current_frame.unsqueeze(0)
        if next_frame.dim() == 3:
            next_frame = next_frame.unsqueeze(0)

        x = torch.cat([current_frame, next_frame], dim=1)  # [B, 6, 128, 128]
        feat = self.backbone(x)                            # [B, 256]

        button_logits = self.button_head(feat)             # [B, 21]
        camera_logits = self.camera_head(feat)             # [B, 121]
        return button_logits, camera_logits


def load_idm_weights(model: InverseDynamicsModel, weights_path: str, device=config.DEVICE) -> bool:
    """
    Robust loader for IDM weights.
    Supports:
      1. Standard calibrated state_dict (idm_model.pth).
      2. Official OpenAI VPT IDM weights (2x_idm.model / 2x_idm.weights) or checkpoints.
      3. Automatic prefix stripping ('model.', 'net.', 'module.', 'idm.', 'policy.').
      4. Graceful partial loading if architectures differ.
    """
    if not os.path.exists(weights_path):
        print(f"[IDM Loader] File '{weights_path}' does not exist.")
        return False

    try:
        loaded = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
        elif isinstance(loaded, dict) and "model" in loaded:
            state_dict = loaded["model"]
        elif isinstance(loaded, dict) and "policy" in loaded:
            state_dict = loaded["policy"]
        elif isinstance(loaded, dict) and "net" in loaded:
            state_dict = loaded["net"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        elif hasattr(loaded, "state_dict"):
            state_dict = loaded.state_dict()
        else:
            print(f"[IDM Loader] Unrecognized weight format in '{weights_path}'.")
            return False

        # Attempt exact match load directly
        try:
            model.load_state_dict(state_dict)
            print(f"[IDM Loader] Successfully loaded exact state dict from '{weights_path}'.")
            return True
        except Exception:
            pass

        # Try prefix-stripped exact load
        prefixes = ["model.", "net.", "module.", "idm.", "policy."]
        cleaned_sd = {}
        for k, v in state_dict.items():
            ck = k
            for pfx in prefixes:
                if ck.startswith(pfx):
                    ck = ck[len(pfx):]
            cleaned_sd[ck] = v

        try:
            model.load_state_dict(cleaned_sd)
            print(f"[IDM Loader] Successfully loaded prefix-stripped state dict from '{weights_path}'.")
            return True
        except Exception:
            pass

        # Partial transfer of matching backbone and projection layers
        model_dict = model.state_dict()
        matched = {}
        for source_sd in (cleaned_sd, state_dict):
            for k, v in source_sd.items():
                if k in model_dict and v.shape == model_dict[k].shape and k not in matched:
                    matched[k] = v

        if matched:
            model_dict.update(matched)
            model.load_state_dict(model_dict)
            print(f"[IDM Loader] Partially transferred {len(matched)}/{len(model_dict)} matching parameter tensors from '{weights_path}'.")
            return True

        print(f"[IDM Loader] No matching parameter shapes found in '{weights_path}'.")
        return False
    except Exception as e:
        print(f"[IDM Loader] Error reading '{weights_path}': {e}")
        return False
