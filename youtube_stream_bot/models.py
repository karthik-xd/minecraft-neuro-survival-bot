"""
models.py - Neural Network Architectures with explicit tensor shape annotations.
1. CNNBackbone: 4-stage convolutional visual feature extractor.
2. MinecraftPolicy: CNN + GRU model mapping stacked frames to discrete keys and mouse deltas.
3. InverseDynamicsModel (IDM): Infers action a_t given current frame s_t and next frame s_{t+1}.
"""
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

            nn.AdaptiveAvgPool2d((1, 1)), # [B, 256, 1, 1]
            nn.Flatten()                  # [B, 256]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, in_channels, 128, 128]
        # Output:  [B, 256]
        return self.conv(x)


class MinecraftPolicy(nn.Module):
    """
    Recurrent Behavioral Cloning Policy with dynamic GRU temporal memory.
    Supports Net2Net Neurogenesis: dynamically sprouts fresh recurrent neurons
    after video training episodes while preserving 100% of previous weights.
    Maps stacked frames to keyboard logits & mouse delta regression.
    """
    def __init__(self, in_channels: int = 12, gru_hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.backbone = CNNBackbone(in_channels=in_channels)
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, batch_first=True)

        # Output Head 1: Continuous camera pitch/yaw delta regression (Ego-motion)
        self.camera_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.NUM_CAMERA_AXES),
            nn.Tanh()  # Continuous camera displacement scaled to [-1.0, 1.0]
        )

        # Output Head 2: Multi-binary button classification
        # Conditioned on visual features + camera ego-motion to disentangle view rotation from intentional actions
        self.button_head = nn.Linear(gru_hidden_dim + config.NUM_CAMERA_AXES, config.NUM_BUTTONS)

    def grow_gru_neurons(self, add_neurons: int = 32) -> int:
        """
        Progressive Neurogenesis (Net2Net):
        Sprouts `add_neurons` fresh recurrent memory units in GRU while preserving
        100% of previously learned weights and maintaining identical functional output.
        Returns the new total GRU neuron count.
        """
        old_h = self.gru_hidden_dim
        new_h = old_h + add_neurons
        device = next(self.parameters()).device

        # 1. Create expanded GRU
        new_gru = nn.GRU(input_size=256, hidden_size=new_h, batch_first=True).to(device)

        with torch.no_grad():
            for g in range(3):
                # Copy old input-hidden weights
                new_gru.weight_ih_l0[g*new_h : g*new_h + old_h, :] = self.gru.weight_ih_l0[g*old_h : (g+1)*old_h, :]
                nn.init.orthogonal_(new_gru.weight_ih_l0[g*new_h + old_h : (g+1)*new_h, :], gain=0.02)

                # Copy old recurrent hidden-hidden weights
                new_gru.weight_hh_l0[g*new_h : g*new_h + old_h, :old_h] = self.gru.weight_hh_l0[g*old_h : (g+1)*old_h, :]
                nn.init.orthogonal_(new_gru.weight_hh_l0[g*new_h + old_h : (g+1)*new_h, :], gain=0.02)
                new_gru.weight_hh_l0[g*new_h : g*new_h + old_h, old_h:] = 0.0

                # Biases
                new_gru.bias_ih_l0[g*new_h : g*new_h + old_h] = self.gru.bias_ih_l0[g*old_h : (g+1)*old_h]
                new_gru.bias_hh_l0[g*new_h : g*new_h + old_h] = self.gru.bias_hh_l0[g*old_h : (g+1)*old_h]
                new_gru.bias_ih_l0[g*new_h + old_h : (g+1)*new_h] = 0.0
                new_gru.bias_hh_l0[g*new_h + old_h : (g+1)*new_h] = 0.0

        # 2. Expand Button Head (new_h + 2 inputs -> config.NUM_BUTTONS)
        cam_dim = config.NUM_CAMERA_AXES
        new_button_head = nn.Linear(new_h + cam_dim, config.NUM_BUTTONS).to(device)
        with torch.no_grad():
            new_button_head.weight[:, :old_h] = self.button_head.weight[:, :old_h]
            new_button_head.weight[:, old_h : new_h] = 0.0
            new_button_head.weight[:, new_h : new_h + cam_dim] = self.button_head.weight[:, old_h : old_h + cam_dim]
            new_button_head.bias.copy_(self.button_head.bias)

        # 3. Expand Camera Head
        old_cam_linear = self.camera_head[0]
        new_cam_linear = nn.Linear(new_h, 64).to(device)
        with torch.no_grad():
            new_cam_linear.weight[:, :old_h] = old_cam_linear.weight
            new_cam_linear.weight[:, old_h:] = 0.0
            new_cam_linear.bias.copy_(old_cam_linear.bias)

        new_camera_head = nn.Sequential(
            new_cam_linear,
            nn.ReLU(inplace=True),
            self.camera_head[2],
            self.camera_head[3]
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
            button_logits: [B, 9] or [B, T, 9] (Multi-binary key probabilities before sigmoid)
            camera_delta:  [B, 2] or [B, T, 2] (Normalized pitch and yaw mouse deltas in [-1.0, 1.0])
            next_h:        [1, B, gru_hidden_dim] (Updated recurrent memory state across time)
        """
        if x.dim() == 5:
            # Multi-step Contiguous Sequence Training (BPTT across T timesteps)
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            features_flat = self.backbone(x_flat)       # [B*T, 256]
            features_seq = features_flat.view(B, T, 256) # [B, T, 256]

            gru_out, next_h = self.gru(features_seq, h) # [B, T, gru_hidden_dim]

            # Predict camera displacement (ego-motion)
            camera_delta = self.camera_head(gru_out)    # [B, T, 2]

            # Condition button actions on visual features + camera ego-motion
            btn_input = torch.cat([gru_out, camera_delta], dim=-1) # [B, T, gru_hidden_dim + 2]
            button_logits = self.button_head(btn_input)            # [B, T, 9]

            return button_logits, camera_delta, next_h
        else:
            # Single-step inference [B, 12, 128, 128]
            features = self.backbone(x)          # [B, 256]
            features_seq = features.unsqueeze(1) # [B, 1, 256] (sequence length = 1)

            gru_out, next_h = self.gru(features_seq, h) # gru_out: [B, 1, gru_hidden_dim]
            gru_out = gru_out.squeeze(1)                # [B, gru_hidden_dim]

            # Predict camera displacement (ego-motion)
            camera_delta = self.camera_head(gru_out)    # [B, 2]

            # Condition button actions on visual features + camera ego-motion
            btn_input = torch.cat([gru_out, camera_delta], dim=-1) # [B, gru_hidden_dim + 2]
            button_logits = self.button_head(btn_input)            # [B, 9]

            return button_logits, camera_delta, next_h


class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model (IDM).
    Takes a visual transition (s_t, s_{t+1}) and predicts the action a_t that caused it.
    Input Channels: 3 (current frame) + 3 (next frame) = 6 channels.
    """
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone(in_channels=6)

        # Multi-binary button prediction
        self.button_head = nn.Linear(256, config.NUM_BUTTONS)

        # Camera displacement prediction
        self.camera_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.NUM_CAMERA_AXES),
            nn.Tanh()
        )

    def forward(self, current_frame: torch.Tensor, next_frame: torch.Tensor):
        """
        current_frame: [B, 3, 128, 128], float32
        next_frame:    [B, 3, 128, 128], float32
        Returns:
            button_logits: [B, 9], float32
            camera_delta:  [B, 2], float32
        """
        # Concatenate along channel dimension: [B, 3+3, 128, 128] = [B, 6, 128, 128]
        x = torch.cat([current_frame, next_frame], dim=1)
        feat = self.backbone(x) # [B, 256]

        button_logits = self.button_head(feat) # [B, 9]
        camera_delta = self.camera_head(feat)   # [B, 2]
        return button_logits, camera_delta
