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
    Recurrent Behavioral Cloning Policy with GRU temporal memory.
    Maps stacked frames to keyboard logits & mouse delta regression.
    """
    def __init__(self, in_channels: int = 12, gru_hidden_dim: int = 256):
        super().__init__()
        self.backbone = CNNBackbone(in_channels=in_channels)
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, batch_first=True)

        # Output Head 1: Multi-binary keyboard & mouse button classification
        self.button_head = nn.Linear(gru_hidden_dim, config.NUM_BUTTONS)

        # Output Head 2: Continuous camera pitch/yaw delta regression
        self.camera_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.NUM_CAMERA_AXES),
            nn.Tanh()  # Continuous camera displacement scaled to [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        """
        x: [B, 12, 128, 128], float32
        h: [1, B, 256], float32 (Optional GRU hidden memory state)
        Returns:
            button_logits: [B, 9] (Multi-binary key probabilities before sigmoid)
            camera_delta:  [B, 2] (Normalized pitch and yaw mouse deltas in [-1.0, 1.0])
            next_h:        [1, B, 256] (Updated recurrent memory state)
        """
        features = self.backbone(x)          # [B, 256]
        features_seq = features.unsqueeze(1) # [B, 1, 256] (sequence length = 1)

        gru_out, next_h = self.gru(features_seq, h) # gru_out: [B, 1, 256]
        gru_out = gru_out.squeeze(1)                # [B, 256]

        button_logits = self.button_head(gru_out)   # [B, 9]
        camera_delta = self.camera_head(gru_out)    # [B, 2]

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
