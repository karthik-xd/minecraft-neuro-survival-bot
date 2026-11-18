"""
config.py - Shared Configuration, Action Space definitions, and Hyperparameters.
Updated for OpenAI VPT Parity:
  - Discretized 11x11 camera action grid (121 categorical classes).
  - Expanded 21-button discrete action space (W/A/S/D, mouse, hotbar 1-9, inventory, drop, offhand).
"""
import shutil
import numpy as np
import torch

# 1. Hardware Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Video Stream Specifications
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_CHANNELS = 3
TARGET_FPS = 15
STACK_SIZE = 4          # Number of stacked frames: 4 * 3 = 12 input channels

# 3. Action Space Definition (Minecraft Controls)
# Expanded to 21 discrete keyboard/mouse buttons (Multi-binary classification)
BUTTON_NAMES = [
    "forward",     # 0: W
    "back",        # 1: S
    "left",        # 2: A
    "right",       # 3: D
    "jump",        # 4: Space
    "sneak",       # 5: Shift
    "sprint",      # 6: Ctrl
    "attack",      # 7: Left Mouse Click
    "use",         # 8: Right Mouse Click
    "hotbar_1",    # 9: 1
    "hotbar_2",    # 10: 2
    "hotbar_3",    # 11: 3
    "hotbar_4",    # 12: 4
    "hotbar_5",    # 13: 5
    "hotbar_6",    # 14: 6
    "hotbar_7",    # 15: 7
    "hotbar_8",    # 16: 8
    "hotbar_9",    # 17: 9
    "inventory",   # 18: E
    "drop",        # 19: Q
    "offhand"      # 20: F
]
NUM_BUTTONS = len(BUTTON_NAMES)  # 21 discrete binary heads

# Camera Delta Discretization (OpenAI VPT Section 3.2: 11x11 grid = 121 categorical classes)
# Foveated bin centers: fine-grained near zero for aim precision, coarse-grained for rapid flicks
CAMERA_BINS_PER_AXIS = np.array(
    [-1.0, -0.4, -0.2, -0.1, -0.02, 0.0, 0.02, 0.1, 0.2, 0.4, 1.0],
    dtype=np.float32
)
NUM_CAMERA_BINS_PER_AXIS = len(CAMERA_BINS_PER_AXIS)  # 11 bins
NUM_CAMERA_CLASSES = NUM_CAMERA_BINS_PER_AXIS * NUM_CAMERA_BINS_PER_AXIS  # 121 classes
NEUTRAL_CAMERA_CLASS = 5 * NUM_CAMERA_BINS_PER_AXIS + 5  # Index 60 is (yaw=0.0, pitch=0.0)

NUM_CAMERA_AXES = 2  # Continuous yaw (dx) and pitch (dy) for legacy reference
TOTAL_ACTION_DIM = NUM_BUTTONS + NUM_CAMERA_CLASSES


def camera_to_bin(cam_input) -> torch.Tensor:
    """
    Converts continuous camera delta [..., 2] (yaw dx, pitch dy in [-1.0, 1.0])
    into discrete class index in [0, 120] (pitch_bin * 11 + yaw_bin).
    Accepts PyTorch Tensor or numpy ndarray.
    """
    if isinstance(cam_input, np.ndarray):
        cam_tensor = torch.from_numpy(cam_input).float()
    elif isinstance(cam_input, (list, tuple)):
        cam_tensor = torch.tensor(cam_input, dtype=torch.float32)
    else:
        cam_tensor = cam_input.float()

    bins = torch.as_tensor(CAMERA_BINS_PER_AXIS, device=cam_tensor.device, dtype=torch.float32)
    yaw = cam_tensor[..., 0].unsqueeze(-1)
    pitch = cam_tensor[..., 1].unsqueeze(-1)

    yaw_bin = torch.argmin(torch.abs(yaw - bins), dim=-1)
    pitch_bin = torch.argmin(torch.abs(pitch - bins), dim=-1)

    return pitch_bin * NUM_CAMERA_BINS_PER_AXIS + yaw_bin


def bin_to_camera(bin_input) -> torch.Tensor:
    """
    Converts discrete class index in [0, 120] back to continuous [..., 2] (yaw dx, pitch dy).
    Index layout: class_idx = pitch_bin * 11 + yaw_bin.
    Accepts integer, list, numpy array, or PyTorch Tensor.
    """
    if isinstance(bin_input, (int, np.integer)):
        bin_tensor = torch.tensor(bin_input, dtype=torch.long)
    elif isinstance(bin_input, np.ndarray):
        bin_tensor = torch.from_numpy(bin_input).long()
    elif isinstance(bin_input, (list, tuple)):
        bin_tensor = torch.tensor(bin_input, dtype=torch.long)
    else:
        bin_tensor = bin_input.long()

    bins = torch.as_tensor(CAMERA_BINS_PER_AXIS, device=bin_tensor.device, dtype=torch.float32)
    pitch_bin = torch.div(bin_tensor, NUM_CAMERA_BINS_PER_AXIS, rounding_mode="floor").clamp(0, NUM_CAMERA_BINS_PER_AXIS - 1)
    yaw_bin = (bin_tensor % NUM_CAMERA_BINS_PER_AXIS).clamp(0, NUM_CAMERA_BINS_PER_AXIS - 1)

    yaw = bins[yaw_bin]
    pitch = bins[pitch_bin]
    return torch.stack([yaw, pitch], dim=-1)


# 4. Training Hyperparameters
BUFFER_CAPACITY = 15_000   # 15,000 transitions in RAM (bounded uint8 memory to prevent OOM)
BATCH_SIZE = 64
POLICY_LR = 1e-4
IDM_LR = 3e-4
DEFAULT_SEQ_LEN = 16       # 16 timesteps (~1.07s) sequence length for Recurrent BPTT


def get_ffmpeg_binary() -> str:
    """Finds system ffmpeg or falls back to imageio-ffmpeg binary automatically."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"
