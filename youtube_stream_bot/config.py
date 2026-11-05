"""
config.py - Shared Configuration, Action Space definitions, and Hyperparameters.
"""
import torch
import shutil

# 1. Hardware Device Configuration
# Uses NVIDIA CUDA GPU if available, otherwise falls back gracefully to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Video Stream Specifications
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_CHANNELS = 3
TARGET_FPS = 15
STACK_SIZE = 4          # Number of stacked frames: 4 * 3 = 12 input channels

# 3. Action Space Definition (Minecraft Controls)
# Discrete keyboard/mouse buttons (Multi-binary classification)
BUTTON_NAMES = [
    "forward",     # W
    "back",        # S
    "left",        # A
    "right",       # D
    "jump",        # Space
    "sneak",       # Shift
    "sprint",      # Ctrl
    "attack",      # Left Mouse Click
    "use"          # Right Mouse Click
]
NUM_BUTTONS = len(BUTTON_NAMES)  # 9 discrete binary heads

# Camera Delta (continuous mouse movements: yaw dx, pitch dy)
# Regression output: [pitch_delta, yaw_delta] normalized to [-1.0, 1.0]
NUM_CAMERA_AXES = 2

TOTAL_ACTION_DIM = NUM_BUTTONS + NUM_CAMERA_AXES  # 11 dimensions total

# 4. Training Hyperparameters
BUFFER_CAPACITY = 50_000   # 50,000 transitions in RAM
BATCH_SIZE = 64
POLICY_LR = 1e-4
IDM_LR = 3e-4

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
