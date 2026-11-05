# Minecraft YouTube In-Memory Streaming Bot (Imitation Learning + IDM)

An end-to-end PyTorch imitation learning pipeline that trains a Minecraft bot directly from YouTube gameplay streams with **zero disk writes**. The pipeline pipes direct media streams into memory, pseudo-labels raw video frames using an Inverse Dynamics Model (IDM), and trains a behavioral cloning policy equipped with a recurrent GRU temporal memory.

---

## 1. System Architecture

```
                          [ YouTube Video Queue ]
                                    │
                                    ▼ (yt-dlp -g: fresh HTTPS stream URL)
                         [ FFmpeg Subprocess ]
                                    │
                       (rawvideo rgb24 pipe:1 to RAM)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ In-Memory Frame Decoder [128x128 @ 15 FPS] (Zero Disk)  │
       └────────────────────────────┬────────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
      [ s_t: Current Frame ]              [ s_{t+1}: Next Frame ]
       [3, 128, 128] float32               [3, 128, 128] float32
                   │                                 │
                   └────────────────┬────────────────┘
                                    │ Concatenate [6, 128, 128]
                                    ▼
                ┌───────────────────────────────────────┐
                │     Inverse Dynamics Model (IDM)      │
                │         CNN Backbone (256-d)          │
                └───────────────────┬───────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
    [ Pseudo-Buttons \hat{a}_t ]            [ Pseudo-Camera \hat{a}_t ]
      [9] multi-binary logits               [2] normalized yaw/pitch dx/dy
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 4-Frame Temporal Stacker [12, 128, 128]│
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ Circular Replay Buffer (50,000 RAM)   │
                └───────────────────┬───────────────────┘
                                    │ Uniform Random Non-Consecutive Mini-Batches
                                    ▼
                ┌───────────────────────────────────────┐
                │       Minecraft Behavior Policy       │
                │        CNN Feature Extractor          │
                │         Recurrent GRU Memory          │
                │     Multi-Task Loss (BCE + SmoothL1)  │
                └───────────────────┬───────────────────┘
                                    │ Checkpointed to 'minecraft_policy.pth'
                                    ▼
                ┌───────────────────────────────────────┐
                │  Real-Time Live Agent (MSS 15 FPS)    │
                │ DirectInput Keyboard & Mouse Execution │
                └───────────────────────────────────────┘
```

---

## 2. Project Folder Structure

```
youtube_stream_bot/
├── config.py              # Central hyperparameters, action definitions, FFmpeg resolver
├── queue_manager.py       # Thread-safe dynamic YouTube link queue
├── video_streamer.py      # yt-dlp + FFmpeg in-memory stream decoder (0 disk bytes)
├── frame_stacker.py       # 4-frame temporal stacking (outputs [12, 128, 128])
├── replay_buffer.py       # 50k capacity in-memory circular replay buffer
├── models.py              # CNN Backbone, MinecraftPolicy (GRU), InverseDynamicsModel
├── record_human.py        # Screen + keyboard/mouse capture for IDM dataset
├── train_idm.py           # IDM training script with episode-level validation splitting
├── train_policy.py        # End-to-end multi-threaded streaming & policy training
├── live_agent.py          # Real-time game runner with DirectInput controls & killswitch
├── requirements.txt       # Dependencies
└── README.md              # Complete guide & documentation
```

---

## 3. Tensor Shapes Reference

| Module | Variable / Layer | Tensor Shape | Dtype & Range | Description |
|---|---|---|---|---|
| `video_streamer.py` | `frame` | `[128, 128, 3]` | `uint8` [0, 255] | Single raw frame read from FFmpeg stdout |
| `frame_stacker.py` | `tensor_frame` | `[3, 128, 128]` | `float32` [0.0, 1.0] | Normalized single RGB frame |
| `frame_stacker.py` | `stacked_state` | `[12, 128, 128]` | `float32` [0.0, 1.0] | 4 temporal stacked RGB frames ($4 \times 3 = 12$) |
| `models.py` (CNN) | `conv(x)` | `[B, 256]` | `float32` | 4-stage convolutional embedding vector |
| `models.py` (IDM) | `x` (concat) | `[B, 6, 128, 128]` | `float32` [0.0, 1.0] | Current frame $s_t$ + Next frame $s_{t+1}$ |
| `models.py` (IDM) | `button_logits` | `[B, 9]` | `float32` | Multi-binary logits for 9 keys |
| `models.py` (IDM) | `camera_delta` | `[B, 2]` | `float32` [-1.0, 1.0] | Normalized pitch $\Delta y$ and yaw $\Delta x$ |
| `models.py` (Policy) | `gru_out` | `[B, 256]` | `float32` | GRU temporal recurrent representation |
| `models.py` (Policy) | `h` / `next_h` | `[1, B, 256]` | `float32` | Recurrent GRU memory state across steps |
| `replay_buffer.py` | `batch_states` | `[B, 12, 128, 128]` | `float32` [0.0, 1.0] | Random non-consecutive mini-batch |
| `replay_buffer.py` | `batch_buttons` | `[B, 9]` | `float32` {0.0, 1.0} | Multi-binary target actions |
| `replay_buffer.py` | `batch_camera` | `[B, 2]` | `float32` [-1.0, 1.0] | Camera displacement target |

---

## 4. Setup & Installation

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```
*Note: FFmpeg is automatically discovered using `imageio-ffmpeg` or system PATH.*

---

## 5. Step-by-Step Execution Guide

### Step 1: Prepare IDM Labeled Data
The IDM requires a small set of labeled gameplay (frames + key/mouse inputs):

- **Option A (Synthetic Demo - Ready Immediately):**
  ```bash
  python record_human.py --synthetic --episodes 5 --frames-per-ep 300
  ```
- **Option B (Record Your Own Live Minecraft Play):**
  ```bash
  python record_human.py --episodes 3 --frames-per-ep 450
  ```
  *Switches to Minecraft, captures 15 FPS gameplay + keys and mouse movements, and saves to `human_data.pt`.*

### Step 2: Train the Inverse Dynamics Model (IDM)
```bash
python train_idm.py --epochs 15 --batch-size 64 --lr 3e-4
```
- Splits validation **strictly by episode** (never by random frames).
- Compensates for button class imbalance via positive weights in `BCEWithLogitsLoss`.
- Checkpoints the trained model to `idm_model.pth`.

### Step 3: Stream YouTube & Train the Policy
```bash
python train_policy.py --steps 3000 --warmup 256 --links "https://www.youtube.com/watch?v=VIDEO_ID"
```
- Fetches fresh ephemeral stream URLs on-the-fly (`yt-dlp`).
- Decodes $128 \times 128$ @ 15 FPS frames via FFmpeg stdout directly into RAM.
- Pseudo-labels frames using `idm_model.pth`.
- You can paste additional YouTube links dynamically into the terminal while it runs.
- Checkpoints policy weights to `minecraft_policy.pth`.

### Step 4: Run the Bot Live in Minecraft
```bash
python live_agent.py --model minecraft_policy.pth
```
- Unpause Minecraft.
- The bot captures the screen at 15 FPS, tracks temporal state using its GRU, and sends DirectInput keyboard and mouse inputs.
- **Emergency Stop:** Press `ESC` at any time to immediately kill the bot and release all keys.

---

## 6. Known Limitations & Next Steps

1. **Resolution vs. Small Items:** $128 \times 128$ resolution is optimal for high-throughput stream decoding and low-latency inference, but small inventory icons or tiny HUD elements are downscaled. Future step: Add a dual-resolution visual branch (e.g., $128 \times 128$ global view + $64 \times 64$ inventory crop).
2. **Streaming Bandwidth:** High network latency can throttle FFmpeg read throughput. The queue manager skips stalled streams safely.
3. **Complex Long-Horizon Planning:** Pure Behavioral Cloning learns immediate reactive reflexes (walking, jumping, looking around, attacking). For crafting trees and multi-step survival goals, combine with the neuro-symbolic behavior tree from `approch 3/server.js`.
