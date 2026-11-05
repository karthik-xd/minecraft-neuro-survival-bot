"""
train_policy.py - End-to-end streaming imitation learning pipeline for Minecraft.
Architecture:
  1. Producer Thread:
     - Pulls YouTube URLs from LinkQueueManager.
     - Fetches fresh ephemeral stream URL via yt-dlp.
     - Pipes raw 128x128 @ 15fps frames directly from FFmpeg stdout into RAM (Zero disk writes).
     - Passes adjacent frames (s_t, s_{t+1}) through pretrained Inverse Dynamics Model (IDM) to infer pseudo-actions.
     - Stacks 4 temporal frames [12, 128, 128].
     - Pushes transitions (state, buttons, camera_delta) into the 50k circular ReplayBuffer.
  2. Consumer / Trainer Thread:
     - Uniformly samples random non-consecutive batches [B, 12, 128, 128] from the replay buffer.
     - Trains the Behavioral Cloning Policy (CNN + GRU temporal memory).
     - Addresses class imbalance with weighted multi-binary BCE loss.
     - Periodically checkpoints weights to 'minecraft_policy.pth'.

Explicit Tensor Shapes:
  - Input Stacked States: [B, 12, 128, 128], float32 in [0.0, 1.0]
  - Pseudo-Labeled Buttons: [B, 9], float32 (binary 0.0 or 1.0)
  - Pseudo-Labeled Camera:  [B, 2], float32 (normalized pitch, yaw deltas [-1.0, 1.0])
  - Policy Output Buttons:  [B, 9], float32 logits
  - Policy Output Camera:   [B, 2], float32 displacements in [-1.0, 1.0]
  - GRU Hidden State:       [1, B, 256], float32
"""
import os
import sys
import time
import threading
import argparse
import torch
import torch.nn as nn
import numpy as np

import config
from queue_manager import LinkQueueManager
from video_streamer import YouTubeFrameStreamer
from frame_stacker import FrameStacker
from replay_buffer import InactiveReplayBuffer
from models import InverseDynamicsModel, MinecraftPolicy


class StreamingTrainer:
    def __init__(self,
                 idm_weights_path: str = "idm_model.pth",
                 policy_checkpoint_path: str = "minecraft_policy.pth",
                 batch_size: int = config.BATCH_SIZE,
                 learning_rate: float = config.POLICY_LR,
                 min_buffer_warmup: int = 256):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_buffer_warmup = min_buffer_warmup
        self.policy_checkpoint_path = policy_checkpoint_path

        # Thread synchronization & state
        self.queue_mgr = LinkQueueManager()
        self.replay_buffer = InactiveReplayBuffer(capacity=config.BUFFER_CAPACITY)
        self.stop_signal = threading.Event()

        # Load Inverse Dynamics Model (frozen pseudo-labeler)
        self.idm = InverseDynamicsModel().to(config.DEVICE)
        if os.path.exists(idm_weights_path):
            print(f"[Trainer] Loading IDM weights from: {idm_weights_path}")
            self.idm.load_state_dict(torch.load(idm_weights_path, map_location=config.DEVICE, weights_only=True))
        else:
            print(f"[Warning] IDM weights '{idm_weights_path}' not found! Using initialized IDM weights.")
        self.idm.eval()
        for param in self.idm.parameters():
            param.requires_grad = False

        # Initialize Behavioral Cloning Policy (CNN + GRU)
        self.policy = MinecraftPolicy().to(config.DEVICE)
        if os.path.exists(policy_checkpoint_path):
            print(f"[Trainer] Resuming policy from checkpoint: {policy_checkpoint_path}")
            self.policy.load_state_dict(torch.load(policy_checkpoint_path, map_location=config.DEVICE, weights_only=True))

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Loss functions: positive weights compensate for rare action buttons
        # e.g., attack, jump, sneak happen far less frequently than idling / running
        default_pos_weight = torch.tensor([1.5, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0], device=config.DEVICE)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=default_pos_weight)
        self.camera_loss_fn = nn.SmoothL1Loss()

        self.streamer = YouTubeFrameStreamer()

    def stream_producer_loop(self):
        """
        Background worker thread that continually fetches YouTube videos, decodes frames in memory,
        pseudo-labels them with the IDM, and deposits transitions into the Replay Buffer.
        """
        print("[Producer Thread] Started YouTube streaming producer...")
        stacker = FrameStacker(stack_size=config.STACK_SIZE)

        while not self.stop_signal.is_set():
            url = self.queue_mgr.get_next_link(timeout=2.0)
            if url is None:
                continue

            print(f"\n[Producer Thread] ---> Streaming from: {url}", flush=True)
            stacker.reset()
            prev_raw_frame_tensor = None  # [1, 3, 128, 128]
            frames_streamed = 0

            try:
                for frame_np in self.streamer.stream_frames(url):
                    if self.stop_signal.is_set():
                        break

                    # Shape: [128, 128, 3], uint8 -> [1, 3, 128, 128], float32
                    curr_frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE) / 255.0

                    # 1. Update temporal 4-frame stack
                    is_stack_full = stacker.add_frame(frame_np)

                    # 2. Pseudo-label if we have consecutive frames and full stack
                    if prev_raw_frame_tensor is not None and is_stack_full:
                        with torch.no_grad():
                            # IDM forward pass: (s_t, s_{t+1}) -> (button_logits, camera_delta)
                            # Input:  current_frame [1, 3, 128, 128], next_frame [1, 3, 128, 128]
                            # Output: button_logits [1, 9], camera_delta [1, 2]
                            btn_logits, cam_delta = self.idm(prev_raw_frame_tensor, curr_frame_tensor)

                            # Convert logits to binary labels: threshold at sigmoid > 0.5 (logit > 0)
                            pseudo_buttons = (torch.sigmoid(btn_logits).squeeze(0) > 0.5).float() # [9]
                            pseudo_camera = cam_delta.squeeze(0)                                   # [2]

                        # 3. Get 4-frame stacked state: [12, 128, 128], float32
                        stacked_state = stacker.get_stacked_state()

                        # 4. Store in RAM replay buffer
                        self.replay_buffer.push(stacked_state, pseudo_buttons, pseudo_camera)
                        frames_streamed += 1

                        if frames_streamed % 30 == 0:
                            active_keys = [config.BUTTON_NAMES[i].upper() for i, val in enumerate(pseudo_buttons) if val > 0.5]
                            keys_str = "+".join(active_keys) if active_keys else "IDLE"
                            dx = float(pseudo_camera[0].item())
                            dy = float(pseudo_camera[1].item())
                            print(f"[STREAMING LOG] Frame #{frames_streamed:05d} | IDM Inferred Keys: [{keys_str}] | Mouse: (dx:{dx:+.2f}, dy:{dy:+.2f}) | Buffer: {len(self.replay_buffer)}/50000", flush=True)

                    prev_raw_frame_tensor = curr_frame_tensor

                print(f"[Producer Thread] Finished video: {url} ({frames_streamed} transitions added to buffer, total buffer: {len(self.replay_buffer)})")
            except Exception as e:
                print(f"[Producer Warning] Stream error on {url}: {e}. Skipping safely.")
            finally:
                self.queue_mgr.task_done()

        print("[Producer Thread] Producer exited.")

    def train_policy_loop(self, total_steps: int = 5000, log_interval: int = 50, save_interval: int = 500):
        """
        Main training loop. Uniformly samples uncorrelated random mini-batches from the Replay Buffer.
        """
        print("[Trainer] Waiting for replay buffer warmup...")
        while len(self.replay_buffer) < self.min_buffer_warmup:
            if self.stop_signal.is_set():
                return
            time.sleep(1.0)
            print(f"[Trainer] Buffer warmup: {len(self.replay_buffer)}/{self.min_buffer_warmup} transitions...")

        print(f"[Trainer] Replay buffer warmed up ({len(self.replay_buffer)} transitions). Starting training for {total_steps} steps...")
        self.policy.train()

        running_loss = 0.0
        running_btn_loss = 0.0
        running_cam_loss = 0.0
        step_count = 0

        start_time = time.time()

        for step in range(1, total_steps + 1):
            if self.stop_signal.is_set():
                break

            # Uniform random non-consecutive batch sampling
            # batch_states:  [B, 12, 128, 128], float32
            # batch_buttons: [B, 9], float32
            # batch_camera:  [B, 2], float32
            batch_states, batch_buttons, batch_camera = self.replay_buffer.sample(self.batch_size)

            self.optimizer.zero_grad()

            # Forward pass through MinecraftPolicy (CNN + GRU)
            # Input:  x [B, 12, 128, 128], h=None (initial hidden state 0)
            # Output: pred_btn_logits [B, 9], pred_camera [B, 2], next_h [1, B, 256]
            pred_btn_logits, pred_camera, _ = self.policy(batch_states, h=None)

            # Losses
            loss_btn = self.bce_loss_fn(pred_btn_logits, batch_buttons)
            loss_cam = self.camera_loss_fn(pred_camera, batch_camera)

            total_loss = loss_btn + (2.0 * loss_cam)
            total_loss.backward()

            # Gradient clipping for recurrent stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += total_loss.item()
            running_btn_loss += loss_btn.item()
            running_cam_loss += loss_cam.item()
            step_count += 1

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step_count / max(1.0, elapsed)
                avg_total = running_loss / log_interval
                avg_btn = running_btn_loss / log_interval
                avg_cam = running_cam_loss / log_interval

                print(f"Step [{step:05d}/{total_steps:05d}] | "
                      f"Total Loss: {avg_total:.4f} | "
                      f"Btn Loss: {avg_btn:.4f} | "
                      f"Cam Loss: {avg_cam:.4f} | "
                      f"Buffer: {len(self.replay_buffer)} | "
                      f"Speed: {steps_per_sec:.1f} steps/s", flush=True)

                running_loss = 0.0
                running_btn_loss = 0.0
                running_cam_loss = 0.0
                step_count = 0
                start_time = time.time()

            if step % save_interval == 0:
                torch.save(self.policy.state_dict(), self.policy_checkpoint_path)
                print(f"  [Checkpoint] Saved policy weights to '{self.policy_checkpoint_path}'")

        # Final save
        torch.save(self.policy.state_dict(), self.policy_checkpoint_path)
        print(f"[Trainer] Policy training complete. Model saved to '{self.policy_checkpoint_path}'")


def cli_link_listener(queue_mgr: LinkQueueManager, stop_signal: threading.Event):
    """Allows user to type or paste new YouTube links into the terminal dynamically."""
    print("\n[CLI Console] You can paste YouTube URLs here anytime and press Enter to enqueue.")
    while not stop_signal.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            url = line.strip()
            if url:
                queue_mgr.add_link(url)
        except Exception:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream YouTube gameplay & train Minecraft Imitation Policy.")
    parser.add_argument("--links", nargs="*", default=[], help="Initial list of YouTube URLs.")
    parser.add_argument("--links-file", type=str, default=None, help="Text file containing YouTube URLs (one per line).")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps.")
    parser.add_argument("--warmup", type=int, default=128, help="Min transitions in replay buffer before training starts.")
    parser.add_argument("--idm-weights", type=str, default="idm_model.pth", help="Path to IDM weights file.")
    parser.add_argument("--output", type=str, default="minecraft_policy.pth", help="Checkpoint output file.")
    args = parser.parse_args()

    trainer = StreamingTrainer(
        idm_weights_path=args.idm_weights,
        policy_checkpoint_path=args.output,
        min_buffer_warmup=args.warmup
    )

    # Enqueue links from file if provided
    if args.links_file and os.path.exists(args.links_file):
        with open(args.links_file, "r") as f:
            for line in f:
                trainer.queue_mgr.add_link(line.strip())

    # Enqueue CLI provided links
    for lk in args.links:
        trainer.queue_mgr.add_link(lk)

    if trainer.queue_mgr.remaining() == 0:
        print("[Info] No initial links passed. Adding example high-quality Minecraft video to queue...")
        # A standard public creative commons / no-copyright Minecraft gameplay stream link
        trainer.queue_mgr.add_link("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # Start background streaming producer thread
    producer_thread = threading.Thread(target=trainer.stream_producer_loop, daemon=True)
    producer_thread.start()

    # Start CLI link listener thread
    cli_thread = threading.Thread(target=cli_link_listener, args=(trainer.queue_mgr, trainer.stop_signal), daemon=True)
    cli_thread.start()

    try:
        trainer.train_policy_loop(total_steps=args.steps)
    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user. Shutting down safely...")
    finally:
        trainer.stop_signal.set()
        print("[Notice] Pipeline stopped.")
