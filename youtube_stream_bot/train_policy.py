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
import json
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
                 min_buffer_warmup: int = 256,
                 resume_checkpoint: bool = True):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_buffer_warmup = min_buffer_warmup
        self.policy_checkpoint_path = policy_checkpoint_path
        self.resume_checkpoint = resume_checkpoint
        self.checkpoint_meta_path = os.path.join(os.path.dirname(self.policy_checkpoint_path) or ".", "checkpoint_meta.json")

        # Thread synchronization & pause control
        self.queue_mgr = LinkQueueManager()
        self.replay_buffer = InactiveReplayBuffer(capacity=config.BUFFER_CAPACITY)
        self.stop_signal = threading.Event()
        self.is_paused = False
        self.is_producing = False
        self.producer_finished = False

        # Live telemetry tracking
        self.current_video_url = None
        self.current_video_sec = 0.0
        self.current_frame = 0
        self.current_step = 0
        self.current_video_title = "Minecraft Gameplay"
        self.current_video_duration = 0.0
        self.current_video_total_frames = 0
        self.last_btn_loss = 0.0
        self.last_cam_loss = 0.0
        self.last_total_loss = 0.0

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
        default_pos_weight = torch.tensor([1.5, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0], device=config.DEVICE)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=default_pos_weight)
        self.camera_loss_fn = nn.SmoothL1Loss()

        self.streamer = YouTubeFrameStreamer()

    def pause(self):
        self.is_paused = True
        print("[PIPELINE PAUSED] Learning suspended by user command.", flush=True)

    def resume(self):
        self.is_paused = False
        print("[PIPELINE RESUMED] Learning active.", flush=True)

    def save_checkpoint(self, is_manual: bool = False):
        try:
            checkpoint_data = {
                "video_source": self.current_video_url or "",
                "video_seconds": round(float(self.current_video_sec), 2),
                "frame": int(self.current_frame),
                "total_frames": int(self.current_video_total_frames),
                "video_duration": round(float(self.current_video_duration), 2),
                "video_title": self.current_video_title,
                "step": int(self.current_step),
                "btn_loss": round(float(self.last_btn_loss), 4),
                "cam_loss": round(float(self.last_cam_loss), 4),
                "total_loss": round(float(self.last_total_loss), 4),
                "is_manual": is_manual,
                "timestamp": time.time()
            }
            tmp_path = self.checkpoint_meta_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.checkpoint_meta_path)

            torch.save(self.policy.state_dict(), self.policy_checkpoint_path)

            tag = "MANUAL CHECKPOINT" if is_manual else "AUTO CHECKPOINT"
            print(f"[{tag} SAVED] Video: '{self.current_video_url}' at {self.current_video_sec:.1f}s | Frame #{self.current_frame} | Step #{self.current_step}", flush=True)
        except Exception as e:
            print(f"[Checkpoint Error] Failed to save checkpoint: {e}", flush=True)

    def stream_producer_loop(self):
        """
        Background worker thread that continually fetches YouTube/local videos, decodes frames in memory,
        pseudo-labels them with the IDM, and deposits transitions into the Replay Buffer.
        Supports pausing, manual/auto checkpoints, and resuming from exact video timestamps.
        """
        print("[Producer Thread] Started YouTube streaming producer...", flush=True)
        stacker = FrameStacker(stack_size=config.STACK_SIZE)
        total_finished_video_seconds = 0.0
        video_index = 0

        while not self.stop_signal.is_set():
            url = self.queue_mgr.get_next_link(timeout=2.0)
            if url is None:
                if video_index > 0 and self.queue_mgr.remaining() == 0:
                    self.producer_finished = True
                continue

            self.is_producing = True
            self.producer_finished = False
            video_index += 1
            total_videos_count = max(video_index, self.queue_mgr.total_added())

            print(f"\n[Producer Thread] ---> Streaming from: {url}", flush=True)
            self.current_video_url = url
            stacker.reset()
            prev_raw_frame_tensor = None  # [1, 3, 128, 128]

            # Inspect checkpoint to resume from exact video timestamp
            start_second = 0.0
            start_frame = 0
            if self.resume_checkpoint and os.path.exists(self.checkpoint_meta_path):
                try:
                    with open(self.checkpoint_meta_path, "r", encoding="utf-8") as f:
                        meta_cp = json.load(f)
                    cp_src = meta_cp.get("video_source", "")
                    if cp_src and (cp_src == url or os.path.normpath(cp_src) == os.path.normpath(url)):
                        saved_sec = float(meta_cp.get("video_seconds", 0.0))
                        if saved_sec > 1.0:
                            start_second = saved_sec
                            start_frame = int(meta_cp.get("frame", int(start_second * config.TARGET_FPS)))
                            print(f"[RESUMING VIDEO] Found saved checkpoint for '{url}'! Resuming from {start_second:.1f}s (Frame #{start_frame})", flush=True)
                except Exception as e:
                    print(f"[Producer Warning] Failed to inspect checkpoint: {e}", flush=True)

            frames_streamed = start_frame
            new_frames_decoded = 0
            video_start_time = time.time()
            current_video_title = "Minecraft Gameplay"
            current_video_duration = 0.0
            current_video_total_frames = 0

            try:
                for frame_np, meta in self.streamer.stream_frames(url, start_second=start_second):
                    while self.is_paused and not self.stop_signal.is_set():
                        time.sleep(0.1)
                    if self.stop_signal.is_set():
                        break

                    current_video_title = meta.get("title", current_video_title)
                    current_video_duration = meta.get("duration_sec", current_video_duration)
                    current_video_total_frames = meta.get("total_frames", current_video_total_frames)

                    self.current_video_title = current_video_title
                    self.current_video_duration = current_video_duration
                    self.current_video_total_frames = current_video_total_frames

                    # Shape: [128, 128, 3], uint8 -> [1, 3, 128, 128], float32
                    curr_frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE) / 255.0

                    # 1. Update temporal 4-frame stack
                    is_stack_full = stacker.add_frame(frame_np)

                    # 2. Pseudo-label if we have consecutive frames and full stack
                    if prev_raw_frame_tensor is not None and is_stack_full:
                        with torch.no_grad():
                            btn_logits, cam_delta = self.idm(prev_raw_frame_tensor, curr_frame_tensor)
                            pseudo_buttons = (torch.sigmoid(btn_logits).squeeze(0) > 0.5).float()
                            pseudo_camera = cam_delta.squeeze(0)

                        stacked_state = stacker.get_stacked_state()
                        self.replay_buffer.push(stacked_state, pseudo_buttons, pseudo_camera)

                        new_frames_decoded += 1
                        frames_streamed = start_frame + new_frames_decoded
                        self.current_frame = frames_streamed

                        elapsed_video_wall = max(0.001, time.time() - video_start_time)
                        current_video_sec = start_second + (new_frames_decoded / config.TARGET_FPS)
                        self.current_video_sec = current_video_sec
                        processing_fps = new_frames_decoded / elapsed_video_wall
                        speed_multiplier = processing_fps / config.TARGET_FPS

                        # Current video progress & ETA
                        if current_video_total_frames > 0:
                            video_pct = min(100.0, (frames_streamed / current_video_total_frames) * 100.0)
                            remaining_frames = max(0, current_video_total_frames - frames_streamed)
                            video_eta_sec = remaining_frames / max(1.0, processing_fps)
                        else:
                            video_pct = 0.0
                            video_eta_sec = 0.0

                        # Whole set progress & ETA
                        remaining_queue = self.queue_mgr.remaining()
                        estimated_avg_video_sec = current_video_duration if current_video_duration > 0 else 1800.0
                        total_set_estimated_sec = total_finished_video_seconds + current_video_duration + (remaining_queue * estimated_avg_video_sec)
                        current_set_finished_sec = total_finished_video_seconds + current_video_sec
                        set_pct = min(100.0, (current_set_finished_sec / max(1.0, total_set_estimated_sec)) * 100.0)
                        remaining_set_sec = max(0.0, total_set_estimated_sec - current_set_finished_sec)
                        set_eta_sec = remaining_set_sec / max(0.5, speed_multiplier)

                        def fmt_sec(s: float) -> str:
                            s = int(max(0, s))
                            h = s // 3600
                            m = (s % 3600) // 60
                            sec = s % 60
                            return f"{h:02d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

                        # Periodic telemetry emission every 15 frames (~1 sec in video time)
                        if new_frames_decoded % 15 == 0 or new_frames_decoded == 1:
                            active_keys = [config.BUTTON_NAMES[i].upper() for i, val in enumerate(pseudo_buttons) if val > 0.5]
                            keys_str = "+".join(active_keys) if active_keys else "IDLE"
                            dx = float(pseudo_camera[0].item())
                            dy = float(pseudo_camera[1].item())

                            cur_time_str = fmt_sec(current_video_sec)
                            tot_time_str = fmt_sec(current_video_duration)
                            v_eta_str = fmt_sec(video_eta_sec)
                            s_eta_str = fmt_sec(set_eta_sec)

                            print(f"[STREAMING PROGRESS] Frame #{frames_streamed:05d}/{current_video_total_frames:05d} ({video_pct:.1f}%) | "
                                  f"Time: {cur_time_str}/{tot_time_str} ({current_video_sec:.1f}s/{current_video_duration:.1f}s) | "
                                  f"Speed: {processing_fps:.1f}fps ({speed_multiplier:.1f}x) | "
                                  f"VideoETA: {v_eta_str} ({video_eta_sec:.0f}s) | "
                                  f"SetProgress: {set_pct:.1f}% | "
                                  f"SetETA: {s_eta_str} ({set_eta_sec:.0f}s) | "
                                  f"Queue: {video_index}/{total_videos_count} | "
                                  f"Keys: [{keys_str}] | "
                                  f"Mouse: (dx:{dx:+.2f}, dy:{dy:+.2f}) | "
                                  f"Buffer: {len(self.replay_buffer)}/50000 | "
                                  f"Title: {current_video_title}", flush=True)

                    prev_raw_frame_tensor = curr_frame_tensor

                total_finished_video_seconds += current_video_duration
                print(f"[Producer Thread] Finished video: {url} ({new_frames_decoded} transitions added to buffer, total buffer: {len(self.replay_buffer)})", flush=True)
            except Exception as e:
                print(f"[Producer Warning] Stream error on {url}: {e}. Skipping safely.", flush=True)
            finally:
                self.queue_mgr.task_done()
                self.is_producing = False

        self.producer_finished = True
        print("[Producer Thread] Producer exited.", flush=True)

    def train_policy_loop(self, total_steps: int = 0, log_interval: int = 50, save_interval: int = 500):
        """
        Main training loop. Uniformly samples uncorrelated random mini-batches from the Replay Buffer.
        If total_steps <= 0, continuously trains alongside video streaming until all videos finish.
        """
        is_continuous = (total_steps <= 0)
        mode_str = "continuous video-driven stream" if is_continuous else f"{total_steps} steps"
        print("[Trainer] Waiting for video stream & replay buffer warmup...", flush=True)
        idle_announced = False
        while len(self.replay_buffer) < self.min_buffer_warmup:
            while self.is_paused and not self.stop_signal.is_set():
                time.sleep(0.1)
            if self.stop_signal.is_set():
                return
            time.sleep(1.0)
            if self.queue_mgr.remaining() == 0 and len(self.replay_buffer) == 0:
                if not idle_announced:
                    print("[Trainer] Link queue empty. Waiting for user to enqueue YouTube links...", flush=True)
                    idle_announced = True
                continue
            idle_announced = False
            print(f"[Trainer] Buffer warmup: {len(self.replay_buffer)}/{self.min_buffer_warmup} transitions...", flush=True)

        print(f"[Trainer] Replay buffer warmed up ({len(self.replay_buffer)} transitions). Starting training for {mode_str}...", flush=True)
        self.policy.train()

        running_loss = 0.0
        running_btn_loss = 0.0
        running_cam_loss = 0.0
        step_count = 0
        start_time = time.time()
        step = 0
        cooldown_steps = 0

        while not self.stop_signal.is_set():
            if not is_continuous and step >= total_steps:
                break

            while self.is_paused and not self.stop_signal.is_set():
                time.sleep(0.1)
            if self.stop_signal.is_set():
                break

            if len(self.replay_buffer) < self.batch_size:
                if self.producer_finished and self.queue_mgr.remaining() == 0:
                    print("[Trainer] Video queue and buffer finished. Training complete.", flush=True)
                    break
                time.sleep(0.05)
                continue

            # In continuous mode, if producer is done and all videos processed, perform a 500-step cooldown convergence
            if is_continuous and self.producer_finished and self.queue_mgr.remaining() == 0:
                cooldown_steps += 1
                if cooldown_steps > 500:
                    print("[Trainer] Completed post-stream convergence passes. Training complete.", flush=True)
                    break

            step += 1
            self.current_step = step

            # Uniform random non-consecutive batch sampling
            batch_states, batch_buttons, batch_camera = self.replay_buffer.sample(self.batch_size)

            self.optimizer.zero_grad()

            pred_btn_logits, pred_camera, _ = self.policy(batch_states, h=None)

            loss_btn = self.bce_loss_fn(pred_btn_logits, batch_buttons)
            loss_cam = self.camera_loss_fn(pred_camera, batch_camera)

            total_loss = loss_btn + (2.0 * loss_cam)
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.last_btn_loss = loss_btn.item()
            self.last_cam_loss = loss_cam.item()
            self.last_total_loss = total_loss.item()

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

                step_hdr = f"Step [{step:05d}/CONT]" if is_continuous else f"Step [{step:05d}/{total_steps:05d}]"
                print(f"{step_hdr} | "
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
                self.save_checkpoint(is_manual=False)

        # Final save
        self.save_checkpoint(is_manual=False)
        print(f"[Trainer] Policy training complete. Model saved to '{self.policy_checkpoint_path}'", flush=True)


def cli_link_listener(trainer: StreamingTrainer):
    """Allows user to type or paste new YouTube links into the terminal or send control commands (PAUSE, RESUME, CHECKPOINT)."""
    print("\n[CLI Console] Commands: PAUSE, RESUME, CHECKPOINT, or paste video URLs/paths anytime.", flush=True)
    while not trainer.stop_signal.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().strip('"\'')
            if not cmd:
                continue
            upper = cmd.upper()
            if upper == "PAUSE":
                trainer.pause()
            elif upper == "RESUME":
                trainer.resume()
            elif upper == "CHECKPOINT":
                trainer.save_checkpoint(is_manual=True)
            elif not cmd.startswith("#"):
                trainer.queue_mgr.add_link(cmd)
        except Exception:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream YouTube gameplay & train Minecraft Imitation Policy.")
    parser.add_argument("--links", nargs="*", default=[], help="Initial list of YouTube URLs.")
    parser.add_argument("--links-file", type=str, default=None, help="Text file containing YouTube URLs (one per line).")
    parser.add_argument("--steps", type=int, default=0, help="Total training steps (0 for continuous streaming).")
    parser.add_argument("--warmup", type=int, default=128, help="Min transitions in replay buffer before training starts.")
    parser.add_argument("--idm-weights", type=str, default="idm_model.pth", help="Path to IDM weights file.")
    parser.add_argument("--output", type=str, default="minecraft_policy.pth", help="Checkpoint output file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume video from checkpoint timestamp.")
    args = parser.parse_args()

    trainer = StreamingTrainer(
        idm_weights_path=args.idm_weights,
        policy_checkpoint_path=args.output,
        min_buffer_warmup=args.warmup,
        resume_checkpoint=not args.no_resume
    )

    # Enqueue links from file if provided
    if args.links_file and os.path.exists(args.links_file):
        with open(args.links_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                clean_line = line.strip().strip('"\'')
                if clean_line and not clean_line.startswith("#"):
                    trainer.queue_mgr.add_link(clean_line)

    # Enqueue CLI provided links
    for lk in args.links:
        clean_lk = lk.strip().strip('"\'')
        if clean_lk and not clean_lk.startswith("#"):
            trainer.queue_mgr.add_link(clean_lk)

    if trainer.queue_mgr.remaining() == 0:
        print("[Info] Link queue empty. Waiting for user to enqueue YouTube links via UI or CLI...", flush=True)

    # Start background streaming producer thread
    producer_thread = threading.Thread(target=trainer.stream_producer_loop, daemon=True)
    producer_thread.start()

    # Start CLI link listener thread
    cli_thread = threading.Thread(target=cli_link_listener, args=(trainer,), daemon=True)
    cli_thread.start()

    try:
        trainer.train_policy_loop(total_steps=args.steps)
    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user. Shutting down safely...", flush=True)
    finally:
        trainer.stop_signal.set()
        print("[Notice] Pipeline stopped.", flush=True)
