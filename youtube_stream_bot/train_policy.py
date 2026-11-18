"""
train_policy.py - End-to-end streaming imitation learning pipeline for Minecraft.
Updated for OpenAI VPT Parity:
  - 121-class categorical camera Cross-Entropy loss (eliminating mode averaging drift).
  - 21-button discrete multi-binary loss with balanced focal weights (breaking back+left collapse).
  - Recurrent BPTT across T=16 timesteps with circular boundary guarantees.
  - Reset of mode-collapsed checkpoints on startup.
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
from models import InverseDynamicsModel, MinecraftPolicy, load_idm_weights


def is_policy_mode_collapsed(state_dict: dict) -> bool:
    """Checks if a saved checkpoint has mode-collapsed button weights (e.g., back+left saturation)."""
    if "button_head.bias" not in state_dict:
        return True
    bias = state_dict["button_head.bias"]
    if bias.shape[0] != config.NUM_BUTTONS:
        return True  # Obsolete shape (e.g., 9 buttons vs 21)
    if "camera_head.2.weight" in state_dict and state_dict["camera_head.2.weight"].shape[0] != config.NUM_CAMERA_CLASSES:
        return True  # Obsolete camera head (e.g., 2 regression vs 121 classes)
    # Check for extreme saturation
    if (bias.abs() > 15.0).any():
        return True
    return False


class StreamingTrainer:
    def __init__(self,
                 idm_weights_path: str = "idm_model.pth",
                 policy_checkpoint_path: str = "minecraft_policy.pth",
                 batch_size: int = config.BATCH_SIZE,
                 learning_rate: float = config.POLICY_LR,
                 min_buffer_warmup: int = 128,
                 seq_len: int = config.DEFAULT_SEQ_LEN,
                 resume_checkpoint: bool = True):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_buffer_warmup = min_buffer_warmup
        self.seq_len = seq_len
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.idm_weights_path = os.path.join(script_dir, idm_weights_path) if not os.path.isabs(idm_weights_path) else idm_weights_path
        self.policy_checkpoint_path = os.path.join(script_dir, policy_checkpoint_path) if not os.path.isabs(policy_checkpoint_path) else policy_checkpoint_path
        self.resume_checkpoint = resume_checkpoint
        self.checkpoint_meta_path = os.path.join(os.path.dirname(self.policy_checkpoint_path) or ".", "checkpoint_meta.json")

        # Thread synchronization & pause control
        self.queue_mgr = LinkQueueManager()
        self.replay_buffer = InactiveReplayBuffer(capacity=config.BUFFER_CAPACITY)
        self.stop_signal = threading.Event()
        self.model_lock = threading.Lock()
        self.train_h = None
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
        if os.path.exists(self.idm_weights_path):
            print(f"[Trainer] Loading IDM weights from: {self.idm_weights_path}")
            load_idm_weights(self.idm, self.idm_weights_path, device=config.DEVICE)
        else:
            print(f"[Warning] IDM weights '{self.idm_weights_path}' not found! Using calibrated initial IDM weights.")
        self.idm.eval()
        for param in self.idm.parameters():
            param.requires_grad = False

        # Initialize Behavioral Cloning Policy (CNN + GRU with 121-class camera & 21 buttons)
        policy_loaded = False
        if self.resume_checkpoint and os.path.exists(self.policy_checkpoint_path):
            try:
                sd = torch.load(self.policy_checkpoint_path, map_location=config.DEVICE, weights_only=True)
                if is_policy_mode_collapsed(sd):
                    print(f"[Trainer Warning] Checkpoint '{self.policy_checkpoint_path}' contains obsolete shapes or mode-collapsed weights! Resetting to balanced initialization.")
                else:
                    detected_gru = sd['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in sd else 256
                    self.policy = MinecraftPolicy(gru_hidden_dim=detected_gru).to(config.DEVICE)
                    self.policy.load_state_dict(sd)
                    print(f"[Trainer] Resumed policy: {self.policy.gru_hidden_dim} GRU neurons from {self.policy_checkpoint_path}")
                    policy_loaded = True
            except Exception as e:
                print(f"[Trainer Warning] Failed loading policy checkpoint ({e}), resetting to clean policy.")

        if not policy_loaded:
            self.policy = MinecraftPolicy().to(config.DEVICE)
            # Save fresh, calibrated policy checkpoint
            torch.save(self.policy.state_dict(), self.policy_checkpoint_path)
            print(f"[Trainer] Initialized fresh balanced policy (21 buttons, 121 camera classes). Saved to '{self.policy_checkpoint_path}'.")

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Balanced positive weights for 21 buttons to prevent mode collapse
        # [forward, back, left, right, jump, sneak, sprint, attack, use, hotbar_1..9, inventory, drop, offhand]
        pos_weights = [1.5, 2.0, 2.0, 2.0, 2.5, 3.0, 2.0, 2.5, 3.0] + [3.0] * 9 + [3.0, 4.0, 4.0]
        pos_weight_tensor = torch.tensor(pos_weights, device=config.DEVICE, dtype=torch.float32)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.camera_loss_fn = nn.CrossEntropyLoss()

        self.streamer = YouTubeFrameStreamer()

    def pause(self):
        self.is_paused = True
        self.save_checkpoint(is_manual=True)
        print(f"[PIPELINE PAUSED] Learning suspended & checkpoint saved at {self.current_video_sec:.1f}s (Frame #{self.current_frame}).", flush=True)

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
                "gru_neurons": int(self.policy.gru_hidden_dim),
                "total_parameters": int(sum(p.numel() for p in self.policy.parameters())),
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
        Background worker that continuously streams gameplay video, pseudo-labels transitions
        using the IDM, and pushes them into the Replay Buffer.
        """
        print("[Producer Thread] Started streaming producer...", flush=True)
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
            self.train_h = None
            stacker.reset()
            prev_raw_frame_tensor = None  # [1, 3, 128, 128]

            # Resume from exact video timestamp if meta exists
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

                    curr_frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE) / 255.0

                    is_stack_full = stacker.add_frame(frame_np)

                    if prev_raw_frame_tensor is not None and is_stack_full:
                        with torch.no_grad():
                            btn_logits, cam_logits = self.idm(prev_raw_frame_tensor, curr_frame_tensor)
                            pseudo_buttons = (torch.sigmoid(btn_logits).squeeze(0) > 0.5).float()
                            # Enforce physical control constraints (prevent conflicting movement keys)
                            b_l = btn_logits.squeeze(0)
                            if pseudo_buttons[0] == 1.0 and pseudo_buttons[1] == 1.0:
                                if b_l[0] >= b_l[1]: pseudo_buttons[1] = 0.0
                                else: pseudo_buttons[0] = 0.0
                            if pseudo_buttons[2] == 1.0 and pseudo_buttons[3] == 1.0:
                                if b_l[2] >= b_l[3]: pseudo_buttons[3] = 0.0
                                else: pseudo_buttons[2] = 0.0

                            pseudo_cam_class = cam_logits.argmax(dim=-1).squeeze(0)

                        stacked_state = stacker.get_stacked_state()
                        self.replay_buffer.push(stacked_state, pseudo_buttons, pseudo_cam_class)

                        new_frames_decoded += 1
                        frames_streamed = start_frame + new_frames_decoded
                        self.current_frame = frames_streamed

                        elapsed_video_wall = max(0.001, time.time() - video_start_time)
                        current_video_sec = start_second + (new_frames_decoded / config.TARGET_FPS)
                        self.current_video_sec = current_video_sec
                        processing_fps = new_frames_decoded / elapsed_video_wall
                        speed_multiplier = processing_fps / config.TARGET_FPS

                        if current_video_total_frames > 0:
                            video_pct = min(100.0, (frames_streamed / current_video_total_frames) * 100.0)
                            remaining_frames = max(0, current_video_total_frames - frames_streamed)
                            video_eta_sec = remaining_frames / max(1.0, processing_fps)
                        else:
                            video_pct = 0.0
                            video_eta_sec = 0.0

                        if new_frames_decoded % 15 == 0 or new_frames_decoded == 1:
                            active_keys = [config.BUTTON_NAMES[i].upper() for i, val in enumerate(pseudo_buttons) if val > 0.5]
                            keys_str = "+".join(active_keys) if active_keys else "IDLE"
                            cam_delta = config.bin_to_camera(pseudo_cam_class)
                            dx = float(cam_delta[0].item())
                            dy = float(cam_delta[1].item())

                            print(f"[STREAMING PROGRESS] Frame #{frames_streamed:05d}/{current_video_total_frames:05d} ({video_pct:.1f}%) | "
                                  f"Time: {current_video_sec:.1f}s/{current_video_duration:.1f}s | "
                                  f"Speed: {processing_fps:.1f}fps ({speed_multiplier:.1f}x) | "
                                  f"Keys: [{keys_str}] | "
                                  f"Camera: (bin:{pseudo_cam_class.item()}, dx:{dx:+.2f}, dy:{dy:+.2f}) | "
                                  f"Buffer: {len(self.replay_buffer)}/{config.BUFFER_CAPACITY}", flush=True)

                    prev_raw_frame_tensor = curr_frame_tensor

                total_finished_video_seconds += current_video_duration
                print(f"[Producer Thread] Finished video: {url} ({new_frames_decoded} transitions added to buffer)", flush=True)

                # Progressive Neurogenesis: Sprout +32 fresh recurrent memory units in GRU
                if new_frames_decoded > 50:
                    with self.model_lock:
                        old_gru = self.policy.gru_hidden_dim
                        new_gru = self.policy.grow_gru_neurons(add_neurons=32)
                        self._rebuild_optimizer_preserving_state()
                        self.train_h = None
                    print(f"\n✨ [NEUROGENESIS] Video episode learned! GRU memory expanded from {old_gru} -> {new_gru} neurons (+32 plastic recurrent units)! Total weights: {sum(p.numel() for p in self.policy.parameters()):,}\n", flush=True)
                    self.save_checkpoint(is_manual=True)
            except Exception as e:
                print(f"[Producer Warning] Stream error on {url}: {e}. Skipping safely.", flush=True)
            finally:
                self.queue_mgr.task_done()
                self.is_producing = False

        self.producer_finished = True
        print("[Producer Thread] Producer exited.", flush=True)

    def _rebuild_optimizer_preserving_state(self):
        """Reconstructs AdamW optimizer while preserving momentum states for untouched and expanded parameters."""
        param_names = {p: name for name, p in self.policy.named_parameters()}
        old_momenta = {}
        if hasattr(self, "optimizer") and hasattr(self.optimizer, "state"):
            for p, s in self.optimizer.state.items():
                if p in param_names:
                    old_momenta[param_names[p]] = {
                        k: (v.clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in s.items()
                    }

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Restore momentum tensors for matching and expanded layers
        for name, p in self.policy.named_parameters():
            if name in old_momenta:
                saved_s = old_momenta[name]
                new_s = {}
                for k, v in saved_s.items():
                    if k == "step":
                        new_s[k] = v.clone() if isinstance(v, torch.Tensor) else v
                    elif isinstance(v, torch.Tensor):
                        if v.shape == p.shape:
                            new_s[k] = v.to(p.device)
                        else:
                            # Pad momentum with zeros for newly sprouted neuron rows/cols
                            pad_v = torch.zeros_like(p)
                            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(pad_v.shape, v.shape))
                            pad_v[slices] = v[slices].to(p.device)
                            new_s[k] = pad_v
                    else:
                        new_s[k] = v
                self.optimizer.state[p] = new_s

    def train_policy_loop(self, total_steps: int = 0, log_interval: int = 50, save_interval: int = 0):
        """
        Main training loop. Uniformly samples contiguous sequences of length seq_len
        from the Replay Buffer for Recurrent BPTT.
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

        print(f"[Trainer] Replay buffer warmed up ({len(self.replay_buffer)} transitions). Starting training for {mode_str} (seq_len={self.seq_len})...", flush=True)
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

            if is_continuous and self.producer_finished and self.queue_mgr.remaining() == 0:
                cooldown_steps += 1
                if cooldown_steps > 500:
                    print("[Trainer] Completed post-stream convergence passes. Training complete.", flush=True)
                    break

            step += 1
            self.current_step = step

            # Contiguous sequence sampling for Recurrent BPTT (resolves short-horizon amnesia)
            batch_states, batch_buttons, batch_camera = self.replay_buffer.sample_sequences(
                self.batch_size, seq_len=self.seq_len
            )

            with self.model_lock:
                self.optimizer.zero_grad()

                # Sequence forward pass through Policy with recurrent state continuity
                curr_h = self.train_h
                if curr_h is not None and (curr_h.shape[1] != self.batch_size or curr_h.shape[2] != self.policy.gru_hidden_dim):
                    curr_h = None

                pred_btn_logits, pred_cam_logits, next_h = self.policy(batch_states, h=curr_h)

                if next_h is not None and next_h.shape[1] == self.batch_size and next_h.shape[2] == self.policy.gru_hidden_dim:
                    self.train_h = next_h.detach()
                else:
                    self.train_h = None

                # Multi-binary button loss
                loss_btn = self.bce_loss_fn(pred_btn_logits, batch_buttons)

                # 121-class categorical camera Cross-Entropy loss
                loss_cam = self.camera_loss_fn(
                    pred_cam_logits.view(-1, config.NUM_CAMERA_CLASSES),
                    batch_camera.view(-1)
                )

                total_loss = loss_btn + (1.5 * loss_cam)
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

            if save_interval > 0 and step % save_interval == 0:
                self.save_checkpoint(is_manual=False)

        # Final save
        self.save_checkpoint(is_manual=False)
        print(f"[Trainer] Policy training complete. Model saved to '{self.policy_checkpoint_path}'", flush=True)


def cli_link_listener(trainer: StreamingTrainer):
    """Allows user to control trainer from CLI."""
    print("\n[CLI Console] Commands: PAUSE, RESUME, CHECKPOINT, GROW, STOP, or paste video URLs/paths anytime.", flush=True)
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
            elif upper == "GROW":
                with trainer.model_lock:
                    old_gru = trainer.policy.gru_hidden_dim
                    new_gru = trainer.policy.grow_gru_neurons(add_neurons=32)
                    trainer._rebuild_optimizer_preserving_state()
                    trainer.train_h = None
                trainer.save_checkpoint(is_manual=True)
                print(f"\n✨ [MANUAL NEUROGENESIS] GRU neurons expanded: {old_gru} -> {new_gru} (+32 recurrent units)! Total weights: {sum(p.numel() for p in trainer.policy.parameters()):,}\n", flush=True)
            elif upper == "STOP":
                print("[CLI Console] Received STOP command. Saving checkpoint before shutdown...", flush=True)
                trainer.save_checkpoint(is_manual=True)
                trainer.stop_signal.set()
                break
            elif not cmd.startswith("#"):
                trainer.queue_mgr.add_link(cmd)
        except Exception:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream YouTube gameplay & train Minecraft Imitation Policy.")
    parser.add_argument("--links", nargs="*", default=[], help="Initial list of YouTube URLs.")
    parser.add_argument("--links-file", type=str, default=None, help="Text file containing YouTube URLs.")
    parser.add_argument("--steps", type=int, default=0, help="Total training steps (0 for continuous streaming).")
    parser.add_argument("--warmup", type=int, default=128, help="Min transitions in replay buffer before training starts.")
    parser.add_argument("--seq-len", type=int, default=config.DEFAULT_SEQ_LEN, help="Sequence length for Recurrent BPTT.")
    parser.add_argument("--idm-weights", type=str, default="idm_model.pth", help="Path to IDM weights file.")
    parser.add_argument("--output", type=str, default="minecraft_policy.pth", help="Checkpoint output file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume video from checkpoint timestamp.")
    args = parser.parse_args()

    trainer = StreamingTrainer(
        idm_weights_path=args.idm_weights,
        policy_checkpoint_path=args.output,
        min_buffer_warmup=args.warmup,
        seq_len=args.seq_len,
        resume_checkpoint=not args.no_resume
    )

    if args.links_file and os.path.exists(args.links_file):
        with open(args.links_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                clean_line = line.strip().strip('"\'')
                if clean_line and not clean_line.startswith("#"):
                    trainer.queue_mgr.add_link(clean_line)

    for lk in args.links:
        clean_lk = lk.strip().strip('"\'')
        if clean_lk and not clean_lk.startswith("#"):
            trainer.queue_mgr.add_link(clean_lk)

    if trainer.queue_mgr.remaining() == 0:
        print("[Info] Link queue empty. Waiting for user to enqueue YouTube links via UI or CLI...", flush=True)

    producer_thread = threading.Thread(target=trainer.stream_producer_loop, daemon=True)
    producer_thread.start()

    cli_thread = threading.Thread(target=cli_link_listener, args=(trainer,), daemon=True)
    cli_thread.start()

    try:
        trainer.train_policy_loop(total_steps=args.steps)
    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user. Shutting down safely...", flush=True)
    finally:
        trainer.stop_signal.set()
        print("[Notice] Pipeline stopped.", flush=True)
