"""
record_human.py - Utility to collect labeled human gameplay for training the Inverse Dynamics Model (IDM).
Captures:
  - Current frame s_t: [3, 128, 128], float32
  - Next frame s_{t+1}: [3, 128, 128], float32
  - Action a_t:
      - buttons: [9], float32 (W, S, A, D, Jump, Sneak, Sprint, Attack, Use)
      - camera_delta: [2], float32 (mouse dx, dy normalized to [-1.0, 1.0])

Data is grouped by episode to strictly preserve episode-level validation splits without frame leakage.
Includes a '--synthetic' generator mode for instant end-to-end testing without external game setup.
"""
import time
import argparse
import numpy as np
import torch
import cv2
import config

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from pynput import keyboard, mouse
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False


class HumanGameplayRecorder:
    def __init__(self, target_fps: int = config.TARGET_FPS):
        self.fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.active_buttons = {name: False for name in config.BUTTON_NAMES}
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.recording = False
        self.key_listener = None
        self.mouse_listener = None

    def _on_press(self, key):
        try:
            char = key.char.lower()
            if char == 'w': self.active_buttons["forward"] = True
            elif char == 's': self.active_buttons["back"] = True
            elif char == 'a': self.active_buttons["left"] = True
            elif char == 'd': self.active_buttons["right"] = True
        except AttributeError:
            if key == keyboard.Key.space: self.active_buttons["jump"] = True
            elif key == keyboard.Key.shift: self.active_buttons["sneak"] = True
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r: self.active_buttons["sprint"] = True
            elif key == keyboard.Key.esc: self.recording = False

    def _on_release(self, key):
        try:
            char = key.char.lower()
            if char == 'w': self.active_buttons["forward"] = False
            elif char == 's': self.active_buttons["back"] = False
            elif char == 'a': self.active_buttons["left"] = False
            elif char == 'd': self.active_buttons["right"] = False
        except AttributeError:
            if key == keyboard.Key.space: self.active_buttons["jump"] = False
            elif key == keyboard.Key.shift: self.active_buttons["sneak"] = False
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r: self.active_buttons["sprint"] = False

    def _on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self.active_buttons["attack"] = pressed
        elif button == mouse.Button.right:
            self.active_buttons["use"] = pressed

    def _on_move(self, x, y):
        # Accumulate mouse deltas between sample ticks
        if not hasattr(self, '_last_mouse_pos'):
            self._last_mouse_pos = (x, y)
            return
        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)
        self.mouse_dx += dx
        self.mouse_dy += dy

    def start_listeners(self):
        if not HAS_PYNPUT:
            print("[Warning] pynput not installed. Keyboard/mouse inputs cannot be captured.")
            return
        self.key_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.mouse_listener = mouse.Listener(on_click=self._on_click, on_move=self._on_move)
        self.key_listener.start()
        self.mouse_listener.start()

    def stop_listeners(self):
        if self.key_listener: self.key_listener.stop()
        if self.mouse_listener: self.mouse_listener.stop()

    def capture_screen_frame(self, sct) -> np.ndarray:
        """Grabs full screen and resizes to [128, 128, 3] RGB uint8."""
        monitor = sct.monitors[1]  # Primary monitor
        sct_img = sct.grab(monitor)
        # Convert BGRA to RGB
        img_np = np.array(sct_img)[:, :, :3]
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        return resized

    def record_episode(self, episode_id: int, max_frames: int = 450) -> list:
        """
        Records a single gameplay episode (default 30 seconds at 15 FPS = 450 frames).
        Returns a list of transition dictionaries.
        """
        if not HAS_MSS:
            raise RuntimeError("mss is required for live screen recording. Install via 'pip install mss'.")

        transitions = []
        self.recording = True
        self.start_listeners()
        print(f"\n[Recorder] --- Starting Episode {episode_id} in 3 seconds... Switch to Minecraft window! ---")
        time.sleep(3.0)
        print(f"[Recorder] Recording Episode {episode_id} NOW! (Press ESC to stop early)")

        with mss.mss() as sct:
            prev_frame_np = self.capture_screen_frame(sct)
            prev_frame = torch.from_numpy(prev_frame_np).permute(2, 0, 1).float() / 255.0  # [3, 128, 128]

            step = 0
            while self.recording and step < max_frames:
                loop_start = time.time()

                # Snapshot actions
                buttons = torch.tensor([
                    1.0 if self.active_buttons[name] else 0.0 for name in config.BUTTON_NAMES
                ], dtype=torch.float32)  # [9]

                # Normalize mouse delta to [-1.0, 1.0]
                norm_dx = np.clip(self.mouse_dx / 50.0, -1.0, 1.0)
                norm_dy = np.clip(self.mouse_dy / 50.0, -1.0, 1.0)
                cam = torch.tensor([norm_dx, norm_dy], dtype=torch.float32)  # [2]

                # Reset delta counters
                self.mouse_dx = 0.0
                self.mouse_dy = 0.0

                # Capture next frame
                curr_frame_np = self.capture_screen_frame(sct)
                curr_frame = torch.from_numpy(curr_frame_np).permute(2, 0, 1).float() / 255.0  # [3, 128, 128]

                # Store transition (s_t, s_{t+1}, a_t)
                transitions.append({
                    "current_frame": prev_frame,      # [3, 128, 128]
                    "next_frame": curr_frame,          # [3, 128, 128]
                    "buttons": buttons,                # [9]
                    "camera": cam                      # [2]
                })

                prev_frame = curr_frame
                step += 1

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.frame_interval - elapsed)
                time.sleep(sleep_time)

        self.stop_listeners()
        print(f"[Recorder] Finished Episode {episode_id} with {len(transitions)} transitions.")
        return transitions


def generate_synthetic_human_data(num_episodes: int = 5, frames_per_ep: int = 300) -> dict:
    """
    Generates synthetic Minecraft-like transitions to test the IDM training pipeline
    without requiring manual recording. Creates visual motion correlated with actions.
    Output:
      dict with keys: 'episodes' -> list of episodes (each is a list of transition dicts).
    """
    print(f"[Synthetic] Generating {num_episodes} synthetic gameplay episodes ({frames_per_ep} frames each)...")
    episodes = []

    for ep in range(num_episodes):
        ep_transitions = []
        # Base frame: simulated green grass / brown dirt textures
        base_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.float32)
        base_frame[64:, :, 1] = 0.55 + 0.1 * np.random.rand(64, config.FRAME_WIDTH) # ground
        base_frame[:64, :, 0] = 0.4  # sky red
        base_frame[:64, :, 1] = 0.6  # sky green
        base_frame[:64, :, 2] = 0.9  # sky blue

        curr_frame_np = base_frame.copy()

        for _ in range(frames_per_ep):
            # Simulate keypresses
            w = 1.0 if np.random.rand() > 0.3 else 0.0
            s = 1.0 if np.random.rand() > 0.85 else 0.0
            a = 1.0 if np.random.rand() > 0.85 else 0.0
            d = 1.0 if np.random.rand() > 0.85 else 0.0
            jump = 1.0 if np.random.rand() > 0.8 else 0.0
            sneak = 1.0 if np.random.rand() > 0.9 else 0.0
            sprint = 1.0 if (w == 1.0 and np.random.rand() > 0.5) else 0.0
            attack = 1.0 if np.random.rand() > 0.8 else 0.0
            use = 1.0 if np.random.rand() > 0.9 else 0.0

            buttons = torch.tensor([w, s, a, d, jump, sneak, sprint, attack, use], dtype=torch.float32)

            # Simulate camera motion
            dx = float(np.clip(np.random.randn() * 0.3, -1.0, 1.0))
            dy = float(np.clip(np.random.randn() * 0.15, -1.0, 1.0))
            camera = torch.tensor([dx, dy], dtype=torch.float32)

            # Correlate next frame visual motion with camera shift and forward movement
            next_frame_np = np.roll(curr_frame_np, int(dx * 4), axis=1) # horizontal shift
            next_frame_np = np.roll(next_frame_np, int(dy * 2), axis=0) # vertical shift
            next_frame_np = np.clip(next_frame_np + np.random.normal(0, 0.02, next_frame_np.shape), 0.0, 1.0)

            # Convert to PyTorch tensors
            s_t = torch.from_numpy(curr_frame_np).permute(2, 0, 1).float()      # [3, 128, 128]
            s_next = torch.from_numpy(next_frame_np).permute(2, 0, 1).float()   # [3, 128, 128]

            ep_transitions.append({
                "current_frame": s_t,
                "next_frame": s_next,
                "buttons": buttons,
                "camera": camera
            })
            curr_frame_np = next_frame_np

        episodes.append(ep_transitions)

    return {"episodes": episodes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record labeled human play for Minecraft IDM.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--frames-per-ep", type=int, default=300, help="Frames per episode (300 = ~20s at 15 FPS).")
    parser.add_argument("--output", type=str, default="human_data.pt", help="Path to save output .pt dataset.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic gameplay without live recording.")
    args = parser.parse_args()

    if args.synthetic:
        data = generate_synthetic_human_data(num_episodes=args.episodes, frames_per_ep=args.frames_per_ep)
    else:
        recorder = HumanGameplayRecorder()
        episodes = []
        for i in range(1, args.episodes + 1):
            ep_data = recorder.record_episode(episode_id=i, max_frames=args.frames_per_ep)
            episodes.append(ep_data)
        data = {"episodes": episodes}

    torch.save(data, args.output)
    total_samples = sum(len(ep) for ep in data["episodes"])
    print(f"\n[Success] Saved {len(data['episodes'])} episodes ({total_samples} total transitions) to '{args.output}'.")
