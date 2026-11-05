"""
live_agent.py - Real-time inference agent that controls Minecraft directly.
Pipeline:
  1. Grabs live screen using mss at 15 FPS.
  2. Resizes and stacks the last 4 frames into tensor [1, 12, 128, 128], float32.
  3. Feeds state + recurrent hidden state h_t into MinecraftPolicy.
  4. Decodes multi-binary button actions and continuous camera deltas.
  5. Sends hardware-level DirectInput keyboard keypresses and mouse movements.
  6. Emergency safety kill switch: press 'ESC' anytime to release keys and exit.

Explicit Tensor Shapes:
  - Input Stacked State: [1, 12, 128, 128], float32
  - Recurrent GRU State: [1, 1, 256], float32
  - Pred Button Logits:  [1, 9], float32
  - Pred Camera Deltas:  [1, 2], float32 (pitch, yaw normalized to [-1.0, 1.0])
"""
import time
import argparse
import numpy as np
import torch
import cv2
import mss

import config
from frame_stacker import FrameStacker
from models import MinecraftPolicy

# DirectInput for Windows Minecraft
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.001
    HAS_DIRECTINPUT = True
except ImportError:
    HAS_DIRECTINPUT = False

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False


# Map internal button names to DirectInput key codes
KEY_MAPPING = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
    "jump": "space",
    "sneak": "shift",
    "sprint": "ctrl"
}


class LiveMinecraftBot:
    def __init__(self, model_path: str = "minecraft_policy.pth"):
        self.device = config.DEVICE
        self.policy = MinecraftPolicy().to(self.device)

        print(f"[Live Bot] Loading policy checkpoint from: {model_path}")
        self.policy.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.policy.eval()

        self.stacker = FrameStacker(stack_size=config.STACK_SIZE)
        self.running = True
        self.active_keys = set()
        self.active_mouse_buttons = set()

        # Recurrent GRU hidden memory state: [1, 1, 256], float32
        self.gru_hidden = torch.zeros(1, 1, 256, device=self.device)

        # Safety kill switch listener
        if HAS_PYNPUT:
            self.esc_listener = keyboard.Listener(on_press=self._on_key_press)
            self.esc_listener.start()

    def _on_key_press(self, key):
        if key == keyboard.Key.esc:
            print("\n[Safety Killswitch Activated] Emergency Stop Triggered by User!")
            self.running = False

    def release_all_inputs(self):
        """Emergency cleanup to ensure keys don't get stuck."""
        if not HAS_DIRECTINPUT:
            return
        for key in list(self.active_keys):
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        self.active_keys.clear()

        for btn in list(self.active_mouse_buttons):
            try:
                pydirectinput.mouseUp(button=btn)
            except Exception:
                pass
        self.active_mouse_buttons.clear()

    def execute_action(self, buttons_binary: np.ndarray, camera_delta: np.ndarray):
        """
        Executes keyboard and mouse commands in the game.
        buttons_binary: ndarray [9] (0 or 1 for each button)
        camera_delta:   ndarray [2] (pitch, yaw deltas normalized [-1.0, 1.0])
        """
        if not HAS_DIRECTINPUT:
            return

        # 1. Handle Keyboard actions
        for i, name in enumerate(config.BUTTON_NAMES):
            is_active = bool(buttons_binary[i] > 0.5)

            if name in KEY_MAPPING:
                key_code = KEY_MAPPING[name]
                if is_active and key_code not in self.active_keys:
                    pydirectinput.keyDown(key_code)
                    self.active_keys.add(key_code)
                elif not is_active and key_code in self.active_keys:
                    pydirectinput.keyUp(key_code)
                    self.active_keys.remove(key_code)

            elif name == "attack": # Left click
                if is_active and "left" not in self.active_mouse_buttons:
                    pydirectinput.mouseDown(button="left")
                    self.active_mouse_buttons.add("left")
                elif not is_active and "left" in self.active_mouse_buttons:
                    pydirectinput.mouseUp(button="left")
                    self.active_mouse_buttons.remove("left")

            elif name == "use": # Right click
                if is_active and "right" not in self.active_mouse_buttons:
                    pydirectinput.mouseDown(button="right")
                    self.active_mouse_buttons.add("right")
                elif not is_active and "right" in self.active_mouse_buttons:
                    pydirectinput.mouseUp(button="right")
                    self.active_mouse_buttons.remove("right")

        # 2. Handle Camera Movement (Continuous relative mouse movement)
        # Scale normalized [-1.0, 1.0] to pixel displacements
        yaw_dx = int(camera_delta[0] * 30.0)
        pitch_dy = int(camera_delta[1] * 20.0)

        if abs(yaw_dx) > 0 or abs(pitch_dy) > 0:
            # relative = True moves relative to current cursor
            pydirectinput.moveRel(yaw_dx, pitch_dy, relative=True)

    def run(self):
        print("\n=======================================================")
        print("          MINECRAFT NEURAL BOT - LIVE RUNNER           ")
        print("=======================================================")
        print("1. Switch focus to the Minecraft game window.")
        print("2. Make sure the game is unpaused (Press Esc in game).")
        print("3. Press ESC on your keyboard at any time to KILL the bot.")
        print("Starting in 5 seconds...")
        time.sleep(5.0)

        interval = 1.0 / config.TARGET_FPS

        with mss.mss() as sct:
            monitor = sct.monitors[1] # Primary monitor

            while self.running:
                loop_start = time.time()

                # 1. Capture screen
                sct_img = sct.grab(monitor)
                frame_bgr = np.array(sct_img)[:, :, :3]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                # 2. Add to 4-frame temporal buffer
                is_full = self.stacker.add_frame(frame_resized)
                if not is_full:
                    time.sleep(0.01)
                    continue

                # 3. Form input tensor [1, 12, 128, 128], float32
                stacked_tensor = self.stacker.get_stacked_state().unsqueeze(0).to(self.device)

                # 4. Policy inference with recurrent GRU memory
                with torch.no_grad():
                    # Input:  stacked_tensor [1, 12, 128, 128], gru_hidden [1, 1, 256]
                    # Output: button_logits [1, 9], camera_delta [1, 2], gru_hidden [1, 1, 256]
                    btn_logits, cam_delta, self.gru_hidden = self.policy(stacked_tensor, self.gru_hidden)

                    # Sigmoid activation on button logits
                    btn_probs = torch.sigmoid(btn_logits).squeeze(0).cpu().numpy()
                    cam_arr = cam_delta.squeeze(0).cpu().numpy()

                # Multi-binary decision (prob > 0.5)
                btn_binary = (btn_probs > 0.5).astype(np.float32)

                # 5. Execute DirectInput control
                self.execute_action(btn_binary, cam_arr)

                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, interval - elapsed)
                time.sleep(sleep_time)

        self.release_all_inputs()
        print("[Live Bot] Bot stopped and all inputs released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live Minecraft neural agent.")
    parser.add_argument("--model", type=str, default="minecraft_policy.pth", help="Path to trained policy checkpoint.")
    args = parser.parse_args()

    bot = LiveMinecraftBot(model_path=args.model)
    try:
        bot.run()
    except Exception as e:
        print(f"[Error] Live loop crashed: {e}")
    finally:
        bot.release_all_inputs()
