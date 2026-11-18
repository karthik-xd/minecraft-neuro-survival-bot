"""
live_agent.py - Real-time inference agent that controls Minecraft directly.
Updated for OpenAI VPT Parity:
  - Dynamically loads GRU hidden dimension from checkpoint (256, 288, 320, etc.).
  - Decodes 121-class categorical camera predictions into smooth relative mouse movements.
  - Supports expanded 21-button action space (movement, hotbar 1-9, inventory, drop, offhand).
  - Handles missing dependencies cleanly without crashing at import time.
"""
import time
import argparse
import numpy as np
import torch

import config
from frame_stacker import FrameStacker
from models import MinecraftPolicy

# Optional dependencies handled gracefully
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

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
    "sprint": "ctrl",
    "hotbar_1": "1",
    "hotbar_2": "2",
    "hotbar_3": "3",
    "hotbar_4": "4",
    "hotbar_5": "5",
    "hotbar_6": "6",
    "hotbar_7": "7",
    "hotbar_8": "8",
    "hotbar_9": "9",
    "inventory": "e",
    "drop": "q",
    "offhand": "f"
}


class LiveMinecraftBot:
    def __init__(self, model_path: str = "minecraft_policy.pth"):
        self.device = config.DEVICE
        self.model_path = model_path
        self.running = True
        self.active_keys = set()
        self.active_mouse_buttons = set()

        print(f"[Live Bot] Loading policy checkpoint from: {model_path}")
        sd = torch.load(model_path, map_location=self.device, weights_only=True)

        # Dynamic detection of GRU hidden dimension
        if "gru.weight_hh_l0" in sd:
            detected_gru = sd["gru.weight_hh_l0"].shape[1]
        else:
            detected_gru = 256
        print(f"[Live Bot] Detected GRU hidden dimension: {detected_gru}")

        self.policy = MinecraftPolicy(gru_hidden_dim=detected_gru).to(self.device)
        self.policy.load_state_dict(sd)
        self.policy.eval()

        self.stacker = FrameStacker(stack_size=config.STACK_SIZE)

        # Recurrent GRU hidden memory state: [1, 1, detected_gru], float32
        self.gru_hidden = torch.zeros(1, 1, detected_gru, device=self.device)

        # Safety kill switch listener
        if HAS_PYNPUT:
            self.esc_listener = keyboard.Listener(on_press=self._on_key_press)
            self.esc_listener.start()
        else:
            print("[Live Bot Notice] pynput not installed; safety killswitch disabled. Run with Ctrl+C in terminal to stop.")

        if not HAS_MSS:
            print("[Live Bot Warning] mss is not installed. Screen capture will not function.")
        if not HAS_DIRECTINPUT:
            print("[Live Bot Warning] pydirectinput is not installed. Keyboard/mouse actions will run in simulation mode.")

    def _on_key_press(self, key):
        if key == keyboard.Key.esc:
            print("\n[Safety Killswitch Activated] Emergency Stop Triggered by User!")
            self.running = False

    def release_all_inputs(self):
        """Emergency cleanup to ensure keys don't get stuck."""
        if not HAS_DIRECTINPUT:
            self.active_keys.clear()
            self.active_mouse_buttons.clear()
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

    def execute_action(self, buttons_binary, camera_delta):
        """
        Executes keyboard and mouse commands in the game.
        buttons_binary: ndarray or Tensor [21] (0 or 1 for each button)
        camera_delta:   ndarray or Tensor [2] (yaw dx, pitch dy deltas normalized [-1.0, 1.0])
        """
        if isinstance(buttons_binary, torch.Tensor):
            buttons_binary = buttons_binary.detach().cpu().numpy()
        else:
            buttons_binary = np.asarray(buttons_binary, dtype=np.float32)

        if isinstance(camera_delta, torch.Tensor):
            camera_delta = camera_delta.detach().cpu().numpy()
        else:
            camera_delta = np.asarray(camera_delta, dtype=np.float32)

        if not HAS_DIRECTINPUT:
            return

        # Enforce movement mutual exclusivity
        if len(buttons_binary) > 3:
            if buttons_binary[0] > 0.5 and buttons_binary[1] > 0.5:
                buttons_binary[1] = 0.0
            if buttons_binary[2] > 0.5 and buttons_binary[3] > 0.5:
                buttons_binary[3] = 0.0

        # 1. Handle Keyboard & Mouse Click actions
        for i, name in enumerate(config.BUTTON_NAMES):
            if i >= len(buttons_binary):
                break
            is_active = bool(buttons_binary[i] > 0.5)

            if name in KEY_MAPPING:
                key_code = KEY_MAPPING[name]
                if is_active and key_code not in self.active_keys:
                    pydirectinput.keyDown(key_code)
                    self.active_keys.add(key_code)
                elif not is_active and key_code in self.active_keys:
                    pydirectinput.keyUp(key_code)
                    self.active_keys.remove(key_code)

            elif name == "attack":  # Left click
                if is_active and "left" not in self.active_mouse_buttons:
                    pydirectinput.mouseDown(button="left")
                    self.active_mouse_buttons.add("left")
                elif not is_active and "left" in self.active_mouse_buttons:
                    pydirectinput.mouseUp(button="left")
                    self.active_mouse_buttons.remove("left")

            elif name == "use":  # Right click
                if is_active and "right" not in self.active_mouse_buttons:
                    pydirectinput.mouseDown(button="right")
                    self.active_mouse_buttons.add("right")
                elif not is_active and "right" in self.active_mouse_buttons:
                    pydirectinput.mouseUp(button="right")
                    self.active_mouse_buttons.remove("right")

        # 2. Handle Camera Movement (Discretized bins decoded to continuous relative pixels)
        # Scale normalized [-1.0, 1.0] to pixel displacements
        yaw_dx = int(camera_delta[0] * 30.0)
        pitch_dy = int(camera_delta[1] * 20.0)

        if abs(yaw_dx) > 0 or abs(pitch_dy) > 0:
            pydirectinput.moveRel(yaw_dx, pitch_dy, relative=True)

    def run(self):
        if not HAS_MSS or not HAS_CV2:
            raise RuntimeError("Live agent requires 'mss' and 'opencv-python' for desktop screen capture. Install them to run live.")

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
            monitor = sct.monitors[1]  # Primary monitor

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
                    # Input:  stacked_tensor [1, 12, 128, 128], gru_hidden [1, 1, H]
                    # Output: btn_logits [1, 21], cam_logits [1, 121], gru_hidden [1, 1, H]
                    btn_logits, cam_logits, self.gru_hidden = self.policy(stacked_tensor, self.gru_hidden)

                    btn_probs = torch.sigmoid(btn_logits).squeeze(0).cpu().numpy()

                    # Categorical camera class decoding:
                    cam_class = torch.argmax(cam_logits, dim=-1).squeeze(0)
                    cam_arr = config.bin_to_camera(cam_class).cpu().numpy()

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
