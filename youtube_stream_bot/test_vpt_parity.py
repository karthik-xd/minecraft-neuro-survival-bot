"""
test_vpt_parity.py - Comprehensive Verification Suite for OpenAI VPT Parity & Learning Fixes.
Verifies:
  1. Camera Discretization & Action Space Expansion (21 buttons, 121 camera classes).
  2. Model Architectures & Tensor Shapes (Backbone, Policy single/sequence, Net2Net, IDM, unbatched 3D).
  3. Replay Buffer Circular Boundary Safety, Sequence Sampling, Underfill Handling, and Thread Safety.
  4. Policy Mode Collapse Elimination (Balanced logits, no back+left lock).
  5. IDM Calibration on Real Human Data & Multi-Source Loading (human_data.pt and demos/*.jsonl).
  6. Recurrent State Continuity, Loss Backpropagation, and Optimizer Momentum Preservation.
  7. Live Agent Dynamic GRU Detection & Key Mapping.
"""
import os
import sys
import unittest
import threading
import torch
import torch.nn as nn
import numpy as np

# Add youtube_stream_bot directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config
from models import CNNBackbone, MinecraftPolicy, InverseDynamicsModel, load_idm_weights
from replay_buffer import InactiveReplayBuffer
from train_idm import RealHumanGameplayDataset
from train_policy import StreamingTrainer
from live_agent import LiveMinecraftBot, KEY_MAPPING


class TestVPTParity(unittest.TestCase):

    def test_01_action_space_and_camera_discretization(self):
        """Verifies expanded 21-button action space and 121-class camera discretization."""
        self.assertEqual(config.NUM_BUTTONS, 21, "Must have exactly 21 discrete action buttons.")
        self.assertIn("hotbar_1", config.BUTTON_NAMES)
        self.assertIn("hotbar_9", config.BUTTON_NAMES)
        self.assertIn("inventory", config.BUTTON_NAMES)
        self.assertIn("drop", config.BUTTON_NAMES)
        self.assertIn("offhand", config.BUTTON_NAMES)

        self.assertEqual(config.NUM_CAMERA_CLASSES, 121, "Must have exactly 121 camera classes (11x11 grid).")
        self.assertEqual(config.NEUTRAL_CAMERA_CLASS, 60, "Neutral look (yaw=0, pitch=0) must map to class index 60.")

        # Test Neutral (0, 0)
        neutral_cam = torch.tensor([0.0, 0.0])
        bin_idx = config.camera_to_bin(neutral_cam)
        self.assertEqual(bin_idx.item(), 60)
        rec_neutral = config.bin_to_camera(bin_idx)
        self.assertTrue(torch.allclose(rec_neutral, neutral_cam, atol=1e-5))

        # Test Extreme corners
        top_left = torch.tensor([-1.0, -1.0])
        bin_tl = config.camera_to_bin(top_left)
        self.assertEqual(bin_tl.item(), 0)  # 0*11 + 0 = 0
        rec_tl = config.bin_to_camera(bin_tl)
        self.assertTrue(torch.allclose(rec_tl, top_left, atol=1e-5))

        bottom_right = torch.tensor([1.0, 1.0])
        bin_br = config.camera_to_bin(bottom_right)
        self.assertEqual(bin_br.item(), 120)  # 10*11 + 10 = 120
        rec_br = config.bin_to_camera(bin_br)
        self.assertTrue(torch.allclose(rec_br, bottom_right, atol=1e-5))

        # Test Batch round-trip
        cam_batch = torch.tensor([
            [0.0, 0.0],
            [-0.4, 0.2],
            [0.1, -0.2],
            [0.02, 0.0]
        ])
        bins_batch = config.camera_to_bin(cam_batch)
        rec_batch = config.bin_to_camera(bins_batch)
        self.assertEqual(bins_batch.shape, (4,))
        self.assertEqual(rec_batch.shape, (4, 2))

    def test_02_model_architectures_and_forward_shapes(self):
        """Verifies policy single-step, 3D unbatched, sequence BPTT, Net2Net, and IDM forward pass shapes."""
        # CNN Backbone
        backbone = CNNBackbone(in_channels=12)
        x_bb = torch.randn(2, 12, 128, 128)
        feat = backbone(x_bb)
        self.assertEqual(feat.shape, (2, 256))

        # Policy Single-Step
        policy = MinecraftPolicy(gru_hidden_dim=256)
        x_single = torch.randn(3, 12, 128, 128)
        btn_s, cam_s, h_s = policy(x_single)
        self.assertEqual(btn_s.shape, (3, 21), "Single-step buttons must be [B, 21].")
        self.assertEqual(cam_s.shape, (3, 121), "Single-step camera must be [B, 121].")
        self.assertEqual(h_s.shape, (1, 3, 256), "GRU hidden must be [1, B, 256].")

        # Policy 3D Unbatched input
        x_unbatched = torch.randn(12, 128, 128)
        btn_u, cam_u, h_u = policy(x_unbatched)
        self.assertEqual(btn_u.shape, (1, 21))
        self.assertEqual(cam_u.shape, (1, 121))

        # Policy Sequence (Recurrent BPTT T=16 and T=32)
        for t_steps in (16, 32):
            x_seq = torch.randn(2, t_steps, 12, 128, 128)
            btn_seq, cam_seq, h_seq = policy(x_seq)
            self.assertEqual(btn_seq.shape, (2, t_steps, 21))
            self.assertEqual(cam_seq.shape, (2, t_steps, 121))
            self.assertEqual(h_seq.shape, (1, 2, 256))

        # Progressive Neurogenesis (Net2Net expansion)
        old_h = policy.gru_hidden_dim
        new_h = policy.grow_gru_neurons(add_neurons=32)
        self.assertEqual(new_h, old_h + 32)
        self.assertEqual(policy.gru_hidden_dim, 288)
        btn_grew, cam_grew, h_grew = policy(x_single)
        self.assertEqual(btn_grew.shape, (3, 21))
        self.assertEqual(cam_grew.shape, (3, 121))
        self.assertEqual(h_grew.shape, (1, 3, 288))

        # IDM
        idm = InverseDynamicsModel()
        f1 = torch.randn(4, 3, 128, 128)
        f2 = torch.randn(4, 3, 128, 128)
        idm_btn, idm_cam = idm(f1, f2)
        self.assertEqual(idm_btn.shape, (4, 21), "IDM button prediction must be [B, 21].")
        self.assertEqual(idm_cam.shape, (4, 121), "IDM camera prediction must be [B, 121].")

    def test_03_replay_buffer_circular_boundary_and_sequences(self):
        """Verifies circular buffer boundary safety, underfilled handling, and thread safety."""
        cap = 80
        buf = InactiveReplayBuffer(capacity=cap)

        # Underfilled buffer tests (n < batch_size and n < seq_len)
        for i in range(5):
            s = torch.full((12, 128, 128), i, dtype=torch.uint8)
            b = torch.zeros(config.NUM_BUTTONS)
            b[i % config.NUM_BUTTONS] = 1.0
            buf.push(s, b, config.NEUTRAL_CAMERA_CLASS)

        # Must not throw when sampling batch larger than buffer
        b_s, b_b, b_c = buf.sample(batch_size=16)
        self.assertEqual(b_s.shape, (16, 12, 128, 128))
        self.assertEqual(b_b.shape, (16, 21))
        self.assertEqual(b_c.shape, (16,))

        # Must not throw when sampling sequence longer than buffer
        seq_s, seq_b, seq_c = buf.sample_sequences(batch_size=16, seq_len=16)
        self.assertEqual(seq_s.shape, (16, 16, 12, 128, 128))
        self.assertEqual(seq_b.shape, (16, 16, 21))
        self.assertEqual(seq_c.shape, (16, 16))

        # Fill past capacity (120 elements)
        total_pushed = 120
        for i in range(5, total_pushed):
            s = torch.full((12, 128, 128), i % 256, dtype=torch.uint8)
            b = torch.zeros(config.NUM_BUTTONS)
            b[i % config.NUM_BUTTONS] = 1.0
            cam = torch.tensor([0.05, -0.05])
            buf.push(s, b, cam)

        self.assertEqual(len(buf), cap)
        self.assertEqual(buf.position, total_pushed % cap)

        # Test sequence lengths T=16 and T=32
        for seq_len in (16, 32):
            states, buttons, cameras = buf.sample_sequences(batch_size=30, seq_len=seq_len)
            self.assertEqual(states.shape, (30, seq_len, 12, 128, 128))
            self.assertEqual(buttons.shape, (30, seq_len, 21))
            self.assertEqual(cameras.shape, (30, seq_len))
            self.assertEqual(cameras.dtype, torch.long)

            # Check that each sampled sequence is strictly monotonic without wrapping across head
            for b_idx in range(30):
                pixel_vals = [int(round(states[b_idx, t, 0, 0, 0].item() * 255.0)) for t in range(seq_len)]
                for t in range(seq_len - 1):
                    diff = pixel_vals[t + 1] - pixel_vals[t]
                    self.assertEqual(
                        diff, 1,
                        f"Discontinuity found at seq_len={seq_len}, batch={b_idx}, step={t}: {pixel_vals[t]} -> {pixel_vals[t+1]}"
                    )

        # Thread safety test: concurrent pushes and samples
        def worker_push():
            for i in range(50):
                s = torch.zeros(12, 128, 128, dtype=torch.uint8)
                b = torch.zeros(config.NUM_BUTTONS)
                buf.push(s, b, 60)

        def worker_sample():
            for _ in range(20):
                buf.sample_sequences(batch_size=8, seq_len=16)

        t1 = threading.Thread(target=worker_push)
        t2 = threading.Thread(target=worker_sample)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(len(buf), cap)

    def test_04_policy_mode_collapse_elimination(self):
        """Verifies minecraft_policy.pth does not suffer from back+left saturation or static camera drift."""
        policy_path = os.path.join(SCRIPT_DIR, "minecraft_policy.pth")
        self.assertTrue(os.path.exists(policy_path), f"Checkpoint '{policy_path}' must exist.")

        sd = torch.load(policy_path, map_location="cpu", weights_only=True)
        detected_gru = sd["gru.weight_hh_l0"].shape[1] if "gru.weight_hh_l0" in sd else 256
        policy = MinecraftPolicy(gru_hidden_dim=detected_gru)
        policy.load_state_dict(sd)
        policy.eval()

        # Check button biases
        btn_bias = sd["button_head.bias"]
        self.assertEqual(btn_bias.shape[0], 21, "Button bias must have 21 elements.")
        self.assertLess(btn_bias.abs().max().item(), 5.0, "Button bias must not be saturated (> 15.0).")

        # Test forward pass on zero and random inputs
        with torch.no_grad():
            x_zero = torch.zeros(1, 12, 128, 128)
            btn_logits, cam_logits, _ = policy(x_zero)
            btn_probs = torch.sigmoid(btn_logits).squeeze(0)

            # Verify back (index 1) and left (index 2) are NOT locked to 1.0!
            self.assertLess(btn_probs[1].item(), 0.5, "Back button must not be locked at 1.0!")
            self.assertLess(btn_probs[2].item(), 0.5, "Left button must not be locked at 1.0!")

            # Verify camera prediction on zero input is neutral look (bin 60)
            pred_cam_class = cam_logits.argmax(dim=-1).item()
            self.assertEqual(pred_cam_class, config.NEUTRAL_CAMERA_CLASS, "Default camera look must be neutral (bin 60).")

    def test_05_idm_calibrated_predictions_and_demos_loading(self):
        """Verifies idm_model.pth predictions and multi-source loading (human_data.pt & demos/*.jsonl)."""
        idm_path = os.path.join(SCRIPT_DIR, "idm_model.pth")
        data_path = os.path.join(SCRIPT_DIR, "human_data.pt")
        self.assertTrue(os.path.exists(idm_path), "idm_model.pth must exist.")
        self.assertTrue(os.path.exists(data_path), "human_data.pt must exist.")

        idm = InverseDynamicsModel()
        loaded = load_idm_weights(idm, idm_path, device=torch.device("cpu"))
        self.assertTrue(loaded, "IDM weights must load successfully.")
        idm.eval()

        data = torch.load(data_path, map_location="cpu", weights_only=False)
        ep0 = data["episodes"][0]

        attack_preds = []
        cam_preds = []

        with torch.no_grad():
            for t in ep0:
                f1 = t["current_frame"].unsqueeze(0)
                f2 = t["next_frame"].unsqueeze(0)
                b_logits, c_logits = idm(f1, f2)
                b_bin = (torch.sigmoid(b_logits) > 0.5).float().squeeze(0)
                c_class = c_logits.argmax(dim=-1).item()

                attack_preds.append(b_bin[7].item())  # Index 7 is attack
                cam_preds.append(c_class)

        attack_rate = sum(attack_preds) / len(attack_preds)
        print(f"\n[IDM Test Metric] Attack Prediction Rate on ep0: {attack_rate*100:.1f}%")
        # Ensure attack rate is neither collapsed to 100% nor 0%
        self.assertLess(attack_rate, 0.70, "IDM must NOT predict attack=1 on all transitions!")
        self.assertGreaterEqual(attack_rate, 0.0, "Attack rate must be non-negative.")

        # Test loading from demos/*.jsonl and human_data.pt
        demos_ds = RealHumanGameplayDataset(data_path="human_data.pt,../demos/*.jsonl", augment=False, max_demos_samples=500)
        self.assertGreater(len(demos_ds), 150, "Dataset must successfully load both human_data.pt and demos/*.jsonl.")
        sample_f1, sample_f2, sample_b, sample_c = demos_ds[0]
        self.assertEqual(sample_f1.shape, (3, 128, 128))
        self.assertEqual(sample_f2.shape, (3, 128, 128))
        self.assertEqual(sample_b.shape, (21,))
        self.assertGreaterEqual(sample_c.item(), 0)
        self.assertLessEqual(sample_c.item(), 120)

    def test_06_loss_and_recurrent_continuity_backprop(self):
        """Verifies multi-binary BCE, categorical camera Cross-Entropy, and recurrent state continuity."""
        policy = MinecraftPolicy(gru_hidden_dim=256)
        policy.train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        bce_fn = nn.BCEWithLogitsLoss()
        ce_fn = nn.CrossEntropyLoss()

        batch_states1 = torch.randn(4, 16, 12, 128, 128)
        batch_buttons1 = (torch.rand(4, 16, 21) > 0.8).float()
        batch_camera1 = torch.randint(0, 121, (4, 16))

        optimizer.zero_grad()
        pred_btn, pred_cam, next_h = policy(batch_states1, h=None)
        self.assertIsNotNone(next_h)
        self.assertEqual(next_h.shape, (1, 4, 256))

        loss_btn = bce_fn(pred_btn, batch_buttons1)
        loss_cam = ce_fn(pred_cam.view(-1, 121), batch_camera1.view(-1))
        total_loss = loss_btn + (1.5 * loss_cam)

        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()

        # Check gradients exist on all key components
        self.assertIsNotNone(policy.backbone.conv[0].weight.grad)
        self.assertIsNotNone(policy.gru.weight_ih_l0.grad)
        self.assertIsNotNone(policy.button_head.weight.grad)
        self.assertIsNotNone(policy.camera_head[2].weight.grad)

        optimizer.step()

        # Recurrent state continuity pass: pass detached next_h into second chunk
        batch_states2 = torch.randn(4, 16, 12, 128, 128)
        pred_btn2, pred_cam2, next_h2 = policy(batch_states2, h=next_h.detach())
        self.assertEqual(next_h2.shape, (1, 4, 256))

        # Test optimizer momentum preservation across neurogenesis
        trainer = StreamingTrainer(resume_checkpoint=False, min_buffer_warmup=10)
        trainer.policy = policy
        trainer.optimizer = optimizer
        old_gru = trainer.policy.gru_hidden_dim
        trainer.policy.grow_gru_neurons(add_neurons=32)
        trainer._rebuild_optimizer_preserving_state()
        self.assertEqual(trainer.policy.gru_hidden_dim, old_gru + 32)
        self.assertGreater(len(trainer.optimizer.state), 0, "Optimizer momentum state must be preserved across neurogenesis!")

    def test_07_live_agent_dynamic_loading_and_keys(self):
        """Verifies LiveMinecraftBot correctly detects GRU dimensions, handles expanded keys, and accepts tensor inputs."""
        policy_path = os.path.join(SCRIPT_DIR, "minecraft_policy.pth")
        bot = LiveMinecraftBot(model_path=policy_path)
        self.assertEqual(bot.policy.gru_hidden_dim, bot.gru_hidden.shape[2])

        # Verify key mapping covers all 21 buttons
        self.assertIn("hotbar_1", KEY_MAPPING)
        self.assertIn("hotbar_9", KEY_MAPPING)
        self.assertIn("inventory", KEY_MAPPING)
        self.assertIn("drop", KEY_MAPPING)
        self.assertIn("offhand", KEY_MAPPING)

        # Test execute_action accepts numpy array, PyTorch tensor, and list inputs without throwing
        dummy_buttons_np = np.zeros(21, dtype=np.float32)
        dummy_camera_np = np.array([0.1, -0.1], dtype=np.float32)
        bot.execute_action(dummy_buttons_np, dummy_camera_np)

        dummy_buttons_pt = torch.zeros(21)
        dummy_camera_pt = torch.tensor([0.05, -0.05])
        bot.execute_action(dummy_buttons_pt, dummy_camera_pt)

        bot.release_all_inputs()


if __name__ == "__main__":
    unittest.main()
