"""
train_idm.py - Trains the Inverse Dynamics Model (IDM) on labeled human gameplay transitions.
Features:
  1. Episode-level validation splitting (NEVER random frame splitting, preventing temporal leakage).
  2. Positive class weighting in BCEWithLogitsLoss to handle severe class imbalance for rare actions (jump, attack, sneak).
  3. Combined multi-task objective: Multi-binary classification (BCE) + continuous camera regression (Smooth L1).
  4. Saves best model checkpoint to 'idm_model.pth'.

Explicit Tensor Shapes:
  - current_frame: [B, 3, 128, 128], float32
  - next_frame:    [B, 3, 128, 128], float32
  - target_buttons: [B, 9], float32
  - target_camera:  [B, 2], float32
  - pred_buttons:  [B, 9], float32 (logits)
  - pred_camera:   [B, 2], float32 (displacements in [-1.0, 1.0])
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config
from models import InverseDynamicsModel


class EpisodeTransitionDataset(Dataset):
    def __init__(self, episodes: list):
        """Flattens transitions from selected episodes."""
        self.samples = []
        for ep in episodes:
            for transition in ep:
                self.samples.append(transition)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            item["current_frame"],  # [3, 128, 128], float32
            item["next_frame"],     # [3, 128, 128], float32
            item["buttons"],        # [9], float32
            item["camera"]          # [2], float32
        )


def compute_class_weights(dataset: EpisodeTransitionDataset) -> torch.Tensor:
    """
    Computes inverse frequency positive weights (num_neg / num_pos) per button action
    to prevent the model from degenerate all-zero predictions on imbalanced buttons.
    Returns:
        pos_weight: Tensor [9], float32
    """
    all_buttons = torch.stack([s["buttons"] for s in dataset.samples], dim=0) # [N, 9]
    total_samples = all_buttons.size(0)
    positives = all_buttons.sum(dim=0) # [9]
    negatives = total_samples - positives # [9]

    # pos_weight = negatives / (positives + 1e-5)
    pos_weight = (negatives + 1.0) / (positives + 1.0)
    # Clamp extreme weights for stability
    pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)
    return pos_weight


def train_idm(data_path: str = "human_data.pt",
              epochs: int = 15,
              batch_size: int = config.BATCH_SIZE,
              learning_rate: float = config.IDM_LR,
              val_split_ratio: float = 0.2):
    """
    Trains the IDM with strict episode-level validation separation.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset '{data_path}' not found! Run 'python record_human.py --synthetic' first."
        )

    print(f"[IDM Training] Loading dataset from: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    episodes = data["episodes"]
    num_episodes = len(episodes)
    print(f"[IDM Training] Found {num_episodes} gameplay episodes.")

    if num_episodes < 2:
        print("[Warning] Only 1 episode provided. Validation split requires >= 2 episodes.")
        num_val = 0
        train_eps = episodes
        val_eps = episodes
    else:
        num_val = max(1, int(num_episodes * val_split_ratio))
        # Split by episode index: early episodes for train, distinct trailing episodes for validation
        train_eps = episodes[:-num_val]
        val_eps = episodes[-num_val:]

    train_ds = EpisodeTransitionDataset(train_eps)
    val_ds = EpisodeTransitionDataset(val_eps)

    print(f"[IDM Training] Split: {len(train_eps)} Train Episodes ({len(train_ds)} samples) | "
          f"{len(val_eps)} Val Episodes ({len(val_ds)} samples)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class weights for multi-binary buttons
    pos_weights = compute_class_weights(train_ds).to(config.DEVICE) # [9]
    print(f"[IDM Training] Action positive weights: {pos_weights.cpu().numpy().round(2)}")

    model = InverseDynamicsModel().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    camera_loss_fn = nn.SmoothL1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        for current_frame, next_frame, target_buttons, target_camera in train_loader:
            # Transfer batch to device
            # Shapes:
            #   current_frame:  [B, 3, 128, 128]
            #   next_frame:     [B, 3, 128, 128]
            #   target_buttons: [B, 9]
            #   target_camera:  [B, 2]
            current_frame = current_frame.to(config.DEVICE)
            next_frame = next_frame.to(config.DEVICE)
            target_buttons = target_buttons.to(config.DEVICE)
            target_camera = target_camera.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            pred_button_logits, pred_camera = model(current_frame, next_frame)
            # pred_button_logits: [B, 9]
            # pred_camera:        [B, 2]

            loss_buttons = bce_loss_fn(pred_button_logits, target_buttons)
            loss_camera = camera_loss_fn(pred_camera, target_camera)

            # Combined multi-task loss
            total_loss = loss_buttons + (2.0 * loss_camera)
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(1, train_batches)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        correct_buttons = 0
        total_button_preds = 0

        with torch.no_grad():
            for current_frame, next_frame, target_buttons, target_camera in val_loader:
                current_frame = current_frame.to(config.DEVICE)
                next_frame = next_frame.to(config.DEVICE)
                target_buttons = target_buttons.to(config.DEVICE)
                target_camera = target_camera.to(config.DEVICE)

                pred_button_logits, pred_camera = model(current_frame, next_frame)
                loss_buttons = bce_loss_fn(pred_button_logits, target_buttons)
                loss_camera = camera_loss_fn(pred_camera, target_camera)
                val_loss = loss_buttons + (2.0 * loss_camera)

                total_val_loss += val_loss.item()
                val_batches += 1

                # Calculate binary accuracy (thresholded at logit > 0 -> prob > 0.5)
                pred_binary = (torch.sigmoid(pred_button_logits) > 0.5).float()
                correct_buttons += (pred_binary == target_buttons).sum().item()
                total_button_preds += target_buttons.numel()

        avg_val_loss = total_val_loss / max(1, val_batches)
        button_acc = (correct_buttons / max(1, total_button_preds)) * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Button Acc: {button_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "idm_model.pth")
            print(f"  --> Saved new best IDM model checkpoint to 'idm_model.pth'")

    print("\n[IDM Training Completed Successfully]")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Inverse Dynamics Model on recorded human play.")
    parser.add_argument("--data", type=str, default="human_data.pt", help="Path to labeled transitions dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=config.IDM_LR, help="AdamW learning rate.")
    args = parser.parse_args()

    train_idm(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
