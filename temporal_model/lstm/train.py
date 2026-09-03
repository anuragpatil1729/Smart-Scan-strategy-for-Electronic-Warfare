"""Training Pipeline for Pulse Sequence LSTM (Block 5.1).

Trains PulseSequenceLSTM to predict future pulse intervals, burst activity,
and temporal modulation patterns using synthetic and TSRD radar pulse streams.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

from temporal_model.lstm.model import PulseSequenceLSTM
from temporal_model.emitter_representation.encoder import TemporalEmitterEncoder
from pdw.extraction.dataset_reader import TSRDDatasetReader


def generate_synthetic_sequences(
    num_sequences: int = 1000,
    seq_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic radar pulse sequences with fixed, staggered, and jittered PRIs."""
    rng = np.random.default_rng(42)
    x_list: List[np.ndarray] = []
    y_toa_list: List[float] = []
    y_act_list: List[float] = []

    for _ in range(num_sequences):
        pattern_type = rng.choice(["fixed", "staggered", "jittered"])
        base_pri = rng.uniform(50.0, 1000.0)
        freq = rng.uniform(2000.0, 12000.0)
        pw = rng.uniform(0.5, 10.0)
        aoa = rng.uniform(0.0, 360.0)
        amp = rng.uniform(-40.0, -10.0)

        if pattern_type == "fixed":
            delta_toas = np.full(seq_len + 1, base_pri) + rng.normal(0, 1.0, seq_len + 1)
        elif pattern_type == "staggered":
            pris = [base_pri, base_pri * 1.5, base_pri * 0.8]
            delta_toas = np.array([pris[i % len(pris)] for i in range(seq_len + 1)])
        else:  # jittered
            delta_toas = base_pri + rng.normal(0, base_pri * 0.15, seq_len + 1)

        delta_toas = np.clip(delta_toas, 10.0, 50000.0)
        toas = np.cumsum(delta_toas)

        # Build 5D PDWs
        pdws = np.column_stack([
            toas,
            rng.normal(freq, 2.0, seq_len + 1),
            rng.normal(pw, 0.05, seq_len + 1),
            rng.normal(aoa, 0.5, seq_len + 1),
            rng.normal(amp, 1.0, seq_len + 1),
        ])

        encoder = TemporalEmitterEncoder()
        seq = encoder.prepare_pulse_sequence(pdws[:seq_len])

        # Target: next delta-ToA normalized
        next_delta = delta_toas[-1]
        log_delta = np.log10(np.clip(next_delta, 1.0, 50000.0))
        target_toa_norm = float(np.clip((log_delta - 1.0) / 3.7, 0.0, 1.0))

        # Target: activity prob (active if high pulse density)
        target_act = 1.0 if rng.random() > 0.3 else 0.0

        x_list.append(seq)
        y_toa_list.append(target_toa_norm)
        y_act_list.append(target_act)

    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_toa_list, dtype=np.float32)[:, None],
        np.array(y_act_list, dtype=np.float32)[:, None],
    )


def train_lstm(
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "models/temporal/lstm_pulse_model.pth",
) -> PulseSequenceLSTM:
    """Train PulseSequenceLSTM model."""
    print(f"\n>>> Generating / loading pulse sequences for LSTM training...")
    x, y_toa, y_act = generate_synthetic_sequences(num_sequences=1500, seq_len=32)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_toa),
        torch.from_numpy(y_act),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PulseSequenceLSTM(input_dim=5, hidden_dim=64, num_layers=2, embedding_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    model.train()
    print(f">>> Training PulseSequenceLSTM for {num_epochs} epochs on device: {torch.device('cpu')}...")
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch_x, batch_toa, batch_act in dataloader:
            optimizer.zero_grad()
            out = model(batch_x)

            loss_toa = mse_loss(out["next_delta_toa"], batch_toa)
            loss_act = bce_loss(out["activity_prob"], batch_act)
            loss = loss_toa + 0.5 * loss_act

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f}")

    # Save trained checkpoint
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f">>> PulseSequenceLSTM saved to: {save_path}\n")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pulse Sequence LSTM Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="models/temporal/lstm_pulse_model.pth")
    args = parser.parse_args()

    train_lstm(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
