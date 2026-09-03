"""PyTorch GRU Model for Pulse Sequence Temporal Dynamics (Block 5.1).

Ablation / comparative alternative to PulseSequenceLSTM with gated recurrent units.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np


class PulseSequenceGRU(nn.Module):
    """Gated Recurrent Unit (GRU) Neural Network for Radar Pulse Sequence Modeling."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.toa_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.activity_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        proj = self.act(self.input_proj(x))
        out, h_n = self.gru(proj, hx)

        last_out = out[:, -1, :]
        embedding = self.embed_head(last_out)

        next_delta_toa = self.toa_head(embedding)
        activity_prob = self.activity_head(embedding)
        uncertainty = self.uncertainty_head(embedding)

        return {
            "embedding": embedding,
            "next_delta_toa": next_delta_toa,
            "activity_prob": activity_prob,
            "uncertainty": uncertainty,
            "hidden": h_n,
        }

    @torch.no_grad()
    def predict_pulse_stream(self, pulse_array: np.ndarray) -> Dict[str, float]:
        self.eval()
        if len(pulse_array) == 0:
            return {"next_delta_toa_norm": 0.5, "activity_prob": 0.5, "uncertainty": 0.5}

        tensor = torch.from_numpy(pulse_array).float().unsqueeze(0)
        out = self.forward(tensor)

        return {
            "next_delta_toa_norm": float(out["next_delta_toa"].squeeze().item()),
            "activity_prob": float(out["activity_prob"].squeeze().item()),
            "uncertainty": float(out["uncertainty"].squeeze().item()),
        }
