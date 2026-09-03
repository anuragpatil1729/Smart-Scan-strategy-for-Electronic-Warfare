"""PyTorch LSTM Model for Pulse Sequence Temporal Dynamics (Block 5.1).

Processes deinterleaved pulse streams (Delta-ToA, Frequency, PulseWidth, Amplitude)
to predict:
1. Future pulse arrival timing (Next Delta-ToA).
2. Probability of transmission in upcoming dwell window (Burst Activity Probability).
3. Temporal latent representation of emitter modulation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class PulseSequenceLSTM(nn.Module):
    """LSTM Neural Network for Radar Pulse Sequence Modeling."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Initialize PulseSequenceLSTM.

        Args:
            input_dim: Feature channels per pulse [delta_toa_norm, freq_norm, pw_norm, aoa_norm, amp_norm]
            hidden_dim: LSTM hidden units
            num_layers: Number of stacked LSTM layers
            embedding_dim: Size of output temporal emitter representation
            dropout: Dropout probability between stacked layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Input feature projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        # Recurrent core
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection for emitter representation embedding
        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Head 1: Predict next Delta-ToA (normalized regression)
        self.toa_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Delta-ToA is strictly positive
        )

        # Head 2: Predict burst activity probability in next dwell window
        self.activity_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Probability in [0, 1]
        )

        # Head 3: Modulation uncertainty (confidence estimation)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over a batch of pulse sequences.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hx: Optional hidden and cell states (h_0, c_0)

        Returns:
            Dictionary containing:
                - "embedding": (batch_size, embedding_dim)
                - "next_delta_toa": (batch_size, 1)
                - "activity_prob": (batch_size, 1)
                - "uncertainty": (batch_size, 1)
                - "hidden": tuple of (h_n, c_n)
        """
        # Linear projection
        proj = self.act(self.input_proj(x))

        # LSTM forward
        out, (h_n, c_n) = self.lstm(proj, hx)

        # Extract last time step representation
        last_out = out[:, -1, :]  # (batch_size, hidden_dim)
        embedding = self.embed_head(last_out)  # (batch_size, embedding_dim)

        # Task heads
        next_delta_toa = self.toa_head(embedding)
        activity_prob = self.activity_head(embedding)
        uncertainty = self.uncertainty_head(embedding)

        return {
            "embedding": embedding,
            "next_delta_toa": next_delta_toa,
            "activity_prob": activity_prob,
            "uncertainty": uncertainty,
            "hidden": (h_n, c_n),
        }

    @torch.no_grad()
    def predict_pulse_stream(
        self,
        pulse_array: np.ndarray,
    ) -> Dict[str, float]:
        """Inference helper for raw numpy pulse streams.

        Args:
            pulse_array: Array of shape (seq_len, 5)

        Returns:
            Dictionary of scalar predictions:
                - next_delta_toa_norm: float
                - activity_prob: float
                - uncertainty: float
        """
        self.eval()
        if len(pulse_array) == 0:
            return {"next_delta_toa_norm": 0.5, "activity_prob": 0.5, "uncertainty": 0.5}

        tensor = torch.from_numpy(pulse_array).float().unsqueeze(0)  # (1, seq_len, 5)
        out = self.forward(tensor)

        return {
            "next_delta_toa_norm": float(out["next_delta_toa"].squeeze().item()),
            "activity_prob": float(out["activity_prob"].squeeze().item()),
            "uncertainty": float(out["uncertainty"].squeeze().item()),
        }
