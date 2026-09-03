"""Temporal Emitter Encoder and Feature Generator (Block 5.2).

Extracts sequence-level representations from deinterleaved pulse streams using
the trained LSTM/GRU model to output:
1. Temporal latent embedding (32-D representation of radar PRI pattern)
2. Predicted activity probability in upcoming dwell window
3. Predicted next PRI timing
4. Temporal parameter uncertainty
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import torch

from temporal_model.lstm.model import PulseSequenceLSTM
from temporal_model.gru.model import PulseSequenceGRU
from pdw.features.pdw_features import PDWFeatureScaler, compute_delta_toa


class TemporalEmitterEncoder:
    """Encodes deinterleaved pulse sequences into temporal emitter representations."""

    def __init__(
        self,
        model: Optional[Union[PulseSequenceLSTM, PulseSequenceGRU]] = None,
        max_seq_len: int = 64,
        scaler: Optional[PDWFeatureScaler] = None,
    ) -> None:
        self.model = model or PulseSequenceLSTM()
        self.max_seq_len = max_seq_len
        self.scaler = scaler or PDWFeatureScaler()

    def prepare_pulse_sequence(self, pdws: np.ndarray) -> np.ndarray:
        """Transform raw 5D PDWs into normalized temporal sequence for LSTM.

        Args:
            pdws: Array of shape (N, 5) [ToA, Frequency, PulseWidth, AoA, Amplitude]

        Returns:
            Normalized array of shape (min(N, max_seq_len), 5):
            [delta_toa_norm, freq_norm, pw_norm, aoa_norm, amp_norm]
        """
        if len(pdws) == 0:
            return np.zeros((1, 5), dtype=np.float32)

        # Sort chronologically by ToA
        sorted_pdws = pdws[np.argsort(pdws[:, 0])]
        if len(sorted_pdws) > self.max_seq_len:
            sorted_pdws = sorted_pdws[-self.max_seq_len :]

        # 1. Delta-ToA
        delta_toas = compute_delta_toa(sorted_pdws[:, 0])
        if len(delta_toas) > 1 and delta_toas[0] == 0.0:
            delta_toas[0] = delta_toas[1]
        elif len(delta_toas) == 1:
            delta_toas[0] = 250.0
        # Log-scale normalize Delta-ToA (typical 10 us to 10,000 us -> [0, 1])
        log_delta = np.log10(np.clip(delta_toas, 1.0, 50000.0))
        delta_toa_norm = (log_delta - 1.0) / 3.7  # maps ~10us to 0.0, ~50000us to 1.0
        delta_toa_norm = np.clip(delta_toa_norm, 0.0, 1.0)

        # 2. Normalized Frequency, PW, AoA, Amplitude
        norm_pdws = self.scaler.extract_all_normalized(sorted_pdws)

        # Combine into 5-channel sequence
        seq = np.column_stack([
            delta_toa_norm,
            norm_pdws[:, 1],  # Freq [0, 1]
            norm_pdws[:, 2],  # PW [0, 1]
            norm_pdws[:, 3],  # AoA [0, 1]
            norm_pdws[:, 4],  # Amp [0, 1]
        ]).astype(np.float32)

        return seq

    @torch.no_grad()
    def encode_cluster(self, cluster_pdws: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Process a deinterleaved pulse cluster to produce temporal features.

        Args:
            cluster_pdws: Array of shape (N, 5)

        Returns:
            Dictionary containing:
                - activity_prob: float in [0.1, 0.95]
                - predicted_next_delta_toa_us: float (denormalized us)
                - temporal_uncertainty: float in [0.05, 0.95]
                - embedding: np.ndarray of shape (32,)
        """
        if len(cluster_pdws) == 0:
            return {
                "activity_prob": 0.5,
                "predicted_next_delta_toa_us": 250.0,
                "temporal_uncertainty": 0.8,
                "embedding": np.zeros(32, dtype=np.float32),
            }

        seq = self.prepare_pulse_sequence(cluster_pdws)
        preds = self.model.predict_pulse_stream(seq)

        # Denormalize predicted delta-ToA from log space back to microseconds
        norm_val = np.clip(preds["next_delta_toa_norm"], 0.0, 1.0)
        predicted_log = norm_val * 3.7 + 1.0
        predicted_pri_us = float(10.0 ** predicted_log)

        # Extract latent temporal embedding
        tensor = torch.from_numpy(seq).float().unsqueeze(0)
        out = self.model(tensor)
        embedding = out["embedding"].squeeze(0).cpu().numpy()

        return {
            "activity_prob": float(np.clip(preds["activity_prob"], 0.05, 0.95)),
            "predicted_next_delta_toa_us": float(np.clip(predicted_pri_us, 10.0, 50000.0)),
            "temporal_uncertainty": float(np.clip(preds["uncertainty"], 0.05, 0.95)),
            "embedding": embedding,
        }
