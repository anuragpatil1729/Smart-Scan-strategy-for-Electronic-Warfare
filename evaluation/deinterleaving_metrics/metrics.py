"""Deinterleaving Evaluation Metrics.

Computes standard clustering and radar pulse deinterleaving benchmarks:
- V-Measure Score
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
- Homogeneity & Completeness
- Pairwise-binary F1 Score and Matthews Correlation Coefficient (MCC)
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)


def pairwise_f1_and_mcc(labels_true: np.ndarray, labels_pred: np.ndarray, max_pairs: int = 500000) -> Dict[str, float]:
    """Compute pairwise-binary precision, recall, F1, and MCC on pulse clustering.

    Two pulses in the same cluster are considered a positive pair (TP / FP).
    Uses random pair subsampling if the number of pairs N*(N-1)/2 is very large.
    """
    n = len(labels_true)
    if n < 2:
        return {"pairwise_f1": 0.0, "pairwise_mcc": 0.0}

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        # Exact pair evaluation
        idx_i, idx_j = np.triu_indices(n, k=1)
    else:
        # Sample random pairs
        rng = np.random.default_rng(42)
        idx_i = rng.integers(0, n, size=max_pairs)
        idx_j = rng.integers(0, n, size=max_pairs)
        valid = idx_i != idx_j
        idx_i, idx_j = idx_i[valid], idx_j[valid]

    y_true = (labels_true[idx_i] == labels_true[idx_j]).astype(int)
    y_pred = (labels_pred[idx_i] == labels_pred[idx_j]).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    denom = float(
        np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    )
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "pairwise_mcc": mcc,
    }


def evaluate_deinterleaving(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics comparing predicted clusters to ground truth."""
    if len(labels_true) != len(labels_pred):
        raise ValueError(
            f"Mismatched label shapes: true={len(labels_true)}, pred={len(labels_pred)}"
        )

    # Standard scikit-learn clustering scores
    ari = float(adjusted_rand_score(labels_true, labels_pred))
    ami = float(adjusted_mutual_info_score(labels_true, labels_pred))
    v_meas = float(v_measure_score(labels_true, labels_pred))
    homo = float(homogeneity_score(labels_true, labels_pred))
    comp = float(completeness_score(labels_true, labels_pred))

    # Pairwise metrics
    pw_metrics = pairwise_f1_and_mcc(labels_true, labels_pred)

    return {
        "adjusted_rand_index": ari,
        "adjusted_mutual_info": ami,
        "v_measure": v_meas,
        "homogeneity": homo,
        "completeness": comp,
        **pw_metrics,
    }
