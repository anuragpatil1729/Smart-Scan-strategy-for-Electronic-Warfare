"""Turing Synthetic Radar Dataset (TSRD) HDF5 Reader.

Loads 5-dimensional Pulse Descriptor Words (PDWs), ground truth emitter labels,
and transmitter/receiver metadata from the TSRD dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import h5py


@dataclass
class PulseTrainSample:
    """Represents a loaded radar pulse train scenario from TSRD."""

    filename: str
    pdws: np.ndarray              # Shape (N, 5): [ToA, Frequency, PulseWidth, AoA, Amplitude]
    labels: np.ndarray            # Shape (N,): Emitter identity per pulse (int)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_pulses(self) -> int:
        return int(self.pdws.shape[0])

    @property
    def num_emitters(self) -> int:
        return int(len(np.unique(self.labels)))

    @property
    def toa(self) -> np.ndarray:
        """Time of Arrival in microseconds."""
        return self.pdws[:, 0]

    @property
    def frequency(self) -> np.ndarray:
        """Center frequency in MHz."""
        return self.pdws[:, 1]

    @property
    def pulse_width(self) -> np.ndarray:
        """Pulse width in microseconds."""
        return self.pdws[:, 2]

    @property
    def aoa(self) -> np.ndarray:
        """Angle of Arrival in degrees."""
        return self.pdws[:, 3]

    @property
    def amplitude(self) -> np.ndarray:
        """Received pulse amplitude in dB."""
        return self.pdws[:, 4]


class TSRDDatasetReader:
    """Reader and loader for HDF5 radar pulse train files."""

    SPLIT_SUBDIRS = {
        "train_scan": "scan/train_scan",
        "val_scan": "scan/val_scan",
        "test_scan": "scan/test_scan",
        "train_stare": "stare/train_stare",
        "val_stare": "stare/val_stare",
        "test_stare": "stare/test_stare",
    }

    def __init__(self, root_path: Union[str, Path] = "datasets/synthetic/turing_radar_data") -> None:
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"TSRD dataset root directory not found: {self.root_path}")

    def list_files(self, split: str = "train_scan") -> List[Path]:
        """List all .h5 files available in a given split directory."""
        subdir = self.SPLIT_SUBDIRS.get(split, split)
        dir_path = self.root_path / subdir
        if not dir_path.exists():
            raise FileNotFoundError(f"Split directory not found: {dir_path}")

        files = sorted(dir_path.glob("*.h5"), key=lambda p: int(p.stem.split("_")[-1]) if "_" in p.stem and p.stem.split("_")[-1].isdigit() else p.stem)
        return files

    def load_sample(
        self,
        split: str = "train_scan",
        index_or_filename: Union[int, str, Path] = 0,
        max_pulses: Optional[int] = None,
    ) -> PulseTrainSample:
        """Load a single scenario file.

        Args:
            split: Subdirectory split (e.g. 'train_scan', 'val_scan', 'train_stare')
            index_or_filename: Integer index in split or filename
            max_pulses: Optional limit on the number of pulses loaded
        """
        if isinstance(index_or_filename, int):
            files = self.list_files(split)
            if not files:
                raise FileNotFoundError(f"No .h5 files found in split '{split}'")
            filepath = files[index_or_filename % len(files)]
        else:
            filepath = Path(index_or_filename)
            if not filepath.exists() and not filepath.is_absolute():
                subdir = self.SPLIT_SUBDIRS.get(split, split)
                filepath = self.root_path / subdir / filepath

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            if "data" not in f:
                raise KeyError(f"Expected 'data' dataset in {filepath}")

            data_ds = f["data"]
            total_n = data_ds.shape[0]
            n_load = min(total_n, max_pulses) if max_pulses is not None else total_n

            pdws = np.array(data_ds[:n_load], dtype=np.float32)

            if "labels" in f:
                labels_raw = f["labels"][:n_load]
                labels = np.array(labels_raw, dtype=np.int64).reshape(-1)
            else:
                labels = np.zeros(n_load, dtype=np.int64)

            metadata: Dict[str, Any] = {}
            if "metadata" in f:
                meta_grp = f["metadata"]
                for k in meta_grp.keys():
                    try:
                        item = meta_grp[k]
                        if isinstance(item, h5py.Dataset):
                            val = item[()]
                            if isinstance(val, bytes):
                                val = val.decode("utf-8", errors="ignore")
                            metadata[k] = val
                        elif isinstance(item, h5py.Group):
                            # Store subgroup keys
                            metadata[k] = list(item.keys())
                    except Exception:
                        pass

        return PulseTrainSample(
            filename=filepath.name,
            pdws=pdws,
            labels=labels,
            metadata=metadata,
        )
