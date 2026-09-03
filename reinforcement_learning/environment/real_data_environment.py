"""Gymnasium Environment driven by Real Turing Radar Dataset Scenarios.

Streams through the 2,500 HDF5 pulse train files in the TSRD dataset:
- Each episode loads a real scenario file from `train_scan` (or test/val splits).
- Partitions the scenario into scheduling dwell windows across time.
- Employs real intercepted pulse arrivals to determine physical detection,
  interception, parameter uncertainty, and track drops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pdw.extraction.dataset_reader import TSRDDatasetReader, PulseTrainSample
from deinterleaving.emitter_tracker import EmitterTracker, EmitterTrack
from reinforcement_learning.state.upstream_interface import (
    UpstreamScanContext,
    UpstreamStateAdapter,
)
from reinforcement_learning.reward.reward_function import (
    EWRewardFunction,
    RewardWeights,
    StepObservationResult,
)


class RealDataEWEnvironment(gym.Env):
    """Real-Data Gymnasium Environment for Electronic Warfare Smart Scan Strategy."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        dataset_path: Union[str, Path] = "datasets/synthetic/turing_radar_data",
        split: str = "train_scan",
        num_emitters: int = 8,
        max_steps: int = 100,
        initial_budget: float = 100.0,
        dwell_cost: float = 1.0,
        max_revisit_window: int = 10,
        drop_timeout: int = 20,
        max_scenarios: Optional[int] = None,
        max_pulses_per_scenario: int = 10000,
        reward_weights: Optional[RewardWeights] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.num_emitters = num_emitters
        self.max_steps = max_steps
        self.initial_budget = initial_budget
        self.dwell_cost = dwell_cost
        self.max_revisit_window = max_revisit_window
        self.drop_timeout = drop_timeout
        self.max_pulses_per_scenario = max_pulses_per_scenario
        self.shuffle = shuffle

        # Dataset reader and tracker
        self.reader = TSRDDatasetReader(root_path=self.dataset_path)
        self.all_files = self.reader.list_files(self.split)
        if max_scenarios is not None:
            self.all_files = self.all_files[:max_scenarios]

        if not self.all_files:
            raise FileNotFoundError(f"No scenario files found in {self.split}")

        self.file_index = 0
        self.tracker = EmitterTracker()
        self.adapter = UpstreamStateAdapter(num_emitters=self.num_emitters)

        self.observation_space = self.adapter.state_builder.observation_space
        self.action_space = spaces.Discrete(self.num_emitters)

        self.reward_fn = EWRewardFunction(weights=reward_weights)
        self.rng = np.random.default_rng()

        # Episode state
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.current_sample: Optional[PulseTrainSample] = None
        self.tracks: List[EmitterTrack] = []
        self.time_windows: np.ndarray = np.array([])
        self.total_detections = 0
        self.total_drops = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Select next scenario file
        if self.shuffle:
            chosen_file = self.rng.choice(self.all_files)
        else:
            chosen_file = self.all_files[self.file_index % len(self.all_files)]
            self.file_index += 1

        self.current_sample = self.reader.load_sample(
            split=self.split,
            index_or_filename=chosen_file,
            max_pulses=self.max_pulses_per_scenario,
        )

        # Deinterleave pulses to extract emitter tracks
        self.tracks = self.tracker.process_pulse_train(self.current_sample.pdws, return_tracks=True)  # type: ignore[assignment]

        # Pad tracks if fewer than num_emitters
        while len(self.tracks) < self.num_emitters:
            idx = len(self.tracks)
            self.tracks.append(
                EmitterTrack(
                    track_id=f"dummy_{idx}",
                    num_pulses=0,
                    mean_freq_mhz=0.0,
                    std_freq_mhz=0.0,
                    mean_pw_us=0.0,
                    std_pw_us=0.0,
                    mean_aoa_deg=0.0,
                    std_aoa_deg=0.0,
                    mean_amplitude_db=-100.0,
                    estimated_pri_us=0.0,
                    pri_type="idle",
                    pri_confidence=0.0,
                    last_intercept_us=0.0,
                    threat_level=0.0,
                    activity_prob=0.0,
                    uncertainty=0.0,
                    identity_confidence=0.0,
                )
            )

        # Truncate to top num_emitters by risk if more
        if len(self.tracks) > self.num_emitters:
            self.tracks.sort(key=lambda t: t.threat_level * (t.activity_prob + t.uncertainty), reverse=True)
            self.tracks = self.tracks[: self.num_emitters]

        # Reset episode counters
        self.current_step = 0
        self.remaining_budget = self.initial_budget
        self.total_detections = 0
        self.total_drops = 0

        # Create time windows across the scenario duration
        total_time_us = float(self.current_sample.pdws[-1, 0] - self.current_sample.pdws[0, 0]) if len(self.current_sample.pdws) > 1 else 1e6
        self.time_windows = np.linspace(0.0, total_time_us, self.max_steps + 1)

        # Track management variables
        self.track_ages = np.zeros(self.num_emitters, dtype=int)
        self.track_confirmed = np.zeros(self.num_emitters, dtype=bool)
        self.track_dropped = np.zeros(self.num_emitters, dtype=bool)

        obs = self._get_observation()
        info = {
            "scenario": self.current_sample.filename,
            "num_tracks": len(self.tracks),
            "step": 0,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        records = [t.to_upstream_record() for t in self.tracks]
        context = UpstreamScanContext(
            current_timestamp=float(self.time_windows[self.current_step] * 1e-6) if len(self.time_windows) > self.current_step else 0.0,
            current_step=self.current_step,
            max_steps=self.max_steps,
            sensor_utilization=float(self.current_step / max(1, self.max_steps)),
            remaining_budget_fraction=float(self.remaining_budget / max(1.0, self.initial_budget)),
        )
        return self.adapter.convert_to_observation(records, context=context)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_int = int(action)
        if not (0 <= action_int < self.num_emitters):
            raise ValueError(f"Action {action_int} out of bounds")

        self.current_step += 1
        target_track = self.tracks[action_int]
        pre_uncertainty = float(target_track.uncertainty)
        time_since_last_visit = float(self.track_ages[action_int])
        is_confirmed = bool(self.track_confirmed[action_int])

        # Deduct dwell cost
        self.remaining_budget = max(0.0, self.remaining_budget - self.dwell_cost)

        # Window boundaries in microseconds
        t_start = self.time_windows[self.current_step - 1]
        t_end = self.time_windows[self.current_step]

        # Check if real pulses occurred for target in this window
        # Target active if pulse arrival is expected and target has pulses
        detected = False
        if target_track.num_pulses > 0 and self.current_sample is not None:
            # Check cluster pulses within window
            window_pulses = (self.current_sample.toa >= t_start) & (self.current_sample.toa <= t_end)
            if np.any(window_pulses):
                detected = bool(self.rng.random() < target_track.activity_prob)

        if detected:
            self.total_detections += 1
            self.track_confirmed[action_int] = True
            self.track_ages[action_int] = 0
            target_track.uncertainty = float(np.clip(target_track.uncertainty * 0.40, 0.01, 1.0))
            target_track.identity_confidence = float(np.clip(target_track.identity_confidence + 0.15, 0.0, 1.0))
        else:
            target_track.uncertainty = float(np.clip(target_track.uncertainty - 0.01, 0.01, 1.0))

        post_uncertainty = float(target_track.uncertainty)

        # Update other unobserved tracks
        dropped_threat_sum = 0.0
        unobserved_active_threats: List[float] = []

        for i, t in enumerate(self.tracks):
            if i != action_int:
                self.track_ages[i] += 1
                t.uncertainty = float(np.clip(t.uncertainty + 0.03, 0.0, 1.0))

                # Check if unobserved track had pulses in this window
                if t.num_pulses > 0 and self.rng.random() < (t.activity_prob * 0.5):
                    unobserved_active_threats.append(float(t.threat_level))

                # Track drop logic
                if self.track_confirmed[i] and not self.track_dropped[i] and self.track_ages[i] > self.drop_timeout:
                    self.track_dropped[i] = True
                    self.total_drops += 1
                    dropped_threat_sum += float(t.threat_level)

        # Multi-objective reward
        step_result = StepObservationResult(
            action=action_int,
            detected=detected,
            threat_level=float(target_track.threat_level),
            pre_uncertainty=pre_uncertainty,
            post_uncertainty=post_uncertainty,
            is_track_confirmed=is_confirmed,
            time_since_last_visit=time_since_last_visit,
            max_revisit_window=float(self.max_revisit_window),
            dropped_threat_sum=dropped_threat_sum,
            dwell_cost=self.dwell_cost,
            initial_budget=self.initial_budget,
            unobserved_active_threats=unobserved_active_threats,
        )
        telemetry = self.reward_fn.compute(step_result)

        terminated = self.remaining_budget <= 0.0
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "detected": detected,
            "total_detections": self.total_detections,
            "total_drops": self.total_drops,
            "telemetry": telemetry.to_dict(),
        }

        return obs, telemetry.total_reward, terminated, truncated, info
