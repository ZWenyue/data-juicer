import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields

OP_NAME = "robot_key_frame_detector_mapper"


@OPERATORS.register_module(OP_NAME)
class RobotKeyFrameDetectorMapper(Mapper):
    """Detect task-critical keyframes in a robot episode.

    Identifies gripper closure/opening events (large delta in gripper signal)
    and decisive state transitions (velocity norm peaks above a percentile
    threshold).  Each detected keyframe is expanded by a protection window.
    Writes a report with protected frame indices into Fields.meta.
    """

    def __init__(
        self,
        signal_source: str = "top_level",
        states_key: str = "states",
        actions_key: str = "actions",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        gripper_dims: list = None,
        gripper_field: str = None,
        gripper_close_direction: str = "decrease",
        gripper_delta_threshold: float = 5.0,
        gripper_delta_threshold_frac: float = None,
        state_velocity_percentile: float = 95.0,
        keyframe_window: int = 3,
        exempt_dims: list = None,
        min_frames: int = 4,
        report_field: str = "key_frame_report",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.signal_source = signal_source
        self.states_key = states_key
        self.actions_key = actions_key
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.gripper_dims = gripper_dims
        self.gripper_field = gripper_field
        self.gripper_close_direction = gripper_close_direction
        self.gripper_delta_threshold = gripper_delta_threshold
        self.gripper_delta_threshold_frac = gripper_delta_threshold_frac
        self.state_velocity_percentile = state_velocity_percentile
        self.keyframe_window = keyframe_window
        self.exempt_dims = exempt_dims or []
        self.min_frames = min_frames
        self.report_field = report_field

    def _load_states(self, sample):
        if self.signal_source == "top_level":
            raw = sample.get(self.top_level_state_key)
            if raw is None:
                return None
            return np.array(raw, dtype=np.float64)
        return None

    def _load_gripper_signal(self, sample, states):
        if self.gripper_field is not None:
            raw = sample.get(self.gripper_field)
            if raw is not None:
                return np.array(raw, dtype=np.float64)
        if self.gripper_dims is not None and states is not None:
            dims = [d for d in self.gripper_dims if 0 <= d < states.shape[1]]
            if dims:
                return states[:, dims]
        return None

    def _detect_gripper_events(self, gripper_signal):
        if gripper_signal is None or len(gripper_signal) < 2:
            return []
        signal = np.asarray(gripper_signal, dtype=np.float64)
        if signal.ndim == 1:
            signal = signal[:, None]

        events = set()
        for d in range(signal.shape[1]):
            values = signal[:, d]
            delta = np.diff(values)
            threshold = float(self.gripper_delta_threshold)
            if self.gripper_delta_threshold_frac is not None:
                q01, q99 = np.percentile(values, [1, 99])
                range_threshold = float(self.gripper_delta_threshold_frac) * float(q99 - q01)
                abs_delta = np.abs(delta)
                med = float(np.median(abs_delta))
                mad = float(np.median(np.abs(abs_delta - med)))
                noise_threshold = med + 6.0 * 1.4826 * mad
                threshold = max(range_threshold, noise_threshold, np.finfo(float).eps)
            if self.gripper_close_direction == "decrease":
                dim_events = np.where(delta < -threshold)[0]
            else:
                dim_events = np.where(delta > threshold)[0]
            events.update(dim_events.tolist())
        return sorted(events)

    def _detect_velocity_peaks(self, states):
        if states is None or len(states) < 2:
            return []
        check_dims = [
            d for d in range(states.shape[1]) if d not in self.exempt_dims
        ]
        if not check_dims:
            return []
        velocity = np.linalg.norm(np.diff(states[:, check_dims], axis=0), axis=1)
        if len(velocity) == 0:
            return []
        threshold = np.percentile(velocity, self.state_velocity_percentile)
        if threshold <= 0:
            return []
        peaks = np.where(velocity > threshold)[0]
        return peaks.tolist()

    def _expand_protection_window(self, keyframe_indices, total_frames):
        protected = set()
        for k in keyframe_indices:
            for offset in range(
                -self.keyframe_window, self.keyframe_window + 1
            ):
                idx = k + offset
                if 0 <= idx < total_frames:
                    protected.add(idx)
        return sorted(protected)

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})
        states = self._load_states(sample)

        if states is None or len(states) < self.min_frames:
            report = {
                "num_gripper_events": 0,
                "num_velocity_peaks": 0,
                "gripper_event_frames": [],
                "velocity_peak_frames": [],
                "protected_frame_indices": [],
                "skipped": True,
                "reason": "insufficient_frames",
            }
            meta[self.report_field] = json.dumps(report, ensure_ascii=False)
            return sample

        T = len(states)
        gripper_signal = self._load_gripper_signal(sample, states)
        gripper_events = self._detect_gripper_events(gripper_signal)
        velocity_peaks = self._detect_velocity_peaks(states)

        all_keyframes = sorted(set(gripper_events + velocity_peaks))
        protected = self._expand_protection_window(all_keyframes, T)

        report = {
            "num_gripper_events": len(gripper_events),
            "num_velocity_peaks": len(velocity_peaks),
            "gripper_event_frames": gripper_events,
            "velocity_peak_frames": velocity_peaks,
            "protected_frame_indices": protected,
        }
        meta[self.report_field] = json.dumps(report, ensure_ascii=False)
        return sample
