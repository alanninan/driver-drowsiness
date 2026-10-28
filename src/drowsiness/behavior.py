"""Causal full-rate features shared by training and streaming inference.

No label, filename, subject, brightness or time-in-recording is a feature.
Calibration is an explicitly supplied known-alert prefix, never a whole test clip.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

RATE = 25.0
CONTEXTS = (30.0, 90.0)
CALIBRATION_SECONDS = 180.0


def frame_array(frames):
    """Columns: time, left EAR, right EAR, MAR, pitch, yaw, roll, valid."""
    return np.asarray(
        [
            [f.timestamp, f.ear_left, f.ear_right, f.mar, f.pitch, f.yaw, f.roll, float(f.valid)]
            for f in frames
        ],
        dtype=np.float64,
    ).reshape(-1, 8)


def summarize(values):
    values = np.asarray(values)
    if not len(values):
        return np.zeros(8)
    return np.r_[
        np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]),
        values.mean(),
        values.std(),
        np.mean(np.abs(np.diff(values))) if len(values) > 1 else 0.0,
    ]


def regular_runs(data):
    """Interpolate within valid runs only, never across tracking gaps."""
    if not len(data):
        return []
    valid = (data[:, 7] > 0) & np.isfinite(data).all(axis=1)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    cuts = np.flatnonzero((np.diff(ids) > 1) | (np.diff(data[ids, 0]) > 0.15)) + 1
    runs = []
    for group in np.split(ids, cuts):
        if len(group) < 4 or data[group[-1], 0] - data[group[0], 0] < 0.12:
            continue
        sample = data[group]
        t = np.arange(sample[0, 0], sample[-1, 0] + 1e-8, 1 / RATE)
        runs.append(
            np.column_stack([t] + [np.interp(t, sample[:, 0], sample[:, c]) for c in range(1, 7)])
        )
    return runs


def closed_durations(closed):
    edges = np.diff(np.r_[False, closed, False].astype(int))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / RATE


def feature_block(data, eye_reference=None):
    """One trailing context; no observations outside data are consulted."""
    names, values = [], []

    def add(name, value):
        names.append(name)
        values.append(float(value))

    def stats(name, value):
        for suffix, number in zip(
            ("q05", "q25", "median", "q75", "q95", "mean", "std", "motion"),
            summarize(value),
            strict=True,
        ):
            add(name + "_" + suffix, number)

    runs = regular_runs(data)
    sample = np.concatenate(runs) if runs else np.empty((0, 7))
    eyes = sample[:, 1:3].mean(axis=1)
    open_eye = max(float(np.quantile(eyes, 0.9)) if len(eyes) else 0.3, 0.08)
    reference = max(eye_reference if eye_reference is not None else open_eye, 0.08)
    stats("eye", eyes)
    stats("relative_eye", eyes / reference)
    stats("asymmetry", np.abs(sample[:, 1] - sample[:, 2]))
    for column, name in [(3, "mouth"), (4, "pitch"), (5, "yaw"), (6, "roll")]:
        stats(name, sample[:, column])
    observed = max(len(sample) / RATE, 1 / RATE)
    for fraction in (0.6, 0.7, 0.8, 0.9):
        durations = []
        for run in runs:
            durations.extend(closed_durations(run[:, 1:3].mean(axis=1) < reference * fraction))
        durations = np.asarray(durations)
        key = f"closure_{fraction:.1f}"
        add(key + "_fraction", durations.sum() / observed)
        add(key + "_slow_fraction", durations[durations >= 0.5].sum() / observed)
        add(key + "_longest", durations.max() if len(durations) else 0)
        add(key + "_rate", len(durations) * 60 / observed)
    durations, amplitudes, reopening, closing, intervals = [], [], [], [], []
    for run in runs:
        ear = run[:, 1:3].mean(axis=1)
        if len(ear) < 7:
            continue
        smoothed = savgol_filter(ear, 5, 2)
        peaks, props = find_peaks(
            -smoothed, prominence=max(0.015, open_eye * 0.08), distance=max(1, int(0.12 * RATE))
        )
        if not len(peaks):
            continue
        widths, _, left, right = peak_widths(-smoothed, peaks, rel_height=0.8)
        keep = (widths / RATE >= 0.08) & (widths / RATE <= 3)
        peaks, widths, left, right = peaks[keep], widths[keep], left[keep], right[keep]
        prominence = props["prominences"][keep]
        durations.extend(widths / RATE)
        amplitudes.extend(prominence / reference)
        reopening.extend(prominence / reference / np.maximum((right - peaks) / RATE, 0.04))
        closing.extend(prominence / reference / np.maximum((peaks - left) / RATE, 0.04))
        intervals.extend(np.diff(peaks) / RATE)
    add("blink_rate", len(durations) * 60 / observed)
    for name, vector in [
        ("blink_duration", durations),
        ("blink_amplitude", amplitudes),
        ("reopening_speed", reopening),
        ("closing_speed", closing),
        ("blink_interval", intervals),
    ]:
        stats(name, vector)
    for threshold in (0.2, 0.35, 0.5):
        add(f"mouth_above_{threshold}", np.mean(sample[:, 3] > threshold) if len(sample) else 0)
    return np.nan_to_num(np.asarray(values, dtype=np.float32)), names


@dataclass
class AlertReference:
    eye: float
    center: np.ndarray
    scale: np.ndarray
    observed_seconds: float

    def to_dict(self):
        return dict(
            eye=self.eye,
            center=self.center.tolist(),
            scale=self.scale.tolist(),
            observed_seconds=self.observed_seconds,
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["eye"],
            np.asarray(value["center"]),
            np.asarray(value["scale"]),
            value["observed_seconds"],
        )


def calibrate_reference(prefix):
    """Requires 120 usable seconds in a 180-second explicit alert enrolment."""
    runs = regular_runs(prefix)
    observed = sum(len(run) / RATE for run in runs)
    if observed < 120:
        raise ValueError(f"Need 120 usable alert seconds; received {observed:.1f}")
    sample = np.concatenate(runs)
    eye = float(np.quantile(sample[:, 1:3].mean(axis=1), 0.9))
    blocks = []
    for start in np.arange(prefix[0, 0], prefix[-1, 0] - 29, 30):
        clip = prefix[(prefix[:, 0] >= start) & (prefix[:, 0] < start + 30)]
        blocks.append(feature_block(clip, eye)[0])
    blocks = np.stack(blocks)
    center = np.median(blocks, axis=0)
    scale = np.maximum(blocks.std(axis=0), np.maximum(np.abs(center) * 0.25, 0.05))
    return AlertReference(eye, center, scale, observed)


def extract_behavior(data, end, reference=None, history_start=0.0):
    """Fixed-dimensional trailing features, also safe on a live buffer."""
    result, names = [], []
    for seconds in CONTEXTS:
        clip = data[(data[:, 0] >= max(history_start, end - seconds)) & (data[:, 0] < end)]
        block, keys = feature_block(clip, reference.eye if reference else None)
        result.extend(block)
        names.extend(f"{int(seconds)}s_{key}" for key in keys)
        if reference is not None:
            result.extend(np.clip((block - reference.center) / reference.scale, -20, 20))
            names.extend(f"{int(seconds)}s_personal_{key}" for key in keys)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite behavioural features")
    return result, names
