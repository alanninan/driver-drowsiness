"""Causal summaries of MediaPipe facial-expression coefficients."""

import numpy as np


def expression_reference(prefix):
    valid = prefix[(prefix[:, 1] > 0) & np.isfinite(prefix).all(1), 2:]
    if len(valid) < 240:
        raise ValueError("Need 120 seconds of valid 2-Hz expression calibration")
    return np.median(valid, axis=0), np.maximum(valid.std(0), 0.03)


def expression_features(data, end, reference, names, history_start=0):
    center, scale = reference
    features, keys = [], []
    for seconds in (30, 90):
        sample = data[
            (data[:, 0] >= max(history_start, end - seconds))
            & (data[:, 0] < end)
            & (data[:, 1] > 0),
            2:,
        ]
        sample = np.clip((sample - center) / scale, -20, 20)
        blocks = (
            [
                sample.mean(0),
                sample.std(0),
                np.quantile(sample, 0.1, axis=0),
                np.quantile(sample, 0.9, axis=0),
            ]
            if len(sample)
            else [np.zeros(len(names))] * 4
        )
        for suffix, block in zip(("mean", "std", "q10", "q90"), blocks, strict=True):
            features.extend(block)
            keys.extend(f"{seconds}s_expression_{name}_{suffix}" for name in names)
    return np.asarray(features, dtype=np.float32), keys
