import numpy as np
import pytest

from drowsiness.behavior import (
    AlertReference,
    calibrate_reference,
    extract_behavior,
    feature_block,
    frame_array,
    regular_runs,
)
from drowsiness.core import Frame
from drowsiness.expressions import expression_features, expression_reference
from drowsiness.learned import LearnedMonitor


def synthetic(seconds=210):
    t = np.arange(0, seconds, 0.04)
    # Smooth, isolated blinks with known duration, amplitude and separation.
    ear = 0.3 - 0.2 * np.exp(-(((t % 4 - 2) / 0.12) ** 2))
    return frame_array(
        Frame(float(a), float(b), float(b), 0.03, brightness=100)
        for a, b in zip(t, ear, strict=True)
    )


def test_future_data_and_enrolment_history_cannot_affect_prediction():
    data = synthetic()
    reference = calibrate_reference(data[data[:, 0] < 180])
    before, names = extract_behavior(data, 200, reference, history_start=180)
    changed = data.copy()
    changed[(changed[:, 0] >= 200) | (changed[:, 0] < 180), 1:7] = 1000
    after, other_names = extract_behavior(changed, 200, reference, history_start=180)
    np.testing.assert_array_equal(before, after)
    assert names == other_names and len(names) == len(set(names))


def test_tracking_gap_is_not_interpolated_into_a_closure():
    data = synthetic(10)
    gap = (data[:, 0] > 3) & (data[:, 0] < 7)
    data[gap, 7] = 0
    data[gap, 1:3] = 0
    runs = regular_runs(data)
    assert len(runs) == 2
    assert all(not np.any((r[:, 0] > 3.1) & (r[:, 0] < 6.9)) for r in runs)
    x, names = feature_block(data)
    assert x[names.index("closure_0.6_longest")] < 0.5


def test_blink_shape_and_rate_survive_feature_extraction():
    x, names = feature_block(synthetic(60))
    assert x[names.index("blink_rate")] == pytest.approx(15, abs=1)
    assert 0.1 < x[names.index("blink_duration_median")] < 0.5
    assert x[names.index("reopening_speed_median")] > 1


def test_calibration_requires_usable_time_and_round_trips():
    with pytest.raises(ValueError, match="120 usable"):
        calibrate_reference(synthetic(60))
    reference = calibrate_reference(synthetic(180))
    restored = AlertReference.from_dict(reference.to_dict())
    assert restored.observed_seconds > 179
    np.testing.assert_array_equal(restored.center, reference.center)
    np.testing.assert_array_equal(restored.scale, reference.scale)
    assert np.all(restored.scale > 0)


def test_empty_tracking_has_finite_fixed_dimension():
    empty, names = feature_block(np.empty((0, 8)))
    full, other = feature_block(synthetic(30))
    assert len(empty) == len(full) and names == other
    assert np.isfinite(empty).all()


def test_streaming_features_match_offline_and_tracking_loss_hides_prediction():
    class RecordingModel:
        seen = None

        def predict(self, x):
            self.seen = x.copy()
            return np.array([1])

    frames = [
        Frame(
            i / 25,
            0.3 - 0.15 * np.exp(-(((i / 25 % 4 - 2) / 0.12) ** 2)),
            0.3 - 0.15 * np.exp(-(((i / 25 % 4 - 2) / 0.12) ** 2)),
            0.03,
            brightness=100,
        )
        for i in range(5526)
    ]
    data = frame_array(frames)
    reference = calibrate_reference(data[data[:, 0] < 180])
    expected, names = extract_behavior(data, 221, reference, 180)
    model = RecordingModel()
    monitor = LearnedMonitor(model, dict(variant="enrolled", feature_names=names))
    for frame in frames:
        result = monitor.update(frame)
    assert result["model_label"] == 1 and result["score"] == 50
    np.testing.assert_allclose(model.seen[0], expected, atol=1e-5)
    result = monitor.update(Frame(222, valid=False, reason="No face"))
    assert result["state"] == "unknown" and result["score"] is None


def test_expression_features_ignore_future_and_calibration_history():
    time = np.arange(0, 240, 0.5)
    data = np.column_stack(
        [time, np.ones(len(time)), 0.1 + 0.05 * np.sin(time), 0.2 + 0.02 * np.cos(time)]
    )
    reference = expression_reference(data[data[:, 0] < 180])
    expected, names = expression_features(data, 221, reference, ["eye", "mouth"], 180)
    changed = data.copy()
    changed[(time < 180) | (time >= 221), 2:] = 100
    actual, keys = expression_features(changed, 221, reference, ["eye", "mouth"], 180)
    np.testing.assert_array_equal(actual, expected)
    assert names == keys and np.isfinite(actual).all()


def test_expression_streaming_matches_the_offline_hybrid_features():
    class RecordingModel:
        seen = None

        def predict(self, x):
            self.seen = x.copy()
            return np.array([0])

    frames = [Frame(i / 30, 0.3, 0.3, 0.03, brightness=100) for i in range(6631)]
    expression = [
        [f.timestamp, 1, 0.1 + 0.05 * np.sin(f.timestamp), 0.2]
        for i, f in enumerate(frames)
        if i % 15 == 0
    ]
    data = frame_array(frames)
    expression = np.asarray(expression)
    reference = calibrate_reference(data[data[:, 0] < 180])
    eye_x, names = extract_behavior(data, 221, reference, 180)
    expr_reference = expression_reference(expression[expression[:, 0] < 180])
    expr_x, expr_names = expression_features(expression, 221, expr_reference, ["eye", "mouth"], 180)
    model = RecordingModel()
    monitor = LearnedMonitor(
        model,
        dict(
            variant="enrolled",
            feature_names=names,
            expression_names=["eye", "mouth"],
            expression_feature_names=expr_names,
        ),
    )
    for frame in frames:
        result = monitor.update(frame, [0.1 + 0.05 * np.sin(frame.timestamp), 0.2])
    assert result["model_label"] == 0
    np.testing.assert_allclose(model.seen[0], np.r_[eye_x, expr_x], atol=1e-5)
