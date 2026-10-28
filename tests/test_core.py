from dataclasses import replace

import numpy as np
import pytest

from drowsiness.core import Baseline, Calibrator, Config, Frame, Monitor


def frame(t, ear=0.30, **kwargs):
    return Frame(float(t), ear, ear, mar=0.05, brightness=120, **kwargs)


def advance(monitor, start, end, ear=0.30, fps=20):
    result = None
    for t in np.arange(start, end, 1 / fps):
        result = monitor.update(frame(t, ear))
    return result


def test_calibration_counts_valid_time_not_wall_time():
    cal = Calibrator(seconds=2)
    for t in np.arange(0, 1, 0.05):
        cal.update(frame(t))
    cal.update(frame(1, valid=False, reason="no face"))
    cal.update(frame(100))
    assert cal.elapsed < 1
    with pytest.raises(ValueError, match="more usable"):
        cal.finish()
    for t in np.arange(100.05, 102, 0.05):
        cal.update(frame(t))
    assert cal.finish().ear_left == pytest.approx(0.30)


def test_natural_blinks_do_not_count_as_slow_closure():
    monitor = Monitor(baseline=Baseline())
    for t in np.arange(0, 30, 0.05):
        result = monitor.update(frame(t, 0.10 if 1 < t % 4 < 1.2 else 0.30))
    assert result["state"] == "alert"
    assert result["closure_ratio"] == 0
    assert result["blink_seconds"] > 0.08


def test_sustained_closure_overrides_warmup():
    monitor = Monitor(baseline=Baseline())
    result = advance(monitor, 0, 2, ear=0.10)
    assert result["state"] == "critical"
    assert result["score"] >= 85


def test_missing_face_breaks_closure_and_never_implies_alert():
    monitor = Monitor(baseline=Baseline())
    advance(monitor, 0, 1, ear=0.10)
    missing = monitor.update(frame(1, valid=False, reason="No face"))
    assert missing["state"] == "unknown" and missing["score"] is None
    result = monitor.update(frame(20, 0.10))
    assert result["closure_seconds"] == 0
    assert result["state"] == "unknown"
    assert result["coverage"] < 0.1


def test_wink_is_not_bilateral_closure():
    monitor = Monitor(baseline=Baseline())
    for t in np.arange(0, 12, 0.05):
        result = monitor.update(replace(frame(t), ear_left=0.10))
    assert result["state"] == "alert"
    assert result["closure_ratio"] == 0


def test_personal_baseline_changes_threshold():
    fixed = Monitor(baseline=Baseline())
    personal = Monitor(baseline=Baseline(ear_left=0.23, ear_right=0.23, personalised=True))
    for t in np.arange(0, 12, 0.05):
        a = fixed.update(frame(t, 0.18))
        b = personal.update(frame(t, 0.18))
    assert a["state"] == "critical"
    assert b["state"] == "alert"


def test_dropped_frames_cannot_accumulate_a_continuous_closure():
    monitor = Monitor(baseline=Baseline())
    for t in range(20):
        result = monitor.update(frame(t, 0.1))
        assert result["closure_seconds"] == 0
        assert result["score"] is None


def test_time_weighting_is_stable_across_frame_rates():
    outputs = []
    for fps in (10, 20, 30):
        monitor = Monitor(baseline=Baseline())
        for t in np.arange(0, 40, 1 / fps):
            result = monitor.update(frame(t, 0.1 if 20 <= t < 21 else 0.3))
        outputs.append(result["closure_ratio"])
    assert max(outputs) - min(outputs) < 0.003


def test_window_expires_old_closures():
    monitor = Monitor(Config(window_seconds=15), baseline=Baseline())
    advance(monitor, 0, 2, 0.1)
    result = advance(monitor, 2, 40)
    assert result["closure_ratio"] == 0
    assert result["state"] == "alert"


def test_timestamp_and_feature_validation():
    monitor = Monitor(baseline=Baseline())
    monitor.update(frame(1))
    with pytest.raises(ValueError, match="increase"):
        monitor.update(frame(1))
    with pytest.raises(ValueError, match="finite"):
        Monitor().update(frame(float("nan")))
    with pytest.raises(ValueError, match="finite"):
        Monitor().update(frame(0, float("inf")))


def test_yawn_counted_once_during_sustained_open_mouth():
    monitor = Monitor(baseline=Baseline())
    for t in np.arange(0, 15, 0.05):
        result = monitor.update(replace(frame(t), mar=0.7))
    assert len(monitor.yawns) == 1
    assert result["yawns_per_minute"] == pytest.approx(4, abs=0.1)


def test_invalid_frames_abort_nod_candidate():
    monitor = Monitor(baseline=Baseline())
    monitor.update(replace(frame(0), pitch=20))
    monitor.update(replace(frame(0.2), pitch=20))
    monitor.update(frame(0.3, valid=False))
    monitor.update(frame(0.5))
    assert not monitor.nods
