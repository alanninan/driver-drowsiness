import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recover_small_face import recover_frame

from drowsiness.core import Frame


def fixture_frame():
    frame = Frame(0, .3, .3, .1, 0, 0, 0, 100, False, "Move closer to the camera")
    overlay = [[.3, .4]] * 18
    overlay[3] = [.34, .4]
    overlay[9] = [.34, .4]
    return frame, overlay


def test_small_face_with_sufficient_native_eye_detail_can_recover():
    frame, overlay = fixture_frame()
    result = recover_frame(frame, overlay, 427, 240)
    assert result.valid and result.reason == ""
    assert result.ear_left == frame.ear_left
    assert not recover_frame(frame, overlay, 240, 135).valid


@pytest.mark.parametrize("changes", [
    {"reason": "No face detected"}, {"brightness": 24}, {"yaw": 36},
    {"pitch": 41}, {"roll": 36}, {"ear_left": .9}, {"mar": 3}, {"yaw": float("nan")},
])
def test_recovery_preserves_other_quality_gates(changes):
    frame, overlay = fixture_frame()
    assert not recover_frame(replace(frame, **changes), overlay, 427, 240).valid


def test_clipped_or_nonfinite_landmarks_are_not_promoted():
    frame, overlay = fixture_frame()
    for point in ([-.01, .4], [1.01, .4], [float("nan"), .4]):
        bad = list(overlay)
        bad[17] = point
        assert not recover_frame(frame, bad, 427, 240).valid
