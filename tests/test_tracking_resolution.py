"""Resolution control preserves real image detail and never upsamples."""

from types import SimpleNamespace

import numpy as np
import pytest

from drowsiness.vision import Tracker


@pytest.mark.parametrize(
    "width,limit,expected",
    [(1280, 960, (540, 960)), (1280, 1280, (720, 1280)), (640, 1280, (720, 640))],
)
def test_tracker_resolution(width, limit, expected):
    tracker = Tracker.__new__(Tracker)
    tracker.max_width, tracker.last_ms, tracker.clahe = limit, -1, None
    tracker.mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB=1), Image=lambda **kwargs: kwargs["data"]
    )
    seen = []

    def detect(image, timestamp):
        seen.append(image.shape[:2])
        return SimpleNamespace(face_landmarks=[])

    tracker.detector = SimpleNamespace(detect_for_video=detect)
    frame, _ = tracker.process(np.zeros((720, width, 3), dtype=np.uint8), 0)
    assert seen == [expected]
    assert not frame.valid
    assert frame.reason == "No face detected"
