import struct
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from drowsiness.behavior import calibrate_reference, frame_array
from drowsiness.core import Baseline, Frame, Monitor
from drowsiness.learned import LearnedMonitor
from drowsiness.server import create_app


def test_learned_resume_preserves_reference_and_clears_events():
    monitor = LearnedMonitor(None, {"variant": "enrolled"})
    prefix = frame_array(Frame(i / 25, .3, .3, .03) for i in range(4500))
    reference = calibrate_reference(prefix)
    monitor.reference = reference
    baseline = Baseline(personalised=True)
    monitor.monitor = Monitor(baseline=baseline)
    monitor.buffer.append(Frame(200, .1, .1))
    monitor.last_prediction = 2
    monitor.ready_at = 180
    monitor.previous = 200
    monitor.monitor.closed_since = 199
    monitor.resume()
    assert monitor.reference is reference
    assert monitor.monitor.baseline is baseline
    assert not monitor.buffer and monitor.last_prediction is None
    result = monitor.update(Frame(1000, .3, .3, .03))
    assert result["state"] == "warming" and result["calibration"] == 1
    assert result["score"] is None and result["closure_seconds"] == 0


def test_websocket_pause_survives_idle_and_retains_baseline(monkeypatch):
    class Tracker:
        def __init__(self, *_):
            self.last_expression = None

        def process(self, image, timestamp):
            return Frame(timestamp, .3, .3, .03, brightness=100), []

        def close(self):
            pass

    monkeypatch.setattr("drowsiness.server.Tracker", Tracker)
    _, jpeg = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    def packet(t):
        return struct.pack("<d", t) + jpeg.tobytes()

    with TestClient(create_app(classifier=None)) as client:
        with client.websocket_connect("/ws/live", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"calibration_seconds": 30})
            assert ws.receive_json()["ready"]
            ws.send_json({"action": "pause"})
            assert "pause_error" in ws.receive_json()
            for i in range(312):
                ws.send_bytes(packet(i / 10))
                result = ws.receive_json()
            baseline = result["metadata"]["baseline"]
            assert result["calibration"] == 1
            ws.send_json({"action": "pause"})
            assert ws.receive_json() == {"paused": True}
            time.sleep(10.1)  # Longer than the normal camera frame timeout.
            ws.send_bytes(packet(100))  # In-flight frames must be discarded while paused.
            ws.send_json({"action": "resume"})
            assert ws.receive_json() == {"paused": False}
            ws.send_bytes(packet(101))
            result = ws.receive_json()
            assert result["metadata"]["baseline"] == baseline
            assert result["calibration"] == 1 and result["state"] in {"unknown", "warming"}
            assert result["score"] is None and result["closure_seconds"] == 0
