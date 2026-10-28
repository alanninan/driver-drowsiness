"""Exercise extraction, cache reuse, replay and live WS on a public sample still.

This verifies software plumbing only. It is not an accuracy or fatigue evaluation.
Run vision_smoke.py first to obtain the sample portrait.
"""

import csv
import json
import struct
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from drowsiness.core import Monitor
from drowsiness.dataset import extract, load_frames
from drowsiness.server import create_app


def main():
    output = Path("outputs/integration")
    output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread("outputs/vision/portrait.jpg")
    assert image is not None, "Run scripts/vision_smoke.py first"
    height = round(image.shape[0] * 480 / image.shape[1] / 2) * 2
    image = cv2.resize(image, (480, height))
    video = output / "portrait.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (480, height))
    assert writer.isOpened()
    for _ in range(570):
        writer.write(image)
    writer.release()
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "video_path", "label"])
        writer.writerow(["sample", str(video.resolve()), 0])
    index = extract(manifest, output / "features")
    feature = output / "features" / json.loads(index.read_text())["videos"][0]["feature_file"]
    mtime = feature.stat().st_mtime_ns
    extract(manifest, output / "features")
    assert feature.stat().st_mtime_ns == mtime
    monitor = Monitor()
    count = 0
    for frame in load_frames(feature):
        result = monitor.update(frame)
        count += 1
    assert monitor.baseline is not None and monitor.baseline.personalised
    assert result["state"] == "alert", result
    _, encoded = cv2.imencode(".jpg", image)
    states = set()
    with TestClient(create_app(classifier=None)) as client:
        with client.websocket_connect("/ws/live", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"calibration_seconds": 30})
            assert ws.receive_json()["ready"]
            for index in range(170):
                ws.send_bytes(struct.pack("<d", index / 4) + encoded.tobytes())
                data = ws.receive_json()
                assert "error" not in data, data
                states.add(data["state"])
            assert data["state"] == "alert", data
    report = {
        "passed": True,
        "frames_extracted": count,
        "cache_reused": True,
        "personal_replay_final_state": result["state"],
        "live_websocket_states": sorted(states),
        "note": "Repeated sample still; no drowsiness accuracy claim",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
