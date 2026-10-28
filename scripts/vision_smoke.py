"""Verify the real model on Google's public sample portrait, not a drowsiness benchmark."""

import json
import time
import urllib.request
from pathlib import Path

import cv2

from drowsiness.vision import Tracker


def main():
    output = Path("outputs/vision")
    output.mkdir(parents=True, exist_ok=True)
    portrait = output / "portrait.jpg"
    if not portrait.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-assets/portrait.jpg", portrait
        )
    image = cv2.imread(str(portrait))
    image = cv2.resize(image, (480, round(image.shape[0] * 480 / image.shape[1])))
    times, frames = [], []
    with Tracker() as tracker:
        for index in range(40):
            started = time.perf_counter()
            frame, overlay = tracker.process(image, index / 30)
            times.append((time.perf_counter() - started) * 1000)
            frames.append(frame)
    assert frames[-1].valid, frames[-1]
    assert 0.1 < frames[-1].ear < 0.6 and len(overlay) == 18
    report = {
        "source": "Google MediaPipe sample portrait",
        "frames": len(frames),
        "valid_frames": sum(f.valid for f in frames),
        "mean_inference_ms_excluding_first": sum(times[1:]) / len(times[1:]),
        "ear": frames[-1].ear,
        "mar": frames[-1].mar,
        "pitch": frames[-1].pitch,
        "note": "Repeated still image verifies inference, not drowsiness accuracy",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
