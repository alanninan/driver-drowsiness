"""Calibration-triggered recovery using native eye detail instead of face width.

Only the redundant 90px whole-face gate may be bypassed. Every remaining quality
gate is reapplied. Images are not upscaled and missing measurements are not filled.
"""
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
from pathlib import Path

import cv2
import numpy as np
from build_behavior_cohort import build, sha

from drowsiness.behavior import calibrate_reference, frame_array
from drowsiness.core import Frame
from drowsiness.dataset import load_frames, video_frames, write_json
from drowsiness.vision import Tracker

ROOT = Path("outputs/folds34-run")
RECOVERY = ROOT / "native-eye-recovery"


def recover_frame(frame, overlay, width, height):
    if frame.reason != "Move closer to the camera" or len(overlay) != 18:
        return frame
    points = np.asarray(overlay, dtype=float) * [width, height]
    if not np.isfinite(points).all() or not np.isfinite([
        frame.ear_left, frame.ear_right, frame.mar, frame.pitch,
        frame.yaw, frame.roll, frame.brightness,
    ]).all():
        return frame
    # Overlay rounding is 0.0001 normalized units. Require a conservative margin
    # above the original 14px threshold, so rounding cannot promote a small eye.
    margin = np.hypot(width, height) * 0.0001
    eye_pixels = min(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[6] - points[9]))
    if eye_pixels < 14 + margin:
        return frame
    if frame.brightness < 25 or abs(frame.yaw) > 35 or abs(frame.pitch) > 40 or abs(frame.roll) > 35:
        return frame
    if np.any(points <= margin) or np.any(points >= np.array([width, height]) - margin):
        return frame
    if not (0 <= frame.ear_left <= .8 and 0 <= frame.ear_right <= .8 and 0 <= frame.mar <= 2):
        return frame
    return replace(frame, valid=True, reason="")


def extract_video(video):
    cv2.setNumThreads(1)
    key = hashlib.sha256(video["video_path"].encode()).hexdigest()[:16]
    target = RECOVERY / f"{key}.csv"
    receipt = target.with_suffix(".json")
    source = Path(video["video_path"])
    inputs = dict(source=str(source), source_bytes=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns,
        code={str(p): sha(p) for p in [Path(__file__), Path("src/drowsiness/vision.py"),
                                     Path("src/drowsiness/dataset.py")]},
        model_sha256=sha(Path("models/face_landmarker.task")), max_width=1280,
        policy="native-eye-resolution-v1", sample_fps=30)
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["inputs"] != inputs or saved["sha256"] != sha(target):
            raise ValueError("Recovery cache mismatch")
    else:
        counts = Counter()
        temporary = target.with_suffix(".partial")
        with Tracker(max_width=1280) as tracker, temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(Frame)])
            writer.writeheader()
            for image, timestamp in video_frames(source, 30):
                frame, overlay = tracker.process(image, timestamp)
                h, w = image.shape[:2]
                if w > 1280:
                    h, w = round(h * 1280 / w), 1280
                recovered = recover_frame(frame, overlay, w, h)
                counts["recovered_frames"] += int(recovered.valid and not frame.valid)
                counts["total_frames"] += 1
                counts["valid_frames"] += int(recovered.valid)
                writer.writerow(asdict(recovered))
        temporary.replace(target)
        write_json(receipt, dict(inputs=inputs, sha256=sha(target), counts=dict(counts)))
    print(f"Recovered {video['subject_id']}/{video['label']}", flush=True)
    return dict(video, feature_file=str(target.resolve()))


def run():
    started = time.monotonic()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    original = ROOT / "features-index.json"
    index = json.loads(original.read_text())
    if {v["subject_id"] for v in index["videos"]} != {f"{n:02}" for n in range(25, 49)}:
        raise ValueError("Only development Folds3/4 are allowed")
    failures = {}
    for video in index["videos"]:
        if video["label"] == 0:
            data = frame_array(load_frames(video["feature_file"]))
            try:
                calibrate_reference(data[data[:, 0] < 180])
            except ValueError as error:
                failures[video["subject_id"]] = str(error)
    write_json(RECOVERY / "protocol.json", dict(failed_references=failures,
        selection="Known-alert calibration failure only; no classification scores consulted",
        policy="Replace whole-face width gate with original native-eye-detail gate and rounding margin",
        calibration_seconds=180, minimum_usable_seconds=120,
        original_index_sha256=sha(original), affected_subjects=sorted(failures)))
    write_json(ROOT / "status.json", {"phase": "native_eye_resolution_recovery", "subjects": sorted(failures)})
    selected = [v for v in index["videos"] if v["subject_id"] in failures]
    with ProcessPoolExecutor(max_workers=3) as pool:
        replacements = {v["video_path"]: v for v in pool.map(extract_video, selected)}
    merged = [replacements.get(v["video_path"], v) for v in index["videos"]]
    references = {}
    for v in merged:
        if v["label"] == 0:
            data = frame_array(load_frames(v["feature_file"]))
            references[v["subject_id"]] = calibrate_reference(data[data[:, 0] < 180]).observed_seconds
    write_json(RECOVERY / "calibration-check.json", references)
    final_index = RECOVERY / "index.json"
    write_json(final_index, dict(settings=dict(index["settings"], native_eye_recovery=sorted(failures)), videos=merged))
    write_json(ROOT / "status.json", {"phase": "behavior_features", "recovered_calibration": references})
    output = Path("outputs/folds34-behavior")
    build(final_index, output)
    write_json(output / "preprocessing.json", dict(
        development_subjects=list(range(25,49)),
        resolution="Original extraction; 1280px retry for failed enrolment",
        native_eye_recovery=sorted(failures), recovery_protocol=str(RECOVERY / "protocol.json"),
        note="Adjusted preprocessing on development calibration data; all 24 drivers retained",
        split_recordings="Physical clips remain separate; temporal history resets",
        final_test="Fold5 has not been extracted or scored"))
    write_json(ROOT / "status.json", dict(phase="complete", subjects=24,
                                          elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        write_json(ROOT / "status.json", {"phase": "failed", "error": repr(error)})
        raise
