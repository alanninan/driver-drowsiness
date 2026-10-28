"""Explicit manifests and reusable feature extraction. Never downloads a dataset."""

import csv
import hashlib
import json
from dataclasses import asdict, fields
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from .core import Frame
from .vision import DEFAULT_MODEL, Tracker

LABELS = {0: 0, 5: 1, 10: 2}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def make_manifest(root, output):
    root, output = Path(root).resolve(), Path(output)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.stem in {"0", "5", "10"}:
            rows.append(
                {"subject_id": path.parent.name, "video_path": str(path), "label": int(path.stem)}
            )
    if not rows:
        raise ValueError("No subject/0,5,10 video files found. Create a manifest manually")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "video_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return {"videos": len(rows), "subjects": len({r["subject_id"] for r in rows})}


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"subject_id", "video_path", "label"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs subject_id,video_path,label columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    for row in rows:
        row["subject_id"] = row["subject_id"].strip()
        if not row["subject_id"]:
            raise ValueError("Subject IDs cannot be empty")
        row["label"] = int(row["label"])
        if row["label"] not in LABELS:
            raise ValueError("Use original UTA labels: 0=alert, 5=low vigilance, 10=drowsy")
        video = Path(row["video_path"])
        video = (path.parent / video).resolve() if not video.is_absolute() else video.resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        if video in seen:
            raise ValueError(f"Duplicate video in manifest: {video}")
        seen.add(video)
        row["video_path"] = str(video)
    return rows


def video_frames(path, sample_fps=30.0):
    if not 2 <= sample_fps <= 60:
        raise ValueError("sample_fps must be between 2 and 60")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has no usable frame rate: {path}")
        next_sample, last_t, index = 0.0, -1.0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            t = pts if np.isfinite(pts) and pts > last_t else index / fps
            if t <= last_t:
                t = last_t + 1 / fps
            last_t = t
            index += 1
            if t + 1e-6 < next_sample:
                continue
            next_sample = t + 1 / sample_fps - 1e-6
            yield frame, t
    finally:
        capture.release()


def extract(manifest, output, model=DEFAULT_MODEL, clahe=False, sample_fps=30.0, max_width=960):
    rows = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "schema": 1,
        "software": {
            name: version(name) for name in ("mediapipe", "opencv-contrib-python", "numpy")
        },
        "clahe": clahe,
        "sample_fps": sample_fps,
        "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
    }
    if max_width != 960:
        settings["max_width"] = max_width
    results = []
    for i, row in enumerate(rows):
        source = Path(row["video_path"])
        key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        target = output / f"{key}.csv"
        metadata_path = target.with_suffix(".json")
        metadata = dict(
            settings,
            source=str(source),
            bytes=source.stat().st_size,
            mtime_ns=source.stat().st_mtime_ns,
        )
        cached = (
            target.exists()
            and metadata_path.exists()
            and json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
        )
        if not cached:
            temporary = target.with_suffix(".partial")
            count = 0
            with (
                Tracker(model, clahe, max_width=max_width) as tracker,
                temporary.open("w", newline="", encoding="utf-8") as stream,
            ):
                writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(Frame)])
                writer.writeheader()
                for image, timestamp in video_frames(source, sample_fps):
                    frame, _ = tracker.process(image, timestamp)
                    writer.writerow(asdict(frame))
                    count += 1
            if count == 0:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"No decoded frames: {source}")
            temporary.replace(target)
            write_json(metadata_path, metadata)
        results.append(dict(row, feature_file=target.name))
        print(f"[{i + 1}/{len(rows)}] {'cached' if cached else 'extracted'} {source.name}")
    write_json(output / "index.json", {"settings": settings, "videos": results})
    return output / "index.json"


def load_frames(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            yield Frame(
                **{
                    key: (
                        value.lower() == "true"
                        if key == "valid"
                        else value
                        if key == "reason"
                        else float(value)
                    )
                    for key, value in row.items()
                }
            )
