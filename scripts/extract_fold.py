"""Parallel full-rate geometry extraction with source-validated resumable caches."""

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import cv2

from drowsiness.dataset import extract, load_frames, read_manifest, write_json


def worker(subject, rows, output):
    cv2.setNumThreads(1)
    directory = output / "features" / f"subject-{subject}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "video_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return str(extract(manifest, directory, sample_fps=30))


def run(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = read_manifest(args.manifest)
    subjects = sorted({r["subject_id"] for r in rows})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), *Path("src/drowsiness").glob("*.py")]
    signature = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    signature["manifest"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    receipt = output / "source-signature.json"
    if receipt.exists() and json.loads(receipt.read_text()) != signature:
        raise ValueError("Extraction source/manifest changed; use a new output directory")
    write_json(receipt, signature)
    started = time.monotonic()
    indexes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(worker, s, [r for r in rows if r["subject_id"] == s], output): s
            for s in subjects
        }
        while pending:
            done, _ = wait(pending, timeout=30, return_when="FIRST_COMPLETED")
            for future in done:
                indexes.append(Path(future.result()))
                del pending[future]
            count = sum(
                p.name != "manifest.csv" for p in (output / "features").glob("subject-*/*.csv")
            )
            status = dict(
                phase="extraction",
                completed_videos=count,
                total_videos=len(rows),
                completed_subjects=len(indexes),
                elapsed_seconds=time.monotonic() - started,
            )
            write_json(output / "status.json", status)
            print(json.dumps(status), flush=True)
    merged, audit = {"settings": None, "videos": []}, []
    for path in sorted(indexes):
        index = json.loads(path.read_text())
        if merged["settings"] is None:
            merged["settings"] = index["settings"]
        if merged["settings"] != index["settings"]:
            raise ValueError("Inconsistent extraction settings")
        for video in index["videos"]:
            count, valid, last = 0, 0, -1
            for frame in load_frames(path.parent / video["feature_file"]):
                if frame.timestamp <= last:
                    raise ValueError("Non-increasing frame timestamps")
                count += 1
                valid += frame.valid
                last = frame.timestamp
            merged["videos"].append(
                dict(video, feature_file=f"{path.parent.name}/{video['feature_file']}")
            )
            audit.append(dict(video, sampled_frames=count, valid_frames=valid, last_timestamp=last))
    write_json(output / "features/index.json", merged)
    write_json(output / "extraction-audit.json", audit)
    write_json(
        output / "status.json",
        dict(
            phase="complete",
            videos=len(rows),
            subjects=len(subjects),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    run(parser.parse_args())
