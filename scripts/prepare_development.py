"""Prepare Fold3/4 geometry and features without opening the final test fold."""

import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from build_behavior_cohort import build, sha

from drowsiness.behavior import calibrate_reference, frame_array
from drowsiness.dataset import extract, load_frames, read_manifest, write_json

ROOT = Path("outputs/folds34-run")


def retry_video(video):
    cv2.setNumThreads(1)
    key = hashlib.sha256(video["video_path"].encode()).hexdigest()[:16]
    output = ROOT / "resolution-fallback" / key
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    write_manifest(manifest, [video])
    index = extract(manifest, output, max_width=1280)
    entry = json.loads(index.read_text())["videos"][0]
    return dict(entry, feature_file=str((index.parent / entry["feature_file"]).resolve()))


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "video_path", "label"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifests = [Path(f"data/fold{f}-manifest.csv") for f in (3, 4)]
    while not all(p.exists() for p in manifests):
        if time.monotonic() - started > 3600:
            raise TimeoutError("Archive preparation did not finish within one hour; inspect logs")
        write_json(ROOT / "status.json", {"phase": "waiting_for_verified_archives"})
        time.sleep(15)
    rows = [r for manifest in manifests for r in read_manifest(manifest)]
    if {r["subject_id"] for r in rows} != {f"{n:02}" for n in range(25, 49)}:
        raise ValueError("Expected development subjects 25-48 only")
    if len(rows) != 73:
        raise ValueError("Expected 73 physical recordings including subject 32's two parts")
    manifest = ROOT / "manifest.csv"
    write_manifest(manifest, rows)
    geometry = Path("outputs/folds34-geometry")
    status = json.loads((geometry / "status.json").read_text()) if (geometry / "status.json").exists() else {}
    if status.get("phase") == "complete":
        sources = [Path("scripts/extract_fold.py").resolve(), *Path("src/drowsiness").glob("*.py")]
        expected = {str(p): sha(p) for p in sources}
        expected["manifest"] = sha(manifest)
        if json.loads((geometry / "source-signature.json").read_text()) != expected:
            raise ValueError("Completed extraction source/manifest mismatch")
        audit = json.loads((geometry / "extraction-audit.json").read_text())
        if len(audit) != 73 or status.get("subjects") != 24:
            raise ValueError("Incomplete extraction audit")
        print("Reusing completed, source-validated Fold3/4 extraction", flush=True)
    else:
        write_json(ROOT / "status.json", {"phase": "geometry", "videos": len(rows)})
        with (ROOT / "geometry.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "scripts/extract_fold.py", "--manifest", str(manifest),
                 "--output", str(geometry), "--workers", "4"],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    original = Path("outputs/folds34-geometry/features/index.json")
    index = json.loads(original.read_text())
    expected_videos = {(r["subject_id"], r["video_path"], int(r["label"])) for r in rows}
    actual_videos = {(v["subject_id"], v["video_path"], int(v["label"])) for v in index["videos"]}
    if actual_videos != expected_videos or len(index["videos"]) != 73:
        raise ValueError("Extraction index does not match the development manifest")
    write_json(ROOT / "status.json", {"phase": "calibration_audit"})
    failed = {}
    for video in index["videos"]:
        if int(video["label"]) == 0:
            data = frame_array(load_frames(original.parent / video["feature_file"]))
            try:
                calibrate_reference(data[data[:, 0] < 180])
            except ValueError as error:
                failed[video["subject_id"]] = str(error)
    write_json(ROOT / "resolution-protocol.json", {
        "selection": "All recordings of subjects failing the first 180s known-alert reference",
        "failed_original": failed, "retry_max_width": 1280,
        "minimum_usable_seconds": 120, "original_index_sha256": sha(original),
    })
    write_json(ROOT / "status.json", {"phase": "resolution_fallback", "failed_original": failed})
    selected = [v for v in index["videos"] if v["subject_id"] in failed]
    with ProcessPoolExecutor(max_workers=3) as pool:
        replacements = {v["video_path"]: v for v in pool.map(retry_video, selected)}
    merged = [replacements.get(v["video_path"], dict(
        v, feature_file=str((original.parent / v["feature_file"]).resolve())
    )) for v in index["videos"]]
    # Use physical video identity, not (subject, label), to preserve split recordings.
    final_index = ROOT / "features-index.json"
    write_json(final_index, {"settings": dict(index["settings"], resolution_fallback=failed),
                            "videos": merged})
    write_json(ROOT / "status.json", {"phase": "behavior_features"})
    build(final_index, Path("outputs/folds34-behavior"))
    write_json(Path("outputs/folds34-behavior/preprocessing.json"), {
        "development_subjects": list(range(25, 49)),
        "resolution": "960px; all recordings retried at 1280px if known-alert enrolment fails",
        "split_recordings": "32/10_1 and 32/10_2 remain separate; temporal context resets",
        "final_test": "Fold5 has not been extracted or scored",
    })
    write_json(ROOT / "status.json", {"phase": "complete", "subjects": 24,
                                      "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        write_json(ROOT / "status.json", {"phase": "failed", "error": repr(error)})
        raise
