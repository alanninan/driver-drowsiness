"""Build source-validated behavioural features with the frozen Fold1 window rules."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from drowsiness.behavior import calibrate_reference, extract_behavior, frame_array
from drowsiness.dataset import load_frames, write_json
from drowsiness.evaluation import build_windows


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(index_path, output):
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text())
    signature = {
        str(p): sha(p)
        for p in [
            index_path,
            Path(__file__),
            Path("src/drowsiness/behavior.py"),
            Path("src/drowsiness/evaluation.py"),
            Path("src/drowsiness/core.py"),
        ]
    }
    signature["frames"] = {
        v["feature_file"]: sha(index_path.parent / v["feature_file"]) for v in index["videos"]
    }
    receipt = output / "cache-receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["inputs"] != signature or any(
            sha(output / name) != digest for name, digest in saved["outputs"].items()
        ):
            raise ValueError("Feature cache provenance mismatch; use a new output directory")
        print(f"Verified feature cache: {output}", flush=True)
        return
    windows, coverage = build_windows(index_path, protocol="fixed")
    windows = [
        {k: v for k, v in w.items() if k != "x"}
        for w in windows
        if w["label"] != 0 or w["start"] >= 191
    ]
    references, failures = {}, {}
    # Only explicit known-alert prefixes may establish a reference.
    for video in index["videos"]:
        if int(video["label"]) != 0:
            continue
        data = frame_array(load_frames(index_path.parent / video["feature_file"]))
        try:
            references[str(video["subject_id"])] = calibrate_reference(data[data[:, 0] < 180])
        except ValueError as error:
            failures[str(video["subject_id"])] = str(error)
    write_json(output / "calibration.json", {s: r.to_dict() for s, r in references.items()})
    write_json(output / "calibration-failures.json", failures)
    if failures:
        # Never silently discard hard participants and inflate the reported score.
        raise ValueError(f"Enrolment failed; inspect calibration-failures.json: {failures}")
    result, names, ordered = [], None, []
    for video in index["videos"]:
        data = frame_array(load_frames(index_path.parent / video["feature_file"]))
        selected = [w for w in windows if w["video"] == video["feature_file"]]
        for window in selected:
            values, keys = extract_behavior(
                data,
                window["end"],
                references[window["subject"]],
                180 if window["label"] == 0 else 0,
            )
            if names is not None and names != keys:
                raise ValueError("Feature order changed")
            names = keys
            result.append(values)
            ordered.append(window)
        print(
            f"Features: {video['subject_id']}/{video['label']}, {len(selected)} windows", flush=True
        )
    if not result:
        raise ValueError("No retained windows")
    np.savez_compressed(output / "features.npz", enrolled=np.stack(result))
    write_json(output / "windows.json", ordered)
    write_json(output / "feature-names.json", {"enrolled": names})
    write_json(output / "coverage.json", coverage)
    write_json(
        receipt,
        dict(
            inputs=signature,
            outputs={
                name: sha(output / name)
                for name in [
                    "features.npz",
                    "windows.json",
                    "feature-names.json",
                    "calibration.json",
                    "coverage.json",
                ]
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.index, args.output)
