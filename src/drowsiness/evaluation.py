"""Subject-disjoint evaluation, with an explicit known-alert enrolment protocol."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from .core import Baseline, Calibrator, Config, Monitor
from .dataset import LABELS, load_frames, write_json

FEATURES = [
    "closure_ratio",
    "blink_seconds",
    "blinks_per_minute",
    "yawns_per_minute",
    "nods_per_minute",
    "closure_seconds",
    "ear_relative",
    "mar_relative",
]


def build_windows(index_path, protocol="enrolled", seconds=30, enrolment_seconds=60.0):
    if protocol not in {"enrolled", "fixed"} or seconds < 5:
        raise ValueError("Use enrolled/fixed protocol and windows of at least 5 seconds")
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    videos = index["videos"]
    if len({v["feature_file"] for v in videos}) != len(videos):
        raise ValueError("A feature file appears more than once in the index")
    baselines, enrolment_files = {}, set()
    subjects = sorted({str(v["subject_id"]) for v in videos})
    if protocol == "enrolled":
        for subject in subjects:
            candidates = [
                v for v in videos if str(v["subject_id"]) == subject and int(v["label"]) == 0
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Subject {subject}: enrolled protocol needs exactly one alert clip"
                )
            clip = candidates[0]
            calibrator = Calibrator(45.0)
            for frame in load_frames(index_path.parent / clip["feature_file"]):
                if frame.timestamp >= enrolment_seconds:
                    break
                calibrator.update(frame)
            try:
                baselines[subject] = calibrator.finish()
            except ValueError as exc:
                raise ValueError(f"Subject {subject} enrolment failed: {exc}") from exc
            enrolment_files.add(clip["feature_file"])
    windows, report = [], []
    for video in videos:
        subject = str(video["subject_id"])
        baseline = baselines.get(subject, Baseline())
        personal, fixed = Monitor(baseline=baseline), Monitor(baseline=Baseline())
        start = enrolment_seconds if video["feature_file"] in enrolment_files else 0.0
        bins = {}
        last_t = start
        for frame in load_frames(index_path.parent / video["feature_file"]):
            if frame.timestamp < start:
                continue
            last_t = frame.timestamp
            result, reference = personal.update(frame), fixed.update(frame)
            # Both methods receive the same frames and evaluation intervals.
            bucket = int(frame.timestamp - start)
            if bucket in bins:
                continue
            eligible = result["score"] is not None and reference["score"] is not None
            if eligible:
                features = [result.get(f, 0) for f in FEATURES[:6]] + [
                    frame.ear / ((baseline.ear_left + baseline.ear_right) / 2),
                    frame.mar - baseline.mar,
                ]
                bins[bucket] = (features, result["score"], reference["score"])
            else:
                bins[bucket] = None
        total, kept = 0, 0
        warmup = int(Config().warmup_seconds) + 1
        for begin in range(warmup, int(last_t - start) - seconds + 1, seconds):
            total += 1
            sequence = [bins.get(second) for second in range(begin, begin + seconds)]
            if any(item is None for item in sequence):
                continue
            x = np.array([item[0] for item in sequence], dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Non-finite evaluation features")
            score = float(np.median([item[1] for item in sequence]))
            fixed_score = float(np.median([item[2] for item in sequence]))
            windows.append(
                {
                    "x": x,
                    "subject": subject,
                    "video": video["feature_file"],
                    "label": LABELS[int(video["label"])],
                    "start": start + begin,
                    "end": start + begin + seconds,
                    "rule": score_label(score),
                    "fixed": score_label(fixed_score),
                }
            )
            kept += 1
        report.append(
            {
                "subject": subject,
                "video": video["feature_file"],
                "label": int(video["label"]),
                "candidate_windows": total,
                "retained_windows": kept,
                "excluded_prefix_seconds": start,
            }
        )
    if not windows:
        raise ValueError("No complete evaluation windows. Inspect duration and tracking quality")
    return windows, {
        "protocol": protocol,
        "sequence_seconds": seconds,
        "enrolment_seconds": enrolment_seconds if protocol == "enrolled" else 0,
        "baselines": {k: asdict(v) for k, v in baselines.items()},
        "coverage": report,
        "extraction": index.get("settings", {}),
    }


def score_label(score):
    return 2 if score >= 65 else 1 if score >= 35 else 0


def metrics(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "class_support": np.bincount(truth, minlength=3).tolist(),
    }


def subject_splits(windows):
    groups = np.array([w["subject"] for w in windows])
    if len(set(groups)) < 3:
        raise ValueError("LOSO evaluation requires at least three retained subjects")
    for train, test in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups):
        assert not set(groups[train]) & set(groups[test])
        if {windows[i]["label"] for i in train} != {0, 1, 2}:
            raise ValueError("Every training fold must contain all three classes")
        yield train, test


def evaluate(
    index, output, protocol="enrolled", model="rf", seconds=30, seed=42, epochs=20, device="auto"
):
    if model not in {"rf", "gru", "rules"}:
        raise ValueError("Unknown evaluation model")
    windows, metadata = build_windows(index, protocol, seconds)
    x = np.stack([w["x"] for w in windows])
    y = np.array([w["label"] for w in windows])
    all_predictions = {
        "fixed_rules": np.array([w["fixed"] for w in windows]),
        "personal_rules" if protocol == "enrolled" else "rules": np.array(
            [w["rule"] for w in windows]
        ),
    }
    if model != "rules":
        all_predictions[model] = np.full(len(y), -1)
    folds = []
    for number, (train, test) in enumerate(subject_splits(windows)):
        test_subject = windows[test[0]]["subject"]
        print(f"LOSO {number + 1}: held-out subject {test_subject}")
        training_info = {}
        if model == "rf":
            summary = np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
            estimator = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            estimator.fit(summary[train], y[train])
            all_predictions[model][test] = estimator.predict(summary[test])
        elif model == "gru":
            from .temporal import fit_predict

            prediction, training_info = fit_predict(
                x[train], y[train], x[test], epochs=epochs, seed=seed, device=device
            )
            all_predictions[model][test] = prediction
        folds.append(
            {
                "test_subject": test_subject,
                "train_subjects": sorted({windows[i]["subject"] for i in train}),
                "training_windows": len(train),
                "test_windows": len(test),
                "training": training_info,
                "metrics": {
                    name: metrics(y[test], pred[test]) for name, pred in all_predictions.items()
                },
            }
        )
    videos = sorted({w["video"] for w in windows})
    video_results = {}
    for name, predictions in all_predictions.items():
        truth, guesses = [], []
        for video in videos:
            ids = [i for i, window in enumerate(windows) if window["video"] == video]
            truth.append(int(y[ids[0]]))
            # Deterministic tie-break: most severe of equally frequent labels.
            counts = Counter(int(predictions[i]) for i in ids)
            guesses.append(max(counts, key=lambda label: (counts[label], label)))
        video_results[name] = metrics(truth, guesses)
    report = dict(
        metadata,
        seed=seed,
        model=model,
        features=FEATURES,
        config=asdict(Config()),
        software={name: version(name) for name in ("numpy", "scikit-learn")},
        folds=folds,
        index_sha256=hashlib.sha256(Path(index).read_bytes()).hexdigest(),
        window_metrics={name: metrics(y, pred) for name, pred in all_predictions.items()},
        video_metrics=video_results,
        mean_subject_macro_f1={
            name: float(np.mean([fold["metrics"][name]["macro_f1"] for fold in folds]))
            for name in all_predictions
        },
        label_names=["alert", "low_vigilance", "drowsy"],
        limitations=[
            "Video labels are weak labels inherited by windows",
            "No event-level early-warning or clinical accuracy claim",
            "Enrolled protocol uses a separate known-alert prefix per subject",
            "Unusable windows are excluded; report coverage alongside accuracy",
            "Heuristics are engineering defaults, not optimised on this dataset",
        ],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_json(
        output / "predictions.json",
        [
            {
                **{k: v for k, v in window.items() if k != "x"},
                "predictions": {name: int(pred[i]) for name, pred in all_predictions.items()},
            }
            for i, window in enumerate(windows)
        ],
    )
    return report
