"""Nested subject-disjoint comparison of richer signals and alert enrolment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from drowsiness.behavior import calibrate_reference, extract_behavior, frame_array
from drowsiness.dataset import load_frames, write_json
from drowsiness.evaluation import metrics

CANDIDATES = ["svm_0.1", "svm_1", "svm_10", "svm_100", "extra_trees", "rf", "hist"]


def estimator(name):
    if name.startswith("svm_"):
        return make_pipeline(
            StandardScaler(),
            SVC(C=float(name.split("_")[1]), class_weight="balanced", kernel="rbf"),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=4
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=4
        )
    if name == "hist":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=7,
            l2_regularization=5,
            random_state=42,
            class_weight="balanced",
            early_stopping=False,
        )
    raise ValueError(name)


def select_model(x, y, groups):
    splits = list(GroupKFold(n_splits=3).split(x, y, groups))
    scores = {}
    for name in CANDIDATES:
        fold_scores = []
        for train, valid in splits:
            assert not set(groups[train]) & set(groups[valid])
            model = estimator(name).fit(x[train], y[train])
            prediction = model.predict(x[valid])
            fold_scores.extend(
                f1_score(
                    y[valid][groups[valid] == subject],
                    prediction[groups[valid] == subject],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
                for subject in sorted(set(groups[valid]))
            )
        scores[name] = float(np.mean(fold_scores))
    return max(CANDIDATES, key=lambda name: scores[name]), scores


def build_features(index_path, baseline_path, output):
    index = json.loads(index_path.read_text())
    original = json.loads(baseline_path.read_text())
    windows = [w for w in original if w["label"] != 0 or w["start"] >= 191]
    frames, references = {}, {}
    for video in index["videos"]:
        key = video["feature_file"]
        frames[key] = frame_array(load_frames(index_path.parent / key))
        if int(video["label"]) == 0:
            prefix = frames[key][frames[key][:, 0] < 180]
            references[str(video["subject_id"])] = calibrate_reference(prefix)
    write_json(output / "calibration.json", {k: v.to_dict() for k, v in references.items()})
    matrices, names = {"raw": [], "enrolled": []}, {}
    for i, window in enumerate(windows):
        data = frames[window["video"]]
        for variant in matrices:
            reference = references[window["subject"]] if variant == "enrolled" else None
            x, names[variant] = extract_behavior(
                data, window["end"], reference, 180 if window["label"] == 0 else 0
            )
            matrices[variant].append(x)
        if (i + 1) % 100 == 0:
            print(f"Features {i + 1}/{len(windows)}", flush=True)
    matrices = {k: np.stack(v) for k, v in matrices.items()}
    np.savez_compressed(output / "features.npz", **matrices)
    write_json(output / "windows.json", windows)
    write_json(output / "feature-names.json", names)
    return windows, matrices


def video_metrics(windows, truth, prediction):
    labels, guesses = [], []
    for video in sorted({w["video"] for w in windows}):
        ids = np.array([i for i, w in enumerate(windows) if w["video"] == video])
        counts = np.bincount(prediction[ids], minlength=3)
        labels.append(truth[ids[0]])
        guesses.append(max(range(3), key=lambda label: (counts[label], label)))
    return metrics(labels, guesses)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    signature = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path("src/drowsiness/behavior.py"),
            Path(args.index),
            Path(args.baseline),
        ]
    }
    metadata = dict(
        signature=signature,
        candidates=CANDIDATES,
        seed=42,
        protocol="nested LOSO; 3-fold grouped inner selection",
        enrolment_seconds=180,
        contexts_seconds=[30, 90],
    )
    restored = {}
    if args.resume:
        args.reuse_features = True
        previous = json.loads((output / "progress.json").read_text())
        restored = {(f["variant"], f["subject"]): f for f in previous["completed_folds"]}
        write_json(output / "progress-before-resume.json", previous)
        metadata["previous_run"] = json.loads((output / "experiment.json").read_text())
        metadata["resume_note"] = (
            "Preserved completed inner selections; deterministic refits "
            "must reproduce stored metrics. BLAS/OpenMP threads reduced to one."
        )
    write_json(output / "experiment.json", metadata)
    (output / "experiment-source.py").write_bytes(Path(__file__).read_bytes())
    (output / "behavior-source.py").write_bytes(Path("src/drowsiness/behavior.py").read_bytes())
    if args.reuse_features:
        windows = json.loads((output / "windows.json").read_text())
        matrices = dict(np.load(output / "features.npz", allow_pickle=False))
    else:
        windows, matrices = build_features(Path(args.index), Path(args.baseline), output)
    y = np.array([w["label"] for w in windows])
    groups = np.array([w["subject"] for w in windows])
    predictions = {
        name: np.array([w["predictions"][name] for w in windows]) for name in ["fixed_rules", "rf"]
    }
    folds = []
    for variant, x in matrices.items():
        predictions[variant] = np.full(len(y), -1)
        for train, test in LeaveOneGroupOut().split(x, y, groups):
            subject = str(groups[test][0])
            saved = restored.get((variant, subject))
            if saved:
                best, scores = saved["selected"], saved["inner_scores"]
            else:
                best, scores = select_model(x[train], y[train], groups[train])
            model = estimator(best).fit(x[train], y[train])
            predictions[variant][test] = model.predict(x[test])
            result = metrics(y[test], predictions[variant][test])
            if saved and result != saved["metrics"]:
                raise ValueError(f"Resumed fold {variant}/{subject} did not reproduce its metrics")
            folds.append(
                dict(
                    variant=variant,
                    subject=subject,
                    selected=best,
                    inner_scores=scores,
                    train_subjects=sorted(set(groups[train])),
                    metrics=result,
                )
            )
            print(
                f"{variant} held out {subject}: {best}, accuracy {result['accuracy']:.3f}, "
                f"F1 {result['macro_f1']:.3f}",
                flush=True,
            )
            write_json(
                output / "progress.json",
                dict(completed_folds=folds, elapsed_seconds=time.monotonic() - started),
            )
    report = dict(
        **metadata,
        windows=len(y),
        subjects=len(set(groups)),
        folds=folds,
        window_metrics={k: metrics(y, p) for k, p in predictions.items()},
        video_metrics={k: video_metrics(windows, y, p) for k, p in predictions.items()},
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "report.json", report)
    write_json(
        output / "predictions.json",
        [
            dict(
                subject=w["subject"],
                video=w["video"],
                start=w["start"],
                end=w["end"],
                label=w["label"],
                predictions={k: int(p[i]) for k, p in predictions.items()},
            )
            for i, w in enumerate(windows)
        ],
    )
    print(json.dumps(report["window_metrics"], indent=2), flush=True)
    print(json.dumps(report["video_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="outputs/fold1-fixed/features/index.json")
    parser.add_argument("--baseline", default="outputs/fold1-fixed/predictions.json")
    parser.add_argument("--output", default="outputs/fold1-improved")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--resume", action="store_true")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
