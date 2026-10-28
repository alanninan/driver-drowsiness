"""Train on Folds1-3, validate once on Fold4, and stop before Fold5 preparation."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from build_behavior_cohort import sha
from evaluate_cohort import load_cohorts, save_predictions, summarize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from drowsiness.dataset import write_json

OUTPUT = Path("outputs/folds123-extra-trees")
COHORTS = [Path("outputs/fold1-improved"), Path("outputs/fold2-resolution-behavior"),
           Path("outputs/folds34-behavior")]
# Fixed before looking at Fold4 scores. The existing configuration is first for ties.
CANDIDATES = [dict(min_samples_leaf=leaf, max_features=features)
              for leaf in (3, 1, 8) for features in ("sqrt", 0.5, 1.0)]


def development_split(groups):
    groups = np.asarray(groups)
    expected = {f"{n:02}" for n in range(1, 49)}
    if set(groups) != expected:
        raise ValueError("Require exactly subjects 01-48; Fold5 subjects are forbidden")
    train = np.array([int(s) <= 36 for s in groups])
    return train, ~train


def grouped_splits(groups):
    groups = np.asarray(groups)
    if set(groups) != {f"{n:02}" for n in range(1, 37)}:
        raise ValueError("Tuning requires training subjects 01-36 only")
    return list(GroupKFold(n_splits=3).split(np.zeros(len(groups)), groups=groups))


def model(parameters):
    return ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=4, **parameters)


def run():
    started = time.monotonic()
    windows, x, names, signature = load_cohorts(COHORTS)
    groups = np.array([w["subject"] for w in windows])
    training, validation = development_split(groups)
    y = np.array([w["label"] for w in windows])
    tx, ty, tg = x[training], y[training], groups[training]
    folds = grouped_splits(tg)
    signature["code"] = {str(p): sha(p) for p in [Path(__file__),
        Path("scripts/evaluate_cohort.py"), Path("scripts/improve_model.py"),
        Path("src/drowsiness/evaluation.py"), Path("uv.lock")]}
    signature["candidates"] = CANDIDATES
    checkpoint = OUTPUT / "tuning.json"
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        "signature": signature, "scores": []}
    if saved["signature"] != signature:
        raise ValueError("Training inputs changed; refusing to mix runs")
    for parameters in CANDIDATES[len(saved["scores"]):]:
        scores = {}
        for fold, (fit, heldout) in enumerate(folds):
            write_json(OUTPUT / "status.json", {"phase": "grouped_training_tuning",
                "candidate": len(saved["scores"]) + 1, "total_candidates": len(CANDIDATES),
                "inner_fold": fold + 1, "fold5_access": False})
            classifier = model(parameters).fit(tx[fit], ty[fit])
            prediction = classifier.predict(tx[heldout])
            for subject in sorted(set(tg[heldout])):
                mask = tg[heldout] == subject
                scores[str(subject)] = float(f1_score(ty[heldout][mask], prediction[mask],
                    labels=[0, 1, 2], average="macro", zero_division=0))
        saved["scores"].append({"parameters": parameters, "per_subject_macro_f1": scores,
                                "mean_subject_macro_f1": float(np.mean(list(scores.values())))})
        write_json(checkpoint, saved)
        print(json.dumps(saved["scores"][-1]), flush=True)
    selected = max(saved["scores"], key=lambda row: row["mean_subject_macro_f1"])
    write_json(OUTPUT / "selection.json", selected)  # Selection precedes validation scoring.
    write_json(OUTPUT / "status.json", {"phase": "fit_selected_and_validate_fold4"})
    classifier = model(selected["parameters"]).fit(tx, ty)
    artifact = OUTPUT / "behavior.joblib"
    joblib.dump(classifier, artifact)
    write_json(artifact.with_suffix(".json"), {"schema": 1, "variant": "enrolled",
        "estimator": "extra_trees", "feature_names": names,
        "subjects": sorted(set(tg)), "windows": len(ty), "sha256": sha(artifact),
        "parameters": classifier.get_params(), "selection": selected,
        "protocol": "Training-only subject-grouped tuning; Fold4 validation; Fold5 untouched"})
    prediction = classifier.predict(x[validation])
    validation_windows = [w for w, keep in zip(windows, validation, strict=True) if keep]
    report = dict(summarize(validation_windows, y[validation], prediction),
        protocol="Offline three-class classification after known-alert personal calibration",
        role="Fold4 validation, not final-test or live accuracy",
        training_subjects=sorted(set(tg)), validation_subjects=sorted(set(groups[validation])),
        selected=selected, signature=signature, fold5_prepared=False, fold5_tested=False,
        calibration="First 180 seconds of known-alert recording; at least 120 usable seconds",
        preprocessing={str(p): json.loads((p / "preprocessing.json").read_text())
                       for p in COHORTS if (p / "preprocessing.json").exists()},
        coverage={str(p): json.loads((p / "coverage.json").read_text())
                  for p in COHORTS if (p / "coverage.json").exists()},
        validation_retained_windows=len(validation_windows),
        validation_retained_videos=len({w["video"] for w in validation_windows}),
        validation_physical_videos=36, elapsed_seconds=time.monotonic() - started)
    write_json(OUTPUT / "report.json", report)
    save_predictions(OUTPUT, validation_windows, prediction)
    write_json(OUTPUT / "status.json", {"phase": "stopped_before_fold5_preparation_and_testing",
        "training_complete": True, "fold4_validation_complete": True,
        "fold5_prepared": False, "fold5_tested": False,
        "live_model_replaced": False, "report": str(OUTPUT / "report.json")})
    print(json.dumps(report["window_metrics"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-preparation", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        if args.wait_preparation:
            write_json(OUTPUT / "status.json", {"phase": "waiting_for_fold34_preparation",
                "stop_boundary": "before_fold5_preparation_and_testing"})
            deadline = time.monotonic() + 24 * 3600
            while True:
                status = json.loads(Path("outputs/folds34-run/status.json").read_text())
                if status["phase"] == "complete":
                    break
                if status["phase"] in {"failed", "paused_after_extraction"}:
                    raise RuntimeError(f"Preparation is not running: {status}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Preparation did not complete within 24 hours")
                time.sleep(15)
        with threadpool_limits(limits=1):
            run()
    except Exception as error:
        write_json(OUTPUT / "status.json", {"phase": "failed", "error": repr(error)})
        raise
