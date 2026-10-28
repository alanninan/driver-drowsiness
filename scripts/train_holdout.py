"""Train on Folds1-4 and evaluate on Fold5. Never change the deployed model."""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from build_behavior_cohort import sha
from evaluate_cohort import load_cohorts
from evaluate_metrics import DEFAULT_MODEL, evaluate
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from drowsiness.dataset import write_json
from drowsiness.learned import load_classifier

TRAIN_COHORTS = [Path("outputs/fold1-improved"), Path("outputs/fold2-resolution-behavior"),
                 Path("outputs/folds34-behavior")]
TRAIN_SUBJECTS = {f"{n:02}" for n in range(1, 49)}


def validate_training(windows):
    if {w["subject"] for w in windows} != TRAIN_SUBJECTS:
        raise ValueError("Training requires exactly subjects 01-48; Fold5 must be excluded")
    if {w["label"] for w in windows} != {0, 1, 2}:
        raise ValueError("Training requires all three classes")


def run(output):
    output = output.resolve()
    allowed = Path("outputs").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("Choose a new subdirectory of outputs; existing artifacts are never overwritten")
    started = time.monotonic()
    windows, x, names, signature = load_cohorts(TRAIN_COHORTS)
    validate_training(windows)
    if len(names) != 464:
        raise ValueError("Expected 464 behavioural features")
    y = np.array([w["label"] for w in windows])
    output.mkdir(parents=True)
    write_json(output / "status.json", {"phase": "training_folds1_to4"})
    # Fixed development-selected settings; no tuning using Fold5 scores.
    classifier = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8,
        max_features=1.0, class_weight="balanced", random_state=42, n_jobs=4).fit(x, y)
    artifact = output / "behavior.joblib"
    joblib.dump(classifier, artifact, compress=3)
    write_json(artifact.with_suffix(".json"), dict(schema=1, variant="enrolled",
        estimator="extra_trees", feature_names=names, subjects=sorted(TRAIN_SUBJECTS),
        training_windows=len(y), parameters=classifier.get_params(), signature=signature,
        sha256=sha(artifact), code_sha256=sha(Path(__file__)),
        protocol="Fit only Folds1-4 (01-48); Fold5 (49-60) reserved for scoring",
        limitation="Fold5 data has been evaluated before; live accuracy unmeasured",
        calibration="First 180 known-alert seconds; minimum 120 usable; prefix excluded from fitting"))
    loaded, _ = load_classifier(artifact)
    np.testing.assert_array_equal(loaded.predict(x[:16]), classifier.predict(x[:16]))
    write_json(output / "status.json", {"phase": "evaluating_fold5"})
    evaluate(artifact, output / "evaluation")
    write_json(output / "status.json", dict(phase="complete", live_model_replaced=False,
        training_subjects=48, training_windows=len(y), elapsed_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL.parent)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)


if __name__ == "__main__":
    main()
