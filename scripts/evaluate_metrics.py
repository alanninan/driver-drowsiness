"""Calculate and display screenshot-friendly held-out Fold5 metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
from build_behavior_cohort import sha
from evaluate_cohort import load_cohorts
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from drowsiness.dataset import write_json
from drowsiness.learned import load_classifier

DEFAULT_MODEL = Path("outputs/folds1234-extra-trees/behavior.joblib")
CLASS_NAMES = ["Alert", "Low vigilance", "Drowsy"]


def validate_holdout(training_subjects, test_subjects):
    train, test = set(training_subjects), set(test_subjects)
    if not train or not train <= {f"{n:02}" for n in range(1, 49)}:
        raise ValueError("Model must be trained only on subjects 01-48 (Folds1-4).")
    if test != {f"{n:02}" for n in range(49, 61)} or train & test:
        raise ValueError("Evaluation requires all 12 unseen Fold5 participants (49-60).")


def calculate_metrics(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if truth.ndim != 1 or truth.shape != predictions.shape or not len(truth):
        raise ValueError("Expected equally sized, non-empty label vectors")
    if not set(truth) <= {0, 1, 2} or not set(predictions) <= {0, 1, 2}:
        raise ValueError("Expected labels 0, 1, 2")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1, 2], zero_division=0)
    weights = support / support.sum()
    return dict(accuracy=float(accuracy_score(truth, predictions)),
        correct=int((truth == predictions).sum()), samples=len(truth),
        macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()), weighted_precision=float(precision @ weights),
        weighted_recall=float(recall @ weights), weighted_f1=float(f1 @ weights),
        per_class=[dict(name=name, precision=float(p), recall=float(r), f1=float(f), support=int(s))
                   for name, p, r, f, s in zip(CLASS_NAMES, precision, recall, f1, support, strict=True)],
        confusion_matrix=confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist())


def format_summary(report):
    m = report["metrics"]
    line = "=" * 76
    training = report["training_subjects"]
    folds = sorted({(int(s) - 1) // 12 + 1 for s in training})
    lines = [line, "DRIVER DROWSINESS DETECTION | EXTRA TREES", line,
        f"Train: Folds {', '.join(map(str, folds))} ({len(training)} participants)",
        f"Test:  Fold 5 ({len(report['test_subjects'])} participants) | 3 classes",
        "Protocol: offline classification after personal calibration",
        f"Test windows: {m['samples']:,} | Correct: {m['correct']:,}",
        f"Retained videos: {report['retained_videos']} / 37 | "
        f"Model SHA256: {report['model_sha256'][:16]}", "-" * 76,
        f"  ACCURACY        {m['accuracy']:>7.2%}    MACRO PRECISION  {m['macro_precision']:>7.2%}",
        f"  MACRO RECALL    {m['macro_recall']:>7.2%}    MACRO F1         {m['macro_f1']:>7.2%}",
        "-" * 76,
        f"{'CLASS':<20}{'PRECISION':>13}{'RECALL':>13}{'F1':>13}{'SUPPORT':>13}"]
    for row in m["per_class"]:
        lines.append(f"{row['name']:<20}{row['precision']:>13.2%}{row['recall']:>13.2%}"
                     f"{row['f1']:>13.2%}{row['support']:>13,}")
    lines += [f"{'Weighted average':<20}{m['weighted_precision']:>13.2%}"
              f"{m['weighted_recall']:>13.2%}{m['weighted_f1']:>13.2%}{m['samples']:>13,}",
              "-" * 76, "CONFUSION MATRIX | Rows: actual / Columns: predicted",
              f"{'':20}{'Alert':>16}{'Low vigilance':>16}{'Drowsy':>16}"]
    for name, row in zip(CLASS_NAMES, m["confusion_matrix"], strict=True):
        lines.append(f"{name:<20}{row[0]:>16,}{row[1]:>16,}{row[2]:>16,}")
    lines += ["-" * 76, "Scores apply to retained windows; live accuracy is not measured.",
              "Fold 5 was previously evaluated in this project; it is not a fresh test.", line]
    return "\n".join(lines) + "\n"


def evaluate(model_path, output):
    if output.exists():
        raise FileExistsError(f"Metrics output exists: {output}; choose a new --output")
    classifier, metadata = load_classifier(model_path)
    validate_holdout(metadata["subjects"], {f"{n:02}" for n in range(49, 61)})
    windows, x, names, signature = load_cohorts([Path("outputs/fold5-behavior")])
    subjects = sorted({w["subject"] for w in windows})
    validate_holdout(metadata["subjects"], subjects)
    if names != metadata["feature_names"]:
        raise ValueError("Training/test feature order mismatch")
    truth = np.array([w["label"] for w in windows])
    prediction = classifier.predict(x)
    report = dict(metrics=calculate_metrics(truth, prediction),
        training_subjects=metadata["subjects"], test_subjects=subjects,
        retained_videos=len({w["video"] for w in windows}),
        model_path=str(model_path), model_sha256=sha(model_path), test_signature=signature,
        protocol="Fold5 excluded from classifier fitting; offline after known-alert calibration",
        limitation="Fold5 evaluated previously; not a fresh research holdout. Live accuracy unmeasured.")
    output.mkdir(parents=True)
    write_json(output / "metrics.json", report)
    text = format_summary(report)
    (output / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/folds1234-extra-trees/evaluation"))
    parser.add_argument("--from-report", type=Path, help="Redisplay saved metrics.json without writing")
    args = parser.parse_args()
    if args.from_report:
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
        validate_holdout(report["training_subjects"], report["test_subjects"])
        print(format_summary(report), end="")
    else:
        if not args.model.exists():
            parser.error("Candidate not found. Run scripts/train_full_dataset.py first. "
                         "The live all-folds model cannot provide held-out Fold5 metrics.")
        evaluate(args.model, args.output)


if __name__ == "__main__":
    main()
