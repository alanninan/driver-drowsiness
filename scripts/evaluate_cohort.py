"""Frozen new-cohort test and resumable nested development evaluation."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from build_behavior_cohort import sha
from improve_model import CANDIDATES, estimator, select_model, video_metrics
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from drowsiness.dataset import write_json
from drowsiness.evaluation import metrics
from drowsiness.learned import load_classifier


def load_cohorts(paths):
    windows, matrices, names, signature = [], [], None, {}
    seen = set()
    for path in paths:
        keys = json.loads((path / "feature-names.json").read_text())["enrolled"]
        if names is not None and keys != names:
            raise ValueError("Incompatible feature names")
        names = keys
        rows = json.loads((path / "windows.json").read_text())
        subjects = {w["subject"] for w in rows}
        if seen & subjects:
            raise ValueError("Overlapping cohort subjects")
        seen.update(subjects)
        x = np.load(path / "features.npz", allow_pickle=False)["enrolled"]
        if x.shape != (len(rows), len(names)) or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        windows.extend(rows)
        matrices.append(x)
        signature[str(path)] = {
            n: sha(path / n) for n in ["features.npz", "windows.json", "feature-names.json"]
        }
    return windows, np.concatenate(matrices), names, signature


def summarize(windows, y, prediction):
    groups = np.array([w["subject"] for w in windows])
    return dict(
        windows=len(y),
        subjects=len(set(groups)),
        videos=len({w["video"] for w in windows}),
        window_metrics=metrics(y, prediction),
        video_metrics=video_metrics(windows, y, prediction),
        per_subject={
            s: metrics(y[groups == s], prediction[groups == s]) for s in sorted(set(groups))
        },
    )


def save_predictions(output, windows, prediction):
    write_json(
        output / "predictions.json",
        [
            dict(
                subject=w["subject"],
                video=w["video"],
                start=w["start"],
                end=w["end"],
                label=w["label"],
                prediction=int(p),
            )
            for w, p in zip(windows, prediction, strict=True)
        ],
    )


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    windows, x, names, signature = load_cohorts(args.cohorts)
    preprocessing_note = getattr(args, "preprocessing_note", None) or {
        str(path): json.loads((path / "preprocessing.json").read_text())
        for path in args.cohorts
        if (path / "preprocessing.json").exists()
    }
    if preprocessing_note:
        signature["preprocessing_note"] = preprocessing_note
    y = np.array([w["label"] for w in windows])
    groups = np.array([w["subject"] for w in windows])
    signature["code"] = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path("scripts/improve_model.py"),
            Path("src/drowsiness/evaluation.py"),
            Path("uv.lock"),
        ]
    }
    if args.frozen:
        model, metadata = load_classifier(args.frozen)
        if metadata["feature_names"] != names or set(metadata["subjects"]) & set(groups):
            raise ValueError("Frozen model feature mismatch or overlapping test subjects")
        signature["model"] = sha(args.frozen)
        prediction = model.predict(x)
        report = dict(
            summarize(windows, y, prediction),
            protocol="Frozen Fold1 model, new Fold2 drivers; explicit known-alert enrolment",
            signature=signature,
            training_subjects=metadata["subjects"],
            test_subjects=sorted(set(groups)),
            preprocessing_note=preprocessing_note,
            elapsed_seconds=time.monotonic() - started,
        )
        if (output / "report.json").exists():
            previous = json.loads((output / "report.json").read_text())
            if (
                previous["signature"] != signature
                or previous["window_metrics"] != report["window_metrics"]
            ):
                raise ValueError("Refusing to overwrite a different frozen benchmark")
        write_json(output / "report.json", report)
        save_predictions(output, windows, prediction)
        print(json.dumps(report, indent=2), flush=True)
        return
    candidates = CANDIDATES
    choose = select_model

    def fit_selected(name, a, b):
        return estimator(name).fit(a, b)

    signature["candidates"] = candidates
    checkpoint = output / "checkpoint.json"
    saved = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else dict(signature=signature, folds=[])
    )
    if saved["signature"] != signature:
        raise ValueError("Nested evaluation source/data changed; use a new output directory")
    write_json(
        output / "experiment.json",
        dict(
            signature=signature,
            protocol="24-driver development LOSO; three subject-grouped inner folds",
            candidates=candidates,
            seed=42,
            enrolment_seconds=180,
            preprocessing_note=preprocessing_note,
        ),
    )
    prediction = np.full(len(y), -1)
    completed = {f["subject"]: f for f in saved["folds"]}
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        subject = str(groups[test][0])
        training_subjects = sorted(set(groups[train]))
        if subject in completed:
            fold = completed[subject]
            if fold["train_subjects"] != training_subjects or fold["test_ids"] != test.tolist():
                raise ValueError("Checkpoint split mismatch")
            prediction[test] = fold["predictions"]
        else:
            best, scores = choose(x[train], y[train], groups[train])
            prediction[test] = fit_selected(best, x[train], y[train]).predict(x[test])
            fold = dict(
                subject=subject,
                selected=best,
                inner_scores=scores,
                train_subjects=training_subjects,
                test_ids=test.tolist(),
                predictions=prediction[test].tolist(),
                metrics=metrics(y[test], prediction[test]),
            )
        folds.append(fold)
        write_json(checkpoint, dict(signature=signature, folds=folds))
        print(
            f"{len(folds)}/{len(set(groups))} held out {subject}: {fold['selected']}; accuracy {fold['metrics']['accuracy']:.3f}",
            flush=True,
        )
    report = dict(
        summarize(windows, y, prediction),
        folds=folds,
        signature=signature,
        elapsed_seconds=time.monotonic() - started,
        protocol="Expanded development nested LOSO; not an untouched final test",
        preprocessing_note=preprocessing_note,
    )
    report["cohort_metrics"] = {}
    for path in args.cohorts:
        subjects = {w["subject"] for w in json.loads((path / "windows.json").read_text())}
        mask = np.array([g in subjects for g in groups])
        report["cohort_metrics"][str(path)] = summarize(
            [w for w, keep in zip(windows, mask, strict=True) if keep], y[mask], prediction[mask]
        )
    write_json(output / "report.json", report)
    save_predictions(output, windows, prediction)
    print(
        json.dumps({k: report[k] for k in ["window_metrics", "video_metrics"]}, indent=2),
        flush=True,
    )
    best, scores = choose(x, y, groups)
    model = fit_selected(best, x, y)
    artifact = output / "behavior-candidate.joblib"
    joblib.dump(model, artifact, compress=3)
    write_json(
        artifact.with_suffix(".json"),
        dict(
            schema=1,
            variant="enrolled",
            estimator=best,
            feature_names=names,
            subjects=sorted(set(groups)),
            training_windows=len(y),
            selection="Three subject-grouped inner folds across all development drivers",
            selection_macro_f1=scores,
            benchmark_report=str((output / "report.json").resolve()),
            sha256=sha(artifact),
            limitation="Development artifact; independent later-fold evaluation pending",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--preprocessing-note")
    parser.add_argument("--suite", choices=["original"], default="original")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
