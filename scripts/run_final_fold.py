"""Frozen Fold5 evaluation: no fitting, model selection, or test-driven adjustments."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import recover_small_face as recovery
from build_behavior_cohort import build, sha
from evaluate_cohort import load_cohorts, save_predictions, summarize
from prepare_fold import prepare
from threadpoolctl import threadpool_limits

from drowsiness.behavior import calibrate_reference, frame_array
from drowsiness.dataset import extract, load_frames, read_manifest, write_json
from drowsiness.learned import load_classifier

ROOT = Path("outputs/fold5-final")
GEOMETRY = Path("outputs/fold5-geometry")
FEATURES = Path("outputs/fold5-behavior")
SOURCE_MODEL = Path("outputs/folds123-extra-trees/behavior.joblib")
TEST_SUBJECTS = {f"{n:02}" for n in range(49, 61)}


def validate_partition(training_subjects, test_subjects):
    if set(training_subjects) != {f"{n:02}" for n in range(1, 37)}:
        raise ValueError("Expected frozen Folds1-3 training participants")
    if set(test_subjects) != TEST_SUBJECTS or set(training_subjects) & set(test_subjects):
        raise ValueError("Final test must contain exactly unseen subjects 49-60")


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    _, metadata = load_classifier(SOURCE_MODEL)
    validate_partition(metadata["subjects"], TEST_SUBJECTS)
    code = [Path(__file__), Path("scripts/prepare_fold.py"), Path("scripts/extract_fold.py"),
            Path("scripts/build_behavior_cohort.py"), Path("scripts/recover_small_face.py"),
            Path("scripts/evaluate_cohort.py"), Path("scripts/improve_model.py"),
            Path("uv.lock"), *Path("src/drowsiness").glob("*.py")]
    protocol = dict(model_sha256=sha(SOURCE_MODEL), metadata_sha256=sha(SOURCE_MODEL.with_suffix(".json")),
        training_subjects=metadata["subjects"], test_subjects=sorted(TEST_SUBJECTS),
        code={str(p): sha(p) for p in code},
        selection="Use the already selected and fitted Folds1-3 Extra Trees model unchanged",
        calibration_seconds=180, minimum_usable_seconds=120,
        recovery_order=["960px default", "1280px for failed alert calibration",
                        "native-eye-detail recovery for remaining failed alert calibration"],
        recovery_scope="All physical recordings of affected participants; selection uses alert prefix only",
        failure_policy="Stop on unresolved calibration; no exclusions, threshold changes or test-based tuning",
        live_model_replaced=False)
    path = ROOT / "frozen-protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen protocol changed; do not silently restart final testing")
    write_json(path, protocol)
    artifact = ROOT / "frozen-model.joblib"
    for source, destination in [(SOURCE_MODEL, artifact),
                                (SOURCE_MODEL.with_suffix(".json"), artifact.with_suffix(".json"))]:
        if destination.exists():
            if sha(source) != sha(destination):
                raise ValueError("Frozen model snapshot changed")
        else:
            shutil.copyfile(source, destination)
    return artifact, protocol


def audit_references(videos):
    observed, failures = {}, {}
    for video in videos:
        if video["label"] != 0:
            continue
        data = frame_array(load_frames(video["feature_file"]))
        try:
            observed[video["subject_id"]] = calibrate_reference(data[data[:, 0] < 180]).observed_seconds
        except ValueError as error:
            failures[video["subject_id"]] = str(error)
    return observed, failures


def retry_resolution(video):
    cv2.setNumThreads(1)
    key = hashlib.sha256(video["video_path"].encode()).hexdigest()[:16]
    output = ROOT / "resolution-fallback" / key
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "video_path", "label"])
        writer.writeheader()
        writer.writerow({k: video[k] for k in writer.fieldnames})
    index = extract(manifest, output, max_width=1280)
    entry = json.loads(index.read_text())["videos"][0]
    return dict(entry, feature_file=str((index.parent / entry["feature_file"]).resolve()))


def retry_native_eye(video):
    # Set in each worker (Windows spawn does not inherit changed module globals).
    recovery.RECOVERY = ROOT / "native-eye-recovery"
    recovery.RECOVERY.mkdir(parents=True, exist_ok=True)
    return recovery.extract_video(video)


def recover(videos):
    audits = []
    for stage, worker in [("resolution_fallback", retry_resolution),
                          ("native_eye_recovery", retry_native_eye)]:
        observed, failed = audit_references(videos)
        audits.append(dict(before_stage=stage, observed=observed, failed=failed))
        write_json(ROOT / "calibration-audits.json", audits)
        if failed:
            write_json(ROOT / "status.json", {"phase": stage, "subjects": sorted(failed)})
            with ProcessPoolExecutor(max_workers=3) as pool:
                replacements = {v["video_path"]: v for v in pool.map(worker,
                    [v for v in videos if v["subject_id"] in failed])}
            videos = [replacements.get(v["video_path"], v) for v in videos]
    observed, failed = audit_references(videos)
    audits.append(dict(before_stage="feature_assembly", observed=observed, failed=failed))
    write_json(ROOT / "calibration-audits.json", audits)
    if failed:
        raise ValueError(f"Frozen calibration policy exhausted; input required: {failed}")
    return videos, audits


def coverage_summary(coverage, windows):
    # Fixed protocol begins at 41s in 30s bins; alert bins before 191s are excluded.
    candidate = sum(max(0, row["candidate_windows"] - (5 if row["label"] == 0 else 0))
                    for row in coverage["coverage"])
    retained = len(windows)
    if not 0 < retained <= candidate:
        raise ValueError("Invalid retained/candidate coverage")
    return dict(candidate_windows_after_enrolment=candidate, retained_windows=retained,
        retained_fraction=retained/candidate, physical_videos=37,
        retained_videos=len({w["video"] for w in windows}),
        videos_without_windows=[r["video"] for r in coverage["coverage"]
                                if r["video"] not in {w["video"] for w in windows}])


def update_report(report):
    path = Path("docs/project-report.md")
    original = path.read_text(encoding="utf-8")
    start, end = "<!-- FOLD5_RESULTS_START -->", "<!-- FOLD5_RESULTS_END -->"
    wm, vm, coverage = report["window_metrics"], report["video_metrics"], report["coverage_summary"]
    section = f'''{start}
### 8.5 Independent Fold5 test

The frozen Extra Trees model trained on subjects 01–36 was tested on subjects
49–60. Model weights, feature order, calibration requirements, and the ordered
recovery procedure were fixed before Fold5 processing. Fold4 supplied development
validation; Fold5 contributed no classifier training or parameter selection.

| Measure | Observed result |
| --- | ---: |
| Test participants | {report['subjects']} |
| Physical videos with predictions | {coverage['retained_videos']} / 37 |
| Retained / eligible windows | {coverage['retained_windows']} / {coverage['candidate_windows_after_enrolment']} |
| Window coverage | {coverage['retained_fraction']:.2%} |
| Three-class window accuracy | **{wm['accuracy']:.2%}** |
| Balanced accuracy | {wm['balanced_accuracy']:.2%} |
| Macro-F1 | {wm['macro_f1']:.4f} |
| Physical-video-majority accuracy | {vm['accuracy']:.2%} |

Confusion matrix (rows actual, columns predicted):

| Class | Alert | Low vigilance | Drowsy |
| --- | ---: | ---: | ---: |
'''
    for name, row in zip(["Alert", "Low vigilance", "Drowsy"], wm["confusion_matrix"], strict=True):
        section += f"| {name} | {row[0]} | {row[1]} | {row[2]} |\n"
    section += f'''
These are offline classification results after explicit personal calibration,
conditional on retained windows. Physical-video scoring treats subject 49's two
drowsy parts separately. This single held-out-fold result does not represent
five-fold cross-validation or live-session accuracy. The complete preprocessing
audit and per-participant scores accompany the saved predictions.

The three-class window result {'meets or exceeds the lower 80% target' if wm['accuracy'] >= .8 else 'does not meet the 80–90% accuracy objective'}.
No model was retuned using this test result.

[Final-test metrics](../outputs/fold5-final/report.json) ·
[Predictions](../outputs/fold5-final/predictions.json) ·
[Frozen protocol](../outputs/fold5-final/frozen-protocol.json)
{end}

'''
    if start in original:
        left, rest = original.split(start, 1)
        _, right = rest.split(end, 1)
        text = left + section.rstrip() + right
    else:
        if original.count("## 9. Conclusion") != 1:
            raise ValueError("Report insertion point missing or ambiguous")
        text = original.replace("## 9. Conclusion", section + "## 9. Conclusion")
    backup = ROOT / "project-report-before-final-test.md"
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def run():
    started = time.monotonic()
    artifact, protocol = freeze()  # Must precede reading any Fold5 contents.
    if (ROOT / "report.json").exists():
        report = json.loads((ROOT / "report.json").read_text())
        if report["frozen_protocol"] != protocol:
            raise ValueError("Existing final report belongs to a different protocol")
        update_report(report)
        write_json(ROOT / "status.json", {"phase": "complete", "report": str(ROOT / "report.json"),
                                           "live_model_replaced": False})
        return
    write_json(ROOT / "status.json", {"phase": "archive_verification_and_extraction"})
    prepare(5)
    rows = read_manifest(Path("data/fold5-manifest.csv"))
    validate_partition(protocol["training_subjects"], {r["subject_id"] for r in rows})
    if len(rows) != 37:
        raise ValueError("Expected 37 physical videos, including both subject49 drowsy parts")
    write_json(ROOT / "status.json", {"phase": "mediapipe_extraction", "videos": 37})
    with (ROOT / "geometry.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "scripts/extract_fold.py", "--manifest", "data/fold5-manifest.csv",
                        "--output", str(GEOMETRY), "--workers", "4"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    index_path = GEOMETRY / "features/index.json"
    index = json.loads(index_path.read_text())
    videos = [dict(v, feature_file=str((index_path.parent / v["feature_file"]).resolve()))
              for v in index["videos"]]
    write_json(ROOT / "status.json", {"phase": "calibration_audit"})
    videos, audits = recover(videos)
    final_index = ROOT / "features-index.json"
    write_json(final_index, dict(settings=dict(index["settings"], recovery_policy="frozen-protocol.json"), videos=videos))
    write_json(ROOT / "status.json", {"phase": "behavior_features"})
    build(final_index, FEATURES)
    write_json(FEATURES / "preprocessing.json", {"protocol": protocol, "calibration_audits": audits})
    write_json(ROOT / "status.json", {"phase": "frozen_final_test"})
    windows, x, names, signature = load_cohorts([FEATURES])
    classifier, metadata = load_classifier(artifact)
    validate_partition(metadata["subjects"], {w["subject"] for w in windows})
    if metadata["feature_names"] != names or sha(artifact) != protocol["model_sha256"]:
        raise ValueError("Frozen model or feature order mismatch")
    prediction = classifier.predict(x)
    truth = np.array([w["label"] for w in windows])
    coverage = json.loads((FEATURES / "coverage.json").read_text())
    report = dict(summarize(windows, truth, prediction),
        protocol="Independent Fold5 test of frozen Folds1-3 Extra Trees after known-alert calibration",
        frozen_protocol=protocol, signature=signature, calibration_audits=audits,
        coverage_summary=coverage_summary(coverage, windows), coverage=coverage,
        elapsed_seconds=time.monotonic()-started, live_model_replaced=False)
    save_predictions(ROOT, windows, prediction)
    write_json(ROOT / "report.json", report)
    update_report(report)
    write_json(ROOT / "status.json", {"phase": "complete", "report": str(ROOT / "report.json"),
        "window_accuracy": report["window_metrics"]["accuracy"], "live_model_replaced": False,
        "elapsed_seconds": time.monotonic()-started})
    print(json.dumps(report["window_metrics"]), flush=True)


if __name__ == "__main__":
    try:
        with threadpool_limits(limits=1):
            run()
    except Exception as error:
        write_json(ROOT / "status.json", {"phase": "failed", "error": repr(error)})
        raise
