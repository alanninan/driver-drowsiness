# Driver Drowsiness

A local webcam research application with personal calibration, time-based fatigue cues,
graded warnings, and subject-disjoint evaluation tooling. Built with Python 3.12 and uv.

The repository contains application code, training scripts and tests. Local reports
in `docs/`, datasets, trained models, experiment outputs and report-generation files
are excluded from Git. Keep `uv.lock` committed for reproducible dependencies.

**Current model:** an Extra Trees classifier selected by grouped cross-validation,
using full-rate blink shape, eye/mouth distributions, head motion and personal
normalization. The live app uses this trained model when `models/behavior.joblib`
is present. It needs a **3-minute known-alert calibration**, then 41 seconds of history.
The original rule engine remains available with `--rules` and for the simulated session.

**Deployment model:** Extra Trees trained on all five UTA-RLDD folds: 60
participants, 2,966 usable windows, 464 features and 300 trees. The complete usable
dataset is used for fitting, with no separate test partition for this model.
Real-time accuracy has not been measured. These describe the existing local model;
its weights and training summary are not included in the repository. Historical
evaluation artifacts remain locally in `outputs/`; they describe earlier models.

## Train on Folds 1–4 and test on Fold 5

The training code uses subjects 01–48 for fitting and subjects 49–60 only for
scoring. Existing features are reused. The live model is never replaced by these
commands. These commands require locally prepared feature caches in
`outputs/fold1-improved`, `outputs/fold2-resolution-behavior`,
`outputs/folds34-behavior` and `outputs/fold5-behavior`; they are not supplied by a
fresh clone. No new training has been run as part of this code change.

The local Fold5 behavioural feature cache was removed on request. Testing requires
rebuilding it first from the retained geometry index:

```powershell
uv run --no-sync python scripts/build_behavior_cohort.py --index outputs/fold5-final/features-index.json --output outputs/fold5-behavior
```

Obsolete visual-model experiments, ensemble experiments and one-off reporting
scripts were removed from the maintained tree. Local recovery copies are kept in
`outputs/legacy-code-2026-10-28/` (ignored by Git). Active CLI functionality and
training/preparation dependencies are retained.

When ready to train a separate candidate and calculate its metrics:

```powershell
uv run --no-sync python scripts/train_holdout.py
```

The legacy `scripts/train_full_dataset.py` command now runs this same workflow.
It no longer trains on all five folds. The candidate is saved under
`outputs/folds1234-extra-trees/`; existing output directories are protected from
overwriting. Use `--output outputs/another-run` for a separate run.

For a clean, screenshot-ready terminal display after training:

```powershell
Clear-Host
uv run --no-sync python scripts/evaluate_metrics.py --from-report outputs/folds1234-extra-trees/evaluation/metrics.json
```

The display includes accuracy, macro precision/recall/F1, per-class and weighted
scores, supports, and a labelled confusion matrix. It fits an 80-column terminal.
The same display is saved as `evaluation/summary.txt` alongside `metrics.json`.
To recalculate from a candidate without retraining:

```powershell
uv run --no-sync python scripts/evaluate_metrics.py --model outputs/folds1234-extra-trees/behavior.joblib --output outputs/folds1234-metrics-refresh
```

The evaluator rejects the live all-folds model because it already saw Fold 5.
Fold 5 has also been evaluated in earlier experiments: excluding it from this
candidate's fitting does not make it a new, untouched research test. Results are
offline classification after calibration, not live-camera accuracy.

## Run

From this folder in PowerShell:

```powershell
uv sync
uv run --no-sync drowsiness download-model
uv run --no-sync drowsiness serve
```

On a fresh clone, `download-model` retrieves the official MediaPipe face model.
The trained Extra Trees files are separate: place a trusted local
`behavior.joblib` and its matching `behavior.json` in `models/` to enable learned
classification. Without them, the application uses its rule engine and the
simulated session remains available. Never load untrusted joblib/pickle files.

Open **http://127.0.0.1:8000** in Edge or Chrome. Click **Try a simulated session** to
exercise the UI without a camera, or **Start camera** for real detection.

Critical alert sound is enabled by default. Starting a session enables browser
audio; critical states play a short tone at most every five seconds. Use **Test
sound** in Session setup to check your speakers or resume blocked playback.
Uncheck **Critical alert sound** to mute it. Audio failures are shown beside the
setting and do not prevent camera monitoring.

1. Start while rested, with one face visible, facing the camera in even light.
2. Allow camera access. Blink naturally, keep your mouth relaxed, and remain in view.
3. With the trained model, remain rested for 3 minutes; at least 120 seconds must have
   usable tracking. Poor calibration ends the session with an explanation.
4. Another 41 seconds builds the first history. The model updates every 5 seconds;
   prolonged eye closure can trigger critical sooner. Missing tracking hides predictions.
5. Stop the session to release the camera. Export JSON if desired; no recording is automatic.

After calibration, **Pause camera** disables capture and alerts while keeping the
connection and personal baseline. **Resume camera** retains calibration and gathers
fresh history (41 seconds for the learned model). Keep the page and server open;
Stop, a reload, or a disconnection ends the session and requires calibration again.

The export contains at most the latest 3,600 one-second measurements, settings, and the
session baseline. It contains no images. The live stream remains in memory on this machine.
The server binds to loopback only. A phone recording can be analysed through the CLI;
a phone connected over the network is not a supported live client in this version.

This is an experimental research prototype, not a validated driving-safety device.
Do not test it while driving. With the classifier, 0 / 50 / 100 are ordinal class values
(alert / low vigilance / drowsy), not probabilities or confidence percentages. The separate
immediate eye-closure override and live-camera behavior have not been accuracy-validated.

## Commands

```powershell
uv run drowsiness --help
uv run drowsiness doctor
uv run drowsiness analyze .\recording.mp4 --output outputs\recording.jsonl
uv run drowsiness analyze .\recording.mp4 --fixed --clahe
uv run pytest -q
uv run ruff check src tests
```

Video replay uses video timestamps, not processing speed. Without `--fixed`, its first
footage is treated as a **known-alert calibration**; only use that mode when this
assumption is true. The trained model uses 180 seconds. `--rules` restores the legacy
calibrated rules, and `--fixed` uses uncalibrated rules. Replay saves JSONL and metadata.

### Reproduce the improved model

```powershell
uv run --no-sync python scripts/improve_model.py
uv run --no-sync python scripts/train_behavior.py
# Separate CUDA experiment, not the selected live classifier:
uv run --no-sync python scripts/evaluate_behavior_gru.py
```

These commands use the supplied Fold1 feature cache. The first performs nested LOSO;
the second selects an estimator using grouped CV and fits the deployment artifact on
all development subjects. That final fitted artifact has no independent test score.
Keep the model's adjacent JSON metadata file: loading verifies its checksum and schema.
The experiment plan and all alternative results are retained under `docs/` and `outputs/`.

## Dataset workflow (when the data is available)

Obtain UTA-RLDD from the [dataset authors](https://sites.google.com/view/utarldd/home).
No command in this project downloads the dataset. Keep the original subject identifiers.

```powershell
uv run drowsiness manifest D:\datasets\UTA-RLDD --output data\manifest.csv
uv run drowsiness extract data\manifest.csv --output data\features
uv run drowsiness evaluate data\features\index.json --output outputs\enrolled --model rf
uv run drowsiness evaluate data\features\index.json --output outputs\fixed --protocol fixed
```

The manifest generator recognises `subject/0.mp4`, `subject/5.mp4`, and `subject/10.mp4`
(also MOV, AVI, MKV). For another layout, create a UTF-8 CSV with these exact columns:

```csv
subject_id,video_path,label
01,D:/datasets/UTA-RLDD/01/0.mp4,0
01,D:/datasets/UTA-RLDD/01/5.mp4,5
01,D:/datasets/UTA-RLDD/01/10.mp4,10
```

Labels are the original **0 / 5 / 10**, internally mapped to alert / low vigilance / drowsy.
Relative paths resolve against the manifest directory. Duplicate source videos are rejected.
Extraction writes one numerical CSV per video and an `index.json`. It resumes matching
caches using source size/mtime, model hash, settings, and schema. Use `--fps 30` (default)
to preserve blink timing; coarse subsampling cannot reliably measure short blinks.

**Enrolled evaluation (default):** each subject must have exactly one alert clip. Its
first 60 wall-clock seconds supplies at least 45 usable seconds for a personal baseline.
That prefix is excluded from both personal and fixed-rule scoring. Each video then
starts with fresh temporal state and a warmup. The known-alert enrolment requirement is
part of the experiment, including for the held-out subject; this is not zero-shot LOSO.

**Fixed evaluation:** no personal enrolment; uses fixed thresholds. Because it retains
the alert prefix, do not compare its headline scores directly with the enrolled run.
Use `fixed_rules` versus `personal_rules` **within the same enrolled report** for the
paired personalisation comparison.

Windows are non-overlapping sequences of 30 one-second snapshots. Each full window
inherits its source video's label. Any unavailable snapshot excludes the window; reports
retain candidate/retained counts per video. LOSO keeps all windows from one subject
out of model training. RF uses sequence means/stds. GRU normalization is fit exclusively
on training subjects. Hyperparameters are fixed; no test-fold tuning or early stopping.

Output: `report.json` with fold membership, coverage, confusion matrices, accuracy,
balanced accuracy, macro-F1, mean subject macro-F1, and video-majority metrics;
`predictions.json` with subject, video, time interval and per-method predictions.
Always report coverage next to accuracy. Additional protocol notes, when available
locally, are in `docs/evaluation.md` (excluded from Git).

### Low-light experiment

```powershell
uv run drowsiness extract data\manifest.csv --clahe --output data\features-clahe
uv run drowsiness evaluate data\features-clahe\index.json --output outputs\clahe
```

Use identical manifests/folds. Compare tracking retention as well as classification,
and compare common retained windows for a paired result. CLAHE cannot restore information
lost to darkness, glare, or occlusion. UTA's video labels do not encode illumination;
any lighting subgroup annotation must be recorded separately and defined before testing.

## RTX 5050 / optional GRU

MediaPipe's Python landmark pipeline runs on CPU here. The light rule engine and RF
do not need a GPU. The optional 32-unit GRU uses PyTorch CUDA for LOSO training:

```powershell
uv sync
uv run --extra gpu drowsiness doctor --gpu
uv run --extra gpu drowsiness evaluate data\features\index.json --model gru --device cuda --epochs 20 --output outputs\gru
```

The `gpu` extra selects official PyTorch CUDA 12.8 wheels on Windows/Linux, supporting
Blackwell hardware. It is a multi-gigabyte dependency and is optional for the application.
Use `--device cpu` to train without CUDA, or `auto` to choose an available GPU.
`doctor --gpu` executes a real GRU forward/backward pass, not just a device lookup.
Keep `--extra gpu` on uv commands that need torch; normal uv sync/run may remove unused extras.

Local setup verified on 14 October 2026: **PyTorch 2.10.0+cu128**, CUDA runtime 12.8,
RTX 5050 capability 12.0. A real GRU forward/backward test and train/predict cycle passed,
and all 12 raw-signal GRU evaluation folds ran on CUDA. That experiment scored 50.52%
window accuracy and did not beat the calibrated feature pipeline. See `outputs/gpu/result.json`.
Use `uv run --no-sync ...` to preserve the already installed optional GPU environment.

## Structure

```text
src/drowsiness/core.py        Calibration and time-based rule engine
src/drowsiness/vision.py      MediaPipe, pixel-space EAR/MAR, pose, quality gates
src/drowsiness/server.py      Local FastAPI/WebSocket service
src/drowsiness/static/        Browser camera, dashboard, charts and session export
src/drowsiness/dataset.py     Manifest and cached feature extraction
src/drowsiness/evaluation.py  Window construction, paired baselines and LOSO reports
src/drowsiness/temporal.py    Optional CUDA GRU benchmark
src/drowsiness/behavior.py    Full-rate causal behavioural features and alert reference
src/drowsiness/learned.py     Trained classifier loading and streaming inference
src/drowsiness/expressions.py Optional expression-feature research branch
tests/                       Signal, leakage, geometry, and service regression tests
```

No credentials, telemetry, cloud processing, or dataset are required for the demo.
The official face model is downloaded separately; its provenance is documented in research notes.

## Troubleshooting

- **Camera denied:** allow camera access for localhost in the browser; close other camera apps.
- **No face / low coverage:** bring the face closer, face forward, remove glasses glare, improve lighting.
- **Calibration fails:** improve tracking and restart while rested. The trained model
  requires 120 usable seconds within its 180-second prefix. Baselines never adapt silently.
- **Missing model:** run `uv run drowsiness download-model` from this project directory.
- **Port in use:** `uv run drowsiness serve --port 8001`.
- **Windows executable is in use during uv sync:** stop the dashboard server before
  changing dependencies, then restart it after syncing.
- **CUDA unavailable:** run the GPU doctor with `--extra gpu`; confirm a current NVIDIA driver.
- **Failed enrolment:** inspect the subject's alert prefix; do not silently calibrate on a
  drowsy clip. Correct the manifest or run the explicitly fixed protocol.
