# Driver Drowsiness

Local research prototype for a student/researcher demonstrating webcam fatigue estimation.
User-provided brief: personal calibration, graded warnings, rolling facial features, and
fair UTA-RLDD evaluation. The user approved implementing the whole app and evaluation
pipeline while deferring dataset download and measured accuracy benchmarking.

Workflow: open local dashboard, start camera while rested, calibrate for 45 valid seconds,
observe signals and fatigue estimates, stop, optionally export numerical session history.
Recorded video analysis and dataset extraction/evaluation are command-line workflows.
Frames stay in memory on the local machine. No recording or telemetry by default.

Implementation choices: browser camera plus Python local server; English interface;
desktop-first with a responsive layout; CPU landmark inference; optional CUDA GRU training.
These routine choices follow the approved brief and are not additional user requirements.
This is a research prototype, with heuristic scores and no measured accuracy claim.
