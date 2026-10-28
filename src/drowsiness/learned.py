"""Streaming inference for locally trained behavioural models."""

import hashlib
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from .behavior import calibrate_reference, extract_behavior, frame_array, regular_runs
from .core import Baseline, Monitor
from .expressions import expression_features, expression_reference

DEFAULT_CLASSIFIER = Path("models/behavior.joblib")


def load_classifier(path=DEFAULT_CLASSIFIER):
    """Load a trusted, locally generated artifact. Never download pickle models."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema") != 1 or metadata.get("variant") not in {"raw", "enrolled"}:
        raise ValueError("Unsupported behavioural classifier artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Classifier checksum does not match its training metadata")
    return joblib.load(path), metadata


class LearnedMonitor:
    """180s alert enrolment, then causal classification every 5s with 30/90s history.

    The ordinal display value 0/50/100 is a class index, not a probability.
    The separate prolonged-closure override retains the immediate existing alarm.
    """

    def __init__(self, model, metadata):
        self.model, self.training = model, metadata
        self.reference = None
        self.prefix = []
        self.buffer = deque()
        self.started = None
        self.ready_at = None
        self.last_prediction = None
        self.predicted_at = -np.inf
        self.monitor = Monitor(baseline=Baseline())
        self.previous = None
        self.expression_buffer = deque()
        self.expression_prefix = []
        self.expression_reference = None
        self.last_expression_time = -np.inf

    def resume(self):
        """Keep the alert reference, but never join events across a camera pause."""
        if self.training["variant"] == "enrolled" and self.reference is None:
            raise ValueError("Complete calibration before pausing")
        self.monitor = Monitor(config=self.monitor.config, baseline=self.monitor.baseline)
        self.buffer.clear()
        self.expression_buffer.clear()
        self.ready_at = None
        self.last_prediction = None
        self.predicted_at = -np.inf
        self.last_expression_time = -np.inf

    def update(self, frame, expression=None):
        t = frame.timestamp
        if self.previous is not None and t <= self.previous:
            raise ValueError("Timestamps must increase strictly")
        self.previous = t
        if self.started is None:
            self.started = t
        result = self.monitor.update(frame)
        result.update(
            scoring="behavioral_classifier",
            model_label=None,
            score_kind="ordinal class index; not confidence or probability",
        )
        needs_expressions = "expression_names" in self.training
        if needs_expressions and t - self.last_expression_time >= 0.5 - 1e-6:
            count = len(self.training["expression_names"])
            valid = (
                frame.valid
                and expression is not None
                and len(expression) == count
                and np.isfinite(expression).all()
            )
            row = [t, float(valid)] + (list(expression) if valid else [0.0] * count)
            if self.reference is None and t - self.started < 180:
                self.expression_prefix.append(row)
            else:
                self.expression_buffer.append(row)
            self.last_expression_time = t
        while self.expression_buffer and self.expression_buffer[0][0] < t - 91:
            self.expression_buffer.popleft()
        enrolled = self.training["variant"] == "enrolled"
        if enrolled and self.reference is None:
            # Keep the exact first 180s; poor tracking is reported, not silently extended.
            elapsed = t - self.started
            if elapsed < 180:
                self.prefix.append(frame)
                result.update(
                    state="calibrating",
                    score=None,
                    calibration=elapsed / 180,
                    reason="Remain rested and blink naturally during the 3-minute baseline",
                )
                if not frame.valid:
                    result["reason"] = frame.reason or "Improve face tracking for calibration"
                return result
            data = frame_array(self.prefix)
            self.reference = calibrate_reference(data)
            if needs_expressions:
                self.expression_reference = expression_reference(np.asarray(self.expression_prefix))
                self.expression_prefix.clear()
            valid = data[(data[:, 7] > 0) & np.isfinite(data).all(axis=1)]
            baseline = Baseline(
                ear_left=float(np.quantile(valid[:, 1], 0.8)),
                ear_right=float(np.quantile(valid[:, 2], 0.8)),
                mar=float(np.median(valid[:, 3])),
                pitch=float(np.median(valid[:, 4])),
                personalised=True,
            )
            self.monitor = Monitor(baseline=baseline)
            self.prefix.clear()
            self.ready_at = t
            result = self.monitor.update(frame)
        if self.ready_at is None:
            self.ready_at = t
        self.buffer.append(frame)
        while self.buffer and self.buffer[0].timestamp < t - 91:
            self.buffer.popleft()
        result.update(
            calibration=1.0,
            scoring="behavioral_classifier",
            model_label=None,
            score_kind="ordinal class index; not confidence or probability",
        )
        # Invalid tracking cannot reuse a stale prediction or produce an alert estimate.
        if not frame.valid:
            result.update(state="unknown", score=None)
            return result
        if result["closure_seconds"] >= 1.5:
            result.update(
                state="critical",
                score=100.0,
                reason="Prolonged eye closure detected; stop safely and rest",
                scoring="prolonged_closure_override",
            )
            return result
        if t - self.ready_at < 41:
            result.update(
                state="warming", score=None, reason="Gathering 41 seconds of recent history"
            )
            return result
        data = frame_array(self.buffer)
        recent = data[(data[:, 0] >= t - 30) & (data[:, 0] < t)]
        observed = sum(len(run) / 25 for run in regular_runs(recent))
        if observed < 21:
            result.update(
                state="unknown", score=None, reason="Insufficient recent tracking coverage"
            )
            return result
        expression_data = None
        if needs_expressions:
            expression_data = np.asarray(self.expression_buffer)
            usable = expression_data[
                (expression_data[:, 0] >= t - 30) & (expression_data[:, 0] < t), 1
            ].sum()
            if usable < 30:
                result.update(
                    state="unknown",
                    score=None,
                    reason="Insufficient recent facial-expression tracking",
                )
                return result
        if t - self.predicted_at >= 5 or self.last_prediction is None:
            x, names = extract_behavior(data, t, self.reference, self.ready_at)
            if names != self.training["feature_names"]:
                raise ValueError("Model feature schema has changed; retrain the classifier")
            if needs_expressions:
                extra, keys = expression_features(
                    expression_data,
                    t,
                    self.expression_reference,
                    self.training["expression_names"],
                    self.ready_at,
                )
                if keys != self.training["expression_feature_names"]:
                    raise ValueError(
                        "Expression feature schema has changed; retrain the classifier"
                    )
                x = np.r_[x, extra]
            self.last_prediction = int(self.model.predict(x[None])[0])
            self.predicted_at = t
        label = self.last_prediction
        state = ("alert", "fatigued", "critical")[label]
        result.update(
            state=state,
            score=float(label * 50),
            model_label=label,
            prediction_age_seconds=round(t - self.predicted_at, 2),
            reason=(
                "Recent behaviour classified as alert",
                "Recent behaviour suggests low vigilance. Plan a rest break",
                "Recent behaviour suggests drowsiness. Stop safely and rest",
            )[label],
        )
        return result

    def metadata(self):
        return dict(
            self.monitor.metadata(),
            scoring="behavioral_classifier",
            classifier=self.training["estimator"],
            variant=self.training["variant"],
            calibration_seconds=180 if self.training["variant"] == "enrolled" else 0,
            model_sha256=self.training["sha256"],
            training_subjects=self.training["subjects"],
            evaluation="Fold1 development benchmark; new-cohort validation pending",
            score_kind="ordinal class index; not confidence or probability",
        )
