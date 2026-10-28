"""Time-based signal processing. No camera, UI, or ML dependencies here."""

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    timestamp: float
    ear_left: float = 0.0
    ear_right: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    brightness: float = 0.0
    valid: bool = True
    reason: str = ""

    @property
    def ear(self):
        return (self.ear_left + self.ear_right) / 2


@dataclass(frozen=True)
class Baseline:
    ear_left: float = 0.30
    ear_right: float = 0.30
    mar: float = 0.05
    pitch: float = 0.0
    blink_seconds: float = 0.20
    blink_rate: float = 15.0
    personalised: bool = False


@dataclass(frozen=True)
class Config:
    calibration_seconds: float = 45.0
    window_seconds: float = 60.0
    warmup_seconds: float = 10.0
    max_gap_seconds: float = 0.5
    min_coverage: float = 0.70
    closure_ratio: float = 0.65
    fixed_ear: float = 0.20
    slow_closure_seconds: float = 0.50
    critical_closure_seconds: float = 1.50
    yawn_seconds: float = 1.50
    fatigue_threshold: float = 35.0
    critical_threshold: float = 65.0

    def __post_init__(self):
        if not (0 < self.calibration_seconds <= 120 and self.window_seconds > 0):
            raise ValueError("Invalid calibration or window duration")
        if not (0 < self.min_coverage <= 1 and 0 < self.closure_ratio < 1):
            raise ValueError("Invalid coverage or closure ratio")


class Calibrator:
    """Accumulate observed, usable time only; never infer that a person is rested."""

    def __init__(self, seconds=45.0, max_gap=0.5):
        self.seconds = seconds
        self.max_gap = max_gap
        self.frames = []
        self.elapsed = 0.0
        self.previous = None

    def update(self, frame):
        usable = frame.valid and abs(frame.yaw) < 20 and abs(frame.pitch) < 25
        if usable:
            if self.previous is not None:
                dt = frame.timestamp - self.previous
                if 0 < dt <= self.max_gap:
                    self.elapsed += dt
            self.previous = frame.timestamp
            self.frames.append(frame)
        else:
            self.previous = None
        return min(1.0, self.elapsed / self.seconds)

    def finish(self):
        if self.elapsed + 1e-6 < self.seconds or len(self.frames) < 20:
            raise ValueError("Calibration needs more usable, forward-facing footage")
        left = float(np.percentile([f.ear_left for f in self.frames], 80))
        right = float(np.percentile([f.ear_right for f in self.frames], 80))
        if min(left, right) < 0.12 or max(left, right) > 0.6:
            raise ValueError("Eye baseline is implausible. Improve lighting and recalibrate")
        durations = []
        start = None
        previous = None
        for f in self.frames:
            if previous is not None and f.timestamp - previous > self.max_gap:
                start = None
            closed = f.ear_left < left * 0.65 and f.ear_right < right * 0.65
            if closed and start is None:
                start = f.timestamp
            if not closed and start is not None:
                duration = f.timestamp - start
                if 0.08 <= duration <= 1.0:
                    durations.append(duration)
                start = None
            previous = f.timestamp
        return Baseline(
            ear_left=left,
            ear_right=right,
            mar=float(np.median([f.mar for f in self.frames])),
            pitch=float(np.median([f.pitch for f in self.frames])),
            blink_seconds=float(np.median(durations)) if durations else 0.20,
            blink_rate=len(durations) * 60 / self.elapsed,
            personalised=True,
        )


class Monitor:
    def __init__(self, config=None, baseline=None):
        self.config = config or Config()
        self.baseline = baseline
        self.calibrator = Calibrator(self.config.calibration_seconds, self.config.max_gap_seconds)
        self.previous = None
        self.started = None
        self.intervals = deque()
        self.closures = deque()
        self.blinks = deque()
        self.yawns = deque()
        self.nods = deque()
        self.closed_since = None
        self.yawn_since = None
        self.nod_since = None
        self.yawn_counted = False
        self.score = 0.0
        self.level = "alert"
        self.recovery_since = None

    def _break_events(self, timestamp):
        if self.closed_since is not None:
            if timestamp - self.closed_since >= self.config.slow_closure_seconds:
                self.closures.append((self.closed_since, timestamp))
        self.closed_since = self.yawn_since = self.nod_since = None
        self.yawn_counted = False

    def update(self, frame):
        c = self.config
        t = frame.timestamp
        if not math.isfinite(t) or t < 0:
            raise ValueError("Timestamps must be finite and nonnegative")
        if self.previous is not None and t <= self.previous.timestamp:
            raise ValueError("Timestamps must increase strictly")
        if frame.valid and not all(
            math.isfinite(v)
            for v in (
                frame.ear_left,
                frame.ear_right,
                frame.mar,
                frame.pitch,
                frame.yaw,
                frame.roll,
                frame.brightness,
            )
        ):
            raise ValueError("Frame features must be finite")
        old = self.previous
        self.previous = frame
        dt = t - old.timestamp if old else 0.0
        continuous = bool(old and old.valid and frame.valid and 0 < dt <= c.max_gap_seconds)
        if not continuous:
            self._break_events(old.timestamp if old else t)
            self.recovery_since = None
        result = {
            "timestamp": t,
            "state": "unknown",
            "score": None,
            "reason": frame.reason,
            "calibration": 1.0 if self.baseline else 0.0,
            "ear": round(frame.ear, 3) if frame.valid else None,
            "mar": round(frame.mar, 3) if frame.valid else None,
            "pitch": round(frame.pitch, 1) if frame.valid else None,
            "brightness": round(frame.brightness, 1),
            "closure_ratio": None,
            "closure_seconds": 0.0,
            "blink_seconds": 0.0,
            "blinks_per_minute": 0.0,
            "yawns_per_minute": 0.0,
            "nods_per_minute": 0.0,
            "coverage": 0.0,
            "observed_seconds": 0.0,
            "components": {},
        }
        if self.baseline is None:
            progress = self.calibrator.update(frame)
            result.update(
                state="calibrating",
                calibration=progress,
                reason="Look ahead, relax your mouth, and blink naturally",
            )
            if not frame.valid:
                result["reason"] = frame.reason + ". Calibration paused"
            if progress >= 1:
                self.baseline = self.calibrator.finish()
                self.started = t
                result["reason"] = "Calibration complete. Gathering a rolling history"
            return result
        b = self.baseline
        if self.started is None:
            self.started = t
        cutoff = t - c.window_seconds
        for items in (self.intervals, self.closures):
            while items and items[0][1] <= cutoff:
                items.popleft()
        for items in (self.blinks, self.yawns, self.nods):
            while items and (items[0][0] if isinstance(items[0], tuple) else items[0]) <= cutoff:
                items.popleft()
        if continuous:
            self.intervals.append((old.timestamp, t))
        observed = sum(end - max(start, cutoff) for start, end in self.intervals)
        span = min(c.window_seconds, t - self.started)
        coverage = min(1.0, observed / span) if span > 0 else 0.0
        result.update(coverage=round(coverage, 3), observed_seconds=round(observed, 2))
        if not frame.valid:
            result["reason"] = frame.reason or "Face tracking unavailable"
            return result
        left = b.ear_left * c.closure_ratio if b.personalised else c.fixed_ear
        right = b.ear_right * c.closure_ratio if b.personalised else c.fixed_ear
        closed = frame.ear_left < left and frame.ear_right < right
        if closed and self.closed_since is None:
            self.closed_since = t
        if not closed and self.closed_since is not None:
            duration = t - self.closed_since
            if 0.08 <= duration <= 1.0:
                self.blinks.append((t, duration))
            if duration >= c.slow_closure_seconds:
                self.closures.append((self.closed_since, t))
            self.closed_since = None
        closure = t - self.closed_since if self.closed_since is not None else 0.0
        yawning = frame.mar > max(0.45, b.mar + 0.30)
        if yawning:
            if self.yawn_since is None:
                self.yawn_since = t
            if t - self.yawn_since >= c.yawn_seconds and not self.yawn_counted:
                if not self.yawns or t - self.yawns[-1] >= 3:
                    self.yawns.append(t)
                self.yawn_counted = True
        else:
            self.yawn_since = None
            self.yawn_counted = False
        # A pitch excursion followed by recovery is a nod candidate, not proof of sleep.
        if abs(frame.pitch - b.pitch) > 15:
            if self.nod_since is None:
                self.nod_since = t
        elif self.nod_since is not None:
            duration = t - self.nod_since
            if 0.3 <= duration <= 3 and (not self.nods or t - self.nods[-1] > 2):
                self.nods.append(t)
            self.nod_since = None
        closed_time = sum(end - max(start, cutoff) for start, end in self.closures)
        if closure >= c.slow_closure_seconds:
            closed_time += t - max(self.closed_since, cutoff)
        ratio = min(1.0, closed_time / observed) if observed else 0.0
        blink = float(np.mean([d for _, d in self.blinks])) if self.blinks else 0.0
        denominator = max(observed, c.warmup_seconds)
        blink_rate = len(self.blinks) * 60 / denominator
        yawn_rate = len(self.yawns) * 60 / denominator
        nod_rate = len(self.nods) * 60 / denominator
        # Explicit engineering heuristics, not a fitted probability or clinical cutoff.
        components = {
            "eye_closure": min(1.0, ratio / 0.30) * 55,
            "slow_blinks": min(1.0, max(0.0, blink - max(0.25, b.blink_seconds)) / 0.35) * 20,
            "yawns": min(1.0, yawn_rate / 3) * 15,
            "head_movement": min(1.0, nod_rate / 4) * 10,
        }
        raw = sum(components.values())
        alpha = 1 - math.exp(-min(dt, c.max_gap_seconds) / 2)
        self.score += alpha * (raw - self.score)
        critical_closure = closure >= c.critical_closure_seconds
        if critical_closure:
            self.score = max(85.0, self.score)
        result.update(
            closure_ratio=round(ratio, 4),
            closure_seconds=round(closure, 3),
            blink_seconds=round(blink, 3),
            blinks_per_minute=round(blink_rate, 2),
            yawns_per_minute=round(yawn_rate, 2),
            nods_per_minute=round(nod_rate, 2),
            components={k: round(v, 2) for k, v in components.items()},
        )
        if not critical_closure and (observed < c.warmup_seconds or coverage < c.min_coverage):
            result.update(
                state="warming" if coverage >= c.min_coverage else "unknown",
                reason="Gathering reliable history"
                if coverage >= c.min_coverage
                else "Insufficient tracking coverage. Face the camera in better light",
            )
            return result
        target = (
            "critical"
            if self.score >= c.critical_threshold
            else ("fatigued" if self.score >= c.fatigue_threshold else "alert")
        )
        rank = {"alert": 0, "fatigued": 1, "critical": 2}
        if rank[target] >= rank[self.level]:
            self.level = target
            self.recovery_since = None
        else:
            boundary = c.critical_threshold if self.level == "critical" else c.fatigue_threshold
            if self.score < boundary - 5:
                if self.recovery_since is None:
                    self.recovery_since = t
                if t - self.recovery_since >= 3:
                    self.level = target
                    self.recovery_since = None
            else:
                self.recovery_since = None
        messages = {
            "alert": "No elevated fatigue cues in the observed window",
            "fatigued": "Fatigue cues are rising. Plan a rest break",
            "critical": "Sustained fatigue cues. Stop safely and rest",
        }
        result.update(state=self.level, score=round(self.score, 1), reason=messages[self.level])
        return result

    def metadata(self):
        return {
            "config": asdict(self.config),
            "baseline": asdict(self.baseline) if self.baseline else None,
        }
