"""Guard against cohort overlap and frozen-test contamination."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from drowsiness.dataset import write_json


@pytest.fixture
def cohort_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_cohort")


def cohort(path, subject):
    path.mkdir()
    windows = [
        dict(
            subject=subject,
            video=f"{subject}/{label}.csv",
            label=label,
            start=191 if label == 0 else 11,
            end=221 if label == 0 else 41,
        )
        for label in range(3)
    ]
    write_json(path / "windows.json", windows)
    write_json(path / "feature-names.json", {"enrolled": ["a", "b"]})
    np.savez_compressed(path / "features.npz", enrolled=np.arange(6).reshape(3, 2))
    return path


def test_duplicate_subjects_between_cohorts_rejected(tmp_path, cohort_tools):
    first, second = cohort(tmp_path / "first", "01"), cohort(tmp_path / "second", "01")
    with pytest.raises(ValueError, match="Overlapping cohort subjects"):
        cohort_tools.load_cohorts([first, second])


def test_incompatible_feature_order_rejected(tmp_path, cohort_tools):
    first, second = cohort(tmp_path / "first", "01"), cohort(tmp_path / "second", "13")
    write_json(second / "feature-names.json", {"enrolled": ["b", "a"]})
    with pytest.raises(ValueError, match="Incompatible feature names"):
        cohort_tools.load_cohorts([first, second])


def test_frozen_model_requires_new_subjects_and_preserves_predictions(tmp_path, cohort_tools):
    path = cohort(tmp_path / "cohort", "13")
    model_path = tmp_path / "model.joblib"
    model = DummyClassifier(strategy="constant", constant=2).fit(
        [[0, 0], [1, 1], [2, 2]], [0, 1, 2]
    )
    joblib.dump(model, model_path)
    metadata = dict(
        schema=1,
        variant="enrolled",
        subjects=["01"],
        feature_names=["a", "b"],
        sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    write_json(model_path.with_suffix(".json"), metadata)
    output = tmp_path / "result"
    args = argparse.Namespace(output=output, cohorts=[path], frozen=model_path)
    cohort_tools.run(args)
    report = json.loads((output / "report.json").read_text())
    assert report["window_metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert report["test_subjects"] == ["13"]
    assert [r["prediction"] for r in json.loads((output / "predictions.json").read_text())] == [
        2,
        2,
        2,
    ]
    metadata["subjects"] = ["13"]
    write_json(model_path.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match="overlapping test subjects"):
        cohort_tools.run(args)

