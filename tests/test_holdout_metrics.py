import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_metrics import calculate_metrics, format_summary, validate_holdout
from train_holdout import TRAIN_COHORTS, run, validate_training


def test_all_fold_model_and_missing_test_subjects_are_rejected():
    train = {f"{s:02}" for s in range(1, 49)}
    test = {f"{s:02}" for s in range(49, 61)}
    validate_holdout(train, test)
    with pytest.raises(ValueError):
        validate_holdout(train | test, test)
    with pytest.raises(ValueError):
        validate_holdout(train, test - {"60"})


def test_training_guard_blocks_leaked_fold5_rows():
    rows = [dict(subject=f"{s:02}", label=label) for s in range(1, 49) for label in range(3)]
    validate_training(rows)
    with pytest.raises(ValueError):
        validate_training(rows + [dict(subject="49", label=0)])
    with pytest.raises(ValueError):
        validate_training(rows[3:])
    assert all("fold5" not in str(path) for path in TRAIN_COHORTS)


def test_metrics_against_hand_computed_confusion_matrix():
    m = calculate_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2])
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["accuracy"] == pytest.approx(4/6)
    assert m["macro_precision"] == pytest.approx((.5 + 2/3 + 1)/3)
    assert m["macro_recall"] == pytest.approx(2/3)
    assert m["macro_f1"] == pytest.approx((.5 + .8 + 2/3)/3)
    report = dict(metrics=m, training_subjects=[f"{s:02}" for s in range(1, 49)],
        test_subjects=[f"{s:02}" for s in range(49, 61)], retained_videos=34, model_sha256="a"*64)
    text = format_summary(report)
    assert "66.67%" in text and "CONFUSION MATRIX" in text
    assert "48 participants" in text and "not a fresh test" in text
    assert max(map(len, text.splitlines())) <= 80


def test_invalid_labels_fail():
    for a, b in [([], []), ([0], [3]), ([0, 1], [0])]:
        with pytest.raises(ValueError):
            calculate_metrics(a, b)


def test_training_cannot_write_to_live_models_or_existing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/existing").mkdir(parents=True)
    for path in (Path("models"), Path("outputs"), Path("outputs/existing")):
        with pytest.raises(ValueError, match="new subdirectory"):
            run(path)
