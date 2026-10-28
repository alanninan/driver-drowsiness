import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_development_trees import development_split, grouped_splits


def test_development_boundary_rejects_test_fold_and_missing_drivers():
    groups = np.repeat([f"{n:02}" for n in range(1, 49)], 3)
    train, validation = development_split(groups)
    assert set(groups[train]) == {f"{n:02}" for n in range(1, 37)}
    assert set(groups[validation]) == {f"{n:02}" for n in range(37, 49)}
    for invalid in (np.append(groups, "49"), groups[groups != "48"]):
        with pytest.raises(ValueError, match="Fold5 subjects are forbidden"):
            development_split(invalid)


def test_tuning_never_separates_windows_of_same_driver_or_accepts_validation():
    groups = np.repeat([f"{n:02}" for n in range(1, 37)], 4)
    held_out = []
    for train, validation in grouped_splits(groups):
        assert not set(groups[train]) & set(groups[validation])
        held_out.extend(validation)
    assert sorted(held_out) == list(range(len(groups)))
    with pytest.raises(ValueError, match="training subjects"):
        grouped_splits(np.append(groups, "37"))
