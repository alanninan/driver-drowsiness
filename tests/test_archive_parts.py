import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_fold import member_label, validate_members


def test_split_recordings_preserve_label_and_require_both_parts():
    paths = [Path("32") / f"{stem}.mp4" for stem in ("0", "5", "10_1", "10_2")]
    validate_members(paths, {"32"})
    assert [member_label(p) for p in paths] == [0, 5, 10, 10]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        validate_members(paths[:-1], {"32"})
    with pytest.raises(ValueError, match="Duplicate archive"):
        validate_members(paths + paths[-1:], {"32"})


def test_unexpected_split_or_extra_unsplit_recording_is_rejected():
    assert member_label(Path("46/5.m4v")) == 5
    with pytest.raises(ValueError, match="Unexpected archive member"):
        member_label(Path("31/10_1.mp4"))
    paths = [Path("49") / f"{stem}.mp4" for stem in ("0", "5", "10_1", "10_2", "10")]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        validate_members(paths, {"49"})
