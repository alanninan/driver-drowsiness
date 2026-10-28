import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_final_fold import coverage_summary, validate_partition


def test_final_partition_forbids_overlap_and_missing_participants():
    train = {f"{n:02}" for n in range(1, 37)}
    test = {f"{n:02}" for n in range(49, 61)}
    validate_partition(train, test)
    for invalid in (test - {"60"}, test | {"36"}, {"01"}):
        with pytest.raises(ValueError):
            validate_partition(train, invalid)
    with pytest.raises(ValueError):
        validate_partition(train | {"49"}, test)


def test_coverage_excludes_calibration_and_preserves_split_video_identity():
    coverage = {"coverage": [
        {"label": 0, "candidate_windows": 10, "video": "alert"},
        {"label": 10, "candidate_windows": 10, "video": "part1"},
        {"label": 10, "candidate_windows": 10, "video": "part2"},
    ]}
    result = coverage_summary(coverage, [{"video": "alert"}, {"video": "part1"}])
    assert result["candidate_windows_after_enrolment"] == 25
    assert result["retained_videos"] == 2
    assert result["videos_without_windows"] == ["part2"]
