"""Coverage test for multi-team ordinary archive replay."""

import glob
from pathlib import Path

from archive_replay import replay_ordinary_segments

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_archive_replays_multi_team_prefix():
    report = replay_ordinary_segments(glob.glob(str(REPO_ROOT / "*余46.json"))[0])
    assert report.replayed == report.total == 1265
    assert report.stopped_team is None
    assert report.stopped_segment is None
    assert report.reason is None
    assert report.day == 93


if __name__ == "__main__":
    test_archive_replays_multi_team_prefix()
    print("Archive replay prefix test passed")
