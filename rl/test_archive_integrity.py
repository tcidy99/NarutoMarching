"""Integrity checks for the current representative GUI archive."""

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = next(Path(path) for path in glob.glob(str(REPO_ROOT / "*余46.json")))


def test_archive_segment_ledgers_are_consistent():
    state = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    checked = 0
    for team_name in ("team1", "team2", "team3"):
        team = state[team_name]
        if team is None:
            continue
        segment_count = len(team["_seg_days"])
        fields = (
            "_seg_lengths", "_seg_foods", "_seg_awards", "_seg_steps",
            "_seg_new_hexes", "_seg_exploration_hexes", "_seg_jumps",
            "_seg_path_nodes", "_seg_end_positions", "_seg_action_sequence",
            "_seg_hex_costs", "_seg_is_fly_skill", "_seg_fly_skill_deltas",
        )
        for field in fields:
            assert len(team[field]) == segment_count, (team_name, field)

        for index in range(segment_count):
            nodes = team["_seg_path_nodes"][index]
            costs = team["_seg_hex_costs"][index]
            assert len(nodes) == team["_seg_lengths"][index], (team_name, index, "length")
            assert len(costs) in (len(nodes), len(nodes) + 1), (team_name, index, "cost alignment")

            totals = tuple(
                sum(entry[column] for entry in costs if len(entry) > column)
                for column in range(3)
            )
            stored = (
                team["_seg_foods"][index],
                team["_seg_awards"][index],
                team["_seg_steps"][index],
            )
            assert totals == stored, (team_name, index, totals, stored)
            checked += 1

    assert checked == 1265


if __name__ == "__main__":
    test_archive_segment_ledgers_are_consistent()
    print(f"Archive integrity passed: {ARCHIVE_PATH.name}, 1265 segments")
