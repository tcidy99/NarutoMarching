"""Golden-trace round-trip and jump visibility tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hex_pathfinding_demo as game
from rl_playground import ActionType, RouteAction
from rl_trace import record_trace, replay_trace, save_trace


def _first_neighbor(start):
    return next(pos for pos in game._neighbors(*start) if game._passable(*pos))


def test_trace_round_trip_is_deterministic():
    origin = game._find_start_position()
    target = _first_neighbor(origin)
    actions = [
        RouteAction(ActionType.MOVE_TO_HEX, target=target, team=1),
        RouteAction(ActionType.MOVE_TO_HEX, target=origin, team=1),
        RouteAction(ActionType.ADVANCE_DAY, team=1),
    ]

    trace = record_trace(actions)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "golden_trace.json"
        save_trace(str(path), trace)
        assert replay_trace(str(path)) is None

    assert trace["records"][1]["jump_count"] == 1
    assert trace["records"][1]["score_delta"] == 0


if __name__ == "__main__":
    test_trace_round_trip_is_deterministic()
    print("RL trace tests passed")
