"""Differential smoke tests for the GUI-logic-backed headless engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction
import hex_pathfinding_demo as game


def _first_neighbor(start):
    return next(pos for pos in game._neighbors(*start) if game._passable(*pos))


def test_reset_and_move_use_game_transition_logic():
    engine = NarutoMarchingEngine()
    initial = engine.reset()
    start = initial["teams"][1]["position"]
    target = _first_neighbor(start)

    observation, reward, done, info = engine.step(
        RouteAction(ActionType.MOVE_TO_HEX, target=target, team=1)
    )

    assert not done
    assert observation["teams"][1]["position"] == target
    assert observation["total_food"] >= 0
    assert observation["total_reward"] - initial["total_reward"] == reward
    assert info["observation_before"]["teams"][1]["position"] == start


def test_invalid_fly_target_does_not_consume_skill():
    engine = NarutoMarchingEngine()
    initial = engine.reset()
    before = initial["fly_skill_limit"]
    observation, _reward, _done, _info = engine.step(
        RouteAction(ActionType.FLY_TO_HEX, target=(0, 0), team=1)
    )
    assert observation["fly_skill_limit"] == before


def test_fly_validation_accepts_another_teams_occupied_bridge():
    engine = NarutoMarchingEngine()
    engine.reset()
    owned = (20, 20)
    bridge = game._neighbors(*owned)[0]
    target = next(
        hex_pos
        for hex_pos in game._neighbors(*bridge)
        if hex_pos != owned and game._passable(*hex_pos)
    )
    engine.team1.visited_hexes = {owned}
    engine.team1.free_exploration_hexes = set()
    engine.team1.full_path = [owned]
    engine.team2 = game.Team(bridge, created_day=1)
    engine.team2.visited_hexes = {bridge}
    engine.team2.free_exploration_hexes = set()
    engine.all_visited_hexes = {owned, bridge}
    engine.active_team = engine.team1

    assert engine._is_valid_fly_destination(target)


def test_fixed_trace_preserves_jump_and_day_state():
    engine = NarutoMarchingEngine()
    initial = engine.reset()
    start = initial["teams"][1]["position"]
    first_target = _first_neighbor(start)

    moved, _reward, _done, move_info = engine.step(
        RouteAction(ActionType.MOVE_TO_HEX, target=first_target, team=1)
    )
    assert moved["teams"][1]["position"] == first_target
    assert engine.team1._seg_action_sequence[-1][-1][0] == "new"
    assert move_info["jump_count"] == 0

    returned, jump_reward, _done, jump_info = engine.step(
        RouteAction(ActionType.MOVE_TO_HEX, target=start, team=1)
    )
    assert returned["teams"][1]["position"] == start
    assert engine.team1._seg_action_sequence[-1][-1][0] == "jump"
    assert jump_info["jump_count"] == 1
    assert jump_reward == 0

    next_day, _reward, _done, _info = engine.step(
        RouteAction(ActionType.ADVANCE_DAY, team=1)
    )
    assert next_day["day"] == 2
    assert next_day["teams"][1]["position"] == start


def test_replay_path_preserves_saved_path_shape():
    engine = NarutoMarchingEngine()
    initial = engine.reset()
    start = initial["teams"][1]["position"]
    target = _first_neighbor(start)

    observation, _reward, _done, _info = engine.step(
        RouteAction(
            ActionType.REPLAY_PATH,
            target=target,
            path=(start, target),
            team=1,
        )
    )

    assert observation["teams"][1]["position"] == target
    assert engine.team1._seg_path_nodes[-1] == [target]


if __name__ == "__main__":
    test_reset_and_move_use_game_transition_logic()
    test_invalid_fly_target_does_not_consume_skill()
    print("RL engine tests passed")
