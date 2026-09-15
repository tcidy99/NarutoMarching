"""Optional Gymnasium adapter checks."""

import importlib.util

from rl_gym_env import NarutoMarchingGymEnv
from rl_playground import ActionType
import hex_pathfinding_demo as game


def test_gym_adapter_has_clear_optional_dependency_behavior():
    if importlib.util.find_spec("gymnasium") is None:
        try:
            NarutoMarchingGymEnv()
        except ImportError as error:
            assert "gymnasium" in str(error)
        else:
            raise AssertionError("adapter should require optional gymnasium")


def test_invalid_action_uses_configured_penalty_when_available():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    _observation, reward, terminated, truncated, info = env.step([1, 0, 0, 0, 0])
    assert reward == -env.reward_config["invalid_action"]
    assert not terminated and not truncated
    assert info["invalid_action"] is True
    assert info["reward_breakdown"]["total"] == reward
    env.close()


def test_completion_milestones_are_one_time_rewards():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    env.reset()
    base = env.engine.get_observation()
    g_hexes = [
        (ir, ic)
        for ir in range(game.ROWS)
        for ic in range(game.COLS)
        if game.RAW_MAP[ir][ic] in ("G", "g")
    ]
    tent_hexes = [
        (ir, ic)
        for ir in range(game.ROWS)
        for ic in range(game.COLS)
        if game.RAW_MAP[ir][ic] == "T"
    ]
    before = dict(base)
    after = dict(base)
    after["visited_hexes"] = frozenset(g_hexes + tent_hexes[:15])
    info = {"score_delta": 0}
    reward = env._transition_reward(before, after, info, False, ActionType.MOVE_TO_HEX)
    assert reward == 800.0
    assert info["reward_breakdown"]["g_complete_bonus"] == 500.0
    assert info["reward_breakdown"]["tent_complete_bonus"] == 300.0

    reward_again = env._transition_reward(after, after, {"score_delta": 0}, False, ActionType.MOVE_TO_HEX)
    assert reward_again == 0.0
    env.close()


def test_advance_day_penalizes_unused_resources():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    before = env.engine.get_observation()
    after = dict(before)
    after["day"] = before["day"] + 1
    info = {"score_delta": 0}
    reward = env._transition_reward(before, after, info, False, ActionType.ADVANCE_DAY)
    assert reward < 0
    assert info["reward_breakdown"]["daily_food_remaining_penalty"] < 0
    assert info["reward_breakdown"]["daily_step_remaining_penalty"] < 0
    env.close()


def test_move_is_rejected_when_selected_team_has_no_steps():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    env.reset()
    env.engine.current_day = 4
    env.engine._rebuild_day_records()
    env.engine.day_records[3]["team1_steps_remain"] = 0
    _observation, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0])
    assert reward == -env.reward_config["invalid_action"]
    assert not terminated and not truncated
    assert info["invalid_action"] is True
    assert "no remaining steps" in info["reason"]
    env.close()


def test_zero_steps_allows_one_adjacent_probe():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    env.reset()
    env.engine.current_day = 4
    env.engine._rebuild_day_records()
    env.engine.day_records[3]["team1_steps_remain"] = 0
    start = env.engine.team1.full_path[-1]
    target = next(
        neighbor
        for neighbor in game._neighbors(*start)
        if neighbor not in env.engine.all_visited_hexes
        and game._terrain(*neighbor).get("step", 1) > 0
    )
    _observation, _reward, _terminated, _truncated, info = env.step(
        [0, 0, target[0], target[1], 0]
    )
    assert info.get("invalid_action") is not True
    assert target in env.engine.team1.free_exploration_hexes
    env.close()


def test_normal_move_rejects_non_adjacent_target():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    env.reset()
    start = env.engine.team1.full_path[-1]
    far_target = (0, 0) if (0, 0) not in game._neighbors(*start) else (1, 1)
    _observation, reward, _terminated, _truncated, info = env.step(
        [0, 0, far_target[0], far_target[1], 0]
    )
    assert reward == -env.reward_config["invalid_action"]
    assert info["invalid_action"] is True
    assert "adjacent" in info["reason"]
    env.close()


def test_action_candidates_have_score_food_weights():
    if importlib.util.find_spec("gymnasium") is None:
        return
    env = NarutoMarchingGymEnv()
    env.reset()
    candidates = env.get_action_candidates()
    assert candidates
    assert all(item["food_cost"] >= 0 for item in candidates)
    assert all(item["weight"] > 0 for item in candidates)
    assert abs(sum(item["probability"] for item in candidates) - 1.0) < 1e-6
    sampled = env.sample_weighted_candidate()
    assert sampled is not None
    assert 0.0 < sampled["sampling_probability"] <= 1.0
    env.close()


if __name__ == "__main__":
    test_gym_adapter_has_clear_optional_dependency_behavior()
    print("Optional Gymnasium adapter test passed")
