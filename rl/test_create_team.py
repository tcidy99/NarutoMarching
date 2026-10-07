# -*- coding: utf-8 -*-
"""建队动作 + 训练奖励与游戏积分分离 的检查。

引擎层的检查不需要 gymnasium; 需要 gymnasium 的部分自己跳过, 与 test_rl_gym_env
的写法保持一致。
"""

import importlib.util

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401  统一把 rl/ 与仓库根加进 sys.path, 并校正工作目录

from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction
import hex_pathfinding_demo as game


def _engine():
    engine = NarutoMarchingEngine()
    engine.reset()
    return engine


def test_reset_starts_with_only_team1():
    engine = _engine()
    observation = engine.get_observation()
    assert observation["teams"][1] is not None
    assert observation["teams"][2] is None
    assert observation["teams"][3] is None


def test_create_team_opens_team_at_active_team_position():
    engine = _engine()
    where = tuple(engine.team1.full_path[-1])
    observation, _score_delta, _done, _info = engine.step(
        RouteAction(action_type=ActionType.CREATE_TEAM, team=2)
    )
    assert engine.team2 is not None
    assert tuple(engine.team2.origin) == where
    assert engine.team2.created_day == engine.current_day
    assert observation["teams"][2]["position"] == where
    # 新队伍随即成为激活队伍, 与 GUI 按钮的行为一致。
    assert engine.active_team is engine.team2
    assert observation["active_team"] == 2


def test_create_team_three_also_works_and_is_independent():
    engine = _engine()
    engine.step(RouteAction(action_type=ActionType.CREATE_TEAM, team=2))
    engine.step(RouteAction(action_type=ActionType.CREATE_TEAM, team=3))
    assert engine.team2 is not None and engine.team3 is not None
    observation = engine.get_observation()
    # 步数是每队独立结算的 —— 这正是建队的意义。
    assert observation["teams"][2]["steps"] > 0
    assert observation["teams"][3]["steps"] > 0


def test_create_team_rejects_bad_targets():
    engine = _engine()
    for bad in (1, 0, 4):
        try:
            engine.step(RouteAction(action_type=ActionType.CREATE_TEAM, team=bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"team={bad} 应该被拒绝")
    engine.step(RouteAction(action_type=ActionType.CREATE_TEAM, team=2))
    try:
        engine.step(RouteAction(action_type=ActionType.CREATE_TEAM, team=2))
    except RuntimeError:
        pass
    else:
        raise AssertionError("重复建同一支队伍应该被拒绝")


def test_create_team_does_not_spend_food_or_score():
    engine = _engine()
    before = engine.get_observation()
    _observation, score_delta, _done, _info = engine.step(
        RouteAction(action_type=ActionType.CREATE_TEAM, team=2)
    )
    after = engine.get_observation()
    assert after["total_food"] == before["total_food"]
    assert score_delta == 0
    assert after["total_reward"] == before["total_reward"]


def test_engine_step_reports_game_score_not_training_reward():
    """引擎那一位返回的是游戏积分增量, 并且在 info 里有带 game_ 前缀的同名字段。"""
    engine = _engine()
    start = tuple(engine.team1.full_path[-1])
    target = next(p for p in game._neighbors(*start) if game._passable(*p))
    before = engine.get_observation()
    observation, score_delta, _done, info = engine.step(
        RouteAction(action_type=ActionType.MOVE_TO_HEX, target=target, team=1)
    )
    assert score_delta == observation["total_reward"] - before["total_reward"]
    assert info["game_score_delta"] == score_delta
    assert info["game_total_reward"] == observation["total_reward"]


# ── 下面这些要 gymnasium ────────────────────────────────────────────────────
def _needs_gym():
    return importlib.util.find_spec("gymnasium") is None


def test_gym_action_space_exposes_create_team():
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv
    env = NarutoMarchingGymEnv()
    assert env.action_space.nvec[0] == 7
    masks = env.action_masks()
    assert len(masks["action_type"]) == 7
    assert masks["action_type"][6]          # 开局还没建队, 应该允许
    assert masks["team"][1] and masks["team"][2]  # 目标队伍尚不存在也要可选


def test_gym_create_team_pays_bonus_in_training_points():
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv, TRAINING_REWARD
    env = NarutoMarchingGymEnv()
    env.reset()
    _obs, reward, _term, _trunc, info = env.step([6, 1, 0, 0, 0])
    assert env.engine.team2 is not None
    breakdown = info["reward_breakdown"]
    assert breakdown["create_team_bonus"] > 0
    # 这里是"day 1 但 team1 一步没走"就建队: 早晚系数满额 1.0, 就位系数是地板
    # 0.2(新队伍会和 team1 叠在同一格)。真正的满额要等创建者把当天步数走完 ——
    # 见 test_gym_create_team_rewards_the_human_day1_ordering。
    factors = breakdown["create_team_factors"]
    assert factors["earliness"] == 1.0
    assert factors["readiness"] == 0.2
    expected = TRAINING_REWARD["create_team"] * 1.0 * 0.2
    assert abs(breakdown["create_team_bonus"] - expected) < 1e-9
    # 建队不产生游戏积分, 所以训练奖励与游戏积分必须是两个数
    assert info["game_score_delta"] == 0
    assert reward > 0


def test_gym_create_team_rewards_the_human_day1_ordering():
    """人工存档 全局70558_6.json 的 day 1 打法: team1 先把 6 步走完, 在它停下的
    那一格开 team2; team2 走完 6 步, 再在它停下的那一格开 team3。

    所以"创建者当天步数已用光"时的建队奖励, 必须明显高于"一步没走就建队"。
    """
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv

    # (a) 一步没走就建队
    env = NarutoMarchingGymEnv()
    env.reset()
    _o, _r, _t, _tr, eager = env.step([6, 1, 0, 0, 0])
    eager_bonus = eager["reward_breakdown"]["create_team_bonus"]

    # (b) 先把 team1 当天的步数走光, 再建队
    env2 = NarutoMarchingGymEnv()
    env2.reset()
    for _ in range(20):
        if env2.engine._get_team_steps_for_day(env2.engine.team1,
                                               env2.engine.current_day) <= 0:
            break
        start = tuple(env2.engine.team1.full_path[-1])
        target = next(
            (p for p in game._neighbors(*start)
             if game._passable(*p) and p not in env2.engine.all_visited_hexes),
            None,
        )
        if target is None:
            break
        env2.step([0, 0, target[0], target[1], 0])
    assert env2.engine._get_team_steps_for_day(
        env2.engine.team1, env2.engine.current_day) <= 0, "没能把 team1 的步数走光"
    _o, _r, _t, _tr, ready = env2.step([6, 1, 0, 0, 0])
    ready_bonus = ready["reward_breakdown"]["create_team_bonus"]

    assert env2.engine.team2 is not None
    # 新队伍确实开在 team1 停下的那一格(和人工存档一样)
    assert tuple(env2.engine.team2.origin) == tuple(env2.engine.team1.full_path[-1])
    assert ready_bonus > eager_bonus, (
        f"走完再建({ready_bonus}) 应该比一步没走就建({eager_bonus}) 奖励更高"
    )
    assert ready["reward_breakdown"]["create_team_factors"]["readiness"] == 1.0


def test_gym_create_team_bonus_decays_with_the_day():
    """人工存档两支队伍都是 day 1 建的; 拖到后面建, 奖励要明显变少。"""
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv
    env = NarutoMarchingGymEnv()
    env.reset()
    _o, _r, _t, _tr, early = env.step([6, 1, 0, 0, 0])

    env2 = NarutoMarchingGymEnv()
    env2.reset()
    for _ in range(40):
        env2.step([4, 0, 0, 0, 0])          # 一路换日, 什么都不做
    assert env2.engine.current_day > 20
    _o, _r, _t, _tr, late = env2.step([6, 1, 0, 0, 0])

    e = early["reward_breakdown"]
    l = late["reward_breakdown"]
    assert e["create_team_factors"]["earliness"] == 1.0
    assert l["create_team_factors"]["earliness"] < e["create_team_factors"]["earliness"]
    assert l["create_team_bonus"] < e["create_team_bonus"]


def test_gym_masks_out_create_team_once_all_teams_exist():
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv
    env = NarutoMarchingGymEnv()
    env.reset()
    env.step([6, 1, 0, 0, 0])
    env.step([6, 2, 0, 0, 0])
    assert env.engine.team2 is not None and env.engine.team3 is not None
    assert not env.action_masks()["action_type"][6]


def test_gym_reward_is_not_the_game_score():
    """训练奖励必须是另一套数: 游戏积分要先除以 SCORE_PER_RP 才能进 reward。"""
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv, SCORE_PER_RP
    env = NarutoMarchingGymEnv()
    env.reset()
    start = tuple(env.engine.team1.full_path[-1])
    target = next(p for p in game._neighbors(*start) if game._passable(*p))
    _obs, reward, _term, _trunc, info = env.step([0, 0, target[0], target[1], 0])
    score = info["game_score_delta"]
    assert score > 0                      # 占了新格子, 游戏积分涨了
    assert abs(reward - score) > 1.0      # 但训练奖励不等于它
    assert abs(info["reward_breakdown"]["score_rp"] - score / SCORE_PER_RP) < 1e-9


def test_gym_advance_day_is_less_punishing_than_an_illegal_action():
    """曾经的死锁: day 1 换日净收益约 -1.61, 乱按非法动作只罚 -0.25, 于是最优策略
    变成赖在第一天乱按。换日可以是负的, 但不能比"什么都不做"还亏。"""
    if _needs_gym():
        return
    from rl_gym_env import NarutoMarchingGymEnv
    env = NarutoMarchingGymEnv()
    env.reset()
    _obs, advance_reward, _t, _tr, _info = env.step([4, 0, 0, 0, 0])
    env.reset()
    _obs, invalid_reward, _t, _tr, info = env.step([1, 0, 0, 0, 0])
    assert info["invalid_action"] is True
    assert advance_reward > invalid_reward, (
        f"换日 {advance_reward} 应该优于乱按 {invalid_reward}"
    )
