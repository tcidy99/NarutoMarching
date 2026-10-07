# -*- coding: utf-8 -*-
"""把贪心 agent 改写成"看着 RouteEnv 的状态, 每次吐一个 18 维动作"。

为什么需要这个: 行为克隆要 (观测, 动作) 成对的数据。直接去翻译人工存档行不通 ——
那要求精确复现当时的控制流(切队顺序、换日时刻、建队瞬间、蹭不蹭步、传不传送),
错一处后面全错, 实测 73% 的步翻译不上。

而贪心的控制流是我们自己写的, 每一步想做什么一清二楚, 翻译成动作是**构造上**成立
的, 不用猜。它 66,822 分(人工 70,558 的 94.7%), 当老师足够好。

策略每一步按这个顺序决定:
    1. 该队当天步数用完 -> 建队(如果该建) / 用飞雷神 / 切到下一队 / 换日
    2. 有该结算的试探格 -> 结算
    3. 有目标(八卦/帐篷/boss...) -> 朝目标走一格(A* 的下一跳)
    4. 否则挑相邻最值钱的未占领格
    5. 都没有 -> 踩着自己的地跳向最近的边界(同样只走一格)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
import run_agent_route as heuristics
from rl_route_env import (RouteEnv, direction_of, N_DIRS, A_DIR_TELEPORT,
                          A_ADVANCE_DAY, A_CREATE_T2, A_CREATE_T3, A_SETTLE,
                          A_FLY, A_SWITCH_TEAM)


def _step_toward(env, target, commit=True):
    """朝 target 走一格, 返回动作; 走不了返回 None。

    commit=True 时会把整条 A* 路径记在队伍上, 之后每步沿着它走 —— 这一点很关键。
    原来的贪心是"点一个远处的格子、引擎 A* 一次走完", 是一个决策; 移植过来每步
    只走一跳、走完又重新挑目标, 就会在两格之间来回震荡(实测方向1 走了 2,512 次、
    方向4 走了 2,545 次, 整季只占了 507 格还剩四万八千粮没花)。
    """
    here = env._here()
    if here is None or target is None:
        return None
    team = env._active_team()
    plan = getattr(team, '_bc_plan', None)
    if commit and plan and plan.get('goal') == tuple(target):
        # 还在既定路线上就接着走, 不重新寻路
        index = plan['index'].get(here)
        if index is not None and index + 1 < len(plan['path']):
            nxt = tuple(plan['path'][index + 1])
            direction = direction_of(here, nxt)
            if direction is not None:
                return _finish(env, direction, nxt)
    direction = direction_of(here, target)
    if direction is None:                      # 不相邻 -> 用 A* 取下一跳
        path, _cost = game._astar(here, tuple(target), set())
        if not path or len(path) < 2:
            return None
        if commit and team is not None:
            team._bc_plan = {'goal': tuple(target), 'path': path,
                             'index': {tuple(p): i for i, p in enumerate(path)}}
        direction = direction_of(here, tuple(path[1]))
        if direction is None:
            return None
        target = tuple(path[1])
    return _finish(env, direction, target)


def _finish(env, direction, target):
    """落点是传送点时, 只有对岸还有没拿的卦才值得传。"""
    if tuple(target) in heuristics.PORTAL_PAIRS and any(
            h not in env.engine.visited_g_lands for h in env.engine.all_g_lands):
        return A_DIR_TELEPORT + direction
    return direction


def _another_team_can_move(env, budget_map=None):
    """除当前队伍外, 还有没有别的队伍今天能动。"""
    engine = env.engine
    # 粮草见底时谁都动不了 —— 得换日等 +1600, 不能在队伍之间空转。
    if engine.current_food < heuristics.FOOD_FLOOR:
        return False
    for index, team in enumerate(env._teams()):
        if team is None or index == env._active_index:
            continue
        if engine._get_team_steps_for_day(team, engine.current_day) <= 0:
            continue
        if budget_map is not None and budget_map.get(index, 1) <= 0:
            continue
        return True
    return False


def greedy_action(env, food_weight=0.30, day_move_budget=None,
                  day_move_budget_map=None):
    """给出这一步该做什么(18 维里的一个动作)。"""
    engine = env.engine
    team = env._active_team()
    if team is None:
        return A_ADVANCE_DAY

    # 试探格先结算, 否则它会一直挂在那里
    if team.free_exploration_hexes and env.action_masks()[A_SETTLE]:
        return A_SETTLE

    steps = engine._get_team_steps_for_day(team, engine.current_day)
    out_of_moves = (steps <= 0
                    or engine.current_food < heuristics.FOOD_FLOOR
                    or (day_move_budget is not None and day_move_budget <= 0))

    if out_of_moves:
        # 走完了就在脚下开下一支队伍 —— 与人工 day 1 的顺序一致
        if engine.team2 is None and env.action_masks()[A_CREATE_T2]:
            return A_CREATE_T2
        if (engine.team2 is not None and engine.team3 is None
                and env.action_masks()[A_CREATE_T3]):
            return A_CREATE_T3
        # 飞雷神不耗步数, 步数用完之后飞一格是白赚
        if (heuristics.USE_FLY and engine.fly_skill_limit > 1
                and env.action_masks()[A_FLY]):
            return A_FLY
        # 只有在**还有别的队伍能动**时才切队, 否则会在三支都走完的情况下
        # 无限地切来切去(第一版就是这样, 一天之内切了五千多次还没换日)。
        if _another_team_can_move(env, day_move_budget_map):
            return A_SWITCH_TEAM
        return A_ADVANCE_DAY

    # 有战略目标就朝目标走
    target = heuristics.objective_step(engine, team)
    if target is not None:
        action = _step_toward(env, target)
        if action is not None:
            return action
    # 相邻最值钱的未占领格
    target = heuristics.greedy_pick(engine, team, food_weight)
    if target is not None:
        action = _step_toward(env, target)
        if action is not None:
            return action
    # 被自己的地围住了 -> 跳向最近的边界
    target = heuristics.nearest_frontier(engine, team, food_weight)
    if target is not None:
        action = _step_toward(env, target)
        if action is not None:
            return action
    if env.action_masks()[A_SWITCH_TEAM]:
        return A_SWITCH_TEAM
    return A_ADVANCE_DAY


def rollout(env=None, food_weight=0.30, collect=False, max_steps=20000,
            moves_per_team_day=120):
    """用贪心策略在 RouteEnv 里打完一季。collect=True 时顺带收集 (观测, 动作)。"""
    env = env or RouteEnv(food_weight=food_weight)
    env.reset()
    obs_globals, obs_dirs, actions = [], [], []
    # 每队每天的动作上限, 防止跳步不耗步数导致原地打转。
    # 注意这个数要给得比原贪心的 30 大得多: 原贪心"点一个远处目标"是**一个**动作
    # (引擎 A* 一次走完整条路), 移植到逐跳动作之后, 穿越自己领地的每一跳都要占一个
    # 名额。给 30 的话预算全耗在赶路上, 整季只占 507 格、还剩四万八千粮没花(34,320 分);
    # 给到 80 以上就稳定在 65,114 分 / 1,120 格, 和原贪心基本一致。
    budget = {}
    last_day = env.engine.current_day
    for _ in range(max_steps):
        if env.engine.current_day != last_day:
            budget.clear()
            last_day = env.engine.current_day
        key = env._active_index
        budget.setdefault(key, moves_per_team_day)
        action = greedy_action(env, food_weight, budget[key], budget)
        if collect:
            observation = env._observation()
            obs_globals.append(observation['globals'])
            obs_dirs.append(observation['dirs'])
            actions.append(action)
        if action < A_ADVANCE_DAY:
            budget[key] -= 1
        elif action == A_SWITCH_TEAM:
            pass
        env.apply(action)
        if env.engine.current_day >= game.TOTAL_DAYS and action == A_ADVANCE_DAY:
            break
    return env, obs_globals, obs_dirs, actions


if __name__ == '__main__':
    import numpy as np
    import time
    t0 = time.perf_counter()
    env, g, d, a = rollout(collect=True)
    engine = env.engine
    print('贪心策略在 18 维动作空间里跑完一季:')
    print('   day %d  积分 %s  占领 %d  耗粮 %s  余粮 %d'
          % (engine.current_day, format(engine.total_reward, ','),
             len(engine.all_visited_hexes), format(engine.total_food, ','),
             engine.current_food))
    print('   动作样本 %d 个, 用时 %.0f 秒' % (len(a), time.perf_counter() - t0))
    counts = np.bincount(np.asarray(a), minlength=18)
    names = ['方向%d' % i for i in range(6)] + ['方向%d+传送' % i for i in range(6)] + \
            ['换日', '建2队', '建3队', '结算', '飞雷神', '切队']
    print('   动作分布:', {names[i]: int(c) for i, c in enumerate(counts) if c})
    print('   对照: 原贪心 66,822 分 / 占领 1,115')
