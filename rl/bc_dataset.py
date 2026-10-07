# -*- coding: utf-8 -*-
"""把一份存档的路线翻译成 (观测, 动作) 序列, 供行为克隆使用。

做法是"跟着放":在 RouteEnv 里从头重放这条路线 —— 每一步先记下当前观测, 再把人工
那一步翻译成 12 维里的某个动作并执行。这样拿到的观测和真实训练时看到的完全同分布,
不是离线拼出来的。

翻译规则(与 rl_route_env 的动作定义一一对应):
    · 路径里的每一格 -> 它相对当前位置的方向下标 0-5(未占领是占格, 已占领是跳步)
    · 段与段之间日期变了 -> 插入换日
    · 轮到别的队伍 -> 插入切队
    · 新队伍出现 -> 插入建队
    · 飞雷神段 -> 动作 10(目标由启发式选, 不保证和人工落点一致)
    · 空段(原地结算试探格) -> 动作 9

翻译不上的步会被统计出来并跳过 —— 跑一次就知道覆盖率。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import contextlib
import io
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from rl_route_env import (RouteEnv, direction_of, N_DIRS, A_DIR_TELEPORT,
                          A_ADVANCE_DAY, A_CREATE_T2, A_CREATE_T3, A_SETTLE,
                          A_FLY, A_SWITCH_TEAM)

import tkinter.messagebox as _mb
import tkinter.filedialog as _fd
_mb.askyesno = lambda *a, **k: True
for _n in ('showinfo', 'showwarning', 'showerror'):
    setattr(_mb, _n, lambda *a, **k: None)


def load_route(path):
    """读出存档里按全局动作顺序排好的段列表。"""
    _fd.askopenfilename = lambda *a, **k: path
    demo = game.PathfindingDemo()
    for name in ('_auto_save_game', '_draw', '_animate_view_to',
                 '_center_view_on_active_team', '_center_view_on_active_team_day'):
        setattr(demo, name, lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()):
        demo._load_game()

    segments = []
    for team_number, team in enumerate((demo.team1, demo.team2, demo.team3), 1):
        if team is None:
            continue
        for j in range(len(team._seg_lengths)):
            order = team._seg_action_orders[j] if j < len(team._seg_action_orders) else -1
            segments.append({
                'order': order if order is not None else -1,
                'team': team_number,
                'day': team._seg_days[j],
                'nodes': [tuple(x) for x in team._seg_path_nodes[j]],
                'end': tuple(team._seg_end_positions[j])
                if j < len(team._seg_end_positions) else None,
                'is_fly': bool(team._seg_is_fly_skill[j])
                if j < len(team._seg_is_fly_skill) else False,
                'created_day': team.created_day,
            })
    segments.sort(key=lambda s: s['order'])
    # 建队时机要靠 origin 还原: 存档记了 created_day, 但没记"当时是在哪一格建的" ——
    # 而 team.origin 就是那一格。所以重放时只要"激活队伍正好站在 origin 上、且日期
    # 到了", 就是人工建这支队伍的那一刻。
    created = {team_number: {'day': t.created_day, 'origin': tuple(t.origin)}
               for team_number, t in enumerate((demo.team1, demo.team2, demo.team3), 1)
               if t is not None}
    return segments, created


def build(path, food_weight=0.30, verbose=True):
    """返回 (观测列表, 动作列表, 统计)。"""
    segments, created = load_route(path)
    env = RouteEnv(food_weight=food_weight)
    env.reset()
    obs_globals, obs_dirs, actions = [], [], []
    stats = Counter()

    def record_and_apply(action):
        observation = env._observation()
        obs_globals.append(observation['globals'])
        obs_dirs.append(observation['dirs'])
        actions.append(action)
        return env.apply(action)

    for seg in segments:
        # 1) 日期对齐
        guard = 0
        while env.engine.current_day < seg['day'] and guard < 200:
            record_and_apply(A_ADVANCE_DAY)
            stats['换日'] += 1
            guard += 1
        # 2) 建队。时机必须和人工一致 —— 新队伍开在"当时激活队伍所在的那一格",
        #    而那一格就是它的 origin。所以条件是"日期到了 + 激活队伍正好站在它的
        #    origin 上", 不能用"轮到它行动了"(team3 是 day1 建的、day3 才第一次动,
        #    等到它第一段再建就晚了三天, 建在别处)。
        for team_number, action in ((2, A_CREATE_T2), (3, A_CREATE_T3)):
            info = created.get(team_number)
            if (info is None or env._teams()[team_number - 1] is not None
                    or env.engine.current_day < info['day']):
                continue
            if team_number == 3 and env.engine.team2 is None:
                continue
            if env._here() != info['origin']:
                continue
            if record_and_apply(action):
                stats['建队'] += 1
            else:
                stats['建队失败'] += 1
        # 3) 切到这一段该动的队伍
        guard = 0
        while (env._active_index + 1) != seg['team'] and guard < 4:
            record_and_apply(A_SWITCH_TEAM)
            stats['切队'] += 1
            guard += 1
        if (env._active_index + 1) != seg['team']:
            stats['切队对不上(跳过该段)'] += 1
            continue
        # 4) 这一段本身
        if seg['is_fly']:
            record_and_apply(A_FLY)
            stats['飞雷神(目标由启发式定)'] += 1
            continue
        if not seg['nodes']:
            if env.action_masks()[A_SETTLE]:
                record_and_apply(A_SETTLE)
                stats['结算试探格'] += 1
            else:
                stats['空段但无试探格(跳过)'] += 1
            continue
        for index, node in enumerate(seg['nodes']):
            here = env._here()
            d = direction_of(here, node) if here else None
            if d is None:
                stats['目标不相邻(跳过)'] += 1
                break
            occupied = node in env.engine.all_visited_hexes
            # 人工这一步传送了没有: 只有本段最后一格、且记录的终点和落脚格不同
            last = index == len(seg['nodes']) - 1
            teleported = (last and seg['end'] is not None
                          and tuple(seg['end']) != node)
            action = (A_DIR_TELEPORT + d) if teleported else d
            if not record_and_apply(action):
                stats['引擎拒绝该步'] += 1
                break
            if teleported:
                stats['传送'] += 1
            else:
                stats['跳步' if occupied else '相邻单步占格'] += 1

    engine = env.engine
    if verbose:
        total = len(actions)
        print('%s -> %d 个动作样本' % (_os.path.basename(path), total))
        for kind, n in stats.most_common():
            print('   %-22s %5d' % (kind, n))
        bad = sum(n for k, n in stats.items() if '跳过' in k or '失败' in k or '拒绝' in k)
        print('   翻译不了的步: %d (%.2f%%)' % (bad, bad / max(1, total + bad) * 100))
        print('   重放结果: day %d, 积分 %s, 占领 %d'
              % (engine.current_day, format(engine.total_reward, ','),
                 len(engine.all_visited_hexes)))
    return (np.asarray(obs_globals, dtype=np.float32),
            np.asarray(obs_dirs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64), stats, engine)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('saves', nargs='*',
                        default=[_os.path.join(_rlpath.REPO_ROOT, '全局70558_6.json')])
    args = parser.parse_args()
    for save in args.saves:
        build(save)
        print()
