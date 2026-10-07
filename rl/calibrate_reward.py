# -*- coding: utf-8 -*-
"""把人工存档整条路线按训练奖励表折算一遍, 看看各项的量级对不对。

判据很简单: 一条已知的好路线(全局70558_6.json, 70,558 分, 打满 97 天)按这张表
算下来必须是一个明显为正、且"占格"占主导的分数。如果某一项(比如粮草惩罚)大到
把它压成负数, 那这张表就是在教 agent 别去占地。

用法:  python rl/calibrate_reward.py [存档路径]
不需要 gymnasium —— 这里只用 TRAINING_REWARD 里的数字重算, 不构造环境。
"""

import io
import os
import sys
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401  统一把 rl/ 与仓库根加进 sys.path, 并校正工作目录

import contextlib

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from rl_gym_env import TRAINING_REWARD, SCORE_PER_RP, DAILY_STEP_GRANT

import tkinter.messagebox as _mb
import tkinter.filedialog as _fd
_mb.askyesno = lambda *a, **k: True
for _n in ('showinfo', 'showwarning', 'showerror'):
    setattr(_mb, _n, lambda *a, **k: None)


def load(path):
    _fd.askopenfilename = lambda *a, **k: path
    demo = game.PathfindingDemo()
    for name in ('_auto_save_game', '_draw', '_animate_view_to',
                 '_center_view_on_active_team', '_center_view_on_active_team_day'):
        setattr(demo, name, lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()):
        demo._load_game()
    return demo


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _rlpath.REPO_ROOT, '全局70558_6.json')
    demo = load(path)
    teams = [(i, t) for i, t in enumerate((demo.team1, demo.team2, demo.team3), 1)
             if t is not None]

    new_hexes = sum(len(seg) for _i, t in teams for seg in t._seg_new_hexes)
    food = sum(sum(t._seg_foods) for _i, t in teams)
    score = sum(sum(t._seg_awards) for _i, t in teams)
    steps = sum(sum(t._seg_steps) for _i, t in teams)
    active_days = sorted({day for _i, t in teams for day in t._seg_days})

    cfg = TRAINING_REWARD
    terms = {}
    terms['new_hex 占新格'] = cfg['new_hex'] * new_hexes
    terms['score 游戏积分折算'] = cfg['score'] * (score / SCORE_PER_RP)
    terms['day_progress 推进天数'] = cfg['day_progress'] * (game.TOTAL_DAYS - 1)
    terms['completion 打满一季'] = cfg['completion']

    # 里程碑: 这条路线确实集齐了八卦, 帐篷拿了几个按实际算
    taken = set(demo.all_visited_hexes)

    def count(tokens):
        return sum(1 for ir, ic in taken
                   if 0 <= ir < game.ROWS and 0 <= ic < game.COLS
                   and game.RAW_MAP[ir][ic] in tokens)

    g_taken, tent_taken = count(('G', 'g')), count(('T',))
    terms['g_complete 八卦集齐'] = cfg['g_complete'] if g_taken >= 8 else 0.0
    terms['tent_complete 帐篷拿完'] = cfg['tent_complete'] if tent_taken >= 15 else 0.0

    # 建队: 这条路线两支都是 day 1、且上一支走完之后才开的 -> 系数都是满的
    created = [(i, t.created_day) for i, t in teams if i in (2, 3)]
    team_bonus = 0.0
    for _i, day in created:
        ramp = max(1.0, 0.15 * float(game.TOTAL_DAYS))
        earliness = max(0.2, 1.0 - (day - 1) / ramp)
        team_bonus += cfg['create_team'] * earliness * 1.0
    terms['create_team 建队'] = team_bonus

    terms['food_cost 粮草'] = -cfg['food_cost'] * food

    # 换日时剩余步数的惩罚: 按每天(三队可用 - 实际用掉)估
    steps_by_day = defaultdict(int)
    for _i, t in teams:
        for day, st in zip(t._seg_days, t._seg_steps):
            steps_by_day[day] += st
    leftover = 0
    for day in range(1, game.TOTAL_DAYS + 1):
        alive = sum(1 for _i, t in teams if t.created_day <= day)
        leftover += max(0, alive * DAILY_STEP_GRANT - steps_by_day.get(day, 0))
    terms['daily_step_remaining 剩步'] = -cfg['daily_step_remaining'] * leftover

    print('校准存档: %s' % os.path.basename(path))
    print('  游戏侧: 积分 %s / 耗粮 %s / 新占格 %d / 步数 %d / 行动日 %d'
          % (format(demo.total_reward, ','), format(food, ','), new_hexes,
             steps, len(active_days)))
    print('  建队: %s' % ', '.join('team%d 于 day%d' % (i, d) for i, d in created))
    print()
    print('  按训练奖励表折算(单位 RP, 与游戏积分无关):')
    width = max(len(k) for k in terms)
    for key, value in sorted(terms.items(), key=lambda kv: -kv[1]):
        print('     %-*s %+10.1f' % (width, key, value))
    total = sum(terms.values())
    print('     %-*s %+10.1f' % (width, '合计', total))
    print()
    print('  每占一格的平均净收益: %+.2f RP' % (total / max(1, new_hexes)))
    print('  正项合计 %+.1f / 负项合计 %+.1f'
          % (sum(v for v in terms.values() if v > 0),
             sum(v for v in terms.values() if v < 0)))
    if total <= 0:
        print('  !! 这条已知的好路线折算出来是负分 —— 奖励表需要重调')
    elif sum(v for v in terms.values() if v < 0) < -0.5 * sum(
            v for v in terms.values() if v > 0):
        print('  !! 负项超过正项的一半, 惩罚偏重')
    else:
        print('  OK 正项主导, 且 new_hex 是最大的一项 ——'
              ' 奖励表在鼓励"多占地", 与人工打法一致')


if __name__ == '__main__':
    main()
