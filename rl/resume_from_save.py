# -*- coding: utf-8 -*-
"""从某份存档的第 N 天接手, 让 agent 打完剩下的赛季。

想法很直接: 人工那条 70,558 的路线前半程比 agent 强(开局布局、八卦顺序、队伍分工),
而 agent 在中后期的"每一格值不值"上算得比人细。那就各取所长 —— 保留人工前 N 天,
后面交给 agent。只要后半程打得比人工当时好, 总分就能超过 70,558。

引擎本来就支持"只保存到第 N 天"(软件里保存时问的那个"是否只保存到当前 Day"),
对应 _build_game_state_for_save(through_day=N)。把截断后的状态写成临时存档, 再用
无头引擎载入, 就得到了一个"打到第 N 天"的真实局面, 之后照常往下打即可。

用法:
    python rl/resume_from_save.py --save 全局70558_6.json --days 60 70 80 85 90
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import argparse
import contextlib
import io
import json
import time

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from rl_engine import NarutoMarchingEngine
import run_agent_route as heuristics

import tkinter.messagebox as _mb
import tkinter.filedialog as _fd
_mb.askyesno = lambda *a, **k: True
for _n in ('showinfo', 'showwarning', 'showerror'):
    setattr(_mb, _n, lambda *a, **k: None)

SCRATCH = _os.path.join(_rlpath.REPO_ROOT, 'rl', '_cut')


def _silence(obj):
    for name in ('_auto_save_game', '_draw', '_animate_view_to',
                 '_center_view_on_active_team', '_center_view_on_active_team_day'):
        setattr(obj, name, lambda *a, **k: None)
    return obj


def load_demo(path):
    _fd.askopenfilename = lambda *a, **k: path
    demo = _silence(game.PathfindingDemo())
    with contextlib.redirect_stdout(io.StringIO()):
        demo._load_game()
    return demo


def load_engine(path):
    """把存档载入无头引擎, 这样之后可以用 step() 继续打。"""
    _fd.askopenfilename = lambda *a, **k: path
    engine = NarutoMarchingEngine()
    _silence(engine)
    with contextlib.redirect_stdout(io.StringIO()):
        engine._load_game()
    return engine


def cut_to_day(demo, day, out_dir=SCRATCH):
    _os.makedirs(out_dir, exist_ok=True)
    path = _os.path.join(out_dir, 'cut_day%d.json' % day)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(demo._build_game_state_for_save(through_day=day), handle,
                  indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=_os.path.join(_rlpath.REPO_ROOT,
                                                        '全局70558_6.json'))
    parser.add_argument('--days', type=int, nargs='+',
                        default=[40, 50, 60, 70, 80, 85, 90])
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--out', default=_os.path.join(_rlpath.REPO_ROOT,
                                                       'hybrid_route_day97.json'))
    args = parser.parse_args()

    demo = load_demo(args.save)
    baseline = demo.total_reward
    print('原存档 %s: day %d, 积分 %s, 占领 %d'
          % (_os.path.basename(args.save), demo.current_day,
             format(baseline, ','), len(demo.all_visited_hexes)), flush=True)
    print()
    print('%-6s %10s %10s %10s %9s %s'
          % ('接手日', '接手时积分', '打完积分', '后半程得分', '占领', '对比原存档'))

    best = None
    for day in args.days:
        path = cut_to_day(demo, day)
        engine = load_engine(path)
        start_score = engine.total_reward
        started = time.perf_counter()
        heuristics.play_greedy(engine, args.food_weight)
        final = engine.total_reward
        delta = final - baseline
        print('%-6d %10s %10s %10s %9d %+9s  (%.0fs)'
              % (day, format(start_score, ','), format(final, ','),
                 format(final - start_score, ','), len(engine.all_visited_hexes),
                 format(delta, ','), time.perf_counter() - started), flush=True)
        if best is None or final > best[0]:
            best = (final, day, engine)

    final, day, engine = best
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(engine._build_game_state_for_save(None), handle,
                  indent=2, ensure_ascii=False)
    print()
    print('最好: 第 %d 天接手, 打完 %s 分 (%s 原存档 %s)'
          % (day, format(final, ','), '超过' if final > baseline else '不及',
             format(baseline, ',')))
    print('已导出 %s' % _os.path.basename(args.out))


if __name__ == '__main__':
    main()
