# -*- coding: utf-8 -*-
"""方向式动作空间: Discrete(12)。为行为克隆而设计。

为什么从"按价值排序的候选下标"换成"方向":

  · 覆盖率。把人工存档 全局70558_6.json 的 723 个动作拆成逐格步(共 1,267 步)之后:
        相邻单步占格          1,150   90.8%
        跳步(踩相邻已占领格)     109    8.6%
        传送/飞雷神/结算试探格      8    0.6%
    也就是说"多格移动"的中间格在走到它的那一刻都还没被占领, 拆成连续单步是**无损**的。
    只要动作里同时允许"走到相邻未占领格"和"走到相邻已占领格(跳步)", 覆盖率就到 99.4%。
    根本不需要几十个远程目标候选。

  · 标签唯一。行为克隆要给每一步一个确定的标签。按价值排序的候选下标会随估值函数
    漂移(同一格今天排第 1、改个权重就排第 3), 方向下标不会。

动作:
     0-5   走到第 d 个方向的相邻格(不传送) —— 未占领就占领, 已占领就是跳步(10 粮)
     6-11  同上, 但落点是传送点时顺势传到对岸(只有落点真是传送点时才合法)
    12     换日
    13     建 2 队
    14     建 3 队
    15     结算脚下的试探格(蹭步之后确认占领)
    16     用飞雷神(飞到本队连通领地外沿最值钱的一格; 全图只有 3 次, 不值得再细分)
    17     切到下一支队伍

方向下标按 hex_core._neighbors 里那两张表的顺序, 偶数列和奇数列各一张 —— 它们是同
一组六个几何方向的两种写法, 所以下标 d 在几何上是一致的。越界的方向直接屏蔽。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import contextlib
import copy
import io

import numpy as np

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction
from rl_gym_env import TRAINING_REWARD, SCORE_PER_RP, DAILY_STEP_GRANT
import run_agent_route as heuristics

import gymnasium as gym
from gymnasium import spaces

import tkinter.messagebox as _mb
_mb.askyesno = lambda *a, **k: True
for _n in ('showinfo', 'showwarning', 'showerror'):
    setattr(_mb, _n, lambda *a, **k: None)

N_DIRS = 6
# 踩到传送点时"传不传"是玩家的选择, 不能自动替他传 —— 人工存档里就有踩上传送点
# 却不传的步(段24 踩 (25,54)=P3 没传, 我这边自动把队伍甩到 (53,41), 之后整条路线
# 全崩)。所以方向动作分成两套: 0-5 不传送, 6-11 落点是传送点时顺势传过去。
A_DIR_TELEPORT = 6          # 6..11 = 方向 0..5 + 传送
A_ADVANCE_DAY = 12
A_CREATE_T2 = 13
A_CREATE_T3 = 14
A_SETTLE = 15
A_FLY = 16
A_SWITCH_TEAM = 17
N_ACTIONS = 18

FEATURES_PER_DIR = 10
MAX_EPISODE_STEPS = 2600     # 人工整季 1,267 步, 贪心 1,442 段, 留足余量


def direction_offsets(ic):
    """第 d 个方向相对当前格的偏移。与 hex_core._neighbors 的两张表一致。"""
    if ic % 2 == 0:
        return [(+1, 0), (0, +1), (-1, +1), (-1, 0), (-1, -1), (0, -1)]
    return [(+1, 0), (+1, +1), (0, +1), (-1, 0), (0, -1), (+1, -1)]


def neighbor_in_direction(pos, d):
    """(ir, ic) 往第 d 个方向走一格; 越界返回 None。"""
    ir, ic = pos
    dr, dc = direction_offsets(ic)[d]
    nr, nc = ir + dr, ic + dc
    if 0 <= nr < game.ROWS and 0 <= nc < game.COLS:
        return (nr, nc)
    return None


def direction_of(pos, target):
    """target 在 pos 的第几个方向; 不相邻返回 None。"""
    for d in range(N_DIRS):
        if neighbor_in_direction(pos, d) == tuple(target):
            return d
    return None


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class RouteEnv(gym.Env):
    """方向式动作 + 动作掩码的远征环境。"""

    metadata = {'render_modes': []}

    def __init__(self, food_weight=0.30, init_save=None):
        """init_save 给一份存档路径时, 每局从那份存档的局面开始打, 而不是从开局。

        用途: 保留人工路线前若干天的强开局, 只让 agent 优化后半程。存档可以用
        _build_game_state_for_save(through_day=N) 先截断到第 N 天(见 resume_from_save.py)。

        每局都重新读盘太慢(五千局就是五千次), 所以只读一次当模板, 之后每局深拷贝。
        """
        super().__init__()
        self.food_weight = food_weight
        self.init_save = init_save
        self._template = None
        self.engine = NarutoMarchingEngine()
        if init_save:
            self._template = self._load_template(init_save)
        self.reward_config = dict(TRAINING_REWARD)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            'globals': spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            'dirs': spaces.Box(-np.inf, np.inf,
                               shape=(N_DIRS, FEATURES_PER_DIR), dtype=np.float32),
        })
        self._steps = 0
        self._active_index = 0
        self._nav_cache_key = None
        self._nav_cache = (None, None)

    def _load_template(self, path):
        import tkinter.filedialog as _fd
        _fd.askopenfilename = lambda *a, **k: path
        engine = NarutoMarchingEngine()
        for name in ('_auto_save_game', '_draw', '_animate_view_to',
                     '_center_view_on_active_team', '_center_view_on_active_team_day'):
            setattr(engine, name, lambda *a, **k: None)
        with contextlib.redirect_stdout(io.StringIO()):
            engine._load_game()
        return engine

    # ── 状态 ────────────────────────────────────────────────────────────
    def _teams(self):
        return [self.engine.team1, self.engine.team2, self.engine.team3]

    def _active_team(self):
        teams = self._teams()
        if teams[self._active_index] is None:
            for i, t in enumerate(teams):
                if t is not None:
                    self._active_index = i
                    break
        return self._teams()[self._active_index]

    def _sync_active_index(self):
        for i, t in enumerate(self._teams()):
            if t is not None and t is self.engine.active_team:
                self._active_index = i
                return

    def _here(self):
        team = self._active_team()
        return tuple(team.full_path[-1]) if team is not None else None

    def _observation(self):
        engine = self.engine
        team = self._active_team()
        here = self._here()
        steps = engine._get_team_steps_for_day(team, engine.current_day) if team else 0
        # 三支队伍各自的剩余步数都要给出来, 不能只给当前这一支 ——
        # "该切队还是该换日"完全取决于别的队伍还能不能动。之前观测里只有激活队伍的
        # 步数, 学生根本判断不了, 于是学会了无限切队(DAgger 模仿率 94.5% 却卡在 day 2)。
        per_team_steps = []
        for t in self._teams():
            if t is None:
                per_team_steps.append(-1.0)
            else:
                per_team_steps.append(
                    engine._get_team_steps_for_day(t, engine.current_day)
                    / float(DAILY_STEP_GRANT))
        others_can_move = any(
            v > 0 for i, v in enumerate(per_team_steps) if i != self._active_index)
        globals_vec = np.array([
            engine.current_day / float(game.TOTAL_DAYS),
            max(0, engine.current_food) / 6800.0,
            engine.total_food / 160000.0,
            engine.total_reward / 70000.0,
            len(engine.all_visited_hexes) / 1200.0,
            len(engine.visited_g_lands) / max(1, len(engine.all_g_lands)),
            sum(1 for t in self._teams() if t is not None) / 3.0,
            steps / float(DAILY_STEP_GRANT),
            engine.fly_skill_limit / 3.0,
            1.0 if (team is not None and team.free_exploration_hexes) else 0.0,
            per_team_steps[0], per_team_steps[1], per_team_steps[2],
            1.0 if others_can_move else 0.0,
            # 粮草见底时谁都动不了, 只能换日等 +1600 —— 这也是教师的判据之一
            1.0 if engine.current_food < heuristics.FOOD_FLOOR else 0.0,
            self._active_index / 2.0,
        ], dtype=np.float32)

        feats = np.zeros((N_DIRS, FEATURES_PER_DIR), dtype=np.float32)
        # 导航线索每一步都要算一次 A*/BFS, 是整个环境最慢的地方(五千局要十小时)。
        # 只要"这支队伍站在哪 + 地图占领了多少格"没变, 结果就不会变, 拿这个当缓存键。
        cache_key = (id(team), here, len(engine.all_visited_hexes),
                     len(engine.visited_g_lands))
        if self._nav_cache_key == cache_key:
            objective_next, frontier_next = self._nav_cache
        else:
            objective_next = self._next_hop_to(heuristics.objective_step(engine, team)
                                               if team is not None else None)
            frontier_next = self._next_hop_to(
                heuristics.nearest_frontier(engine, team, self.food_weight)
                if team is not None else None)
            self._nav_cache_key = cache_key
            self._nav_cache = (objective_next, frontier_next)
        if here is not None:
            for d in range(N_DIRS):
                nb = neighbor_in_direction(here, d)
                if nb is None or not game._passable(*nb):
                    continue
                terrain = game._terrain(*nb)
                if terrain.get('name') == 'empty':
                    continue
                token = game.RAW_MAP[nb[0]][nb[1]]
                occupied = nb in engine.all_visited_hexes
                feats[d] = (
                    1.0,                                   # 这个方向可走
                    0.0 if occupied else 1.0,              # 未占领(能占地)
                    terrain.get('award', 0) / 100.0,
                    terrain.get('food', 0) / 150.0,
                    1.0 if token in ('G', 'g') else 0.0,
                    1.0 if token == 'T' else 0.0,
                    1.0 if token in ('B', 'b', 'L', 'M') else 0.0,
                    1.0 if token.startswith('P') else 0.0,
                    # 导航线索: 这个方向是不是"通往当前战略目标/最近边界"的下一跳。
                    # 没有这两项的话, 选方向这件事在观测里根本没有足够信息 ——
                    # 教师是靠跨图 A*/BFS 决定往哪走的, 而观测只看得见相邻六格,
                    # 实测模仿准确率卡在 84%(留出集 84.5%, 和训练集持平, 不是过拟合),
                    # 且错误全部集中在方向上(换日/切队/传送/结算都是 100%)。
                    1.0 if nb == objective_next else 0.0,
                    1.0 if nb == frontier_next else 0.0,
                )
        return {'globals': globals_vec, 'dirs': feats}

    def _next_hop_to(self, target):
        """朝 target 走的下一格(相邻就是它自己, 否则取 A* 的第二个点)。"""
        here = self._here()
        if here is None or target is None:
            return None
        target = tuple(target)
        if direction_of(here, target) is not None:
            return target
        path, _cost = game._astar(here, target, set())
        return tuple(path[1]) if path and len(path) >= 2 else None

    def action_masks(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        engine = self.engine
        team = self._active_team()
        here = self._here()
        if here is not None:
            for d in range(N_DIRS):
                nb = neighbor_in_direction(here, d)
                if nb is None or not game._passable(*nb):
                    continue
                if game._terrain(*nb).get('name') == 'empty':
                    continue
                mask[d] = True
                # 只有落点确实是传送点时, "顺势传过去"这一版才合法
                if nb in heuristics.PORTAL_PAIRS:
                    mask[A_DIR_TELEPORT + d] = True
        mask[A_ADVANCE_DAY] = True
        mask[A_CREATE_T2] = engine.team2 is None and team is not None
        mask[A_CREATE_T3] = (engine.team2 is not None and engine.team3 is None
                             and team is not None)
        mask[A_SETTLE] = bool(team is not None and team.free_exploration_hexes)
        mask[A_FLY] = engine.fly_skill_limit > 0 and team is not None
        mask[A_SWITCH_TEAM] = sum(1 for t in self._teams() if t is not None) > 1
        return mask

    # ── gym 接口 ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._template is not None:
            self.engine = copy.deepcopy(self._template)
        else:
            _quiet(self.engine.reset)
        self._steps = 0
        self._active_index = 0
        self._nav_cache_key = None
        self._nav_cache = (None, None)
        return self._observation(), {}

    def apply(self, action):
        """执行一个动作。返回 (是否合法)。训练和行为克隆都走这里。"""
        engine = self.engine
        team = self._active_team()
        team_number = self._active_index + 1
        action = int(action)

        if action < A_ADVANCE_DAY:
            teleport = action >= A_DIR_TELEPORT
            direction = action - A_DIR_TELEPORT if teleport else action
            here = self._here()
            nb = neighbor_in_direction(here, direction) if here else None
            if nb is None:
                return False
            if teleport and nb not in heuristics.PORTAL_PAIRS:
                return False
            # 引擎内部拒绝一步时并不抛异常(只写一条状态消息), 所以不能只看
            # try_move 的返回值 —— 要核对队伍是不是真的挪了位置。
            before = tuple(team.full_path[-1]) if team is not None else None
            heuristics.try_move(engine, team_number, nb, teleport)
            self._sync_active_index()
            moved = self._active_team()
            after = tuple(moved.full_path[-1]) if moved is not None else None
            return after is not None and after != before
        if action == A_ADVANCE_DAY:
            heuristics.advance(engine)
            return True
        if action == A_CREATE_T2:
            okay = heuristics.try_create(engine, 2)
            self._sync_active_index()
            return okay
        if action == A_CREATE_T3:
            okay = heuristics.try_create(engine, 3)
            self._sync_active_index()
            return okay
        if action == A_SETTLE:
            if team is None or not team.free_exploration_hexes:
                return False
            engine.active_team = team
            try:
                _quiet(engine.step, RouteAction(
                    action_type=ActionType.SETTLE_EXPLORATION, team=team_number))
            except Exception:
                return False
            return True
        if action == A_FLY:
            return heuristics.try_fly(engine, team_number, team, self.food_weight)
        # 切队
        alive = [i for i, t in enumerate(self._teams()) if t is not None]
        if len(alive) <= 1:
            return False
        cur = self._active_index if self._active_index in alive else alive[0]
        self._active_index = alive[(alive.index(cur) + 1) % len(alive)]
        engine.active_team = self._teams()[self._active_index]
        return True

    def step(self, action):
        engine = self.engine
        before = (engine.total_reward, engine.total_food,
                  len(engine.all_visited_hexes), engine.current_day,
                  len(engine.visited_g_lands))
        legal = self.apply(action)
        self._steps += 1

        cfg = self.reward_config
        reward = cfg['score'] * ((engine.total_reward - before[0]) / SCORE_PER_RP)
        reward += cfg['new_hex'] * (len(engine.all_visited_hexes) - before[2])
        reward -= cfg['food_cost'] * max(0, engine.total_food - before[1])
        reward += cfg['day_progress'] * max(0, engine.current_day - before[3])
        if before[4] < len(engine.all_g_lands) <= len(engine.visited_g_lands):
            reward += cfg['g_complete']
        if not legal:
            reward -= cfg['invalid_action']

        terminated = engine.current_day >= game.TOTAL_DAYS
        truncated = self._steps >= MAX_EPISODE_STEPS
        if terminated:
            reward += cfg['completion']
        info = {
            'game_score': engine.total_reward,
            'occupied': len(engine.all_visited_hexes),
            'day': engine.current_day,
            'illegal': not legal,
        }
        return self._observation(), float(reward), terminated, truncated, info
