# -*- coding: utf-8 -*-
"""把动作空间从 MultiDiscrete[7,3,60,60,2] 换成一个很小的"候选下标"空间。

为什么必须换: 原来的动作空间有 7x3x60x60x2 = 129,600 种组合, 而任一时刻真正
合法的普通移动只有 6~18 种(相邻未占领格), 命中率 0.01% 量级。PPO 靠采样探索,
这种比例下它几乎永远只能拿到"非法动作"的惩罚, 再训多少步也学不会走路 ——
之前那次 100 回合的训练, 一半回合停在 day 1、分数恰好等于"一步不走"的保底 7500,
根源就在这里, 不是训练量不够。

新的动作空间:

    0 .. K-1   走到第 k 个候选格(候选 = 当前队伍相邻的、未占领的、可通行的格子,
               按价值从高到低排序, 最多取 K 个)
    K          换日
    K+1        建 2 队
    K+2        建 3 队
    K+3        用一次飞雷神(飞到本队连通领地外沿最值钱的一格)
    K+4        切换到下一支有步数的队伍

每一步都给出 action_masks(), 非法项直接屏蔽 —— 配合 sb3-contrib 的 MaskablePPO,
采样只会落在合法动作上。

观测里除了原来的全局量, 还给出每个候选格的 (积分, 粮草, 是否帐篷/卦/boss) 特征,
否则策略无从判断"第 3 个候选"到底是什么。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401  统一把 rl/ 与仓库根加进 sys.path, 并校正工作目录

import contextlib
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

K_CANDIDATES = 12          # 每步最多给策略看多少个候选格
FEATURES_PER_CANDIDATE = 6
MAX_EPISODE_STEPS = 2200   # 贪心整季用掉约 1,440 段, 留足余量


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class CandidateRouteEnv(gym.Env):
    """候选下标动作空间 + 动作掩码的远征环境。"""

    metadata = {'render_modes': []}

    def __init__(self, food_weight=0.30):
        super().__init__()
        self.food_weight = food_weight
        self.engine = NarutoMarchingEngine()
        self.reward_config = dict(TRAINING_REWARD)
        self.n_actions = K_CANDIDATES + 5
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            'globals': spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            'candidates': spaces.Box(-np.inf, np.inf,
                                     shape=(K_CANDIDATES, FEATURES_PER_CANDIDATE),
                                     dtype=np.float32),
        })
        self._steps = 0
        self._active_index = 0
        self._cands = []

    # ── 内部 ────────────────────────────────────────────────────────────
    def _teams(self):
        return [self.engine.team1, self.engine.team2, self.engine.team3]

    def _active_team(self):
        teams = self._teams()
        team = teams[self._active_index]
        if team is None:
            for i, t in enumerate(teams):
                if t is not None:
                    self._active_index = i
                    return t
        return team

    def _candidates(self):
        """当前队伍的相邻可占格, 按启发式价值排序, 最多 K 个。"""
        team = self._active_team()
        if team is None:
            return []
        here = tuple(team.full_path[-1])
        out = []
        for pos in game._neighbors(*here):
            if not game._passable(*pos) or pos in self.engine.all_visited_hexes:
                continue
            terrain = game._terrain(*pos)
            if terrain.get('name') == 'empty':
                continue
            out.append((heuristics.hex_value(self.engine, pos, self.food_weight, team),
                        pos, terrain))
        out.sort(key=lambda item: -item[0])
        return out[:K_CANDIDATES]

    def _observation(self):
        engine = self.engine
        team = self._active_team()
        steps = engine._get_team_steps_for_day(team, engine.current_day) if team else 0
        globals_vec = np.array([
            engine.current_day / float(game.TOTAL_DAYS),
            max(0, engine.current_food) / 6800.0,
            engine.total_food / 160000.0,
            engine.total_reward / 70000.0,
            len(engine.all_visited_hexes) / 1200.0,
            len(engine.visited_g_lands) / max(1, len(engine.all_g_lands)),
            sum(1 for t in self._teams() if t is not None) / 3.0,
            steps / float(DAILY_STEP_GRANT),
        ], dtype=np.float32)

        feats = np.zeros((K_CANDIDATES, FEATURES_PER_CANDIDATE), dtype=np.float32)
        for i, (_value, pos, terrain) in enumerate(self._cands):
            token = game.RAW_MAP[pos[0]][pos[1]]
            feats[i] = (
                terrain.get('award', 0) / 100.0,
                terrain.get('food', 0) / 150.0,
                1.0 if token in ('G', 'g') else 0.0,
                1.0 if token == 'T' else 0.0,
                1.0 if token in ('B', 'b', 'L') else 0.0,
                1.0,                      # 这一项存在(用来和补零的行区分)
            )
        return {'globals': globals_vec, 'candidates': feats}

    def action_masks(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[:len(self._cands)] = True
        team = self._active_team()
        engine = self.engine
        mask[K_CANDIDATES] = True                                   # 换日永远合法
        mask[K_CANDIDATES + 1] = (engine.team2 is None and team is not None)
        mask[K_CANDIDATES + 2] = (engine.team2 is not None
                                  and engine.team3 is None and team is not None)
        mask[K_CANDIDATES + 3] = engine.fly_skill_limit > 0 and team is not None
        mask[K_CANDIDATES + 4] = sum(1 for t in self._teams() if t is not None) > 1
        return mask

    # ── gym 接口 ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _quiet(self.engine.reset)
        self._steps = 0
        self._active_index = 0
        self._cands = self._candidates()
        return self._observation(), {}

    def step(self, action):
        engine = self.engine
        action = int(action)
        before_score = engine.total_reward
        before_food = engine.total_food
        before_taken = len(engine.all_visited_hexes)
        before_g = len(engine.visited_g_lands)
        before_day = engine.current_day
        team = self._active_team()
        team_number = self._active_index + 1
        illegal = False

        if action < K_CANDIDATES:
            if action < len(self._cands):
                target = self._cands[action][1]
                if not heuristics.try_move(engine, team_number, target,
                                           target in heuristics.PORTAL_PAIRS):
                    illegal = True
            else:
                illegal = True
        elif action == K_CANDIDATES:
            heuristics.advance(engine)
        elif action == K_CANDIDATES + 1:
            illegal = not heuristics.try_create(engine, 2)
        elif action == K_CANDIDATES + 2:
            illegal = not heuristics.try_create(engine, 3)
        elif action == K_CANDIDATES + 3:
            illegal = not heuristics.try_fly(engine, team_number, team,
                                             self.food_weight)
        else:
            alive = [i for i, t in enumerate(self._teams()) if t is not None]
            if alive:
                self._active_index = alive[(alive.index(self._active_index) + 1)
                                           % len(alive)] if self._active_index in alive \
                    else alive[0]

        # 建队之后新队伍成为激活队伍, 同步下标
        for i, t in enumerate(self._teams()):
            if t is engine.active_team:
                self._active_index = i
                break

        self._steps += 1
        self._cands = self._candidates()

        # ── 奖励(训练分 RP, 与游戏积分是两套数, 见 rl_gym_env 顶部说明) ──
        cfg = self.reward_config
        reward = 0.0
        reward += cfg['score'] * ((engine.total_reward - before_score) / SCORE_PER_RP)
        reward += cfg['new_hex'] * (len(engine.all_visited_hexes) - before_taken)
        reward -= cfg['food_cost'] * max(0, engine.total_food - before_food)
        reward += cfg['day_progress'] * max(0, engine.current_day - before_day)
        if before_g < len(engine.all_g_lands) <= len(engine.visited_g_lands):
            reward += cfg['g_complete']
        if illegal:
            reward -= cfg['invalid_action']

        terminated = engine.current_day >= game.TOTAL_DAYS
        truncated = self._steps >= MAX_EPISODE_STEPS
        if terminated:
            reward += cfg['completion']

        info = {
            'game_score': engine.total_reward,
            'occupied': len(engine.all_visited_hexes),
            'day': engine.current_day,
            'illegal': illegal,
        }
        return self._observation(), float(reward), terminated, truncated, info
