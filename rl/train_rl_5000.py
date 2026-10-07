# -*- coding: utf-8 -*-
"""跑 N 局强化学习微调, 并把**最高分那一局的路线**存成软件能读的存档。

流程:
  1. DAgger 监督预训练 —— 把 SB3 的策略网络先训成"会打完整一季"(约 65,874 分)。
     这一步不是强化学习, 是模仿学习; 目的只是给 PPO 一个像样的起点(纯 PPO 从零
     练一万局最高才 35,337, 因为它连一季该走一千多步都学不会)。
  2. MaskablePPO 微调 N 局。这一步才是强化学习 —— 用的是 rl_gym_env 里那套 RP
     奖励表, 靠试错自己找比教师更好的走法。
  3. 全程盯着每一局的**游戏积分**, 一旦刷新纪录就立刻把那一局的路线导出成 json。

为什么要在回合结束的那一刻存: SB3 的向量环境在回合结束后会自动 reset, 等回调里
看到 episode 信息时路线已经没了。所以用一个 Wrapper 卡在 reset 之前把它存下来。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import argparse
import json
import time

import numpy as np
import gymnasium as gym

import matplotlib
matplotlib.use('Agg')
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_route_env import RouteEnv
from train_dagger import rollout_and_label
from train_ppo_finetune import pretrain_policy, pretrain_value, evaluate


class BestRouteSaver(gym.Wrapper):
    """回合一结束就看分数, 刷新纪录就把这一局的完整路线写成存档。"""

    def __init__(self, env, out_path, record_path, baseline=0):
        super().__init__(env)
        self.out_path = out_path
        self.record_path = record_path
        self.best = baseline
        self.saved = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            engine = self.env.unwrapped.engine
            score = engine.total_reward
            if score > self.best:
                self.best = score
                state = engine._build_game_state_for_save(None)
                with open(self.out_path, 'w', encoding='utf-8') as handle:
                    json.dump(state, handle, indent=2, ensure_ascii=False)
                self.saved += 1
                with open(self.record_path, 'w', encoding='utf-8') as handle:
                    json.dump({'best_score': int(score),
                               'occupied': len(engine.all_visited_hexes),
                               'day': engine.current_day,
                               'saved_times': self.saved,
                               'file': _os.path.basename(self.out_path)},
                              handle, ensure_ascii=False, indent=1)
                print('   *** 新纪录 %s 分 (占领 %d, day %d) -> 已存 %s'
                      % (format(int(score), ','), len(engine.all_visited_hexes),
                         engine.current_day, _os.path.basename(self.out_path)),
                      flush=True)
        return obs, reward, terminated, truncated, info


class EpisodeStop(BaseCallback):
    """数满 N 局就停, 顺便记录进度。"""

    def __init__(self, target, out_json, benchmark, model_path, log_every=25,
                 save_every=250):
        super().__init__()
        self.target = target
        self.out_json = out_json
        self.benchmark = benchmark
        self.model_path = model_path
        self.log_every = log_every
        self.save_every = save_every
        self.rows = []
        self.best = None
        self.started = time.perf_counter()

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' not in info:
                continue
            row = {'episode': len(self.rows) + 1,
                   'game_score': int(info.get('game_score', 0)),
                   'occupied': int(info.get('occupied', 0)),
                   'final_day': int(info.get('day', 0)),
                   'steps': int(info['episode']['l']),
                   'elapsed': round(time.perf_counter() - self.started, 1)}
            self.rows.append(row)
            if self.best is None or row['game_score'] > self.best['game_score']:
                self.best = row
            n = len(self.rows)
            if n % self.log_every == 0:
                recent = self.rows[-self.log_every:]
                rate = n / max(1e-6, row['elapsed']) * 60
                print('   第 %4d/%d 局 | 近 %d 局均 %8s 最高 %8s | 全程最高 %8s '
                      '| %.1f 局/分 | 预计还需 %.1f 小时'
                      % (n, self.target, self.log_every,
                         format(int(np.mean([r['game_score'] for r in recent])), ','),
                         format(max(r['game_score'] for r in recent), ','),
                         format(self.best['game_score'], ','), rate,
                         (self.target - n) / max(1e-6, rate) / 60), flush=True)
                self._flush()
            if n % self.save_every == 0:
                self.model.save(self.model_path)
            if n >= self.target:
                self._flush()
                self.model.save(self.model_path)
                return False
        return True

    def _flush(self):
        with open(self.out_json, 'w', encoding='utf-8') as handle:
            json.dump({'episodes': len(self.rows), 'target': self.target,
                       'benchmark': self.benchmark, 'best': self.best,
                       'results': self.rows}, handle, ensure_ascii=False, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--dagger-rounds', type=int, default=6)
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    parser.add_argument('--pretrain-lr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--route-out', default=_os.path.join(
        _rlpath.REPO_ROOT, 'rl_best_route_day97.json'))
    parser.add_argument('--model', default=_os.path.join(_rlpath.RL_DIR, 'rl5000_model'))
    parser.add_argument('--out', default=_os.path.join(_rlpath.RL_DIR,
                                                       'rl5000_results.json'))
    parser.add_argument('--record', default=_os.path.join(_rlpath.RL_DIR,
                                                          'rl5000_best.json'))
    parser.add_argument('--benchmark', type=int, default=66822)
    parser.add_argument('--pretrain-target', type=int, default=60000,
                        help='预训练实跑达到这个分数就停止采样')
    parser.add_argument('--init-save', default=None,
                        help='从这份存档的局面开始打(通常是人工路线截断到第 N 天)')
    args = parser.parse_args()

    def mask_fn(e):
        return e.unwrapped.action_masks()

    base = RouteEnv(food_weight=args.food_weight, init_save=args.init_save)
    saver = BestRouteSaver(base, args.route_out, args.record)
    env = Monitor(ActionMasker(saver, mask_fn))

    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env,
                        n_steps=2048, batch_size=256, learning_rate=args.lr,
                        ent_coef=0.0, clip_range=0.1, verbose=0, seed=42)

    def sb3_actor(e):
        action, _ = model.predict(e._observation(),
                                  action_masks=e.action_masks(), deterministic=False)
        return int(action)

    print('第一步: DAgger 监督预训练(模仿学习, 不是强化学习)', flush=True)
    all_g, all_d, all_a = [], [], []
    engine = None
    for rnd in range(args.dagger_rounds):
        beta = 1.0 if rnd == 0 else 0.5 ** rnd
        actor = None if rnd == 0 else sb3_actor
        eng, g, d, a = rollout_and_label(None, beta, args.food_weight,
                                         args.max_steps, actor=actor,
                                         init_save=args.init_save)
        all_g.extend(g); all_d.extend(d); all_a.extend(a)
        gs = np.asarray(all_g, dtype=np.float32)
        ds = np.asarray(all_d, dtype=np.float32)
        acts = np.asarray(all_a, dtype=np.int64)
        _loss, acc = pretrain_policy(model, gs, ds, acts, args.pretrain_epochs,
                                     args.pretrain_lr, quiet=True)
        engine = evaluate(model, args.food_weight, init_save=args.init_save)
        print('   第 %d 轮 | 累计 %6d 条 | 模仿 %.1f%% | 实跑 day %2d %8s 分'
              % (rnd, len(acts), acc, engine.current_day,
                 format(engine.total_reward, ',')), flush=True)
        if engine.total_reward >= args.pretrain_target:
            break
    if engine is None or engine.total_reward < args.pretrain_target * 0.6:
        print('!! 预训练没到位, 强化学习意义不大 —— 中止', flush=True)
        return
    pretrain_value(model, gs, ds, epochs=5, lr=args.pretrain_lr)

    # 先把预训练策略的路线存进去当基线, 并把门槛设成它的分数 ——
    # 这样文件里任何时候都是一条能打满全季的好路线, 之后只有**真正超过起点**
    # 才覆盖。否则强化学习早期的随机探索(两万多分)会先把文件糊掉。
    with open(args.route_out, 'w', encoding='utf-8') as handle:
        json.dump(engine._build_game_state_for_save(None), handle,
                  indent=2, ensure_ascii=False)
    saver.best = engine.total_reward
    print('   起点路线已存: %s (%s 分), 之后只有超过它才会覆盖'
          % (_os.path.basename(args.route_out),
             format(engine.total_reward, ',')), flush=True)

    print('\n第二步: 强化学习微调 %d 局 (基准: 贪心 %s / 人工 70,558)'
          % (args.episodes, format(args.benchmark, ',')), flush=True)
    stopper = EpisodeStop(args.episodes, args.out, args.benchmark, args.model)
    try:
        model.learn(total_timesteps=args.episodes * 3000, callback=stopper)
    finally:
        model.save(args.model)
        print('\n跑了 %d 局。训练中最高 %s 分'
              % (len(stopper.rows),
                 format(stopper.best['game_score'], ',') if stopper.best else '-'),
              flush=True)
        print('最高分路线已存: %s (%s 分, 存过 %d 次)'
              % (args.route_out, format(int(saver.best), ','), saver.saved), flush=True)


if __name__ == '__main__':
    main()
