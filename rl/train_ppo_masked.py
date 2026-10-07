# -*- coding: utf-8 -*-
"""用 MaskablePPO 在候选动作空间上训练, 看看能不能超过启发式 agent 的成绩。

和旧的 train_ppo_100.py 的区别:
  · 动作空间是 Discrete(17) 的"候选下标", 不是 MultiDiscrete[7,3,60,60,2]
    (129,600 种组合里只有万分之一合法, 旧版就是卡死在这上面)
  · 每一步都给动作掩码, 采样只落在合法动作上 —— 用 sb3-contrib 的 MaskablePPO
  · 回合步数上限 2,200(启发式整季约 1,440 段; 旧版设 1,000, 连打完都不够)
  · 记录的是**游戏积分**(game_score), 训练奖励是另一套 RP, 两者不混

用法:
    python rl/train_ppo_masked.py --episodes 10000 --out rl/ppo_masked_results.json
    python rl/train_ppo_masked.py --episodes 50 --timesteps 50000    # 快速标定

随时可以看 --out 指定的 json 或同名 .csv 了解进度。

关于存盘和续练(容易踩坑, 写清楚):
  · 模型每 --save-every 个时间步存一次(默认 10 万), 训练结束时再存一次。
    中途被杀/断电只会丢掉最后不到一个存盘间隔的进度。
  · **续练必须显式加 --resume**。不加的话是从零开始练, 而且结束时会把原来那个
    模型覆盖掉 —— 想保住旧模型就换个 --model 路径。
  · --resume 时会带 reset_num_timesteps=False, 这样学习率调度接着上次走,
    而不是从头再来一遍。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import argparse
import csv
import json
import time

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_candidate_env import CandidateRouteEnv, MAX_EPISODE_STEPS


def mask_fn(env):
    return env.unwrapped.action_masks()


def make_env(food_weight):
    env = CandidateRouteEnv(food_weight=food_weight)
    env = ActionMasker(env, mask_fn)
    return Monitor(env)


class EpisodeRecorder(BaseCallback):
    """每局结束时记一行: 游戏积分/占领数/结束天/用时。同时滚动写盘。"""

    def __init__(self, target_episodes, out_json, out_csv, benchmark, log_every=25,
                 save_every=0, model_path=None):
        super().__init__()
        self.save_every = save_every
        self.model_path = model_path
        self._next_save = save_every
        self.target = target_episodes
        self.out_json = out_json
        self.out_csv = out_csv
        self.benchmark = benchmark
        self.log_every = log_every
        self.rows = []
        self.best = None
        self.started = time.perf_counter()

    def _on_step(self):
        # 定期存盘: 中途被杀也只丢最后不到一个间隔的进度
        if (self.save_every and self.model_path
                and self.num_timesteps >= self._next_save):
            self._next_save = self.num_timesteps + self.save_every
            self.model.save(self.model_path)
        for info in self.locals.get('infos', []):
            if 'episode' not in info:
                continue
            row = {
                'episode': len(self.rows) + 1,
                'game_score': int(info.get('game_score', 0)),
                'occupied': int(info.get('occupied', 0)),
                'final_day': int(info.get('day', 0)),
                'train_reward': float(info['episode']['r']),
                'steps': int(info['episode']['l']),
                'elapsed': round(time.perf_counter() - self.started, 1),
            }
            self.rows.append(row)
            if self.best is None or row['game_score'] > self.best['game_score']:
                self.best = row
            if len(self.rows) % self.log_every == 0:
                self._flush()
                recent = self.rows[-self.log_every:]
                print('  第 %5d 局 | 近 %d 局平均积分 %8s 最高 %8s | 全程最高 %8s '
                      '(启发式 %s) | %.0f 步/秒'
                      % (len(self.rows), self.log_every,
                         format(int(np.mean([r['game_score'] for r in recent])), ','),
                         format(max(r['game_score'] for r in recent), ','),
                         format(self.best['game_score'], ','),
                         format(self.benchmark, ','),
                         self.num_timesteps / max(1e-6, row['elapsed'])),
                      flush=True)
            if len(self.rows) >= self.target:
                self._flush()
                return False
        return True

    def _flush(self):
        elapsed = time.perf_counter() - self.started
        payload = {
            'episodes_completed': len(self.rows),
            'episodes_requested': self.target,
            'timesteps': int(self.num_timesteps),
            'elapsed_seconds': round(elapsed, 1),
            'heuristic_benchmark': self.benchmark,
            'best': self.best,
            'results': self.rows,
        }
        with open(self.out_json, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        with open(self.out_csv, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--timesteps', type=int, default=0,
                        help='0 = 按 episodes x MAX_EPISODE_STEPS 估')
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--out', default=_os.path.join(_rlpath.RL_DIR,
                                                       'ppo_masked_results.json'))
    parser.add_argument('--model', default=_os.path.join(_rlpath.RL_DIR,
                                                         'ppo_masked_model'))
    parser.add_argument('--benchmark', type=int, default=66822,
                        help='启发式 agent 的成绩, 只用来在日志里对照')
    parser.add_argument('--resume', action='store_true',
                        help='接着 --model 指定的模型继续练; 不加则从零开始并覆盖它')
    parser.add_argument('--save-every', type=int, default=100000,
                        help='每多少个时间步存一次盘, 0 = 只在结束时存')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    env = make_env(args.food_weight)
    total = args.timesteps or args.episodes * 900   # 早期回合短, 900 是个折中估计

    resuming = args.resume and _os.path.exists(args.model + '.zip')
    if resuming:
        model = MaskablePPO.load(args.model, env=env)
        print('接着上次的模型练:', args.model + '.zip')
    else:
        if _os.path.exists(args.model + '.zip'):
            print('!! 注意: %s.zip 已存在, 没加 --resume, 本次将从零开始练并在结束时'
                  '覆盖它。想保留旧模型请换 --model 路径。' % args.model, flush=True)
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy, env,
            n_steps=2048, batch_size=256, learning_rate=3e-4,
            ent_coef=0.01, verbose=0, seed=args.seed,
        )

    print('目标 %d 局 (约 %s 个时间步), 启发式基准 %s 分'
          % (args.episodes, format(total, ','), format(args.benchmark, ',')), flush=True)
    recorder = EpisodeRecorder(args.episodes, args.out,
                               _os.path.splitext(args.out)[0] + '.csv',
                               args.benchmark, save_every=args.save_every,
                               model_path=args.model)
    try:
        model.learn(total_timesteps=total, callback=recorder,
                    reset_num_timesteps=not resuming)
    finally:
        model.save(args.model)
        if recorder.rows:
            recorder._flush()
        print('已存模型:', args.model + '.zip', flush=True)
        if recorder.best:
            print('最好一局: 第 %d 局, 游戏积分 %s, 占领 %d, 结束于 day %d'
                  % (recorder.best['episode'], format(recorder.best['game_score'], ','),
                     recorder.best['occupied'], recorder.best['final_day']), flush=True)


if __name__ == '__main__':
    main()
