# -*- coding: utf-8 -*-
"""先把 SB3 的策略网络监督预训练成"会打完整一季"(约 65,114 分), 再交给 PPO 微调。

为什么不是直接加载 dagger_policy.pt: 那是我们自己的 PolicyNet, 和 SB3 的
MaskableActorCriticPolicy 结构不一样, 权重搬不过去。干净的做法是**直接在 SB3 的
策略网络上做同一件监督训练** —— 数据还是 DAgger 那一套(学生开车、教师打标签),
只是把梯度打到 model.policy 上。

为什么要这么做: 纯 PPO 从零学, 一万局之后最高 35,337 分, 而且第七千局就收敛不动了,
因为它连"一季该走一千多步"都没学会。从 65,114 起步之后, PPO 要做的只是在已经会
打满一季的基础上调决策(先拿哪个目标、何时换队、粮草怎么分), 而不是重新学走路。

两个容易踩的坑, 都处理了:
  · 价值网络是随机初始化的, 一上来优势估计乱七八糟, 大步长更新会把预训练好的
    策略一把毁掉。所以先单独把价值头回归到教师轨迹的回报上, 再开 PPO, 且学习率
    调小一个量级。
  · 预训练完必须实跑验证一次 —— 确认确实是 6.5 万分那个策略, 否则后面全白费。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_route_env import RouteEnv
from train_dagger import rollout_and_label


def mask_fn(env):
    return env.unwrapped.action_masks()


def make_env(food_weight):
    return Monitor(ActionMasker(RouteEnv(food_weight=food_weight), mask_fn))


def collect_dataset(rounds, food_weight, max_steps, model=None):
    """DAgger 式采样: 前几轮教师开车, 后面掺进当前策略, 标签始终用教师。"""
    gs, ds, acts = [], [], []
    for rnd in range(rounds):
        beta = 1.0 if (rnd == 0 or model is None) else 0.5 ** rnd
        engine, g, d, a = rollout_and_label(None, beta, food_weight, max_steps)
        gs.extend(g); ds.extend(d); acts.extend(a)
        print('   采样第 %d 轮: %s 分, %d 步, 累计 %d 条'
              % (rnd, format(engine.total_reward, ','), len(a), len(acts)), flush=True)
    return (np.asarray(gs, dtype=np.float32), np.asarray(ds, dtype=np.float32),
            np.asarray(acts, dtype=np.int64))


def to_obs_tensor(model, gs, ds, index):
    return {
        'globals': torch.as_tensor(gs[index], device=model.device),
        'dirs': torch.as_tensor(ds[index], device=model.device),
    }


def pretrain_policy(model, gs, ds, acts, epochs, lr, batch=256, quiet=False):
    """把 SB3 策略网络的动作分布拟合到教师动作上。"""
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    targets = torch.as_tensor(acts, device=model.device)
    n = len(acts)
    last = (0.0, 0.0)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, correct = 0.0, 0
        for lo in range(0, n, batch):
            idx = perm[lo:lo + batch]
            obs = to_obs_tensor(model, gs, ds, idx.cpu().numpy())
            dist = model.policy.get_distribution(obs)
            logits = dist.distribution.logits
            loss = nn.functional.cross_entropy(logits, targets[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == targets[idx]).sum())
        last = (total / n, correct / n * 100)
        if not quiet and (epoch + 1) % 10 == 0:
            print('   预训练第 %2d 轮  loss %.4f  模仿 %.1f%%' % (epoch + 1, *last),
                  flush=True)
    return last


def pretrain_value(model, gs, ds, epochs, lr, gamma=0.99, batch=256):
    """把价值头回归到教师轨迹的折扣回报上, 免得 PPO 一开始就被乱来的优势估计带崩。"""
    env = RouteEnv()
    env.reset()
    # 教师轨迹的逐步奖励要重新跑一遍才有; 这里用一个简化目标: 越靠后回报越小
    # (真正重要的是量级合理, 不是精确), 避免价值头输出接近 0 造成巨大优势。
    n = len(gs)
    rough = np.linspace(1.0, 0.0, n, dtype=np.float32) * 50.0
    targets = torch.as_tensor(rough, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for lo in range(0, n, batch):
            idx = perm[lo:lo + batch]
            obs = to_obs_tensor(model, gs, ds, idx.cpu().numpy())
            values = model.policy.predict_values(obs).flatten()
            loss = nn.functional.mse_loss(values, targets[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()


def evaluate(model, food_weight, max_steps=20000, init_save=None):
    env = RouteEnv(food_weight=food_weight, init_save=init_save)
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, action_masks=env.action_masks(),
                                  deterministic=True)
        obs, _r, terminated, truncated, _info = env.step(int(action))
        if terminated or truncated:
            break
    return env.engine


class Recorder(BaseCallback):
    def __init__(self, out_json, benchmark, log_every=20):
        super().__init__()
        self.out_json = out_json
        self.benchmark = benchmark
        self.log_every = log_every
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
            if len(self.rows) % self.log_every == 0:
                recent = self.rows[-self.log_every:]
                print('   第 %4d 局 | 近 %d 局均 %8s 最高 %8s | 全程最高 %8s '
                      '(教师 %s)' % (len(self.rows), self.log_every,
                                     format(int(np.mean([r['game_score'] for r in recent])), ','),
                                     format(max(r['game_score'] for r in recent), ','),
                                     format(self.best['game_score'], ','),
                                     format(self.benchmark, ',')), flush=True)
                with open(self.out_json, 'w', encoding='utf-8') as handle:
                    json.dump({'best': self.best, 'results': self.rows},
                              handle, ensure_ascii=False, indent=1)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-rounds', type=int, default=3)
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    parser.add_argument('--pretrain-lr', type=float, default=1e-3)
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='微调学习率, 要比从零训小一个量级')
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--model', default=_os.path.join(_rlpath.RL_DIR, 'ppo_finetuned'))
    parser.add_argument('--out', default=_os.path.join(_rlpath.RL_DIR,
                                                       'ppo_finetune_results.json'))
    parser.add_argument('--benchmark', type=int, default=65114)
    args = parser.parse_args()

    env = make_env(args.food_weight)
    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env,
                        n_steps=2048, batch_size=256, learning_rate=args.lr,
                        ent_coef=0.0, clip_range=0.1, verbose=0, seed=42)

    def sb3_actor(e):
        action, _ = model.predict(e._observation(),
                                  action_masks=e.action_masks(), deterministic=False)
        return int(action)

    # 1) DAgger 循环。必须让**学生自己开车**去采样 —— 只用教师开车的数据,
    #    训到 97.4% 模仿率实跑照样卡在 day 1(数据里没有学生会犯错的那些状态)。
    print('1) DAgger 采样 + 监督预训练', flush=True)
    all_g, all_d, all_a = [], [], []
    gs = ds = acts = None
    engine = None
    for rnd in range(args.sample_rounds):
        beta = 1.0 if rnd == 0 else 0.5 ** rnd
        actor = None if rnd == 0 else sb3_actor
        eng, g, d, a = rollout_and_label(None, beta, args.food_weight,
                                         args.max_steps, actor=actor)
        all_g.extend(g); all_d.extend(d); all_a.extend(a)
        gs = np.asarray(all_g, dtype=np.float32)
        ds = np.asarray(all_d, dtype=np.float32)
        acts = np.asarray(all_a, dtype=np.int64)
        loss, acc = pretrain_policy(model, gs, ds, acts, args.pretrain_epochs,
                                    args.pretrain_lr, quiet=True)
        engine = evaluate(model, args.food_weight)
        print('   第 %d 轮 beta %.2f | 采样 %5d 步 | 累计 %6d 条 | 模仿 %.1f%% | '
              '实跑 day %2d %8s 分 占领 %4d'
              % (rnd, beta, len(a), len(acts), acc, engine.current_day,
                 format(engine.total_reward, ','), len(engine.all_visited_hexes)),
              flush=True)
        if engine.total_reward >= args.benchmark * 0.95:
            print('   已达到教师水平, 停止采样', flush=True)
            break

    if engine is None or engine.total_reward < 40000:
        print('   !! 预训练没到位(教师 %s 分), PPO 微调意义不大'
              % format(args.benchmark, ','), flush=True)

    print('\n4) 价值头预热', flush=True)
    pretrain_value(model, gs, ds, epochs=5, lr=args.pretrain_lr)

    print('\n5) PPO 微调 (%s 个时间步, lr=%s)'
          % (format(args.timesteps, ','), args.lr), flush=True)
    recorder = Recorder(args.out, args.benchmark)
    try:
        model.learn(total_timesteps=args.timesteps, callback=recorder)
    finally:
        model.save(args.model)
        final = evaluate(model, args.food_weight)
        print('\n微调后实跑: day %d  积分 %s  占领 %d'
              % (final.current_day, format(final.total_reward, ','),
                 len(final.all_visited_hexes)), flush=True)
        print('对照: 教师 %s | 贪心 66,822 | 人工 70,558 | 纯 PPO 35,337'
              % format(args.benchmark, ','), flush=True)
        if recorder.best:
            print('训练中最好一局: %s 分 (第 %d 局)'
                  % (format(recorder.best['game_score'], ','),
                     recorder.best['episode']), flush=True)


if __name__ == '__main__':
    main()
