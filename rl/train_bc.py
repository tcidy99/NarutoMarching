# -*- coding: utf-8 -*-
"""行为克隆: 先让网络学会模仿贪心策略, 再交给 PPO 在这个基础上继续练。

为什么要这一步: 直接用 PPO 从零学, 一万局之后最高只有 35,337 分(贪心 66,822 的
52.9%), 而且在第七千局就收敛不动了 —— 它连"一季该走多少步"都没学会(一季只走 407
步, 贪心要 1,442 步)。行为克隆先把"打满一季长什么样"灌进去, PPO 就能从六万多分
往上微调, 而不是从八千分一路爬。

老师是贪心策略(见 greedy_policy.py), 不是人工存档 —— 人工存档翻译成动作序列需要
精确复现当时的控制流, 实测 73% 的步对不上; 贪心的控制流是自己写的, 构造上就能翻译。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import greedy_policy
from rl_route_env import RouteEnv, N_ACTIONS, N_DIRS, FEATURES_PER_DIR


class PolicyNet(nn.Module):
    """观测 -> 18 个动作的打分。结构刻意做小, 样本只有几千条。"""

    def __init__(self, n_globals=10, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_globals + N_DIRS * FEATURES_PER_DIR, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, globals_vec, dirs):
        x = torch.cat([globals_vec, dirs.flatten(start_dim=1)], dim=1)
        return self.body(x)


def collect(episodes, food_weight):
    gs, ds, acts = [], [], []
    for i in range(episodes):
        env, g, d, a = greedy_policy.rollout(collect=True, food_weight=food_weight)
        gs.extend(g); ds.extend(d); acts.extend(a)
        print('   第 %d 局: %s 分, 占领 %d, 样本 %d'
              % (i + 1, format(env.engine.total_reward, ','),
                 len(env.engine.all_visited_hexes), len(a)), flush=True)
    return (np.asarray(gs, dtype=np.float32), np.asarray(ds, dtype=np.float32),
            np.asarray(acts, dtype=np.int64))


def evaluate(net, food_weight, greedy=True):
    """用学到的网络实跑一季。"""
    env = RouteEnv(food_weight=food_weight)
    env.reset()
    for _ in range(20000):
        obs = env._observation()
        with torch.no_grad():
            logits = net(torch.from_numpy(obs['globals']).unsqueeze(0),
                         torch.from_numpy(obs['dirs']).unsqueeze(0))[0]
        logits[~torch.from_numpy(env.action_masks())] = -1e9   # 屏蔽非法动作
        action = int(torch.argmax(logits)) if greedy else \
            int(torch.distributions.Categorical(logits=logits).sample())
        env.apply(action)
        if env.engine.current_day >= 97:
            break
    return env.engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--out', default=_os.path.join(_rlpath.RL_DIR, 'bc_policy.pt'))
    args = parser.parse_args()

    print('收集教师数据(贪心策略跑 %d 局):' % args.episodes, flush=True)
    gs, ds, acts = collect(args.episodes, args.food_weight)
    print('共 %d 条样本, 动作分布 %s'
          % (len(acts), dict(zip(*np.unique(acts, return_counts=True)))), flush=True)

    net = PolicyNet(n_globals=gs.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tg = torch.from_numpy(gs); td = torch.from_numpy(ds); ta = torch.from_numpy(acts)

    print('\n开始训练:', flush=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(ta))
        total_loss = correct = 0
        for lo in range(0, len(ta), 256):
            idx = perm[lo:lo + 256]
            logits = net(tg[idx], td[idx])
            loss = loss_fn(logits, ta[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss) * len(idx)
            correct += int((logits.argmax(1) == ta[idx]).sum())
        if (epoch + 1) % 10 == 0:
            print('   第 %2d 轮  loss %.4f  模仿准确率 %.1f%%'
                  % (epoch + 1, total_loss / len(ta), correct / len(ta) * 100), flush=True)

    torch.save(net.state_dict(), args.out)
    print('\n已存 %s (%.0f 秒)' % (args.out, time.perf_counter() - started), flush=True)

    print('\n用学到的策略实跑一季:', flush=True)
    engine = evaluate(net, args.food_weight)
    print('   day %d  积分 %s  占领 %d  余粮 %d'
          % (engine.current_day, format(engine.total_reward, ','),
             len(engine.all_visited_hexes), engine.current_food))
    print('   对照: 贪心教师 65,114 / 原贪心 66,822 / 人工 70,558 / 纯 PPO 最高 35,337')


if __name__ == '__main__':
    main()
