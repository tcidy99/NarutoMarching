# -*- coding: utf-8 -*-
"""DAgger: 治行为克隆的"误差累积"。

纯克隆的毛病: 训练时看到的全是教师走过的状态, 一旦自己走偏到教师没去过的地方就
不知道该干什么, 然后一路错下去 —— 实测 84% 的模仿准确率, 实跑却卡死在 day 1。

DAgger 的办法: **让学生自己走, 但让教师来标注它走到的每一个状态**。这样数据集里
就包含了"学生会犯错的那些状态"以及"在那儿该怎么纠正", 迭代几轮分布就收住了。

每一轮:
    1. 用当前策略(按 beta 的概率混入教师动作)跑一季, 记下沿途所有状态
    2. 每个状态都问教师"你会怎么走", 作为标签
    3. 并入数据集, 从头重训
    4. 实跑评估

beta 按 0.5^round 衰减: 第 0 轮全靠教师(就是普通的行为克隆), 之后越来越多地
让学生自己走 —— 否则一开始学生太菜, 跑出来的状态毫无价值。
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
import hex_pathfinding_demo as game
import greedy_policy
from rl_route_env import RouteEnv, A_ADVANCE_DAY, A_SWITCH_TEAM
from train_bc import PolicyNet

TEACHER_BUDGET = 120     # 每队每天的动作上限, 与 greedy_policy.rollout 一致


def _policy_action(net, env, deterministic=True):
    obs = env._observation()
    with torch.no_grad():
        logits = net(torch.from_numpy(obs['globals']).unsqueeze(0),
                     torch.from_numpy(obs['dirs']).unsqueeze(0))[0]
    logits[~torch.from_numpy(env.action_masks())] = -1e9
    if deterministic:
        return int(torch.argmax(logits))
    return int(torch.distributions.Categorical(logits=logits).sample())


def rollout_and_label(net, beta, food_weight, max_steps, deterministic=False,
                      actor=None, init_save=None):
    """学生开车、教师打标签。beta 是"这一步直接用教师动作"的概率。

    actor 给一个 (env) -> 动作 的可调用对象时就用它开车(比如 SB3 的模型);
    否则用 net(我们自己的 PolicyNet)。两个都不给就全程教师开车。
    """
    env = RouteEnv(food_weight=food_weight, init_save=init_save)
    env.reset()
    gs, ds, labels = [], [], []
    budget = {}
    last_day = env.engine.current_day
    rng = np.random.default_rng()
    for _ in range(max_steps):
        if env.engine.current_day != last_day:
            budget.clear()
            last_day = env.engine.current_day
        key = env._active_index
        budget.setdefault(key, TEACHER_BUDGET)

        # 教师在这个状态下的动作 —— 无论谁开车, 标签都用它
        expert = greedy_policy.greedy_action(env, food_weight, budget[key], budget)
        obs = env._observation()
        gs.append(obs['globals']); ds.append(obs['dirs']); labels.append(expert)

        if (net is None and actor is None) or rng.random() < beta:
            action = expert
        elif actor is not None:
            action = actor(env)
        else:
            action = _policy_action(net, env, deterministic)
        if action < A_ADVANCE_DAY:
            budget[key] -= 1
        env.apply(action)
        if env.engine.current_day >= game.TOTAL_DAYS and action == A_ADVANCE_DAY:
            break
    return env.engine, gs, ds, labels


def evaluate(net, food_weight, max_steps=20000, init_save=None):
    env = RouteEnv(food_weight=food_weight, init_save=init_save)
    env.reset()
    for _ in range(max_steps):
        action = _policy_action(net, env, deterministic=True)
        env.apply(action)
        if env.engine.current_day >= game.TOTAL_DAYS:
            break
    return env.engine


def train(net, gs, ds, labels, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    tg, td, ta = (torch.from_numpy(gs), torch.from_numpy(ds), torch.from_numpy(labels))
    last = (0.0, 0.0)
    for _ in range(epochs):
        perm = torch.randperm(len(ta))
        total, correct = 0.0, 0
        for lo in range(0, len(ta), 256):
            idx = perm[lo:lo + 256]
            logits = net(tg[idx], td[idx])
            loss = loss_fn(logits, ta[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == ta[idx]).sum())
        last = (total / len(ta), correct / len(ta) * 100)
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--out', default=_os.path.join(_rlpath.RL_DIR, 'dagger_policy.pt'))
    args = parser.parse_args()

    all_g, all_d, all_a = [], [], []
    net = None
    best = None
    print('教师(贪心策略): 65,114 分 / 占领 1,120   |   纯 PPO 最高 35,337', flush=True)
    for rnd in range(args.rounds):
        beta = 1.0 if rnd == 0 else 0.5 ** rnd
        started = time.perf_counter()
        engine, gs, ds, labels = rollout_and_label(
            net, beta, args.food_weight, args.max_steps)
        all_g.extend(gs); all_d.extend(ds); all_a.extend(labels)
        G = np.asarray(all_g, dtype=np.float32)
        D = np.asarray(all_d, dtype=np.float32)
        A = np.asarray(all_a, dtype=np.int64)

        if net is None:
            net = PolicyNet(n_globals=G.shape[1])
        loss, acc = train(net, G, D, A, args.epochs)
        evaluated = evaluate(net, args.food_weight)
        score = evaluated.total_reward
        if best is None or score > best[0]:
            best = (score, rnd)
            torch.save(net.state_dict(), args.out)
        print('第 %d 轮  beta %.2f | 本轮采样 %5d 步(跑到 day %2d, %s 分) | '
              '累计样本 %6d | 模仿 %.1f%% | 学生实跑 day %2d %8s 分 占领 %4d | %.0fs'
              % (rnd, beta, len(labels), engine.current_day,
                 format(engine.total_reward, ','), len(A), acc,
                 evaluated.current_day, format(score, ','),
                 len(evaluated.all_visited_hexes), time.perf_counter() - started),
              flush=True)

    print()
    print('最好的一轮: 第 %d 轮, %s 分  -> 已存 %s'
          % (best[1], format(best[0], ','), args.out))


if __name__ == '__main__':
    main()
