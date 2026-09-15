# Naruto Marching S25 强化学习设计

## 1. 目标

训练一个 PPO agent，在完整的 97 天游戏流程中自动规划路线，最大化最终积分，同时合理管理：

- 粮草
- 每队步数
- 地块占领顺序
- B/X/Z buff
- G/g 全局减免
- Tent 地块收益
- 飞雷神次数
- 多队伍协同

最终评估指标是 Day 97 结束时的真实游戏积分，而不是单纯的 shaping reward。

## 2. 系统结构

项目分为三层：

### GUI 游戏层

[hex_pathfinding_demo.py](hex_pathfinding_demo.py)

负责：

- Matplotlib/Tkinter 界面
- 鼠标点击
- 地图绘制
- 存档和加载
- 人工操作

GUI 手动操作继续使用原有路径规划和规则，不依赖 RL 训练模块。

### Headless 游戏引擎层

[rl_engine.py](rl_engine.py)

`NarutoMarchingEngine` 继承核心游戏逻辑，但覆盖 GUI 回调：

- 不创建窗口
- 不弹出确认框
- 不保存 autosave
- 不更新按钮和地图视图
- 复用游戏中的状态转移、成本、奖励、buff 和日期逻辑

接口：

```python
observation = engine.reset()
observation, reward, terminated, info = engine.step(action)
```

### Gymnasium 适配层

[rl_gym_env.py](rl_gym_env.py)

提供 Stable-Baselines3 使用的标准环境接口：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

依赖：

- `gymnasium`
- `stable-baselines3`
- `numpy==1.26.4`，用于兼容当前 Matplotlib 3.4.1

## 3. Episode 定义

一次 episode 从 Day 1 开始，目标是推进到 Day 97。

```text
Day 1 reset
    |
    | agent 执行动作
    v
Day 2 ... Day 97
    |
    v
terminated=True
```

当前环境设置最大 action 数，防止 agent 不推进日期而无限循环。训练脚本使用每个 episode 最多 1000 个 action。

注意：

- `day` 是游戏日期。
- `timestep` 是一次 `env.step(action)`。
- 一个 97 天 episode 通常包含远多于 97 个 timesteps。
- `total_timesteps` 是所有 episode 的 action 总数。

## 4. 状态空间

当前 Gym observation 是字典：

```python
{
    "day": np.array([day], dtype=np.int32),
    "food": np.array([current_food], dtype=np.int32),
    "total_food": np.array([total_food], dtype=np.int32),
    "total_reward": np.array([total_reward], dtype=np.int32),
    "fly_skill_limit": np.array([fly_skill_limit], dtype=np.int32),
    "occupied_map": np.ndarray[(ROWS, COLS)],
}
```

### 核心状态

- 当前日期
- 当前共享粮草
- 历史总粮草消耗
- 当前累计积分
- 剩余飞雷神次数
- 全局已占领地图

### 引擎内部还维护

每支队伍包含：

- 当前坐标
- 创建日期
- 已占领地块
- 未结算试探地块
- B/X/Z buff 剩余次数
- 每日步数
- segment 历史
- 每格粮草、积分和步数账本

## 5. Action 空间

当前 action 编码：

```text
[action_type, team_index, target_row, target_col, teleport]
```

### action_type

| 值 | 动作 | 说明 |
|---:|---|---|
| 0 | `MOVE_TO_HEX` | 普通移动到相邻目标地块 |
| 1 | `REPLAY_PATH` | 存档验证专用，不用于训练 |
| 2 | `FLY_TO_HEX` | 飞雷神到合法目标 |
| 3 | `SETTLE_EXPLORATION` | 结算当前试探地块 |
| 4 | `ADVANCE_DAY` | 推进到下一天 |
| 5 | `CHOOSE_PORTAL` | 选择传送门行为 |

### 普通移动

RL agent 每次普通移动只能选择当前队伍所在六边形的相邻地块。

GUI 手动操作仍然可以点击远处目标并由 A* 规划多格路线；这个限制只作用于 RL Gym 环境。

普通移动中的已占领地块回访是跳步，跳步属于移动结果的一部分。

### 试探步

当队伍当天步数为 0 时，仍允许一次合法试探步：

- 目标必须相邻
- 目标必须未占领
- 地形 `step > 0`
- 当前队伍没有 X buff 免费步数

试探地块会暂时记录为未结算地块，之后离开或执行结算动作时支付成本并获得奖励。

### 飞雷神

飞雷神目标必须是未占领地块，并且目标必须与“从飞雷神起点出发、沿任意队伍已占领地块连通形成的区域”相邻。

规则：

- 起点是当前队伍实际所在位置。
- 连通路径可以经过多个已占领地块。
- 路径上的地块可以属于任意队伍。
- 不限制为两跳。
- 未结算试探地块不作为已占领连接区域。
- 飞雷神不消耗普通步数。
- BigBoss 等特殊规则仍由游戏引擎处理。

### 日期推进

`ADVANCE_DAY` 会：

- 增加当前日期
- 更新粮草
- 更新每队步数
- 触发日期相关地形收益
- 在 Day 97 结束 episode

## 6. 合法动作与 action mask

环境提供：

```python
env.action_masks()
```

当前 coarse mask 会屏蔽：

- archive-only 的 `REPLAY_PATH`
- 没有飞雷神次数时的飞雷神
- 没有试探地块时的结算
- 未创建的 Team 2 / Team 3
- 当前 active team 没有步数且没有可试探目标时的普通移动

目标坐标仍是地图大小的 MultiDiscrete 空间。完整的逐格 mask 尚未接入，当前通过 step-time 校验拒绝不合法目标。

非法动作返回：

```python
reward = -invalid_action
info["invalid_action"] = True
```

当前 `invalid_action` 默认值为 `0.25`。

## 7. 候选动作权重

环境提供候选动作分析接口：

```python
env.get_action_candidates()
env.sample_weighted_candidate(temperature=1.0)
```

每个候选包含：

```python
{
    "target": (ir, ic),
    "score_gain": 预计积分,
    "food_cost": 预计粮草,
    "new_hexes": 新占领数量,
    "jump_count": 跳步数量,
    "weight": 推荐权重,
    "probability": 归一化概率,
}
```

当前启发式权重：

```python
weight = score_gain + 8.0 * new_hexes - 0.05 * food_cost
```

这个权重是 action proposal/prior，不会强制修改 PPO 的策略概率。

## 8. Reward Function

Gym reward 是 shaping reward，不改变真实游戏积分。

当前公式：

```python
reward = (
    1.0 * score_delta
    - 0.001 * food_spent
    + 8.0 * new_hexes
    + 0.05 * day_progress
    + g_complete_bonus
    + tent_complete_bonus
    + completion_bonus
    + daily_food_remaining_penalty
    + daily_step_remaining_penalty
)
```

### 各项参数

```python
{
    "score": 1.0,
    "food_cost": 0.001,
    "new_hex": 8.0,
    "day_progress": 0.05,
    "daily_food_remaining": 0.0002,
    "daily_step_remaining": 0.05,
    "completion": 100.0,
    "g_complete": 500.0,
    "tent_complete": 300.0,
    "invalid_action": 0.25,
}
```

### 实际积分

```text
score_delta = 本次动作后的总积分 - 动作前总积分
```

这是 reward 的主项。

### 新地块奖励

每个 action 新增地块数量：

```text
+8 × new_hexes
```

用于给探索行为更密集的学习信号。

### G/g 里程碑

首次完成全部 8 个 G/g 地块：

```text
+500
```

只触发一次。

### Tent 里程碑

首次占领满 15 个 Tent 地块：

```text
+300
```

只触发一次。

### Day 97 完成奖励

完成 Day 97：

```text
+100
```

当前已从原来的 1000 降低，避免 agent 只推进日期。

### 每日剩余资源惩罚

只在 `ADVANCE_DAY` 时计算：

```text
food_penalty = -0.0002 × 当天剩余粮草
step_penalty = -0.05 × 所有队伍剩余步数
```

这样 agent 在推进日期前使用更多资源，惩罚会降低。

### 非法动作惩罚

```text
-0.25
```

避免随机目标造成过强负反馈，同时保留对非法行为的区分。

## 9. 训练脚本

训练入口：

[train_ppo_100.py](train_ppo_100.py)

核心配置：

```python
model = PPO(
    "MultiInputPolicy",
    env,
    n_steps=256,
    batch_size=64,
    seed=42,
)
```

训练结果记录到：

- `ppo_100_episode_results_occupied.json`
- `ppo_100_episode_results_occupied.csv`

每个 episode 记录：

- episode 编号
- 实际游戏积分
- 训练 reward
- 最终日期
- 最终粮草
- 最终占领地块数量
- action steps
- 累计耗时

## 10. 历史存档验证

[archive_replay.py](archive_replay.py)

用于验证 GUI 历史行为与 headless engine 的一致性。

重要原则：

- 新 action 使用当前规则重新计算成本。
- 历史存档 replay 使用存档中的 `_seg_hex_costs`、`_seg_foods`、`_seg_awards` 和 `_seg_steps`。
- 不能用当前 G/g 状态重新计算旧历史，否则会把旧规则误判为当前规则。

当前代表性存档：

`69949余46.json`

验证结果：

```text
1265 / 1265 segments replayed
最终到达 Day 97 历史路径
最终资源、队伍位置和总积分与存档一致
```

存档中的最终总积分：

```text
70138
```

注意存档的 `current_day=93` 是查看日，历史路径可延伸到 Day 97。

## 11. 测试体系

主要测试文件：

- `test_rl_engine.py`
- `test_rl_playground.py`
- `test_rl_trace.py`
- `test_archive_integrity.py`
- `test_archive_replay.py`
- `test_rl_gym_env.py`
- `test_bonus_fix.py`

验证内容：

- 普通移动
- 相邻移动限制
- 零步试探步
- 飞雷神范围
- 跨队伍占领连接
- G/g 和 Tent 里程碑奖励
- 非法动作处理
- action mask
- Gymnasium contract
- 多队伍历史回放
- 存档账本一致性
- 真实积分和粮草

Stable-Baselines3 检查：

```python
check_env(env)
```

当前通过。

## 12. 已知限制

### PPO 仍不是最终路线规划器

目前 observation 和 action 设计已经可训练，但 PPO 在早期实验中容易：

- 推进日期过快
- 重复尝试不合法目标
- 卡在 Day 1
- 只占领极少地块

因此需要更长训练、更好的目标 action mask 或候选动作索引空间。

### 当前 action space 仍为地图坐标

这是一个稀疏动作空间：

```text
所有地图行列坐标都可能被采样
```

虽然环境会拒绝非法目标，但 PPO 学习效率较低。下一步可改为：

```text
candidate_action_index
```

即环境先生成合法候选列表，PPO 只选择候选索引。

### 训练 reward 与真实积分不同

训练 reward 包含：

- 资源效率
- 探索奖励
- 里程碑奖励
- 完成奖励
- 非法动作惩罚

最终评估必须使用真实游戏积分，而不是单纯比较 training reward。

## 13. 推荐后续路线

1. 将 action space 改成合法候选动作索引。
2. 使用 `sb3-contrib` 的 `MaskablePPO`。
3. 将普通移动、飞雷神、试探步和推进日期拆为明确候选动作。
4. 训练时保存最佳 model 和每个 episode 的完整 action trace。
5. 每个 episode 记录真实积分、地块坐标和 97 天完成情况。
6. 与 `69949余46.json` 的 70138 分基线比较。
7. 训练更长时间后再调整 reward 权重。
