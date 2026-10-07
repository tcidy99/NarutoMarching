"""Optional Gymnasium adapter for the Naruto Marching headless engine."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401  统一把 rl/ 与仓库根加进 sys.path, 并校正工作目录
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction, plan_move
import hex_pathfinding_demo as game

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    gym = None
    spaces = None


# ══ 训练奖励(RP)是一套独立的数, 和游戏积分不是一回事 ═══════════════════════
#
# 游戏积分(observation["total_reward"])一局能累到七万上下; 训练奖励下面每一项都
# 刻意压在 O(1)~O(10)。两者唯一的接口就是 SCORE_PER_RP: 游戏积分必须先除以它才
# 能进 reward, 绝不能原样相加 —— 否则 score 这一项会比其余所有 shaping 项大三四
# 个数量级, 等于那些项根本不存在。
#
# 读数的时候也别混: reward / reward_breakdown 里的都是 RP;
# 游戏积分在 info["game_score_delta"] 和 info["game_total_reward"] 里, 名字带
# game_ 前缀。评估模型好坏要看后者, 训练曲线看前者。
SCORE_PER_RP = 100.0  # 100 游戏积分 = 1 RP。一整季七万分 ≈ 700 RP, 与其它项同量级。

DAILY_STEP_GRANT = 6  # 每队每天回复的步数(上限 18), 用来把"剩余步数"归一化。

# 下面的权重是拿人工存档 全局70558_6.json(70,558 分, 打满 97 天)量出来的节奏
# 校准的 —— 那是一条已知的好路线, 奖励函数至少要让它得高分:
#
#     全季新占格 1,155        平均每格 54.6 游戏积分 = 0.546 RP
#     全季粮草  160,394       平均每格 138.9 粮
#     全季步数  1,115         93 个行动日, 每天中位 11 步(三队共 18 步/天, 约六成利用率)
#     每天中位: 11 步 / 1,624 粮 / 11 新格 / 580 分
#
# 按这张表把那条人工路线整季折算下来大约是(见 tools 里的校准脚本):
#     new_hex +1155, score +631, 建队 +60, 推进天数 +19, 里程碑 +130, 打满 +50
#     粮草 -321  ->  净约 +1,720 RP
# 也就是说"多占一格"的净收益约 +1.27 RP, 稳定为正; 而每天剩六步不用只扣 0.12 RP,
# 不会逼着 agent 做人类也不会做的事。
TRAINING_REWARD = {
    # ── 正向 ──
    "score": 1.0,          # 每 SCORE_PER_RP 游戏积分给 1 RP(人工路线平均 0.55 RP/格)
    "new_hex": 1.0,        # 每占一个新格 1 RP —— 主要的密集信号, 也是整张表的基准单位
    "create_team": 30.0,   # 开出 2/3 队, 一次性 30 RP(还要乘早晚与就位系数)
    "day_progress": 0.2,   # 推进一天 0.2 RP, 全季约 19 RP
    "g_complete": 50.0,    # 八个 G/g 集齐(此后所有成本永久 8 折, 值这个价)
    "tent_complete": 30.0, # 帐篷全部拿完
    "completion": 50.0,    # 打满一季
    # ── 负向 ──
    # 人工路线平均每格花 138.9 粮, 乘 0.002 = 0.28 RP, 约占每格毛收益的两成 ——
    # 足以压住"为跳步而跳步"(重复经过只花 10 粮 = 0.02 RP), 又不至于让占格变亏本。
    "food_cost": 0.002,
    "invalid_action": 0.25,
    # 换日时资源没用完的惩罚。这两项曾经是这套环境最大的坑: 原来是
    # 0.0002/粮 + 0.05/步, day 1 满粮时换日净收益约 -1.61 RP, 而乱按一个非法动作
    # 只罚 -0.25 —— 于是最优策略变成"赖在第一天一直乱按", 100 回合里有一半
    # final_day=1、分数恰好等于"一步不走"的保底 7500。
    # 现在: 粮草按比例算(不按绝对数), 步数权重压到 0.02 —— 人工打法每天也要剩
    # 六步左右(利用率只有六成), 罚得太重等于逼 agent 做人类都不做的事。
    "daily_food_remaining": 0.10,  # 乘以"当天剩余粮草占当日可用量的比例"
    "daily_step_remaining": 0.02,  # 乘以三队剩余步数之和(剩 6 步 = -0.12 RP)
}


class NarutoMarchingGymEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible wrapper, loaded only when Gymnasium is installed.

    Action encoding:
      [action_type, team_index, target_row, target_col, teleport]

    action_type: 0=move, 1=replay is intentionally excluded, 2=fly,
    3=settle exploration, 4=advance day, 5=choose portal, 6=create team
    (建 2/3 队, 用 team_index 选哪一支, target 忽略)。
    """

    def __init__(self, portal_teleport: bool = False, reward_config: Optional[Dict[str, float]] = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium to use NarutoMarchingGymEnv")
        self.engine = NarutoMarchingEngine(portal_teleport=portal_teleport)
        self.reward_config = dict(TRAINING_REWARD)
        if reward_config:
            self.reward_config.update(reward_config)
        self._last_team_bonus_factors = {}
        self.action_space = spaces.MultiDiscrete([7, 3, self.engine_rows, self.engine_cols, 2])
        self.observation_space = spaces.Dict({
            "day": spaces.Box(1, 100, shape=(1,), dtype=np.int32),
            "food": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "total_food": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "total_reward": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "fly_skill_limit": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            "teams_alive": spaces.MultiBinary(3),
            "occupied_map": spaces.MultiBinary((self.engine_rows, self.engine_cols)),
        })

    @property
    def engine_rows(self) -> int:
        import hex_pathfinding_demo as game
        return game.ROWS

    @property
    def engine_cols(self) -> int:
        import hex_pathfinding_demo as game
        return game.COLS

    def _observation(self) -> Dict[str, np.ndarray]:
        state = self.engine.get_observation()
        occupied = np.zeros((self.engine_rows, self.engine_cols), dtype=np.int8)
        for ir, ic in state["visited_hexes"]:
            occupied[ir, ic] = 1
        return {
            "day": np.array([state["day"]], dtype=np.int32),
            "food": np.array([max(0, state["food"])], dtype=np.int32),
            "total_food": np.array([state["total_food"]], dtype=np.int32),
            "total_reward": np.array([state["total_reward"]], dtype=np.int32),
            "fly_skill_limit": np.array([state["fly_skill_limit"]], dtype=np.int32),
            # 哪几支队伍已经存在。建队是一个动作, 那"建没建过"就必须可观测,
            # 否则 agent 只能靠撞非法动作来试探。
            "teams_alive": np.array(
                [1 if state["teams"].get(n) is not None else 0 for n in (1, 2, 3)],
                dtype=np.int8,
            ),
            "occupied_map": occupied,
        }

    def _transition_reward(self, before: Dict[str, Any], after: Dict[str, Any], info: Dict[str, Any], done: bool, action_type: ActionType) -> float:
        """Shape training feedback while keeping actual score as the main term."""
        score_delta = float(info.get("score_delta", after["total_reward"] - before["total_reward"]))
        food_spent = max(0.0, float(before["food"] - after["food"]))
        new_hexes = len(after["visited_hexes"] - before["visited_hexes"])
        day_progress = max(0, int(after["day"]) - int(before["day"]))
        # 这里原来是先把已占格子映射成 token 的**集合**再数 —— 集合里 "G"/"g"
        # 最多各出现一次, 于是 g_count 永远 ≤ 2、tent_count 永远 ≤ 1, 那两个
        # "凑齐 8 个卦 / 拿完 15 个帐篷" 的里程碑奖励**一次都不可能触发**。
        # 现在按格子数,而不是按 token 种类数。
        before_g_count = self._count_tokens(before["visited_hexes"], ("G", "g"))
        after_g_count = self._count_tokens(after["visited_hexes"], ("G", "g"))
        before_tent_count = self._count_tokens(before["visited_hexes"], ("T",))
        after_tent_count = self._count_tokens(after["visited_hexes"], ("T",))
        g_complete_bonus = (
            self.reward_config["g_complete"]
            if before_g_count < 8 <= after_g_count
            else 0.0
        )
        tent_complete_bonus = (
            self.reward_config["tent_complete"]
            if before_tent_count < 15 <= after_tent_count
            else 0.0
        )
        food_remaining_penalty = 0.0
        step_remaining_penalty = 0.0
        if action_type == ActionType.ADVANCE_DAY:
            # 按"占当天可用粮草的比例"惩罚, 不按绝对粮草数 —— 后者在开局满粮时
            # 会算出一个远大于 invalid_action 的数, 把 agent 钉死在第一天。
            budget = max(1.0, float(before["food"]) + float(food_spent))
            idle_ratio = min(1.0, max(0.0, float(before["food"])) / budget)
            food_remaining_penalty = -self.reward_config["daily_food_remaining"] * idle_ratio
            remaining_steps = sum(
                team_state["steps"]
                for team_state in before["teams"].values()
                if team_state is not None
            )
            step_remaining_penalty = -self.reward_config["daily_step_remaining"] * remaining_steps
        team_bonus = self._team_creation_bonus(before, after, action_type)
        reward = (
            # 游戏积分先折算成 RP 再进来, 不直接相加 —— 见文件顶部 SCORE_PER_RP。
            self.reward_config["score"] * (score_delta / SCORE_PER_RP)
            - self.reward_config["food_cost"] * food_spent
            + self.reward_config["new_hex"] * new_hexes
            + self.reward_config["day_progress"] * day_progress
            + team_bonus
            + g_complete_bonus
            + tent_complete_bonus
            + food_remaining_penalty
            + step_remaining_penalty
        )
        if done:
            reward += self.reward_config["completion"]
        # 下面全部是 RP。游戏积分只在 game_score_delta 里, 名字带 game_ 前缀。
        info["reward_breakdown"] = {
            "score_rp": self.reward_config["score"] * (score_delta / SCORE_PER_RP),
            "food_cost_penalty": -self.reward_config["food_cost"] * food_spent,
            "new_hex_bonus": self.reward_config["new_hex"] * new_hexes,
            "day_progress_bonus": self.reward_config["day_progress"] * day_progress,
            "create_team_bonus": team_bonus,
            "create_team_factors": (self._last_team_bonus_factors
                                    if team_bonus else None),
            "completion_bonus": self.reward_config["completion"] if done else 0.0,
            "g_complete_bonus": g_complete_bonus,
            "tent_complete_bonus": tent_complete_bonus,
            "daily_food_remaining_penalty": food_remaining_penalty,
            "daily_step_remaining_penalty": step_remaining_penalty,
            "total": reward,
        }
        info["game_score_delta"] = score_delta
        info["game_total_reward"] = after["total_reward"]
        return float(reward)

    @staticmethod
    def _count_tokens(hexes, tokens) -> int:
        """已占格子里属于这些地形的**格子数**(不是地形种类数)。"""
        return sum(
            1
            for ir, ic in hexes
            if 0 <= ir < game.ROWS and 0 <= ic < game.COLS
            and game.RAW_MAP[ir][ic] in tokens
        )

    def _team_creation_bonus(self, before, after, action_type) -> float:
        """开出 2/3 队的一次性奖励 = 基础分 x 早晚系数 x "上一支队伍走完了没"系数。

        为什么单独给奖励: 步数是每队独立结算的(每天 +6, 上限 18), 多一支队伍就是
        多一份行动力。但这份收益要几十步之后才通过 new_hex 体现出来, 对 PPO 来说
        太远, 不给即时奖励它学不会先建队。

        为什么还要看"走完了没": 新队伍开在**当前激活队伍所在的那一格**上。如果
        激活队伍还没动就建队, 两支队伍叠在同一个点, 等于白白浪费了分头占地的机会。
        人工存档 全局70558_6.json 的 day 1 就是标准打法:

            order 0-11   team1 先把当天 6 步走完, 再用 0 步的动作挪到 (14,49)
            order 12     在 (14,49) 开出 team2      <- 此时 team1 步数已归零
            order 12-17  team2 把它自己的 6 步走完, 停在 (19,51)
            order 28     在 (19,51) 开出 team3      <- 此时 team2 步数已归零

        两支队伍都是"上一支当天走完了"才开的, 而且都开在上一支的终点。所以这里按
        创建者当天剩余步数打折: 步数已用光给满额, 还剩满格步数只给两成。
        """
        if action_type != ActionType.CREATE_TEAM:
            return 0.0
        created = [n for n in (2, 3)
                   if before["teams"].get(n) is None and after["teams"].get(n) is not None]
        if not created:
            return 0.0  # 建队失败(已存在/选了 team1 等), 不给奖励

        # 早晚: day 1 满额, 之后在赛季前 15% 里线性衰减到两成, 再往后维持两成
        # (晚建总比不建强, 所以留个地板)。
        ramp = max(1.0, 0.15 * float(game.TOTAL_DAYS))
        earliness = max(0.2, 1.0 - (int(before["day"]) - 1) / ramp)

        # 创建者是动作发生前的激活队伍 —— 新队伍就开在它脚下。
        creator = before["teams"].get(before["active_team"])
        steps_left = float(creator["steps"]) if creator else 0.0
        readiness = max(0.2, 1.0 - steps_left / float(DAILY_STEP_GRANT))

        bonus = (self.reward_config["create_team"] * earliness * readiness
                 * len(created))
        self._last_team_bonus_factors = {
            "earliness": earliness,
            "readiness": readiness,
            "creator_steps_left": steps_left,
        }
        return bonus

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.engine.reset()
        return self._observation(), {}

    def action_masks(self):
        """Return coarse masks for action type and team selection.

        This is compatible with masking wrappers such as sb3-contrib's
        MaskablePPO. Target-row/column validity remains state-dependent and is
        reported through ``valid_action_types`` rather than masking the full
        map-sized MultiDiscrete target space.
        """
        teams = [self.engine.team1, self.engine.team2, self.engine.team3]
        action_type_mask = np.ones(7, dtype=bool)
        action_type_mask[1] = False  # REPLAY_PATH is archive-only.
        # 建队: 只要还有队伍没开, 且有一支激活队伍可以当作出生位置, 就允许。
        can_create = self.engine.active_team is not None and any(t is None for t in teams[1:])
        action_type_mask[6] = can_create
        active_team_index = 0 if self.engine.active_team is self.engine.team1 else (
            1 if self.engine.active_team is self.engine.team2 else 2
        )
        action_type_mask[0] = (
            self._selected_team_can_move(active_team_index)
            or (
                self.engine.active_team is not None
                and any(
                    self._can_probe_from_selected_team(active_team_index, neighbor)
                    for neighbor in game._neighbors(*self.engine.active_team.full_path[-1])
                )
            )
        )
        if self.engine.fly_skill_limit <= 0:
            action_type_mask[2] = False
        if self.engine.active_team is None or not self.engine.active_team.free_exploration_hexes:
            action_type_mask[3] = False
        # team 维度是所有动作共用的, 而建队要选的恰恰是"还不存在"的那一支, 所以
        # 这里不能只放行已存在的队伍, 否则 6 号动作永远选不中目标。
        team_mask = np.array(
            [team is not None or (can_create and index > 0)
             for index, team in enumerate(teams)],
            dtype=bool,
        )
        return {"action_type": action_type_mask, "team": team_mask}

    def _selected_team_can_move(self, team_index: int) -> bool:
        team = (self.engine.team1, self.engine.team2, self.engine.team3)[team_index]
        if team is None:
            return False
        return self.engine._get_team_steps_for_day(team, self.engine.current_day) > 0

    def _can_probe_from_selected_team(self, team_index: int, target) -> bool:
        team = (self.engine.team1, self.engine.team2, self.engine.team3)[team_index]
        if team is None or team.x_bonus_remaining > 0:
            return False
        if self._selected_team_can_move(team_index):
            return False
        current = tuple(team.full_path[-1])
        if target is None:
            return False
        target = tuple(target)
        if target not in game._neighbors(*current) or target in self.engine.all_visited_hexes:
            return False
        terrain = game._terrain(*target)
        return terrain.get("name") != "empty" and terrain.get("step", 1) > 0

    def get_action_candidates(self, team_index: Optional[int] = None):
        """Return legal target candidates with heuristic sampling weights.

        This is a proposal/prior API; PPO may use it for candidate-index action
        spaces without forcing the policy distribution in the environment.
        """
        if team_index is None:
            team_index = 0 if self.engine.active_team is self.engine.team1 else (
                1 if self.engine.active_team is self.engine.team2 else 2
            )
        teams = (self.engine.team1, self.engine.team2, self.engine.team3)
        team = teams[team_index]
        if team is None:
            return []

        occupied = set(self.engine.all_visited_hexes)
        candidates = []
        for ir in range(self.engine_rows):
            for ic in range(self.engine_cols):
                target = (ir, ic)
                if target in occupied or not game._passable(ir, ic):
                    continue
                if target not in game._neighbors(*team.full_path[-1]):
                    continue
                try:
                    outcome = plan_move(team.full_path[-1], target, occupied)
                except ValueError:
                    continue
                if not outcome.new_hexes:
                    continue
                score_gain = outcome.reward_total
                food_cost = outcome.food_total
                priority = max(0.01, score_gain + 8.0 * len(outcome.new_hexes) - 0.05 * food_cost)
                candidates.append({
                    "action": RouteAction(ActionType.MOVE_TO_HEX, target=target, team=team_index + 1),
                    "target": target,
                    "score_gain": score_gain,
                    "food_cost": food_cost,
                    "new_hexes": len(outcome.new_hexes),
                    "jump_count": outcome.jump_count,
                    "weight": priority,
                })
        total_weight = sum(item["weight"] for item in candidates) or 1.0
        for item in candidates:
            item["probability"] = item["weight"] / total_weight
        return sorted(candidates, key=lambda item: item["weight"], reverse=True)

    def sample_weighted_candidate(self, team_index: Optional[int] = None, temperature: float = 1.0):
        """Sample a proposal using score/food weights without overriding PPO."""
        candidates = self.get_action_candidates(team_index)
        if not candidates:
            return None
        temperature = max(float(temperature), 1e-6)
        logits = np.array([np.log(max(item["weight"], 1e-6)) / temperature for item in candidates])
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        selected = int(np.random.choice(len(candidates), p=probabilities))
        result = dict(candidates[selected])
        result["sampling_probability"] = float(probabilities[selected])
        return result

    def step(self, action):
        action_type, team_index, target_row, target_col, teleport = [int(value) for value in action]
        action_map = {
            0: ActionType.MOVE_TO_HEX,
            2: ActionType.FLY_TO_HEX,
            3: ActionType.SETTLE_EXPLORATION,
            4: ActionType.ADVANCE_DAY,
            5: ActionType.CHOOSE_PORTAL,
            6: ActionType.CREATE_TEAM,
        }
        if action_type == 1 or action_type not in action_map:
            penalty = -self.reward_config["invalid_action"]
            return self._observation(), penalty, False, False, {
                "invalid_action": True,
                "reason": "unsupported action type",
                "reward_breakdown": {"invalid_action_penalty": penalty, "total": penalty},
                "valid_action_types": np.flatnonzero(self.action_masks()["action_type"]).tolist(),
            }
        target = (target_row, target_col) if action_type in (0, 2) else None
        route_action = RouteAction(
            action_type=action_map[action_type],
            target=target,
            team=team_index + 1,
            teleport=bool(teleport),
        )
        selected_team = (self.engine.team1, self.engine.team2, self.engine.team3)[team_index]
        if action_type == 0 and (
            selected_team is None
            or target not in game._neighbors(*selected_team.full_path[-1])
        ):
            penalty = -self.reward_config["invalid_action"]
            return self._observation(), penalty, False, False, {
                "invalid_action": True,
                "reason": "normal movement must target one adjacent hex",
                "valid_action_types": np.flatnonzero(self.action_masks()["action_type"]).tolist(),
                "reward_breakdown": {"invalid_action_penalty": penalty, "total": penalty},
            }
        zero_step_probe = self._can_probe_from_selected_team(team_index, target)
        if (
            action_type == 0
            and not self._selected_team_can_move(team_index)
            and not zero_step_probe
            and (selected_team is None or selected_team.x_bonus_remaining <= 0)
        ):
            penalty = -self.reward_config["invalid_action"]
            return self._observation(), penalty, False, False, {
                "invalid_action": True,
                "reason": "selected team has no remaining steps; advance the day before moving",
                "valid_action_types": [2, 4, 5],
                "reward_breakdown": {"invalid_action_penalty": penalty, "total": penalty},
            }
        try:
            observation, reward, done, info = self.engine.step(route_action)
        except (ValueError, RuntimeError) as error:
            penalty = -self.reward_config["invalid_action"]
            return self._observation(), penalty, False, False, {
                "invalid_action": True,
                "reason": str(error),
                "reward_breakdown": {"invalid_action_penalty": penalty, "total": penalty},
                "valid_action_types": np.flatnonzero(self.action_masks()["action_type"]).tolist(),
            }
        before = info.get("observation_before", self.engine.get_observation())
        after = info.get("observation_after", observation)
        shaped_reward = self._transition_reward(before, after, info, bool(done), route_action.action_type)
        return self._observation(), shaped_reward, bool(done), False, info

    def close(self):
        return None

