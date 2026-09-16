"""Optional Gymnasium adapter for the Naruto Marching headless engine."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction, plan_move
import hex_pathfinding_demo as game

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    gym = None
    spaces = None


class NarutoMarchingGymEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible wrapper, loaded only when Gymnasium is installed.

    Action encoding:
      [action_type, team_index, target_row, target_col, teleport]

    action_type: 0=move, 1=replay is intentionally excluded, 2=fly,
    3=settle exploration, 4=advance day, 5=choose portal.
    """

    def __init__(self, portal_teleport: bool = False, reward_config: Optional[Dict[str, float]] = None):
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium to use NarutoMarchingGymEnv")
        self.engine = NarutoMarchingEngine(portal_teleport=portal_teleport)
        self.reward_config = {
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
        if reward_config:
            self.reward_config.update(reward_config)
        self.action_space = spaces.MultiDiscrete([6, 3, self.engine_rows, self.engine_cols, 2])
        self.observation_space = spaces.Dict({
            "day": spaces.Box(1, 100, shape=(1,), dtype=np.int32),
            "food": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "total_food": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "total_reward": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "fly_skill_limit": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
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
            "occupied_map": occupied,
        }

    def _transition_reward(self, before: Dict[str, Any], after: Dict[str, Any], info: Dict[str, Any], done: bool, action_type: ActionType) -> float:
        """Shape training feedback while keeping actual score as the main term."""
        score_delta = float(info.get("score_delta", after["total_reward"] - before["total_reward"]))
        food_spent = max(0.0, float(before["food"] - after["food"]))
        new_hexes = len(after["visited_hexes"] - before["visited_hexes"])
        day_progress = max(0, int(after["day"]) - int(before["day"]))
        before_tokens = {
            game.RAW_MAP[ir][ic]
            for ir, ic in before["visited_hexes"]
            if 0 <= ir < game.ROWS and 0 <= ic < game.COLS
        }
        after_tokens = {
            game.RAW_MAP[ir][ic]
            for ir, ic in after["visited_hexes"]
            if 0 <= ir < game.ROWS and 0 <= ic < game.COLS
        }
        before_g_count = sum(token in ("G", "g") for token in before_tokens)
        after_g_count = sum(token in ("G", "g") for token in after_tokens)
        before_tent_count = sum(token == "T" for token in before_tokens)
        after_tent_count = sum(token == "T" for token in after_tokens)
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
            food_remaining_penalty = -self.reward_config["daily_food_remaining"] * max(0, float(before["food"]))
            remaining_steps = sum(
                team_state["steps"]
                for team_state in before["teams"].values()
                if team_state is not None
            )
            step_remaining_penalty = -self.reward_config["daily_step_remaining"] * remaining_steps
        reward = (
            self.reward_config["score"] * score_delta
            - self.reward_config["food_cost"] * food_spent
            + self.reward_config["new_hex"] * new_hexes
            + self.reward_config["day_progress"] * day_progress
            + g_complete_bonus
            + tent_complete_bonus
            + food_remaining_penalty
            + step_remaining_penalty
        )
        if done:
            reward += self.reward_config["completion"]
        info["reward_breakdown"] = {
            "score": score_delta,
            "food_cost_penalty": -self.reward_config["food_cost"] * food_spent,
            "new_hex_bonus": self.reward_config["new_hex"] * new_hexes,
            "day_progress_bonus": self.reward_config["day_progress"] * day_progress,
            "completion_bonus": self.reward_config["completion"] if done else 0.0,
            "g_complete_bonus": g_complete_bonus,
            "tent_complete_bonus": tent_complete_bonus,
            "daily_food_remaining_penalty": food_remaining_penalty,
            "daily_step_remaining_penalty": step_remaining_penalty,
            "total": reward,
        }
        return float(reward)

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
        action_type_mask = np.ones(6, dtype=bool)
        action_type_mask[1] = False  # REPLAY_PATH is archive-only.
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
        team_mask = np.array([team is not None for team in teams], dtype=bool)
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

