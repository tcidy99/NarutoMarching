"""Headless RL engine facade that reuses the GUI game's transition logic.

The GUI remains unchanged.  This facade creates a PathfindingDemo instance
without opening figures, disables presentation-only callbacks, and exposes a
small reset/step interface for deterministic differential testing.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import hex_pathfinding_demo as game
from rl_playground import ActionType, RouteAction

Hex = Tuple[int, int]


class NarutoMarchingEngine(game.PathfindingDemo):
    """Run the existing game transition code without constructing the GUI."""

    def __init__(self, portal_teleport: bool = False):
        self.portal_teleport = bool(portal_teleport)
        self.reset()

    def _draw(self):
        """Suppress GUI redraws while preserving the original state updates."""

    def _auto_save_game(self):
        """Do not write autosaves during training episodes."""

    def _center_view_on_active_team(self):
        """Suppress view-centering callbacks in headless mode."""

    def _update_switch_button_color(self):
        """Suppress GUI button updates in headless mode."""

    def _update_fly_button_state(self):
        """Suppress GUI button updates in headless mode."""

    def _set_fly_button_border(self, _color, _linewidth):
        """Suppress fly-button styling callbacks in headless mode."""

    def _center_view_on_active_team_day(self, _day, move_if_empty=True):
        """Suppress day-navigation view callbacks in headless mode."""

    def _activate_fly_skill(self):
        """Toggle fly mode without touching a GUI button or timer."""
        if not self._fly_mode:
            if self.fly_skill_limit <= 0:
                self._status_msg = "No fly skills remaining!"
                return
            self._fly_mode = True
            self._status_msg = "Fly mode active."
        else:
            self._fly_mode = False
            self._status_msg = "Fly mode deactivated."

    def _confirm_free_exploration_step(self, _hex_pos):
        return True

    def _confirm_settle_current_exploration(self):
        """Settle a probe hex without opening the GUI confirmation dialog."""
        team = self.active_team
        if team is None or not team.full_path:
            return False
        current_pos = team.full_path[-1]
        if current_pos not in team.free_exploration_hexes or self._is_hex_settled(current_pos):
            return False

        steps_available = self._get_team_steps_for_day(team, self.current_day)
        if steps_available <= 0:
            return False

        terrain = game._terrain(*current_pos)
        challenge_food = game._apply_challenge_discounts(
            game._get_terrain_food(terrain, self.current_day),
            team,
            self._are_all_g_lands_visited(),
            is_tent=terrain.get("name") == "Tent",
        )
        reward = terrain.get("award", 0)
        if team.z_bonus_remaining > 0:
            reward = game._apply_z_bonus(reward, team)
        if self.current_food < challenge_food:
            self._status_msg = "Not enough food to settle exploration."
            return False

        team._seg_lengths.append(0)
        team._seg_foods.append(challenge_food)
        team._seg_awards.append(reward)
        team._seg_steps.append(1)
        team._seg_days.append(self.current_day)
        team._seg_new_hexes.append([current_pos])
        team._seg_exploration_hexes.append([])
        team._seg_jumps.append([])
        team._seg_path_nodes.append([])
        team._seg_end_positions.append(current_pos)
        team._seg_action_sequence.append([("new", current_pos)])
        self._append_segment_action_order(team)
        team._seg_hex_costs.append([(challenge_food, reward, 1)])
        team._seg_is_fly_skill.append(False)
        boss_delta = 1 if terrain.get("name") == "bigBoss" else 0
        team._seg_fly_skill_deltas.append(boss_delta)
        self.fly_skill_limit += boss_delta

        team.free_exploration_hexes.discard(current_pos)
        team.visited_hexes.add(current_pos)
        self.all_visited_hexes.add(current_pos)
        if current_pos in self.all_g_lands:
            self.visited_g_lands.add(current_pos)

        self.current_food -= challenge_food
        self.total_food += challenge_food
        self.total_reward += reward
        team.steps -= 1
        self._activate_land_buffs(current_pos)
        self._rebuild_day_records()
        return True

    def _check_portal_teleport(self):
        """Apply the fixed episode portal policy instead of opening a dialog."""
        current_pos = self.active_team.full_path[-1]
        if not (0 <= current_pos[0] < game.ROWS and 0 <= current_pos[1] < game.COLS):
            return False
        portal_type = game.RAW_MAP[current_pos[0]][current_pos[1]]
        if not portal_type.startswith("P"):
            return False

        paired = [
            (ir, ic)
            for ir in range(game.ROWS)
            for ic in range(game.COLS)
            if game.RAW_MAP[ir][ic] == portal_type and (ir, ic) != current_pos
        ]
        if not paired:
            return False

        self.active_team.visited_hexes.add(current_pos)
        self.all_visited_hexes.add(current_pos)
        if self.active_team.b_discount_remaining > 0:
            self.active_team.b_discount_remaining -= 1
        if self.active_team.z_bonus_remaining > 0:
            self.active_team.z_bonus_remaining -= 1

        if not self.portal_teleport:
            return False

        other_portal = paired[0]
        previous_pos = self.active_team.full_path[-2] if len(self.active_team.full_path) >= 2 else current_pos
        self.active_team.full_path[-1] = other_portal
        self.active_team._no_draw_edges.add((previous_pos, other_portal))
        self.active_team.visited_hexes.add(other_portal)
        self.all_visited_hexes.add(other_portal)
        if self.active_team._seg_end_positions:
            self.active_team._seg_end_positions[-1] = other_portal
        return True

    def _new_episode_state(self) -> None:
        origin = game._find_start_position()
        self.team1 = game.Team(origin, created_day=1)
        self.team2 = None
        self.team3 = None
        self.active_team = self.team1
        self.set_start_mode = None
        self._fly_mode = False
        self._fly_button_timer = None
        self._edit_seg_button_timer = None
        self._segment_edit_mode = False
        self._day_edit_context = None
        self._segment_edit_selected_days = set()
        self._segment_edit_targets = []
        self._segment_edit_focus_seg_idx = None
        self._last_landing_cost_breakdown = {}
        self._next_action_order = 0
        self._status_msg = ""
        self.current_day = 1
        self.current_food = 6800
        self.total_food = 0
        self.total_reward = 0
        self.fly_skill_limit = 1
        self.all_g_lands = {
            (ir, ic)
            for ir in range(game.ROWS)
            for ic in range(game.COLS)
            if game.RAW_MAP[ir][ic] in ("G", "g")
        }
        self.all_visited_hexes = {origin}
        self.visited_g_lands = {origin} & self.all_g_lands
        self._init_day_records()
        self._rebuild_day_records()

    def reset(self) -> Dict[str, Any]:
        """Reset the deterministic episode and return the initial observation."""
        self._new_episode_state()
        return self.get_observation()

    def _team_number(self, team) -> int:
        if team is self.team1:
            return 1
        if team is self.team2:
            return 2
        if team is self.team3:
            return 3
        raise ValueError("unknown team")

    def _select_team(self, team_number: int):
        team = {1: self.team1, 2: self.team2, 3: self.team3}.get(int(team_number))
        if team is None:
            raise ValueError(f"team {team_number} does not exist")
        self.active_team = team
        return team

    def get_observation(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of gameplay state."""
        teams = {}
        for number, team in ((1, self.team1), (2, self.team2), (3, self.team3)):
            if team is None:
                teams[number] = None
                continue
            teams[number] = {
                "position": tuple(team.full_path[-1]),
                "created_day": team.created_day,
                "steps": self._get_team_steps_for_day(team, self.current_day),
                "visited_hexes": frozenset(team.visited_hexes),
                "free_exploration_hexes": frozenset(team.free_exploration_hexes),
                "b_discount_remaining": team.b_discount_remaining,
                "x_bonus_remaining": team.x_bonus_remaining,
                "z_bonus_remaining": team.z_bonus_remaining,
            }
        return {
            "day": self.current_day,
            "food": self.current_food,
            "total_food": self.total_food,
            "total_reward": self.total_reward,
            "fly_skill_limit": self.fly_skill_limit,
            "visited_hexes": frozenset(self.all_visited_hexes),
            "visited_g_lands": frozenset(self.visited_g_lands),
            "active_team": self._team_number(self.active_team),
            "teams": teams,
        }

    def _target_to_data(self, target: Hex) -> Tuple[float, float]:
        x, y = game._center(*target)
        return x * game.X_SCALE, y * game.Y_SCALE

    def _action_info(self, before: Dict[str, Any], action: RouteAction) -> Dict[str, Any]:
        after = self.get_observation()
        team_before = before["teams"].get(action.team)
        team_after = after["teams"].get(action.team)
        path = tuple(self.active_team.full_path)
        return {
            "action": action,
            "path": path,
            "jump_count": sum(1 for h in self.active_team._seg_jumps[-1]) if self.active_team._seg_jumps else 0,
            "food_delta": after["food"] - before["food"],
            "score_delta": after["total_reward"] - before["total_reward"],
            "team_before": deepcopy(team_before),
            "team_after": deepcopy(team_after),
            "status": self._status_msg,
        }

    def step(self, action: RouteAction):
        """Apply one high-level action and return observation, reward, done, info."""
        before = self.get_observation()
        if not isinstance(action, RouteAction):
            raise TypeError("action must be a RouteAction")
        self._select_team(action.team)

        if action.action_type == ActionType.MOVE_TO_HEX:
            if action.target is None:
                raise ValueError("move_to_hex requires target")
            self.current_day = max(self.current_day, self.active_team.max_day_reached)
            self._process_path_click(*self._target_to_data(tuple(action.target)))
        elif action.action_type == ActionType.REPLAY_PATH:
            if not action.path:
                raise ValueError("replay_path requires a non-empty path")
            replay_path = tuple(tuple(hex_pos) for hex_pos in action.path)
            if action.target is not None and tuple(action.target) != replay_path[-1]:
                raise ValueError("replay_path target must match the final path hex")
            self._replay_path_override = replay_path
            try:
                self._process_path_click(*self._target_to_data(replay_path[-1]))
            finally:
                self._replay_path_override = None
        elif action.action_type == ActionType.FLY_TO_HEX:
            if action.target is None:
                raise ValueError("fly_to_hex requires target")
            if not self._fly_mode:
                self._activate_fly_skill()
            self._process_path_click(*self._target_to_data(tuple(action.target)))
        elif action.action_type == ActionType.SETTLE_EXPLORATION:
            self._confirm_settle_current_exploration()
        elif action.action_type == ActionType.ADVANCE_DAY:
            self._advance_day()
        elif action.action_type == ActionType.CHOOSE_PORTAL:
            self.portal_teleport = bool(action.teleport)
            self._check_portal_teleport()
        else:
            raise ValueError(f"unsupported action: {action.action_type}")

        after = self.get_observation()
        info = self._action_info(before, action)
        info["observation_before"] = before
        info["observation_after"] = after
        reward = after["total_reward"] - before["total_reward"]
        done = self.current_day >= game.TOTAL_DAYS
        return after, reward, done, info
