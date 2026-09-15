"""Replay ordinary archive segments across all teams in action order."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import hex_pathfinding_demo as game
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction


@dataclass(frozen=True)
class ReplayReport:
    replayed: int
    total: int
    stopped_team: Optional[int]
    stopped_segment: Optional[int]
    reason: Optional[str]
    day: int


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _apply_saved_segment(engine, team, saved_team, segment_index):
    """Apply a historical segment using persisted costs, without recosting."""
    nodes = [tuple(hex_pos) for hex_pos in saved_team["_seg_path_nodes"][segment_index]]
    end_position = tuple(saved_team["_seg_end_positions"][segment_index])
    new_hexes = [tuple(hex_pos) for hex_pos in saved_team["_seg_new_hexes"][segment_index]]
    exploration_hexes = [tuple(hex_pos) for hex_pos in saved_team["_seg_exploration_hexes"][segment_index]]
    jumps = [tuple(hex_pos) for hex_pos in saved_team["_seg_jumps"][segment_index]]

    team.full_path.extend(nodes)
    if end_position != (nodes[-1] if nodes else team.full_path[-1]):
        team.full_path[-1:] = [end_position]
    team._seg_lengths.append(len(nodes))
    team._seg_foods.append(saved_team["_seg_foods"][segment_index])
    team._seg_awards.append(saved_team["_seg_awards"][segment_index])
    team._seg_steps.append(saved_team["_seg_steps"][segment_index])
    team._seg_days.append(saved_team["_seg_days"][segment_index])
    team._seg_new_hexes.append(new_hexes)
    team._seg_exploration_hexes.append(exploration_hexes)
    team._seg_jumps.append(jumps)
    team._seg_path_nodes.append(nodes)
    team._seg_end_positions.append(end_position)
    team._seg_action_sequence.append([
        (action, tuple(hex_pos) if isinstance(hex_pos, (list, tuple)) else hex_pos)
        for action, hex_pos in saved_team["_seg_action_sequence"][segment_index]
    ])
    engine._append_segment_action_order(team)
    team._seg_hex_costs.append(saved_team["_seg_hex_costs"][segment_index])
    team._seg_is_fly_skill.append(False)
    team._seg_fly_skill_deltas.append(saved_team["_seg_fly_skill_deltas"][segment_index])

    for hex_pos in new_hexes:
        team.visited_hexes.add(hex_pos)
        engine.all_visited_hexes.add(hex_pos)
        if hex_pos in engine.all_g_lands:
            engine.visited_g_lands.add(hex_pos)
    for hex_pos in exploration_hexes:
        team.visited_hexes.add(hex_pos)
        team.free_exploration_hexes.add(hex_pos)
        if hex_pos in engine.all_g_lands:
            engine.visited_g_lands.add(hex_pos)
    for hex_pos in jumps:
        team.visited_hexes.add(hex_pos)

    food = saved_team["_seg_foods"][segment_index]
    award = saved_team["_seg_awards"][segment_index]
    engine.current_food -= food
    engine.total_food += food
    engine.total_reward += award
    engine.fly_skill_limit += saved_team["_seg_fly_skill_deltas"][segment_index]
    engine._derive_team_buff_state(team)
    engine._rebuild_day_records()


def replay_ordinary_segments(path: str) -> ReplayReport:
    """Replay ordinary, non-fly, non-empty-path segments in global action order."""
    state = _load(path)
    engine = NarutoMarchingEngine()
    engine.reset()
    teams = {1: engine.team1}
    events = []
    for team_number, team_name in enumerate(("team1", "team2", "team3"), start=1):
        team = state[team_name]
        for segment_index, day in enumerate(team["_seg_days"]):
            events.append((day, team["_seg_action_orders"][segment_index], team_number, segment_index))
    events.sort()

    replayed = 0
    for day, _order, team_number, segment_index in events:
        saved_team = state[f"team{team_number}"]
        if team_number not in teams:
            origin = tuple(saved_team["origin"])
            team = game.Team(origin, created_day=saved_team["created_day"])
            setattr(engine, f"team{team_number}", team)
            teams[team_number] = team
            first_new = {tuple(hex_pos) for hex_pos in saved_team["_seg_new_hexes"][0]}
            if origin in first_new:
                team.free_exploration_hexes.add(origin)
                engine.all_visited_hexes.discard(origin)
            else:
                engine.all_visited_hexes.add(origin)

        team = teams[team_number]
        engine.active_team = team
        while engine.current_day < day:
            engine._advance_day()

        if saved_team["_seg_is_fly_skill"][segment_index]:
            fly_nodes = saved_team["_seg_path_nodes"][segment_index]
            if not fly_nodes:
                return ReplayReport(replayed, len(events), team_number, segment_index, "fly segment has no target", engine.current_day)
            fly_target = tuple(fly_nodes[-1])
            if not engine._is_valid_fly_destination(fly_target):
                return ReplayReport(
                    replayed,
                    len(events),
                    team_number,
                    segment_index,
                    "historical fly target violates current fly frontier rule",
                    engine.current_day,
                )
            engine.step(RouteAction(ActionType.FLY_TO_HEX, target=fly_target, team=team_number))
            replayed += 1
            continue

        nodes = [tuple(hex_pos) for hex_pos in saved_team["_seg_path_nodes"][segment_index]]
        if not nodes:
            if saved_team["_seg_new_hexes"][segment_index]:
                before_count = len(team._seg_days)
                engine.step(RouteAction(ActionType.SETTLE_EXPLORATION, team=team_number))
                if len(team._seg_days) != before_count + 1:
                    return ReplayReport(replayed, len(events), team_number, segment_index, "engine rejected exploration settlement", engine.current_day)
                replayed += 1
                continue
            return ReplayReport(replayed, len(events), team_number, segment_index, "empty path segment", engine.current_day)

        _apply_saved_segment(engine, team, saved_team, segment_index)
        replayed += 1

    saved_day = int(state.get("current_day", engine.current_day))
    engine.current_day = saved_day
    engine._sync_current_food_for_view_day()

    final_checks = {
        "current_food": engine.current_food,
        "total_food": engine.total_food,
        "total_reward": engine.total_reward,
        "fly_skill_limit": engine.fly_skill_limit,
    }
    for field_name, actual in final_checks.items():
        if actual != state.get(field_name):
            return ReplayReport(
                replayed,
                len(events),
                None,
                None,
                f"final state mismatch: {field_name} expected {state.get(field_name)}, got {actual}",
                engine.current_day,
            )

    for team_number in (1, 2, 3):
        saved_team = state[f"team{team_number}"]
        if saved_team is None:
            continue
        team = teams[team_number]
        if tuple(team.full_path[-1]) != tuple(saved_team["full_path"][-1]):
            return ReplayReport(
                replayed,
                len(events),
                team_number,
                None,
                "final state mismatch: team position",
                engine.current_day,
            )

    return ReplayReport(replayed, len(events), None, None, None, engine.current_day)
