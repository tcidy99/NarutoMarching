"""Headless action helpers for training agents on the S25 map.

This module is intentionally separate from the Matplotlib GUI.  It does not
create a window or mutate PathfindingDemo state.  The GUI can continue using
hex_pathfinding_demo.py unchanged while an RL environment is built on top of
these deterministic action descriptions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hex_pathfinding_demo import RAW_MAP, ROWS, COLS, _astar, _neighbors, _passable, _terrain

Hex = Tuple[int, int]


class ActionType(str, Enum):
    """High-level actions exposed to an RL agent."""

    MOVE_TO_HEX = "move_to_hex"
    REPLAY_PATH = "replay_path"
    FLY_TO_HEX = "fly_to_hex"
    SETTLE_EXPLORATION = "settle_exploration"
    ADVANCE_DAY = "advance_day"
    CHOOSE_PORTAL = "choose_portal"


@dataclass(frozen=True)
class RouteAction:
    """An agent action before it is applied to a game state."""

    action_type: ActionType
    target: Optional[Hex] = None
    team: int = 1
    teleport: Optional[bool] = None
    path: Optional[Tuple[Hex, ...]] = None


@dataclass(frozen=True)
class RouteOutcome:
    """Deterministic route details useful as an RL transition info payload."""

    path: Tuple[Hex, ...]
    new_hexes: Tuple[Hex, ...]
    jumps: Tuple[Hex, ...]
    food_by_hex: Tuple[int, ...]
    reward_by_hex: Tuple[int, ...]

    @property
    def jump_count(self) -> int:
        """Number of revisit/jump actions in the selected route."""
        return len(self.jumps)

    @property
    def food_total(self) -> int:
        return sum(self.food_by_hex)

    @property
    def reward_total(self) -> int:
        return sum(self.reward_by_hex)


def action_space_description() -> Dict[str, Dict[str, object]]:
    """Return a serializable description of the proposed high-level action space."""
    return {
        ActionType.MOVE_TO_HEX.value: {
            "target": "passable hex coordinate",
            "note": "A* expands this into new-land and jump actions.",
        },
        ActionType.REPLAY_PATH.value: {
            "path": "complete entered-hex sequence from the archive",
            "note": "Used for deterministic historical replay; bypasses A* route selection.",
        },
        ActionType.FLY_TO_HEX.value: {
            "target": "unoccupied hex coordinate in the validated fly frontier",
        },
        ActionType.SETTLE_EXPLORATION.value: {
            "target": "current team's unsettled exploration hex",
        },
        ActionType.ADVANCE_DAY.value: {
            "target": None,
        },
        ActionType.CHOOSE_PORTAL.value: {
            "teleport": "boolean",
        },
    }


def _normalise_hex(value: Sequence[int]) -> Hex:
    if len(value) != 2:
        raise ValueError("hex coordinate must contain exactly two integers")
    result = (int(value[0]), int(value[1]))
    if not (0 <= result[0] < ROWS and 0 <= result[1] < COLS):
        raise ValueError(f"hex coordinate out of bounds: {result}")
    return result


def _path_actions(path: Iterable[Hex], occupied: set) -> Tuple[List[Hex], List[Hex]]:
    new_hexes: List[Hex] = []
    jumps: List[Hex] = []
    seen = set(occupied)
    for hex_pos in path:
        if hex_pos in seen:
            jumps.append(hex_pos)
        else:
            new_hexes.append(hex_pos)
            seen.add(hex_pos)
    return new_hexes, jumps


def plan_move(start: Sequence[int], target: Sequence[int], occupied: Iterable[Sequence[int]]) -> RouteOutcome:
    """Plan a normal high-level move and expose its jump actions.

    This is an analysis helper only: it does not charge resources or mutate the
    GUI.  The returned path excludes the starting hex, matching the route data
    recorded by the game after a click.
    """
    start_hex = _normalise_hex(start)
    target_hex = _normalise_hex(target)
    if not _passable(*target_hex):
        raise ValueError(f"target is not passable: {target_hex}")
    if target_hex == start_hex:
        raise ValueError("target must differ from current position")

    path, _ = _astar(start_hex, target_hex, set())
    if not path or len(path) < 2:
        raise ValueError(f"no path found from {start_hex} to {target_hex}")
    path = path[1:]
    if len(path) > 17:
        raise ValueError("route exceeds the 17 entered-hex limit")

    occupied_hexes = {_normalise_hex(hex_pos) for hex_pos in occupied}
    new_hexes, jumps = _path_actions(path, occupied_hexes)
    food_by_hex: List[int] = []
    reward_by_hex: List[int] = []
    for hex_pos in path:
        terrain = _terrain(*hex_pos)
        if hex_pos in occupied_hexes:
            food_by_hex.append(10)
            reward_by_hex.append(0)
        else:
            food_by_hex.append(max(0, terrain.get("food", 0)) + 50)
            reward_by_hex.append(terrain.get("award", 0))

    return RouteOutcome(
        path=tuple(path),
        new_hexes=tuple(new_hexes),
        jumps=tuple(jumps),
        food_by_hex=tuple(food_by_hex),
        reward_by_hex=tuple(reward_by_hex),
    )


def fly_frontier(start: Sequence[int], occupied: Iterable[Sequence[int]]) -> set:
    """Return unoccupied candidates bordering the occupied region connected to start."""
    start_hex = _normalise_hex(start)
    occupied_hexes = {_normalise_hex(hex_pos) for hex_pos in occupied}
    connected_occupied = {start_hex} if start_hex in occupied_hexes else set()
    frontier = [start_hex]
    seen = {start_hex}
    while frontier:
        current = frontier.pop()
        for neighbour in _neighbors(*current):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            if neighbour in occupied_hexes:
                connected_occupied.add(neighbour)
                frontier.append(neighbour)
    return {
        neighbour
        for bridge_hex in connected_occupied
        for neighbour in _neighbors(*bridge_hex)
        if neighbour not in occupied_hexes and _passable(*neighbour)
    }
