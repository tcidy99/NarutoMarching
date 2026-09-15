"""Golden-trace recording and replay helpers for the headless RL engine."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction


def _json_value(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def comparable_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    return _json_value(observation)


def action_to_json(action: RouteAction) -> Dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "target": list(action.target) if action.target is not None else None,
        "team": action.team,
        "teleport": action.teleport,
        "path": [list(hex_pos) for hex_pos in action.path] if action.path is not None else None,
    }


def action_from_json(data: Dict[str, Any]) -> RouteAction:
    target = data.get("target")
    return RouteAction(
        action_type=ActionType(data["action_type"]),
        target=tuple(target) if target is not None else None,
        team=int(data.get("team", 1)),
        teleport=data.get("teleport"),
        path=tuple(tuple(hex_pos) for hex_pos in data["path"]) if data.get("path") is not None else None,
    )


def record_trace(actions: Iterable[RouteAction], engine=None) -> Dict[str, Any]:
    engine = engine or NarutoMarchingEngine()
    initial = comparable_observation(engine.reset())
    records: List[Dict[str, Any]] = []
    for index, action in enumerate(actions):
        before = comparable_observation(engine.get_observation())
        after, reward, done, info = engine.step(action)
        records.append({
            "index": index,
            "action": action_to_json(action),
            "before": before,
            "after": comparable_observation(after),
            "reward": reward,
            "done": done,
            "jump_count": info["jump_count"],
            "food_delta": info["food_delta"],
            "score_delta": info["score_delta"],
        })
    return {"initial": initial, "records": records}


def save_trace(path: str, trace: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def replay_trace(path: str, engine=None):
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    engine = engine or NarutoMarchingEngine()
    actual_initial = comparable_observation(engine.reset())
    if actual_initial != expected["initial"]:
        return {"index": -1, "field": "initial", "expected": expected["initial"], "actual": actual_initial}

    for expected_record in expected["records"]:
        action = action_from_json(expected_record["action"])
        before = comparable_observation(engine.get_observation())
        actual_after, actual_reward, actual_done, info = engine.step(action)
        actual = {
            "before": before,
            "after": comparable_observation(actual_after),
            "reward": actual_reward,
            "done": actual_done,
            "jump_count": info["jump_count"],
            "food_delta": info["food_delta"],
            "score_delta": info["score_delta"],
        }
        expected_view = {key: expected_record[key] for key in actual}
        if actual != expected_view:
            return {
                "index": expected_record["index"],
                "field": next(key for key in actual if actual[key] != expected_view[key]),
                "expected": expected_view,
                "actual": actual,
            }
    return None
