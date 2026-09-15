"""Focused tests for the GUI-independent RL action helpers."""

import rl_playground as rl


def test_action_space_includes_jump_as_move_outcome():
    description = rl.action_space_description()
    assert description[rl.ActionType.MOVE_TO_HEX.value]["target"]
    assert "A*" in description[rl.ActionType.MOVE_TO_HEX.value]["note"]


def test_plan_move_reports_jumps_separately():
    start = next(
        (ir, ic)
        for ir in range(rl.ROWS)
        for ic in range(rl.COLS)
        if rl._passable(ir, ic)
        and any(rl._passable(*hex_pos) for hex_pos in rl._neighbors(ir, ic))
    )
    target = next(hex_pos for hex_pos in rl._neighbors(*start) if rl._passable(*hex_pos))
    outcome = rl.plan_move(start, target, [start])
    assert outcome.path
    assert outcome.path[-1] == target
    assert outcome.jump_count == 0
    assert outcome.reward_total >= 0


def test_fly_frontier_excludes_occupied_hexes():
    owned = (20, 20)
    bridge = rl._neighbors(*owned)[0]
    occupied = {owned, bridge}
    frontier = rl.fly_frontier(owned, occupied)
    assert frontier
    assert not frontier & occupied


def test_fly_frontier_can_use_another_teams_occupied_bridge():
    current_team_hex = (20, 20)
    other_team_bridge = rl._neighbors(*current_team_hex)[0]
    target = next(
        hex_pos
        for hex_pos in rl._neighbors(*other_team_bridge)
        if hex_pos != current_team_hex and rl._passable(*hex_pos)
    )

    frontier = rl.fly_frontier(
        current_team_hex,
        [current_team_hex, other_team_bridge],
    )

    assert target in frontier


if __name__ == "__main__":
    test_action_space_includes_jump_as_move_outcome()
    test_plan_move_reports_jumps_separately()
    test_fly_frontier_excludes_occupied_hexes()
    print("RL playground tests passed")
