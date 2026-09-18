"""Read-only route viewer for Naruto Marching S25 saves.

This reuses the GUI game engine in hex_pathfinding_demo.py unchanged and only
disables the route-drawing/editing interactions, keeping map viewing, day
navigation, team-view switching, stats, and screenshot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hex_pathfinding_demo as game
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Buttons tied to modifying a route (drawing, undoing, editing, saving,
# exporting) are hidden rather than deleted: several of the base class's
# internal _draw()-time refresh helpers (_update_fly_button_state,
# _update_edit_seg_blink_state, ...) unconditionally touch these button
# objects, so removing the attributes would crash every redraw. Hiding them
# and moving them off-canvas keeps those helpers working as harmless no-ops
# while making the buttons impossible to see or click.
_ROUTE_EDITING_BUTTONS = (
    '_btn_undo',
    '_btn_reset',
    '_btn_fly',
    '_btn_edit_seg',
    '_btn_enclosure',
    '_btn_export_xlsx',
    '_btn_save',
    '_btn_map_view',
)


def _lighten_color(hex_color, amount=0.4):
    """Blend a hex color amount of the way toward white."""
    r, g, b = game.hex2color(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _darken_rgb(rgb, amount=0.4):
    """Blend an (r, g, b) tuple amount of the way toward black."""
    return tuple(c * (1 - amount) for c in rgb)


class RouteViewerApp(game.PathfindingDemo):
    """Read-only variant of PathfindingDemo: view saves, no editing."""

    def __init__(self):
        # Viewer-only background color; _draw() and figure/axes setup in the
        # base class read this shared module constant by name, so setting it
        # before construction recolors everything derived from it.
        game.APP_BACKGROUND_COLOR = '#344239'

        # PathfindingDemo.__init__ ends by calling plt.show(), which blocks
        # until the window closes - hiding buttons after super().__init__()
        # would never run. Suppress that call, finish our own setup, then
        # show the window ourselves.
        real_show = plt.show
        plt.show = lambda *args, **kwargs: None
        try:
            super().__init__()
        finally:
            plt.show = real_show

        self._hide_route_editing_buttons()
        self.fig.canvas.manager.set_window_title('远征路线查看器 S25 (只读)')
        self._status_msg = '路线查看器：点击"读取"加载存档。'
        self._show_image_map_on_startup()
        self._draw()
        plt.show()

    def _show_image_map_on_startup(self):
        """Viewer has no map-toggle button, so start directly on the image map."""
        if self._load_map_image():
            self._map_view_mode = 'image'

    def _team_position_for_day(self, team, day):
        """Return the hex to center on when switching to `team` while viewing `day`.

        If the team acted on `day`, this is the position of their first step
        that day. Otherwise it's their position as of their most recent action
        at or before `day` (not their absolute final position in the save).
        """
        infos = self._get_team_segment_infos(team)
        day_segs = [s for s in infos if s['day'] == day]
        if day_segs:
            nodes = day_segs[0].get('path_nodes') or []
            return tuple(nodes[0]) if nodes else tuple(day_segs[0]['end_pos'])

        prior_segs = [s for s in infos if s['day'] <= day]
        if prior_segs:
            return tuple(prior_segs[-1]['end_pos'])
        return tuple(team.origin)

    def _center_view_on_active_team(self):
        """Center on the active team's day-relevant position, not its final one.

        Overrides the base behavior (which always centers on
        team.full_path[-1], i.e. the team's absolute end-of-game position)
        since a viewer browsing a specific day should focus on where that team
        was that day, not where they ended up much later in the archive.
        """
        team = self.active_team
        if team is None:
            return
        target = self._team_position_for_day(team, self.current_day)
        cx, cy = game._center(*target)
        self._center_view_on_scaled_point(cx * game.X_SCALE, cy * game.Y_SCALE)

    def _hide_route_editing_buttons(self):
        for name in _ROUTE_EDITING_BUTTONS:
            btn = getattr(self, name, None)
            if btn is None:
                continue
            btn.ax.set_visible(False)
            # Move far outside the figure so it can never receive a click,
            # regardless of whether the backend still hit-tests hidden axes.
            btn.ax.set_position([2.0, 2.0, 0.001, 0.001])

    def _draw(self):
        super()._draw()
        self._draw_past_day_route_overlay()
        self._draw_day_step_numbers()

    def _draw_past_day_route_overlay(self):
        """Redraw every route edge from before the viewed day 40% darker.

        The base _draw() picks one color per team and reuses it for that
        team's entire path regardless of day, only special-casing the
        currently-viewed day with a highlight outline. This mirrors that same
        per-edge geometry (straight/jump-arc/fly-curve) for edges whose day is
        strictly before self.current_day, and draws them again on top in a
        darkened color so only past days visually dim.

        All darkened edges are batched into one LineCollection instead of one
        ax.plot() call per edge (previously up to several hundred separate
        artists per redraw), since pan/zoom re-render every standing artist
        on the axes each frame - fewer artists means noticeably smoother
        panning and zooming.
        """
        path_line_scale = self._get_path_line_scale()
        line_width_factor = 1.4  # matches the route-line width formula in _draw()

        segments = []
        seg_colors = []
        seg_linewidths = []

        for team, team_num in ((self.team1, 1), (self.team2, 2), (self.team3, 3)):
            if team is None or len(team.full_path) <= 1:
                continue

            base_rgb = game.hex2color(self.team_colors[team_num])
            dark_new_rgba = _darken_rgb(base_rgb, 0.4) + (1.0,)
            jump_rgb = tuple((c * 0.55) + 0.45 for c in base_rgb)
            dark_jump_rgba = _darken_rgb(jump_rgb, 0.4) + (0.9,)
            dark_fly_rgba = _darken_rgb(base_rgb, 0.4) + (0.72,)

            path_to_day = {}
            path_to_seg = {}
            path_to_action = {}
            path_idx = 1
            for seg_idx, seg_len in enumerate(team._seg_lengths):
                seg_day = team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else 1
                seg_actions = team._seg_action_sequence[seg_idx] if seg_idx < len(team._seg_action_sequence) else []
                for offset in range(seg_len):
                    if path_idx < len(team.full_path):
                        path_to_seg[path_idx] = seg_idx
                        path_to_day[path_idx] = seg_day
                        if offset < len(seg_actions) and isinstance(seg_actions[offset], (list, tuple)) and len(seg_actions[offset]) >= 1:
                            path_to_action[path_idx] = seg_actions[offset][0]
                        else:
                            path_to_action[path_idx] = 'new'
                        path_idx += 1

            for i in range(1, len(team.full_path)):
                prev_pos = team.full_path[i - 1]
                curr_pos = team.full_path[i]

                seg_idx_for_edge = path_to_seg.get(i)
                seg_day = path_to_day.get(i, team.created_day)
                if seg_day >= self.current_day:
                    continue

                is_fly_edge = (
                    seg_idx_for_edge is not None
                    and seg_idx_for_edge < len(team._seg_is_fly_skill)
                    and team._seg_is_fly_skill[seg_idx_for_edge]
                )

                token_prev = game.RAW_MAP[prev_pos[0]][prev_pos[1]] if 0 <= prev_pos[0] < game.ROWS and 0 <= prev_pos[1] < game.COLS else ''
                token_curr = game.RAW_MAP[curr_pos[0]][curr_pos[1]] if 0 <= curr_pos[0] < game.ROWS and 0 <= curr_pos[1] < game.COLS else ''
                is_prev_portal = bool(game.re.fullmatch(r'P\d+', token_prev))
                is_curr_portal = bool(game.re.fullmatch(r'P\d+', token_curr))
                is_adjacent_edge = curr_pos in game._neighbors(*prev_pos)

                if (not is_adjacent_edge) and (is_prev_portal or is_curr_portal) and (not is_fly_edge):
                    continue
                if prev_pos != curr_pos and is_prev_portal and is_curr_portal:
                    continue

                is_no_draw_edge = (
                    (prev_pos, curr_pos) in team._no_draw_edges
                    or (curr_pos, prev_pos) in team._no_draw_edges
                )
                prev_x, prev_y = game._center(*prev_pos)
                curr_x, curr_y = game._center(*curr_pos)
                x0, y0 = prev_x * game.X_SCALE, prev_y * game.Y_SCALE
                x1, y1 = curr_x * game.X_SCALE, curr_y * game.Y_SCALE

                if is_no_draw_edge:
                    if not is_fly_edge:
                        continue
                    # Portal-teleport fly connector: same quadratic-bezier curve
                    # the base draw uses, just darkened.
                    dx, dy = x1 - x0, y1 - y0
                    d = self._np_hypot(dx, dy)
                    if d <= 1e-6:
                        continue
                    nx, ny = -dy / d, dx / d
                    bend = max(0.22 * d, 3.0)
                    xm, ym = (x0 + x1) * 0.5 + nx * bend, (y0 + y1) * 0.5 + ny * bend
                    tvals = [t / 23.0 for t in range(24)]
                    curve_x = [(1 - t) ** 2 * x0 + 2 * (1 - t) * t * xm + t ** 2 * x1 for t in tvals]
                    curve_y = [(1 - t) ** 2 * y0 + 2 * (1 - t) * t * ym + t ** 2 * y1 for t in tvals]
                    segments.append(list(zip(curve_x, curve_y)))
                    seg_colors.append(dark_fly_rgba)
                    seg_linewidths.append(max(0.55 * path_line_scale * line_width_factor, 0.06))
                    continue

                action_type = path_to_action.get(i, 'new')
                jump_width_factor = 0.55 if action_type == 'jump' else 1.0
                edge_rgba = dark_jump_rgba if action_type == 'jump' else dark_new_rgba

                if action_type == 'jump' and is_adjacent_edge:
                    xs, ys = game._quarter_circle_arc(x0, y0, x1, y1)
                else:
                    xs, ys = [x0, x1], [y0, y1]

                base_lw = 2.5 * path_line_scale * jump_width_factor * line_width_factor
                segments.append(list(zip(xs, ys)))
                seg_colors.append(edge_rgba)
                seg_linewidths.append(base_lw)

        if not segments:
            return

        overlay = LineCollection(
            segments, colors=seg_colors, linewidths=seg_linewidths,
            zorder=9, capstyle='round', joinstyle='round',
        )
        self.ax.add_collection(overlay)

    @staticmethod
    def _np_hypot(dx, dy):
        return (dx * dx + dy * dy) ** 0.5

    def _compute_day_action_step_numbers(self, day):
        """Return {(ir, ic): {'numbers': [...], 'team_num': int}} for one day.

        Only 'new' (fresh land capture) entries are numbered. 'jump'
        (revisit/traverse-through-taken-hex) and 'fly' (the Flying Thunder
        God move itself) entries are skipped entirely - flying somewhere is
        not itself a numbered step, only actually occupying land is; a fly
        that captures on landing still gets exactly one number, from its
        'new' entry. Segments are ordered by their persisted global action
        order, so e.g. team 1's day actions are numbered before team 2's if
        team 1 acted first that day.
        """
        self._normalize_missing_action_orders()
        action_segments = []
        for team_num, team in ((1, self.team1), (2, self.team2), (3, self.team3)):
            if team is None:
                continue
            for seg_idx, seg_day in enumerate(team._seg_days):
                if seg_day != day:
                    continue
                actions = team._seg_action_sequence[seg_idx] if seg_idx < len(team._seg_action_sequence) else []
                if not actions:
                    continue
                action_segments.append({
                    'order': team._seg_action_orders[seg_idx],
                    'team_num': team_num,
                    'actions': actions,
                })
        action_segments.sort(key=lambda item: item['order'])

        step_number = 0
        step_hexes = {}
        for seg in action_segments:
            for kind, hex_pos in seg['actions']:
                if kind != 'new':
                    continue
                step_number += 1
                hex_pos = tuple(hex_pos)
                entry = step_hexes.setdefault(hex_pos, {'numbers': [], 'team_num': seg['team_num']})
                entry['numbers'].append(step_number)
        return step_hexes

    def _draw_day_step_numbers(self):
        """Overlay each hex on the currently-viewed day's route with its step number(s).

        Circle diameter and font size are both derived from the same
        zoom-aware path-line-width formula _draw() uses for route lines, so
        they refresh together with the route line thickness on zoom. Each
        circle is filled with that hex's team's route color lightened 40%.
        """
        step_hexes = self._compute_day_action_step_numbers(self.current_day)
        if not step_hexes:
            return

        line_width_factor = 1.4  # matches the route-line width formula in _draw()
        path_line_scale = self._get_path_line_scale()
        line_width_pt = 2.5 * path_line_scale * line_width_factor
        circle_diameter_pt = line_width_pt * 3
        font_size_pt = max(4.0, circle_diameter_pt * 0.5)

        # One scatter call for every circle (instead of one ax.plot() per hex)
        # plus per-hex text labels - far fewer standing artists means pan/zoom
        # has much less to re-render each frame.
        xs, ys, facecolors = [], [], []
        for hex_pos, entry in step_hexes.items():
            ir, ic = hex_pos
            if not (0 <= ir < game.ROWS and 0 <= ic < game.COLS):
                continue
            cx, cy = game._center(ir, ic)
            x, y = cx * game.X_SCALE, cy * game.Y_SCALE
            xs.append(x)
            ys.append(y)
            facecolors.append(_lighten_color(self.team_colors[entry['team_num']], 0.4))
            label = ','.join(str(n) for n in sorted(entry['numbers']))
            self.ax.text(
                x, y, label, ha='center', va='center',
                fontsize=font_size_pt, fontweight='bold', color='black', zorder=21,
            )

        if xs:
            self.ax.scatter(
                xs, ys, s=circle_diameter_pt ** 2,
                facecolors=facecolors, edgecolors='black',
                linewidths=max(0.6, line_width_pt * 0.3),
                zorder=20,
            )


    def _on_click(self, event):
        """Keep pan-release and the day-badge picker; drop map path drawing."""
        self._clear_hover_preview()
        self._on_release(event)

        if event.button != 1:
            return

        try:
            if self._day_number_text is not None:
                hit_day_badge, _ = self._day_number_text.contains(event)
                if hit_day_badge:
                    self._show_day_picker()
                    return
        except Exception:
            pass

        # No _process_path_click call here: map clicks do nothing in viewer.

    def _on_key_press(self, event):
        """Keep team-switch and day-navigation hotkeys; drop export ('x')."""
        if event.key in ('1', '2', '3'):
            self._on_team_button_click(int(event.key))
        elif event.key.lower() == 'q':
            self._go_previous_day()
            self.fig.canvas.draw()
        elif event.key.lower() == 'e':
            self._advance_day()
            self.fig.canvas.draw()

    def _on_hover_timeout(self):
        """Disable the hypothetical-move path preview; nothing to draw here."""


if __name__ == '__main__':
    RouteViewerApp()
