"""Read-only route viewer for Naruto Marching S25 saves.

This reuses the GUI engine in hex_core.py unchanged and only disables the
remaining interactions, keeping map viewing, day navigation, team-view
switching, stats, and screenshot.

It imports hex_core, NOT hex_pathfinding_demo. The editor half - route
drawing and pricing, undo, day-range edits, enclosure mode, saving, the Excel
export - lives in hex_pathfinding_demo.py and is deliberately never imported
here, so none of it is bundled into the viewer executable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hex_core as game
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.transforms import Affine2D


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
    '_btn_import_map',
)

# Display-toggle buttons the viewer doesn't offer: future-route preview and
# the B/X/Z buff labels are always off (see __init__), and per-save global
# stats aren't a viewer feature. Hidden the same way as the editing buttons
# above, for the same reason (the base class's redraw helpers still touch
# these button objects unconditionally).
_DISPLAY_TOGGLE_BUTTONS = (
    '_btn_show_future',
    '_btn_chk_labels',
    '_btn_global_stat',
)


# 当日路线的格子底色: 半透明压在实景地图上(地块内容仍要看得见), 压在路线线条下面
# (zorder 比路线的 9/10 低、比地块的 1 和背景图的 0 高), 号码另画在最上层。
#
# 底色取队伍路线色, 但先往白里提一大截: 1 队的路线色是纯黑, 直接铺上去在深色地块上
# 几乎看不出变化; 提亮之后变成灰色薄雾, 压在深浅两种地块上都能一眼认出来, 同时仍然
# 保留队伍的色相(2 队偏红、3 队偏蓝), 三队同一天行动时分得开。
_STEP_HEX_LIGHTEN = 0.55
_STEP_HEX_ALPHA = 0.5
_STEP_HEX_ZORDER = 4
# 号码字号 = 格子在屏幕上的尺寸 x 这个系数 —— 号码始终占格子的固定比例, 缩放时
# 大小跟着格子走, 不会像按线宽算那样放大后涨出格外。
#
# 所有号码一个字号, 不按位数缩: 两位数不能把单个数字画得比一位数小, 否则同一天的
# 号码看着大小不一。两位数因此会占得宽一些, 这是刻意的。
_STEP_LABEL_FONT_RATIO = 0.55


def _lighten_color(hex_color, amount=0.4):
    """Blend a hex color amount of the way toward white."""
    r, g, b = game.hex2color(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _darken_rgb(rgb, amount=0.4):
    """Blend an (r, g, b) tuple amount of the way toward black."""
    return tuple(c * (1 - amount) for c in rgb)


class RouteViewerApp(game.HexCore):
    """Read-only variant of the engine: view saves, no editing."""

    def __init__(self):
        # Viewer-only background color; _draw() and figure/axes setup in the
        # base class read this shared module constant by name, so setting it
        # before construction recolors everything derived from it.
        game.APP_BACKGROUND_COLOR = '#344239'

        # HexCore.__init__ ends by calling plt.show(), which blocks
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

        # 读取 shared its bottom-right slot with 保存 (each half-width); now
        # that 保存 is hidden (in _ROUTE_EDITING_BUTTONS), 读取 can take the
        # whole slot at the same width as the other bottom-row buttons.
        self._btn_load.ax.set_position([0.916, 0.002, 0.084, 0.025])

        # Viewer defaults: no future-route preview, no B/X/Z buff labels -
        # both buttons that would toggle these are hidden above, so these
        # flags are the only way to set the default and they can never be
        # changed back on from the UI.
        self._show_future_paths = False
        self._show_bonus_labels = False

        # 基类会在每次重绘时按 _window_base_title + 存档名(+改动星号)重设标题, 所以
        # 这里只改基础标题, 不能直接 set_window_title —— 那样会被下一次重绘盖掉。
        # 查看器改不了路线, 星号永远不会出现。
        self._window_base_title = '远征路线查看器 S25 (只读)'
        self._update_window_title()
        self._status_msg = '路线查看器：点击"读取"加载存档。'
        self._show_image_map_on_startup()
        self._draw()
        plt.show()

    def _show_image_map_on_startup(self):
        """Viewer has no map-toggle button, so start directly on the image map."""
        if self._load_map_image():
            self._map_view_mode = 'image'

    # Browsing a save is not the same as playing one: whatever the viewer has
    # panned and zoomed to is the thing they want to keep looking at while they
    # step through days or flip between teams. Both of the base class's
    # recentering entry points are disabled so switching team (1/2/3, the team
    # buttons) and switching day (Q/E, the day picker, prev/next) leave
    # the view exactly where it is. Loading a save still refits the whole map,
    # since _load_game clears _has_zoomed.
    def _center_view_on_active_team(self):
        """No-op: switching team must not move the view."""

    def _center_view_on_active_team_day(self, day, move_if_empty=True):
        """No-op: switching day must not move the view."""

    def _hide_route_editing_buttons(self):
        for name in _ROUTE_EDITING_BUTTONS + _DISPLAY_TOGGLE_BUTTONS:
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

    def _hex_screen_size_pt(self):
        """一个格子当前在屏幕上有多大(单位 pt), 取"高"和"宽 x 0.85"里小的那个。

        号码字号按这个数算, 所以无论怎么缩放, 号码永远占格子的固定比例。之前是按
        路线线宽(_get_path_line_scale)推算字号的 —— 那个系数是为线条调的, 放大之后
        号码会明显涨出格子外面, 相邻两格的号码叠在一起。

        取宽的 0.85 是因为格子是平顶六边形: 左右两头是尖角, 能放字的宽度比外接宽度窄。
        """
        try:
            dy = float(game.np.sqrt(3) * game.HEX_SIZE * game.Y_SCALE)  # 行间距 = 格高
            dx = float(2 * game.HEX_SIZE * game.X_SCALE)                # 格子外接宽度
            (x0, y0), (x1, y1) = self.ax.transData.transform([(0.0, 0.0), (dx, dy)])
            px_per_pt = self.fig.dpi / 72.0
            h_pt = abs(y1 - y0) / px_per_pt
            w_pt = abs(x1 - x0) / px_per_pt
            return max(min(h_pt, w_pt * 0.85), 1.0)
        except Exception:
            return 12.0

    def _draw_day_step_numbers(self):
        """给当日要走的每个地块铺一层半透明底色, 并在格心标出步序号。

        以前的做法是在格心画一个小圆圈当号码牌(scatter + text)。圆圈本身挡住了地块
        图案, 而且"今天要走哪几格"得靠一串小圆点去认。现在改成整格上色: 半透明压在
        实景地图上, 地块内容照样看得见, 一眼就能圈出当天的范围。

        底色取该格所属队伍的路线色再往白里提(_STEP_HEX_LIGHTEN), 三队同一天行动时
        仍然分得开。色块压在**当日**路线下面(z=4, 当日路线是 5/6), 所以今天这条线
        照旧完整地画在上面; 号码盖在最上层, 白字加黑描边, 落在深色或浅色底上都读
        得出来。

        色块尺寸就是地块本身, 不随缩放变形; 字号按格子的屏幕尺寸算
        (_hex_screen_size_pt), 所以号码始终占格子的固定比例。
        """
        step_hexes = self._compute_day_action_step_numbers(self.current_day)
        if not step_hexes:
            return

        # 一个字号画到底: 一位数两位数一样大(见 _STEP_LABEL_FONT_RATIO)。
        font_size_pt = max(4.0, self._hex_screen_size_pt() * _STEP_LABEL_FONT_RATIO)

        # 一次 PolyCollection 画完所有色块(而不是每格一个 patch): 平移缩放时每帧都要
        # 重画全部常驻 artist, artist 越少越跟手 —— 和基类批量画地块是同一个理由。
        verts, facecolors = [], []
        for hex_pos, entry in step_hexes.items():
            ir, ic = hex_pos
            if not (0 <= ir < game.ROWS and 0 <= ic < game.COLS):
                continue
            cx, cy = game._center(ir, ic)
            verts.append(game._HEX_VERT_OFFSETS + (cx, cy))
            facecolors.append(_lighten_color(
                self.team_colors[entry['team_num']], _STEP_HEX_LIGHTEN))
            label = ','.join(str(n) for n in sorted(entry['numbers']))
            self.ax.text(
                cx * game.X_SCALE, cy * game.Y_SCALE, label,
                ha='center', va='center',
                fontsize=font_size_pt, fontweight='bold',
                color='white', zorder=21,
                path_effects=[patheffects.withStroke(
                    linewidth=max(0.8, font_size_pt * 0.16), foreground='black')],
            )

        if verts:
            overlay = PolyCollection(
                verts, facecolors=facecolors, edgecolors='none',
                alpha=_STEP_HEX_ALPHA, zorder=_STEP_HEX_ZORDER,
            )
            # 顶点是未缩放的格子坐标, 和基类画地块用的是同一套变换。
            overlay.set_transform(
                Affine2D().scale(game.X_SCALE, game.Y_SCALE) + self.ax.transData)
            self.ax.add_collection(overlay)

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
