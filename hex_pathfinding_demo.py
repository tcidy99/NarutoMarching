"""Naruto Marching S25 - route editor.

The read-only half of the app (map model, rendering, day navigation, save
loading) lives in hex_core.py. Only the editing half is here: drawing and
pricing routes, undo, day-range edits, enclosure mode, saving, and the Excel
export.

Keeping the split at module level is deliberate. viewer/route_viewer.py
imports hex_core alone, so none of the code below is reachable from - or
bundled into - the viewer executable. A button lambda in HexCore.__init__ may
still name a method defined here (e.g. self._undo); attribute lookup happens
at click time, so the reference costs the viewer nothing and those buttons are
hidden there anyway.
"""

import os as _os
import sys as _sys

# This module is routinely loaded by absolute path (tests, tooling) rather than
# imported as part of a package, in which case its own directory is not on
# sys.path and `import hex_core` would fail. Put it there first.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from hex_core import *
from hex_core import HexCore


# ── 远征模拟器 JSON 地图导入 ──────────────────────────────────────────────────
#
# 那个模拟器(Godot 工程)把地图导出成分层 JSON:
#   "图块数据": {"x,y": {"基础地块":1-6, "事件层":.., "特殊事件层":.., "等级信息":{"等级":n}}}
# 层ID 的含义来自它的 核心数据/配置文件/图块配置.gd。
_JM_商人, _JM_营地, _JM_试炼场, _JM_徽章, _JM_瞭望塔, _JM_石板 = 13, 14, 15, 16, 17, 18
_JM_首领, _JM_传送点, _JM_历练, _JM_出生点 = 19, 20, 21, 22

# 首领按基础等级给不同奖励(3→300分+八卦, 5→500, 6→1000+飞雷神), 正好是我们的 G/b/B。
_JM_首领映射 = {3: 'G', 5: 'b', 6: 'B'}

# 瞭望塔增益 → 我们的 token 前缀。名字取自它的 瞭望塔增益配置:
#   迅疾=不消耗行动次数, 兵粮=挑战粮草-40%, 战勋=经验+40%
_JM_增益前缀 = {'迅疾': 'X', '兵粮': 'B', '战勋': 'Z'}


def _jm_to_internal(x, y, rows):
    """他们的 (x,y) → 我们的内存坐标 (行, 列)。

    两边都是平顶六边形, 但偏移系统不同: 他们 even-q(偶数列下移), 我们 odd-q
    (奇数列上移)。互转不是刚性平移, 必须按列奇偶错位移行:
        even-q→axial: r = y-(x+(x&1))/2 ;  odd-q→axial: r = ir-(ic-(ic&1))/2
        相减得        ir = y-(x&1)
    再叠加我们自己的行翻转与列基准, 得到下式。

    这是拿"一乐拉面"(就是我们在用的 S25 地图)反解并逐格验证出来的:
    3600 格中 3596 格一致, 其余 10 格只是传送点编号顺序不同。
    """
    return rows - y + (x & 1), x - 1


def _jm_tent_schedule(阶段数据):
    """远征营地阶段 → 我们 Tent 的 (基础food, degrade表)。

    阶段N 的"粮草奖励"是该阶段帐篷给的粮草, 我们记成负数消耗。第1阶段是基础值,
    之后每个阶段的起始天进入 degrade。用一乐拉面验证过, 结果与现有 landInfo 完全一致。
    """
    阶段 = 阶段数据.get('阶段', [])
    if not 阶段:
        return None, []
    base = -int(阶段[0]['粮草奖励'])
    acc, degrade = 0, []
    for i, s in enumerate(阶段[:-1]):
        acc += int(s['天数'])
        degrade.append({'day': acc + 1, 'food': -int(阶段[i + 1]['粮草奖励'])})
    return base, degrade


def _jm_collect_towers(data):
    """列出图里所有瞭望塔: [(坐标键, 基础等级)], 按等级再按坐标排序。"""
    out = []
    for key, v in data.get('图块数据', {}).items():
        if v.get('事件层') == _JM_瞭望塔:
            out.append((key, v.get('等级信息', {}).get('等级', 0)))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def _jm_sidecar_path(json_path):
    """瞭望塔增益类型的记忆文件, 与地图同名同目录。

    地图 JSON 里没有增益类型(它存在 瞭望塔数据.增益配置, 导出的地图文件没有这个键),
    只能由使用者指定一次, 之后记在这里, 同一张图不再追问。
    """
    return _os.path.splitext(json_path)[0] + '.瞭望塔.json'


def _jm_load_tower_choice(json_path):
    p = _jm_sidecar_path(json_path)
    if not _os.path.exists(p):
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith('_') and v in _JM_增益前缀}
    except Exception as e:
        print(f'DEBUG: 读取瞭望塔记忆文件失败 {p}: {e}')
        return {}


def _jm_save_tower_choice(json_path, choice):
    p = _jm_sidecar_path(json_path)
    payload = {'_说明': '瞭望塔增益类型, 值为 迅疾/兵粮/战勋; 键是地图JSON里的 "x,y" 坐标'}
    payload.update(choice)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def _jm_convert(data, tower_choice, rows, cols):
    """JSON 地图 → 我们的 token 网格(内存行序) + 说明信息。"""
    tiles = data['图块数据']
    grid = [['0'] * cols for _ in range(rows)]
    stats = {'替换': {}, '未覆盖': 0}

    portal_id = {}
    for i, g in enumerate(data.get('传送点数据', {}).get('传送点组列表', []), start=1):
        for side in ('传送点1', '传送点2'):
            p = g[side]
            portal_id[(int(p['x']), int(p['y']))] = f'P{i}'

    徽章印记 = data.get('远征徽章数据', {}).get('印记标记', {})

    def note(msg):
        stats['替换'][msg] = stats['替换'].get(msg, 0) + 1

    for key, v in tiles.items():
        x, y = (int(p) for p in key.split(','))
        ir, ic = _jm_to_internal(x, y, rows)
        if not (0 <= ir < rows and 0 <= ic < cols):
            stats['未覆盖'] += 1
            continue

        lvl = v.get('等级信息', {}).get('等级', 0)
        base = str(lvl) if 1 <= lvl <= 6 else '0'
        ev, sp = v.get('事件层', -1), v.get('特殊事件层', -1)

        # ⚠ 顺序要紧: 3级首领本身就发八卦徽章, 那一格会同时带 特殊事件层=19 和
        # 事件层=16。必须先判首领, 它才会变成 G(200粮/300分) 而不是 g(120粮/45分)。
        # 一乐拉面/尸鬼红唇各有3格是这种双层格; 无论哪张图, 最终都是 3个G + 5个g。
        if sp == _JM_出生点:
            tok = 'ST'
        elif sp == _JM_传送点:
            tok = portal_id.get((x, y), base)
        elif sp == _JM_历练:
            tok = 'L'
        elif sp == _JM_首领:
            tok = _JM_首领映射.get(lvl, base)
        elif ev == _JM_商人:
            tok = 'M'
        elif ev == _JM_营地:
            tok = 'T'
        elif ev == _JM_徽章:
            # 远征徽章是"完全继承基础地块"规则。八卦徽章恒在3级地上, 对应我们的 g;
            # 木叶/晓 我们没有对应概念, 按脚下等级当普通地块(晓恒在4级地)。
            if 徽章印记.get(key) == '八卦':
                tok = 'g'
            else:
                tok = base
                note(f'{徽章印记.get(key)}徽章 → {base}级地')
        elif ev == _JM_石板:
            tok = 'T2'              # 遗迹石板固定 100粮/40分 == 我们的 T2
        elif ev == _JM_试炼场:
            if 2 <= lvl <= 6:
                tok = f'T{lvl}'
            else:
                tok = base          # 1级试炼场我们没有 T1, 按普通1级地
                note(f'{lvl}级试炼场 → {base}级地')
        elif ev == _JM_瞭望塔:
            prefix = _JM_增益前缀.get(tower_choice.get(key, ''))
            if prefix and 1 <= lvl <= 3:
                tok = f'{prefix}{lvl}'
            else:
                tok = base
                note(f'瞭望塔({key}) 未指定增益 → {base}级地')
        else:
            tok = base
        grid[ir][ic] = tok

    return grid, stats


class PathfindingDemo(HexCore):
    """HexCore plus every route-mutating operation."""

    def _ask_watchtower_types(self, towers, preset):
        """一次性询问整张图所有瞭望塔的增益类型。

        每张图只问一次: 选择会存进地图旁边的 .瞭望塔.json, 下次导入同一张图直接
        读取。返回 {坐标键: 增益名} 或 None(用户取消)。
        """
        import tkinter as tk

        result = {}
        root = tk.Tk()
        root.title('指定瞭望塔增益类型')
        root.attributes('-topmost', True)

        tk.Label(root, justify='left', padx=12, pady=8, text=(
            '地图文件里没有记录瞭望塔的增益类型，请为每座塔选择。\n'
            '迅疾=不消耗行动次数  兵粮=挑战粮草-40%  战勋=积分+40%\n'
            '等级决定档位：1级→X1/B1/Z1，2级→…2，3级→…3。\n'
            '选择会记住，同一张地图以后不再询问。'
        )).grid(row=0, column=0, columnspan=3, sticky='w')

        tk.Label(root, text='坐标', padx=10).grid(row=1, column=0)
        tk.Label(root, text='基础等级').grid(row=1, column=1)
        tk.Label(root, text='增益类型').grid(row=1, column=2)

        vars_ = {}
        for i, (key, lvl) in enumerate(towers):
            tk.Label(root, text=key, padx=10).grid(row=2 + i, column=0, sticky='w')
            tk.Label(root, text=f'{lvl}级').grid(row=2 + i, column=1)
            var = tk.StringVar(value=preset.get(key, '兵粮'))
            vars_[key] = var
            tk.OptionMenu(root, var, *_JM_增益前缀.keys()).grid(
                row=2 + i, column=2, sticky='ew', padx=10, pady=2)

        state = {'ok': False}

        def confirm():
            state['ok'] = True
            root.quit()

        btns = tk.Frame(root)
        btns.grid(row=2 + len(towers), column=0, columnspan=3, pady=10)
        tk.Button(btns, text='确定', width=10, command=confirm).pack(side='left', padx=6)
        tk.Button(btns, text='取消', width=10, command=root.quit).pack(side='left', padx=6)

        root.protocol('WM_DELETE_WINDOW', root.quit)
        root.mainloop()
        if state['ok']:
            result = {k: v.get() for k, v in vars_.items()}
        root.destroy()
        return result if state['ok'] else None

    def _import_json_map(self):
        """把远征模拟器的 JSON 地图转成我们的 CSV, 并同步赛季/帐篷配置。"""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title='选择远征模拟器的 JSON 地图',
                filetypes=[('JSON 地图', '*.json'), ('所有文件', '*.*')])
            root.destroy()
            if not path:
                return

            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            if '图块数据' not in data:
                self._status_msg = '导入失败：这不是远征模拟器的地图文件。'
                self._draw()
                return

            size = data.get('地图尺寸', {})
            rows = int(size.get('高度', ROWS))
            cols = int(size.get('宽度', COLS))
            if (rows, cols) != (ROWS, COLS):
                self._status_msg = f'导入失败：地图尺寸 {cols}x{rows} 与本软件的 {COLS}x{ROWS} 不一致。'
                self._draw()
                return

            # 瞭望塔增益类型: 记住过就不再问
            towers = _jm_collect_towers(data)
            choice = _jm_load_tower_choice(path)
            if towers and any(k not in choice for k, _ in towers):
                picked = self._ask_watchtower_types(towers, choice)
                if picked is None:
                    self._status_msg = '导入已取消。'
                    self._draw()
                    return
                choice = picked
                saved_to = _jm_save_tower_choice(path, choice)
                print(f'DEBUG: 瞭望塔选择已记住 -> {saved_to}')

            grid, stats = _jm_convert(data, choice, rows, cols)

            base_dir = _os.path.dirname(_os.path.abspath(
                _sys.executable if getattr(_sys, 'frozen', False) else __file__))
            name = _os.path.splitext(_os.path.basename(path))[0]
            csv_name = f'map_{name}.csv'
            csv_path = _os.path.join(base_dir, csv_name)
            # RAW_MAP 载入后会 reverse(), 所以写文件时要反过来存。
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write('\n'.join(','.join(r) for r in reversed(grid)) + '\n')

            # 同步 landInfo.json 的地图指向、赛季长度与帐篷阶段
            info_path = _os.path.join(base_dir, 'landInfo.json')
            backup = info_path + '.backup-before-import'
            with open(info_path, encoding='utf-8') as f:
                info = json.load(f)
            if not _os.path.exists(backup):
                with open(backup, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

            changes = [f'地图 → {csv_name}']
            info.setdefault('map_files', {})['csv'] = csv_name

            阶段数据 = data.get('远征营地数据', {}).get('阶段数据', {})
            总天数 = int(阶段数据.get('总天数', 0))
            if 总天数:
                info.setdefault('season', {})['duration_days'] = 总天数
                changes.append(f'赛季天数 → {总天数}')
            base_food, degrade = _jm_tent_schedule(阶段数据)
            if base_food is not None:
                info.setdefault('T', {})['food'] = base_food
                info['T']['degrade'] = degrade
                changes.append('帐篷阶段 → ' + '/'.join(str(d['day']) for d in degrade))

            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            cnt = {}
            for row in grid:
                for t in row:
                    if t != '0':
                        cnt[t] = cnt.get(t, 0) + 1
            替换 = '\n'.join(f'  {k}: {v} 处' for k, v in sorted(stats['替换'].items()))

            print(f'DEBUG: 导入地图 {name}: {sum(cnt.values())} 格 -> {csv_path}')
            self._status_msg = f'已导入地图 {name}，请重启软件生效。'
            self._draw()

            root = tk.Tk(); root.withdraw()
            messagebox.showinfo('导入完成', (
                f'地图「{name}」已导入（{sum(cnt.values())} 格）。\n\n'
                f'写出：{csv_name}\n'
                f'已更新 landInfo.json：\n  ' + '\n  '.join(changes) + '\n'
                f'（原文件已备份为 landInfo.json.backup-before-import）\n\n'
                + (f'按规则做的等价替换：\n{替换}\n\n' if 替换 else '')
                + '⚠ 地图与赛季配置在启动时载入，请重启软件后生效。'))
            root.destroy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status_msg = f'导入地图失败：{e}'
            self._draw()

    def _on_hover_timeout(self):
        """Show path preview after hover timer fires - only if hex is reachable."""
        if self._hover_hex is None:
            return

        day_locked, _ = self._is_active_team_locked_by_day()
        if day_locked:
            self._clear_hover_preview()
            return
        
        ir, ic = self._hover_hex
        if not (0 <= ir < ROWS and 0 <= ic < COLS):
            return
        
        current = self.active_team.full_path[-1]
        if (ir, ic) == current:
            return
        
        try:
            # Compute path to hover hex
            segment, _ = _astar(current, (ir, ic), set())
            if segment and len(segment) > 1:
                # Check if path is within 18 hex limit
                if len(segment) > 18:
                    return  # Path too long, don't show preview
                
                # Get available steps for today
                steps_available = self._get_team_steps_for_day(self.active_team, self.current_day)
                
                # Count new hexes (rough estimate: all hexes except current position and revisits)
                new_hexes_estimate = 0
                for h in segment[1:]:  # Skip starting position
                    if h not in self.all_visited_hexes:
                        new_hexes_estimate += 1
                
                # Quick step check: need at least one step per new hex
                effective_new_hexes = max(
                    0,
                    new_hexes_estimate - self.active_team.x_bonus_remaining,
                )
                if effective_new_hexes > steps_available:
                    return  # Not enough steps, don't show preview
                
                # Estimate food cost (conservative: movement 50 + typical terrain cost per hex)
                food_estimate = 0
                for h in segment[1:]:  # Skip starting position
                    if h in self.all_visited_hexes:
                        revisit_cost = 10
                        revisit_cost = _apply_g_reduction(revisit_cost, self._are_all_g_lands_visited())
                        food_estimate += revisit_cost
                    else:
                        t = _terrain(*h)
                        terrain_food = _get_terrain_food(t, self.current_day)
                        challenge_cost = _apply_challenge_discounts(terrain_food, self.active_team, 
                                                                   self._are_all_g_lands_visited(), 
                                                                   is_tent=t.get('name') == 'Tent')
                        movement_cost = _apply_g_reduction(50, self._are_all_g_lands_visited())
                        food_estimate += movement_cost + challenge_cost
                
                # Check if we have enough food
                if (not self._enclosure_mode) and food_estimate > self.current_food:
                    return  # Not enough food, don't show preview
                
                # Path is reachable - draw it
                xs = [_center(r, c)[0] * X_SCALE for r, c in segment]
                ys = [_center(r, c)[1] * Y_SCALE for r, c in segment]
                
                # Get team color and make it lighter (50% opacity)
                team_color = self.team_colors[1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)]
                
                # Draw in lighter color with lower z-order
                line_scale = self._get_path_line_scale()
                line_width_factor = 1.4  # 2x wider than previous path width setting
                self._hover_path_line = self.ax.plot(xs, ys, color=team_color, lw=3.5 * line_scale * line_width_factor,
                                                     zorder=2, alpha=0.4, linestyle='--',
                                                     solid_capstyle='round', solid_joinstyle='round')[0]
                self.fig.canvas.draw_idle()
        except Exception:
            pass  # Silently ignore errors in preview

    def _on_key_press(self, event):
        """Handle keyboard hotkeys - 1/2/3 to switch teams, Q/E for day navigation."""
        if event.key in ('1', '2', '3'):
            team_num = int(event.key)
            # Share the same single-switch vs double-press-jumps-to-next-day logic
            # as the T1/T2/T3 mouse buttons (see _on_team_button_click) instead of
            # always doing a plain switch, so pressing "1" twice quickly behaves
            # the same as double-clicking the T1 button.
            self._on_team_button_click(team_num)
        elif event.key.lower() == 'q':
            self._go_previous_day()
            self.fig.canvas.draw()
        elif event.key.lower() == 'e':
            self._advance_day()
            self.fig.canvas.draw()
        elif event.key.lower() == 'x':
            self._export_day_sheets_xlsx()

    def _append_segment_action_order(self, team):
        """Assign the next global ordering slot to a newly-created segment."""
        self._normalize_missing_action_orders()
        self._ensure_team_segment_action_orders(team)
        team._seg_action_orders.append(self._next_action_order)
        self._next_action_order += 1

    def _truncate_team_history_from_segment(self, team, start_seg_idx):
        """Drop one team's segment history from start_seg_idx onward.

        This is a preparation hook for future segment edit mode. It is intentionally local to
        a team; callers are responsible for replay/rebuild of shared derived state afterward.
        """
        if team is None:
            return
        self._ensure_team_segment_path_nodes(team)

        keep_count = max(0, min(start_seg_idx, len(team._seg_lengths)))
        segment_fields = [
            '_seg_lengths', '_seg_foods', '_seg_awards', '_seg_steps', '_seg_days',
            '_seg_new_hexes', '_seg_exploration_hexes', '_seg_jumps', '_seg_path_nodes', '_seg_end_positions',
            '_seg_action_sequence', '_seg_action_orders', '_seg_hex_costs', '_seg_is_fly_skill', '_seg_fly_skill_deltas',
        ]
        for field_name in segment_fields:
            value = getattr(team, field_name, None)
            if value is not None:
                setattr(team, field_name, value[:keep_count])

        rebuilt_path = [team.origin]
        for seg_nodes in team._seg_path_nodes:
            rebuilt_path.extend(seg_nodes)
        team.full_path = rebuilt_path
        self._rebuild_shared_derived_state_from_segments()
        self._rebuild_day_records()

    def _segment_field_names(self):
        """Canonical list of per-segment parallel arrays."""
        return [
            '_seg_lengths', '_seg_foods', '_seg_awards', '_seg_steps', '_seg_days',
            '_seg_new_hexes', '_seg_exploration_hexes', '_seg_jumps', '_seg_path_nodes', '_seg_end_positions',
            '_seg_action_sequence', '_seg_action_orders', '_seg_hex_costs', '_seg_is_fly_skill', '_seg_fly_skill_deltas',
        ]

    def _capture_team_segment_tail(self, team, start_idx):
        """Capture a tail slice of all per-segment arrays for later re-attach."""
        payload = {}
        for field_name in self._segment_field_names():
            value = getattr(team, field_name, None)
            payload[field_name] = list(value[start_idx:]) if value is not None else []
        return payload

    def _snapshot_team_state(self, team):
        """Deep-copy every attribute a route edit can mutate, for later rollback."""
        if team is None:
            return None
        fields = self._segment_field_names() + [
            'full_path', 'visited_hexes', 'free_exploration_hexes',
            'x_bonus_remaining', 'x_bonus_name', 'b_discount_remaining', 'b_discount_name',
            'z_bonus_remaining', 'z_bonus_name', 'max_day_reached', '_no_draw_edges',
        ]
        return {name: copy.deepcopy(getattr(team, name)) for name in fields if hasattr(team, name)}

    def _restore_team_state(self, team, snapshot):
        """Undo a route edit's mutations on one team using a prior _snapshot_team_state result."""
        if team is None or snapshot is None:
            return
        for name, value in snapshot.items():
            setattr(team, name, copy.deepcopy(value))

    def _snapshot_shared_state(self):
        """Capture the cross-team derived state a route edit rebuilds, for later rollback.

        The engine-global scalars belong here too: beginning an edit truncates
        history and re-derives them (see _truncate_team_history_from_segment),
        and the redraw clicks mutate them again. Restoring only the segment
        arrays would leave a rejected edit's numbers behind - most visibly the
        fly-skill count, since rolling back past a bigBoss capture drops its +1
        and nothing puts it back until the next save/load re-derives it.
        """
        return {
            'all_visited_hexes': set(self.all_visited_hexes),
            'visited_g_lands': set(self.visited_g_lands),
            'day_records': copy.deepcopy(self.day_records),
            'next_action_order': self._next_action_order,
            'current_day': self.current_day,
            'fly_skill_limit': self.fly_skill_limit,
            'total_food': self.total_food,
            'total_reward': self.total_reward,
            'current_food': self.current_food,
        }

    def _restore_shared_state(self, snapshot):
        """Undo a route edit's mutations on shared engine state using a prior snapshot."""
        self.all_visited_hexes = set(snapshot['all_visited_hexes'])
        self.visited_g_lands = set(snapshot['visited_g_lands'])
        self.day_records = copy.deepcopy(snapshot['day_records'])
        self._next_action_order = snapshot['next_action_order']
        self.current_day = snapshot['current_day']
        # Older snapshots (taken before these were captured) simply leave the
        # live value alone rather than crashing the rollback.
        for name in ('fly_skill_limit', 'total_food', 'total_reward', 'current_food'):
            if name in snapshot:
                setattr(self, name, snapshot[name])

    def _compute_tent_and_g_capture_days(self):
        """Map every captured Tent/G/g hex (any team) to the day it was first captured.

        Used to enforce that a route edit is not allowed to shift which day a Tent or
        G/g hex was reached on, since that day drives Tent-degrade scheduling and the
        global G/g completion timeline other segments' costs depend on.
        """
        days = {}
        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue
            for seg_day, seg_new in zip(team._seg_days, team._seg_new_hexes):
                for h in seg_new:
                    h = tuple(h)
                    terrain = _terrain(*h)
                    is_tent = terrain.get('name') == 'Tent'
                    is_g = h in self.all_g_lands
                    if not (is_tent or is_g):
                        continue
                    d = int(seg_day)
                    if h not in days or d < days[h]:
                        days[h] = d
        return days

    def _append_team_segment_payload(self, team, payload):
        """Append captured per-segment payload back to a team."""
        for field_name in self._segment_field_names():
            value = getattr(team, field_name, None)
            if value is None:
                continue
            value.extend(payload.get(field_name, []))

    def _begin_day_segment_edit(self, team, day):
        """Backward-compatible single-day entrypoint."""
        return self._begin_day_segment_edit_range(team, day, day)

    def _begin_day_segment_edit_range(self, team, start_day, end_day):
        """Delete selected day-range segments, keep future segments pending until reconnection."""
        if team is None:
            return False

        start_day = int(start_day)
        end_day = int(end_day)
        if start_day > end_day:
            start_day, end_day = end_day, start_day

        self._ensure_team_segment_path_nodes(team)
        seg_days = list(team._seg_days)
        day_indices = [idx for idx, seg_day in enumerate(seg_days) if start_day <= seg_day <= end_day]
        if not day_indices:
            self._status_msg = f'No segments exist in Day {start_day}-{end_day} for the active team.'
            return False

        first_day_idx = day_indices[0]
        future_start_idx = len(seg_days)
        for idx, seg_day in enumerate(seg_days):
            if seg_day > end_day:
                future_start_idx = idx
                break

        seg_infos = self._get_team_segment_infos(team)
        anchor_prev_end = team.origin if first_day_idx == 0 else seg_infos[first_day_idx - 1]['end_pos']
        anchor_next_start = seg_infos[future_start_idx]['start_pos'] if future_start_idx < len(seg_infos) else None
        pending_future_payload = self._capture_team_segment_tail(team, future_start_idx)
        future_preview_segments = [
            {
                'day': seg_infos[idx]['day'],
                'start_pos': seg_infos[idx]['start_pos'],
                'path_nodes': list(seg_infos[idx]['path_nodes']),
                'end_pos': seg_infos[idx]['end_pos'],
                'is_fly_skill': seg_infos[idx]['is_fly_skill'],
            }
            for idx in range(future_start_idx, len(seg_infos))
        ]

        # Keep only history up to yesterday. The buff snapshot must represent
        # the state at the start of the edited day, not the old route's final
        # state after later days have already consumed more charges.
        pre_edit_team_snapshots = {
            team_num: self._snapshot_team_state(t)
            for team_num, t in ((1, self.team1), (2, self.team2), (3, self.team3))
            if t is not None
        }
        pre_edit_shared_snapshot = self._snapshot_shared_state()
        pre_edit_tent_g_days = self._compute_tent_and_g_capture_days()

        self._truncate_team_history_from_segment(team, first_day_idx)
        self._derive_team_buff_state(team)
        bxz_state_before_edit = {
            'b_discount_remaining': team.b_discount_remaining,
            'b_discount_name': team.b_discount_name,
            'x_bonus_remaining': team.x_bonus_remaining,
            'x_bonus_name': team.x_bonus_name,
            'z_bonus_remaining': team.z_bonus_remaining,
            'z_bonus_name': team.z_bonus_name,
        }

        self._day_edit_context = {
            'team': team,
            'day': start_day,
            'day_start': start_day,
            'day_end': end_day,
            'anchor_prev_end': anchor_prev_end,
            'anchor_next_start': anchor_next_start,
            'pending_future_payload': pending_future_payload,
            'future_preview_segments': future_preview_segments,
            'bxz_state_before_edit': bxz_state_before_edit,
            'pre_edit_team_snapshots': pre_edit_team_snapshots,
            'pre_edit_shared_snapshot': pre_edit_shared_snapshot,
            'pre_edit_tent_g_days': pre_edit_tent_g_days,
        }
        self._segment_edit_mode = True
        self._segment_edit_targets = []
        self._segment_edit_selected_days = set()
        self._segment_edit_focus_seg_idx = None
        self.current_day = start_day

        if anchor_next_start is None:
            self._status_msg = (
                f'Days {start_day}-{end_day} segments deleted. Redraw from {anchor_prev_end}. '
                f'No future-day anchor exists, so edit mode can be closed after redraw.'
            )
        else:
            self._status_msg = (
                f'Days {start_day}-{end_day} segments deleted. Redraw from {anchor_prev_end} and reconnect to '
                f'next-day anchor {anchor_next_start}. Edit mode stays active until connected.'
            )
        return True

    def _is_active_day_edit(self):
        """Return True when active team is currently in day-segment redraw mode."""
        return (
            self._day_edit_context is not None
            and self.active_team is self._day_edit_context.get('team')
        )

    def _finalize_day_segment_edit_if_connected(self):
        """Re-attach future segments and exit edit mode when redraw reaches tomorrow anchor."""
        if not self._is_active_day_edit():
            return False

        ctx = self._day_edit_context
        team = ctx['team']
        day = ctx.get('day_start', ctx.get('day', self.current_day))
        day_end = ctx.get('day_end', day)
        expected_next_start = ctx['anchor_next_start']
        current_end = team.full_path[-1]

        if expected_next_start is not None and current_end != expected_next_start:
            self._status_msg = (
                f'Day {day} edit active. Keep drawing until current endpoint reaches '
                f'{expected_next_start}. Current endpoint: {current_end}.'
            )
            return False

        # Re-attach untouched future days, re-derive every later segment's
        # capture/revisit bookkeeping against the redrawn route (see
        # _recost_segments_from_day), then rebalance and rebuild shared state.
        self._append_team_segment_payload(team, ctx['pending_future_payload'])
        # Freshly drawn segments (for start_day..end_day) already received new,
        # strictly-increasing order ids as they were clicked in, and the
        # re-attached tail's own days are all > end_day, so day-based sort
        # already places the tail after the redraw regardless of its order
        # value. Do NOT reassign the tail's order ids here: they encode the
        # true chronological interleaving with OTHER teams' segments that
        # share those same future days, and overwriting them with a fresh,
        # globally-incrementing sequence can invert that cross-team ordering.
        # _recost_segments_from_day sorts by (day, action_order) to replay
        # history and track global state (e.g. G/g completion) - scrambling
        # a tail segment's order relative to another team's same-day segment
        # can flip which one is considered to have happened "first", which
        # then corrupts the G/g-completion (and buff) state used to price
        # both segments.
        self._normalize_missing_action_orders()
        self._ensure_team_segment_action_orders(team)
        self._recost_segments_from_day(day)
        for t in (self.team1, self.team2, self.team3):
            if t is not None:
                self._derive_team_buff_state(t)
        rebalance_overflow = self._rebalance_all_teams_from_day(day)
        self._rebuild_shared_derived_state_from_segments()
        self._rebuild_day_records()

        # The global G/g-completion timeline depends on exactly which day each
        # G/g hex was reached, so any day shift there is rejected. A Tent's
        # cost only depends on which degrade stage its day falls into (see
        # _get_terrain_food) - a shift that stays within the same stage
        # doesn't change anything any other segment's cost depends on, so
        # only a shift into a *different* stage is treated as a real change.
        tent_g_days_before = ctx.get('pre_edit_tent_g_days', {})
        tent_g_days_after = self._compute_tent_and_g_capture_days()

        def _hex_day_effectively_changed(h, old_day, new_day):
            if new_day is None or old_day == new_day:
                return new_day is None
            terrain = _terrain(*h)
            if terrain.get('name') == 'Tent':
                return _get_terrain_food(terrain, old_day) != _get_terrain_food(terrain, new_day)
            return True

        changed_hexes = sorted(
            h for h, old_day in tent_g_days_before.items()
            if _hex_day_effectively_changed(h, old_day, tent_g_days_after.get(h))
        )
        if changed_hexes:
            pre_edit_team_snapshots = ctx.get('pre_edit_team_snapshots', {})
            for team_num, t in ((1, self.team1), (2, self.team2), (3, self.team3)):
                self._restore_team_state(t, pre_edit_team_snapshots.get(team_num))
            pre_edit_shared_snapshot = ctx.get('pre_edit_shared_snapshot')
            if pre_edit_shared_snapshot is not None:
                self._restore_shared_state(pre_edit_shared_snapshot)

            self._segment_edit_mode = False
            self._segment_edit_targets = []
            self._segment_edit_selected_days = set()
            self._segment_edit_focus_seg_idx = None
            self._day_edit_context = None
            example = changed_hexes[0]
            self._status_msg = (
                f'Edit rejected: it would move Tent/G-g hex {example} '
                f'(and {len(changed_hexes) - 1} other hex(es)) to a different day. '
                f'Reverted to the state before this edit.'
            ) if len(changed_hexes) > 1 else (
                f'Edit rejected: it would move Tent/G-g hex {example} to a different day. '
                f'Reverted to the state before this edit.'
            )
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                root.lift()
                root.attributes('-topmost', True)
                root.update()
                messagebox.showerror(
                    'Edit Rejected',
                    (
                        f'This route edit would change the day on which {len(changed_hexes)} '
                        f'Tent/G-g hex(es) are captured (e.g. {example}).\n\n'
                        'Route edits are not allowed to move Tent or G/g capture days, since '
                        'other segments\' costs and schedules depend on them.\n\n'
                        'The edit has been reverted to the state before it started.'
                    ),
                )
                root.destroy()
            except Exception:
                pass
            return False

        self._segment_edit_mode = False
        self._segment_edit_targets = []
        self._segment_edit_selected_days = set()
        self._segment_edit_focus_seg_idx = None
        self._day_edit_context = None
        self.current_day = day
        self._status_msg = f'Days {day}-{day_end} edit completed. Future segments reconnected successfully.'
        if rebalance_overflow:
            self._status_msg += (
                f' WARNING: rebalancing could not fit all remaining moves within '
                f'Day {TOTAL_DAYS} (the season length) - some segments are piled up on the last day.'
            )
        return True

    def _toggle_segment_edit_mode(self):
        """Toggle UI mode for selecting a segment on the current team/day."""
        if self._enclosure_mode:
            self._status_msg = '请先退出圈地模式，再使用路径编辑。'
            self._draw()
            return

        # In active day-edit redraw mode, user must reconnect to tomorrow anchor first.
        if self._segment_edit_mode and self._day_edit_context is not None:
            ctx = self._day_edit_context
            expected_next_start = ctx.get('anchor_next_start')
            current_end = ctx['team'].full_path[-1]
            if expected_next_start is not None and current_end != expected_next_start:
                self._status_msg = (
                    f'Cannot exit edit mode yet. Redraw Day {ctx["day"]} until endpoint reaches '
                    f'{expected_next_start}.'
                )
                self._draw()
                return

        # While in selection mode (before truncation), second click applies selected day-range.
        if self._segment_edit_mode and self._day_edit_context is None:
            if self._segment_edit_selected_days:
                start_day = min(self._segment_edit_selected_days)
                end_day = max(self._segment_edit_selected_days)

                confirmed = True
                try:
                    import tkinter as tk
                    from tkinter import messagebox
                    root = tk.Tk()
                    root.withdraw()
                    root.lift()
                    root.attributes('-topmost', True)
                    root.update()
                    confirmed = messagebox.askyesno(
                        'Apply Edit Range',
                        (
                            f'Delete active team segments for Day {start_day} to Day {end_day} and redraw?\n\n'
                            f'You must reconnect to the next-day anchor before edit mode can exit.'
                        )
                    )
                    root.destroy()
                except Exception:
                    confirmed = True

                if confirmed:
                    self._begin_day_segment_edit_range(self.active_team, start_day, end_day)
                else:
                    self._segment_edit_mode = False
                    self._segment_edit_targets = []
                    self._segment_edit_selected_days = set()
                    self._segment_edit_focus_seg_idx = None
                    self._status_msg = 'Segment range edit cancelled. Edit mode off.'
                self._draw()
                return

        self._segment_edit_mode = not self._segment_edit_mode
        self._clear_hover_preview()

        if self._segment_edit_mode:
            if self.active_team is None:
                self._segment_edit_mode = False
                self._status_msg = 'No active team selected for segment edit mode.'
            else:
                # Snap the viewed day to the active team's own last action day
                # before looking for its segments - otherwise opening Segment
                # Edit while viewing a day this team never acted on (e.g. it's
                # been idle while another team kept moving on later days)
                # always finds zero segments and refuses to open, even though
                # the team clearly has editable history on its own last day.
                team_last_day = self._get_active_team_last_movement_day()
                if team_last_day is not None and team_last_day != self.current_day:
                    self.current_day = team_last_day
                    self._sync_current_food_for_view_day()

                segs_today = [s for s in self._get_team_segment_infos(self.active_team) if s['day'] == self.current_day]
                if not segs_today:
                    self._segment_edit_mode = False
                    self._status_msg = f'No segments for active team on Day {self.current_day}.'
                else:
                    team_num = 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)
                    self._segment_edit_selected_days = set()
                    self._segment_edit_focus_seg_idx = None
                    self._status_msg = (
                        f'Segment edit mode: Team {team_num}. Click segment paths on any days to select range, '
                        f'then click EditSeg again to apply.'
                    )
        else:
            self._segment_edit_targets = []
            self._segment_edit_selected_days = set()
            self._segment_edit_focus_seg_idx = None
            self._status_msg = 'Segment edit mode off.'

        self._draw()

    def _all_tent_and_g_complete_day(self):
        required = set(self.all_g_lands)
        for ir in range(ROWS):
            for ic in range(COLS):
                if _terrain(ir, ic).get('name') == 'Tent':
                    required.add((ir, ic))
        capture_days = self._compute_tent_and_g_capture_days()
        if not required or any(hex_pos not in capture_days for hex_pos in required):
            return None
        return max(capture_days[hex_pos] for hex_pos in required)

    def _can_start_enclosure_mode(self):
        complete_day = self._all_tent_and_g_complete_day()
        if complete_day is None:
            return False, 'T and G/g lands are not all captured yet.'
        if self.current_day < complete_day + 1:
            return False, f'圈地模式需在 T 和 G/g 全部拿满后的第二天开启（最早 Day {complete_day + 1}）。'
        return True, complete_day

    def _toggle_enclosure_mode(self):
        if self._enclosure_mode:
            self._finish_enclosure_mode()
            return
        if self._segment_edit_mode:
            self._status_msg = '请先退出路径编辑，再开启圈地模式。'
            self._draw()
            return
        can_start, detail = self._can_start_enclosure_mode()
        if not can_start:
            self._status_msg = detail
            self._draw()
            return
        self._enclosure_mode = True
        self._enclosure_start_day = self.current_day
        self._status_msg = f'圈地模式已开启：可自由画路线，退出时将从 Day {self._enclosure_start_day} 自动分配到未来天数。'
        self._draw()

    def _finish_enclosure_mode(self):
        start_day = self._enclosure_start_day or self.current_day
        self._recost_segments_from_day(start_day)
        for team in (self.team1, self.team2, self.team3):
            if team is not None:
                self._derive_team_buff_state(team)
        self._rebalance_all_teams_from_day(start_day, preserve_future_days=False)
        self._rebuild_shared_derived_state_from_segments()
        self._rebuild_day_records()

        has_step_overflow = any(
            self._team_has_step_overflow_from(team, max(start_day, team.created_day))
            for team in (self.team1, self.team2, self.team3)
            if team is not None
        )
        has_food_deficit = self._has_global_food_deficit_from_segments()
        self._enclosure_mode = False
        self._enclosure_start_day = None
        if has_step_overflow or has_food_deficit:
            self._status_msg = '圈地模式已退出，但未来天数仍有步数或粮草超支，请减少路线或继续调整。'
        else:
            self._status_msg = f'圈地模式已退出：路线已从 Day {start_day} 起自动分配到未来天数。'
        self._auto_save_game()
        self._draw()

    def _find_segment_edit_target(self, xdata, ydata):
        """Return clicked edit target in map coordinates, if any."""
        if not self._segment_edit_targets:
            return None

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        hit_radius = max(min(cur_xlim[1] - cur_xlim[0], cur_ylim[1] - cur_ylim[0]) * 0.018, 8.0)
        hit_radius2 = hit_radius * hit_radius
        best = None
        best_d2 = None
        for target in self._segment_edit_targets:
            dx = xdata - target['x']
            dy = ydata - target['y']
            d2 = dx * dx + dy * dy
            if d2 <= hit_radius2 and (best_d2 is None or d2 < best_d2):
                best = target
                best_d2 = d2
        return best

    def _handle_segment_edit_click(self, xdata, ydata):
        """Handle map click while in segment edit mode."""
        target = self._find_segment_edit_target(xdata, ydata)
        if target is None:
            self._status_msg = 'Segment edit mode: click a highlighted segment marker.'
            self._draw()
            return

        day = target['day']
        if day in self._segment_edit_selected_days:
            self._segment_edit_selected_days.remove(day)
        else:
            self._segment_edit_selected_days.add(day)

        self._segment_edit_focus_seg_idx = target['seg_idx']
        self._center_view_on_active_team_day(day)

        if self._segment_edit_selected_days:
            start_day = min(self._segment_edit_selected_days)
            end_day = max(self._segment_edit_selected_days)
            self._status_msg = (
                f'Selected day range: {start_day}-{end_day}. Click EditSeg again to delete this range and redraw.'
            )
        else:
            self._status_msg = 'No day selected. Click segment paths on any days to select edit range.'

        self._draw()

    def _recost_segments_from_day(self, start_day, settle_exploration=False):
        """Re-derive per-hex bookkeeping for every segment on/after start_day.

        A segment's records (which hexes were fresh captures vs. 10-food revisits,
        and the resulting food/score/steps) are computed when the move is made
        and stored. A route edit deletes and redraws earlier segments, so the
        untouched later segments - and other teams' segments - can be stale:
        a hex the redraw now captures is still recorded as a fresh capture later
        (scored twice), and a hex the redraw no longer captures stays recorded as
        a revisit where a later day actually walked onto it first (never scored).

        Segments are replayed in chronological order (each team's own array
        order, interleaved across teams by day then action order). For a
        replayed segment, entries whose kind still matches keep their stored
        cost (preserving any B/Z buff applied at the time); entries that change
        kind are re-priced with the base rules.
        """
        import types
        teams = [t for t in (self.team1, self.team2, self.team3) if t is not None]
        if not teams:
            return
        self._normalize_missing_action_orders()
        for team in teams:
            self._ensure_team_segment_path_nodes(team)

        start_day = int(start_day)
        captured = {self.team1.origin}
        visited_g = {self.team1.origin} & self.all_g_lands
        # Probe / deferred-fly hexes nobody has paid for yet. Tracked globally:
        # whichever team is standing on one when it leaves settles it - its own
        # probe, a deferred fly landing, or a team created on top of it.
        unsettled = set()
        pos_by_team = {team: team.origin for team in teams}
        no_buff = types.SimpleNamespace(
            b_discount_remaining=0,
            b_discount_name=None,
            x_bonus_remaining=0,
            x_bonus_name=None,
            z_bonus_remaining=0,
            z_bonus_name=None,
        )
        recost_buffs = {
            team: types.SimpleNamespace(**self._derive_team_buff_state_before_day(team, start_day))
            for team in teams
        }
        if self._day_edit_context is not None:
            edit_team = self._day_edit_context.get('team')
            saved_bxz = self._day_edit_context.get('bxz_state_before_edit', {})
            if edit_team in recost_buffs and saved_bxz:
                recost_buffs[edit_team] = types.SimpleNamespace(**saved_bxz)
        active_recost_buff = no_buff

        def all_g():
            return len(visited_g) == len(self.all_g_lands) and len(self.all_g_lands) > 0

        def note(hex_pos):
            if hex_pos in self.all_g_lands:
                visited_g.add(hex_pos)

        def fresh_capture(hex_pos, seg_day, is_fly):
            t = _terrain(*hex_pos)
            challenge = _apply_challenge_discounts(
                _get_terrain_food(t, seg_day), active_recost_buff, all_g(), is_tent=t.get('name') == 'Tent')
            reward = t['award']
            if active_recost_buff.z_bonus_remaining > 0:
                reward = _apply_z_bonus(reward, active_recost_buff)
            if is_fly:
                movement = 0
                step = t.get('step', 1)
                steps = step if step < 0 else 0
            else:
                movement = _apply_g_reduction(50, all_g())
                steps = t.get('step', 1)
            if active_recost_buff.b_discount_remaining > 0:
                active_recost_buff.b_discount_remaining -= 1
            if active_recost_buff.z_bonus_remaining > 0:
                active_recost_buff.z_bonus_remaining -= 1
            if active_recost_buff.x_bonus_remaining > 0 and not is_fly and steps > 0:
                steps = 0
                active_recost_buff.x_bonus_remaining -= 1
            return (movement + challenge, reward, steps)

        def fresh_settle(hex_pos, seg_day):
            t = _terrain(*hex_pos)
            challenge = _apply_challenge_discounts(
                _get_terrain_food(t, seg_day), active_recost_buff, all_g(), is_tent=t.get('name') == 'Tent')
            reward = t['award']
            if active_recost_buff.z_bonus_remaining > 0:
                reward = _apply_z_bonus(reward, active_recost_buff)
            if active_recost_buff.b_discount_remaining > 0:
                active_recost_buff.b_discount_remaining -= 1
            if active_recost_buff.z_bonus_remaining > 0:
                active_recost_buff.z_bonus_remaining -= 1
            return (challenge, reward, 1)

        def is_portal(hex_pos):
            token = RAW_MAP[hex_pos[0]][hex_pos[1]] if 0 <= hex_pos[0] < ROWS and 0 <= hex_pos[1] < COLS else ''
            return bool(re.fullmatch(r'P\d+', token))

        # Chronological merge: team-internal order is authoritative; across teams
        # interleave by (day, action order).
        idx = {team: 0 for team in teams}
        order = []
        while True:
            best = None
            for team in teams:
                i = idx[team]
                if i >= len(team._seg_days):
                    continue
                key = (int(team._seg_days[i]), team._seg_action_orders[i] if i < len(team._seg_action_orders) and isinstance(team._seg_action_orders[i], int) else 0)
                if best is None or key < best[0]:
                    best = (key, team)
            if best is None:
                break
            order.append((best[1], idx[best[1]]))
            idx[best[1]] += 1

        for team, i in order:
            seg_day = int(team._seg_days[i])
            nodes = [tuple(h) for h in team._seg_path_nodes[i]]
            dep = pos_by_team[team]
            end_pos = tuple(team._seg_end_positions[i]) if i < len(team._seg_end_positions) and team._seg_end_positions[i] is not None else (nodes[-1] if nodes else dep)

            if seg_day < start_day:
                # Untouched history: apply its stored effects to the running state.
                for h in team._seg_new_hexes[i]:
                    captured.add(tuple(h)); note(tuple(h))
                for h in team._seg_exploration_hexes[i]:
                    unsettled.add(tuple(h)); note(tuple(h))
                unsettled -= captured
                pos_by_team[team] = end_pos
                continue

            active_recost_buff = recost_buffs[team]

            is_fly = bool(team._seg_is_fly_skill[i]) if i < len(team._seg_is_fly_skill) else False
            stored_costs = list(team._seg_hex_costs[i]) if i < len(team._seg_hex_costs) else []
            stored_new = {tuple(h) for h in team._seg_new_hexes[i]}
            stored_jump = {tuple(h) for h in team._seg_jumps[i]} if i < len(team._seg_jumps) else set()
            stored_expl = {tuple(h) for h in team._seg_exploration_hexes[i]}
            prefix = max(0, len(stored_costs) - len(nodes))

            entries = []      # (food, award, steps)
            new_hexes, jumps, explores, actions = [], [], [], []
            fly_delta = -1 if is_fly else 0

            # Leaving (or settling in place on) a still-unsettled exploration hex
            # settles it now; if someone captured it in the meantime, nothing to pay.
            settle_hex = dep if (dep in unsettled or (not nodes and dep in stored_new)) else None
            if settle_hex is not None and settle_hex not in captured:
                entry = stored_costs[0] if (prefix > 0 and dep in stored_new) else fresh_settle(dep, seg_day)
                entries.append(tuple(entry))
                new_hexes.append(dep); actions.append(('new', dep))
                captured.add(dep); note(dep)
                if _terrain(*dep).get('name') == 'bigBoss':
                    fly_delta += 1
            elif not nodes and dep in stored_new:
                entries.append((0, 0, 0))  # in-place settle of a hex already taken by someone else
            unsettled.discard(dep)

            if is_fly and nodes:
                actions.insert(0, ('fly', nodes[-1]))

            seen_in_seg = set()
            for k, h in enumerate(nodes):
                stored = tuple(stored_costs[prefix + k]) if prefix + k < len(stored_costs) else None
                if h in stored_expl and not settle_exploration:
                    kind = 'explore'
                elif h in stored_expl:
                    kind = 'new'
                elif h in stored_new:
                    kind = 'new'
                elif h in stored_jump:
                    kind = 'jump'
                else:
                    kind = 'new' if stored and stored[1] else 'jump'

                if is_portal(h):
                    # Portal bookkeeping is handled by the teleport logic; keep as stored.
                    entries.append(stored or (0, 0, 0))
                    if kind == 'new':
                        new_hexes.append(h); actions.append(('new', h)); captured.add(h); note(h)
                    elif kind == 'jump':
                        jumps.append(h); actions.append(('jump', h))
                    seen_in_seg.add(h)
                    continue

                taken = h in captured or h in seen_in_seg
                if kind == 'explore' and not taken:
                    # A deferred landing charges nothing up front - walking waives
                    # nothing (flat movement fee), but flying waives it entirely.
                    fresh_explore_cost = 0 if is_fly else _apply_g_reduction(50, all_g())
                    entries.append(stored or (fresh_explore_cost, 0, 0))
                    explores.append(h); unsettled.add(h); note(h)
                elif taken:
                    entries.append(stored if (kind == 'jump' and stored) else (_apply_g_reduction(10, all_g()), 0, 0))
                    jumps.append(h); actions.append(('jump', h))
                else:
                    # KNOWN LIMITATION: an unchanged classification keeps its
                    # stored cost, deliberately, so a price that had a B/Z buff
                    # folded into it at the time isn't silently recomputed
                    # without that buff context. The side effect is that if an
                    # edit moves *which day* the last G/g land is captured on,
                    # hexes captured after the shifted completion point keep
                    # prices derived from the old G/g-completion timing (i.e.
                    # the wrong 20% discount state). Fixing it properly needs a
                    # way to re-price only the G/g component while preserving
                    # the recorded B/Z context - left alone until a concrete
                    # case needs it.
                    entry = stored if (kind == 'new' and stored) else fresh_capture(h, seg_day, is_fly)
                    entries.append(tuple(entry))
                    new_hexes.append(h); actions.append(('new', h)); captured.add(h); note(h)
                    if _terrain(*h).get('name') == 'bigBoss':
                        fly_delta += 1
                seen_in_seg.add(h)

            team._seg_hex_costs[i] = entries
            team._seg_foods[i] = sum(e[0] for e in entries)
            team._seg_awards[i] = sum(e[1] for e in entries)
            team._seg_steps[i] = sum(e[2] for e in entries)
            team._seg_new_hexes[i] = new_hexes
            team._seg_jumps[i] = jumps
            team._seg_exploration_hexes[i] = explores
            team._seg_action_sequence[i] = actions
            if i < len(team._seg_fly_skill_deltas):
                team._seg_fly_skill_deltas[i] = fly_delta
            pos_by_team[team] = end_pos

    def _set_team2_start(self):
        """Create Team 2 at the current active team's position."""
        if self.team2 is not None:
            self._status_msg = 'Team 2 already exists. Click "Reset Path" to create a new team.'
        else:
            # Create Team 2 at the active team's current position
            old_team = self.active_team
            current_pos = old_team.full_path[-1]
            self.team2 = Team(current_pos, created_day=self.current_day)
            self._init_team_created_at(old_team, self.team2, current_pos)
            self.active_team = self.team2
            self._update_switch_button_color()
            # Hide the legacy Set Team 2 button if present.
            self._set_legacy_set_team_buttons_visibility()
            self._status_msg = 'Team 2 created at active team position. Building Team 2 path...'
            self._rebuild_day_records()  # Rebuild day records to reflect new team
        self._draw()

    def _set_team3_start(self):
        """Create Team 3 at the current active team's position."""
        if self.team3 is not None:
            self._status_msg = 'Team 3 already exists. Click "Reset Path" to create a new team.'
        else:
            # Create Team 3 at the active team's current position
            old_team = self.active_team
            current_pos = old_team.full_path[-1]
            self.team3 = Team(current_pos, created_day=self.current_day)
            self._init_team_created_at(old_team, self.team3, current_pos)
            self.active_team = self.team3
            self._update_switch_button_color()
            # Hide the legacy Set Team 3 button if present.
            self._set_legacy_set_team_buttons_visibility()
            self._status_msg = 'Team 3 created at active team position. Building Team 3 path...'
            self._rebuild_day_records()  # Rebuild day records to reflect new team
        self._draw()

    def _activate_fly_skill(self):
        """Activate fly skill mode - allows team to teleport to any hex with waived movement food."""
        if not self._fly_mode:
            if self.fly_skill_limit <= 0:
                self._status_msg = 'No fly skills remaining!'
                self._draw()
                return
            self._fly_mode = True
            self._status_msg = '选择飞雷神目的地 (Select Flying Thunder God destination) - Press Fly again to cancel'
            
            # Stop any existing timer
            if self._fly_button_timer is not None:
                self._fly_button_timer.stop()
            
            # Start flashing animation with border
            self._fly_button_flash_state = True
            self._set_fly_button_border('#FF1493', 2.5)  # Bright pink border, thick
            self._fly_button_timer = self.fig.canvas.new_timer()
            self._fly_button_timer.interval = 300  # Flash every 300ms
            self._fly_button_timer.single_shot = False
            self._fly_button_timer.callbacks.append((self._flash_fly_button, (), {}))
            self._fly_button_timer.start()
        else:
            self._fly_mode = False
            self._status_msg = 'Fly skill deactivated.'
            
            # Stop flashing
            if self._fly_button_timer is not None:
                self._fly_button_timer.stop()
                self._fly_button_timer = None
            
            self._set_fly_button_border('#777777', 0.8)  # Reset to default border
        self._draw()

    def _is_valid_fly_destination(self, destination):
        """Return whether a target borders the occupied region connected to start."""
        team = self.active_team
        if team is None or destination in self.all_visited_hexes:
            return False

        unsettled_exploration = {
            hex_pos
            for other_team in (self.team1, self.team2, self.team3)
            if other_team is not None
            for hex_pos in other_team.free_exploration_hexes
            if not self._is_hex_settled(hex_pos)
        }
        occupied_hexes = self.all_visited_hexes - unsettled_exploration
        start = tuple(team.full_path[-1])
        connected_occupied = {start} if start in occupied_hexes else set()
        frontier = [start]
        seen = {start}
        while frontier:
            current = frontier.pop()
            for neighbor in _neighbors(*current):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                if neighbor in occupied_hexes:
                    connected_occupied.add(neighbor)
                    frontier.append(neighbor)

        if not connected_occupied:
            return False
        return any(
            destination in _neighbors(*occupied_hex)
            for occupied_hex in connected_occupied
        )

    def _flash_fly_button(self):
        """Toggle fly button border for flashing animation."""
        if not self._fly_mode:
            # If fly mode is no longer active, stop flashing
            if self._fly_button_timer is not None:
                self._fly_button_timer.stop()
                self._fly_button_timer = None
            return
        
        self._fly_button_flash_state = not self._fly_button_flash_state
        if self._fly_button_flash_state:
            self._set_fly_button_border('#FF1493', 2.5)  # Bright pink, thick
        else:
            self._set_fly_button_border('#FFB6C1', 1.5)  # Light pink, medium
        self.fig.canvas.draw_idle()

    def _animate_view_to(self, target_xlim, target_ylim, duration_ms=90, steps=18):
        """Animate axis limits to target quickly for smooth map panning."""
        try:
            if self._view_anim_timer is not None:
                self._view_anim_timer.stop()
                self._view_anim_timer = None
        except Exception:
            self._view_anim_timer = None

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        sx0, sx1 = float(cur_xlim[0]), float(cur_xlim[1])
        sy0, sy1 = float(cur_ylim[0]), float(cur_ylim[1])
        tx0, tx1 = float(target_xlim[0]), float(target_xlim[1])
        ty0, ty1 = float(target_ylim[0]), float(target_ylim[1])

        if (abs(sx0 - tx0) < 1e-9 and abs(sx1 - tx1) < 1e-9 and
                abs(sy0 - ty0) < 1e-9 and abs(sy1 - ty1) < 1e-9):
            return

        # Preserve view across redraws while animating.
        self._has_zoomed = True

        steps = max(1, int(steps))
        interval = max(10, int(duration_ms / steps))
        state = {'i': 0}

        def _tick():
            state['i'] += 1
            t = state['i'] / steps
            # Ease-out cubic for fast but smooth landing.
            u = 1.0 - (1.0 - t) ** 3
            nx0 = sx0 + (tx0 - sx0) * u
            nx1 = sx1 + (tx1 - sx1) * u
            ny0 = sy0 + (ty0 - sy0) * u
            ny1 = sy1 + (ty1 - sy1) * u
            self.ax.set_xlim(nx0, nx1)
            self.ax.set_ylim(ny0, ny1)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw_idle()

            if state['i'] >= steps and self._view_anim_timer is not None:
                self._view_anim_timer.stop()
                self._view_anim_timer = None

        self._view_anim_timer = self.fig.canvas.new_timer(interval=interval)
        self._view_anim_timer.single_shot = False
        self._view_anim_timer.callbacks.append((_tick, (), {}))
        self._view_anim_timer.start()

    def _center_view_on_active_team(self):
        """Center the view on the active team's current position while keeping current zoom level."""
        if self.active_team is None:
            return
        
        current_pos = self.active_team.full_path[-1]
        cx, cy = _center(current_pos[0], current_pos[1])
        
        # Scale coordinates to match axis coordinate space
        cx_scaled = cx * X_SCALE
        cy_scaled = cy * Y_SCALE
        
        # Get current view size
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        view_width = cur_xlim[1] - cur_xlim[0]
        view_height = cur_ylim[1] - cur_ylim[0]

        # Instant move (no animation)
        self.ax.set_xlim(cx_scaled - view_width / 2, cx_scaled + view_width / 2)
        self.ax.set_ylim(cy_scaled - view_height / 2, cy_scaled + view_height / 2)

    def _center_view_on_scaled_point(self, cx_scaled, cy_scaled):
        """Center map view on a scaled point and preserve current zoom span."""
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        view_width = cur_xlim[1] - cur_xlim[0]
        view_height = cur_ylim[1] - cur_ylim[0]

        # Preserve centered view across redraws.
        self._has_zoomed = True
        self.ax.set_xlim(cx_scaled - view_width / 2, cx_scaled + view_width / 2)
        self.ax.set_ylim(cy_scaled - view_height / 2, cy_scaled + view_height / 2)

    def _center_view_on_active_team_day(self, day, move_if_empty=True):
        """Center view on active-team segment footprint for a selected day.

        If `move_if_empty` is False and the team has no segments on `day`,
        the view is left untouched instead of falling back to the team's
        overall last known position.
        """
        if self.active_team is None:
            return

        segs = [s for s in self._get_team_segment_infos(self.active_team) if s['day'] == day]
        if not segs:
            if move_if_empty:
                self._center_view_on_active_team()
            return

        xs = []
        ys = []
        for seg in segs:
            sx, sy = _center(*seg['start_pos'])
            ex, ey = _center(*seg['end_pos'])
            xs.extend([sx, ex])
            ys.extend([sy, ey])

            for node in seg.get('path_nodes', []):
                nx, ny = _center(*node)
                xs.append(nx)
                ys.append(ny)

        if not xs or not ys:
            if move_if_empty:
                self._center_view_on_active_team()
            return

        cx_scaled = (sum(xs) / len(xs)) * X_SCALE
        cy_scaled = (sum(ys) / len(ys)) * Y_SCALE
        self._center_view_on_scaled_point(cx_scaled, cy_scaled)

    def _start_blinking_animation(self):
        """Start the blinking animation for the active team's marker."""
        pass

    def _stop_blinking_animation(self):
        """Stop the blinking animation."""
        pass

    def _update_blinking(self):
        """Update blinking animation phase and redraw."""
        pass

    def _is_marker_visible(self):
        """Check if active team's marker should be visible (for blinking effect).
        
        Uses a sine wave to determine visibility: visible when sin(phase) > 0,
        which creates a 50/50 on/off blinking effect.
        """
        return True

    def _switch_team(self):
        """Cycle through active teams and center view on new team while keeping zoom level."""
        if self.team2 is None and self.team3 is None:
            self._status_msg = 'Set up Team 2 and Team 3 first.'
        else:
            teams = [self.team1]
            if self.team2:
                teams.append(self.team2)
            if self.team3:
                teams.append(self.team3)
            idx = teams.index(self.active_team)
            self.active_team = teams[(idx + 1) % len(teams)]
            team_num = 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)
            self._status_msg = f'Active team: Team {team_num}'
            self._update_switch_button_color()
            self._update_fly_button_state()  # Update fly button state when switching teams
            
            # Center view on the active team's current position while keeping current zoom
            self._center_view_on_active_team()
        
        self._draw()

    def _confirm_free_exploration_step(self, hex_pos):
        """Ask before a probe step ("蹭步"), which the team can only make because
        it has run out of steps for the day. Returns True to go ahead."""
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            confirmed = messagebox.askyesno(
                '蹭步确认',
                f'队伍今日步数已用完，下一步到 ({hex_pos[0]}, {hex_pos[1]}) 是蹭步。\n\n'
                f'是否继续？'
            )
            root.destroy()
        except Exception:
            confirmed = True
        return confirmed

    def _confirm_settle_current_exploration(self):
        """Ask whether to occupy the active team's current probe hex."""
        team = self.active_team
        if team is None or not team.full_path:
            return False

        current_pos = team.full_path[-1]
        if current_pos not in team.free_exploration_hexes or self._is_hex_settled(current_pos):
            return False

        steps_available = self._get_team_steps_for_day(team, self.current_day)
        if steps_available <= 0:
            return False

        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            confirmed = messagebox.askyesno(
                '占领试探地块',
                f'是否占领当前试探地块 ({current_pos[0]}, {current_pos[1]})？\n\n'
                f'需要 1 step，并结算该地块的奖励与消耗。'
            )
            root.destroy()
        except Exception:
            confirmed = True

        if not confirmed:
            return True

        terrain = _terrain(*current_pos)
        raw_challenge_food = _get_terrain_food(terrain, self.current_day)
        challenge_food = _apply_challenge_discounts(
            raw_challenge_food,
            team,
            self._are_all_g_lands_visited(),
            is_tent=terrain.get('name') == 'Tent',
        )
        b_discount_was_active = raw_challenge_food >= 0 and team.b_discount_remaining > 0
        reward = terrain.get('award', 0)
        z_bonus_was_active = team.z_bonus_remaining > 0
        if z_bonus_was_active:
            reward = _apply_z_bonus(reward, team)

        if self.current_food < challenge_food:
            self._status_msg = self._build_not_enough_food_message(
                challenge_food, steps_available
            )
            self._draw()
            return True

        if b_discount_was_active:
            team.b_discount_remaining -= 1
        if z_bonus_was_active:
            team.z_bonus_remaining -= 1

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
        team._seg_action_sequence.append([('new', current_pos)])
        self._append_segment_action_order(team)
        team._seg_hex_costs.append([(challenge_food, reward, 1)])
        team._seg_is_fly_skill.append(False)
        # Capturing a BigBoss grants +1 fly skill, whether it is captured by
        # walking onto it or settled in place here.
        boss_delta = 1 if terrain.get('name') == 'bigBoss' else 0
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
        self._status_msg = (
            f'已占领试探地块 ({current_pos[0]},{current_pos[1]})。'
            f'消耗 1 step，获得奖励 {reward}。'
        )
        if boss_delta:
            self._status_msg += f' ⭐ Reached BigBoss! Fly skill +1 (now {self.fly_skill_limit})'
        self._status_msg += self._activate_land_buffs(current_pos)
        self._rebuild_day_records()
        self._auto_save_game()
        self._draw()
        return True

    def _derive_team_buff_state(self, team):
        """Recompute a team's X/B/Z buff counters by replaying its segment
        history, mirroring how moves consume and (re)activate them: a segment's
        fresh captures consume one B/Z charge each, captures whose step cost
        was waived consume one X charge each (not on fly moves), and capturing
        an X/B/Z land - as the segment's final hex or as a hex settled on
        departure - resets that buff to its full count."""
        state = self._derive_team_buff_state_before_day(team, None)
        team.x_bonus_remaining, team.x_bonus_name = state['x_bonus_remaining'], state['x_bonus_name']
        team.b_discount_remaining, team.b_discount_name = state['b_discount_remaining'], state['b_discount_name']
        team.z_bonus_remaining, team.z_bonus_name = state['z_bonus_remaining'], state['z_bonus_name']

    def _derive_team_buff_state_before_day(self, team, day):
        """Return a team's B/X/Z counters after replaying segments before day."""
        x = b = z = 0
        xn = bn = zn = None
        for i in range(len(team._seg_days)):
            if day is not None and int(team._seg_days[i]) >= int(day):
                continue
            new_hexes = [tuple(h) for h in team._seg_new_hexes[i]]
            nodes = [tuple(h) for h in team._seg_path_nodes[i]] if i < len(team._seg_path_nodes) else []
            costs = team._seg_hex_costs[i] if i < len(team._seg_hex_costs) else []
            is_fly = bool(team._seg_is_fly_skill[i]) if i < len(team._seg_is_fly_skill) else False
            prefix = max(0, len(costs) - len(nodes))
            if x > 0 and not is_fly:
                waived = sum(
                    1 for k, h in enumerate(nodes)
                    if h in new_hexes and prefix + k < len(costs) and len(costs[prefix + k]) >= 3
                    and costs[prefix + k][2] == 0 and _terrain(*h).get('step', 1) > 0
                )
                x = max(0, x - waived)
            if new_hexes:
                if b > 0:
                    b = max(0, b - len(new_hexes))
                if z > 0:
                    z = max(0, z - len(new_hexes))
            candidates = []
            if prefix > 0 and new_hexes and new_hexes[0] not in nodes:
                candidates.append(new_hexes[0])          # settled on departure
            end = team._seg_end_positions[i] if i < len(team._seg_end_positions) and team._seg_end_positions[i] is not None else (nodes[-1] if nodes else None)
            if end is not None and tuple(end) in new_hexes:
                candidates.append(tuple(end))            # captured as the final hex
            elif not nodes and new_hexes:
                candidates.append(new_hexes[0])          # in-place settle
            for h in candidates:
                name = RAW_MAP[h[0]][h[1]]
                if name in X_BONUS_MAP:
                    x, xn = X_BONUS_MAP[name], name
                elif name in B_DISCOUNT_MAP:
                    b, bn = B_DISCOUNT_MAP[name], name
                elif name in Z_BONUS_MAP:
                    z, zn = Z_BONUS_MAP[name], name
        return {
            'b_discount_remaining': b,
            'b_discount_name': bn,
            'x_bonus_remaining': x,
            'x_bonus_name': xn,
            'z_bonus_remaining': z,
            'z_bonus_name': zn,
        }

    def _activate_land_buffs(self, hex_pos):
        """Start the X/B/Z buff granted by capturing hex_pos (if it is one of
        those lands); returns a status-message suffix."""
        name = RAW_MAP[hex_pos[0]][hex_pos[1]]
        team = self.active_team
        if name in X_BONUS_MAP:
            team.x_bonus_remaining = X_BONUS_MAP[name]
            team.x_bonus_name = name
            return f' +{team.x_bonus_remaining} free movements ({name})!'
        if name in B_DISCOUNT_MAP:
            team.b_discount_remaining = B_DISCOUNT_MAP[name]
            team.b_discount_name = name
            rate = _TERRAIN_DB.get(name, {}).get('food_discount_rate', 0.4)
            return f' {int(rate * 100)}% food discount active for next {team.b_discount_remaining} movements!'
        if name in Z_BONUS_MAP:
            team.z_bonus_remaining = Z_BONUS_MAP[name]
            team.z_bonus_name = name
            rate = _TERRAIN_DB.get(name, {}).get('reward_bonus_rate', 1.4)
            return f' {int((rate - 1) * 100)}% reward bonus active for next {team.z_bonus_remaining} movements!'
        return ''

    def _are_all_g_lands_visited(self):
        """Check if all G/g lands have been visited by any team."""
        return len(self.visited_g_lands) == len(self.all_g_lands) and len(self.all_g_lands) > 0

    def _recompute_visited_g_lands_from_segments(self):
        """Re-derive visited_g_lands from canonical segment history.

        During play visited_g_lands is only ever added to, but it gates the
        global 20% food discount - so undoing a G/g capture would otherwise
        leave that land counted forever and keep the discount switched on
        with fewer than all G/g lands actually taken (every hex drawn after
        that then gets stored at a wrongly discounted price).

        Mirrors the rules _rebuild_shared_derived_state_from_segments() uses:
        team origins, captured hexes, still-unsettled probe/deferred-fly
        hexes, and both ends of a portal segment.
        """
        visited = set()
        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue

            origin = tuple(team.origin)
            if origin in self.all_g_lands:
                visited.add(origin)

            for field_name in ('_seg_new_hexes', '_seg_exploration_hexes'):
                for seg in getattr(team, field_name, None) or []:
                    for hex_pos in seg:
                        hex_pos = tuple(hex_pos)
                        if hex_pos in self.all_g_lands:
                            visited.add(hex_pos)

            for seg_idx, seg_nodes in enumerate(team._seg_path_nodes):
                if not seg_nodes:
                    continue
                raw_end_pos = tuple(seg_nodes[-1])
                token = (
                    RAW_MAP[raw_end_pos[0]][raw_end_pos[1]]
                    if 0 <= raw_end_pos[0] < ROWS and 0 <= raw_end_pos[1] < COLS
                    else ''
                )
                if not re.fullmatch(r'P\d+', token):
                    continue
                # A portal segment marks both the entry hex and the hex it
                # teleported out to as taken.
                if raw_end_pos in self.all_g_lands:
                    visited.add(raw_end_pos)
                if seg_idx < len(team._seg_end_positions) and team._seg_end_positions[seg_idx] is not None:
                    end_pos = tuple(team._seg_end_positions[seg_idx])
                    if end_pos in self.all_g_lands:
                        visited.add(end_pos)

        self.visited_g_lands = visited

    def _find_all_g_lands_complete_day(self):
        """Return the day the last G/g land was captured, or None if not all captured."""
        if not self.all_g_lands:
            return None

        events = []
        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue
            for seg_idx, seg_day in enumerate(team._seg_days):
                order = (
                    team._seg_action_orders[seg_idx]
                    if team._seg_action_orders and seg_idx < len(team._seg_action_orders)
                    else 0
                ) or 0
                new_hexes = team._seg_new_hexes[seg_idx] if seg_idx < len(team._seg_new_hexes) else []
                for h in new_hexes:
                    h = tuple(h)
                    if h in self.all_g_lands:
                        events.append((seg_day, order, h))

        events.sort(key=lambda e: (e[0], e[1]))
        visited = set()
        for day, _order, h in events:
            visited.add(h)
            if visited == self.all_g_lands:
                return day
        return None

    def _build_not_enough_food_message(self, needed_food, steps_available):
        """Build a clearer not-enough-food message with step/revisit context."""
        revisit_cost = _apply_g_reduction(10, self._are_all_g_lands_visited())
        msg = (
            f'Not enough food! Need {needed_food}, have {self.current_food}. '
            f'Steps available: {steps_available}. '
            f'(Food is one shared pool across all 3 teams - other teams spending it '
            f'is why an idle team can run low too, even with steps to spare.)'
        )
        if steps_available > 0:
            if self.current_food >= revisit_cost:
                msg += (
                    f' You can still move on visited hexes '
                    f'(revisit cost {revisit_cost} food each).'
                )
            else:
                msg += (
                    f' Even a revisit needs {revisit_cost} food. '
                    f'Please click "Next Day" to advance.'
                )
        else:
            msg += ' Please click "Next Day" to advance.'
        return msg

    def _save_day_record(self):
        """Save current day record based on all teams' moves this day."""
        teams = [self.team1]
        if self.team2:
            teams.append(self.team2)
        if self.team3:
            teams.append(self.team3)
        
        day_food_used = 0
        day_reward_used = 0
        
        for team in teams:
            for f, d in zip(team._seg_foods, team._seg_days):
                if d == self.current_day:
                    day_food_used += f
            for a, d in zip(team._seg_awards, team._seg_days):
                if d == self.current_day:
                    day_reward_used += a
        
        self.day_records.append({
            'day': self.current_day,
            'food_used': day_food_used,
            'reward_used': day_reward_used,
            'food_remain': self.current_food,
        })

    def _undo(self):
        self._clear_hover_preview()  # Clear preview on undo
        
        if not self.active_team._seg_lengths:
            self._status_msg = 'Nothing to undo.'
            self._draw()
            return
        length = self.active_team._seg_lengths.pop()
        food_undone = self.active_team._seg_foods.pop()
        reward_undone = self.active_team._seg_awards.pop()
        steps_undone = self.active_team._seg_steps.pop()
        seg_day = self.active_team._seg_days.pop()
        
        new_hexes_undone = self.active_team._seg_new_hexes.pop()
        exploration_hexes_undone = self.active_team._seg_exploration_hexes.pop()
        jumps_undone = self.active_team._seg_jumps.pop() if self.active_team._seg_jumps else []
        action_sequence_undone = self.active_team._seg_action_sequence.pop() if self.active_team._seg_action_sequence else []
        action_order_undone = self.active_team._seg_action_orders.pop() if self.active_team._seg_action_orders else None
        seg_path_nodes_undone = self.active_team._seg_path_nodes.pop() if self.active_team._seg_path_nodes else []
        seg_end_pos_undone = self.active_team._seg_end_positions.pop() if self.active_team._seg_end_positions else None
        hex_costs_undone = self.active_team._seg_hex_costs.pop()
        is_fly_skill = self.active_team._seg_is_fly_skill.pop() if self.active_team._seg_is_fly_skill else False
        seg_fly_skill_delta = 0
        if hasattr(self.active_team, '_seg_fly_skill_deltas') and self.active_team._seg_fly_skill_deltas:
            seg_fly_skill_delta = self.active_team._seg_fly_skill_deltas.pop()
        else:
            # Backward compatibility for older saves without fly-delta tracking.
            if is_fly_skill:
                seg_fly_skill_delta -= 1
            # Capturing a new BigBoss grants +1 fly skill for both normal and fly moves.
            if any(_terrain(*h).get('name') == 'bigBoss' for h in new_hexes_undone):
                seg_fly_skill_delta += 1

        # Capture pre-undo resource counters for debug reporting.
        fly_skill_before = self.fly_skill_limit
        x_bonus_before = self.active_team.x_bonus_remaining
        b_discount_before = self.active_team.b_discount_remaining
        z_bonus_before = self.active_team.z_bonus_remaining

        # Reverse the exact fly-skill effect introduced by the undone segment.
        self.fly_skill_limit -= seg_fly_skill_delta
        
        # A zero-length segment (in-place settle) added no nodes; note that
        # `del lst[-0:]` would wipe the whole list.
        if length > 0:
            del self.active_team.full_path[-length:]

        # Update shared resources
        self.total_food -= food_undone
        self.total_reward -= reward_undone
        
        # Restore the team's steps that were consumed
        self.active_team.steps += steps_undone
        
        # Remove hexes from visited sets
        # Remove exploration hexes from team's sets
        for hex_pos in exploration_hexes_undone:
            self.active_team.visited_hexes.discard(hex_pos)
            self.active_team.free_exploration_hexes.discard(hex_pos)
        
        # Remove new hexes from team's sets and global set (if no other team visited them)
        for hex_pos in new_hexes_undone:
            self.active_team.visited_hexes.discard(hex_pos)
            # Only remove from global set if only this team visited it
            # Check if any other team has this hex
            # A team that merely probed the hex (still in its free-exploration
            # set) or was created on it does not own it - only a paid visit does.
            other_teams_visited = False
            for team in [self.team1, self.team2, self.team3]:
                if (team and team is not self.active_team and hex_pos in team.visited_hexes
                        and hex_pos not in team.free_exploration_hexes):
                    other_teams_visited = True
                    break
            if not other_teams_visited:
                self.all_visited_hexes.discard(hex_pos)

        # A hex settled on departure (probe / deferred fly landing) reverts to an
        # unsettled exploration hex the team is still standing on.
        pos_after = self.active_team.full_path[-1]
        if pos_after in new_hexes_undone:
            self.active_team.visited_hexes.add(pos_after)
            self.active_team.free_exploration_hexes.add(pos_after)

        # Undoing a G/g capture has to switch the global 20% discount back
        # off; visited_g_lands is never removed from during play, so it must
        # be re-derived from what is left in segment history.
        self._recompute_visited_g_lands_from_segments()

        # Remove any no-draw edges that involve the removed segment
        # This cleans up portal teleport markers when undoing
        # Exactly the edges the undone segment could have registered (fly hop,
        # portal exit): consecutive pairs from the previous end through its
        # nodes, plus the hop to its resolved end position. Matching on hex
        # membership instead would also strip an earlier fly hop that this
        # segment merely revisited.
        prev_end = self.active_team.full_path[-1] if self.active_team.full_path else self.active_team.origin
        nodes_undone = [tuple(h) for h in (seg_path_nodes_undone or [])]
        chain = [prev_end] + nodes_undone
        undone_edges = set(zip(chain[:-1], chain[1:]))
        if seg_end_pos_undone is not None:
            end_undone = tuple(seg_end_pos_undone)
            undone_edges.add((prev_end, end_undone))
            undone_edges.add((nodes_undone[-2] if len(nodes_undone) >= 2 else prev_end, end_undone))
        self.active_team._no_draw_edges -= undone_edges
        
        # Restore the X/B/Z buff state as it was before the undone segment
        # (previously these were simply zeroed, which lost an active buff when
        # undoing any move made while it was running).
        self._derive_team_buff_state(self.active_team)

        fly_skill_after = self.fly_skill_limit
        x_bonus_after = self.active_team.x_bonus_remaining
        b_discount_after = self.active_team.b_discount_remaining
        z_bonus_after = self.active_team.z_bonus_remaining

        print(
            '[UNDO_APPLIED] '
            f'move_idx={len(self.active_team._seg_foods)+1}, '
            f'length={length}, day={seg_day}, '
            f'food_returned={food_undone}, reward_returned={reward_undone}, steps_returned={steps_undone}, '
            f'fly_skill_delta_reversed={-seg_fly_skill_delta}, fly_skill={fly_skill_before}->{fly_skill_after}, '
            f'x_bonus={x_bonus_before}->{x_bonus_after}, '
            f'b_discount={b_discount_before}->{b_discount_after}, '
            f'z_bonus={z_bonus_before}->{z_bonus_after}, '
            f'remaining_total_food={sum(self.active_team._seg_foods) if self.active_team._seg_foods else 0}'
        )
        
        # Recalculate max_day_reached based on remaining segments
        if self.active_team._seg_days:
            self.active_team.max_day_reached = max(self.active_team._seg_days)
        else:
            self.active_team.max_day_reached = self.active_team.created_day
        
        # Rebuild day records from remaining segments
        self._rebuild_day_records()
        self._auto_save_game()  # Auto-save after undo
        self._status_msg = ''
        self._draw()

    def _confirm_reset(self):
        """Show confirmation dialog before resetting."""
        try:
            from tkinter import messagebox
            if messagebox.askyesno('Confirm Reset', 'Are you sure you want to reset all teams and paths?\nThis cannot be undone.'):
                self._reset_path()
        except Exception as e:
            print(f'Error showing confirmation: {e}')
            self._reset_path()

    def _toggle_checkbox_state(self):
        """Toggle checkbox state and update button appearance."""
        self._chk_state = not self._chk_state
        # Update button text with checkmark/empty box
        text = '[X] 显示buff' if self._chk_state else '[ ] 显示buff'
        self._btn_chk_labels.label.set_text(text)
        # Update button color
        color = '#90EE90' if self._chk_state else '#FFCCCC'
        self._btn_chk_labels.color = color
        self._btn_chk_labels.hovercolor = ('#7FDF7F' if self._chk_state else '#FFB3B3')
        self.fig.canvas.draw_idle()
        # Toggle labels visibility
        self._toggle_bonus_labels()

    def _toggle_bonus_labels(self):
        """Toggle visibility of B/X/Z bonus hex labels."""
        self._show_bonus_labels = not self._show_bonus_labels
        self._draw()

    def _toggle_show_future_paths(self):
        """Toggle map visibility for segments whose day is later than current day."""
        self._show_future_paths = not self._show_future_paths
        state_text = '显示' if self._show_future_paths else '隐藏'
        self._status_msg = f'未来路径已{state_text}'
        self._update_show_future_button_state()
        self._draw()

    def _toggle_map_view(self):
        """Switch the main map between the drawn hex grid and the real-game
        screenshot (S24_map_rectified.png), which is pre-aligned so team
        paths/markers still line up correctly on top of the photo.
        """
        if self._map_view_mode == 'hex':
            if not self._load_map_image():
                # _load_map_image() already set an explanatory _status_msg.
                self._draw()
                return
            self._map_view_mode = 'image'
            self._status_msg = '已切换到实景地图'
        else:
            self._map_view_mode = 'hex'
            self._status_msg = '已切换到六边形地图'
        self._draw()

    def _reset_path(self):
        self._clear_hover_preview()  # Clear preview on reset
        
        # Reset all teams and shared state
        team1_origin = _find_start_position()
        self.team1 = Team(team1_origin, created_day=1)
        self.team2 = None
        self.team3 = None
        self.active_team = self.team1
        self.set_start_mode = None
        self._fly_mode = False  # Reset fly skill mode
        self._has_zoomed = False  # Reset zoom state
        self.fly_skill_limit = 1  # Reset global fly skill limit to 1
        self._next_action_order = 0
        self._enclosure_mode = False
        self._enclosure_start_day = None
        
        # Reset shared state
        self.all_visited_hexes = {team1_origin}
        self.visited_g_lands = set()  # Reset visited G/g lands
        self.current_day = 1
        self.current_food = 6800  # Day 1 starts with 6800 food
        self.total_food = 0
        self.total_reward = 0
        
        # Re-initialize day_records for days 1-TOTAL_DAYS
        self._init_day_records()
        
        # Stop any flashing animation
        if self._fly_button_timer is not None:
            self._fly_button_timer.stop()
            self._fly_button_timer = None
        
        self._update_switch_button_color()
        self._update_fly_button_state()  # Update button state after reset
        self._status_msg = 'Reset all teams.'
        self._has_zoomed = False  # Reset zoom to show full map
        self._draw()

    def _clone_team_through_day_for_save(self, team, through_day):
        if team is None:
            return None
        clone = copy.deepcopy(team)
        keep_count = 0
        for seg_day in clone._seg_days:
            if int(seg_day) <= int(through_day):
                keep_count += 1
            else:
                break
        for field_name in self._segment_field_names():
            value = getattr(clone, field_name, None)
            if value is not None:
                setattr(clone, field_name, value[:keep_count])
        clone.full_path = [clone.origin]
        for seg_nodes in clone._seg_path_nodes:
            clone.full_path.extend(seg_nodes)
        clone.max_day_reached = max(clone.created_day, max(clone._seg_days, default=clone.created_day))
        clone.visited_hexes = set(clone.full_path)
        clone.free_exploration_hexes = set()
        clone._no_draw_edges = set()
        return clone

    def _build_game_state_for_save(self, through_day=None):
        state_owner = self
        if through_day is not None:
            state_owner = PathfindingDemo.__new__(PathfindingDemo)
            state_owner.team1 = self._clone_team_through_day_for_save(self.team1, through_day)
            state_owner.team2 = self._clone_team_through_day_for_save(self.team2, through_day)
            state_owner.team3 = self._clone_team_through_day_for_save(self.team3, through_day)
            active_team_num = 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)
            state_owner.active_team = {
                1: state_owner.team1,
                2: state_owner.team2,
                3: state_owner.team3,
            }.get(active_team_num, state_owner.team1)
            state_owner.all_g_lands = set(self.all_g_lands)
            state_owner.current_day = max(1, min(int(through_day), TOTAL_DAYS))
            state_owner.current_food = self.current_food
            state_owner.total_food = self.total_food
            state_owner.total_reward = self.total_reward
            state_owner.fly_skill_limit = self.fly_skill_limit
            state_owner._next_action_order = self._next_action_order
            state_owner._day_edit_context = None
            state_owner._rebuild_shared_derived_state_from_segments()
            for team in (state_owner.team1, state_owner.team2, state_owner.team3):
                if team is not None:
                    state_owner._derive_team_buff_state(team)
            state_owner._rebuild_day_records()
            if state_owner.day_records and 1 <= state_owner.current_day <= len(state_owner.day_records):
                state_owner.current_food = state_owner.day_records[state_owner.current_day - 1]['food_remain']

        def serialize_team(team):
            if team is None:
                return None
            return {
                'full_path': [list(h) for h in team.full_path],
                'origin': list(team.origin),
                'visited_hexes': [list(h) for h in team.visited_hexes],
                'free_exploration_hexes': [list(h) for h in team.free_exploration_hexes],
                'x_bonus_remaining': team.x_bonus_remaining,
                'x_bonus_name': team.x_bonus_name,
                'b_discount_remaining': team.b_discount_remaining,
                'b_discount_name': team.b_discount_name,
                'z_bonus_remaining': team.z_bonus_remaining,
                'z_bonus_name': team.z_bonus_name,
                'max_day_reached': team.max_day_reached,
                'created_day': team.created_day,
                '_seg_foods': team._seg_foods,
                '_seg_steps': team._seg_steps,
                '_seg_awards': team._seg_awards,
                '_seg_days': team._seg_days,
                '_seg_new_hexes': [[list(h) for h in seg] for seg in team._seg_new_hexes],
                '_seg_exploration_hexes': [[list(h) for h in seg] for seg in team._seg_exploration_hexes],
                '_seg_jumps': [[list(h) for h in seg] for seg in team._seg_jumps],
                '_seg_path_nodes': [[list(h) for h in seg] for seg in team._seg_path_nodes],
                '_seg_end_positions': [list(h) for h in team._seg_end_positions],
                '_seg_action_sequence': [[(action, list(h) if isinstance(h, tuple) else h) for action, h in seg] for seg in team._seg_action_sequence],
                '_seg_action_orders': team._seg_action_orders,
                '_seg_lengths': team._seg_lengths,
                '_seg_hex_costs': team._seg_hex_costs,
                '_seg_is_fly_skill': team._seg_is_fly_skill,
                '_seg_fly_skill_deltas': team._seg_fly_skill_deltas,
                '_no_draw_edges': [[list(e[0]), list(e[1])] for e in team._no_draw_edges],
            }

        return {
            'current_day': state_owner.current_day,
            'current_food': state_owner.current_food,
            'total_food': state_owner.total_food,
            'total_reward': state_owner.total_reward,
            'fly_skill_limit': state_owner.fly_skill_limit,
            'team1': serialize_team(state_owner.team1),
            'team2': serialize_team(state_owner.team2),
            'team3': serialize_team(state_owner.team3),
            'active_team_num': 1 if state_owner.active_team is state_owner.team1 else (2 if state_owner.active_team is state_owner.team2 else 3),
            'all_visited_hexes': [list(h) for h in state_owner.all_visited_hexes],
            'visited_g_lands': [list(h) for h in state_owner.visited_g_lands],
            'day_records': state_owner.day_records,
        }

    def _segment_iter(self, team):
        """Yield segment tuples with stable start/end resolution.

        Returns tuples:
          (seg_idx, seg_day, seg_start, seg_end, seg_actions, seg_len)
        """
        if team is None:
            return

        cursor = 1
        for seg_idx, seg_len in enumerate(team._seg_lengths):
            # Zero-length segments (e.g. in-place settling of a probe hex via the
            # "占领试探地块" dialog) still carry a real action and must be reported;
            # only skip once we've run past the recorded path.
            if cursor - 1 >= len(team.full_path):
                break

            seg_day = team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else 1
            seg_actions = team._seg_action_sequence[seg_idx] if seg_idx < len(team._seg_action_sequence) else []
            seg_start = team.full_path[cursor - 1]

            if seg_idx < len(team._seg_end_positions):
                seg_end = team._seg_end_positions[seg_idx]
            else:
                seg_end_idx = min(cursor + seg_len - 1, len(team.full_path) - 1)
                seg_end = team.full_path[seg_end_idx]

            yield seg_idx, seg_day, seg_start, seg_end, seg_actions, seg_len
            cursor += seg_len

    def _operation_label_from_token(self, token):
        """Map terrain token to workbook operation label."""
        if token is None:
            return ''

        token = str(token)

        if token in ('1', '2', '3', '4', '5', '6'):
            return f'{token}级'
        if token == 'M':
            return '商人'
        if token == 'T':
            return '营地'
        if token == 'L':
            return '历练'
        if token == 'G':
            return '大卦'
        if token == 'g':
            return '3级卦'
        if token == 'B':
            return '大boss'
        if token == 'b':
            return '中boss'

        m_tower = re.fullmatch(r'T([2-6])', token)
        if m_tower:
            return f'草{m_tower.group(1)}'

        # Requirement: BXZ123 are exported as T123-equivalent labels.
        m_bxz = re.fullmatch(r'[BXZ]([123])', token)
        if m_bxz:
            return f'草{m_bxz.group(1)}'

        return ''

    def _infer_segment_portal_info(self, team, seg_start, seg_end):
        """Infer portal interaction for a segment.

        Returns (portal_source, portal_dest) or (None, None).
        """
        if team is None or seg_end is None:
            return None, None

        token_end = RAW_MAP[seg_end[0]][seg_end[1]] if 0 <= seg_end[0] < ROWS and 0 <= seg_end[1] < COLS else ''
        if not re.fullmatch(r'P\d+', token_end):
            return None, None

        portal_dest = seg_end
        teleported = (seg_start, seg_end) in team._no_draw_edges
        if not teleported:
            return seg_end, portal_dest

        paired = []
        for ir in range(ROWS):
            for ic in range(COLS):
                if RAW_MAP[ir][ic] == token_end and (ir, ic) != seg_end:
                    paired.append((ir, ic))

        portal_source = paired[0] if paired else seg_end
        return portal_source, portal_dest

    def _compute_bxz_adjustments(self):
        """Compute BXZ-driven adjustment totals per day.

        Returns:
                    day_food_adj: {day: positive int}    -> write into 调粮 (C4)
                    day_reward_adj: {day: positive int}  -> write into 调分 (C6)
          day_team_step_bonus: {day: {1:int,2:int,3:int}}
        """
        day_food_adj = {}
        day_reward_adj = {}
        day_team_step_bonus = {d: {1: 0, 2: 0, 3: 0} for d in range(1, TOTAL_DAYS + 1)}

        # Load buff parameters from global constants (populated from landInfo.json)
        b_bonus_map = B_DISCOUNT_MAP
        x_bonus_map = X_BONUS_MAP
        z_bonus_map = Z_BONUS_MAP

        for team_num, team in ((1, self.team1), (2, self.team2), (3, self.team3)):
            if team is None:
                continue

            b_rem = 0
            b_active_terrain = None  # Track which B hex is active
            x_rem = 0
            z_rem = 0
            z_active_terrain = None  # Track which Z hex is active

            for seg_idx, seg_day, _seg_start, seg_end, _seg_actions, _seg_len in self._segment_iter(team):
                new_hexes = team._seg_new_hexes[seg_idx] if seg_idx < len(team._seg_new_hexes) else []

                for h in new_hexes:
                    terrain = _terrain(h[0], h[1])
                    challenge_food = _get_terrain_food(terrain, seg_day)
                    reward = terrain.get('award', 0)
                    step_cost = terrain.get('step', 1)

                    if b_rem > 0 and challenge_food > 0:
                        # Get discount rate from landInfo, default to 0.4 (40% discount means 40% food savings)
                        b_discount_rate = _TERRAIN_DB.get(b_active_terrain, {}).get('food_discount_rate', 0.4)
                        day_food_adj[seg_day] = day_food_adj.get(seg_day, 0) + round(challenge_food * b_discount_rate)
                    if z_rem > 0 and reward > 0:
                        # Get reward bonus rate from landInfo, default to 1.4 (40% bonus)
                        z_bonus_rate = _TERRAIN_DB.get(z_active_terrain, {}).get('reward_bonus_rate', 1.4)
                        day_reward_adj[seg_day] = day_reward_adj.get(seg_day, 0) + round(reward * (z_bonus_rate - 1))
                    if x_rem > 0 and step_cost > 0:
                        day_team_step_bonus[seg_day][team_num] += step_cost

                    if b_rem > 0:
                        b_rem -= 1
                    if x_rem > 0:
                        x_rem -= 1
                    if z_rem > 0:
                        z_rem -= 1

                final_token = RAW_MAP[seg_end[0]][seg_end[1]] if seg_end is not None else ''
                final_is_new = seg_end in new_hexes if seg_end is not None else False
                if final_is_new:
                    if final_token in b_bonus_map:
                        b_rem = b_bonus_map[final_token]
                        b_active_terrain = final_token
                    if final_token in x_bonus_map:
                        x_rem = x_bonus_map[final_token]
                    if final_token in z_bonus_map:
                        z_rem = z_bonus_map[final_token]
                        z_active_terrain = final_token

                # Gameplay parity: portal interaction consumes one extra movement
                # for B discount / Z reward bonus (in addition to normal new-hex decrements).
                if re.fullmatch(r'P\d+', final_token):
                    if b_rem > 0:
                        b_rem -= 1
                    if z_rem > 0:
                        z_rem -= 1

        return day_food_adj, day_reward_adj, day_team_step_bonus

    def _build_excel_operations_by_day_team(self):
        """Build operation flow labels per (day, team), applying portal jump rules.

        Portal rule:
        - Untaken portal: 跳5
        - Taken portal: 跳1
        """
        def _merge_consecutive_jump_labels(op_list):
            """Merge consecutive 跳N labels into a single aggregated 跳X label."""
            merged = []
            jump_acc = 0
            for op in op_list:
                m = re.fullmatch(r'跳(\d+)', str(op))
                if m:
                    jump_acc += int(m.group(1))
                    continue
                if jump_acc > 0:
                    merged.append(f'跳{jump_acc}')
                    jump_acc = 0
                merged.append(op)

            if jump_acc > 0:
                merged.append(f'跳{jump_acc}')
            return merged

        operations_by_day_team = {(d, t): [] for d in range(1, TOTAL_DAYS + 1) for t in (1, 2, 3)}
        events = []

        for team_num, team in ((1, self.team1), (2, self.team2), (3, self.team3)):
            if team is None:
                continue

            for seg_idx, seg_day, seg_start, seg_end, seg_actions, _seg_len in self._segment_iter(team):
                entries = []
                seg_is_fly = (
                    seg_idx < len(team._seg_is_fly_skill)
                    and bool(team._seg_is_fly_skill[seg_idx])
                )
                if seg_is_fly:
                    entries.append({'label': '飞雷', 'is_new_g': False})

                for action in seg_actions:
                    if not isinstance(action, (list, tuple)) or len(action) < 2:
                        continue
                    action_type, h = action[0], action[1]
                    if isinstance(h, (list, tuple)):
                        h = tuple(h)
                    if not isinstance(h, tuple) or len(h) != 2:
                        continue

                    token = RAW_MAP[h[0]][h[1]] if 0 <= h[0] < ROWS and 0 <= h[1] < COLS else ''

                    # Portal interactions are handled once per segment with 跳5/跳1,
                    # so we skip portal-node action labels here.
                    if re.fullmatch(r'P\d+', token):
                        continue

                    if action_type == 'jump':
                        entries.append({'label': '跳1', 'is_new_g': False})
                    elif action_type == 'new':
                        label = self._operation_label_from_token(token)
                        if label:
                            entries.append({
                                'label': label,
                                'is_new_g': token in ('G', 'g'),
                                'g_pos': h if token in ('G', 'g') else None,
                            })

                # Backward-compatibility: some legacy saves have empty segment
                # action history. Reconstruct a minimal end-of-segment action so
                # export does not drop labels like 商人 on that day.
                if not seg_actions and seg_end is not None:
                    seg_new_hexes = team._seg_new_hexes[seg_idx] if seg_idx < len(team._seg_new_hexes) else []
                    seg_jumps = team._seg_jumps[seg_idx] if seg_idx < len(team._seg_jumps) else []
                    seg_explore_hexes = team._seg_exploration_hexes[seg_idx] if seg_idx < len(team._seg_exploration_hexes) else []
                    seg_new_set = {
                        tuple(h) for h in seg_new_hexes
                        if isinstance(h, (list, tuple)) and len(h) == 2
                    }
                    seg_jump_set = {
                        tuple(h) for h in seg_jumps
                        if isinstance(h, (list, tuple)) and len(h) == 2
                    }
                    seg_explore_set = {
                        tuple(h) for h in seg_explore_hexes
                        if isinstance(h, (list, tuple)) and len(h) == 2
                    }

                    token_end = (
                        RAW_MAP[seg_end[0]][seg_end[1]]
                        if 0 <= seg_end[0] < ROWS and 0 <= seg_end[1] < COLS
                        else ''
                    )
                    if not re.fullmatch(r'P\d+', token_end):
                        if seg_end in seg_jump_set:
                            entries.append({'label': '跳1', 'is_new_g': False})
                        elif seg_end in seg_explore_set:
                            # Probe step (试探步): the hex is not settled yet, so it
                            # must not appear as an operation on the day it was probed.
                            pass
                        else:
                            label_end = self._operation_label_from_token(token_end)
                            if label_end and (seg_end in seg_new_set or not seg_new_set):
                                entries.append({
                                    'label': label_end,
                                    'is_new_g': token_end in ('G', 'g'),
                                    'g_pos': seg_end if token_end in ('G', 'g') else None,
                                })

                portal_source, portal_dest = self._infer_segment_portal_info(team, seg_start, seg_end)

                events.append({
                    'day': seg_day,
                    'team_num': team_num,
                    'seg_idx': seg_idx,
                    'entries': entries,
                    'portal_source': portal_source,
                    'portal_dest': portal_dest,
                })

        events.sort(key=lambda e: (e['day'], e['team_num'], e['seg_idx']))
        taken_portals = set()
        taken_portal_tokens = set()

        visited_g = set()
        for team in (self.team1, self.team2, self.team3):
            if team is not None and team.origin in self.all_g_lands:
                visited_g.add(team.origin)
        unvisited_g = set(self.all_g_lands) - visited_g

        day_events = {}
        for ev in events:
            # A segment recorded outside the season's exportable day range
            # (e.g. a stray move from an older build/config where the season
            # length was different) has no day sheet to be written to - skip
            # it rather than crash the whole export.
            if not (1 <= ev['day'] <= TOTAL_DAYS):
                continue
            day_events.setdefault(ev['day'], []).append(ev)

        # On the day the G/g set is completed, every G/g capture that day acts
        # as a column barrier: anything recorded after it (any team) is pushed
        # to a column after it, so the sheet reads chronologically across all
        # three rows. A day can complete the set with more than one G/g capture
        # on it, in which case later actions must land after *all* of them -
        # see _export_day_sheets_xlsx's row-17 handling for the matching
        # per-column 八卦齐 split on that same day.
        g_completion_day = self._find_all_g_lands_complete_day()
        decisive_col_by_day = {}

        for day in sorted(day_events.keys()):
            pending = list(day_events[day])
            shared_after_g_col = None

            while pending:
                chosen_idx = 0
                # If exactly one G/g is missing globally, process the segment that
                # captures that decisive last G/g before other teams on this day.
                # Only consider each team's own earliest still-pending segment here:
                # a team's later segment must never jump ahead of that same team's
                # own earlier same-day segments, or the exported action order for
                # that team would no longer match what actually happened.
                if len(unvisited_g) == 1:
                    decisive_pos = next(iter(unvisited_g))
                    seen_teams = set()
                    g_first_idx = None
                    for i, pev in enumerate(pending):
                        if pev['team_num'] in seen_teams:
                            continue
                        seen_teams.add(pev['team_num'])
                        if any(ent.get('g_pos') == decisive_pos for ent in pev.get('entries', [])):
                            g_first_idx = i
                            break
                    if g_first_idx is not None:
                        chosen_idx = g_first_idx

                ev = pending.pop(chosen_idx)
                key = (ev['day'], ev['team_num'])
                entries = list(ev.get('entries', []))

                # Within that segment, move the decisive last G/g action to the front.
                if len(unvisited_g) == 1:
                    decisive_pos = next(iter(unvisited_g))
                    final_g_idx = next((i for i, e in enumerate(entries) if e.get('g_pos') == decisive_pos), None)
                    if final_g_idx is not None and final_g_idx > 0:
                        final_g_entry = entries.pop(final_g_idx)
                        entries.insert(0, final_g_entry)

                # Once this day's decisive G/g has been captured, pad this
                # team's row up to the shared column before appending, so its
                # next entry lines up with whatever column other teams'
                # post-G/g entries are also starting from. Consecutive jump
                # labels get merged into one "跳N" cell in a later pass, so
                # the column a team's entries-so-far actually end up in is
                # the *merged* count, not the raw per-hex entry count.
                if shared_after_g_col is not None:
                    merged_len_so_far = len(_merge_consecutive_jump_labels(operations_by_day_team[key]))
                    if merged_len_so_far < shared_after_g_col:
                        operations_by_day_team[key].extend([None] * (shared_after_g_col - merged_len_so_far))

                operations_by_day_team[key].extend(e.get('label', '') for e in entries)

                captured_g_now = False
                g_completed_now = False
                for e in entries:
                    g_pos = e.get('g_pos')
                    if g_pos in unvisited_g:
                        unvisited_g.remove(g_pos)
                        captured_g_now = True
                        if not unvisited_g:
                            g_completed_now = True

                portal_source = ev['portal_source']
                portal_dest = ev['portal_dest']
                if portal_source is not None:
                    portal_ref = portal_dest if portal_dest is not None else portal_source
                    portal_token = (
                        RAW_MAP[portal_ref[0]][portal_ref[1]]
                        if portal_ref is not None and 0 <= portal_ref[0] < ROWS and 0 <= portal_ref[1] < COLS
                        else ''
                    )

                    is_taken = (
                        portal_token in taken_portal_tokens
                        if re.fullmatch(r'P\d+', portal_token)
                        else portal_source in taken_portals
                    )
                    jump_label = '跳1' if is_taken else '跳5'
                    operations_by_day_team[key].append(jump_label)

                    if re.fullmatch(r'P\d+', portal_token):
                        taken_portal_tokens.add(portal_token)
                    taken_portals.add(portal_source)
                    if portal_dest is not None:
                        taken_portals.add(portal_dest)

                # On the day the set is completed, *every* G/g capture raises the
                # shared column barrier - a single day can hold more than one
                # G/g land, and later actions (by any team) must be written to
                # the right of all of them, not just of the last one.
                if day == g_completion_day and captured_g_now:
                    col_after_this_g = len(_merge_consecutive_jump_labels(operations_by_day_team[key]))
                    shared_after_g_col = max(shared_after_g_col or 0, col_after_this_g)
                    if g_completed_now:
                        decisive_col_by_day[day] = shared_after_g_col

        for key, ops in operations_by_day_team.items():
            operations_by_day_team[key] = _merge_consecutive_jump_labels(ops)

        return operations_by_day_team, decisive_col_by_day

    def _export_day_sheets_xlsx(self):
        """Export operations into day sheets (1..TOTAL_DAYS) of an Excel template workbook."""
        try:
            import os
            import sys

            # Immediate feedback so users can tell the button click was received.
            self._status_msg = 'SaveXLX clicked: exporting...'
            self._draw()
            print('DEBUG: SaveXLX button clicked')

            try:
                # Keep this import explicit so PyInstaller includes openpyxl
                # when the application is built from the Python entry point.
                from openpyxl import load_workbook
            except Exception:
                self._status_msg = 'Export error: openpyxl is required. Please install openpyxl first.'
                self._draw()
                print('ERROR: SaveXLX requires openpyxl')
                try:
                    import tkinter as _tk
                    from tkinter import messagebox as _msgbox
                    _root = _tk.Tk()
                    _root.withdraw()
                    _msgbox.showerror('SaveXLX', 'Export error: openpyxl is required.')
                    _root.destroy()
                except Exception:
                    pass
                return

            # One-click export: in packaged app, prefer the exe folder (not _MEIPASS temp folder).
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(os.path.abspath(sys.executable))
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            search_dirs = [base_dir]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in search_dirs:
                search_dirs.append(script_dir)
            meipass_dir = getattr(sys, '_MEIPASS', None)
            if meipass_dir and meipass_dir not in search_dirs:
                search_dirs.append(meipass_dir)

            template_names = ['S25_分表2.0_导出模板.xlsx']
            template_path = None
            for d in search_dirs:
                for name in template_names:
                    cand = os.path.join(d, name)
                    if os.path.exists(cand):
                        template_path = cand
                        break
                if template_path:
                    break

            if not template_path:
                self._status_msg = 'Export error: template not found (S25_分表2.0_导出模板.xlsx).'
                self._draw()
                print(f'ERROR: template not found in: {search_dirs}')
                try:
                    import tkinter as _tk
                    from tkinter import messagebox as _msgbox
                    _root = _tk.Tk()
                    _root.withdraw()
                    _msgbox.showerror('SaveXLX', 'Template not found:\n' + '\n'.join(search_dirs))
                    _root.destroy()
                except Exception:
                    pass
                return

            output_path = os.path.join(base_dir, 'S25_分表2.0.xlsx')

            wb = load_workbook(template_path)

            operations_by_day_team, decisive_col_by_day = self._build_excel_operations_by_day_team()
            day_food_adj, day_reward_adj, _day_team_step_bonus = self._compute_bxz_adjustments()

            # Rebuild day records so we can write final end-action values from canonical state.
            self._rebuild_day_records()

            op_rows = {1: 8, 2: 11, 3: 14}
            end_action_rows = {1: 2, 2: 3, 3: 4}
            day_record_end_key = {1: 'team1_steps_remain', 2: 'team2_steps_remain', 3: 'team3_steps_remain'}

            # C..AA => 25 operation slots.
            op_start_col = 3
            op_end_col = 27

            all_g_complete_day = self._find_all_g_lands_complete_day()

            for day in range(1, TOTAL_DAYS + 1):
                sheet_name = str(day)
                if sheet_name not in wb.sheetnames:
                    continue

                ws = wb[sheet_name]

                # Fill 操作流 for each team.
                for team_num in (1, 2, 3):
                    row = op_rows[team_num]
                    ops = operations_by_day_team.get((day, team_num), [])
                    ops = ops[:25]

                    for col in range(op_start_col, op_end_col + 1):
                        slot_idx = col - op_start_col
                        ws.cell(row=row, column=col, value=ops[slot_idx] if slot_idx < len(ops) else None)

                # BXZ adjustment columns requested by user.
                ws['C4'] = int(day_food_adj.get(day, 0))
                weekly_extra_reward = 500 if (day == 1 or (day - 1) % 7 == 3) else 0
                ws['C6'] = int(day_reward_adj.get(day, 0)) + weekly_extra_reward

                # 结束行动: write actual end-of-day action counts from rebuilt day_records.
                rec = self.day_records[day - 1] if 1 <= day <= len(self.day_records) else {}
                for team_num in (1, 2, 3):
                    row = end_action_rows[team_num]
                    key = day_record_end_key[team_num]
                    ws.cell(row=row, column=9, value=int(rec.get(key, 0)))  # Column I

                # 八卦齐 (row 17): once all G/g lands have been captured, every
                # slot from the following day onward gets the global discount.
                # On the completion day itself, only the slots after the
                # decisive G/g land's own column (shared across all teams)
                # get it - the G/g land's own slot and everything before it
                # still reflect the pre-completion state.
                if all_g_complete_day is not None and day > all_g_complete_day:
                    for col in range(op_start_col, op_end_col + 1):
                        ws.cell(row=17, column=col, value='是')
                elif day == all_g_complete_day:
                    shared_col = decisive_col_by_day.get(day)
                    if shared_col is not None:
                        for col in range(op_start_col, op_end_col + 1):
                            slot_idx = col - op_start_col
                            if slot_idx >= shared_col:
                                ws.cell(row=17, column=col, value='是')

            wb.save(output_path)

            self._status_msg = f'Excel exported: {os.path.basename(output_path)}'
            self._draw()
            print(f'DEBUG: Excel exported to {output_path}')
            try:
                import tkinter as _tk
                from tkinter import messagebox as _msgbox
                _root = _tk.Tk()
                _root.withdraw()
                _msgbox.showinfo('SaveXLX', f'Excel exported:\n{output_path}')
                _root.destroy()
            except Exception:
                pass
        except Exception as e:
            self._status_msg = f'Export error: {str(e)}'
            self._draw()
            print(f'Excel export error: {e}')
            try:
                import tkinter as _tk
                from tkinter import messagebox as _msgbox
                _root = _tk.Tk()
                _root.withdraw()
                _msgbox.showerror('SaveXLX', f'Export error:\n{str(e)}')
                _root.destroy()
            except Exception:
                pass

    def _save_game(self):
        """Save game state to a JSON file."""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            import os
            
            print('DEBUG: Starting save process...')
            
            root = tk.Tk()
            root.withdraw()

            save_current_day = messagebox.askyesnocancel(
                '保存范围',
                (
                    f'请选择要保存的路线范围：\n\n'
                    f'是：只保存到当前 Day {self.current_day} 的路线\n'
                    f'否：保存所有路线\n'
                    f'取消：不保存'
                ),
            )
            if save_current_day is None:
                print('DEBUG: User cancelled save scope dialog')
                root.destroy()
                return
            save_through_day = self.current_day if save_current_day else None
            initial_file = f'game_save_day{self.current_day}.json' if save_current_day else 'game_save.json'
            
            file_path = filedialog.asksaveasfilename(
                defaultextension='.json',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
                initialfile=initial_file
            )
            
            if not file_path:
                print('DEBUG: User cancelled save dialog')
                root.destroy()
                return
            
            print(f'DEBUG: Selected file path: {file_path}')
            print(f'DEBUG: File path exists before write: {os.path.exists(file_path)}')
            
            print('DEBUG: Serializing game state...')
            game_state = self._build_game_state_for_save(save_through_day)
            
            print(f'DEBUG: Writing to file: {file_path}')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(game_state, f, indent=2)
            
            print(f'DEBUG: File write complete. File size: {os.path.getsize(file_path)} bytes')
            
            root.destroy()
            scope_text = f'Day {self.current_day}' if save_current_day else 'all routes'
            self._status_msg = f'Game saved ({scope_text}) to {os.path.basename(file_path)}'
            self._draw()
            print(f'Game saved successfully to {file_path}')
        except Exception as e:
            print(f'Error saving game: {e}')
            import traceback
            traceback.print_exc()
            try:
                root.destroy()
            except:
                pass
            self._status_msg = f'Save error: {str(e)}'
            self._draw()

    def _on_click(self, event):
        """Handle mouse release to draw path or set team start positions."""
        self._clear_hover_preview()  # Clear preview on click
        
        # Handle pan release
        self._on_release(event)

        # Only track left mouse button for UI clicks.
        if event.button != 1:
            return

        # Click on day/date badge opens quick day picker.
        try:
            if self._day_number_text is not None:
                hit_day_badge, _ = self._day_number_text.contains(event)
                if hit_day_badge:
                    self._show_day_picker()
                    return
        except Exception:
            pass

        # Fallback for backends where matplotlib Button callbacks are unreliable.
        try:
            if hasattr(self, '_btn_export_xlsx') and self._btn_export_xlsx is not None:
                if event.inaxes == self._btn_export_xlsx.ax:
                    self._export_day_sheets_xlsx()
                    return
        except Exception:
            pass
        
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Process the click on release
        self._process_path_click(event.xdata, event.ydata)

    def _process_path_click(self, xdata, ydata):
        """Process a path click - called from _on_mouse_release when click is confirmed."""
        try:
            if self._segment_edit_mode and self._day_edit_context is None:
                self._handle_segment_edit_click(xdata, ydata)
                return

            # Keep only the latest click's landing-cost details for terminal debugging.
            self._last_landing_cost_breakdown = {}

            pos = self._pixel_to_hex(xdata, ydata)
            if pos is None:
                return
            if not (0 <= pos[0] < ROWS and 0 <= pos[1] < COLS):
                return

            # Handle setting starting points for teams 2 and 3
            if self.set_start_mode == 'team2':
                if not _passable(*pos):
                    self._status_msg = 'Team 2 starting point must be on passable terrain.'
                    self._draw()
                    return
                self.team2 = Team(pos, created_day=self.current_day)
                self.all_visited_hexes.add(pos)
                self.active_team = self.team2
                self.set_start_mode = None
                self._update_switch_button_color()
                self._status_msg = 'Team 2 created. Building Team 2 path...'
                self._draw()
                return

            if self.set_start_mode == 'team3':
                if not _passable(*pos):
                    self._status_msg = 'Team 3 starting point must be on passable terrain.'
                    self._draw()
                    return
                self.team3 = Team(pos, created_day=self.current_day)
                self.all_visited_hexes.add(pos)
                self.active_team = self.team3
                self.set_start_mode = None
                self._update_switch_button_color()
                self._status_msg = 'Team 3 created. Building Team 3 path...'
                self._draw()
                return

            # Navigate mode: extend path for active team to clicked hex
            if self.active_team is None:
                self._status_msg = 'No active team selected.'
                self._draw()
                return

            # A normal path click always attributes the new segment to whatever day
            # is currently being viewed (self.current_day) - see the "current viewing
            # day, not max_day_reached" comments below where seg_day is actually set.
            # An earlier version of this code force-snapped the viewed day back to
            # the active team's last action day here, meant to help a team that had
            # gone idle while other teams advanced the globally-displayed day. But it
            # fired on *every* click regardless of why current_day differed from the
            # team's last action day, so it also silently overrode a deliberate,
            # explicit day change - e.g. clicking to continue banked steps on a later
            # day (steps accrue day by day up to a cap, so leaving a day's steps
            # partially unused to act on a later day is a legitimate way to play) got
            # silently re-attributed back to the earlier day instead, and once that
            # earlier day's bank was exhausted this way the team could no longer act
            # at all even though the later day still had steps shown as available.
            # self.current_day is only ever changed by explicit navigation elsewhere
            # (Next/Prev Day, jump-to-day, double-click team button, etc.), so it can
            # be trusted here without a defensive override.
            if self._is_active_day_edit():
                self.current_day = self._day_edit_context['day']

            day_locked, last_move_day = self._is_active_team_locked_by_day()
            if day_locked and not self._is_active_day_edit() and not self._enclosure_mode:
                self._status_msg = (
                    f"Cannot draw path: current day ({self.current_day}) is before "
                    f"this team's last movement day ({last_move_day})."
                )
                self._draw()
                return
                
            if not _passable(*pos):
                self._status_msg = 'Cannot navigate to empty terrain.'
                self._draw()
                return

            # Handle fly skill mode - direct teleportation with waived movement cost
            if self._fly_mode:
                current_pos = self.active_team.full_path[-1]
                if pos == current_pos:
                    self._status_msg = 'Already at this hex.'
                    return

                if not self._is_valid_fly_destination(pos):
                    self._status_msg = (
                        '飞雷神只能到达未占领地块，且目标必须位于当前队伍领地外沿的两跳范围内。'
                    )
                    self._draw()
                    return
                
                try:
                    t = _terrain(*pos)
                    is_new_hex = pos not in self.all_visited_hexes
                    # Landing by fly on an untaken, capturable (step > 0), non-portal
                    # hex does not occupy it yet: like a probe step, it is settled
                    # (challenge food + reward + 1 step) when the team leaves it.
                    # Portals (step 0), Tents and other step<=0 terrain keep the
                    # immediate handling below.
                    defer_capture = (not self._enclosure_mode) and is_new_hex and t.get('step', 1) > 0

                    # Flying off a still-unsettled probe / deferred-fly hex settles
                    # it, exactly like walking off it does.
                    is_leaving_exploration = (
                        current_pos in self.active_team.free_exploration_hexes
                        and not self._is_hex_settled(current_pos)
                    )
                    # Snapshotted so an "expired" message can be shown after both
                    # potential charges below (departure settle + landing) have
                    # been consumed one at a time, rather than bulk-decrementing
                    # by hex count afterward - which could let a second hex in
                    # this same fly click get the discount/bonus even with only
                    # one charge actually left.
                    b_discount_before_seg = self.active_team.b_discount_remaining
                    z_bonus_before_seg = self.active_team.z_bonus_remaining
                    dep_food = dep_award = 0
                    if is_leaving_exploration:
                        dep_t = _terrain(*current_pos)
                        dep_terrain_food = _get_terrain_food(dep_t, self.current_day)
                        dep_food = _apply_challenge_discounts(
                            dep_terrain_food, self.active_team,
                            self._are_all_g_lands_visited(), is_tent=dep_t.get('name') == 'Tent')
                        if dep_terrain_food >= 0 and self.active_team.b_discount_remaining > 0:
                            self.active_team.b_discount_remaining -= 1
                        dep_award = dep_t['award']
                        if self.active_team.z_bonus_remaining > 0:
                            dep_award = _apply_z_bonus(dep_award, self.active_team)
                            self.active_team.z_bonus_remaining -= 1

                    # Calculate base challenge cost for destination terrain.
                    # Fly always waives the movement fee, including for BigBoss.
                    # A deferred capture (defer_capture) hasn't happened yet, so
                    # nothing is charged on landing; the challenge food is charged
                    # later at settle time (is_leaving_exploration / in-place-settle
                    # code paths below).
                    raw_challenge_food = _get_terrain_food(t, self.current_day)
                    challenge_food = _apply_challenge_discounts(
                        raw_challenge_food,
                        self.active_team,
                        self._are_all_g_lands_visited(),
                        is_tent=t.get('name') == 'Tent'
                    )
                    if is_new_hex and not defer_capture and raw_challenge_food >= 0 and self.active_team.b_discount_remaining > 0:
                        self.active_team.b_discount_remaining -= 1
                    if defer_capture:
                        challenge_food = 0
                    seg_food = challenge_food
                    self._last_landing_cost_breakdown[pos] = {
                        'challenge': challenge_food,
                        'movement': 0,
                        'revisit': 0,
                        'total': seg_food,
                    }
                    seg_award = 0 if defer_capture else t['award']
                    # Apply Z bonus if active
                    if self.active_team.z_bonus_remaining > 0 and is_new_hex and not defer_capture:
                        seg_award = _apply_z_bonus(seg_award, self.active_team)
                    seg_steps = 0  # Fly doesn't use steps
                    # If terrain gives steps back (e.g. Tent step=-1), apply benefit for new hexes
                    if pos not in self.all_visited_hexes:
                        terrain_step_val = t.get('step', 1)
                        if terrain_step_val < 0:
                            seg_steps = terrain_step_val  # e.g. Tent gives -1 → team gains 1 step
                    
                    # Check if already visited
                    if pos in self.all_visited_hexes:
                        seg_food = 10
                        # Apply G/g reduction to revisit cost
                        seg_food = _apply_g_reduction(seg_food, self._are_all_g_lands_visited())
                        seg_award = 0
                        self._last_landing_cost_breakdown[pos] = {
                            'challenge': 0,
                            'movement': 0,
                            'revisit': seg_food,
                            'total': seg_food,
                        }

                    landing_food, landing_award = seg_food, seg_award
                    seg_food += dep_food
                    seg_award += dep_award
                    if is_leaving_exploration:
                        seg_steps += 1

                    # Day advancement if needed
                    seg_start_day = self.current_day
                    # Attribute fly skill move to current viewing day, not max_day_reached
                    seg_day = self.current_day
                    
                    # Track the maximum day reached by this team (for auto-display switching)
                    if seg_day > self.active_team.max_day_reached:
                        self.active_team.max_day_reached = seg_day
                    
                    # Check if resources are insufficient
                    allow_borrow_edit_resources = self._enclosure_mode or (
                        self._is_active_day_edit() and self.current_day == self._day_edit_context['day']
                    )
                    if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                        steps_avail = self._get_team_steps_for_day(self.active_team, self.current_day)
                        raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_avail))
                    if is_leaving_exploration and not allow_borrow_edit_resources:
                        steps_avail = self._get_team_steps_for_day(self.active_team, self.current_day)
                        if steps_avail < 1:
                            raise RuntimeError(
                                f'Not enough steps! Settling ({current_pos[0]},{current_pos[1]}) on departure '
                                f'needs 1, have {steps_avail}. Please click "Next Day" button to advance.')

                    # Apply teleportation
                    self.active_team.full_path.append(pos)
                    
                    # Mark this edge as no-draw (similar to portal teleports)
                    self.active_team._no_draw_edges.add((current_pos, pos))
                    
                    self.active_team._seg_lengths.append(1)
                    self.active_team._seg_foods.append(seg_food)
                    
                    self.active_team._seg_awards.append(seg_award)
                    self.active_team._seg_steps.append(seg_steps)
                    self.active_team._seg_days.append(seg_day)
                    is_fly_new_hex = is_new_hex and not defer_capture  # captured on landing
                    captured = ([current_pos] if is_leaving_exploration else []) + ([pos] if is_fly_new_hex else [])
                    self.active_team._seg_new_hexes.append(captured)
                    self.active_team._seg_exploration_hexes.append([pos] if defer_capture else [])
                    self.active_team._seg_jumps.append([])  # No jumps for fly skill
                    self.active_team._seg_path_nodes.append([pos])
                    self.active_team._seg_end_positions.append(pos)
                    self.active_team._seg_action_sequence.append(
                        [('fly', pos)] + [('new', h) for h in captured]
                    )
                    self._append_segment_action_order(self.active_team)
                    self.active_team._seg_hex_costs.append(
                        ([(dep_food, dep_award, 1)] if is_leaving_exploration else [])
                        + [(landing_food, landing_award, 0)])
                    self.active_team._seg_is_fly_skill.append(True)  # Mark as fly skill move
                    
                    if not self._enclosure_mode:
                        self.current_food -= seg_food
                        self.total_food += seg_food
                        self.total_reward += seg_award
                    
                    # Decrement GLOBAL fly skill limit
                    self.fly_skill_limit -= 1
                    seg_fly_skill_delta = -1
                    
                    # BigBoss grants +1 fly skill when captured: on landing if captured
                    # now, or when a deferred-fly / probe hex is settled on departure.
                    dest_terrain = _terrain(pos[0], pos[1])
                    if dest_terrain.get('name') == 'bigBoss' and is_fly_new_hex:
                        self.fly_skill_limit += 1
                        seg_fly_skill_delta += 1
                        self._status_msg = f'✓ Flew to ({pos[0]},{pos[1]})! Reached BigBoss! Fly skill +1 (now {self.fly_skill_limit})'
                    elif defer_capture:
                        self._status_msg = f'✓ Flew to ({pos[0]},{pos[1]}) - settled on departure. Fly skill limit: {self.fly_skill_limit}'
                    else:
                        self._status_msg = f'✓ Flew to ({pos[0]},{pos[1]})! Fly skill limit: {self.fly_skill_limit}'
                    if is_leaving_exploration and _terrain(*current_pos).get('name') == 'bigBoss':
                        self.fly_skill_limit += 1
                        seg_fly_skill_delta += 1
                        self._status_msg += f' ⭐ Settled BigBoss! Fly skill +1 (now {self.fly_skill_limit})'

                    # Check for X1, X2, X3 bonus free movements (only if captured on landing)
                    dest_terrain_name = RAW_MAP[pos[0]][pos[1]]
                    if dest_terrain_name in X_BONUS_MAP and is_fly_new_hex:
                        self._status_msg += self._activate_land_buffs(pos)

                    # B/Z charges were already decremented one at a time, right
                    # where each was actually used (departure settle above,
                    # landing above) - just report if either ran out this move.
                    if b_discount_before_seg > 0 and self.active_team.b_discount_remaining == 0:
                        self._status_msg += ' B discount expired.'
                    if z_bonus_before_seg > 0 and self.active_team.z_bonus_remaining == 0:
                        self._status_msg += ' Z reward bonus expired.'

                    # A hex settled on departure is captured now, so its buff starts now.
                    if is_leaving_exploration:
                        self._status_msg += self._activate_land_buffs(current_pos)

                    # Check for B1, B2, B3 bonus food discount (activate only if landing on NEW B hex)
                    if dest_terrain_name in ('B1', 'B2', 'B3') and is_fly_new_hex:
                        self.active_team.b_discount_remaining = B_DISCOUNT_MAP.get(dest_terrain_name, 5)
                        self.active_team.b_discount_name = dest_terrain_name
                        discount_rate = _TERRAIN_DB.get(dest_terrain_name, {}).get('food_discount_rate', 0.4)
                        self._status_msg += f' {int(discount_rate*100)}% food discount active for next {self.active_team.b_discount_remaining} movements!'
                    
                    # Check for Z1, Z2, Z3 bonus reward (activate only if landing on NEW Z hex)
                    if dest_terrain_name in ('Z1', 'Z2', 'Z3') and is_fly_new_hex:
                        self.active_team.z_bonus_remaining = Z_BONUS_MAP.get(dest_terrain_name, 5)
                        self.active_team.z_bonus_name = dest_terrain_name
                        bonus_rate = _TERRAIN_DB.get(dest_terrain_name, {}).get('reward_bonus_rate', 1.4)
                        self._status_msg += f' {int((bonus_rate-1)*100)}% reward bonus active for next {self.active_team.z_bonus_remaining} movements!'
                    
                    if defer_capture:
                        self.active_team.visited_hexes.add(pos)
                        self.active_team.free_exploration_hexes.add(pos)
                        if pos in self.all_g_lands:
                            self.visited_g_lands.add(pos)
                    elif is_new_hex:
                        self.active_team.visited_hexes.add(pos)
                        self.all_visited_hexes.add(pos)
                        # Track G/g land visits
                        if pos in self.all_g_lands:
                            self.visited_g_lands.add(pos)
                    if is_leaving_exploration:
                        self.all_visited_hexes.add(current_pos)
                    # Unconditional: also drops a stale entry for a hex another team settled.
                    self.active_team.free_exploration_hexes.discard(current_pos)

                    self.active_team._seg_fly_skill_deltas.append(seg_fly_skill_delta)
                    
                    # Check for portal teleportation after flying to destination
                    portal_status = self._check_portal_teleport()
                    if self.active_team._seg_end_positions:
                        self.active_team._seg_end_positions[-1] = self.active_team.full_path[-1]
                    
                    # Center view on team after fly
                    self._center_view_on_active_team()
                    
                    # If team has reached a later day, auto-switch display to that day
                    if self.active_team.max_day_reached > self.current_day:
                        self.current_day = self.active_team.max_day_reached
                        self._status_msg += f' [Auto-switched to Day {self.current_day}]'
                    
                    if not self._enclosure_mode:
                        self._rebuild_day_records()

                    if self._is_active_day_edit():
                        self._finalize_day_segment_edit_if_connected()

                    self._auto_save_game()  # Auto-save after fly
                    self._fly_mode = False  # Deactivate fly mode

                    # Stop flashing animation
                    if self._fly_button_timer is not None:
                        self._fly_button_timer.stop()
                        self._fly_button_timer = None

                    self._set_fly_button_border('#777777', 0.8)  # Reset button border
                    self._draw()
                    return
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    self._status_msg = f'Error during flight: {str(e)} (see terminal for details)'
                    print(f'FLIGHT ERROR:\n{error_details}')
                    self._draw()
                    return

            current = self.active_team.full_path[-1]
            if pos == current:
                if self._enclosure_mode:
                    self._status_msg = '圈地模式中：点击其他地块继续画路线。'
                    self._draw()
                    return
                if self._confirm_settle_current_exploration():
                    return
                return

            segment, cost_map = _astar(current, pos, set())
            replay_path = getattr(self, '_replay_path_override', None)
            if replay_path is not None:
                replay_path = [tuple(hex_pos) for hex_pos in replay_path]
                if not replay_path or replay_path[0] != current or replay_path[-1] != pos:
                    self._status_msg = 'Invalid replay path: start or destination does not match.'
                    self._draw()
                    return
                segment = replay_path
            if not segment or len(segment) < 2:
                self._status_msg = f'No path found to ({pos[0]},{pos[1]}).'
                self._draw()
                return
        except Exception as e:
            self._status_msg = f'Error in pathfinding: {str(e)}'
            self._draw()
            return

        # Limit path length
        if len(segment) > 18:
            self._status_msg = f'Path too long ({len(segment)} hexes). Max 18 hexes per segment allowed.'
            self._draw()
            return

        try:
            # Append segment (skip the first node – it's already in full_path)
            added = segment[1:]
            
            # Get current position for exploration and departure checks
            current_pos = self.active_team.full_path[-1]

            # Leaving a hex this team explored "for free" settles it (challenge food
            # + reward) - unless another team already paid for it: a team created
            # on someone's exploration hex inherits it, and whichever team moves
            # off first settles it, so the other must not be charged again.
            is_leaving_exploration = (
                current_pos in self.active_team.free_exploration_hexes
                and not self._is_hex_settled(current_pos)
            )

            # Snapshot B/Z buff charges before this segment's pricing consumes
            # any of them, so an "expired" message can be shown afterward - the
            # charges themselves are now decremented one at a time as each new
            # hex in this segment is actually priced (see below), not in one
            # lump sum by hex count at the end, which used to let every hex in
            # an over-sized multi-hex segment get the discount/bonus even once
            # the charge count ran out partway through it.
            b_discount_before_seg = self.active_team.b_discount_remaining
            z_bonus_before_seg = self.active_team.z_bonus_remaining

            # 圈地模式只记录路线，退出时再统一分配到未来 days。
            steps_available_for_day = 0 if self._enclosure_mode else self._get_team_steps_for_day(self.active_team, self.current_day)

            # Check for free exploration mode: if team has 0 steps but can move to adjacent untaken hex
            free_exploration = False
            if (not self._enclosure_mode) and steps_available_for_day == 0 and self.active_team.x_bonus_remaining <= 0 and len(added) == 1:
                hex_to_explore = added[0]
                if hex_to_explore not in self.all_visited_hexes:
                    explore_terrain = _terrain(*hex_to_explore)
                    if explore_terrain.get('step', 1) > 0:
                        if not is_leaving_exploration:
                            free_exploration = True

            # Confirm before a probe step. Nothing has been mutated yet at this
            # point, so declining just returns.
            if free_exploration and not self._confirm_free_exploration_step(added[0]):
                self._draw()
                return

            # If leaving a free exploration hex, calculate its challenge food cost
            departure_challenge = 0
            departure_is_tent = False
            if is_leaving_exploration:
                _dep_terrain = _terrain(*current_pos)
                departure_challenge = _get_terrain_food(_dep_terrain, self.current_day)
                departure_is_tent = _dep_terrain.get('name') == 'Tent'
            
            # The last G/g land's own capture is still charged at the
            # pre-completion price, but hexes captured after it *in the same
            # click* must already get the global 20% discount. visited_g_lands
            # is only updated once the whole segment is committed, so track
            # completion as this segment's pricing progresses instead of
            # evaluating it once for the entire segment.
            g_taken_in_seg = set()

            def _all_g_done():
                if not self.all_g_lands:
                    return False
                return len(self.visited_g_lands | g_taken_in_seg) == len(self.all_g_lands)

            # Calculate costs
            # Apply B discount and G reduction additively to departure challenge
            seg_food = _apply_challenge_discounts(departure_challenge, self.active_team, _all_g_done(), is_tent=departure_is_tent)
            if is_leaving_exploration and self.active_team.b_discount_remaining > 0:
                self.active_team.b_discount_remaining -= 1
            seg_award = seg_steps = 0
            new_hexes = []
            exploration_hexes = []
            jumps = []  # Track revisits (jumps)
            action_sequence = []  # Track order of actions: ('new', hex) or ('jump', hex)
            segment_hexes = set()
            hex_costs = []

            if is_leaving_exploration:
                departure_terrain = _terrain(*current_pos)
                departure_reward = departure_terrain['award']
                # Apply Z bonus if active
                if self.active_team.z_bonus_remaining > 0:
                    departure_reward = _apply_z_bonus(departure_reward, self.active_team)
                    self.active_team.z_bonus_remaining -= 1
                seg_award += departure_reward
                seg_steps += 1
                new_hexes.append(current_pos)
                action_sequence.append(('new', current_pos))
                hex_costs.append((seg_food, departure_reward, 1))
            
            for h in added:
                t = _terrain(*h)
                terrain_step_cost = t.get('step', 1)
                
                if h in self.all_visited_hexes or h in segment_hexes:
                    revisit_cost = 10
                    # Apply G/g reduction to revisit cost
                    revisit_cost = _apply_g_reduction(revisit_cost, _all_g_done())
                    seg_food += revisit_cost
                    seg_award += 0
                    seg_steps += 0
                    jumps.append(h)  # Track this jump
                    action_sequence.append(('jump', h))  # Track in action order
                    hex_costs.append((revisit_cost, 0, 0))
                    self._last_landing_cost_breakdown[h] = {
                        'challenge': 0,
                        'movement': 0,
                        'revisit': revisit_cost,
                        'total': revisit_cost,
                    }
                elif terrain_step_cost <= 0:
                    terrain_food = _get_terrain_food(t, self.current_day)
                    terrain_reward = t['award']
                    # Apply Z bonus if active
                    if self.active_team.z_bonus_remaining > 0:
                        terrain_reward = _apply_z_bonus(terrain_reward, self.active_team)
                        self.active_team.z_bonus_remaining -= 1
                    # Apply B discount first, then G/g reduction
                    # Apply B discount and G reduction additively to challenge food
                    challenge_cost = _apply_challenge_discounts(terrain_food, self.active_team, _all_g_done(), is_tent=t.get('name') == 'Tent')
                    if terrain_food >= 0 and self.active_team.b_discount_remaining > 0:
                        self.active_team.b_discount_remaining -= 1
                    movement_cost = _apply_g_reduction(50, _all_g_done())
                    cost = movement_cost + challenge_cost
                    seg_food += cost
                    seg_award += terrain_reward
                    seg_steps += terrain_step_cost
                    if h in self.all_g_lands:
                        g_taken_in_seg.add(h)
                    new_hexes.append(h)
                    action_sequence.append(('new', h))  # Track in action order
                    segment_hexes.add(h)
                    hex_costs.append((cost, terrain_reward, terrain_step_cost))
                    self._last_landing_cost_breakdown[h] = {
                        'challenge': challenge_cost,
                        'movement': movement_cost,
                        'revisit': 0,
                        'total': cost,
                    }
                elif free_exploration:
                    exploration_cost = 50
                    # Apply G/g reduction to exploration cost
                    exploration_cost = _apply_g_reduction(exploration_cost, _all_g_done())
                    seg_food += exploration_cost
                    seg_award += 0
                    seg_steps += 0
                    # Probing a G/g land already counts it as visited (see
                    # _rebuild_shared_derived_state_from_segments).
                    if h in self.all_g_lands:
                        g_taken_in_seg.add(h)
                    exploration_hexes.append(h)
                    segment_hexes.add(h)
                    hex_costs.append((exploration_cost, 0, 0))
                    self._last_landing_cost_breakdown[h] = {
                        'challenge': 0,
                        'movement': exploration_cost,
                        'revisit': 0,
                        'total': exploration_cost,
                    }
                else:
                    challenge_food = _get_terrain_food(t, self.current_day)
                    # Apply B discount and G reduction additively to challenge food
                    challenge_with_all_discounts = _apply_challenge_discounts(challenge_food, self.active_team, _all_g_done(), is_tent=t.get('name') == 'Tent')
                    if challenge_food >= 0 and self.active_team.b_discount_remaining > 0:
                        self.active_team.b_discount_remaining -= 1
                    terrain_reward = t['award']
                    # Apply Z bonus if active
                    if self.active_team.z_bonus_remaining > 0:
                        terrain_reward = _apply_z_bonus(terrain_reward, self.active_team)
                        self.active_team.z_bonus_remaining -= 1
                    # Apply G/g reduction: movement cost (50) always reduced
                    movement_cost = _apply_g_reduction(50, _all_g_done())
                    total_cost = movement_cost + challenge_with_all_discounts
                    seg_food += total_cost
                    seg_award += terrain_reward
                    seg_steps += terrain_step_cost
                    if h in self.all_g_lands:
                        g_taken_in_seg.add(h)
                    new_hexes.append(h)
                    action_sequence.append(('new', h))  # Track in action order
                    segment_hexes.add(h)
                    hex_costs.append((total_cost, terrain_reward, terrain_step_cost))
                    self._last_landing_cost_breakdown[h] = {
                        'challenge': challenge_with_all_discounts,
                        'movement': movement_cost,
                        'revisit': 0,
                        'total': total_cost,
                    }
            
            seg_start_day = self.current_day
            # Attribute segment to the current viewing day, not max_day_reached
            # This ensures proper step carryover calculation across days
            seg_day = self.current_day

            # X buffs waive step cost for upcoming new lands, but their zero-cost
            # values are applied to the segment below. Mirror that relief here so
            # a team with 0 displayed steps can still make an X-buffed move.
            step_cost_relief = 0
            if self.active_team.x_bonus_remaining > 0:
                x_bonus_used = 0
                hex_costs_offset = 1 if is_leaving_exploration else 0
                for i, hex_pos in enumerate(added):
                    if hex_pos not in new_hexes or x_bonus_used >= self.active_team.x_bonus_remaining:
                        continue
                    cost_idx = hex_costs_offset + i
                    if cost_idx < len(hex_costs):
                        step_cost_relief += max(0, hex_costs[cost_idx][2])
                        x_bonus_used += 1
            steps_required_for_check = max(0, seg_steps - step_cost_relief)
            
            # Track the maximum day reached by this team (for auto-display switching)
            if seg_day > self.active_team.max_day_reached:
                self.active_team.max_day_reached = seg_day
            
            # If moving to a new day, restore team.steps to 6 (capped at 18)
            # This handles the case where we navigate to a new day and then make a move
            if self.active_team._seg_days:
                last_seg_day = self.active_team._seg_days[-1]
                if seg_day > last_seg_day:
                    # Transitioning to a new day - restore 6 steps (capped at 18)
                    self.active_team.steps = min(self.active_team.steps + 6, 18)
            
            needs_day_check_for_departure = is_leaving_exploration
            debug_msg = f'Move: seg_food={seg_food}, seg_steps={seg_steps}, current_food={self.current_food}, team_steps={self.active_team.steps}'
            
            if free_exploration and not needs_day_check_for_departure:
                # Check if resources are insufficient
                allow_borrow_edit_resources = self._enclosure_mode or (
                    self._is_active_day_edit() and seg_day == self._day_edit_context['day']
                )
                if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                    raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_available_for_day))
            elif free_exploration and needs_day_check_for_departure:
                # Check if resources are insufficient
                allow_borrow_edit_resources = self._enclosure_mode or (
                    self._is_active_day_edit() and seg_day == self._day_edit_context['day']
                )
                if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                    raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_available_for_day))
            else:
                # Check if resources are insufficient
                allow_borrow_edit_resources = self._enclosure_mode or (
                    self._is_active_day_edit() and seg_day == self._day_edit_context['day']
                )
                if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                    raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_available_for_day))
                if (not allow_borrow_edit_resources) and steps_available_for_day < steps_required_for_check:
                    raise RuntimeError(f'Not enough steps! Need {steps_required_for_check}, have {steps_available_for_day}. Please click "Next Day" button to advance.')

            # Apply X bonus: for new hexes covered by bonus, reduce their step costs to 0 BEFORE storing
            if self.active_team.x_bonus_remaining > 0:
                # hex_costs is indexed: [0] = departure (if is_leaving_exploration), then added[0], added[1], ...
                hex_costs_offset = 1 if is_leaving_exploration else 0
                bonus_applied = 0
                
                # Apply bonus to new hexes
                for i, hex_pos in enumerate(added):
                    if hex_pos in new_hexes and bonus_applied < self.active_team.x_bonus_remaining:
                        cost_idx = hex_costs_offset + i
                        if cost_idx < len(hex_costs):
                            food, reward, step_cost = hex_costs[cost_idx]
                            seg_steps -= step_cost  # Remove original step cost
                            hex_costs[cost_idx] = (food, reward, 0)  # Mark as zero cost
                            bonus_applied += 1
                
                self.active_team.x_bonus_remaining -= bonus_applied

            # Apply move
            self.active_team.full_path.extend(added)
            self.active_team._seg_lengths.append(len(added))
            self.active_team._seg_foods.append(seg_food)
            self.active_team._seg_awards.append(seg_award)
            self.active_team._seg_steps.append(seg_steps)
            self.active_team._seg_days.append(seg_day)
            self.active_team._seg_new_hexes.append(new_hexes.copy())
            self.active_team._seg_exploration_hexes.append(exploration_hexes.copy())
            self.active_team._seg_jumps.append(jumps.copy())
            self.active_team._seg_path_nodes.append(added.copy())
            self.active_team._seg_end_positions.append(added[-1] if added else current_pos)
            self.active_team._seg_action_sequence.append(action_sequence.copy())
            self._append_segment_action_order(self.active_team)
            self.active_team._seg_hex_costs.append(hex_costs.copy())
            self.active_team._seg_is_fly_skill.append(False)  # Normal pathfinding, not fly skill
            self.active_team._seg_fly_skill_deltas.append(0)
            
            if not self._enclosure_mode:
                self.current_food -= seg_food
                self.active_team.steps -= seg_steps
                self.total_food += seg_food
                self.total_reward += seg_award
            
            self.active_team.visited_hexes.update(new_hexes)
            self.all_visited_hexes.update(new_hexes)
            
            # Track G/g land visits for global food reduction
            for h in new_hexes:
                if h in self.all_g_lands:
                    self.visited_g_lands.add(h)
            
            if exploration_hexes:
                self.active_team.free_exploration_hexes.update(exploration_hexes)
                self.active_team.visited_hexes.update(exploration_hexes)
                # Track G/g land visits in exploration hexes too
                for h in exploration_hexes:
                    if h in self.all_g_lands:
                        self.visited_g_lands.add(h)
            
            # Unconditional: also drops a stale entry for a hex another team settled.
            self.active_team.free_exploration_hexes.discard(current_pos)

            portal_status = self._check_portal_teleport()
            if self.active_team._seg_end_positions:
                self.active_team._seg_end_positions[-1] = self.active_team.full_path[-1]
            
            # Check for X1, X2, X3 bonus steps AFTER portal teleport (based on final position)
            # Only activate on NEW hexes
            final_pos = self.active_team.full_path[-1]
            final_terrain_name = RAW_MAP[final_pos[0]][final_pos[1]]
            bonus_steps = 0
            if final_terrain_name == 'X1':
                bonus_steps = 5
            elif final_terrain_name == 'X2':
                bonus_steps = 8
            elif final_terrain_name == 'X3':
                bonus_steps = 10
            
            if bonus_steps > 0 and final_pos in new_hexes:
                self.active_team.x_bonus_remaining = bonus_steps
                self.active_team.x_bonus_name = final_terrain_name
                self._status_msg = f'{self._status_msg} +{bonus_steps} free movements ({final_terrain_name})!'
            
            # B/Z charges were already decremented one at a time, right where
            # each was actually used (per hex, in the loop above) - just
            # report if either ran out during this segment.
            if b_discount_before_seg > 0 and self.active_team.b_discount_remaining == 0:
                self._status_msg = f'{self._status_msg} B discount expired.'
            if z_bonus_before_seg > 0 and self.active_team.z_bonus_remaining == 0:
                self._status_msg = f'{self._status_msg} Z reward bonus expired.'
            
            # Check for bigBoss hex - increment fly skill if visiting for first time
            for hex_pos in new_hexes:
                hex_terrain = _terrain(*hex_pos)
                if hex_terrain.get('name') == 'bigBoss':
                    self.fly_skill_limit += 1
                    if self.active_team._seg_fly_skill_deltas:
                        self.active_team._seg_fly_skill_deltas[-1] += 1
                    self._status_msg = f'{self._status_msg} ⭐ Reached BigBoss! Fly skill +1 (now {self.fly_skill_limit})'
                    break  # Only count one bigBoss per segment

            # A hex settled on departure (probe / deferred fly landing) is captured
            # now, so its X/B/Z buff starts now.
            if is_leaving_exploration:
                self._status_msg = f'{self._status_msg}{self._activate_land_buffs(current_pos)}'

            # Check for B1, B2, B3 bonus food discount (activate only if landing on NEW B hex)
            if final_terrain_name in ('B1', 'B2', 'B3') and final_pos in new_hexes:
                self.active_team.b_discount_remaining = B_DISCOUNT_MAP.get(final_terrain_name, 5)
                self.active_team.b_discount_name = final_terrain_name
                discount_rate = _TERRAIN_DB.get(final_terrain_name, {}).get('food_discount_rate', 0.4)
                self._status_msg = f'{self._status_msg} {int(discount_rate*100)}% food discount active for next {self.active_team.b_discount_remaining} movements!'
            
            # Check for Z1, Z2, Z3 bonus reward (activate only if landing on NEW Z hex)
            if final_terrain_name in ('Z1', 'Z2', 'Z3') and final_pos in new_hexes:
                self.active_team.z_bonus_remaining = Z_BONUS_MAP.get(final_terrain_name, 5)
                self.active_team.z_bonus_name = final_terrain_name
                bonus_rate = _TERRAIN_DB.get(final_terrain_name, {}).get('reward_bonus_rate', 1.4)
                self._status_msg = f'{self._status_msg} {int((bonus_rate-1)*100)}% reward bonus active for next {self.active_team.z_bonus_remaining} movements!'
            
            # If team has reached a later day, auto-switch display to that day
            if self.active_team.max_day_reached > self.current_day:
                self.current_day = self.active_team.max_day_reached
                self._status_msg = f'{self._status_msg} [Auto-switched to Day {self.current_day}]'
            
            # Rebuild day records AFTER all bonuses are applied and day is correct.
            #
            # Deliberately NOT calling _rebalance_all_teams_from_day() here on every
            # intermediate click of a day-edit redraw: that function rebalances ALL
            # 3 teams' segment-day assignments from scratch, and while a redraw is
            # still in progress the active team's day total is incomplete, so
            # rebalancing mid-redraw was reshuffling the OTHER two teams' segments
            # (who aren't even being edited) based on a partial view of the day's
            # cost - and each subsequent click reshuffled them again from a
            # different partial state, so the final result didn't reliably converge
            # back to the original arrangement even when the redrawn route was
            # identical to the one that was deleted. _finalize_day_segment_edit_if_
            # connected() below already calls _rebalance_all_teams_from_day() once,
            # after the full redraw is complete and reconnected - that single call
            # is sufficient and stable.
            if not self._enclosure_mode:
                self._rebuild_day_records()

            if self._is_active_day_edit():
                self._finalize_day_segment_edit_if_connected()

            self._auto_save_game()  # Auto-save after normal pathfinding
            
            self._draw()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._status_msg = f'Error applying move: {str(e)} (see terminal for details)'
            print(f'CRASH DETAILS:\n{error_details}')
            self._draw()

    def _on_hscroll_changed(self, val):
        """Handle horizontal scrollbar change."""
        if self._updating_scrollbar or self._default_xlim is None:
            return
        
        self._updating_scrollbar = True
        
        # Map scrollbar value (0 to 1) to axis limits
        cur_width = self._default_xlim[1] - self._default_xlim[0]
        cur_xlim = self.ax.get_xlim()
        zoomed_width = cur_xlim[1] - cur_xlim[0]
        
        # Position: 0 = leftmost, 1 = rightmost
        left_limit = self._default_xlim[0] + val * (cur_width - zoomed_width)
        right_limit = left_limit + zoomed_width
        
        self.ax.set_xlim([left_limit, right_limit])
        self.fig.canvas.draw_idle()
        
        self._updating_scrollbar = False

    def _on_vscroll_changed(self, val):
        """Handle vertical scrollbar change."""
        if self._updating_scrollbar or self._default_ylim is None:
            return
        
        self._updating_scrollbar = True
        
        # Map scrollbar value (0 to 1) to axis limits
        cur_height = self._default_ylim[1] - self._default_ylim[0]
        cur_ylim = self.ax.get_ylim()
        zoomed_height = cur_ylim[1] - cur_ylim[0]
        
        # Position: 0 = bottom, 1 = top (but scrollbar goes 1 = bottom, 0 = top, so invert)
        bottom_limit = self._default_ylim[0] + (1 - val) * (cur_height - zoomed_height)
        top_limit = bottom_limit + zoomed_height
        
        self.ax.set_ylim([bottom_limit, top_limit])
        self.fig.canvas.draw_idle()
        
        self._updating_scrollbar = False

    def _check_portal_teleport(self):
        """Check if active team is on a portal and offer teleportation.
        
        Portals in the CSV are labeled P1, P2, P3 and come in pairs.
        Both the current portal and destination portal (if teleported) are marked as taken.
        Returns True if teleportation occurred, False otherwise.
        """
        current_pos = self.active_team.full_path[-1]
        current_ir, current_ic = current_pos
        portal_entry_pos = current_pos
        
        # Get the raw CSV value to check if it's a portal
        if not (0 <= current_ir < ROWS and 0 <= current_ic < COLS):
            return False
        
        portal_type = RAW_MAP[current_ir][current_ic]
        
        print(f'DEBUG: Current hex ({current_ir},{current_ic}) has CSV value: {portal_type}')  # DEBUG
        cost_dbg = self._last_landing_cost_breakdown.get(current_pos)
        if cost_dbg is not None:
            print(
                f"DEBUG: Charged at ({current_ir},{current_ic}) [{portal_type}] -> "
                f"challenge={cost_dbg['challenge']}, movement={cost_dbg['movement']}, "
                f"revisit={cost_dbg['revisit']}, total={cost_dbg['total']}"
            )
        
        # Check if it's a portal (P1, P2, P3, P4, P5, P6, P7, P8)
        if portal_type not in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'):
            return False  # Not a portal
        
        print(f'DEBUG: Found portal type {portal_type}')  # DEBUG
        
        # Mark this portal as taken (visited) by the team and globally
        self.active_team.visited_hexes.add(current_pos)
        self.all_visited_hexes.add(current_pos)
        # Track G/g land visits
        if current_pos in self.all_g_lands:
            self.visited_g_lands.add(current_pos)
        print(f'DEBUG: Marked current portal {current_pos} as taken')  # DEBUG
        
        # Find all portals of the same type
        other_portals = []
        for ir in range(ROWS):
            for ic in range(COLS):
                if RAW_MAP[ir][ic] == portal_type and (ir, ic) != current_pos:
                    other_portals.append((ir, ic))
                    print(f'DEBUG: Found paired portal {portal_type} at ({ir}, {ic})')  # DEBUG
        
        if not other_portals:
            print(f'DEBUG: No paired portal found!')  # DEBUG
            return False  # No other portal found (shouldn't happen)
        
        # There should be exactly one other portal (they come in pairs)
        other_portal = other_portals[0]
        
        # Show message box using tkinter
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            print(f'DEBUG: Attempting to show messagebox...')  # DEBUG
            
            # Create root window
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            
            msg = f"You've reached a {portal_type} portal!\n\nTeleport to the other side?"
            result = messagebox.askyesno("Portal Teleportation", msg)
            
            print(f'DEBUG: User response: {result}')  # DEBUG
            root.destroy()
            
            if result:
                # Teleport to other portal
                # The edge to skip is from the position BEFORE the current portal to the destination
                if len(self.active_team.full_path) >= 2:
                    prev_pos = self.active_team.full_path[-2]
                    self.active_team.full_path[-1] = other_portal
                    
                    # Mark the edge that will be drawn (from prev to destination) as no-draw
                    self.active_team._no_draw_edges.add((prev_pos, other_portal))
                else:
                    # Edge case: teleporting from first move
                    self.active_team.full_path[-1] = other_portal
                
                # Check if destination is new BEFORE marking
                is_portal_dest_new = other_portal not in self.all_visited_hexes
                
                # Mark BOTH portals as taken
                self.active_team.visited_hexes.add(other_portal)
                self.all_visited_hexes.add(other_portal)
                # Track G/g land visits
                if other_portal in self.all_g_lands:
                    self.visited_g_lands.add(other_portal)
                print(f'DEBUG: Teleported to {other_portal} and marked as taken')  # DEBUG

                # Keep fly segment canonical raw target at portal ENTRY so fly curve points to begin portal.
                if self.active_team._seg_is_fly_skill and self.active_team._seg_is_fly_skill[-1]:
                    if self.active_team._seg_path_nodes:
                        self.active_team._seg_path_nodes[-1] = [portal_entry_pos]
                
                # Portal interaction counts as a movement for B discount
                final_portal_dest = other_portal
                teleport_msg = f'✓ Teleported via {portal_type}!'
            else:
                is_portal_dest_new = False  # Staying on portal doesn't count as new destination
                final_portal_dest = current_pos
                teleport_msg = f'✓ Stayed on {portal_type} portal (marked as taken).'
                print(f'DEBUG: User declined teleportation, portal marked as taken')  # DEBUG
            
            # Portal interaction always counts as 1 movement
            if self.active_team.b_discount_remaining > 0:
                self.active_team.b_discount_remaining -= 1
                if self.active_team.b_discount_remaining == 0:
                    teleport_msg += ' B discount expired.'
            
            # Portal interaction always counts as 1 movement for Z bonus
            if self.active_team.z_bonus_remaining > 0:
                self.active_team.z_bonus_remaining -= 1
                if self.active_team.z_bonus_remaining == 0:
                    teleport_msg += ' Z reward bonus expired.'
            
            # Check for B1, B2, B3 bonus food discount (activate only if destination is NEW)
            final_terrain_name = RAW_MAP[final_portal_dest[0]][final_portal_dest[1]]
            if final_terrain_name in ('B1', 'B2', 'B3') and is_portal_dest_new:
                self.active_team.b_discount_remaining = B_DISCOUNT_MAP.get(final_terrain_name, 5)
                self.active_team.b_discount_name = final_terrain_name
                discount_rate = _TERRAIN_DB.get(final_terrain_name, {}).get('food_discount_rate', 0.4)
                teleport_msg += f' {int(discount_rate*100)}% food discount active for next {self.active_team.b_discount_remaining} movements!'
            
            # Check for Z1, Z2, Z3 bonus reward (activate only if destination is NEW)
            if final_terrain_name in ('Z1', 'Z2', 'Z3') and is_portal_dest_new:
                self.active_team.z_bonus_remaining = Z_BONUS_MAP.get(final_terrain_name, 5)
                self.active_team.z_bonus_name = final_terrain_name
                bonus_rate = _TERRAIN_DB.get(final_terrain_name, {}).get('reward_bonus_rate', 1.4)
                teleport_msg += f' {int((bonus_rate-1)*100)}% reward bonus active for next {self.active_team.z_bonus_remaining} movements!'
            
            self._status_msg = teleport_msg
            print(f'DEBUG: Teleported to {final_portal_dest}')  # DEBUG
            self._center_view_on_active_team()
            return True
        except ImportError as e:
            print(f'ERROR: tkinter not available: {e}')
            self._status_msg = 'Portal found but tkinter unavailable. Check terminal.'
            return False
        except Exception as e:
            import traceback
            print(f'ERROR in portal teleportation: {e}')
            print(traceback.format_exc())
            self._status_msg = f'Portal error: {str(e)}'
            return False

    def _show_global_stat_window(self):
        """Show global map statistics popup window."""
        try:
            self._ensure_global_stat_window()
            self._draw_global_stat_window()
            if hasattr(self, 'global_stat_fig') and self.global_stat_fig is not None:
                self.global_stat_fig.show()
                self._global_stat_window_open = True
                self.global_stat_fig.canvas.draw_idle()
        except Exception as e:
            print(f'Error showing global stat window: {e}')


if __name__ == '__main__':
    print(f'Map: {ROWS} rows × {COLS} cols')
    start = _find_start_position()
    print(f'Resolved start: ir={start[0]}, ic={start[1]}')
    print('Launching interactive window…')
    PathfindingDemo()
