"""hex_pathfinding_demo.py
Interactive A* pathfinding demo on the Naruto hex marching map.
Reference: https://www.redblobgames.com/grids/hexagons/

Day-by-day resource system:
  - Day 1 starts with 800 food and 6 steps
  - Each following day: +1600 food, +6 steps (max 18 steps at day start)
  - Unused resources roll to next day
  - If move exceeds available food or steps, it happens next day

Controls
--------
  Click hexes to extend the path. Each segment costs food (based on terrain)
  and steps (counted as new hexes visited). Steps don't count revisits.
  
  [Undo Last]  – remove the previous move
  [Reset Path] – return to origin, clear all stats
"""

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

# Disable matplotlib's default 'q' quit key to use Q for day navigation
matplotlib.rcParams['keymap.quit'] = []

# Hide the default TkAgg navigation toolbar (home/back/forward/pan/zoom/
# subplot-config/save icons) and its coordinate status bar under every
# window - the app has its own on-canvas buttons for pan/zoom/save, and the
# toolbar's own pan/zoom-mode buttons would otherwise conflict with the
# custom mouse handling (right-click drag to pan, scroll to zoom, etc.).
matplotlib.rcParams['toolbar'] = 'None'

# Prefer installed CJK fonts so Chinese text does not render as squares.
try:
    from matplotlib import font_manager as _fm

    _cjk_candidates = [
        'Microsoft YaHei',
        'SimHei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'PingFang SC',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS',
    ]
    _installed_font_names = {f.name for f in _fm.fontManager.ttflist}
    _picked_cjk_fonts = [n for n in _cjk_candidates if n in _installed_font_names]
    if _picked_cjk_fonts:
        matplotlib.rcParams['font.sans-serif'] = _picked_cjk_fonts + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle, Circle, Polygon
from matplotlib.collections import PolyCollection
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button
from matplotlib.colors import hex2color
import numpy as np
import csv
import json
import heapq
import threading
import time
import re
import ctypes
import struct
from datetime import date, timedelta

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(path):
    with open(path, newline='') as f:
        return [row for row in csv.reader(f)]


with open('landInfo.json') as f:
    _TERRAIN_DB = json.load(f)

_MAP_FILES = _TERRAIN_DB['map_files']
_SEASON = _TERRAIN_DB.get('season', {})
TOTAL_DAYS = _SEASON.get('duration_days', 90)
CALENDAR_DAY1 = date(*(int(p) for p in _SEASON.get('start_date', '2026-06-12').split('-')))

_DEFAULT_TEAM_COLORS = {
    1: '#000000',
    2: '#FF0000',
    3: '#87CEFA',
}
TEAM_COLORS = {
    team_num: _TERRAIN_DB.get('team_colors', {}).get(str(team_num), default_color)
    for team_num, default_color in _DEFAULT_TEAM_COLORS.items()
}
APP_BACKGROUND_COLOR = '#4C4A55'

# Load BXZ buff parameters from landInfo.json
def _load_buff_parameters():
    """Load B, X, Z buff parameters from landInfo.json"""
    b_discount_map = {}
    x_bonus_map = {}
    z_bonus_map = {}
    
    for key in ['B1', 'B2', 'B3']:
        if key in _TERRAIN_DB:
            b_discount_map[key] = _TERRAIN_DB[key].get('effective_steps', 5)
    
    for key in ['X1', 'X2', 'X3']:
        if key in _TERRAIN_DB:
            x_bonus_map[key] = _TERRAIN_DB[key].get('effective_steps', 5)
    
    for key in ['Z1', 'Z2', 'Z3']:
        if key in _TERRAIN_DB:
            z_bonus_map[key] = _TERRAIN_DB[key].get('effective_steps', 5)
    
    return b_discount_map, x_bonus_map, z_bonus_map

B_DISCOUNT_MAP, X_BONUS_MAP, Z_BONUS_MAP = _load_buff_parameters()

RAW_MAP = _load_csv(_MAP_FILES['csv'])
RAW_MAP.reverse()          # row 0 → bottom of the board
ROWS = len(RAW_MAP)
COLS = len(RAW_MAP[0])

HEX_SIZE = 5
# Display-only angled-view transform: stretch x wider, compress y
X_SCALE = 1
Y_SCALE = 0.6

# Unit hexagon vertex offsets - flat-top orientation (vertices at 0/60/120/.../300
# degrees), matching redblobgames.com/grids/hexagons/ "flat topped" layout and the
# original RegularPolygon(numVertices=6, orientation=radians(30)) output (matplotlib's
# RegularPolygon starts its own vertex 0 at 90 degrees *before* adding `orientation`,
# so orientation=30 deg there lands on the same {0,60,...,300} vertex set as the plain
# 0-based angles used here - no extra offset needed).
# Precomputed once so per-hex drawing only needs a cheap (cx, cy) translation instead
# of building a new patch/transform object per hex.
_HEX_VERT_ANGLES = np.arange(6) * (2 * np.pi / 6)
_HEX_VERT_OFFSETS = np.column_stack((
    np.cos(_HEX_VERT_ANGLES), np.sin(_HEX_VERT_ANGLES),
)) * (HEX_SIZE * 0.97)

# Real-game screenshot for the "切换地图" view. S24_map_rectified.png is a
# pre-processed (offline, not at app runtime - see tools/rectify_map_image.py)
# version of the raw S24_map.png screenshot: the script self-calibrates a
# single rigid affine mapping (ir, ic) hex coords <-> source pixel coords
# (connected-component blob detection, anchored on the unique 'ST' start
# hex, refined by iterative least-squares), then places the WHOLE original
# screenshot (only downsampled for size, never cropped or masked) using that
# affine. The resulting placement extent (in this module's data-coordinate
# space) is saved alongside the PNG as a JSON sidecar since - unlike a
# per-hex warp resampled onto the exact grid - a single affine's image
# bounds generally don't equal the full hex-grid bounds.
_MAP_IMAGE_FILENAME = _MAP_FILES['image']
_MAP_IMAGE_EXTENT_FILENAME = _MAP_FILES['image_extent']

# Marching game start: game coords (row=53, col=8) → internal 0-based (ir, ic)
# NOTE: the raw game-start cell may be empty terrain (a boundary marker);
# we resolve the nearest passable hex at runtime in _find_passable_near().
_GAME_START_IR = ROWS - 53   # e.g. 60-53 = 7
_GAME_START_IC = 8 - 1       # e.g. 7


# ── Terrain / geometry helpers ────────────────────────────────────────────────

def _terrain(ir, ic):
    return _TERRAIN_DB.get(RAW_MAP[ir][ic], _TERRAIN_DB['default'])


def _get_terrain_food(terrain_data, current_day):
    """Get the effective food value for terrain, accounting for a data-driven
    'degrade' schedule (e.g. Tent) defined entirely in landInfo.json - both the
    day thresholds AND the resulting food value at each stage live in the JSON,
    nothing about the schedule is hardcoded here.

    'degrade' is a list of {"day": N, "food": V} entries, each meaning "starting
    Day N, this terrain's food value becomes V". The terrain's base 'food' value
    applies before the first entry's day. If current_day matches multiple
    entries, the one with the largest 'day' <= current_day wins (list order in
    the JSON doesn't matter).

    Example: Tent with food=-300 and degrade=[{day:11,food:-250}, {day:21,food:-200},
    {day:36,food:-150}, {day:51,food:-100}]:
      Days 1-10: -300
      Days 11-20: -250
      Days 21-35: -200
      Days 36-50: -150
      Days 51+: -100
    """
    base_food = terrain_data.get('food', 0)
    schedule = terrain_data.get('degrade', [])

    if not schedule:
        return base_food

    applicable = [entry for entry in schedule if current_day >= entry['day']]
    if not applicable:
        return base_food

    return max(applicable, key=lambda entry: entry['day'])['food']


def _apply_b_discount(challenge_food, team):
    """Apply B1/B2/B3 discount if team has discount active.
    
    Discount rate is loaded from landInfo.json (default 40% off = keep 60% of cost).
    Returns the discounted food value (rounded properly).
    """
    if team.b_discount_remaining > 0:
        # Get discount rate from landInfo, default to 0.4 (40% discount)
        b_discount_rate = _TERRAIN_DB.get(team.b_discount_name, {}).get('food_discount_rate', 0.4)
        # Apply discount: multiply by (1 - discount_rate)
        return round(challenge_food * (1 - b_discount_rate))
    return challenge_food


def _apply_g_reduction(food_value, all_g_lands_visited):
    """Apply G/g land global reduction (20% off all food) if all 8 G/g lands are visited.
    
    Returns the reduced food value (rounded properly).
    """
    if all_g_lands_visited:
        # 20% reduction = keep 80% of cost = multiply by 0.8
        return round(food_value * 0.8)
    return food_value


def _apply_challenge_discounts(challenge_food, team, all_g_lands_visited, is_tent=False):
    """Apply B discount and G reduction additively to challenge food.
    
    B discount (from landInfo.json, default 40%) and G reduction (20%) are added together:
    e.g. both active: 1 - 0.4 - 0.2 = 0.4 multiplier
    G reduction is NOT applied to Tent terrain.
    Discounts are NOT applied when food is negative (terrain gives food as a benefit).
    """
    # Don't reduce benefits: negative food means the terrain gives food to the team.
    if challenge_food < 0:
        return challenge_food
    discount = 0.0
    if team.b_discount_remaining > 0:
        # Get discount rate from landInfo, default to 0.4 (40% discount)
        b_discount_rate = _TERRAIN_DB.get(team.b_discount_name, {}).get('food_discount_rate', 0.4)
        discount += b_discount_rate
    if all_g_lands_visited and not is_tent:
        discount += 0.2
    if discount > 0:
        return round(challenge_food * (1 - discount))
    return challenge_food


def _apply_z_bonus(reward, team):
    """Apply Z1/Z2/Z3 bonus if team has bonus active.
    
    Bonus rate is loaded from landInfo.json (default 40% additional = multiply by 1.4).
    Returns the bonus reward value (rounded properly).
    """
    if team.z_bonus_remaining > 0:
        # Get reward bonus rate from landInfo, default to 1.4 (40% bonus)
        z_bonus_rate = _TERRAIN_DB.get(team.z_bonus_name, {}).get('reward_bonus_rate', 1.4)
        return round(reward * z_bonus_rate)
    return reward


def _passable(ir, ic):
    return _terrain(ir, ic)['name'] != 'empty'


def _cost(ir, ic):
    """Movement cost to enter hex (ir, ic). Uses terrain food, minimum 1."""
    return max(1, _terrain(ir, ic)['food'])


def _center(ir, ic):
    """Pixel centre of the hex at internal (ir, ic)."""
    x = ic * 1.5 * HEX_SIZE
    y = ir * np.sqrt(3) * HEX_SIZE
    if ic % 2 == 1:
        y += np.sqrt(3) / 2 * HEX_SIZE
    return x, y


def _neighbors(ir, ic):
    """
    Six hex neighbours in internal (0-based) offset coordinates.

    The map uses a flat-top, odd-column-offset scheme where odd *internal*
    columns are shifted up in the display.  Derivation of the direction vectors
    below matches the adjacency logic in marchGame.py (game coords) converted
    to internal (ir = ROWS-r, ic = c-1) coordinates.

    ic even  →  game col odd:  dirs = [(+1,0),(0,+1),(-1,+1),(-1,0),(-1,-1),(0,-1)]
    ic odd   →  game col even: dirs = [(+1,0),(+1,+1),(0,+1),(-1,0),(0,-1),(+1,-1)]
    """
    if ic % 2 == 0:
        dirs = [(+1, 0), (0, +1), (-1, +1), (-1, 0), (-1, -1), (0, -1)]
    else:
        dirs = [(+1, 0), (+1, +1), (0, +1), (-1, 0), (0, -1), (+1, -1)]
    return [(ir + dr, ic + dc)
            for dr, dc in dirs
            if 0 <= ir + dr < ROWS and 0 <= ic + dc < COLS]


# ── Hex distance (admissible A* heuristic) ────────────────────────────────────
# From redblobgames.com/grids/hexagons/:
#   Convert offset → axial coordinates (odd-q layout):
#       q = ic,  r = ir - ic // 2
#   Then: distance = (|dq| + |dr| + |dq+dr|) / 2

def _to_axial(ir, ic):
    return ic, ir - ic // 2


def _hex_dist(ir0, ic0, ir1, ic1):
    q0, r0 = _to_axial(ir0, ic0)
    q1, r1 = _to_axial(ir1, ic1)
    dq, dr = q1 - q0, r1 - r0
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


# ── A* pathfinding ────────────────────────────────────────────────────────────

def _astar(start, goal, extra_walls):
    """
    A* on the hex grid.

    Returns
    -------
    path : list[(ir, ic)]
        Hexes from start to goal inclusive.  Empty when no path exists.
    cost_map : dict {(ir,ic): accumulated_cost}
        All nodes opened by A* (useful for visualising the search frontier).
    """
    gr, gc = goal
    heap = [(0, start)]
    came_from   = {start: None}
    cost_so_far = {start: 0}

    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            break
        for nb in _neighbors(*cur):
            if nb in extra_walls or not _passable(*nb):
                continue
            new_cost = cost_so_far[cur] + _cost(*nb)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                priority = new_cost + _hex_dist(*nb, gr, gc)
                heapq.heappush(heap, (priority, nb))
                came_from[nb] = cur

    if goal not in came_from:
        return [], cost_so_far

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, cost_so_far


# ── Resolve passable start / default goal ────────────────────────────────────

def _find_passable_near(ir, ic):
    """Return (ir, ic) itself if passable, else the nearest passable neighbour."""
    if 0 <= ir < ROWS and 0 <= ic < COLS and _passable(ir, ic):
        return (ir, ic)
    # BFS outward
    from collections import deque
    q = deque([(ir, ic)])
    seen = {(ir, ic)}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c):
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            if _passable(nr, nc):
                return (nr, nc)
            q.append((nr, nc))
    return (ir, ic)  # fallback (should never reach here on a non-trivial map)


def _find_start_position():
    """Find the start position marked with 'ST' in the map. Falls back to default if not found."""
    for ir in range(ROWS):
        for ic in range(COLS):
            if RAW_MAP[ir][ic] == 'ST':
                return _find_passable_near(ir, ic)
    # Fallback to default start position if ST is not found
    return _find_passable_near(_GAME_START_IR, _GAME_START_IC)


def _default_goal(start_ir, start_ic):
    """Scan outward from start for the first passable hex well to the right."""
    for dc in range(15, 2, -1):
        for dr in range(0, dc + 1):
            for dr_sign in (1, -1):
                r, c = start_ir + dr_sign * dr, start_ic + dc
                if 0 <= r < ROWS and 0 <= c < COLS and _passable(r, c):
                    return (r, c)
    # fallback
    for r in range(start_ir, ROWS):
        for c in range(start_ic + 3, COLS):
            if _passable(r, c):
                return (r, c)
    return (start_ir, start_ic + 3)


# ── Team class ───────────────────────────────────────────────────────────────

class Team:
    """Encapsulates state for a single team exploring the map."""
    
    def __init__(self, origin, created_day=1):
        self.origin = origin
        self.full_path = [origin]
        self._seg_lengths = []
        self._seg_foods = []
        self._seg_awards = []
        self._seg_steps = []
        self._seg_days = []
        self._seg_new_hexes = []      # Track new hexes per segment (for undo)
        self._seg_exploration_hexes = []  # Track exploration hexes per segment (for undo)
        self._seg_jumps = []           # Track revisited hexes (jumps) per segment
        self._seg_path_nodes = []      # Canonical added path nodes per segment (future edit/replay support)
        self._seg_end_positions = []   # Final segment endpoint after portal/teleport resolution
        self._seg_action_sequence = [] # Track order of actions: [('new', hex), ('jump', hex), ...]
        self._seg_action_orders = []  # Cross-team sequence number for right-side symbol ordering
        self._seg_hex_costs = []  # Track per-hex costs [food_per_hex, reward_per_hex, ...] for proper allocation
        self._seg_is_fly_skill = []  # Track which segments are fly skill moves (for undo)
        self._seg_fly_skill_deltas = []  # Per-segment delta applied to global fly skill limit
        self.steps = 6  # Team's step balance (independent per team)
        self.created_day = created_day  # Day this team was created
        self.max_day_reached = created_day  # Maximum day this team has reached (for proper land attribution)
        self.visited_hexes = {origin}  # Hexes visited by THIS team only
        self.free_exploration_hexes = set()  # Hexes explored via free exploration (0 steps, not taken globally)
        self._no_draw_edges = set()  # Edges (from_pos, to_pos) not to draw (e.g., portal teleports)
        # B1/B2/B3 food discount effect
        self.b_discount_remaining = 0  # Movements remaining with food discount (0 = no discount active)
        self.b_discount_name = None  # Which B hex triggered this ('B1', 'B2', 'B3')
        # X1/X2/X3 free movement effect
        self.x_bonus_remaining = 0  # Movements remaining with no step cost (0 = no bonus active)
        self.x_bonus_name = None  # Which X hex triggered this ('X1', 'X2', 'X3')
        # Z1/Z2/Z3 reward bonus effect
        self.z_bonus_remaining = 0  # Movements remaining with 40% reward bonus (0 = no bonus active)
        self.z_bonus_name = None  # Which Z hex triggered this ('Z1', 'Z2', 'Z3')


# ── Interactive demo ──────────────────────────────────────────────────────────

class PathfindingDemo:
    """
    Three teams exploring the map independently.
    Each team has its own path, resources, and day tracker.
    """

    def __init__(self):
        # Team 1 starts at the start position marked with 'ST' in the map
        team1_origin = _find_start_position()
        self.team1 = Team(team1_origin, created_day=1)
        self.team2 = None  # Will be set when user clicks "Set Team 2 Start"
        self.team3 = None  # Will be set when user clicks "Set Team 3 Start"
        
        self.active_team = self.team1  # Currently active team for path building
        self.set_start_mode = None  # 'team2' or 'team3' when setting starting points
        self._fly_mode = False  # True when fly skill is active
        self._fly_button_timer = None  # Timer for flashing animation
        self._fly_button_flash_state = False  # Current flash state (on/off)
        self._breathing_timer = None  # Timer for breathing animation on active team marker
        self._breathing_phase = 0.0  # Tracks animation time for breathing effect
        self.fly_skill_limit = 1  # Global fly skill limit shared by all teams (starts at 1, increases when taking bigBoss)
        self._next_action_order = 0
        
        self._hover_timer = None  # Timer for hover preview
        self._hover_hex = None  # Current hex being hovered
        self._hover_path_line = None  # Line object for preview path
        self._last_landing_cost_breakdown = {}  # {(ir, ic): {'challenge': x, 'movement': y, 'revisit': z, 'total': t}}
        
        # Pan state (right-click drag to move map)
        self._pan_active = False  # True when right-click drag is active
        self._pan_start_x = None  # Starting x position for pan
        self._pan_start_y = None  # Starting y position for pan
        self._pan_start_xlim = None  # Map x-axis limits at pan start
        self._pan_start_ylim = None  # Map y-axis limits at pan start
        self._pan_transform = None  # Data<->pixel transform frozen at pan start
        
        self._show_bonus_labels = True  # Toggle for showing B/X/Z bonus hex labels
        
        self._status_msg = ''
        self._has_zoomed = False  # Track if user has zoomed the view
        
        # Scrollbar state (replaces drag panning)
        self._default_xlim = None
        self._default_ylim = None
        self._updating_scrollbar = False  # Flag to prevent feedback loops
        self._view_anim_timer = None  # Timer for smooth center-view transitions
        
        # Find all G/g lands on the map (for 20% food reduction when all visited)
        self.all_g_lands = set()  # Positions of all G/g lands
        for ir in range(ROWS):
            for ic in range(COLS):
                cell_value = RAW_MAP[ir][ic]
                if cell_value in ('G', 'g'):
                    self.all_g_lands.add((ir, ic))
        
        # Shared state across all teams
        self.all_visited_hexes = {team1_origin}  # All hexes visited by any team (for visualization/counting)
        self.visited_g_lands = set()  # G/g lands that have been visited by any team
        
        # TEST MODE: Uncomment the next line to activate G/g reduction from the start
        # self.visited_g_lands = self.all_g_lands.copy()  # PRE-ACTIVATE G/G REDUCTION FOR TESTING
        
        # Check if team1_origin is a G/g land
        if team1_origin in self.all_g_lands:
            self.visited_g_lands.add(team1_origin)
        self.current_day = 1
        self.current_food = 6800  # Day 1 starts with 6800 food, subsequent days +1600
        self.total_food = 0     # Total food consumed across all teams
        self.total_reward = 0   # Total reward across all teams
        self._calendar_day1 = CALENDAR_DAY1  # Day 1 baseline date (from landInfo.json)
        self._data_window_open = False
        self._global_stat_window_open = False
        self._map_view_mode = 'hex'  # 'hex' or 'image' (real-game screenshot)
        self._map_image_array = None  # lazily loaded/cached on first switch to image mode
        self._map_image_extent = None
        self._segment_edit_mode = False
        self._segment_edit_targets = []
        self._segment_edit_selected_days = set()
        self._segment_edit_focus_seg_idx = None
        self._edit_seg_button_timer = None
        self._edit_seg_button_flash_state = False
        self._day_edit_context = None
        self._show_future_paths = True
        
        # Initialize day records for days 1-TOTAL_DAYS with proper remaining steps
        self._init_day_records()
        
        # Team colors
        self.team_colors = dict(TEAM_COLORS)

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(APP_BACKGROUND_COLOR)
        self.fig.canvas.manager.set_window_title(
            '微信157-亡夜迫邪-远征模拟器S25')
        
        # Create main map axes - enlarged so map uses more of the window area.
        self.ax = self.fig.add_axes([0.01, 0.05, 0.905, 0.93])
        self.ax.set_facecolor(APP_BACKGROUND_COLOR)
        
        # Create separate data window figure
        self.data_fig = plt.figure(figsize=(8, 10))
        self.data_fig.canvas.manager.set_window_title('Game Data')
        self.data_ax = self.data_fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.data_fig.canvas.mpl_connect('close_event', self._on_data_window_closed)
        plt.close(self.data_fig)  # Close it initially so it doesn't show on startup

        # Create global stat window figure (hidden initially)
        self.global_stat_fig = plt.figure(figsize=(8, 10))
        self.global_stat_fig.canvas.manager.set_window_title('全局统计')
        self.global_stat_ax = self.global_stat_fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.global_stat_fig.canvas.mpl_connect('close_event', self._on_global_stat_window_closed)
        plt.close(self.global_stat_fig)

        # Buttons (right side, below symbol display area)
        bax_undo = self.fig.add_axes([0.914, 0.08, 0.043, 0.025])
        self._btn_undo = Button(bax_undo, '撤销', color='#aaddff')
        self._btn_undo.label.set_fontsize(12)
        self._btn_undo.on_clicked(lambda _evt: self._undo())

        bax_rst = self.fig.add_axes([0.957, 0.08, 0.043, 0.025])
        self._btn_reset = Button(bax_rst, '重置', color='#e09050')
        self._btn_reset.label.set_fontsize(12)
        self._btn_reset.on_clicked(lambda _evt: self._confirm_reset())

        bax_fly = self.fig.add_axes([0.914, 0.032, 0.043, 0.025])
        self._btn_fly = Button(bax_fly, '飞雷神', color='#FFB6C1')
        self._btn_fly.label.set_fontsize(12)
        self._btn_fly.on_clicked(lambda _evt: self._activate_fly_skill())

        # Toggle visibility of future-day paths.
        bax_future_paths = self.fig.add_axes([0.914, 0.056, 0.086, 0.022])
        self._btn_show_future = Button(bax_future_paths, '显示未来', color='#90EE90', hovercolor='#7FDF7F')
        self._btn_show_future.label.set_fontsize(11)
        self._btn_show_future.on_clicked(lambda _evt: self._toggle_show_future_paths())
        self._update_show_future_button_state()
        
        # Add text label to show fly skill limit on the button's axes
        self._fly_skill_label_text = bax_fly.text(0.95, 0.5, str(self.fly_skill_limit),
                                                   transform=bax_fly.transAxes,
                                                   fontsize=5, fontweight='bold',
                                                   ha='right', va='center')
        
        # Checkbox to toggle bonus labels (B, X, Z) - same size as other buttons
        bax_chk = self.fig.add_axes([0.957, 0.032, 0.043, 0.025])
        self._chk_state = True
        self._btn_chk_labels = Button(bax_chk, '显示buff', color='#90EE90', hovercolor='#7FDF7F')
        self._btn_chk_labels.label.set_fontsize(12)
        self._btn_chk_labels.on_clicked(lambda _evt: self._toggle_checkbox_state())

        # Prev/Next Day buttons - side by side at bottom left
        bax_prev_day = self.fig.add_axes([0.02, 0.002, 0.084, 0.025])
        self._btn_prev_day = Button(bax_prev_day, '前一天(Q)', color='#ffe699')
        self._btn_prev_day.label.set_fontsize(12)
        self._btn_prev_day.on_clicked(lambda _evt: self._go_previous_day())

        bax_next_day = self.fig.add_axes([0.11, 0.002, 0.084, 0.025])
        self._btn_next_day = Button(bax_next_day, '后一天(E)', color='#ffeb99')
        self._btn_next_day.label.set_fontsize(12)
        self._btn_next_day.on_clicked(lambda _evt: self._advance_day())

        # Toggle between the drawn hex grid and the real-game map screenshot
        bax_map_view = self.fig.add_axes([0.20, 0.002, 0.084, 0.025])
        self._btn_map_view = Button(bax_map_view, '切换地图', color='#c9b3ff')
        self._btn_map_view.label.set_fontsize(12)
        self._btn_map_view.on_clicked(lambda _evt: self._toggle_map_view())

        bax_screenshot = self.fig.add_axes([0.29, 0.002, 0.084, 0.025])
        self._btn_screenshot = Button(bax_screenshot, '截图', color='#d9f2ff')
        self._btn_screenshot.label.set_fontsize(12)
        self._btn_screenshot.on_clicked(lambda _evt: self._copy_map_screenshot())

        # Day/date display on right side, above the food/reward table
        self._day_number_text = self.fig.text(0.95, 0.86, self._format_day_with_date(1),
                              ha='center', va='center', fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff9d6', edgecolor="#f1ef62", linewidth=2.0))
        self._map_image_credit_text = self.fig.text(
            0.95, 0.94, '图片来自sx弑邪\n贴吧群659503174',
            ha='center', va='center', fontsize=19.2, color="#b6b6b6cf",
            alpha=0.16, linespacing=1.15, visible=False,
        )

        # Load/Save buttons at bottom right - touching edge
        bax_load = self.fig.add_axes([0.914, 0.002, 0.043, 0.025])
        self._btn_load = Button(bax_load, '读取', color='#b3d9ff')
        self._btn_load.label.set_fontsize(12)
        self._btn_load.on_clicked(lambda _evt: self._load_game())

        bax_save = self.fig.add_axes([0.957, 0.002, 0.043, 0.025])
        self._btn_save = Button(bax_save, '保存', color='#99ff99')
        self._btn_save.label.set_fontsize(12)
        self._btn_save.on_clicked(lambda _evt: self._save_game())

        # Global Stat button at lower-right area
        bax_global_stat = self.fig.add_axes([0.827, 0.002, 0.084, 0.025])
        self._btn_global_stat = Button(bax_global_stat, '全局统计', color='#ffd9b3')
        self._btn_global_stat.label.set_fontsize(12)
        self._btn_global_stat.on_clicked(lambda _evt: self._show_global_stat_window())

        # Segment edit mode button
        bax_edit_seg = self.fig.add_axes([0.737, 0.002, 0.084, 0.025])
        self._btn_edit_seg = Button(bax_edit_seg, '路径编辑', color='#ffe0b3')
        self._btn_edit_seg.label.set_fontsize(12)
        self._btn_edit_seg.on_clicked(lambda _evt: self._toggle_segment_edit_mode())

        # Export day sheets to workbook template
        bax_export_xlsx = self.fig.add_axes([0.647, 0.002, 0.084, 0.025])
        self._btn_export_xlsx = Button(bax_export_xlsx, '导出表', color='#d9f2ff')
        self._btn_export_xlsx.label.set_fontsize(12)
        self._btn_export_xlsx.on_clicked(lambda _evt: self._export_day_sheets_xlsx())

        # Add three team switch buttons at right side, just above symbol display area
        bax_team1 = self.fig.add_axes([0.909, 0.735, 0.022, 0.025])
        self._btn_team1 = Button(bax_team1, 'T1', color=self.team_colors[1])
        self._btn_team1.label.set_fontsize(17)
        self._btn_team1.label.set_color('#FFFF00')
        self._btn_team1.on_clicked(lambda _evt: self._on_team_button_click(1))

        bax_team2 = self.fig.add_axes([0.937, 0.735, 0.022, 0.025])
        self._btn_team2_switch = Button(bax_team2, 'T2', color=self.team_colors[2])
        self._btn_team2_switch.label.set_fontsize(17)
        self._btn_team2_switch.label.set_color('#FFFF00')
        self._btn_team2_switch.on_clicked(lambda _evt: self._on_team_button_click(2))

        bax_team3 = self.fig.add_axes([0.965, 0.735, 0.022, 0.025])
        self._btn_team3_switch = Button(bax_team3, 'T3', color=self.team_colors[3])
        self._btn_team3_switch.label.set_fontsize(17)
        self._btn_team3_switch.label.set_color('#FFFF00')
        self._btn_team3_switch.on_clicked(lambda _evt: self._on_team_button_click(3))

        # Team-button double-click detection state (backend-independent).
        self._team_button_last_click_time = {1: 0.0, 2: 0.0, 3: 0.0}
        self._team_button_dblclick_window_sec = 0.40

        # Team action display area (large enough for 30+ symbols)
        self._team_action_ax = self.fig.add_axes([0.91, 0.15, 0.09, 0.60])
        self._team_action_ax.axis('off')

        # Day stats display area (food left and cumulated reward) - above team buttons
        self._map_stats_ax = self.fig.add_axes([0.90, 0.765, 0.10, 0.08])
        self._map_stats_ax.axis('off')

        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('close_event', self._on_main_window_closed)

        # Initialize team button colors
        self._update_switch_button_color()
        
        self._draw()
        plt.show()

    def _format_day_with_date(self, day_num):
        """Format day/date label with the current Tent stage from landInfo."""
        safe_day = max(1, int(day_num))
        dt = self._calendar_day1 + timedelta(days=safe_day - 1)
        tent_stage = 1
        for entry in _TERRAIN_DB.get('T', {}).get('degrade', []):
            if safe_day >= int(entry.get('day', 0)):
                tent_stage += 1
        return f'Day {safe_day} ({dt.month}/{dt.day}) | 帐篷{tent_stage}阶段'

    def _on_data_window_closed(self, _event):
        """Track when the data window is closed by the user."""
        self._data_window_open = False

    def _on_main_window_closed(self, _event):
        """Close dependent stat windows when the main map window closes."""
        self._data_window_open = False
        self._global_stat_window_open = False

        try:
            if (hasattr(self, 'global_stat_fig') and self.global_stat_fig is not None and
                    plt.fignum_exists(self.global_stat_fig.number)):
                plt.close(self.global_stat_fig)
        except Exception:
            pass

    def _on_global_stat_window_closed(self, _event):
        """Track when the global stat window is closed by the user."""
        self._global_stat_window_open = False

    def _get_scaled_map_bounds(self):
        """Return full-map bounds in scaled display coordinates."""
        xmin = -HEX_SIZE * 2 * X_SCALE
        xmax = ((COLS - 1) * 1.5 * HEX_SIZE + 2 * HEX_SIZE) * X_SCALE
        ymin = -HEX_SIZE * 2 * Y_SCALE
        ymax = ((ROWS - 1) * np.sqrt(3) * HEX_SIZE + np.sqrt(3) * HEX_SIZE) * Y_SCALE
        return xmin, xmax, ymin, ymax

    def _compute_fit_limits_for_axes(self):
        """Compute x/y limits that fit the full map into current axes size."""
        xmin, xmax, ymin, ymax = self._get_scaled_map_bounds()
        map_w = max(1e-6, xmax - xmin)
        map_h = max(1e-6, ymax - ymin)

        try:
            bbox = self.ax.get_window_extent()
            ax_w = max(1.0, float(bbox.width))
            ax_h = max(1.0, float(bbox.height))
        except Exception:
            ax_w = map_w
            ax_h = map_h

        ax_ratio = ax_w / ax_h
        map_ratio = map_w / map_h

        if ax_ratio >= map_ratio:
            # Axes are wider than map: expand x-span to keep aspect and fit height.
            target_w = map_h * ax_ratio
            pad_x = (target_w - map_w) * 0.5
            return (xmin - pad_x, xmax + pad_x), (ymin, ymax)

        # Axes are taller than map: expand y-span to keep aspect and fit width.
        target_h = map_w / ax_ratio
        pad_y = (target_h - map_h) * 0.5
        return (xmin, xmax), (ymin - pad_y, ymax + pad_y)

    def _on_resize(self, _event):
        """Keep map view fitted to window size while preserving hex proportions."""
        if self._has_zoomed:
            return
        try:
            fit_xlim, fit_ylim = self._compute_fit_limits_for_axes()
            self._default_xlim = fit_xlim
            self._default_ylim = fit_ylim
            self.ax.set_xlim(fit_xlim)
            self.ax.set_ylim(fit_ylim)
            self.ax.set_aspect('equal', adjustable='box')
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _refresh_open_stat_windows(self):
        """Refresh stat windows that are currently open so tables stay live."""
        if self._data_window_open:
            try:
                if hasattr(self, 'data_fig') and self.data_fig is not None and plt.fignum_exists(self.data_fig.number):
                    self._draw_data_window()
                else:
                    self._data_window_open = False
            except Exception:
                self._data_window_open = False

        if self._global_stat_window_open:
            try:
                if hasattr(self, 'global_stat_fig') and self.global_stat_fig is not None and plt.fignum_exists(self.global_stat_fig.number):
                    self._draw_global_stat_window()
                else:
                    self._global_stat_window_open = False
            except Exception:
                self._global_stat_window_open = False

    # ── Hover preview ────────────────────────────────────────────────────────────

    def _clear_hover_preview(self):
        """Clear the hover preview path and cancel timer."""
        if self._hover_timer is not None:
            self._hover_timer.cancel()
            self._hover_timer = None
        if self._hover_path_line is not None:
            # The artist can already be detached (e.g. a _draw() -> ax.clear()
            # happened while a preview was showing) - remove() on an already-
            # detached artist raises NotImplementedError. This runs at the top
            # of every click/mouse-move handler, so letting that exception
            # escape here would leave self._hover_path_line non-None forever,
            # permanently breaking all map interaction from then on (matplotlib
            # swallows exceptions raised inside callbacks, so it fails silently
            # instead of crashing - see _draw()'s comment for the full story).
            try:
                self._hover_path_line.remove()
            except NotImplementedError:
                pass
            self._hover_path_line = None
            self.fig.canvas.draw_idle()
        self._hover_hex = None
    
    def _on_motion(self, event):
        """Handle mouse motion - track hover and show preview after 0.5s, also handle panning."""
        # Handle right-click pan
        if self._pan_active and event.x is not None and event.y is not None:
            # Convert the raw pixel position through the transform frozen at
            # drag-start (not self.ax.transData, which shifts every frame as we
            # pan) so the delta is measured in one consistent reference frame.
            cur_x, cur_y = self._pan_transform.transform((event.x, event.y))
            dx = cur_x - self._pan_start_x
            dy = cur_y - self._pan_start_y

            # Pan the map by adjusting axis limits (pan in opposite direction of mouse movement)
            new_xlim = (self._pan_start_xlim[0] - dx, self._pan_start_xlim[1] - dx)
            new_ylim = (self._pan_start_ylim[0] - dy, self._pan_start_ylim[1] - dy)

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.fig.canvas.draw_idle()
            return  # Skip hover preview during pan
        
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self._clear_hover_preview()
            return

        if self._segment_edit_mode:
            self._clear_hover_preview()
            return

        day_locked, _ = self._is_active_team_locked_by_day()
        if day_locked:
            self._clear_hover_preview()
            return
        
        # Get the hex under cursor
        hex_under_cursor = self._pixel_to_hex(event.xdata, event.ydata)
        if hex_under_cursor is None:
            self._clear_hover_preview()
            return
        
        # If different hex, clear old preview and start new timer
        if hex_under_cursor != self._hover_hex:
            self._clear_hover_preview()
            self._hover_hex = hex_under_cursor
            
            # Check if hex is valid and reachable
            ir, ic = hex_under_cursor
            if 0 <= ir < ROWS and 0 <= ic < COLS:
                terrain = _terrain(ir, ic)
                if terrain['name'] != 'empty' and terrain['name'] != 'Tent':
                    # Start timer for 0.5 seconds
                    if self._hover_timer is not None:
                        self._hover_timer.cancel()
                    self._hover_timer = threading.Timer(0.5, self._on_hover_timeout)
                    self._hover_timer.start()
    
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
                if food_estimate > self.current_food:
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

    def _set_legacy_set_team_buttons_visibility(self):
        """Safely update visibility for legacy Set Team buttons if they exist."""
        if hasattr(self, '_btn_set2') and getattr(self, '_btn_set2', None) is not None:
            self._btn_set2.ax.set_visible(self.team2 is None)
        if hasattr(self, '_btn_set3') and getattr(self, '_btn_set3', None) is not None:
            self._btn_set3.ax.set_visible(self.team3 is None)

    def _ensure_team_segment_path_nodes(self, team):
        """Ensure each segment has its own canonical node list.

        Older saves only persisted full_path + _seg_lengths. Future segment editing needs
        per-segment path nodes so a middle segment can be identified and rewritten safely.
        """
        if team is None:
            return

        expected = len(team._seg_lengths)
        existing = getattr(team, '_seg_path_nodes', None)
        paths_are_valid = (
            existing is not None
            and len(existing) == expected
            and all(len(seg_nodes) == seg_len for seg_nodes, seg_len in zip(existing, team._seg_lengths))
        )

        if not paths_are_valid:
            rebuilt = []
            cursor = 1  # Skip origin at full_path[0]
            for seg_len in team._seg_lengths:
                seg_nodes = [tuple(h) for h in team.full_path[cursor:cursor + seg_len]]
                rebuilt.append(seg_nodes)
                cursor += seg_len
            team._seg_path_nodes = rebuilt
        else:
            rebuilt = [[tuple(h) for h in seg_nodes] for seg_nodes in existing]
            team._seg_path_nodes = rebuilt

        existing_end_positions = getattr(team, '_seg_end_positions', None)
        if existing_end_positions is None or len(existing_end_positions) != expected:
            team._seg_end_positions = [seg_nodes[-1] if seg_nodes else team.origin for seg_nodes in rebuilt]
        self._ensure_team_segment_action_orders(team)

    def _ensure_team_segment_action_orders(self, team):
        """Keep persisted cross-team action-order entries aligned with segments."""
        expected = len(team._seg_lengths)
        orders = list(getattr(team, '_seg_action_orders', []))[:expected]
        orders.extend([None] * (expected - len(orders)))
        team._seg_action_orders = orders

    def _append_segment_action_order(self, team):
        """Assign the next global ordering slot to a newly-created segment."""
        self._normalize_missing_action_orders()
        self._ensure_team_segment_action_orders(team)
        team._seg_action_orders.append(self._next_action_order)
        self._next_action_order += 1

    def _normalize_missing_action_orders(self):
        """Give legacy segments deterministic order values before appending new ones."""
        segments = []
        for team_num, team in ((1, self.team1), (2, self.team2), (3, self.team3)):
            if team is None:
                continue
            self._ensure_team_segment_action_orders(team)
            for seg_idx, seg_day in enumerate(team._seg_days):
                segments.append((int(seg_day), team_num, seg_idx, team))

        existing_orders = [
            order
            for _day, _team_num, seg_idx, team in segments
            for order in [team._seg_action_orders[seg_idx]]
            if isinstance(order, int)
        ]
        next_order = max(existing_orders, default=-1) + 1
        for _day, _team_num, seg_idx, team in sorted(segments):
            if team._seg_action_orders[seg_idx] is None:
                team._seg_action_orders[seg_idx] = next_order
                next_order += 1
        self._next_action_order = next_order

    def _get_team_segment_infos(self, team):
        """Return normalized segment descriptors for a team.

        Each descriptor includes start/end positions and canonical nodes. This is intended as
        the future hand-off surface for segment edit mode rather than direct access to the
        parallel arrays.
        """
        if team is None:
            return []

        self._ensure_team_segment_path_nodes(team)
        infos = []
        seg_start = team.origin
        for seg_idx, seg_nodes in enumerate(team._seg_path_nodes):
            seg_end = seg_nodes[-1] if seg_nodes else seg_start
            infos.append({
                'seg_idx': seg_idx,
                'day': team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else team.created_day,
                'length': len(seg_nodes),
                'start_pos': seg_start,
                'end_pos': team._seg_end_positions[seg_idx] if seg_idx < len(team._seg_end_positions) else seg_end,
                'path_nodes': list(seg_nodes),
                'is_fly_skill': team._seg_is_fly_skill[seg_idx] if seg_idx < len(team._seg_is_fly_skill) else False,
            })
            seg_start = infos[-1]['end_pos']
        return infos

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

    def _append_team_segment_payload(self, team, payload):
        """Append captured per-segment payload back to a team."""
        for field_name in self._segment_field_names():
            value = getattr(team, field_name, None)
            if value is None:
                continue
            value.extend(payload.get(field_name, []))

    def _draw_day_edit_future_preview(self, team, team_color):
        """Draw future-day path preview while selected day is being redrawn."""
        if self._day_edit_context is None or team is not self._day_edit_context.get('team'):
            return

        preview_segments = self._day_edit_context.get('future_preview_segments', [])
        if not preview_segments:
            return

        preview_alpha = 0.42
        preview_lw = max(1.6 * self._get_path_line_scale() * 1.4, 0.08)

        for seg in preview_segments:
            start_pos = tuple(seg['start_pos'])
            seg_nodes = [tuple(h) for h in seg.get('path_nodes', [])]
            if not seg_nodes:
                continue

            prev = start_pos
            for node in seg_nodes:
                token_prev = RAW_MAP[prev[0]][prev[1]] if 0 <= prev[0] < ROWS and 0 <= prev[1] < COLS else ''
                token_curr = RAW_MAP[node[0]][node[1]] if 0 <= node[0] < ROWS and 0 <= node[1] < COLS else ''
                is_prev_portal = bool(re.fullmatch(r'P\d+', token_prev))
                is_curr_portal = bool(re.fullmatch(r'P\d+', token_curr))

                # Keep portal teleport connectors hidden in preview too.
                if prev != node and is_prev_portal and is_curr_portal:
                    prev = node
                    continue
                if node not in _neighbors(*prev) and (is_prev_portal or is_curr_portal):
                    prev = node
                    continue

                x0, y0 = _center(*prev)
                x1, y1 = _center(*node)
                self.ax.plot(
                    [x0 * X_SCALE, x1 * X_SCALE],
                    [y0 * Y_SCALE, y1 * Y_SCALE],
                    color=team_color,
                    lw=preview_lw,
                    alpha=preview_alpha,
                    linestyle=(0, (3.0, 3.0)),
                    zorder=2,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                )
                prev = node

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

        # Keep only history up to yesterday.
        self._truncate_team_history_from_segment(team, first_day_idx)

        self._day_edit_context = {
            'team': team,
            'day': start_day,
            'day_start': start_day,
            'day_end': end_day,
            'anchor_prev_end': anchor_prev_end,
            'anchor_next_start': anchor_next_start,
            'pending_future_payload': pending_future_payload,
            'future_preview_segments': future_preview_segments,
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

        # Re-attach untouched future days and rebuild shared derived state.
        self._append_team_segment_payload(team, ctx['pending_future_payload'])
        self._rebalance_all_teams_from_day(day)
        self._rebuild_shared_derived_state_from_segments()
        self._rebuild_day_records()

        self._segment_edit_mode = False
        self._segment_edit_targets = []
        self._segment_edit_selected_days = set()
        self._segment_edit_focus_seg_idx = None
        self._day_edit_context = None
        self.current_day = day
        self._status_msg = f'Days {day}-{day_end} edit completed. Future segments reconnected successfully.'
        return True

    def _rebalance_team_segment_days_from(self, team, start_day):
        """Reassign segment days from start_day onward to avoid per-day step overflow.

        Segments stay in original order; when a day runs out of steps, remaining segments
        are pushed to following days.
        """
        if team is None or not team._seg_days:
            return

        earliest = max(start_day, team.created_day)

        # Fixed food usage from other teams by day; used as immutable baseline while
        # reassigning only the target team's segment days.
        other_food_by_day = {}
        for other in (self.team1, self.team2, self.team3):
            if other is None or other is team:
                continue
            for seg_food, seg_day in zip(other._seg_foods, other._seg_days):
                d = int(seg_day)
                other_food_by_day[d] = other_food_by_day.get(d, 0) + seg_food

        # Step bank immediately before earliest day.
        step_bank = 0
        for day in range(team.created_day, earliest):
            step_bank = min(step_bank + 6, 18)
            day_used = 0
            for seg_idx, seg_day in enumerate(team._seg_days):
                if seg_day == day:
                    day_used += team._seg_steps[seg_idx]
            step_bank -= day_used

        # Food remaining for earliest day after fixed usage before/at earliest.
        prev_food_end = None
        for day in range(1, earliest + 1):
            day_start_food = 6800 if day == 1 else (prev_food_end + 1600)
            fixed_team_food = 0
            if day < earliest:
                for seg_idx, seg_day in enumerate(team._seg_days):
                    if seg_day == day:
                        fixed_team_food += team._seg_foods[seg_idx]
            day_used_food = other_food_by_day.get(day, 0) + fixed_team_food
            prev_food_end = day_start_food - day_used_food

        day_food_remaining = prev_food_end

        # Enter earliest day.
        current_day = earliest
        step_bank = min(step_bank + 6, 18)

        seg_idx = 0
        while seg_idx < len(team._seg_days):
            if team._seg_days[seg_idx] < earliest:
                seg_idx += 1
                continue

            seg_steps = team._seg_steps[seg_idx]
            seg_food = team._seg_foods[seg_idx]

            if seg_steps > 0 or seg_food > 0:
                # Keep trying on each day: first split if partially fit, otherwise advance day.
                while True:
                    needs_split = (seg_steps > 0 and step_bank > 0 and step_bank < seg_steps) or (
                        seg_food > 0 and day_food_remaining > 0 and day_food_remaining < seg_food
                    )
                    if needs_split:
                        did_split = self._split_team_segment_by_step_budget(
                            team,
                            seg_idx,
                            max(step_bank, 0),
                            max(day_food_remaining, 0),
                        )
                        if did_split:
                            seg_steps = team._seg_steps[seg_idx]
                            seg_food = team._seg_foods[seg_idx]

                    if (
                        (seg_steps <= 0 or step_bank >= seg_steps)
                        and (seg_food <= 0 or day_food_remaining >= seg_food)
                    ):
                        break

                    current_day += 1
                    step_bank = min(step_bank + 6, 18)
                    day_food_remaining = day_food_remaining + 1600 - other_food_by_day.get(current_day, 0)

                team._seg_days[seg_idx] = current_day
                step_bank -= seg_steps
                if step_bank > 18:
                    step_bank = 18
                day_food_remaining -= seg_food
            else:
                # Zero/negative step segments can stay today; negative steps refill bank.
                team._seg_days[seg_idx] = current_day
                step_bank -= seg_steps
                if step_bank > 18:
                    step_bank = 18
                day_food_remaining -= seg_food

            seg_idx += 1

        if team._seg_days:
            team.max_day_reached = max(team.created_day, max(team._seg_days))
        else:
            team.max_day_reached = team.created_day

    def _split_team_segment_by_step_budget(self, team, seg_idx, step_budget, food_budget):
        """Split one segment into [fits_today, overflow] using available step budget.

        Returns True if a split was performed.
        """
        if team is None:
            return False
        if seg_idx < 0 or seg_idx >= len(team._seg_lengths):
            return False
        if step_budget <= 0 and food_budget <= 0:
            return False

        # Avoid splitting fly or special delta segments; keep them atomic.
        is_fly = seg_idx < len(team._seg_is_fly_skill) and team._seg_is_fly_skill[seg_idx]
        seg_delta = team._seg_fly_skill_deltas[seg_idx] if seg_idx < len(team._seg_fly_skill_deltas) else 0
        if is_fly or seg_delta != 0:
            return False

        seg_nodes = list(team._seg_path_nodes[seg_idx]) if seg_idx < len(team._seg_path_nodes) else []
        if not seg_nodes or len(seg_nodes) <= 1:
            return False

        seg_hex_costs = list(team._seg_hex_costs[seg_idx]) if seg_idx < len(team._seg_hex_costs) else []
        extra_prefix_costs = max(0, len(seg_hex_costs) - len(seg_nodes))

        # Derive per-node step costs from hex-cost entries aligned to path nodes.
        node_step_costs = []
        for i in range(len(seg_nodes)):
            cost_idx = extra_prefix_costs + i
            if 0 <= cost_idx < len(seg_hex_costs) and len(seg_hex_costs[cost_idx]) >= 3:
                node_step_costs.append(seg_hex_costs[cost_idx][2])
            else:
                node_step_costs.append(0)

        # Find the largest node-prefix that fits within today's step/food budget.
        acc = 0
        food_acc = 0
        split_node_count = 0
        prefix_food = 0
        used_exact_cost_split = False
        if extra_prefix_costs > 0:
            for i in range(extra_prefix_costs):
                if i < len(seg_hex_costs) and len(seg_hex_costs[i]) >= 1:
                    prefix_food += seg_hex_costs[i][0]

        for node_i, step_cost in enumerate(node_step_costs):
            node_food = 0
            cost_idx = extra_prefix_costs + node_i
            if 0 <= cost_idx < len(seg_hex_costs) and len(seg_hex_costs[cost_idx]) >= 1:
                node_food = seg_hex_costs[cost_idx][0]

            next_step = acc + step_cost
            next_food = food_acc + node_food
            budget_step_ok = (step_cost <= 0) or (next_step <= step_budget)
            budget_food_ok = (prefix_food + next_food <= food_budget)
            if not budget_step_ok or not budget_food_ok:
                break

            acc = next_step
            food_acc = next_food
            split_node_count += 1

        if split_node_count > 0 and split_node_count < len(seg_nodes):
            used_exact_cost_split = True

        # Fallback split when exact per-node costs are missing/incomplete.
        # This prevents pushing an entire multi-node segment to next day when part can fit today.
        if not used_exact_cost_split:
            node_count = len(seg_nodes)
            if node_count > 1:
                max_by_step = node_count - 1
                if team._seg_steps[seg_idx] > 0:
                    max_by_step = int(np.floor((step_budget * node_count) / max(team._seg_steps[seg_idx], 1)))

                max_by_food = node_count - 1
                if team._seg_foods[seg_idx] > 0:
                    max_by_food = int(np.floor((food_budget * node_count) / max(team._seg_foods[seg_idx], 1)))

                fallback_count = min(node_count - 1, max_by_step, max_by_food)
                if fallback_count > 0:
                    split_node_count = fallback_count

        if split_node_count <= 0 or split_node_count >= len(seg_nodes):
            return False

        nodes_a = list(seg_nodes[:split_node_count])
        nodes_b = list(seg_nodes[split_node_count:])

        action_seq = list(team._seg_action_sequence[seg_idx]) if seg_idx < len(team._seg_action_sequence) else []
        split_action_count = min(split_node_count, len(action_seq))
        action_a = action_seq[:split_action_count]
        action_b = action_seq[split_action_count:]

        hex_costs_a = seg_hex_costs[:extra_prefix_costs + split_node_count] if seg_hex_costs else []
        hex_costs_b = seg_hex_costs[extra_prefix_costs + split_node_count:] if seg_hex_costs else []

        node_set_a = set(nodes_a)
        node_set_b = set(nodes_b)

        new_hexes = list(team._seg_new_hexes[seg_idx]) if seg_idx < len(team._seg_new_hexes) else []
        jumps = list(team._seg_jumps[seg_idx]) if seg_idx < len(team._seg_jumps) else []
        explores = list(team._seg_exploration_hexes[seg_idx]) if seg_idx < len(team._seg_exploration_hexes) else []

        new_a = [h for h in new_hexes if h in node_set_a]
        new_b = [h for h in new_hexes if h in node_set_b]
        jumps_a = [h for h in jumps if h in node_set_a]
        jumps_b = [h for h in jumps if h in node_set_b]
        explores_a = [h for h in explores if h in node_set_a]
        explores_b = [h for h in explores if h in node_set_b]

        if hex_costs_a or hex_costs_b:
            food_a = sum(c[0] for c in hex_costs_a if len(c) >= 1)
            award_a = sum(c[1] for c in hex_costs_a if len(c) >= 2)
            steps_a = sum(c[2] for c in hex_costs_a if len(c) >= 3)

            food_b = sum(c[0] for c in hex_costs_b if len(c) >= 1)
            award_b = sum(c[1] for c in hex_costs_b if len(c) >= 2)
            steps_b = sum(c[2] for c in hex_costs_b if len(c) >= 3)
        else:
            # Proportional fallback when per-hex costs are unavailable.
            ratio = split_node_count / max(len(seg_nodes), 1)

            food_a = int(round(team._seg_foods[seg_idx] * ratio))
            if team._seg_foods[seg_idx] > 0 and food_a <= 0:
                food_a = 1
            food_a = min(food_a, max(food_budget, 0), team._seg_foods[seg_idx])
            food_b = team._seg_foods[seg_idx] - food_a

            award_a = int(round(team._seg_awards[seg_idx] * ratio))
            award_b = team._seg_awards[seg_idx] - award_a

            steps_a = int(np.floor(team._seg_steps[seg_idx] * ratio))
            if team._seg_steps[seg_idx] > 0 and steps_a <= 0:
                steps_a = 1
            steps_a = min(steps_a, max(step_budget, 0), team._seg_steps[seg_idx])
            steps_b = team._seg_steps[seg_idx] - steps_a

            if steps_b < 0:
                steps_b = 0
                steps_a = team._seg_steps[seg_idx]

        # Update first half in-place.
        team._seg_lengths[seg_idx] = len(nodes_a)
        team._seg_foods[seg_idx] = food_a
        team._seg_awards[seg_idx] = award_a
        team._seg_steps[seg_idx] = steps_a
        team._seg_new_hexes[seg_idx] = new_a
        team._seg_exploration_hexes[seg_idx] = explores_a
        team._seg_jumps[seg_idx] = jumps_a
        team._seg_path_nodes[seg_idx] = nodes_a
        team._seg_end_positions[seg_idx] = nodes_a[-1]
        team._seg_action_sequence[seg_idx] = action_a
        team._seg_hex_costs[seg_idx] = hex_costs_a

        # Insert overflow as a new segment right after current segment.
        insert_at = seg_idx + 1
        team._seg_lengths.insert(insert_at, len(nodes_b))
        team._seg_foods.insert(insert_at, food_b)
        team._seg_awards.insert(insert_at, award_b)
        team._seg_steps.insert(insert_at, steps_b)
        team._seg_days.insert(insert_at, team._seg_days[seg_idx])
        team._seg_new_hexes.insert(insert_at, new_b)
        team._seg_exploration_hexes.insert(insert_at, explores_b)
        team._seg_jumps.insert(insert_at, jumps_b)
        team._seg_path_nodes.insert(insert_at, nodes_b)
        team._seg_end_positions.insert(insert_at, nodes_b[-1])
        team._seg_action_sequence.insert(insert_at, action_b)
        team._seg_action_orders.insert(insert_at, team._seg_action_orders[seg_idx])
        team._seg_hex_costs.insert(insert_at, hex_costs_b)
        team._seg_is_fly_skill.insert(insert_at, False)
        team._seg_fly_skill_deltas.insert(insert_at, 0)

        return True

    def _team_has_step_overflow_from(self, team, start_day):
        """Return True if segment assignment causes negative step bank from start_day onward."""
        if team is None or not team._seg_days:
            return False

        earliest = max(start_day, team.created_day)
        step_bank = 0

        # Reconstruct bank before earliest day from existing assignment.
        for day in range(team.created_day, earliest):
            step_bank = min(step_bank + 6, 18)
            day_used = 0
            for seg_idx, seg_day in enumerate(team._seg_days):
                if seg_day == day:
                    day_used += team._seg_steps[seg_idx]
            step_bank -= day_used

        for day in range(earliest, TOTAL_DAYS + 1):
            step_bank = min(step_bank + 6, 18)
            day_used = 0
            for seg_idx, seg_day in enumerate(team._seg_days):
                if seg_day == day:
                    day_used += team._seg_steps[seg_idx]
            step_bank -= day_used
            if step_bank < 0:
                return True
        return False

    def _has_global_food_deficit_from_segments(self):
        """Return True if segment day assignment causes food to drop below 0 on any day."""
        food_remaining = 6800
        teams = [t for t in (self.team1, self.team2, self.team3) if t is not None]

        for day in range(1, TOTAL_DAYS + 1):
            used_today = 0
            for team in teams:
                for seg_food, seg_day in zip(team._seg_foods, team._seg_days):
                    if int(seg_day) == day:
                        used_today += seg_food

            food_remaining -= used_today
            if food_remaining < 0:
                return True

            if day < TOTAL_DAYS:
                food_remaining += 1600

        return False

    def _rebalance_all_teams_from_day(self, start_day, max_passes=8):
        """Iteratively rebalance all teams from a day to satisfy shared food and team steps."""
        teams = [t for t in (self.team1, self.team2, self.team3) if t is not None]
        if not teams:
            return

        rebalance_start = max(1, int(start_day))

        for _ in range(max_passes):
            before = [tuple(t._seg_days) for t in teams]

            for team in teams:
                self._rebalance_team_segment_days_from(team, max(rebalance_start, team.created_day))

            has_step_overflow = any(
                self._team_has_step_overflow_from(team, max(rebalance_start, team.created_day))
                for team in teams
            )
            has_food_deficit = self._has_global_food_deficit_from_segments()
            after = [tuple(t._seg_days) for t in teams]

            if (not has_step_overflow) and (not has_food_deficit):
                break
            if before == after:
                break

    def _toggle_segment_edit_mode(self):
        """Toggle UI mode for selecting a segment on the current team/day."""
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

    def _update_segment_edit_button_state(self):
        """Reflect segment edit mode state on the button UI."""
        if not hasattr(self, '_btn_edit_seg') or self._btn_edit_seg is None:
            return
        if self._segment_edit_mode:
            self._btn_edit_seg.color = '#ffb870'
            self._btn_edit_seg.hovercolor = '#ffaa55'
            self._btn_edit_seg.label.set_color('#7a2e00')
        else:
            self._btn_edit_seg.color = '#ffe0b3'
            self._btn_edit_seg.hovercolor = '#ffd199'
            self._btn_edit_seg.label.set_color('black')

    def _build_segment_edit_targets(self):
        """Build clickable segment markers for current team/day in edit mode."""
        self._segment_edit_targets = []
        if not self._segment_edit_mode or self.active_team is None:
            return

        segs = self._get_team_segment_infos(self.active_team)
        for seg in segs:
            sx, sy = _center(*seg['start_pos'])
            ex, ey = _center(*seg['end_pos'])
            marker_x = (sx + ex) * 0.5 * X_SCALE
            marker_y = (sy + ey) * 0.5 * Y_SCALE
            self._segment_edit_targets.append({
                'seg_idx': seg['seg_idx'],
            'label': 'Edit',
                'x': marker_x,
                'y': marker_y,
                'start_pos': seg['start_pos'],
                'end_pos': seg['end_pos'],
                'day': seg['day'],
            })

    def _draw_segment_edit_targets(self):
        """Draw segment markers that the user can click in edit mode."""
        self._build_segment_edit_targets()
        # Edit selection now uses path highlighting instead of overlay tags.
        return

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

    def _rebuild_shared_derived_state_from_segments(self):
        """Replay segment history to reconstruct shared derived runtime state.

        This is the preparation layer for future segment-edit mode. It rebuilds the parts of
        runtime state that should be derivable from canonical segment history rather than
        trusted from incremental mutation.
        """
        teams = [team for team in (self.team1, self.team2, self.team3) if team is not None]

        self.all_visited_hexes = set()
        self.visited_g_lands = set()
        self.fly_skill_limit = 1
        self.total_food = 0

        # Origins that are still-unsettled free-exploration hexes of some team
        # (a team was created on a probe hex nobody has paid for yet) must not
        # be marked taken - see _init_team_created_at.
        explored = {tuple(h) for t in teams for seg in t._seg_exploration_hexes for h in seg}
        unsettled_origins = {h for h in explored if not self._is_hex_settled(h)}

        for team in teams:
            self._ensure_team_segment_path_nodes(team)

            if len(team._seg_end_positions) != len(team._seg_lengths):
                team._seg_end_positions = [
                    (seg_nodes[-1] if seg_nodes else team.origin)
                    for seg_nodes in team._seg_path_nodes
                ]

            team.full_path = [team.origin]
            team.visited_hexes = {team.origin}
            team.free_exploration_hexes = set()
            team._no_draw_edges = set()
            team.max_day_reached = team.created_day

            if team.origin in unsettled_origins:
                # Created on a probe hex: inherits the obligation to settle it.
                team.free_exploration_hexes.add(team.origin)
            else:
                self.all_visited_hexes.add(team.origin)
            if team.origin in self.all_g_lands:
                self.visited_g_lands.add(team.origin)

        for team in teams:
            current_pos = team.origin

            for seg_idx, seg_nodes in enumerate(team._seg_path_nodes):
                seg_nodes = [tuple(h) for h in seg_nodes]
                raw_end_pos = seg_nodes[-1] if seg_nodes else current_pos
                end_pos = team._seg_end_positions[seg_idx] if seg_idx < len(team._seg_end_positions) else raw_end_pos
                seg_day = team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else team.created_day
                is_fly_skill = team._seg_is_fly_skill[seg_idx] if seg_idx < len(team._seg_is_fly_skill) else False

                team.max_day_reached = max(team.max_day_reached, seg_day)
                if seg_idx < len(team._seg_foods):
                    self.total_food += team._seg_foods[seg_idx]
                if seg_idx < len(team._seg_fly_skill_deltas):
                    self.fly_skill_limit += team._seg_fly_skill_deltas[seg_idx]

                if current_pos in team.free_exploration_hexes:
                    team.free_exploration_hexes.discard(current_pos)

                if is_fly_skill:
                    team.full_path.append(end_pos)
                    if seg_nodes:
                        team._no_draw_edges.add((current_pos, raw_end_pos))
                    if end_pos != raw_end_pos:
                        team._no_draw_edges.add((current_pos, end_pos))
                else:
                    if seg_nodes:
                        rebuilt_nodes = list(seg_nodes)
                        if end_pos != raw_end_pos:
                            rebuilt_nodes[-1] = end_pos
                            prev_before_exit = current_pos if len(seg_nodes) == 1 else seg_nodes[-2]
                            team._no_draw_edges.add((prev_before_exit, end_pos))
                        team.full_path.extend(rebuilt_nodes)

                if seg_idx < len(team._seg_new_hexes):
                    for hex_pos in team._seg_new_hexes[seg_idx]:
                        hex_pos = tuple(hex_pos)
                        team.visited_hexes.add(hex_pos)
                        self.all_visited_hexes.add(hex_pos)
                        if hex_pos in self.all_g_lands:
                            self.visited_g_lands.add(hex_pos)

                if seg_idx < len(team._seg_exploration_hexes):
                    for hex_pos in team._seg_exploration_hexes[seg_idx]:
                        hex_pos = tuple(hex_pos)
                        team.visited_hexes.add(hex_pos)
                        team.free_exploration_hexes.add(hex_pos)
                        if hex_pos in self.all_g_lands:
                            self.visited_g_lands.add(hex_pos)

                # Portal segments mark both entry and exit as taken even though full_path only keeps the end.
                token_raw_end = RAW_MAP[raw_end_pos[0]][raw_end_pos[1]] if 0 <= raw_end_pos[0] < ROWS and 0 <= raw_end_pos[1] < COLS else ''
                if re.fullmatch(r'P\d+', token_raw_end):
                    team.visited_hexes.add(raw_end_pos)
                    self.all_visited_hexes.add(raw_end_pos)
                    if raw_end_pos in self.all_g_lands:
                        self.visited_g_lands.add(raw_end_pos)
                    if end_pos != raw_end_pos:
                        team.visited_hexes.add(end_pos)
                        self.all_visited_hexes.add(end_pos)
                        if end_pos in self.all_g_lands:
                            self.visited_g_lands.add(end_pos)

                current_pos = end_pos

        # Keep fly skill limit non-negative even if history is partially edited.
        self.fly_skill_limit = max(0, self.fly_skill_limit)

        # Team step balances are refreshed after day_records is rebuilt.

    def _get_active_team_last_movement_day(self):
        """Return latest movement day for the active team, or None if no movement exists."""
        if self.active_team is None:
            return None
        if getattr(self.active_team, '_seg_days', None):
            return max(self.active_team._seg_days)
        return None

    def _is_active_team_locked_by_day(self):
        """Check whether editing is locked because view day is before team's latest movement day."""
        last_move_day = self._get_active_team_last_movement_day()
        if last_move_day is None:
            return False, None
        return self.current_day < last_move_day, last_move_day

    # ── Team management ───────────────────────────────────────────────────────

    def _init_team_created_at(self, old_team, new_team, current_pos):
        """Shared bookkeeping for a team created on the active team's current hex.

        A still-unsettled free-exploration hex stays unsettled: the new team
        inherits the obligation (whichever team leaves first settles it) and it
        must NOT be marked globally taken yet - otherwise a third team walking
        onto it is treated as a 10-food revisit instead of occupying it."""
        if current_pos in old_team.free_exploration_hexes and not self._is_hex_settled(current_pos):
            new_team.free_exploration_hexes.add(current_pos)
        else:
            self.all_visited_hexes.add(current_pos)
        if current_pos in self.all_g_lands:
            self.visited_g_lands.add(current_pos)

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
    
    def _set_fly_button_border(self, color, linewidth):
        """Set the border color and width of the fly skill button."""
        for spine in self._btn_fly.ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(linewidth)
    
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

    def _set_edit_seg_button_border(self, color, linewidth):
        """Set the border color and width of the EditSeg button."""
        if not hasattr(self, '_btn_edit_seg') or self._btn_edit_seg is None:
            return
        for spine in self._btn_edit_seg.ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(linewidth)

    def _flash_edit_seg_button(self):
        """Toggle EditSeg button border for blinking animation in selection mode."""
        should_blink = self._segment_edit_mode and self._day_edit_context is None
        if not should_blink:
            if self._edit_seg_button_timer is not None:
                self._edit_seg_button_timer.stop()
                self._edit_seg_button_timer = None
            self._set_edit_seg_button_border('#777777', 0.8)
            return

        self._edit_seg_button_flash_state = not self._edit_seg_button_flash_state
        if self._edit_seg_button_flash_state:
            self._set_edit_seg_button_border('#FF8A00', 2.4)
        else:
            self._set_edit_seg_button_border('#FFD199', 1.3)
        self.fig.canvas.draw_idle()

    def _update_edit_seg_blink_state(self):
        """Start/stop EditSeg blinking based on whether user is selecting paths to edit."""
        should_blink = self._segment_edit_mode and self._day_edit_context is None

        if should_blink:
            if self._edit_seg_button_timer is None:
                self._edit_seg_button_flash_state = True
                self._set_edit_seg_button_border('#FF8A00', 2.4)
                self._edit_seg_button_timer = self.fig.canvas.new_timer()
                self._edit_seg_button_timer.interval = 300
                self._edit_seg_button_timer.single_shot = False
                self._edit_seg_button_timer.callbacks.append((self._flash_edit_seg_button, (), {}))
                self._edit_seg_button_timer.start()
        else:
            if self._edit_seg_button_timer is not None:
                self._edit_seg_button_timer.stop()
                self._edit_seg_button_timer = None
            self._set_edit_seg_button_border('#777777', 0.8)

    def _update_fly_button_state(self):
        """Update fly button appearance based on global fly skill limit."""
        if self.fly_skill_limit <= 0:
            # Disabled state: gray border, dimmed label
            self._set_fly_button_border('#999999', 1.2)  # Gray border, medium thick
            self._btn_fly.label.set_color('#999999')  # Gray text
        else:
            # Enabled state: default border, normal label
            self._set_fly_button_border('#777777', 0.8)  # Default border
            self._btn_fly.label.set_color('black')

    def _update_team_button_labels(self):
        """Update team button labels to show remaining steps for each team."""
        # Keep team button text styling consistent after text refresh.
        self._btn_team1.label.set_color('#FFFF00')
        self._btn_team2_switch.label.set_color('#FFFF00')
        self._btn_team3_switch.label.set_color('#FFFF00')

        # Team 1 - always exists
        team1_steps = self._get_team_steps_for_day(self.team1, self.current_day)
        self._btn_team1.label.set_text(f'{team1_steps}')
        
        # Team 2 - only if it exists
        if self.team2 is not None:
            team2_steps = self._get_team_steps_for_day(self.team2, self.current_day)
            self._btn_team2_switch.label.set_text(f'{team2_steps}')
        else:
            self._btn_team2_switch.label.set_text('—')
        
        # Team 3 - only if it exists
        if self.team3 is not None:
            team3_steps = self._get_team_steps_for_day(self.team3, self.current_day)
            self._btn_team3_switch.label.set_text(f'{team3_steps}')
        else:
            self._btn_team3_switch.label.set_text('—')

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

    def _switch_to_team(self, team_num):
        """Switch to a specific team (1, 2, or 3). If team doesn't exist, create it at current active team's position."""
        if team_num == 1:
            target_team = self.team1
        elif team_num == 2:
            if self.team2 is None:
                # Create Team 2 at the current active team's position
                old_team = self.active_team
                current_pos = old_team.full_path[-1]
                self.team2 = Team(current_pos, created_day=self.current_day)
                self._init_team_created_at(old_team, self.team2, current_pos)
                self.active_team = self.team2
                self._update_switch_button_color()
                self._status_msg = 'Team 2 created at active team position.'
                self._rebuild_day_records()
                self._draw()
                return
            target_team = self.team2
        elif team_num == 3:
            if self.team3 is None:
                # Create Team 3 at the current active team's position
                old_team = self.active_team
                current_pos = old_team.full_path[-1]
                self.team3 = Team(current_pos, created_day=self.current_day)
                self._init_team_created_at(old_team, self.team3, current_pos)
                self.active_team = self.team3
                self._update_switch_button_color()
                self._status_msg = 'Team 3 created at active team position.'
                self._rebuild_day_records()
                self._draw()
                return
            target_team = self.team3
        else:
            return
        
        if self.active_team is not target_team:
            self.active_team = target_team
            # Food is one pool shared by all 3 teams, so surface it right away when
            # switching - otherwise an idle team with plenty of steps left looks
            # "stuck" for no visible reason once another team has spent it down.
            self._status_msg = f'Switched to Team {team_num} (shared food remaining: {self.current_food})'
            self._update_switch_button_color()
            self._update_fly_button_state()

        # Always center on the selected team's current stopping position. The
        # team may have no segment on the currently viewed day, so day-based
        # centering would leave the map at the previous team's location.
        self._center_view_on_active_team()

        self._draw()

    def _update_switch_button_color(self):
        """Update team button colors to highlight active team."""
        # Highlight the active team button
        team_num = 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)
        
        # Update button colors - active team gets normal color, others get dimmed
        for i, btn in enumerate([self._btn_team1, self._btn_team2_switch, self._btn_team3_switch], 1):
            if i == team_num:
                # Active team: bright color
                btn.color = self.team_colors[i]
                btn.hovercolor = self.team_colors[i]
                btn.label.set_fontweight('bold')
            else:
                # Inactive team: dimmed color
                original_color = self.team_colors[i]
                # Simple dimming: blend with gray
                r, g, b = int(original_color[1:3], 16), int(original_color[3:5], 16), int(original_color[5:7], 16)
                dimmed_r, dimmed_g, dimmed_b = int(r * 0.6), int(g * 0.6), int(b * 0.6)
                dimmed_color = f'#{dimmed_r:02x}{dimmed_g:02x}{dimmed_b:02x}'
                btn.color = dimmed_color
                btn.hovercolor = dimmed_color
                btn.label.set_fontweight('normal')
        
        self.fig.canvas.draw_idle()

    def _draw_team_action_symbols(self):
        """Display team action symbols (lands and jumps) vertically below team buttons in map window."""
        self._team_action_ax.clear()
        self._team_action_ax.set_facecolor(APP_BACKGROUND_COLOR)
        self._team_action_ax.set_xlim(0, 3)
        self._team_action_ax.axis('off')
        
        # Import Rectangle and Circle for drawing symbols
        from matplotlib.patches import Rectangle, Circle
        
        # First pass: collect each movement segment independently. Sorting by
        # its persisted global order keeps Team 1 -> Team 2 -> Team 1 actions
        # visually interleaved in the same order they were made.
        self._normalize_missing_action_orders()
        action_segments = []
        total_symbols = 0
        for team_idx, (team, team_num) in enumerate([(self.team1, 1), (self.team2, 2), (self.team3, 3)]):
            if team is None:
                continue
            for seg_idx, seg_day in enumerate(team._seg_days):
                if seg_day != self.current_day:
                    continue
                actions = team._seg_action_sequence[seg_idx] if seg_idx < len(team._seg_action_sequence) else []
                symbol_count = sum(1 for action_idx, action in enumerate(actions)
                                   if action[0] in ('new', 'fly') or (action[0] == 'jump' and
                                   (action_idx == 0 or actions[action_idx - 1][0] != 'jump')))
                if symbol_count:
                    action_segments.append({
                        'order': team._seg_action_orders[seg_idx],
                        'team_idx': team_idx,
                        'actions': actions,
                        'symbol_count': symbol_count,
                    })
                    total_symbols += symbol_count

        action_segments.sort(key=lambda item: item['order'])

        # A player can create adjacent path segments while continuing to jump
        # through taken hexes. Combine that uninterrupted run into one symbol;
        # another team's segment remains a visible ordering boundary.
        merged_segments = []
        for action_segment in action_segments:
            actions = list(action_segment['actions'])
            previous = merged_segments[-1] if merged_segments else None
            if (
                previous is not None
                and previous['team_idx'] == action_segment['team_idx']
                and previous['actions'] and actions
                and previous['actions'][-1][0] == 'jump'
                and actions[0][0] == 'jump'
            ):
                previous['actions'].extend(actions)
                continue
            merged_segments.append({
                'order': action_segment['order'],
                'team_idx': action_segment['team_idx'],
                'actions': actions,
            })

        def _count_symbols(actions):
            return sum(
                1 for action_idx, action in enumerate(actions)
                if action[0] in ('new', 'fly') or (
                    action[0] == 'jump'
                    and (action_idx == 0 or actions[action_idx - 1][0] != 'jump')
                )
            )

        action_segments = merged_segments
        for action_segment in action_segments:
            action_segment['symbol_count'] = _count_symbols(action_segment['actions'])
        total_symbols = sum(item['symbol_count'] for item in action_segments)
        
        # Set y-axis range based on max actions (each symbol takes 0.375 units now)
        # Ensure we can display at least 30 symbols by using a larger coordinate system
        max_actions = max(total_symbols, 30)  # Minimum 30 symbols support
        y_start = max_actions * 0.45 + 1  # Starting y position with spacing for larger symbols
        self._team_action_ax.set_ylim(0, y_start + 1)

        # Align symbol columns with the actual on-screen centers of team buttons.
        x_positions = [0.4, 1.1, 1.8]  # Fallback when layout info is unavailable.
        try:
            action_bbox = self._team_action_ax.get_position()
            ax_x0 = float(action_bbox.x0)
            ax_w = max(1e-6, float(action_bbox.width))
            x_min, x_max = self._team_action_ax.get_xlim()
            x_span = max(1e-6, float(x_max - x_min))

            buttons = [self._btn_team1, self._btn_team2_switch, self._btn_team3_switch]
            mapped = []
            for btn in buttons:
                b = btn.ax.get_position()
                btn_center_fig_x = float(b.x0 + b.width * 0.5)
                rel_x = (btn_center_fig_x - ax_x0) / ax_w
                mapped_x = x_min + rel_x * x_span
                mapped_x = min(max(mapped_x, x_min + 0.08 * x_span), x_max - 0.08 * x_span)
                mapped.append(mapped_x)

            if len(mapped) == 3:
                x_positions = mapped
        except Exception:
            pass
        
        # Second pass: draw segments in the global operation timeline.
        symbols_before_segment = 0
        for action_segment in action_segments:
            actions_today = action_segment['actions']
            x_pos = x_positions[action_segment['team_idx']]
            y_pos = y_start - (symbols_before_segment * 0.375)
            
            # Draw symbols in the order they appear in actions_today
            i = 0
            while i < len(actions_today):
                action_type, action_data = actions_today[i]
                
                if action_type == 'new':
                    # Draw new hex symbol (terrain-colored rectangle)
                    hex_pos = action_data
                    
                    terrain = _terrain(hex_pos[0], hex_pos[1])
                    fc = terrain.get('face', '#cccccc')
                    ec = terrain.get('edge', '#000000')
                    if ec in ('none', ''):
                        ec = '#777777'
                    hatch_raw = terrain.get('hatch', '')
                    hatch = ''.join(ch * 2 for ch in hatch_raw) if hatch_raw else None
                    
                    # Draw small rectangle (50% bigger)
                    rect = Rectangle((x_pos - 0.12, y_pos - 0.1125), 0.24, 0.225,
                                    transform=self._team_action_ax.transData,
                                    facecolor=fc, edgecolor=ec, linewidth=0.5,
                                    hatch=hatch, zorder=2)
                    self._team_action_ax.add_patch(rect)
                    y_pos -= 0.375  # Increased spacing for larger symbols
                    i += 1
                    
                elif action_type == 'fly':
                    # Horizontal kunai: ring, wrapped handle, guard, and blade.
                    ring = Circle((x_pos - 0.105, y_pos), 0.055,
                                  transform=self._team_action_ax.transData,
                                  facecolor='none', edgecolor='#4a4a4a', linewidth=1.2, zorder=3)
                    handle = Rectangle((x_pos - 0.05, y_pos - 0.037), 0.11, 0.074,
                                       transform=self._team_action_ax.transData,
                                       facecolor='#6c4b32', edgecolor='#2d2118', linewidth=0.8, zorder=3)
                    guard = Rectangle((x_pos + 0.052, y_pos - 0.075), 0.022, 0.15,
                                      transform=self._team_action_ax.transData,
                                      facecolor='#d4a72c', edgecolor='#72550d', linewidth=0.7, zorder=4)
                    blade = Polygon(
                        [(x_pos + 0.07, y_pos - 0.09), (x_pos + 0.07, y_pos + 0.09),
                         (x_pos + 0.22, y_pos)],
                        transform=self._team_action_ax.transData,
                        facecolor='#bfc7cc', edgecolor='#40484d', linewidth=0.8, zorder=3,
                    )
                    for patch in (ring, handle, guard, blade):
                        self._team_action_ax.add_patch(patch)
                    y_pos -= 0.375
                    i += 1

                elif action_type == 'jump':
                    # Count consecutive jumps starting from current position
                    jump_count = 1
                    j = i + 1
                    while j < len(actions_today) and actions_today[j][0] == 'jump':
                        jump_count += 1
                        j += 1
                    
                    # Draw merged jump circle with count (50% bigger, centered at y_pos like rectangles)
                    circle = Circle((x_pos, y_pos), 0.18,
                                   transform=self._team_action_ax.transData,
                                   facecolor='#FFB6C1', edgecolor='#FF69B4', 
                                   linewidth=1, zorder=3)
                    self._team_action_ax.add_patch(circle)
                    
                    # Add jump count text (larger)
                    self._team_action_ax.text(x_pos, y_pos, str(jump_count),
                                             transform=self._team_action_ax.transData,
                                             fontsize=9, fontweight='bold',
                                             ha='center', va='center', zorder=4)
                    y_pos -= 0.375  # Increased spacing for larger symbols
                    
                    # Skip the consecutive jumps we just processed
                    i = j
                else:
                    i += 1

            symbols_before_segment += action_segment['symbol_count']

    def _draw_map_stats_table(self):
        """Display food left and cumulated reward table above team buttons in map window."""
        self._map_stats_ax.clear()
        self._map_stats_ax.set_xlim(0, 10)
        self._map_stats_ax.set_ylim(0, 10)
        self._map_stats_ax.axis('off')
        
        # Calculate food left and cumulated reward for current day
        if not self.day_records or self.current_day - 1 >= len(self.day_records):
            self._init_day_records()
        
        day_record = self.day_records[self.current_day - 1] if self.current_day - 1 < len(self.day_records) else None
        
        if day_record:
            food_left = day_record['food_remain']
            # Calculate cumulated reward up to current day
            cumulated_reward = 0
            for i in range(min(self.current_day, len(self.day_records))):
                cumulated_reward += self.day_records[i]['reward_used']
        else:
            food_left = self.total_food
            cumulated_reward = self.total_reward
        
        # Create table data
        table_data = [
            ['余粮', f'{food_left}'],
            ['总分', f'{cumulated_reward}']
        ]
        
        # Create table
        tbl = self._map_stats_ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(14)
        tbl.scale(1, 1.5)
        
        # Style cells
        for i in range(len(table_data)):
            tbl[(i, 0)].set_facecolor('#ffffcc')
            tbl[(i, 0)].set_text_props(weight='bold', fontsize=12)
            tbl[(i, 1)].set_facecolor('#e6f2ff')
            tbl[(i, 1)].set_text_props(fontsize=12)

    def _get_team_steps_for_day(self, team, day):
        """Calculate remaining steps available to a team on a given day.
        
        Days 1-3: Cumulative allocation (6, 12, 18)
        Days 4+: Fresh 6 per day with carryover, capped at 18
        """
        if team is None:
            return 0
        
        if day < team.created_day:
            return 0

        # Prefer day_records as single source of truth when available.
        if self.day_records and 1 <= day <= len(self.day_records):
            if team is self.team1:
                return max(0, self.day_records[day - 1].get('team1_steps_remain', 0))
            if team is self.team2:
                return max(0, self.day_records[day - 1].get('team2_steps_remain', 0))
            if team is self.team3:
                return max(0, self.day_records[day - 1].get('team3_steps_remain', 0))
        
        days_elapsed = day - team.created_day + 1
        
        if days_elapsed <= 3:
            # Days 1-3: cumulative allocation
            total_allocated = days_elapsed * 6
            total_consumed = sum(team._seg_steps)
            return max(0, min(total_allocated - total_consumed, 18))
        else:
            # Days 4+: fresh 6 per day with carryover
            # Determine team number
            if team == self.team1:
                team_num = 1
            elif team == self.team2:
                team_num = 2
            elif team == self.team3:
                team_num = 3
            else:
                return 0
            
            # Get yesterday's remaining from day_records
            if self.day_records and day > 1 and day - 2 < len(self.day_records):
                yesterday_remaining = self.day_records[day - 2].get(f'team{team_num}_steps_remain', 0)
            else:
                yesterday_remaining = 0
            
            # Add today's fresh 6
            available_today = min(yesterday_remaining + 6, 18)
            
            # Subtract today's consumption
            today_consumed = sum(seg_steps for seg_idx, seg_steps in enumerate(team._seg_steps) if team._seg_days[seg_idx] == day)
            
            return max(0, available_today - today_consumed)

    def _is_hex_settled(self, hex_pos):
        """True once some team has paid this hex's challenge (it appears in a
        segment's new-hex list). all_visited_hexes can't answer this: creating a
        team on another team's free-exploration hex adds it there for display
        before anyone has paid for it."""
        hex_pos = tuple(hex_pos)
        return any(
            tuple(h) == hex_pos
            for team in (self.team1, self.team2, self.team3) if team is not None
            for seg_new in team._seg_new_hexes
            for h in seg_new
        )

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
        challenge_food = _apply_challenge_discounts(
            _get_terrain_food(terrain, self.current_day),
            team,
            self._are_all_g_lands_visited(),
            is_tent=terrain.get('name') == 'Tent',
        )
        reward = terrain.get('award', 0)
        if team.z_bonus_remaining > 0:
            reward = _apply_z_bonus(reward, team)

        if self.current_food < challenge_food:
            self._status_msg = self._build_not_enough_food_message(
                challenge_food, steps_available
            )
            self._draw()
            return True

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
        team._seg_fly_skill_deltas.append(0)

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
        self._status_msg += self._activate_land_buffs(current_pos)
        self._rebuild_day_records()
        self._auto_save_game()
        self._draw()
        return True

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

    def _sync_current_food_for_view_day(self):
        """Sync current_food to the current viewing day using day_records."""
        # Always derive food from cumulative daily usage for robustness, then normalize
        # day_records food_remain so UI tables stay consistent during day navigation.
        base_food = 6800 + 1600 * (self.current_day - 1)
        if self.day_records:
            days_to_sum = min(max(self.current_day, 0), len(self.day_records))
            spent = sum(r.get('food_used', 0) for r in self.day_records[:days_to_sum])
            self.current_food = base_food - spent

            if 1 <= self.current_day <= len(self.day_records):
                self.day_records[self.current_day - 1]['food_remain'] = self.current_food
        else:
            self.current_food = base_food

    def _get_path_line_scale(self, xlim=None, ylim=None):
        """Return a zoom-aware scale factor based on current on-screen hex size."""
        def _hex_radius_px(xl, yl):
            try:
                bbox = self.ax.get_window_extent()
                ax_w = max(float(bbox.width), 1.0)
                ax_h = max(float(bbox.height), 1.0)
            except Exception:
                return None

            x_span = max(abs(float(xl[1]) - float(xl[0])), 1e-6)
            y_span = max(abs(float(yl[1]) - float(yl[0])), 1e-6)
            px_per_x = ax_w / x_span
            px_per_y = ax_h / y_span

            # Radius is anisotropically scaled in data-space by X/Y display transforms.
            rx = HEX_SIZE * 0.97 * X_SCALE * px_per_x
            ry = HEX_SIZE * 0.97 * Y_SCALE * px_per_y
            return max(min(rx, ry), 1e-6)

        if xlim is None or ylim is None:
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
        else:
            cur_xlim = xlim
            cur_ylim = ylim

        fit_xlim, fit_ylim = self._compute_fit_limits_for_axes()

        cur_hex_px = _hex_radius_px(cur_xlim, cur_ylim)
        base_hex_px = _hex_radius_px(fit_xlim, fit_ylim)
        if cur_hex_px is None or base_hex_px is None:
            return 1.0

        scale = cur_hex_px / base_hex_px
        return max(0.25, min(scale, 40.0))

    def _advance_day(self):
        """Manually advance to next day (just changes viewing index, doesn't modify team state)."""
        if self.current_day >= TOTAL_DAYS:
            self.current_day = TOTAL_DAYS
            self._status_msg = f'Already at Day {TOTAL_DAYS}. Cannot advance further.'
            self._draw()
            return

        # Advance to next day
        self.current_day += 1
        self._sync_current_food_for_view_day()
        
        # Calculate displayed steps for each team for the new current day
        team_steps = []
        if self.team1:
            steps = self._get_team_steps_for_day(self.team1, self.current_day)
            team_steps.append(f'T1: {steps}')
        if self.team2:
            steps = self._get_team_steps_for_day(self.team2, self.current_day)
            team_steps.append(f'T2: {steps}')
        if self.team3:
            steps = self._get_team_steps_for_day(self.team3, self.current_day)
            team_steps.append(f'T3: {steps}')
        steps_str = ' | '.join(team_steps)
        
        self._status_msg = f'Day {self.current_day} | Food: {self.current_food} | Steps: {steps_str}'
        # Only recenter if the active team actually acted on the newly-viewed
        # day; otherwise leave the view exactly where the user had it.
        self._center_view_on_active_team_day(self.current_day, move_if_empty=False)
        self._auto_save_game()  # Auto-save day change
        self._draw()

    def _go_previous_day(self):
        """Go back to previous day (just changes viewing index, doesn't modify team state)."""
        if self.current_day <= 1:
            self._status_msg = 'Already on Day 1. Cannot go back further.'
            self._draw()
            return
        
        # Go back to previous day
        self.current_day -= 1
        self._sync_current_food_for_view_day()
        
        # Calculate displayed steps for each team for the new current day
        team_steps = []
        if self.team1:
            steps = self._get_team_steps_for_day(self.team1, self.current_day)
            team_steps.append(f'T1: {steps}')
        if self.team2:
            steps = self._get_team_steps_for_day(self.team2, self.current_day)
            team_steps.append(f'T2: {steps}')
        if self.team3:
            steps = self._get_team_steps_for_day(self.team3, self.current_day)
            team_steps.append(f'T3: {steps}')
        steps_str = ' | '.join(team_steps)
        
        self._status_msg = f'Day {self.current_day} | Food: {self.current_food} | Steps: {steps_str}'
        # Only recenter if the active team actually acted on the newly-viewed
        # day; otherwise leave the view exactly where the user had it.
        self._center_view_on_active_team_day(self.current_day, move_if_empty=False)
        self._auto_save_game()  # Auto-save day change
        self._draw()

    def _jump_to_day(self, target_day):
        """Jump directly to a specific day and refresh view state."""
        try:
            day = int(target_day)
        except Exception:
            return

        day = max(1, min(TOTAL_DAYS, day))
        self.current_day = day
        self._sync_current_food_for_view_day()

        team_steps = []
        if self.team1:
            team_steps.append(f'T1: {self._get_team_steps_for_day(self.team1, self.current_day)}')
        if self.team2:
            team_steps.append(f'T2: {self._get_team_steps_for_day(self.team2, self.current_day)}')
        if self.team3:
            team_steps.append(f'T3: {self._get_team_steps_for_day(self.team3, self.current_day)}')
        steps_str = ' | '.join(team_steps)

        self._status_msg = f'Day {self.current_day} | Food: {self.current_food} | Steps: {steps_str}'
        # Only recenter if the active team actually acted on the newly-viewed
        # day; otherwise leave the view exactly where the user had it.
        self._center_view_on_active_team_day(self.current_day, move_if_empty=False)
        self._auto_save_game()
        self._draw()

    def _show_day_picker(self):
        """Show a list picker for Day 1..TOTAL_DAYS and jump to selected day."""
        try:
            import tkinter as tk

            picker = tk.Tk()
            picker.title('Jump To Day')
            picker.geometry('220x340')
            picker.attributes('-topmost', True)

            frame = tk.Frame(picker)
            frame.pack(fill='both', expand=True, padx=8, pady=8)

            lbl = tk.Label(frame, text='Select Day')
            lbl.pack(anchor='w')

            listbox = tk.Listbox(frame, height=14)
            scrollbar = tk.Scrollbar(frame, orient='vertical', command=listbox.yview)
            listbox.configure(yscrollcommand=scrollbar.set)

            for d in range(1, TOTAL_DAYS + 1):
                listbox.insert('end', self._format_day_with_date(d))

            listbox.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')

            try:
                listbox.selection_set(self.current_day - 1)
                listbox.see(self.current_day - 1)
            except Exception:
                pass

            def _go_selected(_evt=None):
                sel = listbox.curselection()
                if not sel:
                    return
                day = sel[0] + 1
                picker.destroy()
                self._jump_to_day(day)

            btn_row = tk.Frame(picker)
            btn_row.pack(fill='x', padx=8, pady=(0, 8))

            btn_go = tk.Button(btn_row, text='Go', command=_go_selected)
            btn_cancel = tk.Button(btn_row, text='Cancel', command=picker.destroy)
            btn_go.pack(side='left', expand=True, fill='x', padx=(0, 4))
            btn_cancel.pack(side='left', expand=True, fill='x', padx=(4, 0))

            listbox.bind('<Double-1>', _go_selected)
            listbox.bind('<Return>', _go_selected)

            picker.mainloop()
        except Exception as e:
            self._status_msg = f'Day picker error: {str(e)}'
            self._draw()

    # ── Day management helpers ─────────────────────────────────────────────────

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

    # ── Mode / interaction ────────────────────────────────────────────────────

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

        # Remove any no-draw edges that involve the removed segment
        # This cleans up portal teleport markers when undoing
        edges_to_remove = set()
        for edge in self.active_team._no_draw_edges:
            # Check if either endpoint of the edge is in the removed segment
            if edge[0] in self.active_team.full_path[-length-1:-1] or edge[1] in self.active_team.full_path[-length-1:-1]:
                edges_to_remove.add(edge)
        self.active_team._no_draw_edges -= edges_to_remove
        
        # Clear X and B bonuses when undoing (they were tied to the undone move)
        self.active_team.x_bonus_remaining = 0
        self.active_team.x_bonus_name = None
        self.active_team.b_discount_remaining = 0
        self.active_team.b_discount_name = None
        self.active_team.z_bonus_remaining = 0
        self.active_team.z_bonus_name = None

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
        text = '☑ 显示buff' if self._chk_state else '☐ 显示buff'
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

    def _update_show_future_button_state(self):
        """Refresh the future-path toggle button visual state."""
        if not hasattr(self, '_btn_show_future') or self._btn_show_future is None:
            return

        text = '☑ 显示未来' if self._show_future_paths else '☐ 显示未来'
        self._btn_show_future.label.set_text(text)
        self._btn_show_future.color = '#90EE90' if self._show_future_paths else '#FFCCCC'
        self._btn_show_future.hovercolor = '#7FDF7F' if self._show_future_paths else '#FFB3B3'

    def _toggle_show_future_paths(self):
        """Toggle map visibility for segments whose day is later than current day."""
        self._show_future_paths = not self._show_future_paths
        state_text = '显示' if self._show_future_paths else '隐藏'
        self._status_msg = f'未来路径已{state_text}'
        self._update_show_future_button_state()
        self._draw()

    def _resolve_asset_path(self, filename):
        """Find `filename` next to the running script/exe (same search-dir logic
        used for the Excel export template, so it works both frozen and unfrozen).
        Returns the path if found, else None.
        """
        import os
        import sys

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

        for d in search_dirs:
            cand = os.path.join(d, filename)
            if os.path.exists(cand):
                return cand
        return None

    def _load_map_image(self):
        """Load and cache the pre-rectified real-game map screenshot plus its
        placement extent. See the _MAP_IMAGE_FILENAME comment above for how
        it was rectified - it's a single whole-image affine placement (not a
        per-hex warp), so its extent generally isn't the full hex-grid
        bounds and must be read from the JSON sidecar the offline script
        produced, not recomputed here."""
        if self._map_image_array is not None:
            return True

        path = self._resolve_asset_path(_MAP_IMAGE_FILENAME)
        if path is None:
            self._status_msg = f'Map image not found: {_MAP_IMAGE_FILENAME} (place it next to the app).'
            return False

        extent_path = self._resolve_asset_path(_MAP_IMAGE_EXTENT_FILENAME)
        if extent_path is None:
            self._status_msg = f'Map image extent not found: {_MAP_IMAGE_EXTENT_FILENAME} (place it next to the app).'
            return False

        try:
            from PIL import Image as _PILImage
            img = np.asarray(_PILImage.open(path))
            with open(extent_path, 'r', encoding='utf-8') as f:
                extent = json.load(f)['extent']
        except Exception as e:
            self._status_msg = f'Failed to load map image: {e}'
            return False

        self._map_image_array = img
        self._map_image_extent = extent
        return True

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

    def _copy_map_screenshot(self):
        """Copy the information layout to the clipboard without UI buttons."""
        try:
            self.fig.canvas.draw()
            image = np.asarray(self.fig.canvas.buffer_rgba()).copy()
            canvas_height, canvas_width = image.shape[:2]

            # Keep map, date, statistics, and action symbols, but remove every
            # interactive button from the captured canvas.
            button_names = (
                '_btn_undo', '_btn_reset', '_btn_fly', '_btn_show_future',
                '_btn_chk_labels', '_btn_prev_day', '_btn_next_day',
                '_btn_map_view', '_btn_screenshot', '_btn_load', '_btn_save',
                '_btn_global_stat', '_btn_edit_seg', '_btn_export_xlsx',
            )
            button_mask_padding = 3
            for button_name in button_names:
                button = getattr(self, button_name, None)
                if button is None:
                    continue
                bbox = button.ax.get_window_extent()
                left = max(0, int(np.floor(bbox.x0)) - button_mask_padding)
                right = min(canvas_width, int(np.ceil(bbox.x1)) + button_mask_padding)
                top = max(0, canvas_height - int(np.ceil(bbox.y1)) - button_mask_padding)
                bottom = min(canvas_height, canvas_height - int(np.floor(bbox.y0)) + button_mask_padding)
                rgb = tuple(int(channel * 255) for channel in hex2color(APP_BACKGROUND_COLOR))
                image[top:bottom, left:right] = (*rgb, 255)

            height, width = image.shape[:2]
            # CF_DIB expects bottom-up BGRA pixel order.
            bgra = np.flipud(image[:, :, [2, 1, 0, 3]]).tobytes()
            header = struct.pack(
                '<IiiHHIIiiII',
                40, width, height, 1, 32, 0, len(bgra), 0, 0, 0, 0,
            )
            clipboard_data = header + bgra

            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
            kernel32.GlobalFree.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.restype = ctypes.c_int
            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = ctypes.c_int
            user32.EmptyClipboard.restype = ctypes.c_int
            user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            user32.SetClipboardData.restype = ctypes.c_void_p
            user32.CloseClipboard.restype = ctypes.c_int
            handle = kernel32.GlobalAlloc(0x0002, len(clipboard_data))
            if not handle:
                raise RuntimeError('Could not allocate clipboard memory.')
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                kernel32.GlobalFree(handle)
                raise RuntimeError('Could not lock clipboard memory.')
            ctypes.memmove(pointer, clipboard_data, len(clipboard_data))
            kernel32.GlobalUnlock(handle)

            if not user32.OpenClipboard(None):
                kernel32.GlobalFree(handle)
                raise RuntimeError('Clipboard is currently unavailable.')
            try:
                user32.EmptyClipboard()
                if not user32.SetClipboardData(8, handle):  # CF_DIB
                    kernel32.GlobalFree(handle)
                    raise RuntimeError('Could not write image to clipboard.')
                handle = None  # Clipboard owns this memory after SetClipboardData.
            finally:
                user32.CloseClipboard()

            self._status_msg = '地图截图已复制到剪切板。'
        except Exception as e:
            self._status_msg = f'截图失败: {e}'
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
    
    def _auto_save_game(self):
        """Auto-save game state to a default auto-save file (silent, no dialog)."""
        try:
            import os
            project_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(project_dir, 'save', 'autosave')
            os.makedirs(save_dir, exist_ok=True)

            latest_move_day = 1
            for team in [self.team1, self.team2, self.team3]:
                if team and team._seg_days:
                    latest_move_day = max(latest_move_day, max(team._seg_days))

            file_path = os.path.join(save_dir, f'auto_save_day{latest_move_day}.json')
            
            # Serialize team data
            def serialize_team(team):
                if team is None:
                    return None
                return {
                    'full_path': list(team.full_path),
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
            
            game_state = {
                'current_day': self.current_day,
                'current_food': self.current_food,
                'total_food': self.total_food,
                'total_reward': self.total_reward,
                'fly_skill_limit': self.fly_skill_limit,
                'team1': serialize_team(self.team1),
                'team2': serialize_team(self.team2),
                'team3': serialize_team(self.team3),
                'active_team_num': 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3),
                'all_visited_hexes': [list(h) for h in self.all_visited_hexes],
                'visited_g_lands': [list(h) for h in self.visited_g_lands],
                'day_records': self.day_records,
            }
            
            with open(file_path, 'w') as f:
                json.dump(game_state, f, indent=2)
        except Exception as e:
            print(f'Auto-save error: {e}')

    def _segment_iter(self, team):
        """Yield segment tuples with stable start/end resolution.

        Returns tuples:
          (seg_idx, seg_day, seg_start, seg_end, seg_actions, seg_len)
        """
        if team is None:
            return

        cursor = 1
        for seg_idx, seg_len in enumerate(team._seg_lengths):
            if seg_len <= 0:
                continue
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
                    seg_new_set = {
                        tuple(h) for h in seg_new_hexes
                        if isinstance(h, (list, tuple)) and len(h) == 2
                    }
                    seg_jump_set = {
                        tuple(h) for h in seg_jumps
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
            day_events.setdefault(ev['day'], []).append(ev)

        for day in sorted(day_events.keys()):
            pending = list(day_events[day])

            while pending:
                chosen_idx = 0
                # If exactly one G/g is missing globally, process the segment that
                # captures that decisive last G/g before other teams on this day.
                if len(unvisited_g) == 1:
                    decisive_pos = next(iter(unvisited_g))
                    g_first_idx = next(
                        (
                            i for i, pev in enumerate(pending)
                            if any(ent.get('g_pos') == decisive_pos for ent in pev.get('entries', []))
                        ),
                        None
                    )
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

                operations_by_day_team[key].extend(e.get('label', '') for e in entries)

                for e in entries:
                    g_pos = e.get('g_pos')
                    if g_pos in unvisited_g:
                        unvisited_g.remove(g_pos)

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

        for key, ops in operations_by_day_team.items():
            operations_by_day_team[key] = _merge_consecutive_jump_labels(ops)

        return operations_by_day_team

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
                import importlib
                openpyxl_mod = importlib.import_module('openpyxl')
                load_workbook = openpyxl_mod.load_workbook
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

            template_names = ['S24_分表2.0_导出模板.xlsx', 'S24_分表2.0.xlsx']
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
                self._status_msg = 'Export error: template not found (S24_分表2.0_导出模板.xlsx / S24_分表2.0.xlsx).'
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

            output_path = os.path.join(base_dir, 'S24_分表2.0_export.xlsx')

            wb = load_workbook(template_path)

            operations_by_day_team = self._build_excel_operations_by_day_team()
            day_food_adj, day_reward_adj, _day_team_step_bonus = self._compute_bxz_adjustments()

            # Rebuild day records so we can write final end-action values from canonical state.
            self._rebuild_day_records()

            op_rows = {1: 8, 2: 11, 3: 14}
            end_action_rows = {1: 2, 2: 3, 3: 4}
            day_record_end_key = {1: 'team1_steps_remain', 2: 'team2_steps_remain', 3: 'team3_steps_remain'}

            # C..AA => 25 operation slots.
            op_start_col = 3
            op_end_col = 27

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
            from tkinter import filedialog
            import os
            
            print('DEBUG: Starting save process...')
            
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.asksaveasfilename(
                defaultextension='.json',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
                initialfile='game_save.json'
            )
            
            if not file_path:
                print('DEBUG: User cancelled save dialog')
                root.destroy()
                return
            
            print(f'DEBUG: Selected file path: {file_path}')
            print(f'DEBUG: File path exists before write: {os.path.exists(file_path)}')
            
            # Serialize team data
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
            
            print('DEBUG: Serializing game state...')
            game_state = {
                'current_day': self.current_day,
                'current_food': self.current_food,
                'total_food': self.total_food,
                'total_reward': self.total_reward,
                'fly_skill_limit': self.fly_skill_limit,
                'team1': serialize_team(self.team1),
                'team2': serialize_team(self.team2),
                'team3': serialize_team(self.team3),
                'active_team_num': 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3),
                'all_visited_hexes': [list(h) for h in self.all_visited_hexes],
                'visited_g_lands': [list(h) for h in self.visited_g_lands],
                'day_records': self.day_records,
            }
            
            print(f'DEBUG: Writing to file: {file_path}')
            with open(file_path, 'w') as f:
                json.dump(game_state, f, indent=2)
            
            print(f'DEBUG: File write complete. File size: {os.path.getsize(file_path)} bytes')
            
            root.destroy()
            self._status_msg = f'Game saved to {os.path.basename(file_path)}'
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
    
    def _load_game(self):
        """Load game state from a JSON file."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            import os
            
            print('DEBUG: Starting load process...')
            
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
            )
            
            if not file_path:
                print('DEBUG: User cancelled load dialog')
                root.destroy()
                return
            
            print(f'DEBUG: Selected file path: {file_path}')
            print(f'DEBUG: File exists: {os.path.exists(file_path)}')
            
            print('DEBUG: Loading JSON file...')
            with open(file_path, 'r') as f:
                game_state = json.load(f)
            
            # Deserialize team data
            def deserialize_team(team_data):
                if team_data is None:
                    return None
                print('DEBUG: Deserializing team...')
                team = Team(tuple(team_data['origin']), created_day=team_data['created_day'])
                team.full_path = [tuple(h) for h in team_data['full_path']]
                team.visited_hexes = set(tuple(h) for h in team_data['visited_hexes'])
                team.free_exploration_hexes = set(tuple(h) for h in team_data['free_exploration_hexes'])
                team.x_bonus_remaining = team_data['x_bonus_remaining']
                team.x_bonus_name = team_data['x_bonus_name']
                team.b_discount_remaining = team_data['b_discount_remaining']
                team.b_discount_name = team_data['b_discount_name']
                team.z_bonus_remaining = team_data['z_bonus_remaining']
                team.z_bonus_name = team_data['z_bonus_name']
                team.max_day_reached = team_data['max_day_reached']
                team._seg_foods = team_data['_seg_foods']
                team._seg_steps = team_data['_seg_steps']
                team._seg_awards = team_data['_seg_awards']
                team._seg_days = team_data['_seg_days']
                team._seg_new_hexes = [[tuple(h) for h in seg] for seg in team_data['_seg_new_hexes']]
                team._seg_exploration_hexes = [[tuple(h) for h in seg] for seg in team_data['_seg_exploration_hexes']]
                team._seg_jumps = [[tuple(h) for h in seg] for seg in team_data.get('_seg_jumps', [])]
                team._seg_path_nodes = [[tuple(h) for h in seg] for seg in team_data.get('_seg_path_nodes', [])]
                team._seg_end_positions = [tuple(h) for h in team_data.get('_seg_end_positions', [])]
                team._seg_action_sequence = [[(action, tuple(h) if isinstance(h, (list, tuple)) else h) for action, h in seg] for seg in team_data.get('_seg_action_sequence', [])]
                team._seg_lengths = team_data['_seg_lengths']
                team._seg_action_orders = team_data.get('_seg_action_orders', [None] * len(team._seg_lengths))
                team._seg_hex_costs = team_data['_seg_hex_costs']
                team._seg_is_fly_skill = team_data['_seg_is_fly_skill']
                team._seg_fly_skill_deltas = team_data.get('_seg_fly_skill_deltas', [0] * len(team._seg_lengths))
                team._no_draw_edges = set((tuple(e[0]), tuple(e[1])) for e in team_data['_no_draw_edges'])
                return team
            
            print('DEBUG: Restoring game state...')
            # Restore game state
            self.current_day = game_state['current_day']
            self.current_food = game_state['current_food']
            self.total_food = game_state['total_food']
            self.total_reward = game_state['total_reward']
            self.fly_skill_limit = game_state['fly_skill_limit']
            self.team1 = deserialize_team(game_state['team1'])
            self.team2 = deserialize_team(game_state['team2'])
            self.team3 = deserialize_team(game_state['team3'])
            self._ensure_team_segment_path_nodes(self.team1)
            self._ensure_team_segment_path_nodes(self.team2)
            self._ensure_team_segment_path_nodes(self.team3)
            saved_orders = [
                order
                for team in (self.team1, self.team2, self.team3) if team is not None
                for order in team._seg_action_orders
                if isinstance(order, int)
            ]
            self._next_action_order = max(saved_orders, default=-1) + 1
            self.all_visited_hexes = set(tuple(h) for h in game_state['all_visited_hexes'])
            self.visited_g_lands = set(tuple(h) for h in game_state['visited_g_lands'])
            self.day_records = game_state['day_records']
            
            # Set active team
            active_team_num = game_state['active_team_num']
            if active_team_num == 1:
                self.active_team = self.team1
            elif active_team_num == 2:
                self.active_team = self.team2
            else:
                self.active_team = self.team3
            
            # Update legacy set-team button visibility safely (buttons may not exist in current UI).
            self._set_legacy_set_team_buttons_visibility()
            
            root.destroy()
            
            self._update_switch_button_color()
            self._update_fly_button_state()

            needs_rebalance = self._has_global_food_deficit_from_segments()
            if not needs_rebalance:
                for team in (self.team1, self.team2, self.team3):
                    if team is not None and self._team_has_step_overflow_from(team, team.created_day):
                        needs_rebalance = True
                        break

            if needs_rebalance:
                self._rebalance_all_teams_from_day(1)

            # Reconstruct shared derived runtime state from canonical segment history.
            self._rebuild_shared_derived_state_from_segments()
            
            # Rebuild day records from segment data (ensures consistency with loaded segments)
            # and sync current_food with the rebuilt food_remain for the current day.
            # This fixes cases where current_food in the save is out of sync with actual segment history.
            self._rebuild_day_records()
            if self.day_records and 1 <= self.current_day <= len(self.day_records):
                self.current_food = self.day_records[self.current_day - 1]['food_remain']
            
            self._status_msg = f'Game loaded from {os.path.basename(file_path)}'
            self._has_zoomed = False
            self._draw()
            print(f'Game loaded successfully from {file_path}')
        except Exception as e:
            print(f'Error loading game: {e}')
            import traceback
            traceback.print_exc()
            try:
                root.destroy()
            except:
                pass
            self._status_msg = f'Load error: {str(e)}'
            self._draw()
    
    def _init_day_records(self):
        """Pre-generate day records for days 1-TOTAL_DAYS with base resources (no team moves yet).
        
        Each day gets:
        - Day 1: 6800 food, 6 steps per team
        - Day N (N>1): base_food = 6800 + 1600*(N-1), 6*N steps per team (remaining capped at 18)
        """
        self.day_records = []
        for day in range(1, TOTAL_DAYS + 1):
            # Food: starts at 6800, +1600 per day
            food_available = 6800 + 1600 * (day - 1)
            
            # Calculate allocated steps for each team on this day (assuming day 1 start)
            # Always add 6 per day, no cap on allocation
            days_elapsed = day
            allocated_steps = days_elapsed * 6
            # But remaining will be capped at 18 in _rebuild_day_records()
            
            self.day_records.append({
                'day': day,
                'food_used': 0,          # Will be updated as teams make moves
                'reward_used': 500 if (day == 1 or (day - 1) % 7 == 3) else 0,  # Pre-seed Monday bonus
                'food_remain': food_available,  # Base food available (before moves)
                'team1_steps_remain': min(allocated_steps, 18),  # Capped at 18 for init
                'team2_steps_remain': min(allocated_steps, 18),  # Capped at 18 for init
                'team3_steps_remain': min(allocated_steps, 18),  # Capped at 18 for init
            })
    
    def _rebuild_day_records(self):
        """Rebuild day_records by summing segment data for each day.
        
        This updates the food_used, reward_used, and per-team remaining steps for each day
        based on teams' moves, while preserving the pre-generated day structure for all TOTAL_DAYS days.
        """
        # Initialize pre-generated days (1-TOTAL_DAYS) with base resources
        self._init_day_records()
        
        # Collect all segments from all teams
        all_teams = [self.team1] + ([self.team2] if self.team2 else []) + ([self.team3] if self.team3 else [])
        
        # Sum segments by day
        day_totals = {}  # day -> {food_used, reward_used}
        team_steps_by_day = {team: {} for team in all_teams}  # team -> day -> steps_used
        team_key_map = {}
        if self.team1 is not None:
            team_key_map[self.team1] = 'team1_steps_remain'
        if self.team2 is not None:
            team_key_map[self.team2] = 'team2_steps_remain'
        if self.team3 is not None:
            team_key_map[self.team3] = 'team3_steps_remain'
        
        for team in all_teams:
            for seg_steps, seg_food, seg_reward, seg_day in zip(team._seg_steps, team._seg_foods, team._seg_awards, team._seg_days):
                if seg_day not in day_totals:
                    day_totals[seg_day] = {'food': 0, 'reward': 0}
                day_totals[seg_day]['food'] += seg_food
                day_totals[seg_day]['reward'] += seg_reward
                
                # Track steps per team per day
                if seg_day not in team_steps_by_day[team]:
                    team_steps_by_day[team][seg_day] = 0
                team_steps_by_day[team][seg_day] += seg_steps
        
        # Update day_records with totals and per-team remaining steps
        current_food = 6800  # Starting food
        # Step bank per team. Positive bank is capped at 18. Negative bank carries debt to future days,
        # allowing edit-day overuse to auto-reduce future available steps.
        team_step_bank = {team: 0 for team in all_teams}
        
        for day in range(1, TOTAL_DAYS + 1):
            if day in day_totals:
                food_used = day_totals[day]['food']
                reward_used = day_totals[day]['reward']
                # Add global Monday bonus unconditionally for all qualifying days
                if day == 1 or (day - 1) % 7 == 3:
                    reward_used += 500
                self.day_records[day - 1]['food_used'] = food_used
                self.day_records[day - 1]['reward_used'] = reward_used
                current_food -= food_used
            else:
                # No team moves on this day — still apply Monday bonus if qualifying
                if day == 1 or (day - 1) % 7 == 3:
                    self.day_records[day - 1]['reward_used'] = 500
                else:
                    self.day_records[day - 1]['reward_used'] = 0
                current_food -= self.day_records[day - 1]['food_used']
            
            self.day_records[day - 1]['food_remain'] = current_food
            
            # Calculate remaining steps per team for this day.
            for team in all_teams:
                team_key = team_key_map.get(team)
                if team_key is None:
                    continue

                if team.created_day > day:
                    self.day_records[day - 1][team_key] = 0
                    continue

                steps_used_today = team_steps_by_day[team].get(day, 0)
                team_step_bank[team] += 6
                if team_step_bank[team] > 18:
                    team_step_bank[team] = 18
                team_step_bank[team] -= steps_used_today

                self.day_records[day - 1][team_key] = max(0, min(team_step_bank[team], 18))
            
            # Add 1600 food for next day
            if day < TOTAL_DAYS:
                current_food += 1600
        
        # Recompute total_reward from day_records (single source of truth)
        self.total_reward = sum(r['reward_used'] for r in self.day_records)
        self.total_food = sum(
            sum(team._seg_foods)
            for team in (self.team1, self.team2, self.team3)
            if team is not None
        )

        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue
            latest_day = max(team.created_day, getattr(team, 'max_day_reached', team.created_day))
            latest_day = max(1, min(latest_day, TOTAL_DAYS))
            team.steps = self._get_team_steps_for_day(team, latest_day)

        # Keep current food aligned with the currently viewed day.
        self._sync_current_food_for_view_day()

    def _pixel_to_hex(self, px, py):
        """Return the (ir, ic) of the hex closest to pixel (px, py)."""
        px = px / X_SCALE
        py = py / Y_SCALE
        ic0 = int(round(px / (1.5 * HEX_SIZE)))
        best, best_d2 = None, float('inf')
        for ic in range(max(0, ic0 - 2), min(COLS, ic0 + 3)):
            y_shift = (np.sqrt(3) / 2 * HEX_SIZE) if ic % 2 == 1 else 0.0
            ir0 = int(round((py - y_shift) / (np.sqrt(3) * HEX_SIZE)))
            for ir in range(max(0, ir0 - 2), min(ROWS, ir0 + 3)):
                cx, cy = _center(ir, ic)
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 < best_d2:
                    best_d2, best = d2, (ir, ic)
        return best

    def _on_press(self, event):
        """Handle mouse button press - track right-click for panning."""
        if event.button == 3:  # Right mouse button
            self._pan_active = True
            self._pan_start_x = event.xdata
            self._pan_start_y = event.ydata
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()
            # Freeze the data<->pixel transform at drag start. Motion deltas must be
            # measured against this fixed transform, not the live one (which itself
            # shifts every time we call set_xlim/set_ylim below), otherwise the
            # reference frame moves out from under the drag and the map jitters.
            self._pan_transform = self.ax.transData.inverted()

    def _switch_to_team_and_latest_plus_one_day(self, team_num):
        """Switch to team and jump to one day after that team's latest move day."""
        self._switch_to_team(team_num)

        target_team = self.team1 if team_num == 1 else (self.team2 if team_num == 2 else self.team3)
        if target_team is None:
            return

        # Prefer the most reliable/latest day marker across persisted and runtime state.
        latest_move_day = target_team.created_day
        if target_team._seg_days:
            latest_move_day = max(latest_move_day, max(target_team._seg_days))
        latest_move_day = max(latest_move_day, getattr(target_team, 'max_day_reached', target_team.created_day))
        target_day = latest_move_day + 1

        # Clamp into the supported day-record range.
        target_day = max(1, min(target_day, TOTAL_DAYS))

        print(
            f'[TEAM_DAY_JUMP] team={team_num}, latest_move_day={latest_move_day}, '
            f'target_day={target_day}, current_day_before={self.current_day}'
        )

        self.current_day = target_day
        self._sync_current_food_for_view_day()
        self._status_msg = (
            f'Switched to Team {team_num} and jumped to Day {self.current_day} '
            f'(shared food remaining: {self.current_food})'
        )
        self._draw()

    def _on_team_button_click(self, team_num):
        """Handle single/double click behavior for team buttons."""
        target_team = self.team1 if team_num == 1 else (self.team2 if team_num == 2 else self.team3)
        if target_team is not None and self.active_team is target_team:
            # Robust path: second click (or any click on current team button) jumps day.
            self._switch_to_team_and_latest_plus_one_day(team_num)
            return

        now = time.monotonic()
        last = self._team_button_last_click_time.get(team_num, 0.0)
        self._team_button_last_click_time[team_num] = now

        # Second click within window => treat as double-click action.
        if now - last <= self._team_button_dblclick_window_sec:
            self._team_button_last_click_time[team_num] = 0.0
            self._switch_to_team_and_latest_plus_one_day(team_num)
            return

        # First click => normal team switch.
        self._switch_to_team(team_num)

    def _on_release(self, event):
        """Handle mouse button release - clear pan state."""
        self._pan_active = False
        self._pan_start_x = None
        self._pan_start_y = None

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
            if day_locked and not self._is_active_day_edit():
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
                
                try:
                    t = _terrain(*pos)
                    is_new_hex = pos not in self.all_visited_hexes
                    # Landing by fly on an untaken, capturable (step > 0), non-portal
                    # hex does not occupy it yet: like a probe step, it is settled
                    # (challenge food + reward + 1 step) when the team leaves it.
                    # Portals (step 0), Tents and other step<=0 terrain keep the
                    # immediate handling below.
                    defer_capture = is_new_hex and t.get('step', 1) > 0

                    # Flying off a still-unsettled probe / deferred-fly hex settles
                    # it, exactly like walking off it does.
                    is_leaving_exploration = (
                        current_pos in self.active_team.free_exploration_hexes
                        and not self._is_hex_settled(current_pos)
                    )
                    dep_food = dep_award = 0
                    if is_leaving_exploration:
                        dep_t = _terrain(*current_pos)
                        dep_food = _apply_challenge_discounts(
                            _get_terrain_food(dep_t, self.current_day), self.active_team,
                            self._are_all_g_lands_visited(), is_tent=dep_t.get('name') == 'Tent')
                        dep_award = dep_t['award']
                        if self.active_team.z_bonus_remaining > 0:
                            dep_award = _apply_z_bonus(dep_award, self.active_team)

                    # Calculate base challenge cost for destination terrain.
                    # Fly usually waives movement food, but BigBoss still charges movement food.
                    challenge_food = _get_terrain_food(t, self.current_day)
                    challenge_food = _apply_challenge_discounts(
                        challenge_food,
                        self.active_team,
                        self._are_all_g_lands_visited(),
                        is_tent=t.get('name') == 'Tent'
                    )
                    if defer_capture:
                        challenge_food = 0
                    movement_food = 0
                    if t.get('name') == 'bigBoss':
                        movement_food = _apply_g_reduction(50, self._are_all_g_lands_visited())
                    seg_food = challenge_food + movement_food
                    self._last_landing_cost_breakdown[pos] = {
                        'challenge': challenge_food,
                        'movement': movement_food,
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
                    allow_borrow_edit_resources = self._is_active_day_edit() and self.current_day == self._day_edit_context['day']
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

                    # Captured hexes (departure settle + landing) count as movements
                    # for existing B/Z buffs; a deferred landing does not.
                    movement_count = len(captured)

                    if movement_count > 0 and self.active_team.b_discount_remaining > 0:
                        self.active_team.b_discount_remaining = max(0, self.active_team.b_discount_remaining - movement_count)
                        if self.active_team.b_discount_remaining == 0:
                            self._status_msg += ' B discount expired.'

                    if movement_count > 0 and self.active_team.z_bonus_remaining > 0:
                        self.active_team.z_bonus_remaining = max(0, self.active_team.z_bonus_remaining - movement_count)
                        if self.active_team.z_bonus_remaining == 0:
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
                    
                    self._rebuild_day_records()
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
                if self._confirm_settle_current_exploration():
                    return
                return

            segment, cost_map = _astar(current, pos, set())
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

            # Get the display steps for the current viewing day (use this for all step checks)
            steps_available_for_day = self._get_team_steps_for_day(self.active_team, self.current_day)

            # Check for free exploration mode: if team has 0 steps but can move to adjacent untaken hex
            free_exploration = False
            if steps_available_for_day == 0 and self.active_team.x_bonus_remaining <= 0 and len(added) == 1:
                hex_to_explore = added[0]
                if hex_to_explore not in self.all_visited_hexes:
                    explore_terrain = _terrain(*hex_to_explore)
                    if explore_terrain.get('step', 1) > 0:
                        if not is_leaving_exploration:
                            free_exploration = True

            # If leaving a free exploration hex, calculate its challenge food cost
            departure_challenge = 0
            departure_is_tent = False
            if is_leaving_exploration:
                _dep_terrain = _terrain(*current_pos)
                departure_challenge = _get_terrain_food(_dep_terrain, self.current_day)
                departure_is_tent = _dep_terrain.get('name') == 'Tent'
            
            # Calculate costs
            # Apply B discount and G reduction additively to departure challenge
            seg_food = _apply_challenge_discounts(departure_challenge, self.active_team, self._are_all_g_lands_visited(), is_tent=departure_is_tent)
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
                    revisit_cost = _apply_g_reduction(revisit_cost, self._are_all_g_lands_visited())
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
                    # Apply B discount first, then G/g reduction
                    # Apply B discount and G reduction additively to challenge food
                    challenge_cost = _apply_challenge_discounts(terrain_food, self.active_team, self._are_all_g_lands_visited(), is_tent=t.get('name') == 'Tent')
                    movement_cost = _apply_g_reduction(50, self._are_all_g_lands_visited())
                    cost = movement_cost + challenge_cost
                    seg_food += cost
                    seg_award += terrain_reward
                    seg_steps += terrain_step_cost
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
                    exploration_cost = _apply_g_reduction(exploration_cost, self._are_all_g_lands_visited())
                    seg_food += exploration_cost
                    seg_award += 0
                    seg_steps += 0
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
                    challenge_with_all_discounts = _apply_challenge_discounts(challenge_food, self.active_team, self._are_all_g_lands_visited(), is_tent=t.get('name') == 'Tent')
                    terrain_reward = t['award']
                    # Apply Z bonus if active
                    if self.active_team.z_bonus_remaining > 0:
                        terrain_reward = _apply_z_bonus(terrain_reward, self.active_team)
                    # Apply G/g reduction: movement cost (50) always reduced
                    movement_cost = _apply_g_reduction(50, self._are_all_g_lands_visited())
                    total_cost = movement_cost + challenge_with_all_discounts
                    seg_food += total_cost
                    seg_award += terrain_reward
                    seg_steps += terrain_step_cost
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
                allow_borrow_edit_resources = self._is_active_day_edit() and seg_day == self._day_edit_context['day']
                if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                    raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_available_for_day))
            elif free_exploration and needs_day_check_for_departure:
                # Check if resources are insufficient
                allow_borrow_edit_resources = self._is_active_day_edit() and seg_day == self._day_edit_context['day']
                if (not allow_borrow_edit_resources) and self.current_food < seg_food:
                    raise RuntimeError(self._build_not_enough_food_message(seg_food, steps_available_for_day))
            else:
                # Check if resources are insufficient
                allow_borrow_edit_resources = self._is_active_day_edit() and seg_day == self._day_edit_context['day']
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
            
            # Count movements: only NEW hexes count as movements for B discount
            # Revisiting taken hexes doesn't decrement the discount counter
            movement_count = len(new_hexes)
            if movement_count > 0 and self.active_team.b_discount_remaining > 0:
                self.active_team.b_discount_remaining -= movement_count
                if self.active_team.b_discount_remaining <= 0:
                    self.active_team.b_discount_remaining = 0
                    self._status_msg = f'{self._status_msg} B discount expired.'
            
            # Count movements: only NEW hexes count as movements for Z bonus
            # Revisiting taken hexes doesn't decrement the bonus counter
            if movement_count > 0 and self.active_team.z_bonus_remaining > 0:
                self.active_team.z_bonus_remaining -= movement_count
                if self.active_team.z_bonus_remaining <= 0:
                    self.active_team.z_bonus_remaining = 0
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

    def _on_scroll(self, event):
        """Handle mouse scroll for zoom in/out."""
        if event.inaxes != self.ax:
            return  # Only zoom if scrolling over the main map
        
        # Mark that user has zoomed
        self._has_zoomed = True
        
        # Get current axis limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # Get event location (in data coordinates)
        xdata = event.xdata
        ydata = event.ydata
        
        # Zoom factor: scroll up = zoom in (0.8), scroll down = zoom out (1.2)
        if event.button == 'up':
            scale_factor = 0.8  # Zoom in
        elif event.button == 'down':
            scale_factor = 1.2  # Zoom out
        else:
            return
        
        # Calculate new limits centered on cursor
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.fig.canvas.draw_idle()

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

    # ── Portal teleportation ────────────────────────────────────────────────────

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

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        # Save zoom limits before clearing if user has zoomed
        saved_xlim = None
        saved_ylim = None
        if self._has_zoomed:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        
        self.ax.clear()
        self.ax.set_facecolor(APP_BACKGROUND_COLOR)

        # ax.clear() just destroyed every artist that was in the axes, including
        # any live hover-preview line - but it doesn't know about (and can't null
        # out) our own self._hover_path_line reference to that now-dead artist.
        # _clear_hover_preview() calling .remove() on it afterward raises
        # NotImplementedError ("cannot remove artist") since the artist is already
        # detached; that happens at the very top of _on_click/_on_motion, before
        # self._hover_path_line is set back to None, and matplotlib's callback
        # dispatcher swallows the exception (prints a traceback, doesn't propagate)
        # - so it never reaches None, and *every* future click/mouse-move hits the
        # same exception again, permanently freezing all map interaction until the
        # app is restarted. Drop the stale reference here, right where it's
        # invalidated, instead of relying on _clear_hover_preview() to catch up.
        self._hover_path_line = None
        if self._hover_timer is not None:
            self._hover_timer.cancel()
            self._hover_timer = None
        self._hover_hex = None

        # Pre-compute fitted limits early; after ax.clear() temporary limits are (0,1),
        # which can make width scaling explode if used directly.
        fit_xlim, fit_ylim = self._compute_fit_limits_for_axes()

        # Background watermark tiled across the whole map (behind terrain/path layers).
        y = -0.05
        row_idx = 0
        while y <= 1.05:
            x_offset = 0.0 if row_idx % 2 == 0 else 0.13
            x = -0.10 + x_offset
            while x <= 1.10:
                self.ax.text(
                    x,
                    y,
                    '亡夜迫邪',
                    transform=self.ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=44,
                    color="#b6b6b6cf",
                    alpha=0.16,
                    rotation=20,
                    zorder=0,
                    clip_on=True,
                )
                x += 0.26
            y += 0.22
            row_idx += 1

        # Use previous view span for line scaling when zoom is active.
        if self._has_zoomed and saved_xlim is not None and saved_ylim is not None:
            path_line_scale = self._get_path_line_scale(saved_xlim, saved_ylim)
        else:
            path_line_scale = self._get_path_line_scale(fit_xlim, fit_ylim)

        C_ORIGIN  = '#22cc55'
        C_CURRENT = '#ff9900'

        # Draw terrain hexes.
        #
        # Perf note: this used to create a brand-new RegularPolygon + Affine2D
        # transform + add_patch() call for each of the ~1450 non-empty hexes on
        # *every* _draw() (i.e. every click/day-change/undo/etc), which profiled
        # at ~450-500ms per redraw - the dominant source of UI lag. Hexes are
        # now batched into a handful of PolyCollections (grouped by hatch
        # pattern, since a collection can only carry one hatch for all of its
        # members) and drawn with a couple of vectorized add_collection() calls
        # instead of ~1450 individual add_patch() calls.
        label_types = {'B1', 'B2', 'B3', 'Z1', 'Z2', 'Z3', 'X1', 'X2', 'X3'}
        show_labels = self._show_bonus_labels
        label_texts = []  # (cx_scaled, cy_scaled, terrain_name)

        if self._map_view_mode == 'image' and self._map_image_array is not None:
            # Real-game screenshot background, pre-rectified (see
            # tools/rectify_map_image.py and the _MAP_IMAGE_FILENAME comment
            # above) so that hex (ir, ic) positions line up with this same
            # data-coordinate system that team paths/markers already use -
            # nothing below this block needs to know which view mode is active.
            self.ax.imshow(
                self._map_image_array, extent=self._map_image_extent,
                zorder=0, aspect='auto', interpolation='bilinear',
            )
            if show_labels:
                for ir in range(ROWS):
                    for ic in range(COLS):
                        t = _terrain(ir, ic)
                        terrain_name = t.get('name', '')
                        if terrain_name and RAW_MAP[ir][ic] in label_types:
                            cx, cy = _center(ir, ic)
                            label_texts.append((cx * X_SCALE, cy * Y_SCALE, terrain_name))
        else:
            # Perf note: this used to create a brand-new RegularPolygon + Affine2D
            # transform + add_patch() call for each of the ~1450 non-empty hexes on
            # *every* _draw() (i.e. every click/day-change/undo/etc), which profiled
            # at ~450-500ms per redraw - the dominant source of UI lag. Hexes are
            # now batched into a handful of PolyCollections (grouped by hatch
            # pattern, since a collection can only carry one hatch for all of its
            # members) and drawn with a couple of vectorized add_collection() calls
            # instead of ~1450 individual add_patch() calls.
            # group key -> lists of verts / facecolors / edgecolors / linewidths
            hex_groups = {}
            for ir in range(ROWS):
                for ic in range(COLS):
                    t = _terrain(ir, ic)
                    if t['name'] == 'empty':
                        continue
                    cx, cy = _center(ir, ic)
                    pos = (ir, ic)

                    # Determine if this is an origin for any team (only color as origin if team is still there)
                    if pos == self.team1.origin and pos == self.team1.full_path[-1]:
                        fc = C_ORIGIN
                    elif self.team2 and pos == self.team2.origin and pos == self.team2.full_path[-1]:
                        fc = C_ORIGIN
                    elif self.team3 and pos == self.team3.origin and pos == self.team3.full_path[-1]:
                        fc = C_ORIGIN
                    else:
                        fc = t['face']

                    ec = t['edge'] if t['edge'] not in ('none', '') else '#777777'
                    is_black_edge = str(ec).lower() in ('k', 'black', '#000', '#000000')
                    hex_border_lw = 0.4 if is_black_edge else 0.9
                    hatch_raw = t.get('hatch', '') if t.get('hatch', '') else None
                    # Make hatch pattern denser by repeating each character
                    hatch = ''.join(ch * 3 for ch in hatch_raw) if hatch_raw else None

                    group = hex_groups.get(hatch)
                    if group is None:
                        group = {'verts': [], 'fc': [], 'ec': [], 'lw': []}
                        hex_groups[hatch] = group
                    group['verts'].append(_HEX_VERT_OFFSETS + (cx, cy))
                    group['fc'].append(fc)
                    group['ec'].append(ec)
                    group['lw'].append(hex_border_lw)

                    # Names for specific terrain types: B1, B2, B3, Z1, Z2, Z3, X1, X2, X3
                    if show_labels:
                        terrain_name = t.get('name', '')
                        if terrain_name and RAW_MAP[ir][ic] in label_types:
                            label_texts.append((cx * X_SCALE, cy * Y_SCALE, terrain_name))

            hex_transform = Affine2D().scale(X_SCALE, Y_SCALE) + self.ax.transData
            for hatch, group in hex_groups.items():
                coll = PolyCollection(
                    group['verts'], facecolors=group['fc'], edgecolors=group['ec'],
                    linewidths=group['lw'], zorder=1, hatch=hatch)
                coll.set_transform(hex_transform)
                self.ax.add_collection(coll)

        for cx_scaled, cy_scaled, terrain_name in label_texts:
            self.ax.text(cx_scaled, cy_scaled, terrain_name,
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        zorder=5, color='#000000')

        # Draw paths for each team
        line_width_factor = 1.4  # 2x wider than previous path width setting
        teams = [(self.team1, 1), (self.team2, 2), (self.team3, 3)]
        for team, team_num in teams:
            if team is None:
                continue
            
            # Draw path line(s), skipping no-draw edges (e.g., portal teleports)
            if len(team.full_path) > 1:
                color = self.team_colors[team_num]
                color_rgb = hex2color(color)
                highlight_color = '#FFEE88'  # Lighter yellow for current day border
                
                # Build a map of path index to (segment index, day)
                path_to_day = {}  # Maps path index to day
                path_to_seg = {}  # Maps path index to segment index
                path_to_action = {}  # Maps path index to action type: 'new' or 'jump'
                path_idx = 1  # Start after origin
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

                # Draw edge-by-edge so revisit/jump edges can be thinner and not block original paths.
                for i in range(1, len(team.full_path)):
                    prev_pos = team.full_path[i - 1]
                    curr_pos = team.full_path[i]

                    seg_idx_for_edge = path_to_seg.get(i)
                    is_fly_edge = (
                        seg_idx_for_edge is not None
                        and seg_idx_for_edge < len(team._seg_is_fly_skill)
                        and team._seg_is_fly_skill[seg_idx_for_edge]
                    )

                    prev_day_for_edge = path_to_day.get(i - 1, team.created_day)
                    seg_day = path_to_day.get(i, prev_day_for_edge)
                    if (not self._show_future_paths) and seg_day > self.current_day:
                        continue

                    # Never draw straight connections between portal hexes.
                    token_prev = RAW_MAP[prev_pos[0]][prev_pos[1]] if 0 <= prev_pos[0] < ROWS and 0 <= prev_pos[1] < COLS else ''
                    token_curr = RAW_MAP[curr_pos[0]][curr_pos[1]] if 0 <= curr_pos[0] < ROWS and 0 <= curr_pos[1] < COLS else ''
                    is_prev_portal = bool(re.fullmatch(r'P\d+', token_prev))
                    is_curr_portal = bool(re.fullmatch(r'P\d+', token_curr))
                    is_adjacent_edge = curr_pos in _neighbors(*prev_pos)

                    # Teleport edges are non-adjacent; if a portal is involved, never draw straight lines.
                    if (not is_adjacent_edge) and (is_prev_portal or is_curr_portal) and (not is_fly_edge):
                        continue

                    if (
                        prev_pos != curr_pos
                        and is_prev_portal
                        and is_curr_portal
                    ):
                        continue

                    # Skip no-draw edges (portal teleport visuals)
                    is_no_draw_edge = (
                        (prev_pos, curr_pos) in team._no_draw_edges
                        or (curr_pos, prev_pos) in team._no_draw_edges
                    )
                    if is_no_draw_edge:
                        # For fly-skill movement, show a thin curved connector instead of no line.
                        if is_fly_edge:
                            curve_target = curr_pos

                            # Use canonical segment raw end as fly-curve target (portal entry if teleported).
                            if (
                                seg_idx_for_edge is not None
                                and seg_idx_for_edge < len(team._seg_path_nodes)
                                and team._seg_path_nodes[seg_idx_for_edge]
                            ):
                                curve_target = team._seg_path_nodes[seg_idx_for_edge][-1]

                            # Legacy fallback: fly segment action history may still hold the portal entry
                            # even when reconstructed path nodes drifted to the portal exit.
                            if (
                                seg_idx_for_edge is not None
                                and seg_idx_for_edge < len(team._seg_action_sequence)
                                and team._seg_action_sequence[seg_idx_for_edge]
                            ):
                                first_action = team._seg_action_sequence[seg_idx_for_edge][0]
                                if isinstance(first_action, (list, tuple)) and len(first_action) >= 2:
                                    action_kind = first_action[0]
                                    action_pos = tuple(first_action[1])
                                    if 0 <= action_pos[0] < ROWS and 0 <= action_pos[1] < COLS:
                                        action_token = RAW_MAP[action_pos[0]][action_pos[1]]
                                        curr_token = RAW_MAP[curr_pos[0]][curr_pos[1]] if 0 <= curr_pos[0] < ROWS and 0 <= curr_pos[1] < COLS else ''
                                        if (
                                            action_kind == 'new'
                                            and action_pos != curr_pos
                                            and re.fullmatch(r'P\d+', action_token)
                                            and action_token == curr_token
                                        ):
                                            curve_target = action_pos

                            # Fallback for legacy/reconstructed history: infer portal entry from no-draw edges.
                            # If current edge ends at a portal exit, there is usually another no-draw edge from
                            # the same previous node to the portal-entry hex.
                            curr_token = RAW_MAP[curr_pos[0]][curr_pos[1]] if 0 <= curr_pos[0] < ROWS and 0 <= curr_pos[1] < COLS else ''
                            if re.fullmatch(r'P\d+', curr_token):
                                inferred_entry = None
                                for e0, e1 in team._no_draw_edges:
                                    if e0 != prev_pos:
                                        continue
                                    if e1 == curr_pos:
                                        continue
                                    if not (0 <= e1[0] < ROWS and 0 <= e1[1] < COLS):
                                        continue
                                    e1_token = RAW_MAP[e1[0]][e1[1]]
                                    if e1_token == curr_token:
                                        inferred_entry = e1
                                        break
                                if inferred_entry is not None:
                                    curve_target = inferred_entry

                            prev_x, prev_y = _center(*prev_pos)
                            curr_x, curr_y = _center(*curve_target)
                            x0, y0 = prev_x * X_SCALE, prev_y * Y_SCALE
                            x2, y2 = curr_x * X_SCALE, curr_y * Y_SCALE

                            dx = x2 - x0
                            dy = y2 - y0
                            d = np.hypot(dx, dy)
                            if d > 1e-6:
                                nx = -dy / d
                                ny = dx / d
                                # Push the control point farther away so fly arc avoids covering main paths.
                                bend = max(0.22 * d, 3.0)
                                x1 = (x0 + x2) * 0.5 + nx * bend
                                y1 = (y0 + y2) * 0.5 + ny * bend

                                tvals = np.linspace(0.0, 1.0, 24)
                                curve_x = (1 - tvals) ** 2 * x0 + 2 * (1 - tvals) * tvals * x1 + tvals ** 2 * x2
                                curve_y = (1 - tvals) ** 2 * y0 + 2 * (1 - tvals) * tvals * y1 + tvals ** 2 * y2

                                self.ax.plot(
                                    curve_x,
                                    curve_y,
                                    color=color,
                                    lw=max(0.55 * path_line_scale * line_width_factor, 0.06),
                                    alpha=0.72,
                                    zorder=6,
                                    solid_capstyle='round',
                                    solid_joinstyle='round',
                                )
                        continue

                    is_selected_edit_day_edge = (
                        self._segment_edit_mode
                        and self._day_edit_context is None
                        and team is self.active_team
                        and seg_day in self._segment_edit_selected_days
                    )
                    action_type = path_to_action.get(i, 'new')
                    jump_width_factor = 0.55 if action_type == 'jump' else 1.0
                    edge_color = color
                    edge_alpha = 1.0
                    if action_type == 'jump':
                        # Lighten revisit/jump edges so they don't dominate normal path segments.
                        edge_color = tuple((c * 0.55) + 0.45 for c in color_rgb)
                        edge_alpha = 0.9

                    prev_x, prev_y = _center(*prev_pos)
                    curr_x, curr_y = _center(*curr_pos)
                    xs = [prev_x * X_SCALE, curr_x * X_SCALE]
                    ys = [prev_y * Y_SCALE, curr_y * Y_SCALE]

                    # Day boundary edge: cut solid line and use dashed connector.
                    if seg_day != prev_day_for_edge:
                        if is_selected_edit_day_edge:
                            self.ax.plot(xs, ys, color='white',
                                         lw=3.0 * (3.1 * path_line_scale * line_width_factor),
                                         linestyle=(0, (2.2, 3.2)), alpha=0.95, zorder=8,
                                         solid_capstyle='round', solid_joinstyle='round')
                        if seg_day == self.current_day:
                            dash_lw = 0.95 * path_line_scale * line_width_factor
                            self.ax.plot(xs, ys, color=highlight_color,
                                         lw=0.75 * (dash_lw + (0.95 * path_line_scale * line_width_factor)),
                                         linestyle=(0, (2.2, 3.2)), alpha=1.0, zorder=6,
                                         solid_capstyle='round', solid_joinstyle='round')
                        self.ax.plot(xs, ys, color=edge_color,
                                     lw=0.95 * path_line_scale * line_width_factor,
                                     linestyle=(0, (2.2, 3.2)), alpha=edge_alpha, zorder=7,
                                     solid_capstyle='round', solid_joinstyle='round')
                        continue

                    if seg_day == self.current_day:
                        # Keep core stroke width identical, but add a visible yellow border.
                        base_lw = 2.5 * path_line_scale * jump_width_factor * line_width_factor
                        outline_lw = base_lw + (1.45 * path_line_scale * line_width_factor)
                        if is_selected_edit_day_edge:
                            self.ax.plot(xs, ys, color='white',
                                         lw=3.0 * (outline_lw + (1.15 * path_line_scale * line_width_factor)),
                                         alpha=0.95, zorder=4,
                                         solid_capstyle='round', solid_joinstyle='round')
                        self.ax.plot(xs, ys, color=highlight_color,
                                     lw=1.0 * outline_lw, alpha=1.0, zorder=5,
                                     solid_capstyle='round', solid_joinstyle='round')
                        self.ax.plot(xs, ys, color=edge_color,
                                     lw=base_lw, alpha=edge_alpha, zorder=6,
                                     solid_capstyle='round', solid_joinstyle='round')
                    else:
                        if is_selected_edit_day_edge:
                            self.ax.plot(xs, ys, color='white',
                                         lw=3.0 * ((2.5 * path_line_scale * jump_width_factor * line_width_factor) + (1.6 * path_line_scale * line_width_factor)),
                                         alpha=0.95,
                                         zorder=2, solid_capstyle='round', solid_joinstyle='round')
                        self.ax.plot(xs, ys, color=edge_color,
                                     lw=2.5 * path_line_scale * jump_width_factor * line_width_factor,
                                     alpha=edge_alpha,
                                     zorder=3, solid_capstyle='round', solid_joinstyle='round')

            # Draw current position marker (end of currently visible path).
            if self._show_future_paths:
                cur = team.full_path[-1]
            else:
                visible_indices = [idx for idx, d in path_to_day.items() if d <= self.current_day and idx < len(team.full_path)]
                cur_idx = max(visible_indices) if visible_indices else 0
                cur = team.full_path[cur_idx]
            if cur != team.origin:
                cx2, cy2 = _center(*cur)
                self.ax.plot(cx2 * X_SCALE, cy2 * Y_SCALE, 'D',
                             color=self.team_colors[team_num], ms=9, zorder=6)

            # During day-edit redraw, keep future-day paths visible as preview overlay.
            if self._show_future_paths:
                self._draw_day_edit_future_preview(team, self.team_colors[team_num])

        self._draw_segment_edit_targets()

        # Status bar (show SHARED resources)
        if self._status_msg:
            status = self._status_msg
        else:
            team_num = 1 if self.active_team is self.team1 else (2 if self.active_team is self.team2 else 3)
            status = f'Team {team_num} | Food Left: {self.current_food} | Reward: {self.total_reward} | Day: {self.current_day}'
        
        # Add Chinese text if fly skill is active
        if self._fly_mode:
            status = '选择飞雷神目的地 (Select Flying Thunder God destination)\n' + status
        
        self.ax.set_xlabel(status, fontsize=9)

        # Axes limits - keep full map fitted to current window size when not zoomed.
        self._default_xlim = fit_xlim
        self._default_ylim = fit_ylim

        if not self._has_zoomed:
            self.ax.set_xlim(fit_xlim)
            self.ax.set_ylim(fit_ylim)
        else:
            # Restore saved zoom limits if user had zoomed
            if saved_xlim is not None and saved_ylim is not None:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
        # Preserve hex geometry when the window is resized.
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')

        # Update scrollbar visibility and position based on zoom
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        default_width = self._default_xlim[1] - self._default_xlim[0]
        default_height = self._default_ylim[1] - self._default_ylim[0]
        
        cur_width = cur_xlim[1] - cur_xlim[0]
        cur_height = cur_ylim[1] - cur_ylim[0]

        # Update fly skill limit label
        self._fly_skill_label_text.set_text(str(self.fly_skill_limit))
        if self.fly_skill_limit <= 0:
            self._fly_skill_label_text.set_color('red')
        else:
            self._fly_skill_label_text.set_color('black')
        
        # Update fly button state based on skill limit
        self._update_fly_button_state()

        # Update day number display
        self._day_number_text.set_text(self._format_day_with_date(self.current_day))
        self._map_image_credit_text.set_visible(self._map_view_mode == 'image')

        # Update team button labels with remaining steps
        self._update_team_button_labels()
        self._update_segment_edit_button_state()
        self._update_edit_seg_blink_state()

        self._draw_team_action_symbols()
        self._draw_map_stats_table()
        self._refresh_open_stat_windows()
        
        self.fig.canvas.draw_idle()

    # ── Data window ──────────────────────────────────────────────────

    def _draw_data_window(self):
        # Check if figure/axis are valid before drawing
        if not hasattr(self, 'data_fig') or not hasattr(self, 'data_ax') or self.data_ax is None or self.data_fig is None:
            return
        
        try:
            ax = self.data_ax
            ax.clear()
            ax.axis('off')
            self.data_fig.patch.set_facecolor('#f5f5f5')
            
            # Ensure day_records is initialized
            if not self.day_records:
                self._init_day_records()

            # Current day data from pre-generated day_records
            col_labels = ['Day', 'Food Used', 'Food Left', 'Reward', 'Cumulative']
            rows = []
            
            # Get current day's record (with safe bounds checking)
            if 1 <= self.current_day <= len(self.day_records):
                current_day_record = self.day_records[self.current_day - 1]
                food_used = current_day_record['food_used']
                food_left = current_day_record['food_remain']
                current_day_reward = current_day_record['reward_used']
                
                # Calculate cumulative reward from day 1 to current day
                cumulative_reward = 0
                for i in range(self.current_day):
                    cumulative_reward += self.day_records[i]['reward_used']
                
                rows.append([
                    str(self.current_day),
                    str(food_used),
                    str(food_left),
                    str(current_day_reward),
                    str(cumulative_reward),
                ])
            
            if rows:
                tbl = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    loc='center',
                    bbox=[0.05, 0.55, 0.9, 0.35],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)

                # Style header row - center text
                for c in range(len(col_labels)):
                    cell = tbl[0, c]
                    cell.set_facecolor('#4488aa')
                    cell.set_text_props(color='white', weight='bold', ha='center')

                # Shade and center the single data row
                for c in range(len(col_labels)):
                    cell = tbl[1, c]
                    cell.set_facecolor('#eef4fb')
                    cell.set_text_props(ha='center')
                    # Make day number 6 times bigger (8 * 6 = 48)
                    if c == 0:  # Day column
                        cell.set_fontsize(48)

            # Display remaining steps per team on current day
            if 1 <= self.current_day <= len(self.day_records):
                current_day_record = self.day_records[self.current_day - 1]
                steps_text = f"Remaining Steps - Team 1: {current_day_record.get('team1_steps_remain', 0) or 0}  |  Team 2: {current_day_record.get('team2_steps_remain', 0) or 0}  |  Team 3: {current_day_record.get('team3_steps_remain', 0) or 0}"
                ax.text(0.05, 0.52, steps_text,
                        transform=ax.transAxes,
                        fontsize=8, fontweight='bold',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.3))

            # Lands visited by each team on current day (visual boxes and jumps in order)
            y_pos = 0.48
            for team, team_num in [(self.team1, 1), (self.team2, 2), (self.team3, 3)]:
                if team is None:
                    continue
                
                # Get action sequence for this team on current day (hexes and jumps in order)
                actions_today = []
                for seg_idx, seg_day in enumerate(team._seg_days):
                    if seg_day == self.current_day:
                        actions_today.extend(team._seg_action_sequence[seg_idx])
                
                # Draw team label
                team_color = self.team_colors[team_num]
                label_text = f'Team {team_num} lands today:'
                ax.text(0.05, y_pos, label_text,
                        transform=ax.transAxes,
                        fontsize=7, fontweight='bold',
                        verticalalignment='top')
                
                current_line_y = y_pos
                if not actions_today:
                    ax.text(0.35, current_line_y, 'No new lands',
                            transform=ax.transAxes,
                            fontsize=7, style='italic',
                            verticalalignment='top',
                            color='#666666')
                    y_pos -= 0.06
                else:
                    # Draw terrain boxes and jump circles in action order
                    x_pos = 0.35
                    max_x = 0.95
                    box_size = 0.025
                    circle_size = 0.025
                    
                    # Track consecutive jumps to merge them
                    jump_count = 0
                    i = 0
                    while i < len(actions_today):
                        action_type, hex_pos = actions_today[i]
                        
                        if action_type == 'new':
                            if x_pos + box_size > max_x:
                                # Move to next line
                                current_line_y -= 0.03
                                x_pos = 0.35
                            
                            # Get terrain info
                            terrain = _terrain(hex_pos[0], hex_pos[1])
                            fc = terrain.get('face', '#cccccc')
                            ec = terrain.get('edge', '#000000')
                            if ec in ('none', ''):
                                ec = '#777777'
                            hatch_raw = terrain.get('hatch', '')
                            hatch = ''.join(ch * 2 for ch in hatch_raw) if hatch_raw else None
                            
                            # Draw rectangle for new hex
                            rect = Rectangle((x_pos, current_line_y - 0.02), box_size, 0.02,
                                            transform=ax.transAxes,
                                            facecolor=fc, edgecolor=ec, linewidth=0.5,
                                            hatch=hatch, zorder=2)
                            ax.add_patch(rect)
                            x_pos += box_size + 0.005
                            
                        elif action_type == 'jump':
                            # Count consecutive jumps
                            jump_count = 1
                            j = i + 1
                            while j < len(actions_today) and actions_today[j][0] == 'jump':
                                jump_count += 1
                                j += 1
                            
                            if x_pos + circle_size > max_x:
                                # Move to next line
                                current_line_y -= 0.03
                                x_pos = 0.35
                            
                            # Draw jump circle with merged count
                            circle = Circle((x_pos + circle_size/2, current_line_y - 0.01), 
                                            circle_size/2.5,
                                            transform=ax.transAxes,
                                            facecolor='#FFB6C1', edgecolor='#FF69B4', 
                                            linewidth=1.5, zorder=3)
                            ax.add_patch(circle)
                            
                            # Add jump count text in the circle
                            ax.text(x_pos + circle_size/2, current_line_y - 0.01, str(jump_count),
                                   transform=ax.transAxes,
                                   fontsize=8, fontweight='bold',
                                   ha='center', va='center', zorder=4)
                            
                            x_pos += circle_size + 0.005
                            # Skip the merged jumps
                            i = j - 1
                        
                        i += 1
                    
                    # Move down for next team
                    y_pos = current_line_y - 0.06

            # Team overall stats at bottom
            y_start = 0.26
            for team, team_num in [(self.team1, 1), (self.team2, 2), (self.team3, 3)]:
                if team is None:
                    continue
                
                # Calculate food and reward for CURRENT DAY ONLY (not cumulative)
                day_team_food = 0
                day_team_reward = 0
                for seg_idx, seg_day in enumerate(team._seg_days):
                    if seg_day == self.current_day:
                        day_team_food += team._seg_foods[seg_idx]
                        day_team_reward += team._seg_awards[seg_idx]
                
                bonus_str = ''
                if team.x_bonus_remaining > 0:
                    bonus_str = f'  |  X bonus: {team.x_bonus_remaining} free mvmt left'
                if team.b_discount_remaining > 0:
                    bonus_str += f'  |  B discount: {team.b_discount_remaining} mvmt left'
                if team.z_bonus_remaining > 0:
                    bonus_str += f'  |  Z bonus: {team.z_bonus_remaining} mvmt left'
                
                # Get remaining steps from day_records (rebuilt after each move)
                if 1 <= self.current_day <= len(self.day_records):
                    current_day_record = self.day_records[self.current_day - 1]
                    steps_remain = current_day_record.get(f'team{team_num}_steps_remain', 0) or 0
                else:
                    steps_remain = 0
                
                team_text = f'Team {team_num}: Food today: {day_team_food}  |  Reward today: {day_team_reward}  |  Steps: {steps_remain}{bonus_str}'
                ax.text(0.05, y_start, team_text,
                        transform=ax.transAxes,
                        fontsize=8, family='monospace',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=self.team_colors[team_num],
                                  alpha=0.2, edgecolor=self.team_colors[team_num], linewidth=0.8))
                y_start -= 0.08

            # Jump summary table: before/after global G/g bonus activation.
            jump_stats = self._compute_jump_summary_before_after_g()
            jump_rows = []
            for team_num in (1, 2, 3):
                s = jump_stats.get(team_num, {'before': 0, 'after': 0})
                total = s['before'] + s['after']
                jump_rows.append([f'Team {team_num}', str(s['before']), str(s['after']), str(total)])

            jump_labels = ['Team', 'Jumps Before G/g', 'Jumps After G/g', 'Total']
            jump_tbl = ax.table(
                cellText=jump_rows,
                colLabels=jump_labels,
                loc='center',
                bbox=[0.50, 0.01, 0.48, 0.20],
                cellLoc='center',
            )
            jump_tbl.auto_set_font_size(False)
            jump_tbl.set_fontsize(7)

            for c in range(len(jump_labels)):
                h = jump_tbl[0, c]
                h.set_facecolor('#6b4f9d')
                h.set_text_props(color='white', weight='bold', ha='center')

            for r in range(1, len(jump_rows) + 1):
                shade = '#f1ebfb' if r % 2 == 1 else '#f8f4ff'
                for c in range(len(jump_labels)):
                    jump_tbl[r, c].set_facecolor(shade)

            ax.text(0.50, 0.215,
                    'Jump Summary (Portal first=5, retake=1, fly-to-portal=0)',
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='bottom')

            self.data_fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error drawing data window: {e}")
            self.data_ax = None
            self.data_fig = None

    def _compute_jump_summary_before_after_g(self):
        """Compute jump totals by team before/after global G/g bonus activation.

        Rules:
        - Normal revisit jump action counts as 1 jump.
        - Taking a portal first time counts as 5 jumps.
        - Taking a previously taken portal counts as 1 jump.
        - Flying to a portal counts as 0 jumps.
        """
        stats = {
            1: {'before': 0, 'after': 0},
            2: {'before': 0, 'after': 0},
            3: {'before': 0, 'after': 0},
        }

        teams = [(self.team1, 1), (self.team2, 2), (self.team3, 3)]
        events = []

        for team, team_num in teams:
            if team is None:
                continue

            cursor = 1  # team.full_path index where current segment starts
            for seg_idx, seg_len in enumerate(team._seg_lengths):
                if seg_len <= 0:
                    continue
                if cursor - 1 >= len(team.full_path):
                    break

                seg_day = team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else 1
                seg_actions = team._seg_action_sequence[seg_idx] if seg_idx < len(team._seg_action_sequence) else []
                is_fly = team._seg_is_fly_skill[seg_idx] if seg_idx < len(team._seg_is_fly_skill) else False

                seg_start = team.full_path[cursor - 1]
                seg_end_idx = min(cursor + seg_len - 1, len(team.full_path) - 1)
                seg_end = team.full_path[seg_end_idx]

                portal_source = None
                portal_dest = None
                token_end = RAW_MAP[seg_end[0]][seg_end[1]] if 0 <= seg_end[0] < ROWS and 0 <= seg_end[1] < COLS else ''

                if re.fullmatch(r'P\d+', token_end):
                    portal_dest = seg_end
                    teleported = (seg_start, seg_end) in team._no_draw_edges
                    if teleported:
                        paired = []
                        for ir in range(ROWS):
                            for ic in range(COLS):
                                if RAW_MAP[ir][ic] == token_end and (ir, ic) != seg_end:
                                    paired.append((ir, ic))
                        portal_source = paired[0] if paired else seg_end
                    else:
                        portal_source = seg_end

                events.append({
                    'day': seg_day,
                    'team_num': team_num,
                    'seg_idx': seg_idx,
                    'actions': seg_actions,
                    'is_fly': is_fly,
                    'portal_source': portal_source,
                    'portal_dest': portal_dest,
                })
                cursor += seg_len

        # Deterministic replay order. Cross-team same-day ordering is approximated by team number.
        events.sort(key=lambda e: (e['day'], e['team_num'], e['seg_idx']))

        visited_g = set()
        for team, _ in teams:
            if team is not None and team.origin in self.all_g_lands:
                visited_g.add(team.origin)

        taken_portals = set()
        total_g = len(self.all_g_lands)

        def _bucket():
            return 'after' if (total_g > 0 and len(visited_g) >= total_g) else 'before'

        for ev in events:
            team_num = ev['team_num']
            portal_source = ev['portal_source']
            portal_dest = ev['portal_dest']

            # Replay action sequence in order so G/g activation can happen mid-segment.
            for action in ev['actions']:
                if not isinstance(action, (list, tuple)) or len(action) < 2:
                    continue
                action_type, h = action[0], action[1]
                if not isinstance(h, tuple):
                    h = tuple(h) if isinstance(h, (list, tuple)) else h

                if action_type == 'jump':
                    # Portal jump is handled by portal rule below (avoid double-counting).
                    if portal_source is not None and h == portal_source:
                        continue
                    stats[team_num][_bucket()] += 1
                elif action_type == 'new' and isinstance(h, tuple) and h in self.all_g_lands:
                    visited_g.add(h)

            # Portal taking rule (including fly exclusion) is applied once per segment endpoint.
            if portal_source is not None:
                if not ev['is_fly']:
                    jump_inc = 5 if portal_source not in taken_portals else 1
                    stats[team_num][_bucket()] += jump_inc

                # Mark taken state regardless of fly, matching gameplay state changes.
                taken_portals.add(portal_source)
                if portal_dest is not None:
                    taken_portals.add(portal_dest)

        return stats

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

    def _ensure_global_stat_window(self):
        """Ensure global stat figure/axes exist and are valid, recreate if needed."""
        needs_recreate = False

        if (not hasattr(self, 'global_stat_fig') or not hasattr(self, 'global_stat_ax') or
                self.global_stat_fig is None or self.global_stat_ax is None):
            needs_recreate = True
        else:
            try:
                # Figure may have been fully closed; check fignum existence.
                if not plt.fignum_exists(self.global_stat_fig.number):
                    needs_recreate = True
                else:
                    # Accessing manager title raises when Tk app/window is already destroyed.
                    manager = self.global_stat_fig.canvas.manager
                    if manager is None:
                        needs_recreate = True
                    else:
                        _ = manager.get_window_title()
            except Exception:
                needs_recreate = True

        if needs_recreate:
            self.global_stat_fig = plt.figure(figsize=(8, 10))
            self.global_stat_fig.canvas.manager.set_window_title('全局统计')
            self.global_stat_ax = self.global_stat_fig.add_axes([0.05, 0.05, 0.9, 0.9])
            self.global_stat_fig.canvas.mpl_connect('close_event', self._on_global_stat_window_closed)

    def _draw_global_stat_window(self):
        """Draw global statistics table: total and taken counts for each hex type."""
        self._ensure_global_stat_window()
        if (not hasattr(self, 'global_stat_fig') or not hasattr(self, 'global_stat_ax') or
                self.global_stat_ax is None or self.global_stat_fig is None):
            return

        try:
            ax = self.global_stat_ax
            ax.clear()
            ax.axis('off')
            self.global_stat_fig.patch.set_facecolor('#f7f7f7')

            # Exclusions per requirement: Portals, ST, default, B/X/Z bonus lands.
            bonus_tokens = {'B1', 'B2', 'B3', 'X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'}

            stats = {}  # key -> {'token','terrain','total','taken'}

            for ir in range(ROWS):
                for ic in range(COLS):
                    token = RAW_MAP[ir][ic]

                    # Exclude portals and ST marker.
                    if token == 'ST' or re.fullmatch(r'P\d+', token):
                        continue

                    # Exclude B/X/Z bonus lands.
                    if token in bonus_tokens:
                        continue

                    terrain = _terrain(ir, ic)
                    terrain_name = terrain.get('name', 'unknown')

                    # Exclude default-resolved and empty/default terrain cells.
                    if token not in _TERRAIN_DB:
                        continue
                    if terrain_name in ('default', 'empty'):
                        continue

                    key = (token, terrain_name)
                    if key not in stats:
                        stats[key] = {
                            'token': token,
                            'terrain': terrain_name,
                            'total': 0,
                            'taken': 0,
                        }

                    stats[key]['total'] += 1
                    if (ir, ic) in self.all_visited_hexes:
                        stats[key]['taken'] += 1

            # Stable ordering: token then terrain name.
            ordered = sorted(stats.values(), key=lambda r: (r['token'], r['terrain']))

            rows = []
            terrain_styles = []
            total_all = 0
            taken_all = 0
            for row in ordered:
                remaining = row['total'] - row['taken']
                terrain_def = _TERRAIN_DB.get(row['token'], {})
                terrain_fc = terrain_def.get('face', '#cccccc')
                terrain_ec = terrain_def.get('edge', '#000000')
                if terrain_ec in ('none', ''):
                    terrain_ec = '#777777'
                hatch_raw = terrain_def.get('hatch', '')
                terrain_hatch = ''.join(ch * 2 for ch in hatch_raw) if hatch_raw else None

                rows.append([
                    row['token'],
                    '',
                    str(row['total']),
                    str(row['taken']),
                    str(remaining),
                ])
                terrain_styles.append({
                    'name': row['terrain'],
                    'face': terrain_fc,
                    'edge': terrain_ec,
                    'hatch': terrain_hatch,
                })
                total_all += row['total']
                taken_all += row['taken']

            if rows:
                col_labels = ['地块', '地形', '总数', '已占领', '剩余']
                tbl = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    loc='upper center',
                    bbox=[0.02, 0.26, 0.96, 0.70],
                    cellLoc='center'
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

                # Header styling.
                for c in range(len(col_labels)):
                    cell = tbl[0, c]
                    cell.set_facecolor('#446688')
                    cell.set_text_props(color='white', weight='bold', ha='center')

                # Row striping.
                for r in range(1, len(rows) + 1):
                    shade = '#eef4fb' if r % 2 == 1 else '#f8fbff'
                    for c in range(len(col_labels)):
                        tbl[r, c].set_facecolor(shade)

                # Emphasize hex-grid label in HEX column.
                for r in range(1, len(rows) + 1):
                    tbl[r, 0].set_text_props(weight='bold', color='#1f2d3d')

                # Draw terrain symbol only in Terrain column.
                # Force a draw first so table cell extents are finalized.
                self.global_stat_fig.canvas.draw()
                renderer = self.global_stat_fig.canvas.get_renderer()

                for r, style in enumerate(terrain_styles, start=1):
                    terrain_cell = tbl[r, 1]
                    terrain_cell.get_text().set_text('')
                    cell_bbox = terrain_cell.get_window_extent(renderer=renderer)
                    (x0, y0) = ax.transAxes.inverted().transform((cell_bbox.x0, cell_bbox.y0))
                    (x1, y1) = ax.transAxes.inverted().transform((cell_bbox.x1, cell_bbox.y1))
                    cx, cy = x0, y0
                    cw = max(0.0, x1 - x0)
                    ch = max(0.0, y1 - y0)

                    symbol_center = (cx + cw * 0.50, cy + ch * 0.50)
                    symbol_radius = ch * 0.28
                    symbol = RegularPolygon(
                        symbol_center,
                        numVertices=6,
                        radius=symbol_radius,
                        orientation=np.radians(30),
                        transform=ax.transAxes,
                        facecolor=style['face'],
                        edgecolor=style['edge'],
                        linewidth=0.9,
                        hatch=style['hatch'],
                        zorder=6,
                        clip_on=True,
                    )
                    # Stretch symbol horizontally (2.5x wider) while keeping center fixed.
                    symbol.set_transform(
                        Affine2D()
                        .translate(-symbol_center[0], -symbol_center[1])
                        .scale(2.5, 1.0)
                        .translate(symbol_center[0], symbol_center[1])
                        + ax.transAxes
                    )
                    ax.add_patch(symbol)

            # Add jump summary table in Global Stat window.
            jump_stats = self._compute_jump_summary_before_after_g()
            jump_rows = []
            for team_num in (1, 2, 3):
                s = jump_stats.get(team_num, {'before': 0, 'after': 0})
                jump_rows.append([
                    f'{team_num}队',
                    str(s['before']),
                    str(s['after']),
                    str(s['before'] + s['after'])
                ])

            jump_labels = ['队伍', '八卦前跳步', '八卦后跳步', '合计']
            jump_tbl = ax.table(
                cellText=jump_rows,
                colLabels=jump_labels,
                loc='center',
                bbox=[0.18, 0.07, 0.64, 0.14],
                cellLoc='center',
            )
            jump_tbl.auto_set_font_size(False)
            jump_tbl.set_fontsize(8)

            for c in range(len(jump_labels)):
                h = jump_tbl[0, c]
                h.set_facecolor('#6b4f9d')
                h.set_text_props(color='white', weight='bold', ha='center')

            for r in range(1, len(jump_rows) + 1):
                shade = '#f1ebfb' if r % 2 == 1 else '#f8f4ff'
                for c in range(len(jump_labels)):
                    jump_tbl[r, c].set_facecolor(shade)

            ax.text(0.5, 0.225,
                    '跳步汇总（首传送门=5跳，重复传送=1跳，飞雷神到传送=0跳）',
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='center', va='bottom')

            title = (
                '全局地块统计'
            )
            ax.text(0.5, 0.99, title,
                    transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold')

            ax.text(0.02, 0.02,
                    f'总地块：{total_all}   |   已占领：{taken_all}   |   剩余：{total_all - taken_all}',
                    transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='bottom')

            self.global_stat_fig.canvas.draw_idle()
        except Exception as e:
            print(f'Error drawing global stat window: {e}')
            self.global_stat_ax = None
            self.global_stat_fig = None



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Map: {ROWS} rows × {COLS} cols')
    start = _find_start_position()
    print(f'Resolved start: ir={start[0]}, ic={start[1]}')
    print('Launching interactive window…')
    PathfindingDemo()
