"""hex_core.py
Shared engine for the Naruto hex marching map: terrain model, segment
history, rendering, day navigation and save loading.

This is the read-only half of the app. Route editing - drawing and pricing
moves, undo, day-range edits, enclosure mode, saving and the Excel export -
lives in the editor module that subclasses HexCore, so a build that imports
only this module carries none of it. HexCore.__init__ still creates the
editor's buttons and names their handlers in lambdas; those are resolved at
click time and cost nothing where the methods are absent.

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
import copy
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

# 这两处必须显式写 encoding='utf-8'。Windows 下 open() 默认用本地编码(cp936/
# cp1252), 而 landInfo.json 是 UTF-8 存的 —— 一旦里面出现非 ASCII(例如导入的
# 地图叫 map_尸鬼红唇.csv), 不指定编码读回来就是乱码, 然后找不到文件。
def _load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return [row for row in csv.reader(f)]


with open('landInfo.json', encoding='utf-8') as f:
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

# 全局统计 window size. The per-phase capture columns (八卦前/后, 帐篷1/2阶段,
# 圈地阶段) make the land table twice as wide as it used to be, so the window is
# sized for ten columns. _GLOBAL_STAT_BASE_FIGSIZE records the proportions the
# terrain-symbol stretch in _draw_global_stat_window was originally tuned
# against, so that symbol keeps its shape whatever this window is resized to.
GLOBAL_STAT_FIGSIZE = (14, 10)
_GLOBAL_STAT_BASE_FIGSIZE = (8, 10)

# Readable name for each map token, used for the 全局统计 land table's first
# column. A token with no entry here falls back to the raw token.
LAND_DISPLAY_NAMES = {
    **{str(n): f'{n}级地' for n in range(1, 7)},
    **{f'T{n}': f'塔{n}' for n in range(1, 7)},
    'B': '大boss',
    'b': '中boss',
    'G': '大卦',
    'g': '3级卦',
    'L': '历练',
    'M': '商人',
    'T': '帐篷',
}

# A B/X/Z buff land is a tower of the matching level carrying a buff - same
# food and award as the plain tower (B2/X2/Z2 cost 100 for 40, exactly like
# T2) - so the 全局统计 land table counts it as one of that tower level
# rather than giving it a row of its own.
BUFF_TOWER_LEVEL = {
    f'{prefix}{level}': f'T{level}'
    for level in (1, 2, 3)
    for prefix in ('B', 'X', 'Z')
}

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


def _quarter_circle_arc(x0, y0, x1, y1, n=16):
    """Points along a 90-degree circular arc from (x0,y0) to (x1,y1), bulging
    to a fixed side of the travel direction (consistent curve orientation for
    every jump edge, so back-and-forth jumps over the same pair of hexes draw
    as mirrored arcs rather than overlapping straight lines).

    For a chord of length L subtending a 90-degree angle at the circle's
    center, radius r = L / sqrt(2) and the center sits exactly L/2 out along
    the chord's perpendicular bisector - both follow directly from the
    isosceles right triangle formed by the center and the two chord endpoints.
    """
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return np.array([x0, x1]), np.array([y0, y1])
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # unit normal, rotated 90 deg CCW from travel direction
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cx, cy = mx + nx * (length / 2.0), my + ny * (length / 2.0)
    a0 = np.arctan2(y0 - cy, x0 - cx)
    a1 = np.arctan2(y1 - cy, x1 - cx)
    sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi  # shortest signed turn, always ~+-90 deg here
    angles = a0 + np.linspace(0.0, sweep, n)
    radius = length / np.sqrt(2.0)
    return cx + radius * np.cos(angles), cy + radius * np.sin(angles)


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

class HexCore:
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
        self._zoom_refresh_timer = None  # Delayed route redraw after zoom settles
        
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
        self._enclosure_mode = False
        self._enclosure_start_day = None
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
        self._global_stat_table_rows = []
        self._create_global_stat_figure()
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

        # 导入远征模拟器的 JSON 地图。处理函数定义在编辑器模块里, 这里只是按名字
        # 引用(点击时才解析), 所以只读查看器里这个按钮是空壳, 且会被隐藏。
        bax_import_map = self.fig.add_axes([0.38, 0.002, 0.084, 0.025])
        self._btn_import_map = Button(bax_import_map, '导入地图', color='#ffd9b3')
        self._btn_import_map.label.set_fontsize(12)
        self._btn_import_map.on_clicked(lambda _evt: self._import_json_map())

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

        # Enclosure mode: freehand future-route planning after all Tent/G-g goals.
        bax_enclosure = self.fig.add_axes([0.557, 0.002, 0.084, 0.025])
        self._btn_enclosure = Button(bax_enclosure, '圈地模式', color='#d8f0c8')
        self._btn_enclosure.label.set_fontsize(12)
        self._btn_enclosure.on_clicked(lambda _evt: self._toggle_enclosure_mode())

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
        self._action_symbol_hits = []
        self._action_tooltip = self.fig.text(
            0.0, 0.0, '', ha='right', va='bottom', fontsize=9,
            color='#222222', visible=False,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fffde6', edgecolor='#777777'),
            zorder=20,
        )

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
        if event.inaxes == self._team_action_ax and event.xdata is not None and event.ydata is not None:
            hit = None
            for candidate in self._action_symbol_hits:
                if (candidate['x0'] <= event.xdata <= candidate['x1'] and
                        candidate['y0'] <= event.ydata <= candidate['y1']):
                    hit = candidate
                    break
            if hit is None:
                self._action_tooltip.set_visible(False)
            else:
                # The symbol column sits at the figure's right edge (x~0.91-1.0),
                # so the tooltip grows leftward from the cursor (ha='right')
                # instead of rightward - growing rightward needed a large clamp
                # to avoid running off the figure, which was the actual cause
                # of the tooltip appearing far from the cursor.
                fig_x, fig_y = self.fig.transFigure.inverted().transform((event.x, event.y))
                self._action_tooltip.set_position((min(max(fig_x - 0.003, 0.05), 0.99),
                                                   min(max(fig_y + 0.003, 0.0), 0.94)))
                self._action_tooltip.set_text(
                    f"粮草: {hit['food']}\n积分: {hit['award']}"
                )
                self._action_tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
            return

        self._action_tooltip.set_visible(False)

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





    def _rebalance_team_segment_days_from(self, team, start_day, preserve_future_days=True):
        """Reassign segment days from start_day onward to avoid per-day step overflow.

        Segments stay in original order; when a day runs out of steps, remaining segments
        are pushed to following days. Never pushes a segment past TOTAL_DAYS (there is no
        such day to view/export); returns True if some segment still didn't fit even after
        reaching the last day (a real overflow the redraw can't be fully squeezed into the
        season), False otherwise.
        """
        if team is None or not team._seg_days:
            return False

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
        overflow = False

        seg_idx = 0
        while seg_idx < len(team._seg_days):
            if team._seg_days[seg_idx] < earliest:
                seg_idx += 1
                continue

            # Preserve any slack the existing schedule had (e.g. a day where the
            # team didn't use all its steps/food): never pull a segment onto an
            # earlier day than it already recorded, only catch up to it. This
            # keeps the rebalance a no-op for an untouched route instead of
            # re-flowing everything as tightly as the greedy budget allows,
            # which would otherwise shift every later segment - and any
            # Tent/G-g hex on it - to an earlier day for no real reason.
            preferred_day = int(team._seg_days[seg_idx]) if preserve_future_days else current_day
            while current_day < preferred_day:
                current_day += 1
                step_bank = min(step_bank + 6, 18)
                day_food_remaining = day_food_remaining + 1600 - other_food_by_day.get(current_day, 0)

            seg_steps = team._seg_steps[seg_idx]
            seg_food = team._seg_foods[seg_idx]

            group_end = seg_idx
            group_steps = 0
            group_food = 0
            while group_end < len(team._seg_days) and int(team._seg_days[group_end]) == preferred_day:
                group_steps += team._seg_steps[group_end]
                group_food += team._seg_foods[group_end]
                group_end += 1

            if group_end > seg_idx + 1 and (
                (group_steps <= 0 or step_bank >= group_steps)
                and (group_food <= 0 or day_food_remaining >= group_food)
            ):
                for j in range(seg_idx, group_end):
                    team._seg_days[j] = current_day
                step_bank -= group_steps
                if step_bank > 18:
                    step_bank = 18
                day_food_remaining -= group_food
                seg_idx = group_end
                continue

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

                    fits = (
                        (seg_steps <= 0 or step_bank >= seg_steps)
                        and (seg_food <= 0 or day_food_remaining >= seg_food)
                    )
                    if fits:
                        break

                    # The season only has TOTAL_DAYS days - a segment that still
                    # doesn't fit even on the last day has nowhere real to go, so
                    # stop advancing (there is no such day to view/export) and
                    # flag the overflow instead of inventing a day past the end
                    # of the season.
                    if current_day >= TOTAL_DAYS:
                        overflow = True
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

        return overflow

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

    def _rebalance_enclosure_segments_from_day(self, start_day):
        """Distribute enclosure-mode future routes across teams day by day."""
        teams = [t for t in (self.team1, self.team2, self.team3) if t is not None]
        if not teams:
            return

        start_day = max(1, int(start_day))

        food_remaining = 6800
        for day in range(1, start_day):
            used = 0
            for team in teams:
                for seg_food, seg_day in zip(team._seg_foods, team._seg_days):
                    if int(seg_day) == day:
                        used += seg_food
            food_remaining = food_remaining - used + 1600

        step_bank = {}
        next_idx = {}
        for team in teams:
            bank = 0
            for day in range(team.created_day, max(start_day, team.created_day)):
                bank = min(bank + 6, 18)
                day_used = 0
                for seg_idx, seg_day in enumerate(team._seg_days):
                    if int(seg_day) == day:
                        day_used += team._seg_steps[seg_idx]
                bank -= day_used
            step_bank[team] = bank

            idx = 0
            while idx < len(team._seg_days) and int(team._seg_days[idx]) < start_day:
                idx += 1
            next_idx[team] = idx

        current_day = start_day
        for team in teams:
            if team.created_day <= current_day:
                step_bank[team] = min(step_bank[team] + 6, 18)

        while any(next_idx[team] < len(team._seg_days) for team in teams):
            placed_today = False

            while True:
                placed_this_round = False
                for team in teams:
                    idx = next_idx[team]
                    if idx >= len(team._seg_days) or team.created_day > current_day:
                        continue

                    seg_steps = team._seg_steps[idx]
                    seg_food = team._seg_foods[idx]
                    if not (
                        (seg_steps <= 0 or step_bank[team] >= seg_steps)
                        and (seg_food <= 0 or food_remaining >= seg_food)
                    ):
                        did_split = self._split_team_segment_by_step_budget(
                            team,
                            idx,
                            max(step_bank[team], 0),
                            max(food_remaining, 0),
                        )
                        if did_split:
                            seg_steps = team._seg_steps[idx]
                            seg_food = team._seg_foods[idx]

                    if (
                        (seg_steps <= 0 or step_bank[team] >= seg_steps)
                        and (seg_food <= 0 or food_remaining >= seg_food)
                    ):
                        team._seg_days[idx] = current_day
                        step_bank[team] -= seg_steps
                        if step_bank[team] > 18:
                            step_bank[team] = 18
                        food_remaining -= seg_food
                        next_idx[team] += 1
                        placed_this_round = True
                        placed_today = True

                if not placed_this_round:
                    break

            if not placed_today:
                # A single oversized segment may need future accumulated food/steps.
                pass

            current_day += 1
            food_remaining += 1600
            for team in teams:
                if team.created_day <= current_day:
                    step_bank[team] = min(step_bank[team] + 6, 18)

        for team in teams:
            if team._seg_days:
                team.max_day_reached = max(team.created_day, max(team._seg_days))
            else:
                team.max_day_reached = team.created_day

    def _rebalance_all_teams_from_day(self, start_day, max_passes=8, preserve_future_days=True):
        """Iteratively rebalance all teams from a day to satisfy shared food and team steps.

        Returns True if some team's segments couldn't all be squeezed into the
        season's TOTAL_DAYS days even after rebalancing (a redraw that simply
        needs more days than the season has left).
        """
        teams = [t for t in (self.team1, self.team2, self.team3) if t is not None]
        if not teams:
            return False

        if not preserve_future_days:
            self._rebalance_enclosure_segments_from_day(start_day)
            return False

        rebalance_start = max(1, int(start_day))
        overflow = False

        for _ in range(max_passes):
            before = [tuple(t._seg_days) for t in teams]

            overflow = False
            for team in teams:
                if self._rebalance_team_segment_days_from(
                    team,
                    max(rebalance_start, team.created_day),
                    preserve_future_days=preserve_future_days,
                ):
                    overflow = True

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

        return overflow


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



    def _update_enclosure_button_state(self):
        if not hasattr(self, '_btn_enclosure') or self._btn_enclosure is None:
            return
        if self._enclosure_mode:
            self._btn_enclosure.label.set_text('退出圈地')
            self._btn_enclosure.color = '#b8e986'
            self._btn_enclosure.hovercolor = '#a8df70'
            self._btn_enclosure.label.set_color('#1f5b00')
        else:
            self._btn_enclosure.label.set_text('圈地模式')
            self._btn_enclosure.color = '#d8f0c8'
            self._btn_enclosure.hovercolor = '#c8e8b8'
            self._btn_enclosure.label.set_color('black')



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




    
    def _set_fly_button_border(self, color, linewidth):
        """Set the border color and width of the fly skill button."""
        for spine in self._btn_fly.ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(linewidth)
    

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
        self._action_symbol_hits = []
        
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
                        'team': team,
                        'seg_idx': seg_idx,
                        'actions': actions,
                        'symbol_count': symbol_count,
                    })
                    total_symbols += symbol_count

        action_segments.sort(key=lambda item: item['order'])

        def _action_costs_for_segment(team, seg_idx, actions):
            """Return one (food, award) pair for each recorded action."""
            if team is None or seg_idx is None:
                return [(0, 0) for _ in actions]

            stored_costs = team._seg_hex_costs[seg_idx] if seg_idx < len(team._seg_hex_costs) else []
            costs = [
                (entry[0], entry[1])
                for entry in stored_costs
                if isinstance(entry, (list, tuple)) and len(entry) >= 2
            ]

            if len(costs) == len(actions):
                return costs
            if actions and actions[0][0] == 'fly' and len(costs) == len(actions) - 1:
                return [(0, 0)] + costs
            if len(costs) > len(actions):
                return costs[-len(actions):]
            return costs + [(0, 0) for _ in range(len(actions) - len(costs))]

        for action_segment in action_segments:
            action_segment['action_costs'] = _action_costs_for_segment(
                action_segment['team'],
                action_segment['seg_idx'],
                action_segment['actions'],
            )

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
                previous['action_costs'].extend(action_segment['action_costs'])
                continue
            merged_segments.append({
                'order': action_segment['order'],
                'team_idx': action_segment['team_idx'],
                'team': action_segment.get('team'),
                'seg_idx': action_segment.get('seg_idx'),
                'actions': actions,
                'action_costs': list(action_segment['action_costs']),
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
            action_costs = action_segment.get('action_costs', [])
            
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
                    self._action_symbol_hits.append({
                        'x0': x_pos - 0.18, 'x1': x_pos + 0.18,
                        'y0': y_pos - 0.22, 'y1': y_pos + 0.22,
                        'food': action_costs[i][0] if i < len(action_costs) else 0,
                        'award': action_costs[i][1] if i < len(action_costs) else 0,
                    })
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
                    self._action_symbol_hits.append({
                        'x0': x_pos - 0.18, 'x1': x_pos + 0.28,
                        'y0': y_pos - 0.22, 'y1': y_pos + 0.22,
                        'food': action_costs[i][0] if i < len(action_costs) else 0,
                        'award': action_costs[i][1] if i < len(action_costs) else 0,
                    })
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
                    self._action_symbol_hits.append({
                        'x0': x_pos - 0.22, 'x1': x_pos + 0.22,
                        'y0': y_pos - 0.22, 'y1': y_pos + 0.22,
                        'food': sum(
                            action_costs[k][0]
                            for k in range(i, min(j, len(action_costs)))
                        ),
                        'award': sum(
                            action_costs[k][1]
                            for k in range(i, min(j, len(action_costs)))
                        ),
                    })
                    
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


    # ── Mode / interaction ────────────────────────────────────────────────────


    
    

    def _update_show_future_button_state(self):
        """Refresh the future-path toggle button visual state."""
        if not hasattr(self, '_btn_show_future') or self._btn_show_future is None:
            return

        text = '[X] 显示未来' if self._show_future_paths else '[ ] 显示未来'
        self._btn_show_future.label.set_text(text)
        self._btn_show_future.color = '#90EE90' if self._show_future_paths else '#FFCCCC'
        self._btn_show_future.hovercolor = '#7FDF7F' if self._show_future_paths else '#FFB3B3'


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
                '_btn_global_stat', '_btn_edit_seg', '_btn_enclosure', '_btn_export_xlsx',
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
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(game_state, f, indent=2)
        except Exception as e:
            print(f'Auto-save error: {e}')








    
    
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
            with open(file_path, 'r', encoding='utf-8') as f:
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
            self._enclosure_mode = False
            self._enclosure_start_day = None
            
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

            rebalance_overflow = False
            if needs_rebalance:
                rebalance_overflow = self._rebalance_all_teams_from_day(1)

            # Reconstruct shared derived runtime state from canonical segment history.
            self._rebuild_shared_derived_state_from_segments()
            
            # Rebuild day records from segment data (ensures consistency with loaded segments)
            # and sync current_food with the rebuilt food_remain for the current day.
            # This fixes cases where current_food in the save is out of sync with actual segment history.
            self._rebuild_day_records()
            if self.day_records and 1 <= self.current_day <= len(self.day_records):
                self.current_food = self.day_records[self.current_day - 1]['food_remain']
            
            self._status_msg = f'Game loaded from {os.path.basename(file_path)}'
            if rebalance_overflow:
                self._status_msg += (
                    f' WARNING: some segments could not be rebalanced within Day {TOTAL_DAYS} '
                    f'(the season length) and are piled up on the last day.'
                )
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

        # Keep zoom interaction light while the wheel is still moving. Route
        # artists are rebuilt once the view has been idle for one second.
        if self._zoom_refresh_timer is not None:
            self._zoom_refresh_timer.stop()
            self._zoom_refresh_timer = None

        def _refresh_after_zoom():
            self._zoom_refresh_timer = None
            self._draw()

        self._zoom_refresh_timer = self.fig.canvas.new_timer(interval=1000)
        self._zoom_refresh_timer.single_shot = True
        self._zoom_refresh_timer.callbacks.append((_refresh_after_zoom, (), {}))
        self._zoom_refresh_timer.start()
        self.fig.canvas.draw_idle()



    # ── Portal teleportation ────────────────────────────────────────────────────


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

            # Read unconditionally below (current-position marker, future-path
            # preview) regardless of whether this team has moved yet, so it
            # must exist even when the "len(full_path) > 1" block below is
            # skipped for a team with zero moves.
            path_to_day = {}  # Maps path index to day

            # Draw path line(s), skipping no-draw edges (e.g., portal teleports)
            if len(team.full_path) > 1:
                color = self.team_colors[team_num]
                color_rgb = hex2color(color)
                highlight_color = '#FFEE88'  # Lighter yellow for current day border

                # Build a map of path index to (segment index, day)
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
                    x0, y0 = prev_x * X_SCALE, prev_y * Y_SCALE
                    x1, y1 = curr_x * X_SCALE, curr_y * Y_SCALE
                    if action_type == 'jump' and is_adjacent_edge:
                        # Jump/revisit edges curve as a quarter-circle arc between
                        # the two hexes instead of a straight line, so a jump path
                        # reads distinctly from a fresh-capture path at a glance.
                        xs, ys = _quarter_circle_arc(x0, y0, x1, y1)
                    else:
                        xs, ys = [x0, x1], [y0, y1]

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
        self._update_enclosure_button_state()
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

    def _tent_stage_start_days(self):
        """Day each Tent-food stage begins, e.g. [1, 15, 25, 43, 64].

        Read from the terrain definition rather than hard-coded, so re-tuning
        the Tent degrade schedule in landInfo.json moves these boundaries too.
        """
        for terrain in _TERRAIN_DB.values():
            if not isinstance(terrain, dict) or terrain.get('name') != 'Tent':
                continue
            degrade = terrain.get('degrade') or []
            days = sorted(
                {int(d['day']) for d in degrade if isinstance(d, dict) and 'day' in d}
            )
            return [1] + [d for d in days if d > 1]
        return [1]

    def _tent_stage_for_day(self, day):
        """1-based Tent-food stage number containing `day`."""
        starts = self._tent_stage_start_days()
        stage = 1
        for idx, start in enumerate(starts, start=1):
            if day >= start:
                stage = idx
            else:
                break
        return stage

    def _compute_hex_capture_phases(self):
        """Classify every captured hex by the game phase it was captured in.

        Returns (phases, max_tent_stage) where phases maps
        {hex_pos: {'bagua': 'before'|'after',
                   'tent': 'tent<N>'|'enclosure',
                   'day': int}}
        and max_tent_stage is the highest Tent-food stage any hex was actually
        captured in - i.e. how many 帐篷N阶段 columns the table needs. If the
        last Tent is only taken in the third stage, that is 3.

        History is replayed in true chronological order - sorted by
        (day, action_order), walking each segment's action sequence - because
        both boundaries can fall in the middle of a day, or even in the middle
        of one click's multi-hex run.

        Boundaries (both are evaluated *before* the hex being classified is
        folded into the running state, so the hex that ends a phase belongs to
        the phase it ends, not the next one - same rule the export's 八卦齐
        column and _compute_jump_summary_before_after_g already use):
          - 八卦前/八卦后: the last of the 8 G/g lands.
          - 帐篷N阶段:  the Tent-food degrade stage (1 = base, 2 = from the
                        first degrade day, ...) the hex was captured in, for
                        everything up to and including the last Tent captured.
          - 圈地阶段:   everything captured after the final Tent.
        """
        phases = {}
        total_g = len(self.all_g_lands)
        total_tents = sum(
            1
            for ir in range(ROWS)
            for ic in range(COLS)
            if _passable(ir, ic) and _terrain(ir, ic).get('name') == 'Tent'
        )

        visited_g = set()
        tents_taken = 0
        max_tent_stage = 1
        self._decisive_g_moment = None

        def classify(day):
            nonlocal max_tent_stage
            bagua = 'after' if (total_g > 0 and len(visited_g) >= total_g) else 'before'
            if total_tents > 0 and tents_taken >= total_tents:
                tent = 'enclosure'
            else:
                stage = self._tent_stage_for_day(day)
                max_tent_stage = max(max_tent_stage, stage)
                tent = f'tent{stage}'
            return {'bagua': bagua, 'tent': tent, 'day': day}

        def take(hex_pos, day, order=None):
            nonlocal tents_taken
            if hex_pos in phases:
                return
            phases[hex_pos] = classify(day)
            if hex_pos in self.all_g_lands:
                visited_g.add(hex_pos)
                # Remember the exact operation that completed the set: the
                # 八卦前 resource snapshot is taken right after it, not at the
                # end of the whole day it happened to fall on.
                if total_g > 0 and len(visited_g) == total_g and self._decisive_g_moment is None:
                    self._decisive_g_moment = {'hex': hex_pos, 'day': day, 'order': order}
            if _terrain(*hex_pos).get('name') == 'Tent':
                tents_taken += 1

        # Team origins are owned from the moment the team exists, and are not
        # part of any segment's captures, so they are seeded first.
        for team in sorted(
            (t for t in (self.team1, self.team2, self.team3) if t is not None),
            key=lambda t: t.created_day,
        ):
            take(tuple(team.origin), team.created_day)

        events = []
        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue
            for seg_idx in range(len(team._seg_lengths)):
                day = team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else 1
                order = (
                    team._seg_action_orders[seg_idx]
                    if seg_idx < len(team._seg_action_orders)
                    and team._seg_action_orders[seg_idx] is not None
                    else 0
                )
                events.append((day, order, team, seg_idx))
        events.sort(key=lambda e: (e[0], e[1]))

        for day, order, team, seg_idx in events:
            actions = (
                team._seg_action_sequence[seg_idx]
                if seg_idx < len(team._seg_action_sequence)
                else []
            )
            if actions:
                for action in actions:
                    if not isinstance(action, (list, tuple)) or len(action) < 2:
                        continue
                    kind, hex_pos = action[0], action[1]
                    if kind != 'new':
                        continue
                    take(tuple(hex_pos), day, order)
            else:
                # Legacy segment with no recorded action sequence: fall back to
                # its capture list (order within the segment is unknown, but
                # the segment's own position in the timeline still is).
                seg_new = team._seg_new_hexes[seg_idx] if seg_idx < len(team._seg_new_hexes) else []
                for hex_pos in seg_new:
                    take(tuple(hex_pos), day, order)

        return phases, max_tent_stage

    def _spend_after_moment_same_day(self, moment):
        """(food, award) priced on `moment`'s day strictly after that operation.

        Lets a mid-day boundary be costed exactly: subtracting nothing gives
        the end-of-day figure day_records holds, and subtracting this residual
        rewinds it back to the instant just after the operation.

        Within a segment, _seg_hex_costs lines up with _seg_path_nodes from the
        back - a leading entry exists only when the segment also settled the
        hex it departed from - which is the same alignment rule
        _recost_segments_from_day uses to re-price stored entries.
        """
        if not moment or moment.get('order') is None:
            return 0, 0

        day = moment['day']
        target_order = moment['order']
        target_hex = moment['hex']

        food_after = 0
        award_after = 0
        for team in (self.team1, self.team2, self.team3):
            if team is None:
                continue
            for seg_idx in range(len(team._seg_lengths)):
                if (team._seg_days[seg_idx] if seg_idx < len(team._seg_days) else None) != day:
                    continue
                order = (
                    team._seg_action_orders[seg_idx]
                    if seg_idx < len(team._seg_action_orders)
                    and team._seg_action_orders[seg_idx] is not None
                    else 0
                )
                if order < target_order:
                    continue

                costs = team._seg_hex_costs[seg_idx] if seg_idx < len(team._seg_hex_costs) else []
                if order > target_order:
                    for entry in costs:
                        food_after += entry[0]
                        award_after += entry[1]
                    continue

                # The segment holding the operation itself: keep only the part
                # priced after it.
                nodes = [tuple(h) for h in (team._seg_path_nodes[seg_idx] or [])]
                prefix = max(0, len(costs) - len(nodes))
                cut = None
                if target_hex in nodes:
                    cut = prefix + nodes.index(target_hex)
                elif prefix > 0:
                    # Settled on departure (probe / deferred fly landing), so
                    # it is priced by one of the leading entries.
                    cut = prefix - 1
                if cut is None:
                    continue
                for entry in costs[cut + 1:]:
                    food_after += entry[0]
                    award_after += entry[1]

        return food_after, award_after

    def _compute_phase_resource_snapshots(self, hex_phases, phase_keys):
        """Food left and cumulative score as each phase ends.

        Returns {phase_key: {'day', 'food', 'score'}} - or None for a phase
        nothing was captured in.

        A phase's end day is the last day it captured anything on, and the
        figures are that day's closing numbers straight out of day_records:
        `food_remain` (余粮) and the running sum of `reward_used` (累计积分).

        八卦前 is the exception: it ends at the exact operation that takes the
        8th G/g, not at the end of whatever day that fell on, so whatever was
        spent and earned later that same day is rewound back out of the
        end-of-day figures. The remaining phases end when the day does.
        """
        snapshots = {pk: None for pk in phase_keys}
        if not self.day_records:
            return snapshots

        end_day = {}
        for info in hex_phases.values():
            day = info.get('day')
            if day is None:
                continue
            for pk in (info['bagua'], info['tent']):
                if pk in snapshots:
                    end_day[pk] = max(end_day.get(pk, day), day)

        cumulative_score = []
        running = 0
        for rec in self.day_records:
            running += rec.get('reward_used', 0)
            cumulative_score.append(running)

        moment = getattr(self, '_decisive_g_moment', None)
        for pk, day in end_day.items():
            idx = min(max(int(day), 1), len(self.day_records)) - 1
            food = self.day_records[idx].get('food_remain', 0)
            score = cumulative_score[idx]
            exact = False
            if pk == 'before' and moment and moment['day'] == day:
                food_after, award_after = self._spend_after_moment_same_day(moment)
                food += food_after
                score -= award_after
                exact = True
            snapshots[pk] = {
                'day': int(day),
                'food': food,
                'score': score,
                'exact': exact,
            }
        return snapshots


    def _create_global_stat_figure(self):
        """Build the 全局统计 figure, its axes and its 复制 button."""
        self.global_stat_fig = plt.figure(figsize=GLOBAL_STAT_FIGSIZE)
        self.global_stat_fig.canvas.manager.set_window_title('全局统计')
        self.global_stat_ax = self.global_stat_fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.global_stat_fig.canvas.mpl_connect('close_event', self._on_global_stat_window_closed)

        # Sits just right of the 全局地块统计 title. Its own axes, so the
        # table redraw's ax.clear() never touches it.
        copy_ax = self.global_stat_fig.add_axes([0.55, 0.927, 0.055, 0.03])
        self._btn_global_stat_copy = Button(copy_ax, '复制', color='#cfe6ff')
        self._btn_global_stat_copy.label.set_fontsize(9)
        self._btn_global_stat_copy.on_clicked(lambda _evt: self._copy_global_stat_to_clipboard())

    def _global_stat_tk_widget(self):
        """A live Tk widget to own the clipboard, or None outside Tk backends."""
        for fig in (getattr(self, 'global_stat_fig', None), getattr(self, 'fig', None)):
            if fig is None:
                continue
            try:
                widget = fig.canvas.get_tk_widget()
            except Exception:
                continue
            if widget is not None:
                return widget
        return None

    def _copy_global_stat_to_clipboard(self):
        """Put the land table on the clipboard as TSV, ready to paste into Excel.

        The rows are the ones the table was last drawn from, so what lands in
        Excel is exactly what is on screen - header, one line per land type,
        and the two end-of-phase resource rows.
        """
        rows = getattr(self, '_global_stat_table_rows', None)
        if not rows:
            self._status_msg = '全局统计：没有可复制的表格内容。'
            return

        text = '\n'.join(
            '\t'.join('' if cell is None else str(cell) for cell in row)
            for row in rows
        )

        widget = self._global_stat_tk_widget()
        try:
            if widget is not None:
                # Clipboard stays valid while the app runs because a live
                # widget owns it.
                widget.clipboard_clear()
                widget.clipboard_append(text)
                widget.update()
            else:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(text)
                # Flush to the OS before the owning interpreter goes away.
                root.update()
                root.destroy()
        except Exception as e:
            print(f'Error copying global stat table: {e}')
            self._status_msg = f'复制失败：{e}'
            return

        print(f'DEBUG: copied global stat table to clipboard ({len(rows)} rows)')
        self._status_msg = f'全局统计表已复制到剪贴板（{len(rows)} 行，可直接粘贴到 Excel）。'
        self._flash_global_stat_copy_button()

    def _flash_global_stat_copy_button(self, revert_ms=1200):
        """Briefly relabel the 复制 button so the click visibly registered."""
        btn = getattr(self, '_btn_global_stat_copy', None)
        if btn is None:
            return
        btn.label.set_text('已复制')
        try:
            self.global_stat_fig.canvas.draw_idle()
        except Exception:
            pass

        widget = self._global_stat_tk_widget()
        if widget is None:
            return

        def _revert():
            try:
                btn.label.set_text('复制')
                self.global_stat_fig.canvas.draw_idle()
            except Exception:
                pass

        try:
            widget.after(revert_ms, _revert)
        except Exception:
            _revert()

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
            self._create_global_stat_figure()

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

            # Exclusions per requirement: Portals, ST, default terrain. B/X/Z
            # buff lands are not excluded - they fold into their tower level
            # via BUFF_TOWER_LEVEL, so e.g. a captured B2 counts as one 塔2.
            stats = {}  # key -> {'token','terrain','total','taken', per-phase counts}
            hex_phases, max_tent_stage = self._compute_hex_capture_phases()
            # How many 帐篷N阶段 columns to show follows the save: if the last
            # Tent was only taken in the third Tent-food stage, three columns
            # are needed, not two.
            tent_keys = tuple(f'tent{n}' for n in range(1, max_tent_stage + 1))
            phase_keys = ('before', 'after') + tent_keys + ('enclosure',)

            for ir in range(ROWS):
                for ic in range(COLS):
                    token = RAW_MAP[ir][ic]

                    # Exclude portals and ST marker.
                    if token == 'ST' or re.fullmatch(r'P\d+', token):
                        continue

                    terrain = _terrain(ir, ic)
                    terrain_name = terrain.get('name', 'unknown')

                    # Exclude default-resolved and empty/default terrain cells.
                    if token not in _TERRAIN_DB:
                        continue
                    if terrain_name in ('default', 'empty'):
                        continue

                    # A buff land is counted as its tower level. Group on that
                    # token alone: B1/X1/Z1 carry three different terrain names
                    # but all belong in the single 塔1 row.
                    group_token = BUFF_TOWER_LEVEL.get(token, token)
                    key = group_token
                    if key not in stats:
                        grouped = group_token != token
                        stats[key] = {
                            'token': group_token,
                            # The tower level may have no terrain entry of its
                            # own (a map can hold B1/X1/Z1 but no plain T1), so
                            # fall back to the buff land's own look.
                            'style_token': (
                                group_token if group_token in _TERRAIN_DB else token
                            ),
                            'terrain': (
                                _TERRAIN_DB.get(group_token, {}).get('name', terrain_name)
                                if grouped else terrain_name
                            ),
                            'total': 0,
                            'taken': 0,
                        }
                        stats[key].update({pk: 0 for pk in phase_keys})

                    stats[key]['total'] += 1
                    if (ir, ic) in self.all_visited_hexes:
                        stats[key]['taken'] += 1
                        # 八卦前/后 and 帐篷1/2/圈地 are two independent splits
                        # of the same captured hexes, so each taken hex adds
                        # one to exactly one column of each pair of groups.
                        phase = hex_phases.get((ir, ic))
                        if phase is not None:
                            stats[key][phase['bagua']] += 1
                            stats[key][phase['tent']] += 1

            # Stable ordering: token then terrain name.
            ordered = sorted(stats.values(), key=lambda r: (r['token'], r['terrain']))

            rows = []
            terrain_styles = []
            total_all = 0
            taken_all = 0
            phase_totals = {pk: 0 for pk in phase_keys}
            snapshots = {}
            for row in ordered:
                remaining = row['total'] - row['taken']
                terrain_def = _TERRAIN_DB.get(row.get('style_token', row['token']), {})
                terrain_fc = terrain_def.get('face', '#cccccc')
                terrain_ec = terrain_def.get('edge', '#000000')
                if terrain_ec in ('none', ''):
                    terrain_ec = '#777777'
                hatch_raw = terrain_def.get('hatch', '')
                terrain_hatch = ''.join(ch * 2 for ch in hatch_raw) if hatch_raw else None

                rows.append([
                    LAND_DISPLAY_NAMES.get(row['token'], row['token']),
                    '',
                    str(row['total']),
                    str(row['taken']),
                    str(remaining),
                ] + [str(row[pk]) for pk in phase_keys])
                terrain_styles.append({
                    'name': row['terrain'],
                    'face': terrain_fc,
                    'edge': terrain_ec,
                    'hatch': terrain_hatch,
                })
                total_all += row['total']
                taken_all += row['taken']
                for pk in phase_keys:
                    phase_totals[pk] += row[pk]

            if rows:
                col_labels = (
                    ['地块', '地形', '总数', '已占领', '剩余', '八卦前', '八卦后']
                    + [f'帐篷{n}阶段' for n in range(1, max_tent_stage + 1)]
                    + ['圈地阶段']
                )
                # Vertical budget, top to bottom: land table, the two
                # end-of-phase resource rows, the jump summary, then the
                # footer lines. The land table stops well above the old 0.26
                # so the rows added under it do not collide with the jump
                # summary below.
                main_bottom = 0.40
                tbl = ax.table(
                    cellText=rows,
                    colLabels=col_labels,
                    loc='upper center',
                    bbox=[0.02, main_bottom, 0.96, 0.96 - main_bottom],
                    cellLoc='center'
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

                # Header styling. The two phase groups get their own header
                # tints so it reads at a glance that 八卦前+八卦后 and
                # 帐篷1+帐篷2+圈地 are two separate splits of 已占领, rather
                # than five more columns that all add up with it.
                bagua_cols = range(5, 7)
                tent_cols = range(7, len(col_labels))
                for c in range(len(col_labels)):
                    if c in bagua_cols:
                        header_fc = '#6b4f9d'
                    elif c in tent_cols:
                        header_fc = '#3f7d5a'
                    else:
                        header_fc = '#446688'
                    cell = tbl[0, c]
                    cell.set_facecolor(header_fc)
                    cell.set_text_props(color='white', weight='bold', ha='center')

                # Row striping, matching each phase group's header tint.
                for r in range(1, len(rows) + 1):
                    odd = r % 2 == 1
                    for c in range(len(col_labels)):
                        if c in bagua_cols:
                            shade = '#f1ebfb' if odd else '#f8f4ff'
                        elif c in tent_cols:
                            shade = '#e9f4ee' if odd else '#f4faf7'
                        else:
                            shade = '#eef4fb' if odd else '#f8fbff'
                        tbl[r, c].set_facecolor(shade)

                # Emphasize hex-grid label in HEX column.
                for r in range(1, len(rows) + 1):
                    tbl[r, 0].set_text_props(weight='bold', color='#1f2d3d')

                # Draw terrain symbol only in Terrain column.
                # Force a draw first so table cell extents are finalized.
                self.global_stat_fig.canvas.draw()
                renderer = self.global_stat_fig.canvas.get_renderer()

                # The symbol's radius is in axes coordinates, which are not
                # square, so the horizontal stretch below has to track the
                # window's aspect or the hexagon smears as the window widens.
                # 2.5 was tuned against _GLOBAL_STAT_BASE_FIGSIZE; rescaling by
                # the aspect ratio keeps the same on-screen shape at any size.
                base_aspect = _GLOBAL_STAT_BASE_FIGSIZE[0] / _GLOBAL_STAT_BASE_FIGSIZE[1]
                fig_w, fig_h = self.global_stat_fig.get_size_inches()
                symbol_stretch = 2.5 * base_aspect / max(fig_w / fig_h, 1e-6)

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
                    # Stretch symbol horizontally while keeping center fixed.
                    symbol.set_transform(
                        Affine2D()
                        .translate(-symbol_center[0], -symbol_center[1])
                        .scale(symbol_stretch, 1.0)
                        .translate(symbol_center[0], symbol_center[1])
                        + ax.transAxes
                    )
                    ax.add_patch(symbol)

                # Two more rows directly under the land table, sharing its
                # column grid: what 余粮 and 累计积分 stood at as each phase
                # ended. Drawn as their own table (rather than extra rows of
                # the one above) so they keep their own styling and are not
                # swept up by the per-row terrain-symbol pass.
                snapshots = self._compute_phase_resource_snapshots(hex_phases, phase_keys)
                res_labels = ('阶段末 余粮', '阶段末 累计积分')
                res_rows = []
                # Same values without thousands separators, for the clipboard:
                # a comma in "2,510" is a decimal separator in some Excel
                # locales, which would silently paste as 2.51.
                res_rows_plain = []
                for label, field in zip(res_labels, ('food', 'score')):
                    row = [label] + [''] * 4
                    plain = [label] + [''] * 4
                    for pk in phase_keys:
                        snap = snapshots.get(pk)
                        row.append('-' if snap is None else f'{snap[field]:,}')
                        plain.append('' if snap is None else str(snap[field]))
                    res_rows.append(row)
                    res_rows_plain.append(plain)

                row_h = (0.96 - main_bottom) / (len(rows) + 1)
                res_tbl = ax.table(
                    cellText=res_rows,
                    loc='upper center',
                    bbox=[0.02, main_bottom - 2 * row_h, 0.96, 2 * row_h],
                    cellLoc='center',
                )
                res_tbl.auto_set_font_size(False)
                res_tbl.set_fontsize(9)

                for r in range(len(res_rows)):
                    for c in range(len(col_labels)):
                        cell = res_tbl[r, c]
                        if c == 0:
                            cell.set_facecolor('#446688')
                            cell.set_text_props(color='white', weight='bold')
                        elif c < 5:
                            # Spacer cells under 地形/总数/已占领/剩余: these
                            # columns have no phase meaning, so leave them blank
                            # and unobtrusive.
                            cell.set_facecolor('#f2f2f2')
                            cell.set_edgecolor('#f2f2f2')
                        elif c in bagua_cols:
                            cell.set_facecolor('#e4d9f5')
                            cell.set_text_props(weight='bold')
                        else:
                            cell.set_facecolor('#dcecE3')
                            cell.set_text_props(weight='bold')

                # Snapshot what was drawn, for the 复制 button - with the
                # resource rows in their unformatted form so Excel reads them
                # as numbers, and without the 地形 column, whose content is a
                # drawn hexagon rather than text and would paste as a blank.
                def _without_terrain_col(row):
                    row = list(row)
                    return row[:1] + row[2:]

                self._global_stat_table_rows = [
                    _without_terrain_col(r)
                    for r in [col_labels] + rows + res_rows_plain
                ]
            else:
                self._global_stat_table_rows = []

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
                # Four columns against the land table's ten: keep this one
                # centered at its own modest width instead of letting it
                # stretch across the widened window.
                bbox=[0.32, 0.155, 0.36, 0.13],
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

            jump_caption = '跳步汇总（首传送门=5跳，重复传送=1跳，飞雷神到传送=0跳）'
            ax.text(0.5, 0.295, jump_caption,
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', ha='center', va='bottom')

            # 复制 takes this table too, as a second block below the land one.
            # An empty row separates them, so pasting lands each block on its
            # own rows in Excel instead of running them together.
            self._global_stat_table_rows = (
                list(self._global_stat_table_rows)
                + [[], [jump_caption], list(jump_labels)]
                + [list(r) for r in jump_rows]
            )

            title = (
                '全局地块统计'
            )
            ax.text(0.5, 0.99, title,
                    transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold')

            ax.text(0.02, 0.065,
                    f'总地块：{total_all}   |   已占领：{taken_all}   |   剩余：{total_all - taken_all}',
                    transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='bottom')

            starts = self._tent_stage_start_days()
            stage_spans = []
            for n in range(1, max_tent_stage + 1):
                lo = starts[n - 1] if n - 1 < len(starts) else starts[-1]
                hi = (starts[n] - 1) if n < len(starts) else None
                stage_spans.append(
                    f'帐篷{n}阶段=Day{lo}~{hi}' if hi is not None else f'帐篷{n}阶段=Day{lo}起'
                )
            stage_spans[-1] += '（至最后一个帐篷）'
            ax.text(0.02, 0.028,
                    f'八卦前 {phase_totals["before"]} / 八卦后 {phase_totals["after"]}'
                    + '   ‖   '
                    + ' / '.join(
                        f'帐篷{n}阶段 {phase_totals[f"tent{n}"]}'
                        for n in range(1, max_tent_stage + 1)
                    )
                    + f' / 圈地阶段 {phase_totals["enclosure"]}'
                    + f'      （八卦前后以第8个G/g为界；{"，".join(stage_spans)}，之后为圈地阶段）',
                    transform=ax.transAxes,
                    fontsize=8, va='bottom', color='#333333')

            phase_label = dict(
                zip(phase_keys, ['八卦前', '八卦后']
                    + [f'帐篷{n}阶段' for n in range(1, max_tent_stage + 1)]
                    + ['圈地阶段'])
            )
            ends = '  '.join(
                f'{phase_label[pk]} D{snapshots[pk]["day"]}'
                + ('(该步结束)' if snapshots[pk].get('exact') else '')
                for pk in phase_keys if snapshots.get(pk)
            )
            if ends:
                ax.text(0.02, 0.105,
                        f'各阶段截至： {ends}'
                        '     （八卦前算到拿下第8个G/g那一步为止，其余算到该日结束）',
                        transform=ax.transAxes,
                        fontsize=8, va='bottom', color='#333333')

            self.global_stat_fig.canvas.draw_idle()
        except Exception as e:
            print(f'Error drawing global stat window: {e}')
            self.global_stat_ax = None
            self.global_stat_fig = None



# ── Entry point ───────────────────────────────────────────────────────────────


# Everything defined above is re-exported so the editor module can do
# `from hex_core import *` and keep referring to RAW_MAP, _terrain, _neighbors
# and friends by their bare names, exactly as when this was one file.
__all__ = [_n for _n in dir() if not _n.startswith('__')]
