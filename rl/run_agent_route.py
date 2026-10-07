# -*- coding: utf-8 -*-
"""让 agent 打完整整一季, 并把路线导出成软件能直接读入的存档 json。

两种策略:

  random  只从合法动作里均匀乱选(用 action_masks + get_action_candidates 过滤)。
          这是环境的连通性自检: 只要它能走出第一天、把三队建起来、一路推到 day 97,
          就说明动作空间/掩码/建队/换日这条链是通的。不要指望分数。

  greedy  贪心: 每一步在相邻可占的格子里挑 "积分 - λ x 粮草" 最高的那个; 步数用完
          就换下一支队伍; 第一天复刻人工打法 —— team1 走完开 team2, team2 走完开
          team3(奖励函数里 readiness 系数鼓励的正是这个顺序)。

用法:
    python rl/run_agent_route.py --policy greedy --out rl/agent_route.json
    python rl/run_agent_route.py --policy random --episodes 3

导出的存档用的就是软件保存时同一个函数(_build_game_state_for_save), 所以可以直接
在编辑器里"读取存档"打开。
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _rlpath  # noqa: F401  统一把 rl/ 与仓库根加进 sys.path, 并校正工作目录

import matplotlib
matplotlib.use('Agg')
import hex_pathfinding_demo as game
from rl_engine import NarutoMarchingEngine
from rl_playground import ActionType, RouteAction

import tkinter.messagebox as _mb
_mb.askyesno = lambda *a, **k: True
for _n in ('showinfo', 'showwarning', 'showerror'):
    setattr(_mb, _n, lambda *a, **k: None)

# 粮草低于这个数就不再主动占新格了 —— 留一点余量给帐篷之类的负粮地块和收尾。
FOOD_FLOOR = 250

# 每支队伍每天最多走多少个动作。跳步不耗步数, 没有这个上限会死循环。
MAX_MOVES_PER_TEAM_DAY = 30

# 踩自己已占领的格子(跳步)的固定粮草开销, 见规则 4.2。
JUMP_FOOD = 10

# beam search 的搜索深度与宽度(--policy beam 时生效)
BEAM_DEPTH = 6
BEAM_WIDTH = 24
BEAM_JUMP_PENALTY = 10   # 搜索里跳步的额外惩罚倍数, 见 beam_plan 的说明

# 特性开关。默认值是实测最优 —— 在 λ=0.30 下逐项消融(同一张图, 同一套参数):
#
#     全开(默认)          66,146 分   占领 1102   卦 8/8  帐篷 15/15  六级 165/165
#     关掉飞雷神           66,146 分   占领 1102   卦 8/8  帐篷 15/15  六级 165/165
#     不看 B/Z 窗口估值    66,146 分   占领 1102   卦 8/8  帐篷 15/15  六级 165/165
#     不追分层目标         66,146 分   占领 1102   卦 8/8  帐篷 15/15  六级 165/165
#     关掉传送点           64,108 分   占领 1130   卦 8/8  帐篷 15/15  六级  87/165
#
# 值得记下来的两件事:
#
# 1) 传送点仍然是唯一有量级影响的一项(+2,038)。南半边的四个卦只能从传送点过去,
#    而且关掉之后六级地只剩 87/165 —— 南边那一半的好地也一起丢了。
#
# 2) "B/Z 窗口内逐格重估值"实测是零收益。原因不是实现不对, 是这张图上增益地块
#    统共只有 6 块(B2/B3/X1x2/Z2/Z3), 窗口 5~10 步, 占整季 1,150 次占领的 3%;
#    而且窗口里相邻几个格子的成本/积分差不多, 乘同一个系数不改变谁最优。
#    真正有用的是把**高价目标**排进窗口 —— 见 objective_step 里战勋窗口那段:
#    窗口开着时专门去够 1000 分的大boss / 520 分的历练, 单这一项 +1,921 分
#    (64,225 -> 66,146)。人工所谓"把高价地块排进窗口", 关键在"高价", 不在"每格重估"。
USE_PORTAL = True
USE_TENT_OBJECTIVE = True
USE_FLY = True


def _quiet(fn, *a, **k):
    """引擎内部有大量 DEBUG print, 全部吞掉。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def new_engine():
    engine = NarutoMarchingEngine()
    _quiet(engine.reset)
    return engine


def teams_of(engine):
    return [(i, t) for i, t in enumerate((engine.team1, engine.team2, engine.team3), 1)
            if t is not None]


def steps_left(engine, team):
    return engine._get_team_steps_for_day(team, engine.current_day)


def candidates(engine, team):
    """该队当前能走到的相邻新格, 附上地形给的积分/粮草/步数。"""
    here = tuple(team.full_path[-1])
    out = []
    for pos in game._neighbors(*here):
        if not game._passable(*pos) or pos in engine.all_visited_hexes:
            continue
        terrain = game._terrain(*pos)
        if terrain.get('name') == 'empty':
            continue
        out.append((pos, terrain.get('award', 0), terrain.get('food', 0),
                    terrain.get('step', 1)))
    return out


# 地形本身的积分之外, 额外的"战略价值"。单位和积分一样, 只用来排序。
#   G/g  八个全部占完 -> 此后所有成本永久 8 折, 比它自己那点积分值钱得多。
#   T    帐篷倒贴 300 粮、还倒还 1 步。这个 agent 卡的就是粮草, 所以一个帐篷等于
#        多出十来个六级地的预算 —— 权重要压过任何普通地块。
#   B/b  大/中 boss, 积分本身就高(1000/500), 大boss 还 +1 次飞雷神。
#   B/X/Z 增益地块本身积分平平, 但占了之后接下来 5~10 步有折扣/加成, 值得专程去踩。
STRATEGIC_BONUS = {
    'G': 4000, 'g': 4000,
    'T': 2500,
    'B': 400, 'b': 200,
    'B2': 600, 'B3': 800, 'B1': 400,   # 兵粮: 之后 5/8/10 步粮草打六折
    'Z2': 600, 'Z3': 800, 'Z1': 400,   # 战勋: 之后 5/8/10 步积分 x1.4
    'X1': 300, 'X2': 400, 'X3': 500,   # 迅疾: 之后 5/8/10 步不耗步数
    # 注意: 这里**故意没有**给 4/5/6 级地、塔、商人加额外权重。试过, 反而掉分:
    #   旧权重(不加)        66,202 分   六级 164/165  塔 33/75  商人 27/40
    #   只加商人+塔         63,391 分   六级  70/165  塔 58/75  商人 34/40
    #   温和加 4/5/6+塔+商人 63,336 分   六级  66/165  塔 59/75  商人 33/40
    #   全套加              64,943 分   六级 122/165  塔 47/75  商人 30/40
    # 原因是这个 agent 卡粮草: 给塔/商人加权重会把队伍拽到一片低价地里去, 沿途
    # 每踩一格自己的地还要付 10 粮跳步钱, 拿到的塔和商人补不回绕路的开销 ——
    # 而且六级地反而从 164 掉到 70。光靠 award - λ*food 排序, 六级地本来就排在
    # 一/二/三级地前面(见 hex_value), 已经能拿到 164/165。
}


def hex_value(engine, pos, food_weight, team=None):
    """这一格现在值多少。

    关键是"现在": B/Z 增益是**有窗口期**的, 同一个格子在窗口里和窗口外价值不同 ——
    人工打法就是刻意把贵的、高分的地块排进窗口里打(规则 §5):

      · 兵粮(B) 生效时, 粮草成本打六折 -> 窗口里应该去啃**贵**的地块, 省得最多;
        窗口外去啃贵地块就是亏。
      · 战勋(Z) 生效时, 积分 x1.4 -> 窗口里应该去啃**高分**的地块。
      · 迅疾(X) 生效时不耗步数 -> 窗口里能多走几格, 但这个 agent 卡粮草不卡步数,
        所以对估值没影响, 只在"还能不能动"那里起作用。
    """
    terrain = game._terrain(*pos)
    token = game.RAW_MAP[pos[0]][pos[1]]
    food = float(terrain.get('food', 0))
    award = float(terrain.get('award', 0))
    if team is not None:
        if getattr(team, 'b_discount_remaining', 0) > 0 and food > 0:
            rate = game._TERRAIN_DB.get(getattr(team, 'b_discount_name', '') or '',
                                        {}).get('food_discount_rate', 0.4)
            food *= (1.0 - rate)
        if getattr(team, 'z_bonus_remaining', 0) > 0 and award > 0:
            rate = game._TERRAIN_DB.get(getattr(team, 'z_bonus_name', '') or '',
                                        {}).get('reward_bonus_rate', 1.4)
            award *= rate
    return award + STRATEGIC_BONUS.get(token, 0) - food_weight * food


def greedy_pick(engine, team, food_weight):
    """挑价值最高的相邻新格; 没有可占的就返回 None。"""
    best, best_value = None, None
    for pos, _award, food, _step in candidates(engine, team):
        if food > 0 and engine.current_food - food < FOOD_FLOOR:
            continue
        value = hex_value(engine, pos, food_weight, team)
        if best_value is None or value > best_value:
            best, best_value = pos, value
    return best


def nearest_frontier(engine, team, food_weight=0.30, max_radius=22):
    """从队伍当前位置广度优先, 找最近的可占新格。

    队伍把自己围在已占领地里之后, 相邻格全是自己的地, greedy_pick 会一直返回 None,
    队伍就永远停在原地了(第一版就是这么整季只走了 51 格)。人的打法是踩着自己的
    地块"跳步"穿过去(每格固定 10 粮), 到边界再继续占 —— 这里就是找那个边界:
    交给引擎自己 A* 寻路过去(和在界面上点一个远处格子一样)。

    最近不等于最好: 八个 G/g 值得多走几步去拿(拿齐之后全局永久 8 折), 帐篷是倒贴
    粮草的。所以这里按 "价值 - 每步折价" 在搜索半径内挑最划算的那个, 而不是见到
    第一个新格就返回。
    """
    start = tuple(team.full_path[-1])
    seen = {start}
    frontier = [start]
    best, best_value = None, None
    for distance in range(1, max_radius + 1):
        nxt = []
        for pos in frontier:
            for nb in game._neighbors(*pos):
                if nb in seen or not game._passable(*nb):
                    continue
                seen.add(nb)
                terrain = game._terrain(*nb)
                if terrain.get('name') == 'empty':
                    continue
                if nb not in engine.all_visited_hexes:
                    # 每多走一步要多花约 10 粮的跳步钱, 折价按距离线性算。
                    value = hex_value(engine, nb, food_weight, team) - 12.0 * distance
                    if best_value is None or value > best_value:
                        best, best_value = nb, value
                else:
                    nxt.append(nb)
        # 已经找到很值钱的目标(比如 G/g)就不用再往外搜了
        if best is not None and best_value is not None and best_value > 1000:
            break
        if not nxt:
            break
        frontier = nxt
    return best


# ── 传送点: 进入地图南半边的唯一通道 ──────────────────────────────────────
#
# S25 地图被不可通行的地形切成两块。从出生点 (6,41) 走路只能到北半边 —— 八个卦
# 里的 (19,9) (20,38) (25,19) (28,33) 在这边; 另外四个 (36,13) (48,12) (52,37)
# (55,22) 在南半边, A* 直接返回"无路"。第一版 agent 整季只拿到 4 个卦就是卡在这里。
#
# 连接两边的只有传送点。踩上传送点格、并在弹窗里选"传送", 就会被送到配对的另一端:
#     P2 (21,42) <-> (51,19)     P3 (25,54) <-> (53,41)
#     P4 (22,43) <-> (56,26)     P5 ( 9,16) <-> (40, 6)
#     P1 (22, 5) <-> (24, 4)     (两端都在北边, 只是条近路)
# 引擎里对应 portal_teleport 这个开关(见 rl_engine._check_portal_teleport):
# 打开之后, 只要某一步的**终点**落在传送点上就会自动传送过去。


def _build_portal_pairs():
    pairs = {}
    by_token = {}
    for row in range(game.ROWS):
        for col in range(game.COLS):
            token = game.RAW_MAP[row][col]
            if game.re.fullmatch(r'P\d+', token):
                by_token.setdefault(token, []).append((row, col))
    for ends in by_token.values():
        if len(ends) == 2:
            pairs[ends[0]] = ends[1]
            pairs[ends[1]] = ends[0]
    return pairs


PORTAL_PAIRS = _build_portal_pairs()

def hexes_with(tokens):
    return [(r, c) for r in range(game.ROWS) for c in range(game.COLS)
            if game.RAW_MAP[r][c] in tokens]


TENT_HEXES = hexes_with(('T',))

# 卦拿齐之后, 值得专程跑一趟的目标, 按优先级排; 第二个数字是允许的绕路格数
# (跨自己领地每格 10 粮跳步钱, 所以越不值钱的目标, 绕路上限越小)。
#   帐篷   倒贴 300 粮 = 后面十来个格子的预算, 走 18 格(180 粮)去拿也划算
#   boss   1000/500 分
#   增益地 占了之后 5~10 步打折/加成
#   商人   0 粮 30 分, 白捡, 但只值得顺路
# 全图最值钱的几块 —— 战勋窗口开着时优先往这里排。
HIGH_AWARD_HEXES = hexes_with(('B', 'L', 'b', 'G'))

OBJECTIVE_TIERS = [
    (TENT_HEXES, 18),
    (hexes_with(('B', 'b')), 14),
    (hexes_with(('B1', 'B2', 'B3', 'Z1', 'Z2', 'Z3', 'X1', 'X2', 'X3')), 12),
    (hexes_with(('M',)), 8),
    (hexes_with(('T6', 'T5', 'T4', 'T3')), 8),
]


def walk_region(origin):
    """从某格出发, 光靠走路能到达的全部格子。"""
    seen = {origin}
    stack = [origin]
    while stack:
        pos = stack.pop()
        for nb in game._neighbors(*pos):
            if nb in seen or not game._passable(*nb):
                continue
            seen.add(nb)
            stack.append(nb)
    return seen


def portal_for(here, goal):
    """要去 goal 但走不到时, 返回该踩哪个传送点(必须是 here 走得到的那一端)。"""
    here_region = walk_region(here)
    if goal in here_region:
        return None
    for near_end, far_end in PORTAL_PAIRS.items():
        if near_end not in here_region:
            continue
        if goal in walk_region(far_end):
            return near_end
    return None


def hex_distance(a, b):
    """odd-q 偏移坐标下的六边形距离(先转成立方坐标再算)。"""
    def to_cube(pos):
        col, row = pos[1], pos[0]
        x = col
        z = row - (col - (col & 1)) // 2
        return x, -x - z, z
    ax, ay, az = to_cube(a)
    bx, by, bz = to_cube(b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def objective_step(engine, team, max_detour=26):
    """八个 G/g 还没凑齐时, 朝最近的那个走一步(返回下一格); 否则返回 None。

    凑齐八个卦之后, 此后所有挑战成本/移动成本/跳步成本永久打八折(规则 §6)。整季
    十六万粮草, 两成就是三万多 —— 折合两百多个格子, 远超过绕路那几步的代价。
    第一版 agent 整季只拿到 1 个卦, 这是它和人工 70,558 分之间最大的一块差距。

    这里只返回**下一格**, 让队伍沿 A* 路径一步步挪过去, 而不是一次点到终点 ——
    一天只有 6 步, 一次点太远会把好几天的步数一口气吃掉。
    """
    # 战勋窗口开着的时候, 优先去啃全图最值钱的那几块(大boss 1000 分、历练 520、
    # 中boss 500、大卦 300)。积分 x1.4 对普通地块只多二十来分, 但把一个 1000 分的
    # 大boss 排进窗口里就是白赚 400 分 —— 人工刻意做的就是这件事。
    # 只在窗口剩余步数够走过去时才去, 否则走到一半窗口就没了。
    z_left = getattr(team, 'z_bonus_remaining', 0)
    if z_left > 0:
        here_now = tuple(team.full_path[-1])
        prizes = [h for h in HIGH_AWARD_HEXES if h not in engine.all_visited_hexes
                  and hex_distance(here_now, h) <= z_left]
        if prizes:
            remaining, chasing_g, max_detour = prizes, False, z_left
            goal = min(prizes, key=lambda h: hex_distance(here_now, h))
            path, _cost = game._astar(here_now, goal, set())
            if path and len(path) >= 2:
                return tuple(path[1])

    remaining = [h for h in engine.all_g_lands if h not in engine.visited_g_lands]
    chasing_g = bool(remaining)
    if not remaining:
        # 卦拿齐之后按"值不值得专程跑一趟"依次追: 帐篷(倒贴 300 粮, 直接变成后面
        # 十来个格子的预算) -> boss/增益地 -> 商人(0 粮 30 分, 白捡)。
        # 每一类都有自己的绕路上限: 跨自己领地每格要付 10 粮跳步钱, 跑太远就亏了。
        for targets, budget in OBJECTIVE_TIERS:
            remaining = [h for h in targets if h not in engine.all_visited_hexes]
            if remaining:
                max_detour = budget
                break
    if not remaining:
        return None
    here = tuple(team.full_path[-1])

    # 每走一步就对八个目标各跑一次 A*(每次要搜三千多格)实在太慢 —— 第一版这么写
    # 直接跑了十分钟还没结束。改成: 先用六边形距离挑最近的那个目标, 只对它跑一次
    # A*, 并把整条路径缓存在队伍上, 之后每步直接取下一格。
    goal = min(remaining, key=lambda g: hex_distance(here, g))

    # 走不到就改走传送点: 把"要去的地方"换成本侧的那个传送点入口, 踩上去之后
    # 引擎会把队伍送到对岸(见 try_move 里对 portal_teleport 的处理)。
    via_portal = None
    reachable, _ = game._astar(here, goal, set())
    if not reachable:
        # 只有"卦"值得为它跨大陆: 拿齐八个是永久八折。帐篷不值 —— 跨图要踩几十个
        # 已占领格, 每格 10 粮的跳步钱比帐篷倒贴回来的还多(实测这么干反而掉 9 千分)。
        if not chasing_g or not USE_PORTAL:
            return None
        via_portal = portal_for(here, goal)
        if via_portal is None:
            return None
        goal = via_portal
    elif hex_distance(here, goal) > max_detour:
        return None

    cached = getattr(team, '_agent_objective', None)
    if (cached is None or cached['goal'] != goal
            or here not in cached['index']):
        path, _cost = game._astar(here, goal, set())
        if not path or len(path) < 2:
            return None
        cached = {'goal': goal, 'path': path,
                  'index': {tuple(p): i for i, p in enumerate(path)}}
        team._agent_objective = cached
    position = cached['index'][here]
    if position + 1 >= len(cached['path']):
        return None
    return tuple(cached['path'][position + 1])


def try_move(engine, team_number, target, teleport=False):
    """走一步。teleport=True 时, 如果这一步的终点是传送点就顺势传过去。

    引擎把"是否传送"做成了一个回合级开关 portal_teleport, 落点是传送点时才会看它。
    所以这里只在需要的那一步临时打开, 走完立刻关回去 —— 否则以后每次不小心踩到
    传送点都会被莫名其妙地传走。
    """
    previous = engine.portal_teleport
    engine.portal_teleport = bool(teleport)
    try:
        _quiet(engine.step,
               RouteAction(action_type=ActionType.MOVE_TO_HEX,
                           target=target, team=team_number))
        return True
    except Exception:
        return False
    finally:
        engine.portal_teleport = previous


def try_create(engine, team_number):
    try:
        _quiet(engine.step,
               RouteAction(action_type=ActionType.CREATE_TEAM, team=team_number))
        return True
    except Exception:
        return False


def try_fly(engine, team_number, team, food_weight):
    """用一次飞雷神。飞不成就返回 False, 不改任何状态。

    规则(§8): 目标必须紧邻"本队所连通的已占领区域"的边缘, 且本身未被占领; 落地
    **不消耗步数**、免移动粮草。所以最划算的用法是等这支队伍当天步数已经用完、
    走不动了, 再飞出去白捡一格 —— 这一步完全是白赚的。

    落地的格子算"探索"状态, 要等队伍离开或原地确认才结算(和免费探索一样), 引擎
    会自己处理。
    """
    if engine.fly_skill_limit <= 0:
        return False
    # 本队连通领地的外沿: 从队伍当前位置沿"已占领"格扩散, 收集紧邻的未占领格
    here = tuple(team.full_path[-1])
    seen = {here}
    stack = [here]
    rim = set()
    while stack:
        pos = stack.pop()
        for nb in game._neighbors(*pos):
            if nb in seen or not game._passable(*nb):
                continue
            seen.add(nb)
            if nb in engine.all_visited_hexes:
                stack.append(nb)
            else:
                terrain = game._terrain(*nb)
                if terrain.get('name') != 'empty':
                    rim.add(nb)
    if not rim:
        return False
    target = max(rim, key=lambda p: hex_value(engine, p, food_weight, team))
    try:
        _quiet(engine.step,
               RouteAction(action_type=ActionType.FLY_TO_HEX,
                           target=target, team=team_number))
    except Exception:
        return False
    return True


def advance(engine):
    _quiet(engine.step, RouteAction(action_type=ActionType.ADVANCE_DAY,
                                    team=teams_of(engine)[0][0]))


def beam_plan(engine, team, food_weight, depth, width=24):
    """给这支队伍规划接下来 depth 步的整条路线, 而不是一步一步地贪。

    贪心只看下一格, 所以会为了眼前一个三级地放弃"穿过两个便宜格子就能吃到一片
    六级地"的走法 —— 这正是人和它之间剩下的那点差距。这里做一次定深 beam search:
    每一层把所有候选展开, 按累计价值留最好的 width 条, 最后把最优那条的第一步
    交出去(下一步会重新规划, 相当于滚动时域)。

    只用地形表估值, 不调引擎 —— 引擎 deepcopy 在后期有七百多段历史, 一次几十毫秒,
    放进搜索里整季要跑几十分钟。地形估值和引擎的真实计费差一个全局折扣和 buff 系数,
    但那些对**排序**没有影响。

    实测结论: **前瞻在这张图上没用, 反而不如贪心。**(同为 λ=0.30)

        贪心                              66,202 分   占领 1102   六级 164/165
        beam 深度4 (跳步罚x5, 首步必占)     65,405 分   占领 1114   六级  89/165
        beam 深度6 (跳步罚x10, 首步必占)    65,915 分   占领 1107   六级 149/165
        beam 深度8 (跳步罚x5,  首步必占)    65,535 分   占领 1128   六级  99/165
        beam 深度6 (早期版本: 不罚跳步)     42,842 分   占领  666   <- 见下

    两点原因:
      1. 最初那版把跳步只按 10 粮计, 结果深度越大越爱"穿过自己的地去更好的片区",
         整季的粮草都花在赶路上, 占领数从 1102 掉到 666。加重跳步惩罚、并且规定
         第一步必须是占格之后才回到 65,915。
      2. 就算修好了也赢不过贪心。因为真正的约束是**全局粮草预算**, 决策本质是
         "这一格值不值它那份粮" —— 这是个背包问题, 按分/粮比贪心就是它的最优松弛解。
         地图又很密(相邻格几乎都能占), 不存在"穿过两个废格才能吃到宝藏"这种
         需要前瞻才能发现的结构; 唯一需要前瞻的目标(八卦、战勋窗口里的大boss)
         已经在 objective_step 里单独处理了。
    """
    start = tuple(team.full_path[-1])
    taken = engine.all_visited_hexes
    # 每条候选: (累计价值, 当前位置, 这一路占掉的新格, 第一步)
    beam = [(0.0, start, frozenset(), None)]
    best = (0.0, None)
    for level in range(depth):
        nxt = []
        for value, pos, claimed, first in beam:
            for nb in game._neighbors(*pos):
                if not game._passable(*nb) or nb in claimed:
                    continue
                terrain = game._terrain(*nb)
                if terrain.get('name') == 'empty':
                    continue
                if nb in taken:
                    # 第一步不许是跳步: 否则搜索会热衷于"先赶路再说", 把粮草烧在
                    # 路上。后面几步的跳步也按 BEAM_JUMP_PENALTY 倍加重惩罚。
                    if level == 0:
                        continue
                    gain = -JUMP_FOOD * food_weight * BEAM_JUMP_PENALTY
                    new_claimed = claimed
                else:
                    gain = hex_value(engine, nb, food_weight, team)
                    new_claimed = claimed | {nb}
                step_first = first if first is not None else nb
                nxt.append((value + gain, nb, new_claimed, step_first))
        if not nxt:
            break
        nxt.sort(key=lambda item: -item[0])
        beam = nxt[:width]
        if beam[0][0] > best[0]:
            best = (beam[0][0], beam[0][3])
    return best[1]


def rushing_gua(engine):
    """八卦还没拿齐 —— 此时应该"少占地、快赶路"。

    八个 G/g 拿齐之后, 此后所有成本永久打八折。在那之前占下的每一格都是按原价买的,
    同一格晚几天再占就便宜两成。所以拿齐之前要克制: 只做三件事 ——
    赶路去卦、顺手捡帐篷(倒贴粮草, 而且越早倒贴越多)、捡免费的商人;
    **不要**为了把当天步数和粮草用完而去啃普通地块。
    """
    return len(engine.visited_g_lands) < len(engine.all_g_lands)


def worth_taking_while_rushing(pos):
    """赶卦阶段允许顺手拿的格子: 只有倒贴粮草的帐篷和完全免费的商人/历练。"""
    token = game.RAW_MAP[pos[0]][pos[1]]
    if token in ('T', 'M', 'L'):
        return True
    terrain = game._terrain(*pos)
    return terrain.get('food', 0) <= 0


def play_greedy(engine, food_weight=0.30, verbose=False, use_beam=False):
    """整季贪心。返回每天的简要日志。"""
    log = []
    # 用 while True + 末尾判断, 不要写成 while current_day < TOTAL_DAYS ——
    # 那样最后一天(day 97)会在"还没轮到它行动"的时候就退出循环, 整季白丢
    # 一天。人工存档 day 97 是走了 11 段的。
    while True:
        day = engine.current_day
        last_day = day >= game.TOTAL_DAYS
        # 最后一天没有"以后"了, 余粮留着一分不值 —— 把预留额度降到 0,
        # 能买多少格就买多少格。(平时留 FOOD_FLOOR 是为了接得住帐篷之类
        # 的负粮地块和收尾。)
        floor = 0 if last_day else FOOD_FLOOR
        moved_today = 0
        # 每建一支队伍, teams_of 的结果就变了, 所以用下标循环而不是一次性取列表。
        index = 0
        while index < len(teams_of(engine)):
            team_number, team = teams_of(engine)[index]
            index += 1
            # 这一支先走到没步数为止。
            # 注意不能只看步数: 踩在已占领的格子上是"跳步", 固定 10 粮、**不消耗
            # 步数**, 所以朝远处目标穿越自己领地时 steps_left 一直不降, 光靠
            # while steps_left > 0 会死循环(第一版就卡在这儿跑不完)。这里加一个
            # 每队每天的动作上限兜底。
            moves_this_team = 0
            while steps_left(engine, team) > 0 and moves_this_team < MAX_MOVES_PER_TEAM_DAY:
                if engine.current_food < floor:
                    break
                moves_this_team += 1
                # 八个 G/g 没凑齐之前, 优先朝最近的那个进发 —— 凑齐之后所有成本
                # 永久 8 折, 整季相当于白得两成粮草, 比眼前多占一格值钱得多。
                target = objective_step(engine, team)
                if target is None and rushing_gua(engine):
                    # 赶卦阶段: 只顺手捡倒贴粮草/免费的格子, 其余一律不占 ——
                    # 早占一格就是多付两成原价。捡不到就收手, 步数留着明天用。
                    pick = greedy_pick(engine, team, food_weight)
                    target = pick if (pick and worth_taking_while_rushing(pick)) else None
                    if target is None:
                        break
                if target is None and use_beam:
                    target = beam_plan(engine, team, food_weight,
                                       BEAM_DEPTH, BEAM_WIDTH)
                if target is None:
                    target = greedy_pick(engine, team, food_weight)
                if target is None:
                    # 周围全是自己的地了, 踩着自己的地跳到最近的边界去
                    target = nearest_frontier(engine, team, food_weight)
                    if target is None:
                        break
                # 落点是传送点, 而且对岸还有没拿的卦 -> 顺势传过去
                want_teleport = (
                    target in PORTAL_PAIRS
                    and any(h not in engine.visited_g_lands
                            for h in engine.all_g_lands)
                )
                if not try_move(engine, team_number, target, want_teleport):
                    break
                if want_teleport:
                    team._agent_objective = None  # 换了大陆, 旧路径作废
                moved_today += 1
            # 步数走光了还剩飞雷神, 就飞一格 —— 飞雷神不耗步数, 纯白赚。
            # 留一次在手上应急(比如后面要跨过一片已占领区去够边角)。
            if (USE_FLY and steps_left(engine, team) <= 0 and engine.fly_skill_limit > 1
                    and engine.current_food > floor):
                if try_fly(engine, team_number, team, food_weight):
                    moved_today += 1

            # 走完了就在它脚下开下一支队伍 —— 与人工存档 day 1 的顺序一致
            if engine.team2 is None and steps_left(engine, team) <= 0:
                try_create(engine, 2)
            elif (engine.team2 is not None and engine.team3 is None
                  and steps_left(engine, team) <= 0):
                try_create(engine, 3)
        log.append({
            'day': day,
            'moves': moved_today,
            'food': engine.current_food,
            'score': engine.total_reward,
            'occupied': len(engine.all_visited_hexes),
            'teams': len(teams_of(engine)),
        })
        if verbose and day % 10 == 0:
            print('   day %-3d 走 %-3d 步  余粮 %-6d 积分 %-7s 占领 %d'
                  % (day, moved_today, engine.current_food,
                     format(engine.total_reward, ','), len(engine.all_visited_hexes)))
        if last_day:
            break
        advance(engine)
    return log


def play_random(engine, rng, max_steps=4000):
    """只从合法动作里乱选。用来验证环境是通的, 不是用来拿分的。"""
    invalid = 0
    for _ in range(max_steps):
        if engine.current_day >= game.TOTAL_DAYS:
            break
        movers = [(i, t) for i, t in teams_of(engine) if steps_left(engine, t) > 0]
        choices = ['advance']
        if movers:
            choices += ['move'] * 6
        if engine.team2 is None or engine.team3 is None:
            choices += ['create'] * 2
        what = rng.choice(choices)
        if what == 'create':
            target_team = 2 if engine.team2 is None else 3
            if not try_create(engine, target_team):
                invalid += 1
        elif what == 'move' and movers:
            team_number, team = rng.choice(movers)
            options = candidates(engine, team)
            if not options:
                advance(engine)
                continue
            pos = rng.choice(options)[0]
            if not try_move(engine, team_number, pos):
                invalid += 1
        else:
            advance(engine)
    return invalid


def export_save(engine, path):
    """用软件保存时同一个序列化函数导出, 保证能被编辑器读回去。"""
    state = engine._build_game_state_for_save(None)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    return os.path.getsize(path)


def summarize(engine, label, elapsed):
    print('%s: day %d / 积分 %s / 耗粮 %s / 余粮 %d / 占领 %d 格 / 队伍 %d 支 / %.1fs'
          % (label, engine.current_day, format(engine.total_reward, ','),
             format(engine.total_food, ','), engine.current_food,
             len(engine.all_visited_hexes), len(teams_of(engine)), elapsed))
    for i, t in teams_of(engine):
        print('     team%d  建于 day%-3d 段数 %-4d 终点 %s'
              % (i, t.created_day, len(t._seg_lengths), tuple(t.full_path[-1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['greedy', 'beam', 'random'], default='greedy')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--food-weight', type=float, default=0.30)
    parser.add_argument('--out', default=None)
    parser.add_argument('--seed', type=int, default=0)
    # 三个开关都用 --xxx / --no-xxx 成对给出, 默认 None = 用模块里的实测最优值。
    # (之前只有 --fly 这种 store_true, 结果命令行默认 False 会把模块里的 True
    #  盖掉 —— 不给参数跑出来的反而不是默认配置。)
    for name, helptext in (
            ('portal', '传送点: 进入地图南半边的唯一通道, 关掉约掉 2 千分'),
            ('tent-objective', '卦拿齐后按分层专程去够帐篷/boss/增益地/商人'),
            ('fly', '步数用完后用飞雷神白捡一格')):
        flag = name.replace('-', '_')
        parser.add_argument('--' + name, dest=flag, action='store_true',
                            default=None, help=helptext)
        parser.add_argument('--no-' + name, dest=flag, action='store_false',
                            help='关闭: ' + helptext)
    args = parser.parse_args()

    global USE_TENT_OBJECTIVE, USE_FLY, USE_PORTAL
    if args.portal is not None:
        USE_PORTAL = args.portal
    if args.tent_objective is not None:
        USE_TENT_OBJECTIVE = args.tent_objective
    if args.fly is not None:
        USE_FLY = args.fly

    best = None
    for episode in range(args.episodes):
        engine = new_engine()
        started = time.perf_counter()
        if args.policy in ('greedy', 'beam'):
            play_greedy(engine, args.food_weight, verbose=(args.episodes == 1),
                        use_beam=(args.policy == 'beam'))
        else:
            rng = random.Random(args.seed + episode)
            invalid = play_random(engine, rng)
            print('   (乱选策略, 非法动作 %d 次)' % invalid)
        elapsed = time.perf_counter() - started
        summarize(engine, '第 %d 局 [%s]' % (episode + 1, args.policy), elapsed)
        if best is None or engine.total_reward > best.total_reward:
            best = engine

    if args.out and best is not None:
        size = export_save(best, args.out)
        print()
        print('已导出存档: %s  (%s 字节, 积分 %s)'
              % (args.out, format(size, ','), format(best.total_reward, ',')))
    return best


if __name__ == '__main__':
    main()
