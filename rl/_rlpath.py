# -*- coding: utf-8 -*-
"""rl/ 下所有模块共用的导入环境修正。用法: 每个模块开头写

    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import _rlpath  # noqa: F401

前两行只是为了让这一行能找到本文件, 剩下的事情本文件全包了。

要修的是两件事:

1. sys.path 里必须同时有 rl/ 和仓库根。rl/ 下的模块彼此用扁平名互相 import
   (``from rl_engine import ...``), 而 hex_pathfinding_demo 在上一层。只有用
   ``python rl/xxx.py`` 直接启动时 Python 才会自动把 rl/ 加进去 —— 从别处
   ``import rl_engine`` 时不会, 于是六个测试文件会集体 ImportError。

2. 工作目录必须是仓库根。hex_core.py 在 **import 阶段** 就用相对路径读
   landInfo.json 和地图 csv(见 hex_core.py 的 _load_csv / _TERRAIN_DB), 所以在
   rl/ 目录里跑会直接 FileNotFoundError: 'landInfo.json'。这里只在确实找不到
   landInfo.json 时才切目录 —— 已经在正确位置的调用方不会被动到。
"""

import os
import sys

RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)

for _path in (RL_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if not os.path.exists('landInfo.json'):
    os.chdir(REPO_ROOT)
