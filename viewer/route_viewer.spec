# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# route_viewer.py imports hex_pathfinding_demo.py from the repo root (one
# level up from this spec's own folder), so that root must be on pathex
# regardless of the current working directory pyinstaller is invoked from.
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

openpyxl_hiddenimports = collect_submodules('openpyxl')
openpyxl_datas = collect_data_files('openpyxl')

a = Analysis(
    [os.path.join(SPECPATH, 'route_viewer.py')],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=openpyxl_datas,
    hiddenimports=openpyxl_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # route_viewer imports hex_core only. Naming the editor module here is a
    # hard guarantee: if some future import ever reaches for it, the build
    # fails loudly instead of silently shipping the editor's bytecode inside
    # the read-only viewer.
    excludes=['hex_pathfinding_demo'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='NarutoMarching_Viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon.ico 由仓库根目录的 icon.jpg 生成(白底补边成正方, 内含 16~256 六种
    # 尺寸)。Windows 只认 .ico, 直接给 .jpg 会打包失败。
    icon=os.path.join(REPO_ROOT, 'icon.ico'),
)
