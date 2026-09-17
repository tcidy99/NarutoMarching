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
    excludes=[],
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
)
