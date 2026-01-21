# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEEG Updater.

This creates MEEG_updater.exe which handles automatic updates.
Run with: pyinstaller MEEG_updater.spec
"""

import sys
from pathlib import Path

project_root = Path(SPECPATH)
is_windows = sys.platform == 'win32'

block_cipher = None

a = Analysis(
    ['MEEG_updater.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MEEG_updater',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,  # Show console for update progress
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/img/Logo.ico' if is_windows else None,
)
