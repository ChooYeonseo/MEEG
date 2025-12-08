# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEEG Application.

This creates a macOS .app bundle (or Windows .exe on Windows).
Run with: pyinstaller MEEG.spec
"""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH)

block_cipher = None

# Data files to include
added_files = [
    # Assets
    ('assets/img/Logo.png', 'assets/img'),
    ('assets', 'assets'),
    
    # Electrode map configuration files
    ('electrode_map', 'electrode_map'),
    
    # Theme files
    ('theme', 'theme'),
    
    # Data directory (for any required data files)
    ('data', 'data'),
    
    # Utils
    ('utils', 'utils'),
    
    # Windows package
    ('windows', 'windows'),
    
    # Label directory
    ('label', 'label'),
    
    # RHD loader
    ('load-rhd-notebook-python', 'load-rhd-notebook-python'),
]

a = Analysis(
    ['meeg.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
        'numpy',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        'scipy',
        'scipy.signal',
        'scipy.interpolate',
        'scipy.spatial',
        'cv2',
        'tqdm',
        'sklearn',
        'json',
        'hashlib',
    ],
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
    [],
    exclude_binaries=True,
    name='MEEG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for GUI app (no terminal window)
    disable_windowed_traceback=False,
    argv_emulation=True,  # Important for macOS
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/img/Logo.png',  # App icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MEEG',
)

# macOS specific: Create .app bundle
app = BUNDLE(
    coll,
    name='MEEG.app',
    icon='assets/img/Logo.png',
    bundle_identifier='com.koreauni.meeg',
    info_plist={
        'CFBundleName': 'MEEG',
        'CFBundleDisplayName': 'MEEG',
        'CFBundleGetInfoString': 'MEEG - EEG Analysis Tool',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
    },
)
