# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MEEG Application.

This creates a Windows .exe (or macOS .app bundle on macOS).
Run with: pyinstaller MEEG.spec
"""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH)

# Determine if running on Windows
is_windows = sys.platform == 'win32'

block_cipher = None

# Data files to include
added_files = [
    # Assets (including icon)
    ('assets', 'assets'),
    
    # Electrode map configuration files
    ('electrode_map', 'electrode_map'),
    
    # Theme files
    ('theme', 'theme'),
    
    # Utils
    ('utils', 'utils'),
    
    # Windows package (PyQt windows)
    ('windows', 'windows'),
    
    # Label directory
    ('label', 'label'),
    
    # RHD loader
    ('load-rhd-notebook-python', 'load-rhd-notebook-python'),
    
    # Images (Mouse head, etc.)
    ('img', 'img'),
    
    # Config file (for version info)
    ('config.py', '.'),
]

# FFmpeg binaries - bundle if available in project root
# To bundle ffmpeg: download from https://www.gyan.dev/ffmpeg/builds/
# and extract ffmpeg.exe + ffprobe.exe to a 'ffmpeg' folder in project root
ffmpeg_binaries = []
ffmpeg_dir = project_root / 'ffmpeg'
if ffmpeg_dir.exists():
    ffmpeg_exe = ffmpeg_dir / 'ffmpeg.exe'
    ffprobe_exe = ffmpeg_dir / 'ffprobe.exe'
    if ffmpeg_exe.exists():
        ffmpeg_binaries.append((str(ffmpeg_exe), '.'))
        print(f"Bundling ffmpeg: {ffmpeg_exe}")
    if ffprobe_exe.exists():
        ffmpeg_binaries.append((str(ffprobe_exe), '.'))
        print(f"Bundling ffprobe: {ffprobe_exe}")
else:
    print("Note: ffmpeg folder not found. Video segment extraction will use fallback mode.")
    print("      To bundle ffmpeg, create 'ffmpeg/' folder with ffmpeg.exe and ffprobe.exe")

from PyInstaller.utils.hooks import collect_dynamic_libs

# Get OpenCV binaries (specifically ffmpeg dll)
cv2_binaries = collect_dynamic_libs('cv2')

a = Analysis(
    ['meeg.py'],
    pathex=[str(project_root)],
    binaries=ffmpeg_binaries + cv2_binaries,
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
        'scipy.signal.windows',
        'scipy.interpolate',
        'scipy.spatial',
        'scipy.stats',
        'cv2',
        'tqdm',
        'sklearn',
        'json',
        'hashlib',
        'requests',  # For auto-update
        'pickle',
        'openpyxl',
        'importrhdutilities',  # RHD file loading module
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
    console=True,  # Set to True for debugging (shows terminal window with errors)
    disable_windowed_traceback=False,
    argv_emulation=False if is_windows else True,  # Only for macOS
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/img/Logo.ico' if is_windows else 'assets/img/Logo.png',  # Windows uses ICO
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
