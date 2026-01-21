"""
MEEG Release Build Script

This script automates the build process for creating a distributable MEEG release:
1. Builds MEEG.exe and MEEG_updater.exe
2. Copies updater to the distribution folder
3. Creates a release zip ready for GitHub upload

Usage:
    python build_release.py
    
The output zip will be created at: releases/MEEG_v{VERSION}_win64.zip
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Get version from config
sys.path.insert(0, str(Path(__file__).parent))
from config import APP_VERSION


def run_command(cmd, description):
    """Run a command and display progress."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True


def main():
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("\n" + "="*60)
    print(f"  MEEG Release Builder v{APP_VERSION}")
    print("="*60)
    print(f"  Project: {project_root}")
    print(f"  Version: {APP_VERSION}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Build MEEG_updater.exe
    if not run_command(
        "python -m PyInstaller MEEG_updater.spec --noconfirm",
        "Building MEEG_updater.exe"
    ):
        return 1
    
    # Step 2: Build MEEG.exe
    if not run_command(
        "python -m PyInstaller MEEG.spec --noconfirm",
        "Building MEEG.exe"
    ):
        return 1
    
    # Step 3: Copy updater to distribution folder
    dist_dir = project_root / "dist" / "MEEG"
    updater_src = project_root / "dist" / "MEEG_updater.exe"
    updater_dst = dist_dir / "MEEG_updater.exe"
    
    print(f"\n{'='*60}")
    print(f"  Copying MEEG_updater.exe to distribution")
    print(f"{'='*60}")
    
    if updater_src.exists():
        shutil.copy2(updater_src, updater_dst)
        print(f"  Copied: {updater_dst}")
    else:
        print(f"  ERROR: Updater not found at {updater_src}")
        return 1
    
    # Step 4: Create releases directory
    releases_dir = project_root / "releases"
    releases_dir.mkdir(exist_ok=True)
    
    # Step 5: Create release zip
    zip_name = f"MEEG_v{APP_VERSION}_win64"
    zip_path = releases_dir / f"{zip_name}.zip"
    
    print(f"\n{'='*60}")
    print(f"  Creating release zip")
    print(f"{'='*60}")
    
    # Remove old zip if exists
    if zip_path.exists():
        zip_path.unlink()
    
    # Create zip
    shutil.make_archive(
        str(releases_dir / zip_name),
        'zip',
        root_dir=str(project_root / "dist"),
        base_dir="MEEG"
    )
    
    # Get zip size
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE!")
    print(f"{'='*60}")
    print(f"  Release zip: {zip_path}")
    print(f"  Size: {zip_size:.1f} MB")
    print(f"\n  Next steps:")
    print(f"  1. Go to: https://github.com/ChooYeonseo/MEEG/releases/new")
    print(f"  2. Create tag: v{APP_VERSION}")
    print(f"  3. Upload: {zip_path.name}")
    print(f"  4. Publish release")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
