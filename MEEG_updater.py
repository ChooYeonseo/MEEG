"""
MEEG Updater - Standalone update script.

This script is bundled as MEEG_updater.exe and is launched by the main app
to apply updates. It:
1. Waits for the main MEEG.exe to close
2. Extracts the downloaded update
3. Replaces old files with new ones
4. Launches the updated MEEG.exe
5. Cleans up and exits

Usage:
    MEEG_updater.exe <zip_path> <app_dir> [--main-pid <pid>]
"""

import os
import sys
import time
import shutil
import zipfile
import subprocess
import argparse
from pathlib import Path


def wait_for_process_exit(pid, timeout=30):
    """Wait for a process to exit (Windows)."""
    if pid is None:
        return True
    
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        
        # Open the process
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            # Wait for process to exit
            WAIT_OBJECT_0 = 0x00000000
            result = kernel32.WaitForSingleObject(handle, timeout * 1000)
            kernel32.CloseHandle(handle)
            return result == WAIT_OBJECT_0
    except:
        pass
    
    # Fallback: just wait a bit
    time.sleep(3)
    return True


def extract_update(zip_path: Path, target_dir: Path, backup_dir: Path):
    """Extract update zip and replace files."""
    print(f"[Updater] Extracting {zip_path} to {target_dir}...")
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary extraction directory
    temp_extract = target_dir / "_update_temp"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir()
    
    try:
        # Extract the zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_extract)
        
        # Find the root directory in the extracted content
        # GitHub zips usually have a root folder like "MEEG-main/"
        extracted_items = list(temp_extract.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]
        else:
            source_dir = temp_extract
        
        # Files/folders to preserve (don't delete or overwrite)
        preserve = {'cache', 'MEEG_update', '_update_temp', 'MEEG_updater.exe'}
        
        # Copy new files, backing up old ones
        for item in source_dir.iterdir():
            dest_item = target_dir / item.name
            
            if item.name in preserve:
                print(f"[Updater] Preserving: {item.name}")
                continue
            
            # Backup existing item
            if dest_item.exists():
                backup_item = backup_dir / item.name
                print(f"[Updater] Backing up: {item.name}")
                if dest_item.is_dir():
                    if backup_item.exists():
                        shutil.rmtree(backup_item)
                    shutil.move(str(dest_item), str(backup_item))
                else:
                    shutil.move(str(dest_item), str(backup_item))
            
            # Copy new item
            print(f"[Updater] Installing: {item.name}")
            if item.is_dir():
                shutil.copytree(str(item), str(dest_item))
            else:
                shutil.copy2(str(item), str(dest_item))
        
        print("[Updater] Update applied successfully!")
        return True
        
    except Exception as e:
        print(f"[Updater] ERROR: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_extract.exists():
            try:
                shutil.rmtree(temp_extract)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='MEEG Updater')
    parser.add_argument('zip_path', help='Path to the downloaded update zip')
    parser.add_argument('app_dir', help='Path to the MEEG application directory')
    parser.add_argument('--main-pid', type=int, help='PID of the main MEEG process to wait for')
    
    args = parser.parse_args()
    
    zip_path = Path(args.zip_path)
    app_dir = Path(args.app_dir)
    backup_dir = app_dir / "_backup"
    
    print("=" * 50)
    print("MEEG Updater")
    print("=" * 50)
    print(f"Update file: {zip_path}")
    print(f"Application: {app_dir}")
    print()
    
    # Validate inputs
    if not zip_path.exists():
        print(f"[Updater] ERROR: Update file not found: {zip_path}")
        input("Press Enter to exit...")
        return 1
    
    if not app_dir.exists():
        print(f"[Updater] ERROR: Application directory not found: {app_dir}")
        input("Press Enter to exit...")
        return 1
    
    # Wait for main process to exit
    if args.main_pid:
        print(f"[Updater] Waiting for MEEG.exe (PID {args.main_pid}) to close...")
        if not wait_for_process_exit(args.main_pid):
            print("[Updater] Warning: Main process may still be running")
        else:
            print("[Updater] Main process closed.")
    else:
        # Just wait a moment for cleanup
        print("[Updater] Waiting for cleanup...")
        time.sleep(2)
    
    # Apply the update
    if extract_update(zip_path, app_dir, backup_dir):
        # Clean up the downloaded zip
        try:
            update_dir = app_dir / "MEEG_update"
            if update_dir.exists():
                shutil.rmtree(update_dir)
        except:
            pass
        
        # Launch the updated application
        meeg_exe = app_dir / "MEEG.exe"
        if meeg_exe.exists():
            print(f"[Updater] Launching updated MEEG...")
            subprocess.Popen([str(meeg_exe)], cwd=str(app_dir))
        else:
            print(f"[Updater] Warning: MEEG.exe not found at {meeg_exe}")
            print("[Updater] Please launch manually.")
            input("Press Enter to exit...")
        
        print("[Updater] Update complete!")
        return 0
    else:
        print("[Updater] Update failed! Restoring backup...")
        # Restore from backup if update failed
        for item in backup_dir.iterdir():
            dest_item = app_dir / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(str(item), str(dest_item))
            else:
                shutil.copy2(str(item), str(dest_item))
        
        input("Press Enter to exit...")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[Updater] Fatal error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
