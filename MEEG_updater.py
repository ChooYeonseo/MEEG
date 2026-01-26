import sys
import time
import shutil
import zipfile
import subprocess
import argparse
from pathlib import Path


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def wait_for_process_exit(pid, timeout=30):
    if pid is None:
        return True

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.WaitForSingleObject(handle, timeout * 1000)
            kernel32.CloseHandle(handle)
    except Exception:
        pass

    time.sleep(2)
    return True


def replace_with_move(src: Path, dst: Path):
    """
    HARD replacement:
    - delete dst
    - MOVE src → dst
    """
    if not src.exists():
        return False

    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    shutil.move(str(src), str(dst))
    return True


# ------------------------------------------------------------
# Update logic
# ------------------------------------------------------------

def extract_update(zip_path: Path, app_dir: Path) -> bool:
    backup_dir = app_dir / "_backup"
    temp_dir = app_dir / "_update_temp"

    try:
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        backup_dir.mkdir()
        temp_dir.mkdir()

        # Extract update
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        extracted = list(temp_dir.iterdir())
        source_dir = extracted[0] if len(extracted) == 1 and extracted[0].is_dir() else temp_dir

        preserve = {
            "_backup",
            "_update_temp",
            "meeg_updater.exe",
            "meeg_update",
        }

        # ----------------------------
        # Phase 1: Backup old version
        # ----------------------------
        for item in source_dir.iterdir():
            dst = app_dir / item.name
            if item.name.lower() in preserve:
                continue
            if dst.exists():
                shutil.move(str(dst), str(backup_dir / item.name))

        # ----------------------------
        # Phase 2: Install new version
        # ----------------------------
        for item in source_dir.iterdir():
            dst = app_dir / item.name
            if item.name.lower() in preserve:
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        # ----------------------------
        # Phase 3: Restore user data (MOVE)
        # ----------------------------
        replace_with_move(backup_dir / "cache", app_dir / "cache")
        replace_with_move(backup_dir / "electrode_map", app_dir / "electrode_map")
        replace_with_move(backup_dir / "config.py", app_dir / "config.py")

        # ----------------------------
        # Phase 4: RESTORE _internal DATA (CRITICAL FIX)
        # ----------------------------
        internal_backup = backup_dir / "_internal"
        internal_target = app_dir / "_internal"

        data_moved = False

        if internal_backup.exists() and internal_target.exists():
            if replace_with_move(internal_backup / "cache",
                                 internal_target / "cache"):
                data_moved = True

            if replace_with_move(internal_backup / "electrode_map",
                                 internal_target / "electrode_map"):
                data_moved = True

        if data_moved:
            print("[Updater] data moved")

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"[Updater] ERROR: {e}")
        return False


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MEEG Updater")
    parser.add_argument("zip_path")
    parser.add_argument("app_dir")
    parser.add_argument("--main-pid", type=int)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    app_dir = Path(args.app_dir)

    if not zip_path.exists() or not app_dir.exists():
        print("[Updater] Invalid path")
        input("Press Enter to exit...")
        return 1

    if args.main_pid:
        wait_for_process_exit(args.main_pid)

    if extract_update(zip_path, app_dir):
        exe = app_dir / "MEEG.exe"
        if exe.exists():
            subprocess.Popen([str(exe)], cwd=str(app_dir))
        print("[Updater] Update complete")
        return 0

    print("[Updater] Update failed")
    input("Press Enter to exit...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
