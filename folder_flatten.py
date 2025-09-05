#!/usr/bin/env python3
import os, sys, shutil
from argparse import ArgumentParser

def main():
    ap = ArgumentParser(description="Restore .rhd files from numbered subfolders back to parent directory.")
    ap.add_argument("dir", help="Path to the main folder containing subfolders with .rhd files.")
    ap.add_argument("--keep-folders", action="store_true",
                    help="Keep the numbered subfolders (default: remove if empty).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without moving files.")
    args = ap.parse_args()

    base = os.path.abspath(args.dir)
    if not os.path.isdir(base):
        print(f"Not a directory: {base}", file=sys.stderr)
        sys.exit(1)

    moved = 0
    subfolders = [f for f in os.listdir(base)
                  if os.path.isdir(os.path.join(base, f))]

    for folder in subfolders:
        folder_path = os.path.join(base, folder)
        for name in os.listdir(folder_path):
            if name.lower().endswith(".rhd") and not name.startswith("._"):
                src = os.path.join(folder_path, name)
                dst = os.path.join(base, name)
                print(f"Move: {src} -> {dst}")
                if not args.dry_run:
                    if os.path.exists(dst):
                        print(f"WARNING: {dst} already exists. Skipping.", file=sys.stderr)
                    else:
                        shutil.move(src, dst)
                        moved += 1

        # remove folder if empty
        if not args.keep_folders and not args.dry_run:
            try:
                os.rmdir(folder_path)
                print(f"Removed empty folder: {folder_path}")
            except OSError:
                pass  # folder not empty

    print(f"\nDone. {'Would have moved' if args.dry_run else 'Moved'} {moved} files.")

if __name__ == "__main__":
    main()


'''
250828 daytime EEG A_WT+pilo B_KO+pilo_250828_091607
python3 folder_flatten.py "/Volumes/CHOO'S SSD/LINK/EEG A_WT+pilo B_KO+pilo/250828 overnight EEG A_WT+pilo B_KO+pilo_250828_191638" --dry-run

'''