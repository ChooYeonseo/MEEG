#########################################################################
# Script to group .rhd files in a directory into numbered subfolders with a common prefix
# This action is required if the file size exceeds about 10GB. 
# If not, the program will crash due to allocation failure. - This may depend on the system.
#########################################################################


#!/usr/bin/env python3
import os, sys, shutil, re
from argparse import ArgumentParser

def natsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.findall(r'\d+|\D+', s)]

def longest_common_prefix(strings):
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i].rstrip("_- ")
    return s1

def main():
    ap = ArgumentParser(description="Group .rhd files into numbered folders with common prefix.")
    ap.add_argument("dir", help="Path to the folder containing .rhd files (not recursive).")
    ap.add_argument("--per", type=int, default=180, help="Files per folder (default: 180).")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without moving files.")
    args = ap.parse_args()

    base = os.path.abspath(args.dir)
    if not os.path.isdir(base):
        print(f"Not a directory: {base}", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if (os.path.isfile(full)
            and name.lower().endswith(".rhd")
            and not name.startswith("._")
            and not name.startswith(".")):
            candidates.append(name)

    if not candidates:
        print("No .rhd files found.")
        return

    candidates.sort(key=natsort_key)

    # derive common prefix
    prefix = longest_common_prefix(candidates)
    if not prefix:
        prefix = "chunk"

    total = len(candidates)
    per = max(1, args.per)
    num_folders = (total + per - 1) // per

    print(f"Found {total} .rhd files. Common prefix: '{prefix}'. Making {num_folders} folders.\n")

    idx = 0
    folder_num = 1
    while idx < total:
        chunk = candidates[idx:idx+per]
        folder_name = f"{prefix}_{folder_num}"
        folder_path = os.path.join(base, folder_name)
        if not os.path.exists(folder_path):
            print(f"Create folder: {folder_path}")
            if not args.dry_run:
                os.makedirs(folder_path, exist_ok=True)
        else:
            if not os.path.isdir(folder_path):
                print(f"ERROR: {folder_path} exists and is not a directory.", file=sys.stderr)
                sys.exit(2)

        for name in chunk:
            src = os.path.join(base, name)
            dst = os.path.join(folder_path, name)
            print(f"Move: {src} -> {dst}")
            if not args.dry_run:
                shutil.move(src, dst)

        idx += per
        folder_num += 1

    print("\nDone.")
    if args.dry_run:
        print("This was a dry run. Re-run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()

'''
python3 folder_subset.py "/Volumes/CHOO'S SSD/LINK/EEG A_WT+pilo B_KO+pilo/250828 overnight EEG A_WT+pilo B_KO+pilo_250828_191638"

'''