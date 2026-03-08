#!/usr/bin/env python3
"""
Organize US state map TIF files into a clean dataset structure.

Copies TIF files from Test_data/States_map/<State>/*.tif
to datasets/us_mapdata/<State>/*.tif

Usage:
    python -m tools.organize_us_mapdata
    python -m tools.organize_us_mapdata --source Test_data/States_map --dest datasets/us_mapdata
    python -m tools.organize_us_mapdata --states Alabama California Florida
"""

import argparse
import os
import shutil
from pathlib import Path


def organize_mapdata(
    source_dir: str = "Test_data/States_map",
    dest_dir: str = "datasets/us_mapdata",
    state_filter: list = None,
    dry_run: bool = False,
):
    """
    Copy TIF files from source to destination with clean structure.

    Args:
        source_dir: Source directory containing state folders
        dest_dir: Destination directory for organized data
        state_filter: Optional list of state names to include
        dry_run: If True, only print what would be done
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    # Get all state directories
    state_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    # Filter if specified
    if state_filter:
        state_dirs = [d for d in state_dirs if d.name in state_filter]

    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"States to process: {len(state_dirs)}")
    print()

    total_files = 0
    states_with_tifs = 0

    for state_dir in sorted(state_dirs):
        state_name = state_dir.name

        # Find TIF files
        tif_files = list(state_dir.glob("*.tif"))
        # Exclude auxiliary files
        tif_files = [f for f in tif_files if not str(f).endswith('.aux.xml')]

        if not tif_files:
            continue

        states_with_tifs += 1
        state_dest = dest_path / state_name

        print(f"{state_name}: {len(tif_files)} TIF files")

        if not dry_run:
            state_dest.mkdir(parents=True, exist_ok=True)

        for tif_file in tif_files:
            dest_file = state_dest / tif_file.name

            if dry_run:
                print(f"  Would copy: {tif_file.name}")
            else:
                if not dest_file.exists():
                    shutil.copy2(tif_file, dest_file)
                    print(f"  Copied: {tif_file.name}")
                else:
                    print(f"  Exists: {tif_file.name}")

            total_files += 1

    print()
    print(f"Summary:")
    print(f"  States with TIF files: {states_with_tifs}")
    print(f"  Total TIF files: {total_files}")
    print(f"  Destination: {dest_path}")

    if dry_run:
        print("\n(Dry run - no files were copied)")


def main():
    parser = argparse.ArgumentParser(
        description="Organize US state map TIF files into clean dataset structure"
    )
    parser.add_argument(
        '--source', '-s',
        default='Test_data/States_map',
        help='Source directory containing state folders'
    )
    parser.add_argument(
        '--dest', '-d',
        default='datasets/us_mapdata',
        help='Destination directory for organized data'
    )
    parser.add_argument(
        '--states',
        nargs='+',
        help='Filter to specific states (e.g., --states Alabama California)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without copying files'
    )

    args = parser.parse_args()

    organize_mapdata(
        source_dir=args.source,
        dest_dir=args.dest,
        state_filter=args.states,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
