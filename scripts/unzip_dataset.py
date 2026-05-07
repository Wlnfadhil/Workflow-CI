#!/usr/bin/env python3
"""Safely extract a zip archive.

Usage:
    python scripts/unzip_dataset.py 
    python scripts/unzip_dataset.py path/to/archive.zip output_dir

Defaults to: data/student-performance-analytics-dataset.zip
"""
from pathlib import Path
import zipfile
import argparse
import sys


def safe_extract(zip_file: zipfile.ZipFile, dest: Path):
    dest = dest.resolve()
    for member in zip_file.namelist():
        member_path = (dest / member).resolve()
        if not str(member_path).startswith(str(dest)):
            raise Exception(f"Unsafe path detected in archive: {member}")
        # create parent directories as needed
        member_parent = member_path.parent
        member_parent.mkdir(parents=True, exist_ok=True)
        # if member is a directory, ensure it exists
        if member.endswith('/'):
            member_parent.mkdir(parents=True, exist_ok=True)
            continue
        with zip_file.open(member) as source, open(member_path, "wb") as target:
            target.write(source.read())


def extract_zip(zip_path: Path, out_dir: Path) -> list:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        safe_extract(zf, out_dir)
        return zf.namelist()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract a zip file safely.")
    parser.add_argument('zip', nargs='?', default='data/student-performance-analytics-dataset.zip',
                        help='Path to the zip file (default: data/student-performance-analytics-dataset.zip)')
    parser.add_argument('-o', '--out', default=None, help='Output directory (default: same folder, named after zip)')
    args = parser.parse_args(argv)

    zip_path = Path(args.zip)
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = zip_path.parent / zip_path.stem

    try:
        members = extract_zip(zip_path, out_dir)
    except zipfile.BadZipFile:
        print(f"ERROR: Bad zip file: {zip_path}")
        return 2
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 3
    except Exception as e:
        print(f"ERROR: {e}")
        return 4

    print(f"Extracted {len(members)} entries to: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
