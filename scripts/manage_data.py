#!/usr/bin/env python3
"""Audit raw-data availability from the public dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "datasets.csv"


def matches(raw_root: Path, pattern: str) -> list[Path]:
    path = raw_root / pattern
    if any(char in pattern for char in "*?["):
        return sorted(raw_root.glob(pattern))
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return [path] if path.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "raw_data")
    parser.add_argument(
        "--target",
        action="append",
        help="Only report rows whose required_for field contains this target.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error when a selected external or bundled input is missing.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST).fillna("")
    if args.target:
        selected = set(args.target)
        manifest = manifest[
            manifest["required_for"].map(
                lambda value: bool(selected & set(str(value).split(";")))
            )
        ]

    missing = []
    for row in manifest.itertuples(index=False):
        found = matches(args.raw_root.resolve(), row.relative_path)
        status = "OK" if found else "MISSING"
        print(
            f"{status:7s} {row.dataset_id:28s} {len(found):3d} file(s)  "
            f"{row.relative_path}"
        )
        if not found:
            missing.append(row)

    if missing:
        print("\nMissing inputs:")
        for row in missing:
            location = row.doi_or_url or "no direct download URL recorded"
            print(f"- {row.dataset_id}: {location}")
    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
