#!/usr/bin/env python3
"""Record immutable Figure 9 inputs without hashing multi-GB symlink targets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from common import RAW_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_dir() or path == RAW_DIR / "source_manifest.csv":
            continue
        relative = path.relative_to(RAW_DIR)
        if path.is_symlink():
            target = path.resolve()
            rows.append(
                {
                    "path": str(relative),
                    "kind": "symlink_to_large_original",
                    "size_bytes": target.stat().st_size,
                    "sha256": "not_computed_for_large_symlink_target",
                    "source_target": str(target),
                }
            )
        else:
            rows.append(
                {
                    "path": str(relative),
                    "kind": "frozen_pack_input",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "source_target": "",
                }
            )
    output = RAW_DIR / "source_manifest.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} entries to {output}")


if __name__ == "__main__":
    main()
