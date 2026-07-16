#!/usr/bin/env python3
"""Record Figure 12 compact inputs and their original source paths."""

from __future__ import annotations

import hashlib

import pandas as pd

from common import CONFIG, RAW_DIR, compact_sc_path, original_sc_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for subject in CONFIG["subjects"]:
        compact = compact_sc_path(subject)
        original = original_sc_path(subject)
        rows.append(
            {
                "Subject": subject,
                "compact_path": str(compact.relative_to(RAW_DIR)),
                "compact_size_bytes": compact.stat().st_size,
                "compact_sha256": sha256(compact),
                "original_source": str(original),
                "original_size_bytes": original.stat().st_size,
            }
        )
    output = RAW_DIR / "source_manifest.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
