#!/usr/bin/env python3
"""Validate public invertebrate derived tables without private references."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    derived = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", ROOT / "derived_data"))
    for target in ("celegans", "drosophila"):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_derived.py"),
                target,
                "--derived-root",
                str(derived),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
