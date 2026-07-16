#!/usr/bin/env python3
"""Rebuild all manuscript figures included in this public release."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "figure_code"

FIGURE_SCRIPTS = [
    "figure9_final_v2.py",
    "figure12_final_v2.py",
    "figure_fcv_fcs_sc_corr_forest.py",
    "figure_stimulus_delta_fcv_acd_combined.py",
    "figure13_final_v2.py",
    "figure_invertebrate_oo_fcv_relationships.py",
    "figure_supply_1.py",
    "figure_supply_2_proc.py",
    "figure_supply_5.py",
    "figure_supply_10_proc.py",
    "figure_supply_13.py",
    "figure_supply_15.py",
    "figure_supplement_te_structural_controls.py",
    "figure_stimulus_condition_region_profiles.py",
    "figure_invertebrate_anatomical_group_summary.py",
]


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "statistics").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE), str(ROOT), env.get("PYTHONPATH", "")]
    )

    for index, script_name in enumerate(FIGURE_SCRIPTS, start=1):
        print(f"[{index:02d}/{len(FIGURE_SCRIPTS)}] {script_name}", flush=True)
        subprocess.run(
            [sys.executable, str(CODE / script_name)],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
