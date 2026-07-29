#!/usr/bin/env python3
"""Rebuild all manuscript figures included in this public release."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "figure_code"

FIGURE_COMMANDS = [
    ("figure9_final_v2.py",),
    ("figure12_final_v3.py",),
    (
        "figure_fcv_fcs_fu_sc_corr_forest.py",
        "--sc-source", "fcs_calibrated_skeleton_kmeans_nearest_r12",
        "--panel-ab-distribution", "two_way_subsampling",
        "--panel-ab-p-mode", "raw",
    ),
    ("figure_stimulus_delta_fcv_acd_combined_skeleton_kmeans_nearest_r12_two_way_subsampling.py",),
    ("figure13_final_v2.py",),
    ("figure_invertebrate_oo_fcv_relationships_with_137_subunits.py",),
    ("figure_supply_1.py",),
    ("figure_supply_2_proc.py",),
    ("figure_supply_10_proc.py",),
    ("figure_supplement_te_structural_controls.py",),
    ("figure_supply_13.py",),
    ("figure_stimulus_condition_region_profiles.py",),
    ("figure_supply_0.py",),
    ("figure_supply_5.py",),
    ("figure_supply_15.py",),
    ("figure_invertebrate_multiscale_sc_fc_matrices.py",),
    ("figure_invertebrate_anatomical_group_summary.py",),
]


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "statistics").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE), str(ROOT), env.get("PYTHONPATH", "")]
    )

    for index, command in enumerate(FIGURE_COMMANDS, start=1):
        script_name, *arguments = command
        print(f"[{index:02d}/{len(FIGURE_COMMANDS)}] {' '.join(command)}", flush=True)
        subprocess.run(
            [sys.executable, str(CODE / script_name), *arguments],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
