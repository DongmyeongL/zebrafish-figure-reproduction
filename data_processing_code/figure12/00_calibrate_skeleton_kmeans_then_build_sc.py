#!/usr/bin/env python3
"""FCS calibration using skeleton-path KMeans endpoint classes.

This variant calibrates the affine endpoint transform with the same
source/target skeleton-path endpoint classes used by the primary r12
Figure 12 reconstruction. By default it runs calibration only; pass
``--build-sc`` to generate subject-wise compact SC files after calibration.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RECONSTRUCTION = HERE / "sc_reconstruction"
CALIBRATE = RECONSTRUCTION / "26_calibrate_subject12_transform_to_fcs.py"
BUILD = HERE / "01_generate_subject_full_affine_endpoint_edges.py"


def run(command: list[str]) -> None:
    print("\n[Figure 12 skeleton-KMeans SC]", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate endpoint transforms using skeleton-path KMeans endpoint classes."
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(12, 19)))
    parser.add_argument("--coarse", type=int, default=24)
    parser.add_argument("--refine", type=int, default=16)
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--squared-threshold", type=float, default=144.0)
    parser.add_argument(
        "--build-sc",
        action="store_true",
        help="After calibration, build skeleton-nearest compact SC files.",
    )
    args = parser.parse_args()
    subjects = [str(subject) for subject in args.subjects]

    run([
        sys.executable,
        str(CALIBRATE),
        "--full-affine",
        "--endpoint-label-rule",
        "skeleton_path_kmeans",
        "--subjects",
        *subjects,
        "--coarse",
        str(args.coarse),
        "--refine",
        str(args.refine),
        "--seed",
        str(args.seed),
    ])
    if args.build_sc:
        run([
            sys.executable,
            str(BUILD),
            "--subjects",
            *subjects,
            "--sc-source",
            "fcs_calibrated_skeleton_kmeans_nearest_r12",
            "--squared-threshold",
            str(args.squared_threshold),
        ])


if __name__ == "__main__":
    main()
