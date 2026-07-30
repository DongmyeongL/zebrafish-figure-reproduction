#!/usr/bin/env python3
"""Generate the reconstruction-dependent SI robustness inputs from source data.

This driver runs the primary skeleton-r12 SC reconstruction when requested,
then regenerates the anatomical-unit-size/radius grid, morphology subsamples,
and weighted strength-preserving topology nulls. It finally exports the compact
tables consumed by the public SI statistics scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


RELEASE = Path(__file__).resolve().parents[2]
FIGURE12 = RELEASE / "data_processing_code" / "figure12"
VALIDATION = FIGURE12 / "validation"
REFERENCE = RELEASE / "derived_data"
SUBJECTS = tuple(range(12, 19))
SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"


def run(script: Path, env: dict[str, str], *arguments: str) -> None:
    command = [sys.executable, str(script), *arguments]
    print("\n[robustness generation] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=script.parent, env=env, check=True)


def require(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(f"Missing {label}:\n{preview}")


def seed_reference_tables(derived_root: Path) -> None:
    for relative in (
        Path("figure9/figure9_region_summary.csv"),
        Path("figure9/figure9_recording_region_measures.csv"),
        Path("common/legacy_stimulus_forest_42_regions_no_rOB.csv"),
    ):
        source = REFERENCE / relative
        destination = derived_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def export_compact_inputs(derived_root: Path, destination: Path) -> None:
    base = derived_root / "figure12" / "validation" / "r12_primary"
    files = {
        base / "unit_target_endpoint_radius" / "fcv_structure_correlations.csv":
            "reconstruction_grid_correlations.csv",
        base / "morphology_subsample_size" / "morphology_subsample_iterations.csv":
            "morphology_subsampling_iterations.csv",
        base / "morphology_subsample_size" / "analysis_settings.csv":
            "morphology_subsampling_settings.csv",
        base / "morphology_support_topology_null" / "topology_preserving_null_fcv_correlations.csv":
            "topology_null_correlations.csv",
        base / "morphology_support_topology_null" / "support_filter_fcv_correlations.csv":
            "topology_null_observed_correlations.csv",
    }
    require(list(files), "generated robustness tables")
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in files.items():
        shutil.copy2(source, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--n-null", type=int, default=250)
    parser.add_argument("--subsample-iterations", type=int, default=200)
    parser.add_argument("--seed-null", type=int, default=20260722)
    parser.add_argument("--seed-subsampling", type=int, default=20260727)
    parser.add_argument(
        "--skip-sc-reconstruction", action="store_true",
        help="Use existing calibration and endpoint-edge files in derived-root.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    derived_root = args.derived_root.resolve()
    output = args.output.resolve()
    subjects = [str(subject) for subject in args.subjects]
    anatomy = raw_root / "figure12" / "anatomy"
    subject_mats = raw_root / "figure12" / "original_subject_mat"
    prepared_subjects = raw_root / "figure9" / "original_subjects"
    require(
        [
            anatomy / "neuronEndpoints_data.mat",
            anatomy / "somaCoordinates_data.mat",
            anatomy / "signle_neuron_poistion_data.mat",
            *[subject_mats / f"subject_{subject}_data.mat" for subject in args.subjects],
            *[
                prepared_subjects / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
                for subject in args.subjects
            ],
        ],
        "Figure 12 source data",
    )
    seed_reference_tables(derived_root)

    env = os.environ.copy()
    env.update({
        "ZF_RAW_DATA_ROOT": str(raw_root),
        "ZF_DERIVED_DATA_ROOT": str(derived_root),
        "ZF_ANALYSIS_INPUT_ROOT": str(derived_root),
        "ZF_ANATOMY_ROOT": str(anatomy),
        "ZF_ORIGINAL_MAT_ROOT": str(subject_mats),
        "ZF_PREPARED_SUBJECT_ROOT": str(prepared_subjects),
        "ZF_COMPACT_SC_ROOT": str(derived_root / "figure12" / "compact_sc"),
    })

    if not args.skip_sc_reconstruction:
        run(
            FIGURE12 / "00_calibrate_skeleton_kmeans_then_build_sc.py", env,
            "--subjects", *subjects, "--squared-threshold", "144", "--build-sc",
        )

    edge_dir = derived_root / "figure12" / "sc_reconstruction" / SC_SOURCE
    require(
        [edge_dir / f"subject_{subject}_full_affine_endpoint_edges.npz" for subject in args.subjects],
        "primary skeleton-r12 endpoint-edge files",
    )

    run(
        VALIDATION / "11_r12_unit_target_endpoint_radius_sensitivity.py", env,
        "--subjects", *subjects,
    )
    run(
        VALIDATION / "12_r12_morphology_support_topology_null.py", env,
        "--subjects", *subjects, "--n-null", str(args.n_null),
        "--seed", str(args.seed_null),
    )
    subsampling_arguments = [
        "--subjects", *subjects, "--iterations", str(args.subsample_iterations),
        "--seed", str(args.seed_subsampling),
    ]
    if set(args.subjects) != set(SUBJECTS):
        subsampling_arguments.append("--allow-incomplete-primary-regions")
    run(
        VALIDATION / "15_r12_morphology_subsample_size_sensitivity.py", env,
        *subsampling_arguments,
    )
    export_compact_inputs(derived_root, output)
    print(f"\nExported complete reconstruction-dependent inputs to {output}")


if __name__ == "__main__":
    main()
