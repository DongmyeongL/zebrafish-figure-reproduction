#!/usr/bin/env python3
"""Run public raw-to-derived pipelines and optionally rebuild figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROCESSING = ROOT / "data_processing_code"
SUBJECTS = [12, 13, 14, 15, 16, 17, 18]


def run(path: Path, env: dict[str, str], *extra: str) -> None:
    print(f"\n[run] {path.relative_to(ROOT)} {' '.join(extra)}", flush=True)
    subprocess.run(
        [sys.executable, str(path), *extra],
        cwd=path.parent,
        env=env,
        check=True,
    )


def require_files(paths: list[Path], description: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:8])
        raise FileNotFoundError(f"Missing {description}:\n{preview}")


def seed_static_inputs(derived_root: Path) -> None:
    source = ROOT / "derived_data" / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"
    destination = derived_root / "common" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def run_figure9(raw_root: Path, derived_root: Path, env: dict[str, str], from_primary: bool) -> None:
    if from_primary:
        required = [
            raw_root / "figure9" / "original_subjects" / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
            for subject in SUBJECTS
        ]
    else:
        required = [
            raw_root / "figure9" / "functional_unit_traces" / f"zebrafish_subject_{subject}_raw_cluster_traces.pkl"
            for subject in SUBJECTS
        ]
    require_files(required, "zebrafish functional-unit inputs")
    require_files([raw_root / "figure9" / "region_labels.csv"], "region label table")

    directory = PROCESSING / "figure9"
    run(directory / "01_extract_functional_units.py", env, *(["--from-original"] if from_primary else []))
    run(directory / "02_compute_fc_measures.py", env)
    run(directory / "03_prepare_te_measures.py", env)
    run(directory / "04_build_figure9_table.py", env)
    run(ROOT / "scripts" / "validate_derived.py", env, "figure9", "--derived-root", str(derived_root))


def run_figure12(raw_root: Path, derived_root: Path, env: dict[str, str]) -> None:
    source = "fcs_calibrated_skeleton_kmeans_nearest_r12"
    compact_dir = raw_root / "figure12" / f"{source}_sc"
    compact = [compact_dir / f"subject_{subject}_compact_sc.npz" for subject in SUBJECTS]
    subject_mats = [
        raw_root / "figure12" / "original_subject_mat" / f"subject_{subject}_data.mat"
        for subject in SUBJECTS
    ]
    directory = PROCESSING / "figure12"
    if not all(path.is_file() for path in compact):
        require_files(subject_mats, "subject soma-coordinate MAT files")
        anatomy = raw_root / "figure12" / "anatomy"
        require_files(
            [
                anatomy / "neuronEndpoints_data.mat",
                anatomy / "somaCoordinates_data.mat",
                anatomy / "signle_neuron_poistion_data.mat",
            ],
            "morphology and endpoint MAT files",
        )
        env["ZF_ORIGINAL_MAT_ROOT"] = str(raw_root / "figure12" / "original_subject_mat")
        env["ZF_ANATOMY_ROOT"] = str(anatomy)
        run(directory / "00_calibrate_skeleton_kmeans_then_build_sc.py", env, "--build-sc")
    require_files(compact, "seven skeleton-KMeans nearest-endpoint r12 compact SC files")
    run(directory / "02_compute_cell_dca.py", env, "--sc-source", source)
    run(directory / "03_compute_region_sc_measures.py", env, "--sc-source", source)
    run(directory / "04_build_figure12_table.py", env, "--sc-source", source)
    run(ROOT / "scripts" / "validate_derived.py", env, "figure12", "--derived-root", str(derived_root))


def run_stimulus(raw_root: Path, derived_root: Path, env: dict[str, str]) -> None:
    primary = [
        raw_root / "figure9" / "original_subjects" / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
        for subject in SUBJECTS
    ]
    require_files(primary, "primary zebrafish subject bundles for stimulus analysis")
    directory = PROCESSING / "figure_stimulus"
    run(directory / "01_extract_stimulus_region_traces.py", env)
    run(directory / "02_compute_stimulus_fc_measures.py", env)
    run(ROOT / "scripts" / "validate_derived.py", env, "stimulus", "--derived-root", str(derived_root))


def run_invertebrate(raw_root: Path, derived_root: Path, env: dict[str, str], species: str) -> None:
    directory = PROCESSING / "invertebrates"
    if species == "celegans":
        require_files(
            [
                raw_root / "invertebrates" / "celegans" / "herm_full_edgelist.csv",
                raw_root / "invertebrates" / "celegans" / "celegans_cell_classes.csv",
                raw_root / "invertebrates" / "celegans" / "celegans_fine_class_annotations.csv",
            ],
            "C. elegans structural and annotation inputs",
        )
        if not list((raw_root / "invertebrates" / "celegans" / "recordings").glob("*_raw_traces.pkl")):
            raise FileNotFoundError("No C. elegans recording exports were found")
        run(directory / "01_compute_celegans_metrics.py", env)
    else:
        fly = raw_root / "invertebrates" / "drosophila"
        require_files(
            [
                fly / "proofread_connections_783.feather",
                fly / "proofread_root_ids_783.npy",
                fly / "per_neuron_neuropil_count_pre_783.feather",
                fly / "per_neuron_neuropil_count_post_783.feather",
                fly / "ito_region_order.csv",
            ],
            "Drosophila structural inputs",
        )
        if not list((fly / "recordings").glob("*_raw_traces.pkl")):
            raise FileNotFoundError("No Drosophila recording exports were found")
        run(directory / "02_compute_drosophila_metrics.py", env)
    run(ROOT / "scripts" / "validate_derived.py", env, species, "--derived-root", str(derived_root))


def run_layer(derived_root: Path, env: dict[str, str], full_model: bool) -> None:
    if not full_model:
        raise RuntimeError(
            "The layer raw-to-derived simulation is computationally intensive. "
            "Re-run with --full-model, or use the bundled frozen summaries."
        )
    directory = PROCESSING / "figure13"
    run(directory / "01_run_layer_linear_model.py", env)
    run(directory / "layer_fcv_dense_model.py", env)
    run(directory / "02_compute_layer_fcv_summary.py", env)
    run(PROCESSING / "figure_supply_15" / "compute_dense_energy_width.py", env)
    run(ROOT / "scripts" / "validate_derived.py", env, "layer", "--derived-root", str(derived_root))


def run_supply13(derived_root: Path, env: dict[str, str], full_controls: bool) -> None:
    if not full_controls:
        raise RuntimeError(
            "Supply 13 uses 200 whole-network surrogates per recording. "
            "Re-run with --full-controls, or use the bundled frozen summaries."
        )
    run(PROCESSING / "figure_supply_13" / "compute_te_surrogate_validation.py", env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["figure9", "figure12", "stimulus", "celegans", "drosophila", "layer", "supply13", "all"],
        default="figure9",
    )
    parser.add_argument("--stage", choices=["derived", "figures", "all"], default="derived")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "raw_data")
    parser.add_argument("--derived-root", type=Path, default=ROOT / "derived_data")
    parser.add_argument("--from-primary", action="store_true")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--full-controls", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    derived_root = args.derived_root.expanduser().resolve()
    derived_root.mkdir(parents=True, exist_ok=True)
    seed_static_inputs(derived_root)

    env = os.environ.copy()
    env["ZF_RAW_DATA_ROOT"] = str(raw_root)
    env["ZF_DERIVED_DATA_ROOT"] = str(derived_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "figure_code"), env.get("PYTHONPATH", "")]
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    targets = (
        ["figure9", "figure12", "stimulus", "celegans", "drosophila", "layer", "supply13"]
        if args.target == "all"
        else [args.target]
    )
    if args.stage in {"derived", "all"}:
        for target in targets:
            try:
                if target == "figure9":
                    run_figure9(raw_root, derived_root, env, args.from_primary)
                elif target == "figure12":
                    run_figure12(raw_root, derived_root, env)
                elif target == "stimulus":
                    run_stimulus(raw_root, derived_root, env)
                elif target in {"celegans", "drosophila"}:
                    run_invertebrate(raw_root, derived_root, env, target)
                elif target == "layer":
                    run_layer(derived_root, env, args.full_model)
                elif target == "supply13":
                    run_supply13(derived_root, env, args.full_controls)
            except (FileNotFoundError, RuntimeError) as error:
                if not args.skip_missing:
                    raise
                print(f"\n[skip] {target}: {error}", file=sys.stderr)

    if args.stage in {"figures", "all"}:
        if derived_root != (ROOT / "derived_data").resolve():
            raise RuntimeError(
                "Figure scripts read the repository derived_data directory. "
                "Use the default --derived-root for --stage figures/all."
            )
        run(ROOT / "run_all_figures.py", env)


if __name__ == "__main__":
    main()
