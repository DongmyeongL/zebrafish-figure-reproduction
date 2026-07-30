#!/usr/bin/env python3
"""Morphology-subset size sensitivity for the canonical skeleton-r12 SC.

Each reconstructed morphology contributes support to each directed
anatomical-unit relation that it can generate. For subsampling, the canonical
cell-edge weight of a relation is divided equally among its supporting
morphologies. A subset receives the corresponding fraction of that canonical
weight, so retaining all morphologies exactly recovers the canonical SC.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
FIG12 = HERE.parent
PACK = FIG12.parents[1]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data"))
REFERENCE_ROOT = Path(os.environ.get("ZF_ANALYSIS_INPUT_ROOT", PACK / "derived_data"))
BASE_SCRIPT = HERE / "04_endpoint_provenance_support_and_null.py"
SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"
EDGE_DIR = DERIVED_ROOT / "figure12" / "sc_reconstruction" / SC_SOURCE
PRIMARY_REGION_PATH = (
    REFERENCE_ROOT / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
OUTPUT = (
    DERIVED_ROOT / "figure12" / "validation" / "r12_primary"
    / "morphology_subsample_size"
)
FIGURE = (
    PACK / "figures" / "validation" / "r12_primary"
    / "figure12_morphology_subsample_size_sensitivity.png"
)
FCV_PATH = REFERENCE_ROOT / "figure9" / "figure9_region_summary.csv"

SUBJECTS = tuple(range(12, 19))
FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
METRICS = ("DCApost", "OO_fraction")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if str(FIG12) not in sys.path:
    sys.path.insert(0, str(FIG12))
base = load_module(BASE_SCRIPT, "morphology_support_base")
from skeleton_path_endpoint_classes import build_endpoint_classes  # noqa: E402


ENDPOINTS = None
SOURCE_LABELS = None


def load_endpoint_classes() -> None:
    global ENDPOINTS, SOURCE_LABELS
    if ENDPOINTS is None:
        ENDPOINTS, SOURCE_LABELS, _ = build_endpoint_classes()


def morphology_relation_contributions(
    subject: int,
    cell_xyz: np.ndarray,
    membership: np.ndarray,
    unit_region: np.ndarray,
    affine: np.ndarray,
    squared_radius: float,
    calibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one row per morphology-supported directed unit relation."""
    load_endpoint_classes()
    cell_tree = cKDTree(cell_xyz.astype(float))
    block_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    radius = np.sqrt(squared_radius)

    for block_id, block in enumerate(ENDPOINTS):
        if block_id % 250 == 0:
            print(
                f"[morphology subsets] subject {subject}: "
                f"morphology {block_id}/{len(ENDPOINTS)}",
                flush=True,
            )
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed = calibration.transform(block, affine.ravel())

        candidate_lists = cell_tree.query_ball_point(transformed, radius, workers=1)
        nonempty = [ids for ids in candidate_lists if len(ids)]
        if not nonempty:
            continue
        candidates = np.unique(np.concatenate(nonempty)).astype(np.int64)
        endpoint_tree = cKDTree(transformed)
        distance, endpoint_id = endpoint_tree.query(
            cell_xyz[candidates].astype(float), k=1, workers=1
        )
        keep = (distance * distance) < squared_radius
        candidates = candidates[keep]
        endpoint_id = endpoint_id[keep]
        if not candidates.size:
            continue

        nearby = [candidates[endpoint_id == index] for index in range(len(block))]
        labels = np.asarray(SOURCE_LABELS[block_id], dtype=bool)
        source_parts = [nearby[index] for index in np.flatnonzero(labels) if nearby[index].size]
        target_parts = [nearby[index] for index in np.flatnonzero(~labels) if nearby[index].size]
        if not source_parts or not target_parts:
            continue

        source_units = np.unique(membership[np.concatenate(source_parts)])
        target_units = np.unique(membership[np.concatenate(target_parts)])
        source_grid = np.repeat(source_units, len(target_units))
        target_grid = np.tile(target_units, len(source_units))
        valid = (
            (source_grid != target_grid)
            & (unit_region[source_grid] != unit_region[target_grid])
        )
        source_grid = source_grid[valid]
        target_grid = target_grid[valid]
        if not source_grid.size:
            continue

        block_rows.append(np.full(source_grid.size, block_id, dtype=np.int32))
        source_rows.append(source_grid.astype(np.int32, copy=False))
        target_rows.append(target_grid.astype(np.int32, copy=False))

    if not block_rows:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.concatenate(block_rows),
        np.concatenate(source_rows),
        np.concatenate(target_rows),
    )


def load_or_build_subject(subject: int, calibration) -> dict[str, np.ndarray]:
    cache_dir = OUTPUT / "subject_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"subject_{subject}_morphology_unit_relations.npz"
    if cache_path.exists():
        print(f"[morphology subsets] subject {subject}: loading cache", flush=True)
        with np.load(cache_path, allow_pickle=False) as payload:
            cached = {name: np.asarray(payload[name]) for name in payload.files}
        if "canonical_adjacency" in cached:
            return cached
        # Upgrade the initial relation-only cache without repeating endpoint matching.
        edge_path = EDGE_DIR / f"subject_{subject}_full_affine_endpoint_edges.npz"
        with np.load(edge_path, allow_pickle=False) as payload:
            source = np.asarray(payload["source"], dtype=np.int64)
            target = np.asarray(payload["target"], dtype=np.int64)
            neuron_region = np.asarray(payload["neuron_root_area"], dtype=np.int64)
        membership, unit_region = base.unit_membership(subject, neuron_region)
        cached["canonical_adjacency"] = base.aggregate_unit_adjacency(
            source, target, membership, unit_region
        )
        cached["unit_region"] = unit_region
        np.savez_compressed(cache_path, **cached)
        return cached

    edge_path = EDGE_DIR / f"subject_{subject}_full_affine_endpoint_edges.npz"
    with np.load(edge_path, allow_pickle=False) as payload:
        cell_xyz = np.asarray(payload["CellXYZ"], dtype=np.int32)
        neuron_region = np.asarray(payload["neuron_root_area"], dtype=np.int64)
        affine = np.asarray(payload["affine_matrix"], dtype=float)
        squared_radius = float(payload["squared_distance_threshold"])
    membership, unit_region = base.unit_membership(subject, neuron_region)
    with np.load(edge_path, allow_pickle=False) as payload:
        canonical_source = np.asarray(payload["source"], dtype=np.int64)
        canonical_target = np.asarray(payload["target"], dtype=np.int64)
    canonical_adjacency = base.aggregate_unit_adjacency(
        canonical_source, canonical_target, membership, unit_region
    )
    block, source, target = morphology_relation_contributions(
        subject,
        cell_xyz,
        membership,
        unit_region,
        affine,
        squared_radius,
        calibration,
    )
    np.savez_compressed(
        cache_path,
        block_id=block,
        source_unit=source,
        target_unit=target,
        unit_region=unit_region,
        canonical_adjacency=canonical_adjacency,
        squared_distance_threshold=np.asarray(squared_radius),
    )
    return {
        "block_id": block,
        "source_unit": source,
        "target_unit": target,
        "unit_region": unit_region,
        "canonical_adjacency": canonical_adjacency,
        "squared_distance_threshold": np.asarray(squared_radius),
    }


def adjacency_from_subset(data: dict[str, np.ndarray], selected: np.ndarray) -> np.ndarray:
    n_units = len(data["unit_region"])
    selected_lookup = np.zeros(len(ENDPOINTS), dtype=bool)
    selected_lookup[selected] = True
    keep = selected_lookup[data["block_id"].astype(np.int64)]
    selected_support = np.zeros((n_units, n_units), dtype=float)
    np.add.at(
        selected_support,
        (data["source_unit"][keep].astype(np.int64), data["target_unit"][keep].astype(np.int64)),
        1.0,
    )
    total_support = np.zeros((n_units, n_units), dtype=float)
    np.add.at(
        total_support,
        (
            data["source_unit"].astype(np.int64),
            data["target_unit"].astype(np.int64),
        ),
        1.0,
    )
    support_fraction = np.divide(
        selected_support,
        total_support,
        out=np.zeros_like(selected_support),
        where=total_support > 0,
    )
    return np.asarray(data["canonical_adjacency"], dtype=float) * support_fraction


def average_regional(
    subjects: dict[int, dict[str, np.ndarray]],
    selected: np.ndarray,
    n_regions: int,
) -> pd.DataFrame:
    rows = []
    for subject, data in subjects.items():
        adjacency = adjacency_from_subset(data, selected)
        regional = base.regional_measures(adjacency, data["unit_region"], n_regions)
        regional["subject"] = subject
        rows.append(regional)
    return pd.concat(rows, ignore_index=True).groupby("root_area_id", as_index=False).agg(
        DCApost=("DCApost", "mean"),
        OO_fraction=("OO_fraction", "mean"),
    )


def valid_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue


def evaluate_regional(
    regional: pd.DataFrame,
    full: pd.DataFrame,
    fcv: pd.DataFrame,
    fixed_ids: np.ndarray,
) -> list[dict]:
    table = (
        regional.merge(full, on="root_area_id", suffixes=("", "_full"))
        .merge(fcv, on="root_area_id")
    )
    table = table[table["root_area_id"].isin(fixed_ids)]
    rows = []
    for metric in METRICS:
        value = table[metric].to_numpy(float)
        full_value = table[f"{metric}_full"].to_numpy(float)
        fcv_value = table["EdgeStdFCV"].to_numpy(float)
        reproducibility = valid_correlation(value, full_value)
        fcv_result = valid_correlation(value, fcv_value)
        rows.append({
            "metric": metric,
            "n_regions": int(np.isfinite(value + full_value + fcv_value).sum()),
            "full_reproducibility_r": reproducibility[0],
            "full_reproducibility_p": reproducibility[1],
            "fcv_pearson_r": fcv_result[0],
            "fcv_pearson_p": fcv_result[1],
            "fcv_spearman_rho": fcv_result[2],
            "fcv_spearman_p": fcv_result[3],
        })
    return rows


def summarize(iterations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fraction, metric), frame in iterations.groupby(["subset_fraction", "metric"]):
        rows.append({
            "subset_fraction": fraction,
            "metric": metric,
            "n_iterations": len(frame),
            "n_morphologies": int(frame["n_morphologies"].median()),
            "median_full_reproducibility_r": frame["full_reproducibility_r"].median(),
            "full_reproducibility_ci_low": frame["full_reproducibility_r"].quantile(0.025),
            "full_reproducibility_ci_high": frame["full_reproducibility_r"].quantile(0.975),
            "median_fcv_pearson_r": frame["fcv_pearson_r"].median(),
            "fcv_pearson_r_ci_low": frame["fcv_pearson_r"].quantile(0.025),
            "fcv_pearson_r_ci_high": frame["fcv_pearson_r"].quantile(0.975),
            "proportion_fcv_r_positive": (frame["fcv_pearson_r"] > 0).mean(),
            "proportion_fcv_p_below_0_05": (frame["fcv_pearson_p"] < 0.05).mean(),
        })
    return pd.DataFrame(rows).sort_values(["metric", "subset_fraction"])


def make_figure(summary: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    colors = {"OO_fraction": "#D55E00", "DCApost": "#0072B2"}
    labels = {"OO_fraction": "OO fraction", "DCApost": r"$\mathrm{DCA}_{\mathrm{post}}$"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for metric in METRICS:
        frame = summary[summary["metric"].eq(metric)].sort_values("subset_fraction")
        x = 100 * frame["subset_fraction"].to_numpy(float)
        for ax, center, low, high in (
            (
                axes[0],
                "median_full_reproducibility_r",
                "full_reproducibility_ci_low",
                "full_reproducibility_ci_high",
            ),
            (
                axes[1],
                "median_fcv_pearson_r",
                "fcv_pearson_r_ci_low",
                "fcv_pearson_r_ci_high",
            ),
        ):
            y = frame[center].to_numpy(float)
            lo = frame[low].to_numpy(float)
            hi = frame[high].to_numpy(float)
            ax.plot(x, y, marker="o", color=colors[metric], lw=1.8, label=labels[metric])
            ax.fill_between(x, lo, hi, color=colors[metric], alpha=0.16, linewidth=0)

    axes[0].set(
        xlabel="Reconstructed morphologies retained (%)",
        ylabel="Correlation with full-support measure",
        title="Structural-measure reproducibility",
        ylim=(-0.05, 1.05),
    )
    axes[1].axhline(0, color="0.65", lw=0.8)
    axes[1].set(
        xlabel="Reconstructed morphologies retained (%)",
        ylabel="FCV Pearson $r$",
        title="Structure--FCV association",
    )
    for panel, ax in zip("AB", axes):
        ax.text(-0.13, 1.06, panel, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def morphology_root_coverage(
    subjects: dict[int, dict[str, np.ndarray]], n_regions: int
) -> np.ndarray:
    """Map each morphology to root areas from which it generates output."""
    coverage = np.zeros((len(ENDPOINTS), n_regions), dtype=bool)
    for data in subjects.values():
        block = data["block_id"].astype(np.int64)
        source_unit = data["source_unit"].astype(np.int64)
        source_root = data["unit_region"][source_unit].astype(np.int64)
        coverage[block, source_root] = True
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=SUBJECTS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--allow-incomplete-primary-regions", action="store_true",
        help="For partial-subject smoke tests, restrict to primary regions with morphology support.",
    )
    args = parser.parse_args()

    load_endpoint_classes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration = base.load_calibration_module()
    subjects = {subject: load_or_build_subject(subject, calibration) for subject in args.subjects}

    eligible = np.unique(np.concatenate([
        data["block_id"].astype(np.int64) for data in subjects.values() if data["block_id"].size
    ]))
    if not eligible.size:
        raise RuntimeError("No reconstructed morphology generated an inter-regional unit relation")
    pd.DataFrame({"block_id": eligible}).to_csv(OUTPUT / "eligible_morphologies.csv", index=False)
    pd.DataFrame([{
        "weighting_rule": "canonical_unit_edge_weight_times_selected_over_total_morphology_support",
        "n_eligible_morphologies": len(eligible),
        "n_iterations_per_incomplete_fraction": args.iterations,
        "random_seed": args.seed,
    }]).to_csv(OUTPUT / "analysis_settings.csv", index=False)

    fcv = pd.read_csv(FCV_PATH)[["root_area_id", "EdgeStdFCV"]].dropna()
    # Match the canonical 42-region primary analysis exactly.  The earlier
    # morphology-support audit used a 41-region complete-case set that omitted
    # OB because it lacked values under stricter support thresholds; no such
    # restriction is needed for morphology subsampling.
    primary_regions = pd.read_csv(PRIMARY_REGION_PATH)
    fixed_ids = primary_regions["root_area_id"].drop_duplicates().to_numpy(int)
    primary_regions.to_csv(OUTPUT / "fixed_primary_42_regions.csv", index=False)
    n_regions = int(fcv["root_area_id"].max()) + 1
    full = average_regional(subjects, eligible, n_regions)
    full.to_csv(OUTPUT / "full_morphology_support_region_measures.csv", index=False)

    root_coverage = morphology_root_coverage(subjects, n_regions)
    covered = root_coverage[eligible][:, fixed_ids].any(axis=0)
    if not covered.all():
        missing = fixed_ids[~covered]
        if not args.allow_incomplete_primary_regions:
            raise RuntimeError(f"Primary regions without morphology output support: {missing.tolist()}")
        print(
            f"[morphology subsets] partial-subject test: excluding unsupported regions "
            f"{missing.tolist()}", flush=True,
        )
        fixed_ids = fixed_ids[covered]
        pd.DataFrame({"root_area_id": fixed_ids}).to_csv(
            OUTPUT / "fixed_primary_42_regions.csv", index=False
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    fractions = sorted(set(float(value) for value in args.fractions))
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"Subset fraction must be in (0, 1], got {fraction}")
        n_selected = min(len(eligible), max(1, int(round(fraction * len(eligible)))))
        n_iterations = 1 if np.isclose(fraction, 1.0) else args.iterations
        for iteration in range(n_iterations):
            if iteration % 25 == 0:
                print(
                    f"[morphology subsets] fraction={fraction:.2f}, "
                    f"iteration={iteration}/{n_iterations}",
                    flush=True,
                )
            attempts = 1
            if n_selected == len(eligible):
                selected = eligible
            else:
                while True:
                    selected = np.sort(rng.choice(eligible, size=n_selected, replace=False))
                    if root_coverage[selected][:, fixed_ids].any(axis=0).all():
                        break
                    attempts += 1
                    if attempts > 100_000:
                        raise RuntimeError(
                            f"Could not draw a {fraction:.2f} morphology subset "
                            "covering all 42 primary regions"
                        )
            regional = average_regional(subjects, selected, n_regions)
            for result in evaluate_regional(regional, full, fcv, fixed_ids):
                if result["n_regions"] != len(fixed_ids):
                    raise RuntimeError(
                        f"Expected {len(fixed_ids)} regions, got {result['n_regions']} "
                        f"for fraction={fraction:.2f}, iteration={iteration}, "
                        f"metric={result['metric']}"
                    )
                rows.append({
                    "subset_fraction": fraction,
                    "n_morphologies": n_selected,
                    "iteration": iteration,
                    "sampling_attempts": attempts,
                    **result,
                })

    iterations = pd.DataFrame(rows)
    summary = summarize(iterations)
    iterations.to_csv(OUTPUT / "morphology_subsample_iterations.csv", index=False)
    summary.to_csv(OUTPUT / "morphology_subsample_summary.csv", index=False)
    make_figure(summary)
    print(summary.to_string(index=False))
    print(f"Saved results to {OUTPUT}")
    print(f"Saved figure to {FIGURE}")


if __name__ == "__main__":
    main()
