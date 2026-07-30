#!/usr/bin/env python3
"""Audit endpoint support and topology-preserving nulls for Figure 12 FU-SC.

The endpoint reconstruction uses morphology-block overwrite semantics, so a
final cell edge records only its last generating block.  Support is therefore
defined at the directed anatomical-unit (FU) level as the number of distinct
morphology blocks that can generate the same FU-to-FU relation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
FIG12 = HERE.parent
PACK = FIG12.parents[1]
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK / "raw_data"))
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data"))
REFERENCE_ROOT = Path(os.environ.get("ZF_ANALYSIS_INPUT_ROOT", PACK / "derived_data"))
RECONSTRUCTION = FIG12 / "sc_reconstruction"
CALIBRATION_PATH = RECONSTRUCTION / "26_calibrate_subject12_transform_to_fcs.py"
EDGE_DIR = DERIVED_ROOT / "figure12" / "sc_reconstruction"
OUTPUT = DERIVED_ROOT / "figure12" / "validation" / "endpoint_provenance"
CACHE_DIR = OUTPUT / "subject_cache"
FIGURE_DIR = PACK / "figures" / "validation"
ANATOMY_DIR = Path(os.environ.get("ZF_ANATOMY_ROOT", RAW_ROOT / "figure12" / "anatomy"))
ORIGINAL_SC_DIR = Path(
    os.environ.get("ZF_PREPARED_SUBJECT_ROOT", RAW_ROOT / "figure9" / "original_subjects")
)
FCV_PATH = REFERENCE_ROOT / "figure9" / "figure9_region_summary.csv"
CANONICAL_PATH = REFERENCE_ROOT / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"

SUBJECTS = tuple(range(12, 19))
SUPPORT_THRESHOLDS = (1, 2, 3)
SOMA_DISTANCE_THRESHOLD = 230.0


def load_calibration_module():
    spec = importlib.util.spec_from_file_location("fcs_calibration", CALIBRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rank1_dca(adjacency: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Rank-1 directed coreness asymmetry used in the Figure 12 FU analysis."""
    graph = sparse.csr_matrix(adjacency)
    n = graph.shape[0]
    c_out = np.full(n, 1.0 / np.sqrt(max(n, 1)), dtype=float)
    c_in = c_out.copy()
    for _ in range(max_iter):
        next_out = graph @ c_in
        norm_out = np.linalg.norm(next_out)
        if norm_out == 0:
            return np.full(n, np.nan)
        next_out /= norm_out
        next_in = graph.T @ next_out
        norm_in = np.linalg.norm(next_in)
        if norm_in == 0:
            return np.full(n, np.nan)
        next_in /= norm_in
        delta = max(np.linalg.norm(next_out - c_out), np.linalg.norm(next_in - c_in))
        c_out, c_in = next_out, next_in
        if delta < tol:
            break
    return c_out - c_in


def unit_membership(subject: int, neuron_region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = ORIGINAL_SC_DIR / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    unit_region = np.asarray(raw["root_area"], dtype=np.int64)
    membership = np.full(len(neuron_region), -1, dtype=np.int64)
    for unit_id, cell_ids in enumerate(raw["final_id_cluster"]):
        membership[np.asarray(cell_ids, dtype=np.int64)] = unit_id
    if (membership < 0).any() or not np.array_equal(unit_region[membership], neuron_region):
        raise ValueError(f"Subject {subject}: endpoint SC ordering does not match final_id_cluster")
    return membership, unit_region


def aggregate_unit_adjacency(
    source: np.ndarray, target: np.ndarray, membership: np.ndarray, unit_region: np.ndarray
) -> np.ndarray:
    """Aggregate source/target arrays without making a second full edge copy."""
    source_unit = membership[source]
    target_unit = membership[target]
    keep = (source_unit != target_unit) & (unit_region[source_unit] != unit_region[target_unit])
    adjacency = np.zeros((len(unit_region), len(unit_region)), dtype=float)
    np.add.at(adjacency, (source_unit[keep], target_unit[keep]), 1.0)
    return adjacency


def strict_nearby(tree: cKDTree, cell_xyz: np.ndarray, endpoints: np.ndarray, squared_radius: float) -> list[np.ndarray]:
    candidates = tree.query_ball_point(endpoints, np.sqrt(squared_radius), workers=1)
    nearby = []
    for endpoint, ids in zip(endpoints, candidates):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size:
            delta = cell_xyz[ids].astype(float) - endpoint
            ids = ids[np.einsum("ij,ij->i", delta, delta) < squared_radius]
        nearby.append(ids)
    return nearby


def morphology_support(
    subject: int,
    cell_xyz: np.ndarray,
    membership: np.ndarray,
    unit_region: np.ndarray,
    affine: np.ndarray,
    squared_radius: float,
    calibration,
) -> np.ndarray:
    """Count morphology blocks independently supporting each directed FU pair."""
    endpoints = sio.loadmat(
        ANATOMY_DIR / "neuronEndpoints_data.mat", squeeze_me=True, struct_as_record=False
    )["neuronEndpoints"]
    soma = np.asarray(sio.loadmat(
        ANATOMY_DIR / "somaCoordinates_data.mat", squeeze_me=True, struct_as_record=False
    )["somaCoordinates"])
    tree = cKDTree(cell_xyz.astype(float))
    support = np.zeros((len(unit_region), len(unit_region)), dtype=np.int32)
    for block_id, block in enumerate(endpoints):
        if block_id % 250 == 0:
            print(f"[support] subject {subject}: morphology {block_id}/{len(endpoints)}", flush=True)
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed = calibration.transform(block, affine.ravel())
        nearby = strict_nearby(tree, cell_xyz, transformed, squared_radius)
        distances = np.sqrt(
            (soma[block_id, 0] - block[:, 0]) ** 2
            + (soma[block_id, 1] - block[:, 1]) ** 2
            + 4.0 * (soma[block_id, 2] - block[:, 2]) ** 2
        )
        source_cells = [nearby[i] for i in np.flatnonzero(distances < SOMA_DISTANCE_THRESHOLD) if nearby[i].size]
        target_cells = [nearby[i] for i in np.flatnonzero(distances >= SOMA_DISTANCE_THRESHOLD) if nearby[i].size]
        if not source_cells or not target_cells:
            continue
        source_units = np.unique(membership[np.concatenate(source_cells)])
        target_units = np.unique(membership[np.concatenate(target_cells)])
        for source_unit in source_units:
            valid_targets = target_units[
                (target_units != source_unit) & (unit_region[target_units] != unit_region[source_unit])
            ]
            support[source_unit, valid_targets] += 1
    return support


def regional_measures(adjacency: np.ndarray, unit_region: np.ndarray, n_regions: int) -> pd.DataFrame:
    dca = rank1_dca(adjacency)
    out_strength, in_strength = adjacency.sum(axis=1), adjacency.sum(axis=0)
    positive = dca > 0
    post_num = adjacency @ dca
    oo_num = (adjacency * np.outer(positive, positive)).sum(axis=1)
    out_region = np.bincount(unit_region, weights=out_strength, minlength=n_regions)
    in_region = np.bincount(unit_region, weights=in_strength, minlength=n_regions)
    post_region = np.bincount(unit_region, weights=post_num, minlength=n_regions)
    oo_region = np.bincount(unit_region, weights=oo_num, minlength=n_regions)
    return pd.DataFrame({
        "root_area_id": np.arange(n_regions),
        "DCApost": np.divide(post_region, out_region, out=np.full(n_regions, np.nan), where=out_region > 0),
        "OO_fraction": np.divide(oo_region, out_region, out=np.full(n_regions, np.nan), where=out_region > 0),
        "inter_out_strength": out_region,
        "inter_in_strength": in_region,
    })


def correlations(regional: pd.DataFrame, fcv: pd.DataFrame, region_ids: np.ndarray) -> list[dict]:
    merged = regional.merge(fcv, on="root_area_id", how="inner")
    merged = merged[merged["root_area_id"].isin(region_ids)]
    rows = []
    for metric in ("DCApost", "OO_fraction"):
        data = merged[[metric, "EdgeStdFCV"]].dropna()
        pearson = pearsonr(data[metric], data["EdgeStdFCV"])
        spearman = spearmanr(data[metric], data["EdgeStdFCV"])
        rows.append({
            "metric": metric, "n_regions": len(data),
            "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
            "spearman_rho": spearman.statistic, "spearman_p": spearman.pvalue,
        })
    return rows


def maximum_entropy_null(
    adjacency: np.ndarray,
    unit_region: np.ndarray,
    rng: np.random.Generator,
    constrained_root_pairs: bool,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Sample a weighted maximum-entropy null with prescribed strength margins.

    Both nulls retain each FU's total outgoing and incoming weight.  The
    root-pair-constrained version also retains total weight for every ordered
    source-root/target-root pair.  Within those constraints, edge weights are
    randomly redistributed over admissible inter-root FU pairs.
    """
    out_strength, in_strength = adjacency.sum(axis=1), adjacency.sum(axis=0)
    source_root = unit_region[:, None]
    target_root = unit_region[None, :]
    allowed = source_root != target_root
    n_regions = int(unit_region.max()) + 1
    pair_weight = np.zeros((n_regions, n_regions), dtype=float)
    np.add.at(pair_weight, (source_root.repeat(len(unit_region), axis=1), target_root.repeat(len(unit_region), axis=0)), adjacency)
    if constrained_root_pairs:
        allowed &= pair_weight[source_root, target_root] > 0
    randomized = rng.exponential(scale=1.0, size=adjacency.shape) * allowed
    randomized[out_strength == 0, :] = 0.0
    randomized[:, in_strength == 0] = 0.0

    for _ in range(iterations):
        row_sum = randomized.sum(axis=1)
        row_scale = np.divide(out_strength, row_sum, out=np.zeros_like(out_strength), where=row_sum > 0)
        randomized *= row_scale[:, None]
        column_sum = randomized.sum(axis=0)
        column_scale = np.divide(in_strength, column_sum, out=np.zeros_like(in_strength), where=column_sum > 0)
        randomized *= column_scale[None, :]
        if constrained_root_pairs:
            current_pair = np.zeros_like(pair_weight)
            np.add.at(current_pair, (source_root.repeat(len(unit_region), axis=1), target_root.repeat(len(unit_region), axis=0)), randomized)
            pair_scale = np.divide(pair_weight, current_pair, out=np.zeros_like(pair_weight), where=current_pair > 0)
            randomized *= pair_scale[source_root, target_root]
    max_relative_margin_error = max(
        np.max(np.abs(randomized.sum(axis=1) - out_strength) / np.maximum(out_strength, 1.0)),
        np.max(np.abs(randomized.sum(axis=0) - in_strength) / np.maximum(in_strength, 1.0)),
    )
    return randomized, float(max_relative_margin_error)


def make_figure(support: pd.DataFrame, nulls: pd.DataFrame, observed: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "figure12_endpoint_provenance_support_null.png"
    null_types = list(nulls["null_type"].drop_duplicates())
    fig, axes = plt.subplots(
        1, 1 + len(null_types), figsize=(5.0 * (1 + len(null_types)), 4.3),
        constrained_layout=True, squeeze=False,
    )
    axes = axes.ravel()
    metric_labels = {"DCApost": r"$\mathrm{DCA}_{\mathrm{post}}$", "OO_fraction": "OO fraction"}

    for metric, frame in support.groupby("metric", sort=False):
        axes[0].plot(frame["support_threshold"], frame["pearson_r"], marker="o", linewidth=2,
                     label=metric_labels[metric])
    axes[0].axhline(0, color="0.65", linewidth=0.8)
    axes[0].set(xlabel="Minimum morphology support per FU edge", ylabel="FCV Pearson r",
                xticks=sorted(support["support_threshold"].unique()), title="Support-strength sensitivity")
    axes[0].legend(frameon=False)

    titles = {
        "strength_preserving": "Strength-preserving null",
        "root_pair_preserving": "Strength + root-pair null",
    }
    for ax, null_type in zip(axes[1:], null_types, strict=True):
        for metric, color in (("DCApost", "#4c78a8"), ("OO_fraction", "#e45756")):
            values = nulls[(nulls["null_type"] == null_type) & (nulls["metric"] == metric)]["pearson_r"]
            ax.hist(values, bins=24, density=True, alpha=0.55, color=color, label=metric_labels[metric])
            observed_value = observed[observed["metric"] == metric]["pearson_r"].iloc[0]
            ax.axvline(observed_value, color=color, linewidth=2.5, linestyle="--")
        ax.set(xlabel="FCV Pearson r", ylabel="Null density", title=titles[null_type])
        ax.legend(frameon=False, fontsize=8)
    for panel, ax in zip("ABC", axes):
        ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontweight="bold", fontsize=14)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--n-null", type=int, default=250)
    parser.add_argument("--ipf-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--include-root-pair-null", action="store_true",
        help="Also run the exploratory root-pair-preserving null omitted from the SI.",
    )
    parser.add_argument("--cache-only", action="store_true", help="Build subject-level FU provenance caches only.")
    parser.add_argument("--from-cache", action="store_true", help="Use existing subject-level FU provenance caches.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration_module()
    fcv = pd.read_csv(FCV_PATH)[["root_area_id", "EdgeStdFCV"]].dropna()
    canonical = set(pd.read_csv(CANONICAL_PATH)["root_area_id"].astype(int))
    n_regions = int(fcv["root_area_id"].max()) + 1

    subjects: dict[int, dict] = {}
    all_rows, edge_rows = [], []
    for subject in args.subjects:
        cache_path = CACHE_DIR / f"subject_{subject}_fu_endpoint_provenance.npz"
        if args.from_cache:
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing provenance cache: {cache_path}")
            print(f"[provenance] subject {subject}: loading FU cache", flush=True)
            cached = np.load(cache_path, allow_pickle=False)
            adjacency = np.asarray(cached["adjacency"], dtype=float)
            support = np.asarray(cached["morphology_support"], dtype=np.int32)
            unit_region = np.asarray(cached["unit_region"], dtype=np.int64)
        else:
            print(f"[provenance] subject {subject}: loading endpoint SC", flush=True)
            with np.load(EDGE_DIR / f"subject_{subject}_full_affine_endpoint_edges.npz", allow_pickle=False) as payload:
                source = np.asarray(payload["source"], dtype=np.int64)
                target = np.asarray(payload["target"], dtype=np.int64)
                cell_xyz = np.asarray(payload["CellXYZ"], dtype=np.int32)
                neuron_region = np.asarray(payload["neuron_root_area"], dtype=np.int64)
                affine = np.asarray(payload["affine_matrix"], dtype=float)
                squared_radius = float(payload["squared_distance_threshold"])
            membership, unit_region = unit_membership(subject, neuron_region)
            adjacency = aggregate_unit_adjacency(source, target, membership, unit_region)
            support = morphology_support(subject, cell_xyz, membership, unit_region, affine, squared_radius, calibration)
            np.savez_compressed(
                cache_path,
                adjacency=adjacency,
                morphology_support=support,
                unit_region=unit_region,
                squared_distance_threshold=np.asarray(squared_radius),
            )
        final_pairs = np.argwhere(adjacency > 0)
        pair_support = support[final_pairs[:, 0], final_pairs[:, 1]]
        edge_rows.extend({
            "subject": subject, "source_unit": int(source), "target_unit": int(target),
            "source_root_area": int(unit_region[source]), "target_root_area": int(unit_region[target]),
            "cell_edge_weight": float(adjacency[source, target]), "morphology_support_count": int(count),
        } for (source, target), count in zip(final_pairs, pair_support))
        subjects[subject] = {"adjacency": adjacency, "support": support, "unit_region": unit_region}

        for threshold in SUPPORT_THRESHOLDS:
            filtered = adjacency.copy()
            filtered[support < threshold] = 0.0
            regional = regional_measures(filtered, unit_region, n_regions)
            regional.insert(0, "subject", subject)
            regional.insert(1, "support_threshold", threshold)
            all_rows.append(regional)
        if not args.from_cache:
            del source, target, cell_xyz, neuron_region, membership
        del pair_support
        gc.collect()

    if args.cache_only:
        print(f"Saved subject-level provenance caches to {CACHE_DIR}")
        return

    support_table = pd.concat(all_rows, ignore_index=True)
    support_table.to_csv(OUTPUT / "subject_region_support_sensitivity.csv", index=False)
    edge_table = pd.DataFrame(edge_rows)
    edge_table.to_csv(OUTPUT / "functional_unit_edge_morphology_support.csv", index=False)

    average = support_table.groupby(["support_threshold", "root_area_id"], as_index=False).agg(
        DCApost=("DCApost", "mean"), OO_fraction=("OO_fraction", "mean"),
        n_subjects=("subject", "nunique"),
    )
    fixed = canonical & set(fcv["root_area_id"].astype(int))
    for _, frame in average.groupby("support_threshold"):
        fixed &= set(frame.dropna(subset=["DCApost", "OO_fraction"])["root_area_id"].astype(int))
    fixed_ids = np.asarray(sorted(fixed), dtype=int)
    pd.DataFrame({"root_area_id": fixed_ids}).to_csv(OUTPUT / "fixed_complete_regions.csv", index=False)

    observed_rows = []
    for threshold, frame in average.groupby("support_threshold", sort=True):
        for row in correlations(frame, fcv, fixed_ids):
            observed_rows.append({"analysis": "support_filter", "support_threshold": threshold, **row})
    observed = pd.DataFrame(observed_rows)
    observed.to_csv(OUTPUT / "support_filter_fcv_correlations.csv", index=False)

    observed_full = observed[observed["support_threshold"] == 1].copy()
    observed_full["null_type"] = "observed"
    rng = np.random.default_rng(args.seed)
    null_specs = [("strength_preserving", False)]
    if args.include_root_pair_null:
        null_specs.append(("root_pair_preserving", True))
    null_rows = []
    for realization in range(args.n_null):
        if realization % 25 == 0:
            print(f"[null] realization {realization}/{args.n_null}", flush=True)
        for null_type, constrained in null_specs:
            regional_rows = []
            margin_errors = []
            for subject, data in subjects.items():
                randomized, margin_error = maximum_entropy_null(
                    data["adjacency"], data["unit_region"], rng, constrained, args.ipf_iterations
                )
                regional = regional_measures(randomized, data["unit_region"], n_regions)
                regional["subject"] = subject
                regional["max_relative_margin_error"] = margin_error
                regional_rows.append(regional)
                margin_errors.append(margin_error)
            mean_regional = pd.concat(regional_rows, ignore_index=True).groupby("root_area_id", as_index=False).agg(
                DCApost=("DCApost", "mean"), OO_fraction=("OO_fraction", "mean")
            )
            for row in correlations(mean_regional, fcv, fixed_ids):
                null_rows.append({
                    "realization": realization, "null_type": null_type,
                    "max_relative_margin_error": max(margin_errors), **row,
                })
    nulls = pd.DataFrame(null_rows)
    nulls.to_csv(OUTPUT / "topology_preserving_null_fcv_correlations.csv", index=False)

    summary_rows = []
    for null_type, frame in nulls.groupby("null_type"):
        for metric, values in frame.groupby("metric"):
            observed_r = observed_full.loc[observed_full["metric"] == metric, "pearson_r"].iloc[0]
            null_r = values["pearson_r"].to_numpy(float)
            empirical_p = (1 + np.sum(np.abs(null_r) >= abs(observed_r))) / (len(null_r) + 1)
            summary_rows.append({
                "null_type": null_type, "metric": metric, "n_null": len(null_r),
                "observed_pearson_r": observed_r, "null_mean_r": null_r.mean(),
                "null_sd_r": null_r.std(ddof=1),
                "null_ci_low": np.quantile(null_r, 0.025), "null_ci_high": np.quantile(null_r, 0.975),
                "empirical_two_sided_p": empirical_p,
                "median_max_relative_margin_error": values["max_relative_margin_error"].median(),
            })
    null_summary = pd.DataFrame(summary_rows)
    null_summary.to_csv(OUTPUT / "topology_preserving_null_summary.csv", index=False)
    figure = make_figure(observed[observed["analysis"] == "support_filter"], nulls, observed_full)
    print(f"Fixed complete regions: {len(fixed_ids)}")
    print(f"Saved provenance audit to {OUTPUT}")
    print(f"Saved figure to {figure}")


if __name__ == "__main__":
    main()
