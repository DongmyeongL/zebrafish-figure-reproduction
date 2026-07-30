#!/usr/bin/env python3
"""Joint sensitivity analysis for anatomical-unit size and endpoint radius.

The subject-specific affine transform is fixed to the existing FCS-calibrated
solution.  For every configuration, the same K-means anatomical-unit assignment
is used to aggregate endpoint-derived cell edges and calculate unit DCA, OO,
and root-area downstream measures.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans


HERE = Path(__file__).resolve().parent
FIG12_DIR = HERE.parent
PACK = FIG12_DIR.parents[1]
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK / "raw_data"))
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data"))
REFERENCE_ROOT = Path(os.environ.get("ZF_ANALYSIS_INPUT_ROOT", PACK / "derived_data"))
RECONSTRUCTION = FIG12_DIR / "sc_reconstruction"
CALIBRATION_PATH = RECONSTRUCTION / "26_calibrate_subject12_transform_to_fcs.py"
BASE_OUT = DERIVED_ROOT / "figure12" / "validation" / "unit_target_endpoint_radius"
OUT = BASE_OUT
CALIBRATION_OUT = DERIVED_ROOT / "figure12" / "sc_reconstruction"
ANATOMY = Path(os.environ.get("ZF_ANATOMY_ROOT", RAW_ROOT / "figure12" / "anatomy"))
SUBJECT_DATA = Path(
    os.environ.get("ZF_ORIGINAL_MAT_ROOT", RAW_ROOT / "figure12" / "original_subject_mat")
)
FCV_TABLE = REFERENCE_ROOT / "figure9" / "figure9_region_summary.csv"
CANONICAL_REGIONS = REFERENCE_ROOT / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"

SUBJECTS = tuple(range(12, 19))
TARGET_SIZES = (250, 300, 400, 500, 600)
RADII = (10.0, 12.0, 13.86, 15.0)
KMEANS_SEED = 42
SOMA_DISTANCE_THRESHOLD = 230.0
PARAMETERS = (
    "m00", "m01", "m02", "bx",
    "m10", "m11", "m12", "by",
    "m20", "m21", "m22", "bz",
)


def load_calibration_module():
    spec = importlib.util.spec_from_file_location("fcs_calibration", CALIBRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rank1_dca(adjacency: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Rank-1 directed coreness asymmetry on a weighted unit adjacency matrix."""
    graph = sparse.csr_matrix(adjacency)
    n_units = graph.shape[0]
    c_out = np.full(n_units, 1.0 / np.sqrt(max(n_units, 1)), dtype=float)
    c_in = c_out.copy()
    for _ in range(max_iter):
        next_out = graph @ c_in
        out_norm = np.linalg.norm(next_out)
        if out_norm <= 0:
            return np.full(n_units, np.nan)
        next_out /= out_norm
        next_in = graph.T @ next_out
        in_norm = np.linalg.norm(next_in)
        if in_norm <= 0:
            return np.full(n_units, np.nan)
        next_in /= in_norm
        delta = max(np.linalg.norm(next_out - c_out), np.linalg.norm(next_in - c_in))
        c_out, c_in = next_out, next_in
        if delta < tol:
            break
    return c_out - c_in


def subject_cells(subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return a fixed cell order, root-area labels, and members per root area."""
    raw = sio.loadmat(SUBJECT_DATA / f"subject_{subject}_data.mat")
    cell_xyz = np.asarray(raw["CellXYZ"])
    result_idx = raw["result_idx"].flatten()
    region_members = [np.asarray(ids).ravel().astype(np.int64) - 1 for ids in result_idx]
    cell_ids = np.unique(np.concatenate(region_members))
    raw_to_compact = np.full(cell_xyz.shape[0], -1, dtype=np.int64)
    raw_to_compact[cell_ids] = np.arange(cell_ids.size)
    neuron_region = np.full(cell_ids.size, -1, dtype=np.int64)
    compact_members = []
    for region_id, members in enumerate(region_members):
        compact = raw_to_compact[members]
        compact = compact[compact >= 0]
        neuron_region[compact] = region_id
        compact_members.append(compact)
    if (neuron_region < 0).any():
        raise RuntimeError(f"Subject {subject}: cells lack a root-area assignment")
    cluster_coordinates = cell_xyz[cell_ids].astype(np.int32, copy=True)
    coordinates = cluster_coordinates.copy()
    coordinates[:, 2] *= 2
    return coordinates, cluster_coordinates, neuron_region, compact_members


def make_unit_membership(coordinates: np.ndarray, region_members: list[np.ndarray], target_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute K-means anatomical units using the historical 400-cell rule generalized to target size."""
    membership = np.full(len(coordinates), -1, dtype=np.int64)
    unit_regions: list[int] = []
    unit_sizes: list[int] = []
    unit_id = 0
    for region_id, members in enumerate(region_members):
        n_cells = len(members)
        n_clusters = int(n_cells / target_size)
        if n_cells > target_size and n_cells % target_size < target_size / 4:
            n_clusters -= 1
        if n_clusters > 1:
            labels = KMeans(n_clusters=n_clusters, random_state=KMEANS_SEED).fit_predict(coordinates[members])
            groups = [members[labels == label] for label in range(n_clusters)]
        else:
            groups = [members]
        for group in groups:
            if not len(group):
                continue
            membership[group] = unit_id
            unit_regions.append(region_id)
            unit_sizes.append(len(group))
            unit_id += 1
    if (membership < 0).any():
        raise RuntimeError("Some cells were not assigned to an anatomical unit")
    return membership, np.asarray(unit_regions), np.asarray(unit_sizes)


def load_affine(subject: int) -> np.ndarray:
    table = pd.read_csv(CALIBRATION_OUT / f"subject{subject}_fcs_transform_search_full_affine.csv")
    best = table.loc[table["rho_fcs_sc_strength"].idxmax(), list(PARAMETERS)]
    return best.to_numpy(float).reshape(3, 4)


def load_transformed_blocks(subject: int, calibration) -> tuple[list[np.ndarray], list[np.ndarray]]:
    affine = load_affine(subject)
    endpoints = sio.loadmat(ANATOMY / "neuronEndpoints_data.mat", squeeze_me=True, struct_as_record=False)["neuronEndpoints"]
    soma = np.asarray(sio.loadmat(ANATOMY / "somaCoordinates_data.mat", squeeze_me=True, struct_as_record=False)["somaCoordinates"])
    transformed_blocks, source_masks = [], []
    for block_id, block in enumerate(endpoints):
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed_blocks.append(calibration.transform(block, affine.ravel()))
        distance = np.sqrt(
            (soma[block_id, 0] - block[:, 0]) ** 2
            + (soma[block_id, 1] - block[:, 1]) ** 2
            + 4.0 * (soma[block_id, 2] - block[:, 2]) ** 2
        )
        source_masks.append(distance < SOMA_DISTANCE_THRESHOLD)
    return transformed_blocks, source_masks


def cell_adjacency(tree: cKDTree, coordinates: np.ndarray, blocks: list[np.ndarray], source_masks: list[np.ndarray], radius: float) -> list[set[int]]:
    """Rebuild binary cell SC with the validated morphology-block overwrite rule."""
    adjacency = [set() for _ in range(len(coordinates))]
    radius_squared = radius * radius
    for block, source_mask in zip(blocks, source_masks, strict=False):
        nearby = tree.query_ball_point(block, radius, workers=1)
        nearby = [
            np.asarray(ids, dtype=np.int64)[
                np.einsum("ij,ij->i", coordinates[np.asarray(ids, dtype=np.int64)].astype(float) - endpoint,
                          coordinates[np.asarray(ids, dtype=np.int64)].astype(float) - endpoint) < radius_squared
            ] if len(ids) else np.empty(0, dtype=np.int64)
            for endpoint, ids in zip(block, nearby, strict=False)
        ]
        nonempty = [ids for ids in nearby if ids.size]
        if not nonempty:
            continue
        selected = np.unique(np.concatenate(nonempty))
        selected_set = set(selected.tolist())
        for source in selected:
            adjacency[int(source)].difference_update(selected_set)
        source_chunks = [nearby[index] for index in np.flatnonzero(source_mask) if nearby[index].size]
        target_chunks = [nearby[index] for index in np.flatnonzero(~source_mask) if nearby[index].size]
        if source_chunks and target_chunks:
            source_nodes = np.unique(np.concatenate(source_chunks))
            target_set = set(np.unique(np.concatenate(target_chunks)).tolist())
            for source in source_nodes:
                adjacency[int(source)].update(target_set)
                adjacency[int(source)].discard(int(source))
    return adjacency


def region_measures(adjacency: list[set[int]], membership: np.ndarray, unit_regions: np.ndarray, n_regions: int) -> tuple[pd.DataFrame, dict]:
    n_units = len(unit_regions)
    unit_adjacency = np.zeros((n_units, n_units), dtype=float)
    for source, targets in enumerate(adjacency):
        if not targets:
            continue
        source_unit = membership[source]
        target_cells = np.fromiter(targets, dtype=np.int64)
        target_units = membership[target_cells]
        keep = (target_units != source_unit) & (unit_regions[target_units] != unit_regions[source_unit])
        np.add.at(unit_adjacency, (np.full(int(keep.sum()), source_unit), target_units[keep]), 1.0)
    dca = rank1_dca(unit_adjacency)
    out_strength = unit_adjacency.sum(axis=1)
    in_strength = unit_adjacency.sum(axis=0)
    post_num = unit_adjacency @ dca
    pre_num = unit_adjacency.T @ dca
    positive = dca > 0
    oo_num = (unit_adjacency * np.outer(positive, positive)).sum(axis=1)
    out_region = np.bincount(unit_regions, weights=out_strength, minlength=n_regions)
    in_region = np.bincount(unit_regions, weights=in_strength, minlength=n_regions)
    post_region_num = np.bincount(unit_regions, weights=post_num, minlength=n_regions)
    pre_region_num = np.bincount(unit_regions, weights=pre_num, minlength=n_regions)
    oo_region_num = np.bincount(unit_regions, weights=oo_num, minlength=n_regions)
    result = pd.DataFrame({
        "root_area_id": np.arange(n_regions),
        "DCApost": np.divide(post_region_num, out_region, out=np.full(n_regions, np.nan), where=out_region > 0),
        "DCApre": np.divide(pre_region_num, in_region, out=np.full(n_regions, np.nan), where=in_region > 0),
        "OO_fraction": np.divide(oo_region_num, out_region, out=np.full(n_regions, np.nan), where=out_region > 0),
        "inter_out_edges": out_region,
        "inter_in_edges": in_region,
    })
    qc = {
        "n_anatomical_units": n_units,
        "n_inter_unit_edges": int(unit_adjacency.sum()),
        "n_finite_dca_units": int(np.isfinite(dca).sum()),
    }
    return result, qc


def correlation_rows(measures: pd.DataFrame, fcv: pd.DataFrame, fixed_regions: np.ndarray) -> list[dict]:
    merged = measures.merge(fcv, on="root_area_id", how="inner")
    merged = merged[merged["root_area_id"].isin(fixed_regions)].dropna(subset=["DCApost", "OO_fraction", "EdgeStdFCV"])
    rows = []
    for metric in ("DCApost", "OO_fraction"):
        x, y = merged[metric].to_numpy(float), merged["EdgeStdFCV"].to_numpy(float)
        pearson = pearsonr(x, y) if len(x) >= 3 and np.unique(x).size > 1 else None
        spearman = spearmanr(x, y) if pearson is not None else None
        rows.append({
            "metric": metric, "n_regions": len(x),
            "pearson_r": pearson.statistic if pearson else np.nan,
            "pearson_p": pearson.pvalue if pearson else np.nan,
            "spearman_rho": spearman.statistic if spearman else np.nan,
            "spearman_p": spearman.pvalue if spearman else np.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--target-sizes", type=int, nargs="+", default=TARGET_SIZES)
    parser.add_argument("--radii", type=float, nargs="+", default=RADII)
    parser.add_argument("--run-name", type=str, default="", help="Write one independent subject-radius run below runs/.")
    args = parser.parse_args()
    global OUT
    if args.run_name:
        OUT = BASE_OUT / "runs" / args.run_name
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration_module()
    fcv = pd.read_csv(FCV_TABLE)[["root_area_id", "EdgeStdFCV"]]
    canonical = pd.read_csv(CANONICAL_REGIONS)["root_area_id"].to_numpy(int)
    n_regions = int(fcv["root_area_id"].max()) + 1
    measure_rows, qc_rows = [], []

    for subject in args.subjects:
        print(f"[sensitivity] subject {subject}", flush=True)
        coordinates, cluster_coordinates, neuron_region, region_members = subject_cells(subject)
        tree = cKDTree(coordinates.astype(float))
        blocks, source_masks = load_transformed_blocks(subject, calibration)
        memberships = {
            target: make_unit_membership(cluster_coordinates, region_members, target)
            for target in args.target_sizes
        }
        for radius in args.radii:
            print(f"  radius={radius:g}", flush=True)
            adjacency = cell_adjacency(tree, coordinates, blocks, source_masks, radius)
            for target, (membership, unit_regions, unit_sizes) in memberships.items():
                regional, qc = region_measures(adjacency, membership, unit_regions, n_regions)
                regional.insert(0, "subject", subject)
                regional.insert(1, "target_size", target)
                regional.insert(2, "endpoint_radius", radius)
                regional["endpoint_squared_threshold"] = radius * radius
                measure_rows.append(regional)
                qc_rows.append({
                    "subject": subject, "target_size": target, "endpoint_radius": radius,
                    "endpoint_squared_threshold": radius * radius,
                    "n_cells": len(coordinates), "median_unit_size": float(np.median(unit_sizes)),
                    "min_unit_size": int(unit_sizes.min()), "max_unit_size": int(unit_sizes.max()), **qc,
                })
            adjacency.clear()
            del adjacency
            gc.collect()

    all_measures = pd.concat(measure_rows, ignore_index=True)
    all_measures.to_csv(OUT / "subject_region_measures.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(OUT / "subject_configuration_qc.csv", index=False)

    summary = all_measures.groupby(["target_size", "endpoint_radius", "root_area_id"], as_index=False).agg(
        DCApost=("DCApost", "mean"), DCApre=("DCApre", "mean"),
        OO_fraction=("OO_fraction", "mean"), n_subjects=("subject", "nunique"),
    )
    complete_by_configuration = summary.groupby(["target_size", "endpoint_radius"]).apply(
        lambda table: set(table.dropna(subset=["DCApost", "OO_fraction"])["root_area_id"].tolist())
    )
    fixed_regions = set(canonical.tolist()) & set(fcv.dropna()["root_area_id"].tolist())
    for region_set in complete_by_configuration:
        fixed_regions &= region_set
    fixed_regions = np.asarray(sorted(fixed_regions), dtype=int)
    pd.DataFrame({"root_area_id": fixed_regions}).to_csv(OUT / "fixed_complete_regions.csv", index=False)

    correlation = []
    for (target, radius), configuration in summary.groupby(["target_size", "endpoint_radius"], sort=True):
        for row in correlation_rows(configuration, fcv, fixed_regions):
            correlation.append({"target_size": target, "endpoint_radius": radius,
                                "endpoint_squared_threshold": radius * radius, **row})
    pd.DataFrame(correlation).to_csv(OUT / "fcv_structure_correlations.csv", index=False)
    summary.to_csv(OUT / "region_measures_mean_across_subjects.csv", index=False)
    print(f"Saved sensitivity outputs to {OUT}")
    print(f"Fixed complete regions: {len(fixed_regions)}")


if __name__ == "__main__":
    main()
