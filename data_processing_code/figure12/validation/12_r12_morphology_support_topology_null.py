#!/usr/bin/env python3
"""Morphology-support and topology-null controls for the canonical r12 SC."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
FIG12 = HERE.parent
PACK = FIG12.parents[1]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data"))
BASE_SCRIPT = HERE / "04_endpoint_provenance_support_and_null.py"
SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"
OUTPUT = DERIVED_ROOT / "figure12" / "validation" / "r12_primary" / "morphology_support_topology_null"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "historical_endpoint_null")
if str(FIG12) not in sys.path:
    sys.path.insert(0, str(FIG12))
from skeleton_path_endpoint_classes import build_endpoint_classes  # noqa: E402


ENDPOINTS = None
SOURCE_LABELS = None


def load_endpoint_classes() -> None:
    global ENDPOINTS, SOURCE_LABELS
    if ENDPOINTS is None:
        ENDPOINTS, SOURCE_LABELS, _ = build_endpoint_classes()


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment without an optional statsmodels import."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def morphology_support(subject, cell_xyz, membership, unit_region, affine, squared_radius, calibration):
    """Count blocks supporting each unit pair under nearest-endpoint assignment."""
    load_endpoint_classes()
    cell_tree = cKDTree(cell_xyz.astype(float))
    support = np.zeros((len(unit_region), len(unit_region)), dtype=np.int32)
    for block_id, block in enumerate(ENDPOINTS):
        if block_id % 250 == 0:
            print(f"[r12 support] subject {subject}: morphology {block_id}/{len(ENDPOINTS)}", flush=True)
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed = calibration.transform(block, affine.ravel())
        candidate_lists = cell_tree.query_ball_point(transformed, np.sqrt(squared_radius), workers=1)
        candidate_lists = [ids for ids in candidate_lists if len(ids)]
        if not candidate_lists:
            continue
        candidates = np.unique(np.concatenate(candidate_lists)).astype(np.int64)
        endpoint_tree = cKDTree(transformed)
        distance, endpoint_id = endpoint_tree.query(cell_xyz[candidates].astype(float), k=1, workers=1)
        keep = (distance * distance) < squared_radius
        candidates, endpoint_id = candidates[keep], endpoint_id[keep]
        if candidates.size == 0:
            continue
        nearby = [candidates[endpoint_id == index] for index in range(len(transformed))]
        labels = np.asarray(SOURCE_LABELS[block_id], dtype=bool)
        source_cells = [nearby[index] for index in np.flatnonzero(labels) if nearby[index].size]
        target_cells = [nearby[index] for index in np.flatnonzero(~labels) if nearby[index].size]
        if not source_cells or not target_cells:
            continue
        source_units = np.unique(membership[np.concatenate(source_cells)])
        target_units = np.unique(membership[np.concatenate(target_cells)])
        for source_unit in source_units:
            valid = target_units[
                (target_units != source_unit) & (unit_region[target_units] != unit_region[source_unit])
            ]
            support[source_unit, valid] += 1
    return support


def write_topology_inference() -> None:
    """Test whether each observed positive correlation differs from its null."""
    nulls = pd.read_csv(OUTPUT / "topology_preserving_null_fcv_correlations.csv")
    observed = pd.read_csv(OUTPUT / "support_filter_fcv_correlations.csv")
    observed = observed.loc[observed["support_threshold"].eq(1)].set_index("metric")
    rows = []
    for (null_type, metric), group in nulls.groupby(["null_type", "metric"], sort=True):
        values = group["pearson_r"].to_numpy(float)
        observed_r = float(observed.loc[metric, "pearson_r"])
        upper_p = (1 + np.count_nonzero(values >= observed_r)) / (len(values) + 1)
        lower_p = (1 + np.count_nonzero(values <= observed_r)) / (len(values) + 1)
        null_mean = float(values.mean())
        centered_p = (
            1 + np.count_nonzero(np.abs(values - null_mean) >= abs(observed_r - null_mean))
        ) / (len(values) + 1)
        rows.append({
            "null_type": null_type,
            "metric": metric,
            "n_null": len(values),
            "observed_pearson_r": observed_r,
            "null_mean_r": null_mean,
            "null_ci_low": np.quantile(values, 0.025),
            "null_ci_high": np.quantile(values, 0.975),
            "empirical_null_centered_two_sided_p_primary": centered_p,
            "empirical_upper_tail_p_sensitivity": upper_p,
            "empirical_equal_tail_two_sided_p": min(1.0, 2 * min(upper_p, lower_p)),
            "inference_note": (
                "Primary alternative: the observed positive FCV correlation differs from "
                "the center of the topology-null distribution."
            ),
        })
    inference = pd.DataFrame(rows)
    inference["empirical_null_centered_two_sided_q_bh"] = np.nan
    for _, indices in inference.groupby("null_type", sort=False).groups.items():
        p_values = inference.loc[
            indices,
            "empirical_null_centered_two_sided_p_primary",
        ].to_numpy(float)
        inference.loc[indices, "empirical_null_centered_two_sided_q_bh"] = bh_fdr(p_values)

    column_order = [
        "null_type",
        "metric",
        "n_null",
        "observed_pearson_r",
        "null_mean_r",
        "null_ci_low",
        "null_ci_high",
        "empirical_null_centered_two_sided_p_primary",
        "empirical_null_centered_two_sided_q_bh",
        "empirical_upper_tail_p_sensitivity",
        "empirical_equal_tail_two_sided_p",
        "inference_note",
    ]
    inference = inference[column_order]
    inference.to_csv(OUTPUT / "topology_preserving_null_inference_summary.csv", index=False)

    publication = inference.pivot(
        index="metric",
        columns="null_type",
        values=[
            "observed_pearson_r",
            "null_mean_r",
            "empirical_null_centered_two_sided_p_primary",
            "empirical_null_centered_two_sided_q_bh",
        ],
    )
    publication.columns = [f"{value}__{null_type}" for value, null_type in publication.columns]
    publication.reset_index().to_csv(
        OUTPUT / "topology_preserving_null_primary_tests.csv",
        index=False,
    )


def main() -> None:
    base.EDGE_DIR = DERIVED_ROOT / "figure12" / "sc_reconstruction" / SC_SOURCE
    base.OUTPUT = OUTPUT
    base.CACHE_DIR = OUTPUT / "subject_cache"
    base.FIGURE_DIR = PACK / "figures" / "validation" / "r12_primary"
    base.morphology_support = morphology_support
    base.main()
    write_topology_inference()


if __name__ == "__main__":
    main()
