#!/usr/bin/env python3
"""Unit-size/radius sensitivity using the canonical r12 reconstruction rules.

This wraps the historical grid implementation but replaces all construction
steps that define the revised analysis: skeleton-path KMeans endpoint labels,
the skeleton-KMeans FCS affine calibration, and one nearest endpoint per soma
within each morphology block.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
FIG12 = HERE.parent
PACK = FIG12.parents[1]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data"))
BASE_SCRIPT = HERE / "01_unit_target_endpoint_radius_sensitivity.py"
OUT = DERIVED_ROOT / "figure12" / "validation" / "r12_primary" / "unit_target_endpoint_radius"
CALIBRATION_OUT = DERIVED_ROOT / "figure12" / "sc_reconstruction"
PARAMETERS = (
    "m00", "m01", "m02", "bx", "m10", "m11", "m12", "by",
    "m20", "m21", "m22", "bz",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "historical_sensitivity")
calibration = base.load_calibration_module()
if str(FIG12) not in sys.path:
    sys.path.insert(0, str(FIG12))
from skeleton_path_endpoint_classes import build_endpoint_classes  # noqa: E402


def load_affine(subject: int) -> np.ndarray:
    path = CALIBRATION_OUT / f"subject{subject}_fcs_transform_search_full_affine_skeleton_path_kmeans.csv"
    table = pd.read_csv(path)
    best = table.loc[table["rho_fcs_sc_strength"].idxmax(), list(PARAMETERS)]
    return best.to_numpy(float).reshape(3, 4)


def load_transformed_blocks(subject: int, _calibration):
    affine = load_affine(subject)
    endpoints, labels, _ = build_endpoint_classes()
    transformed_blocks, source_masks = [], []
    for block_id, block in enumerate(endpoints):
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed_blocks.append(calibration.transform(block, affine.ravel()))
        source_masks.append(np.asarray(labels[block_id], dtype=bool))
    return transformed_blocks, source_masks


def nearest_cell_adjacency(tree, coordinates, blocks, source_masks, radius):
    """Exact nearest-endpoint assignment, restricted to radius candidates."""
    adjacency = [set() for _ in range(len(coordinates))]
    radius_squared = radius * radius
    for block, source_mask in zip(blocks, source_masks, strict=False):
        candidate_lists = tree.query_ball_point(block, radius, workers=1)
        candidate_lists = [ids for ids in candidate_lists if len(ids)]
        if not candidate_lists:
            continue
        candidates = np.unique(np.concatenate(candidate_lists)).astype(np.int64)
        endpoint_tree = cKDTree(np.asarray(block, dtype=float))
        distance, endpoint_id = endpoint_tree.query(coordinates[candidates].astype(float), k=1, workers=1)
        keep = (distance * distance) < radius_squared
        candidates = candidates[keep]
        endpoint_id = endpoint_id[keep]
        if candidates.size == 0:
            continue
        nearby = [candidates[endpoint_id == index] for index in range(len(block))]
        selected_set = set(candidates.tolist())
        for source in candidates:
            adjacency[int(source)].difference_update(selected_set)
        source_chunks = [nearby[index] for index in np.flatnonzero(source_mask) if nearby[index].size]
        target_chunks = [nearby[index] for index in np.flatnonzero(~source_mask) if nearby[index].size]
        if source_chunks and target_chunks:
            source_nodes = np.unique(np.concatenate(source_chunks))
            target_nodes = np.unique(np.concatenate(target_chunks))
            if np.intersect1d(source_nodes, target_nodes, assume_unique=True).size:
                raise RuntimeError("Nearest-endpoint assignment produced source/target overlap")
            target_set = set(target_nodes.tolist())
            for source in source_nodes:
                adjacency[int(source)].update(target_set)
                adjacency[int(source)].discard(int(source))
    return adjacency


def main() -> None:
    base.BASE_OUT = OUT
    base.OUT = OUT
    base.RADII = (10.0, 11.0, 12.0, 13.0, 13.86, 15.0)
    base.load_affine = load_affine
    base.load_transformed_blocks = load_transformed_blocks
    base.cell_adjacency = nearest_cell_adjacency
    base.main()


if __name__ == "__main__":
    main()
