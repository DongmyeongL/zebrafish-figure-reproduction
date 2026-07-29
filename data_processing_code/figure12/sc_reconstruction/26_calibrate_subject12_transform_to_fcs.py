#!/usr/bin/env python3
"""Calibrate endpoint coordinates to subject-specific regional FCS.

This is an exploratory calibration tool.  It deliberately uses an
endpoint-derived regional SC-strength surrogate, without interpolation or
cell-level DCA/OO calculations.  The selected transform must be evaluated on
held-out functional data before it can support an independent SC--FC claim.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parents[2]
FIGURE12_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FIGURE12_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE12_DIR))

from importlib.util import module_from_spec, spec_from_file_location
from skeleton_path_endpoint_classes import build_endpoint_classes


SUBJECT = 12
N_REGIONS = 72
POST_PRE_THRESHOLD = 230.0
CPP_SQUARED_THRESHOLD = 192.0
RADIUS = np.sqrt(CPP_SQUARED_THRESHOLD)
BASE = np.asarray([1.18, 54.42795, 1.204, -32.267997, 0.749148, 18.437146])
BASE_FULL_AFFINE = np.asarray([
    1.18, 0.0, 0.0, 54.42795,
    0.0, 1.204, 0.0, -32.267997,
    0.0, 0.0, 0.749148, 18.437146,
])
RESTRICTED_NAMES = ("ax", "bx", "ay", "by", "az", "bz")
FULL_AFFINE_NAMES = (
    "m00", "m01", "m02", "bx",
    "m10", "m11", "m12", "by",
    "m20", "m21", "m22", "bz",
)
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK_ROOT / "raw_data"))
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data"))
OUT_DIR = DERIVED_ROOT / "figure12" / "sc_reconstruction"
FCS_TABLE = DERIVED_ROOT / "figure9" / "figure9_recording_region_measures.csv"
FUNCTIONAL_UNIT_TRACE_TEMPLATE = (
    RAW_ROOT / "figure9" / "functional_unit_traces"
    / "zebrafish_subject_{subject}_raw_cluster_traces.pkl"
)
ORIGINAL_SUBJECTS = RAW_ROOT / "figure9" / "original_subjects"
ANATOMY_DIR = Path(os.environ.get("ZF_ANATOMY_ROOT", RAW_ROOT / "figure12" / "anatomy"))
ORIGINAL_MAT_ROOT = Path(
    os.environ.get("ZF_ORIGINAL_MAT_ROOT", RAW_ROOT / "figure12" / "original_subject_mat")
)


def rebuild_units(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Recreate endpoint-matching cells and their root-area assignments.

    Coordinates follow the historical generator: cells are concatenated in
    saved functional-unit order, cast to integer coordinates, and have their
    z coordinate doubled before endpoint matching.
    """
    path = ORIGINAL_MAT_ROOT / f"subject_{subject}_data.mat"
    raw = sio.loadmat(path)
    cell_xyz = np.asarray(raw["CellXYZ"])
    result_idx = raw["result_idx"].flatten()
    region_members = [np.asarray(ids).ravel().astype(np.int64) - 1 for ids in result_idx]
    means = np.asarray([cell_xyz[ids].mean(axis=0) for ids in region_members])
    by_x = np.argsort(means[:, 0])
    region_order = np.concatenate((by_x[means[by_x, 1] <= 300], by_x[means[by_x, 1] > 300]))
    chunks = []
    regions = []
    from sklearn.cluster import KMeans
    for region in region_order:
        members = region_members[region]
        n_cells = len(members)
        n_clusters = int(n_cells / 400)
        if n_cells > 400 and n_cells % 400 < 100:
            n_clusters -= 1
        if n_cells > 400 and n_clusters > 1:
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(cell_xyz[members])
            member_groups = [members[labels == cluster] for cluster in range(n_clusters)]
        else:
            member_groups = [members]
        for member_ids in member_groups:
            if member_ids.size:
                chunks.append(cell_xyz[member_ids])
                regions.append(np.full(member_ids.size, region, dtype=np.int32))
    coordinates = np.concatenate(chunks, axis=0).astype(np.int32, copy=False)
    coordinates[:, 2] *= 2
    return coordinates, np.concatenate(regions)


def load_endpoint_blocks(endpoint_label_rule: str = "direct_soma_distance") -> tuple[list[np.ndarray], np.ndarray]:
    endpoints = sio.loadmat(
        ANATOMY_DIR / "neuronEndpoints_data.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["neuronEndpoints"]
    if endpoint_label_rule == "skeleton_path_kmeans":
        _, skeleton_labels, _ = build_endpoint_classes(ANATOMY_DIR)
        blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for block, block_labels in zip(endpoints, skeleton_labels):
            if len(block.shape) <= 1:
                continue
            blocks.append(np.asarray(block, dtype=float))
            labels.append(np.asarray(block_labels, dtype=bool))
        return blocks, np.concatenate(labels)
    if endpoint_label_rule != "direct_soma_distance":
        raise ValueError(f"Unsupported endpoint label rule: {endpoint_label_rule}")
    soma = np.asarray(
        sio.loadmat(
            ANATOMY_DIR / "somaCoordinates_data.mat",
            squeeze_me=True,
            struct_as_record=False,
        )["somaCoordinates"]
    )
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for block_id, block in enumerate(endpoints):
        if len(block.shape) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        distance = np.sqrt(
            (soma[block_id, 0] - block[:, 0]) ** 2
            + (soma[block_id, 1] - block[:, 1]) ** 2
            + 4.0 * (soma[block_id, 2] - block[:, 2]) ** 2
        )
        blocks.append(block)
        labels.append(distance < POST_PRE_THRESHOLD)
    return blocks, np.concatenate(labels)


def transform(block: np.ndarray, params: np.ndarray) -> np.ndarray:
    if len(params) == len(RESTRICTED_NAMES):
        ax, bx, ay, by, az, bz = params
        return np.column_stack((ax * block[:, 1] + bx, ay * block[:, 0] + by, az * block[:, 2] + bz))
    matrix = np.asarray(params, dtype=float).reshape(3, 4)
    source = block[:, [1, 0, 2]]
    return source @ matrix[:, :3].T + matrix[:, 3]


def fcs_vector() -> np.ndarray:
    table = pd.read_csv(FCS_TABLE)
    table = table.loc[table["recording_id"].eq(f"subject_{SUBJECT}")]
    values = table.set_index("root_area_id")["FCS"]
    return values.reindex(np.arange(N_REGIONS)).to_numpy(dtype=float)


def regional_sc_strength(
    params: np.ndarray,
    blocks: list[np.ndarray],
    endpoint_labels: np.ndarray,
    tree: cKDTree,
    neuron_region: np.ndarray,
    region_sizes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return a symmetrized, region-pair-normalized endpoint SC surrogate."""
    regional_counts = np.zeros((N_REGIONS, N_REGIONS), dtype=np.float64)
    label_offset = 0
    nonempty_endpoints = 0
    for block in blocks:
        n_endpoint = len(block)
        labels = endpoint_labels[label_offset : label_offset + n_endpoint]
        label_offset += n_endpoint
        nearby = tree.query_ball_point(transform(block, params), RADIUS, workers=1)
        source_parts = [np.asarray(nearby[i], dtype=np.int32) for i in np.flatnonzero(labels) if len(nearby[i])]
        target_parts = [np.asarray(nearby[i], dtype=np.int32) for i in np.flatnonzero(~labels) if len(nearby[i])]
        nonempty_endpoints += sum(bool(x) for x in nearby)
        if not source_parts or not target_parts:
            continue
        source_regions = neuron_region[np.unique(np.concatenate(source_parts))]
        target_regions = neuron_region[np.unique(np.concatenate(target_parts))]
        source_count = np.bincount(source_regions, minlength=N_REGIONS)
        target_count = np.bincount(target_regions, minlength=N_REGIONS)
        regional_counts += np.outer(source_count, target_count)
        del nearby, source_parts, target_parts

    symmetric = regional_counts + regional_counts.T
    np.fill_diagonal(symmetric, 0.0)
    possible_pairs = np.outer(region_sizes, region_sizes).astype(float)
    np.fill_diagonal(possible_pairs, np.nan)
    density = np.divide(symmetric, possible_pairs, out=np.full_like(symmetric, np.nan), where=np.isfinite(possible_pairs) & (possible_pairs > 0))
    strength = np.nanmean(np.log1p(density), axis=1)
    return strength, nonempty_endpoints


def regional_endpoint_density(
    params: np.ndarray,
    blocks: list[np.ndarray],
    endpoint_labels: np.ndarray,
    tree: cKDTree,
    neuron_region: np.ndarray,
    region_sizes: np.ndarray,
) -> np.ndarray:
    """Return the directed regional endpoint-density matrix used by the surrogate."""
    counts = np.zeros((N_REGIONS, N_REGIONS), dtype=float)
    label_offset = 0
    for block in blocks:
        n_endpoint = len(block)
        labels = endpoint_labels[label_offset : label_offset + n_endpoint]
        label_offset += n_endpoint
        nearby = tree.query_ball_point(transform(block, params), RADIUS, workers=1)
        source_parts = [np.asarray(nearby[i], dtype=np.int32) for i in np.flatnonzero(labels) if len(nearby[i])]
        target_parts = [np.asarray(nearby[i], dtype=np.int32) for i in np.flatnonzero(~labels) if len(nearby[i])]
        if not source_parts or not target_parts:
            continue
        source_count = np.bincount(neuron_region[np.unique(np.concatenate(source_parts))], minlength=N_REGIONS)
        target_count = np.bincount(neuron_region[np.unique(np.concatenate(target_parts))], minlength=N_REGIONS)
        counts += np.outer(source_count, target_count)
    possible = np.outer(region_sizes, region_sizes).astype(float)
    density = np.divide(counts, possible, out=np.full_like(counts, np.nan), where=possible > 0)
    np.fill_diagonal(density, np.nan)
    return density


def subject_regional_fc_matrix(subject: int) -> tuple[np.ndarray, dict[int, str]]:
    """Recompute Figure-9-style mean FC and average it between root areas."""
    trace_path = Path(str(FUNCTIONAL_UNIT_TRACE_TEMPLATE).format(subject=subject))
    with trace_path.open("rb") as handle:
        raw = pickle.load(handle)
    traces = np.asarray(raw["traces"], dtype=float)
    rate = float(raw["sampling_rate_hz"])
    b, a = butter(2, 0.03 / (0.5 * rate), btype="high")
    traces = filtfilt(b, a, traces, axis=1)
    traces = (traces - np.nanmean(traces, axis=1, keepdims=True)) / np.nanstd(traces, axis=1, keepdims=True)
    roots = np.asarray(raw["root_area_ids"], dtype=int)
    names = {int(region): str(name) for region, name in zip(raw["root_area_ids"], raw["root_area_names"])}
    fc_sum = np.zeros((len(traces), len(traces)), dtype=float)
    n_windows = 0
    for start in range(0, traces.shape[1] - 20 + 1, 5):
        fc_sum += np.corrcoef(traces[:, start : start + 20])
        n_windows += 1
    mean_fc = fc_sum / n_windows
    matrix = np.full((N_REGIONS, N_REGIONS), np.nan)
    for source_region in np.unique(roots):
        source_idx = np.flatnonzero(roots == source_region)
        for target_region in np.unique(roots):
            if source_region == target_region:
                continue
            target_idx = np.flatnonzero(roots == target_region)
            matrix[source_region, target_region] = np.nanmean(mean_fc[np.ix_(source_idx, target_idx)])
    return matrix, names


def save_fc_sc_matrices(fc_matrix, sc_density, keep, names, tag):
    """Save matched root-area FC and endpoint-SC matrices for the best transform."""
    fc = fc_matrix[np.ix_(keep, keep)]
    sc = np.log1p(sc_density[np.ix_(keep, keep)])
    labels = [names.get(int(region), str(region)) for region in keep]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    fc_limit = np.nanmax(np.abs(fc))
    im_fc = axes[0].imshow(fc, cmap="coolwarm", vmin=-fc_limit, vmax=fc_limit, interpolation="nearest")
    im_sc = axes[1].imshow(sc, cmap="magma", vmin=0, vmax=np.nanpercentile(sc, 99), interpolation="nearest")
    for ax, title in zip(axes, ("Functional connectivity", "Endpoint-derived directed SC density")):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Root area")
        ax.set_ylabel("Root area")
        tick_step = max(1, len(labels) // 12)
        ticks = np.arange(0, len(labels), tick_step)
        ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=90, fontsize=6)
        ax.set_yticks(ticks, [labels[i] for i in ticks], fontsize=6)
    fig.colorbar(im_fc, ax=axes[0], shrink=0.8, label="Mean sliding-window FC")
    fig.colorbar(im_sc, ax=axes[1], shrink=0.8, label="log(1 + directed density)")
    fig.suptitle(f"Subject {SUBJECT}: matched regional FC and endpoint-derived SC matrices", fontweight="bold")
    path = OUT_DIR / f"subject{SUBJECT}_fc_sc_matrices_{tag}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(OUT_DIR / f"subject{SUBJECT}_fc_sc_matrices_{tag}.npz", root_area_id=keep, fc=fc, sc_log_density=sc)
    return path


def evaluate(params, blocks, labels, tree, neuron_region, region_sizes, fcs):
    strength, nonempty = regional_sc_strength(params, blocks, labels, tree, neuron_region, region_sizes)
    valid = np.isfinite(strength) & np.isfinite(fcs)
    rho, p = spearmanr(strength[valid], fcs[valid])
    return float(rho), float(p), int(valid.sum()), int(nonempty), strength


def save_fcs_sc_diagnostic(fcs, baseline_strength, best_strength, tag):
    """Save the in-sample FCS calibration diagnostic for visual inspection."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)
    for ax, label, strength in zip(
        axes,
        ("Original transform", "Best FCS-calibrated transform"),
        (baseline_strength, best_strength),
    ):
        valid = np.isfinite(fcs) & np.isfinite(strength)
        rho, p = spearmanr(fcs[valid], strength[valid])
        ax.scatter(fcs[valid], strength[valid], s=24, color="#3A7CA5", alpha=0.8, linewidths=0)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(fcs[valid], strength[valid], 1)
            x = np.linspace(fcs[valid].min(), fcs[valid].max(), 100)
            ax.plot(x, slope * x + intercept, color="black", lw=1.2)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Subject {SUBJECT} FCS")
        ax.set_ylabel("Endpoint SC-strength surrogate")
        ax.text(
            0.04, 0.96, f"Spearman $\\rho$ = {rho:.3f}\np = {p:.2e}\nn = {valid.sum()}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "pad": 2.5},
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Subject {SUBJECT} FCS-based endpoint-transform calibration (in-sample)", fontsize=11, fontweight="bold")
    path = OUT_DIR / f"subject{SUBJECT}_fcs_vs_endpoint_sc_strength_{tag}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def candidate_parameters(
    rng: np.random.Generator,
    n_coarse: int,
    n_refine: int,
    full_affine: bool,
    best: np.ndarray | None = None,
):
    if best is not None:
        for _ in range(n_refine):
            if full_affine:
                candidate = best.reshape(3, 4).copy()
                candidate[np.arange(3), np.arange(3)] *= rng.uniform(0.97, 1.03, size=3)
                candidate[:, [0, 1, 2]] += rng.uniform(-0.025, 0.025, size=(3, 3)) * (1 - np.eye(3))
                candidate[:, 3] += rng.uniform([-5, -5, -3], [5, 5, 3])
                yield candidate.ravel(), "refine"
            else:
                scale = rng.uniform(0.97, 1.03, size=3)
                shift = rng.uniform([-5, -5, -3], [5, 5, 3])
                yield np.asarray([
                    best[0] * scale[0], best[1] + shift[0],
                    best[2] * scale[1], best[3] + shift[1],
                    best[4] * scale[2], best[5] + shift[2],
                ]), "refine"
        return

    baseline = BASE_FULL_AFFINE if full_affine else BASE
    yield baseline.copy(), "baseline"
    if best is None:
        for _ in range(n_coarse):
            if full_affine:
                candidate = BASE_FULL_AFFINE.reshape(3, 4).copy()
                candidate[np.arange(3), np.arange(3)] *= rng.uniform(0.88, 1.12, size=3)
                candidate[:, [0, 1, 2]] += rng.uniform(-0.12, 0.12, size=(3, 3)) * (1 - np.eye(3))
                candidate[:, 3] += rng.uniform([-25, -25, -15], [25, 25, 15])
                yield candidate.ravel(), "coarse"
            else:
                scale = rng.uniform(0.88, 1.12, size=3)
                shift = rng.uniform([-25, -25, -15], [25, 25, 15])
                yield np.asarray([
                    BASE[0] * scale[0], BASE[1] + shift[0],
                    BASE[2] * scale[1], BASE[3] + shift[1],
                    BASE[4] * scale[2], BASE[5] + shift[2],
                ]), "coarse"


def calibrate_subject(subject: int, args) -> dict:
    global SUBJECT
    SUBJECT = subject
    print(f"\n=== Subject {subject} ===", flush=True)
    coordinates, neuron_region = rebuild_units(subject)
    blocks, labels = load_endpoint_blocks(args.endpoint_label_rule)
    fcs = fcs_vector()
    tree = cKDTree(coordinates.astype(float))
    region_sizes = np.bincount(neuron_region, minlength=N_REGIONS)
    rng = np.random.default_rng(args.seed + subject)
    rows = []
    parameter_names = FULL_AFFINE_NAMES if args.full_affine else RESTRICTED_NAMES
    for params, stage in candidate_parameters(rng, args.coarse, args.refine, args.full_affine):
        rho, p, n, nonempty, _ = evaluate(params, blocks, labels, tree, neuron_region, region_sizes, fcs)
        rows.append({"stage": stage, "rho_fcs_sc_strength": rho, "p": p, "n_regions": n, "nonempty_endpoints": nonempty, **dict(zip(parameter_names, params))})
        print(f"{stage}: rho={rho:.4f}, mapped_endpoints={nonempty}", flush=True)
        gc.collect()
    coarse = pd.DataFrame(rows)
    initial_best = coarse.loc[coarse["rho_fcs_sc_strength"].idxmax(), list(parameter_names)].to_numpy(float)
    for params, stage in candidate_parameters(rng, 0, args.refine, args.full_affine, initial_best):
        rho, p, n, nonempty, _ = evaluate(params, blocks, labels, tree, neuron_region, region_sizes, fcs)
        rows.append({"stage": stage, "rho_fcs_sc_strength": rho, "p": p, "n_regions": n, "nonempty_endpoints": nonempty, **dict(zip(parameter_names, params))})
        print(f"{stage}: rho={rho:.4f}, mapped_endpoints={nonempty}", flush=True)
        gc.collect()
    result = pd.DataFrame(rows).sort_values("rho_fcs_sc_strength", ascending=False).reset_index(drop=True)
    best = result.iloc[0]
    best_params = best[list(parameter_names)].to_numpy(float)
    strength, _ = regional_sc_strength(best_params, blocks, labels, tree, neuron_region, region_sizes)
    baseline_params = BASE_FULL_AFFINE if args.full_affine else BASE
    baseline_strength, _ = regional_sc_strength(baseline_params, blocks, labels, tree, neuron_region, region_sizes)
    regional = pd.DataFrame({"root_area_id": np.arange(N_REGIONS), f"FCS_subject{subject}": fcs, "endpoint_sc_strength": strength})
    for name, value in zip(parameter_names, best_params):
        regional[f"best_{name}"] = value
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "full_affine" if args.full_affine else "restricted_affine"
    if args.endpoint_label_rule != "direct_soma_distance":
        tag = f"{tag}_{args.endpoint_label_rule}"
    result.to_csv(OUT_DIR / f"subject{subject}_fcs_transform_search_{tag}.csv", index=False)
    regional.to_csv(OUT_DIR / f"subject{subject}_fcs_transform_best_region_strength_{tag}.csv", index=False)
    figure_path = save_fcs_sc_diagnostic(fcs, baseline_strength, strength, tag)
    fc_matrix, region_names = subject_regional_fc_matrix(subject)
    sc_density = regional_endpoint_density(best_params, blocks, labels, tree, neuron_region, region_sizes)
    matched_regions = np.flatnonzero(np.isfinite(fcs) & np.isfinite(strength) & np.isfinite(np.nanmean(fc_matrix, axis=1)))
    matrix_path = save_fc_sc_matrices(fc_matrix, sc_density, matched_regions, region_names, tag)
    summary = {"subject": subject, "tag": tag, **best.to_dict(), "scatter_path": str(figure_path), "matrix_path": str(matrix_path)}
    print(best.to_string())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=int, default=24)
    parser.add_argument("--refine", type=int, default=16)
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--full-affine", action="store_true")
    parser.add_argument("--subjects", type=int, nargs="+", default=[12])
    parser.add_argument(
        "--endpoint-label-rule",
        choices=("direct_soma_distance", "skeleton_path_kmeans"),
        default="direct_soma_distance",
        help="Endpoint source/target class rule used during FCS calibration.",
    )
    args = parser.parse_args()
    summaries = [calibrate_subject(subject, args) for subject in args.subjects]
    tag = "full_affine" if args.full_affine else "restricted_affine"
    if args.endpoint_label_rule != "direct_soma_distance":
        tag = f"{tag}_{args.endpoint_label_rule}"
    summary_path = OUT_DIR / f"all_subjects_fcs_transform_summary_{tag}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"saved={summary_path}")


if __name__ == "__main__":
    main()
