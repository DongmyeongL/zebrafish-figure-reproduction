#!/usr/bin/env python3
"""Compute Figure 9 NetTE measures directly from calcium traces.

The calculation uses the same saved spatial functional units as the FC
measures. Transfer entropy is estimated after 0.03-Hz high-pass filtering and
temporal standardization, then functional-unit values are averaged within each
anatomical root area.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from statsmodels.stats.multitest import multipletests

from common import CONFIG, DERIVED_DIR, ensure_output_dirs


def highpass_filter(traces: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    nyquist = 0.5 * rate_hz
    b, a = butter(2, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, traces, axis=1)


def zscore_rows(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=float)
    mean = np.nanmean(traces, axis=1, keepdims=True)
    sd = np.nanstd(traces, axis=1, keepdims=True)
    sd[sd <= 1e-12] = np.nan
    return np.nan_to_num((traces - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def discretize_quantiles(trace: np.ndarray, bins: int) -> np.ndarray:
    if np.nanstd(trace) <= 1e-12:
        return np.full(trace.shape, -1, dtype=np.int16)
    edges = np.unique(np.nanquantile(trace, np.linspace(0, 1, bins + 1)[1:-1]))
    if edges.size == 0:
        return np.full(trace.shape, -1, dtype=np.int16)
    return np.digitize(trace, edges, right=False).astype(np.int16)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    probabilities = counts[counts > 0].astype(float) / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def net_te_from_traces(traces: np.ndarray, bins: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise NetTE and each unit's outgoing-minus-incoming drive."""
    traces = zscore_rows(traces)
    discrete = np.vstack([discretize_quantiles(trace, bins) for trace in traces])
    n_units, n_time = discrete.shape
    tmax = n_time - lag
    te = np.full((n_units, n_units), np.nan, dtype=float)
    if tmax <= 10:
        return te, np.full(n_units, np.nan)

    valid = np.zeros(n_units, dtype=bool)
    entropy_y = np.full(n_units, np.nan)
    entropy_yn_y = np.full(n_units, np.nan)
    for target in range(n_units):
        y = discrete[target, :tmax]
        y_next = discrete[target, lag : lag + tmax]
        ok = (y >= 0) & (y_next >= 0)
        if ok.sum() > 10:
            valid[target] = True
            entropy_y[target] = entropy_from_counts(np.bincount(y[ok], minlength=bins))
            entropy_yn_y[target] = entropy_from_counts(
                np.bincount(y_next[ok] * bins + y[ok], minlength=bins * bins)
            )

    for source in range(n_units):
        if not valid[source]:
            continue
        x = discrete[source, :tmax]
        for target in range(n_units):
            if source == target or not valid[target]:
                continue
            y = discrete[target, :tmax]
            y_next = discrete[target, lag : lag + tmax]
            ok = (x >= 0) & (y >= 0) & (y_next >= 0)
            if ok.sum() <= 10:
                continue
            counts_y_x = np.bincount(y[ok] * bins + x[ok], minlength=bins * bins)
            counts_yn_y_x = np.bincount(
                (y_next[ok] * bins + y[ok]) * bins + x[ok],
                minlength=bins * bins * bins,
            )
            te[source, target] = (
                entropy_yn_y[target]
                - entropy_y[target]
                - entropy_from_counts(counts_yn_y_x)
                + entropy_from_counts(counts_y_x)
            )

    net_te = te - te.T
    np.fill_diagonal(net_te, np.nan)
    return net_te, np.nanmean(net_te, axis=1)


def mean_fc_from_traces(traces: np.ndarray, window: int, step: int) -> tuple[np.ndarray, int]:
    traces = zscore_rows(traces)
    n_units, n_time = traces.shape
    if n_time < window:
        return np.full((n_units, n_units), np.nan), 0
    fc_sum = np.zeros((n_units, n_units), dtype=float)
    n_windows = 0
    for start in range(0, n_time - window + 1, step):
        local = zscore_rows(traces[:, start : start + window])
        corr = np.clip((local @ local.T) / local.shape[1], -1.0, 1.0)
        np.fill_diagonal(corr, np.nan)
        fc_sum += np.nan_to_num(corr, nan=0.0)
        n_windows += 1
    mean_fc = fc_sum / max(n_windows, 1)
    np.fill_diagonal(mean_fc, np.nan)
    return mean_fc, n_windows


def pearson_pvalues(correlation: np.ndarray, n_time: int) -> np.ndarray:
    pvalues = np.full(correlation.shape, np.nan, dtype=float)
    finite = np.isfinite(correlation) & (np.abs(correlation) < 1)
    if n_time <= 4:
        return pvalues
    zstat = np.arctanh(np.clip(correlation[finite], -0.999999, 0.999999)) * np.sqrt(n_time - 3)
    pvalues[finite] = 2.0 * stats.norm.sf(np.abs(zstat))
    return pvalues


def neighbor_net_te(fc: np.ndarray, net_drive: np.ndarray, n_time: int, fdr_q: float) -> np.ndarray:
    pvalues = pearson_pvalues(fc, n_time)
    diagonal = np.eye(fc.shape[0], dtype=bool)
    pvalues[diagonal] = np.nan
    finite = np.isfinite(pvalues)
    neighbors = np.zeros(fc.shape, dtype=bool)
    if finite.any():
        reject = np.zeros(fc.shape, dtype=bool)
        reject[finite] = multipletests(pvalues[finite], alpha=fdr_q, method="fdr_bh")[0]
        neighbors = reject | reject.T
        neighbors[diagonal] = False

    output = np.full(fc.shape[0], np.nan, dtype=float)
    for unit in range(fc.shape[0]):
        indices = np.flatnonzero(neighbors[unit] & np.isfinite(net_drive))
        if indices.size:
            output[unit] = float(np.nanmean(net_drive[indices]))
    return output


def main() -> None:
    ensure_output_dirs()
    bins = int(CONFIG["te_bins"])
    lag = int(CONFIG["te_lag"])
    fc_window = int(CONFIG["neighbor_fc_window_frames"])
    fc_step = int(CONFIG["neighbor_fc_step_frames"])
    fdr_q = float(CONFIG["neighbor_fdr_q"])
    unit_rows = []
    qc_rows = []

    for path in sorted((DERIVED_DIR / "functional_unit_traces").glob("*_raw_cluster_traces.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        rate = float(raw["sampling_rate_hz"])
        traces = highpass_filter(np.asarray(raw["traces"], dtype=float), rate, float(CONFIG["highpass_hz"]))
        traces = zscore_rows(traces)

        net_te, net_drive = net_te_from_traces(traces, bins=bins, lag=lag)
        mean_fc, n_fc_windows = mean_fc_from_traces(traces, window=fc_window, step=fc_step)
        neighbor_drive = neighbor_net_te(mean_fc, net_drive, traces.shape[1], fdr_q=fdr_q)

        subject_units = pd.DataFrame(
            {
                "recording_id": str(raw["recording_id"]),
                "cluster_id": list(raw["cluster_ids"]),
                "root_area_id": list(raw["root_area_ids"]),
                "node": list(raw["root_area_names"]),
                "n_cells": list(raw["n_cells_per_cluster"]),
                "NetTE": net_drive,
                "NeighborNetTE": neighbor_drive,
            }
        )
        unit_rows.append(subject_units)
        qc_rows.append(
            {
                "recording_id": raw["recording_id"],
                "n_functional_units": traces.shape[0],
                "n_timepoints": traces.shape[1],
                "n_finite_net_te_edges": int(np.isfinite(net_te).sum()),
                "n_units_with_nette": int(np.isfinite(net_drive).sum()),
                "n_units_with_neighbor_nette": int(np.isfinite(neighbor_drive).sum()),
                "te_bins": bins,
                "te_lag": lag,
                "neighbor_fc_window": fc_window,
                "neighbor_fc_step": fc_step,
                "neighbor_fdr_q": fdr_q,
            }
        )
        print(
            f"{raw['recording_id']}: units={traces.shape[0]}, "
            f"NetTE={np.isfinite(net_drive).sum()}, NeighborNetTE={np.isfinite(neighbor_drive).sum()}",
            flush=True,
        )

    units = pd.concat(unit_rows, ignore_index=True)
    regions = (
        units.groupby(["recording_id", "root_area_id", "node"], as_index=False)
        .agg(
            n_clusters=("cluster_id", "size"),
            n_cells=("n_cells", "sum"),
            NetTE=("NetTE", "mean"),
            NeighborNetTE=("NeighborNetTE", "mean"),
        )
    )
    regions["te_source"] = "computed_from_spatial_unit_calcium"
    units.to_csv(DERIVED_DIR / "figure9_te_measures_functional_unit.csv", index=False)
    regions.to_csv(DERIVED_DIR / "figure9_te_measures_recording_region.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(DERIVED_DIR / "figure9_te_qc.csv", index=False)
    print(
        f"Saved {len(regions)} recording-region TE rows across "
        f"{regions['node'].nunique()} regions",
        flush=True,
    )


if __name__ == "__main__":
    main()
