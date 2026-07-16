"""Shared FCV and directed-embedding calculations for invertebrates."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt


def zscore_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out
    sd = np.std(values[finite])
    out[finite] = 0.0 if sd <= 1e-12 else (values[finite] - np.mean(values[finite])) / sd
    return out


def preprocess_traces(traces: np.ndarray, sampling_rate_hz: float, highpass_hz: float) -> np.ndarray:
    """Linear detrend, high-pass filter, and row-standardize traces."""
    traces = detrend(np.asarray(traces, dtype=float), axis=1, type="linear")
    if highpass_hz > 0:
        nyquist = 0.5 * sampling_rate_hz
        b, a = butter(2, highpass_hz / nyquist, btype="high")
        traces = filtfilt(b, a, traces, axis=1)
    mean = np.nanmean(traces, axis=1, keepdims=True)
    sd = np.nanstd(traces, axis=1, keepdims=True)
    sd[sd <= 1e-12] = np.nan
    return np.nan_to_num((traces - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def edge_std_fcv(traces: np.ndarray, window: int, step: int) -> np.ndarray:
    """Mean edge-wise temporal standard deviation of sliding-window FC."""
    n = traces.shape[0]
    corr_sum = np.zeros((n, n), dtype=float)
    corr2_sum = np.zeros((n, n), dtype=float)
    n_windows = 0
    for start in range(0, traces.shape[1] - window + 1, step):
        corr = np.nan_to_num(np.corrcoef(traces[:, start : start + window]), nan=0.0)
        np.fill_diagonal(corr, 0.0)
        corr_sum += corr
        corr2_sum += corr * corr
        n_windows += 1
    if n_windows < 3:
        return np.full(n, np.nan)
    mean = corr_sum / n_windows
    edge_sd = np.sqrt(np.maximum(corr2_sum / n_windows - mean * mean, 0.0))
    np.fill_diagonal(edge_sd, 0.0)
    return edge_sd.sum(axis=1) / max(n - 1, 1)


def rank1_dca_dense(sc: np.ndarray, max_iter: int = 1000, tol: float = 1e-10):
    """Return non-negative rank-1 coreness and DCA for a dense directed SC."""
    sc = np.asarray(sc, dtype=float)
    n = sc.shape[0]
    c_out = np.full(n, 1.0 / np.sqrt(n))
    c_in = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(max_iter):
        next_out = np.maximum(sc @ c_in, 0.0)
        out_norm = np.linalg.norm(next_out)
        if out_norm > 1e-12:
            next_out /= out_norm
        next_in = np.maximum(sc.T @ next_out, 0.0)
        in_norm = np.linalg.norm(next_in)
        if in_norm > 1e-12:
            next_in /= in_norm
        if max(np.linalg.norm(next_out - c_out), np.linalg.norm(next_in - c_in)) < tol:
            c_out, c_in = next_out, next_in
            break
        c_out, c_in = next_out, next_in
    return c_out, c_in, c_out - c_in


def rank1_dca_sparse(sc, max_iter: int = 300, tol: float = 1e-9):
    """Sparse equivalent used for large FlyWire within-region subnetworks."""
    n = sc.shape[0]
    c_out = np.full(n, 1.0 / np.sqrt(n))
    c_in = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(max_iter):
        next_out = np.asarray(sc @ c_in).ravel()
        norm = np.linalg.norm(next_out)
        if norm <= 0:
            break
        next_out /= norm
        next_in = np.asarray(sc.T @ next_out).ravel()
        norm = np.linalg.norm(next_in)
        if norm <= 0:
            break
        next_in /= norm
        delta = max(np.linalg.norm(next_out - c_out), np.linalg.norm(next_in - c_in))
        c_out, c_in = next_out, next_in
        if delta < tol:
            break
    return c_out, c_in, c_out - c_in
