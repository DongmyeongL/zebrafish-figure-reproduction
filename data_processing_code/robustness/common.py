"""Shared paths and statistics for the SI robustness analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


RELEASE = Path(__file__).resolve().parents[2]
DATA = RELEASE / "derived_data" / "robustness"
OUTPUT = RELEASE / "statistics" / "robustness"


def save_table(frame: pd.DataFrame, filename: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / filename
    frame.to_csv(path, index=False)
    return path


def residualize_by_group(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    dummies = pd.get_dummies(groups, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(values)), dummies))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def random_effects_summary(effects: pd.DataFrame) -> dict[str, float]:
    """DerSimonian-Laird random-effects summary on Fisher-z correlations."""
    z = effects["fisher_z"].to_numpy(float)
    variance = effects["fisher_variance"].to_numpy(float)
    fixed_weight = 1.0 / variance
    fixed_z = np.sum(fixed_weight * z) / np.sum(fixed_weight)
    q = np.sum(fixed_weight * (z - fixed_z) ** 2)
    df = len(z) - 1
    denominator = np.sum(fixed_weight) - np.sum(fixed_weight**2) / np.sum(fixed_weight)
    tau2 = max(0.0, (q - df) / denominator)
    weight = 1.0 / (variance + tau2)
    summary_z = np.sum(weight * z) / np.sum(weight)
    se = np.sqrt(1.0 / np.sum(weight))
    return {
        "n_subjects": len(z),
        "random_effect_r": np.tanh(summary_z),
        "random_effect_ci_low": np.tanh(summary_z - 1.96 * se),
        "random_effect_ci_high": np.tanh(summary_z + 1.96 * se),
        "random_effect_p": 2 * norm.sf(abs(summary_z / se)),
        "cochran_q": q,
        "heterogeneity_df": df,
        "heterogeneity_p": chi2.sf(q, df),
        "i2_percent": max(0.0, (q - df) / q) * 100 if q > 0 else 0.0,
        "tau2": tau2,
        "n_positive_subject_effects": int((effects["pearson_r"] > 0).sum()),
    }
