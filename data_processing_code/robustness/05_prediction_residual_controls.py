#!/usr/bin/env python3
"""Test whether SC-model residual magnitude tracks measurement proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, save_table


STRUCTURAL = ["Hard_OO_fraction", "FU_DCApost", "FU_DCApre", "Reciprocity", "LogOutIn"]
PROXIES = ["FCV_SEM", "calcium_variance_sem", "log10_mean_neuron_count"]


def main() -> None:
    frame = pd.read_csv(DATA / "prediction_residual_region_input_no_ob.csv").dropna(
        subset=["EdgeStdFCV", *STRUCTURAL, *PROXIES]
    )
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    prediction = cross_val_predict(
        model, frame[STRUCTURAL].to_numpy(float), frame["EdgeStdFCV"].to_numpy(float), cv=cv
    )
    residual = frame["EdgeStdFCV"].to_numpy(float) - prediction
    rows = []
    for proxy in PROXIES:
        values = frame[proxy].to_numpy(float)
        absolute = pearsonr(values, np.abs(residual))
        squared = pearsonr(values, residual**2)
        rows.append({
            "proxy": proxy,
            "n_regions": len(frame),
            "abs_residual_r": absolute.statistic,
            "abs_residual_p": absolute.pvalue,
            "squared_residual_r": squared.statistic,
            "squared_residual_p": squared.pvalue,
            "excluded_region": "OB",
            "cross_validation": "5-fold shuffled KFold, random_state=0",
        })
    summary = pd.DataFrame(rows)
    path = save_table(summary, "prediction_residual_proxy_summary_no_ob.csv")
    print(summary.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
