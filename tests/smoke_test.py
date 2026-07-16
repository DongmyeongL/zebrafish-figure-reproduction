#!/usr/bin/env python3
"""Fast numerical smoke tests for the public analysis kernels."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
FIGURE9_CODE = ROOT / "data_processing_code" / "figure9"
FIGURE12_CODE = ROOT / "data_processing_code" / "figure12"
FIGURE13_CODE = ROOT / "data_processing_code" / "figure13"
for path in [FIGURE9_CODE, FIGURE12_CODE, FIGURE13_CODE]:
    sys.path.insert(0, str(path))

from importlib.util import module_from_spec, spec_from_file_location


def load(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fc = load(FIGURE9_CODE / "02_compute_fc_measures.py", "smoke_fc")
dca = load(FIGURE12_CODE / "02_compute_cell_dca.py", "smoke_dca")
layer = load(FIGURE13_CODE / "layer_linear_model.py", "smoke_layer")


def main() -> None:
    rng = np.random.default_rng(4)
    traces = rng.normal(size=(6, 120))
    measures, windows = fc.compute_measures(traces, window=20, step=5)
    assert windows == 21
    assert measures.shape == (6, 3)
    assert np.isfinite(measures.to_numpy()).all()

    matrix = sparse.csr_matrix(
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )
    c_out, c_in, _, _ = dca.rank1_dca(matrix, max_iter=300, tol=1e-9)
    assert np.isclose(np.linalg.norm(c_out), 1.0)
    assert np.isclose(np.linalg.norm(c_in), 1.0)

    jacobian, slices = layer.build_layer_jacobian(
        layer_sizes=[2, 3, 4, 5],
        gamma=12.7,
        w_intra=0.25,
        w_inter=1.85,
        intra_epsilon=0.0,
        epsilon=0.5,
        inter_epsilon_scales=[0.8, 0.6, 0.1],
        layer_decay_offsets=[0.0, 0.0, 0.0, 0.0],
    )
    assert jacobian.shape == (14, 14)
    assert len(slices) == 4
    assert np.max(np.real(np.linalg.eigvals(jacobian))) < 0
    print("Smoke tests passed: FC measures, rank-1 DCA, and layer Jacobian")


if __name__ == "__main__":
    main()
