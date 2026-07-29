# SI robustness analyses

These scripts reproduce the numerical summaries used in the SI robustness
sections for the primary zebrafish structural-connectivity reconstruction:
skeleton-path KMeans endpoint classification, nearest-endpoint soma assignment,
an endpoint-matching radius of 12 imaging-coordinate units, and a target size of
400 neurons per anatomical unit.

Run all analyses from the repository root with:

```bash
python data_processing_code/robustness/run_all.py
```

The scripts write tables to `statistics/robustness/`.

| Script | SI control | Released input |
|---|---|---|
| `01_reconstruction_parameter_sensitivity.py` | anatomical-unit size and endpoint radius | correlations for the 30 reconstruction settings |
| `02_morphology_subsampling.py` | random subsampling of reconstructed morphologies | 200 iteration-level results per incomplete fraction |
| `03_strength_preserving_topology_null.py` | weighted in/out-strength-preserving topology null | 250 null correlations and observed correlations |
| `04_division_and_subject_controls.py` | division adjustment and subject-specific meta-analysis | region and subject-by-region analysis tables |
| `05_prediction_residual_controls.py` | measurement and sampling proxies | 41-region predictor and proxy table, excluding OB |

## Reconstruction-dependent inputs

The parameter-grid, morphology-subsampling, and topology-null inputs were
generated from the full soma and reconstructed-morphology endpoint data. Those
raw coordinate arrays are not distributed in this lightweight repository.
Their released iteration-level tables preserve the sampling unit, random-null
realizations, and correlation output needed to reproduce every reported SI
summary without distributing the large source arrays. The primary SC pipeline
in `data_processing_code/figure12/` documents the shared endpoint
classification, affine alignment, nearest-endpoint assignment, and regional
aggregation rules.

The strength-preserving null retained each anatomical unit's weighted incoming
and outgoing strength, prohibited within-region edges, and randomized weight
allocation over admissible inter-regional pairs. The released file also
contains exploratory root-pair-preserving realizations; the public analysis
script explicitly selects only `null_type == "strength_preserving"`, matching
the SI analysis.
