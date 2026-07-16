# Figure Supply 15 inputs

This directory contains the compact simulation traces and Figure 13 summaries
required to reproduce `figure_supply_15.png` without accessing the legacy
figure pack or the large original simulation files.

- `figure_supply_15_rsp_rmos5_trace.npz`: compact Base, Null-In, and Null-Out
  P/SP-associated traces.
- `figure_supply_15_layer_trace_cache.npz`: cached layer-1 and layer-4 traces
  for the three asymmetry values shown in the figure.
- `figure13_inputs/`: layer-model summary, compact whole-brain null-model
  summaries, and effective-potential tables used by the two upper panels.

Checksums for all immutable inputs are recorded in `source_manifest.csv`.
The dense 21-epsilon, four-layer energy-well-width table is recomputed from the
Figure 13 model and saved under `derived_data/figure13/`.
