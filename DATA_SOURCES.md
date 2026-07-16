# Data inventory and provenance

All files under `derived_data/` are frozen analysis-level inputs exported from
the curated manuscript workflow. They contain no imputed replacement for
missing raw observations unless explicitly encoded by the original analysis.

| Data group | Contents | Upstream boundary |
|---|---|---|
| `derived_data/figure9/` | Recording-by-region FCV, FCS, FC reconfiguration, and TE measures | Derived from zebrafish calcium recordings and saved functional units |
| `derived_data/figure12/` | Subject-by-region DCA, OO fraction, modularity, and out/in measures | Derived from cell-level directed SC |
| `derived_data/common/` | Canonical 42-region set and matched spontaneous/stimulus plotting tables | Final matched region-level observations |
| `derived_data/figure_stimulus/` | Subject-condition-region raw and standardized FCV/FCS | Derived from stimulus-period calcium activity |
| `derived_data/invertebrates/` | Matched node/region FCV and directed structural summaries | Derived from published worm and fly datasets |
| `derived_data/figure13/` | Dense layer-model summaries and potential curves | Model outputs; model implementation is included |
| `derived_data/figure_supply_13/` | TE null, leave-one-animal-out, and example-pair summaries | Derived from zebrafish functional-unit traces |
| `raw_data/figure_supply_1/` | Compact histogram and fitted distance-bin parameters | Replaces large pair-level distance/FC arrays for plotting only |
| `raw_data/figure_supply_5/` | Fixed network diagrams and compact null statistics | Diagram assets are not regenerated in this release |
| `raw_data/figure_supply_15/` | Representative empirical/model traces and Figure 13 panel inputs | Frozen trace and simulation outputs |
| `raw_data/figure13/` | Whole-brain perturbation observations | Large simulation itself is not rerun |

The term `raw_data` is retained in a few paths for compatibility with the
curated plotting scripts. In this public bundle, those files are compact figure
inputs or fixed assets, not the primary experimental raw datasets.

## Raw-to-derived availability

The executable processing paths are summarized below. Exact filenames and
schemas are listed in `RAW_DATA.md` and `config/datasets.csv`.

| Target | Public processing code | Input availability |
|---|---|---|
| Figure 9 functional measures | `data_processing_code/figure9/` | Seven compact functional-unit trace files bundled |
| Figure 12 structural measures | `data_processing_code/figure12/` | External prepared subject bundles or compact SC files required |
| Stimulus FCV/FCS | `data_processing_code/figure_stimulus/` | External prepared subject bundles required |
| *C. elegans* metrics | `data_processing_code/invertebrates/01_compute_celegans_metrics.py` | Annotations bundled; recordings and connectome export external |
| *Drosophila* metrics | `data_processing_code/invertebrates/02_compute_drosophila_metrics.py` | Prepared Branson999 and FlyWire exports external |
| Four-layer model | `data_processing_code/figure13/` | Code and parameters bundled; full run is computationally intensive |
| Supply 1 compact input | `data_processing_code/figure_supply_1/` | Optional external pair-level arrays required |
| Supply 13 TE controls | `data_processing_code/figure_supply_13/` | Uses bundled Figure 9 traces; full surrogate run is intensive |

Provider records do not always expose files in the prepared schema consumed by
these scripts. Accordingly, DOI links document scientific provenance, while a
separate archival deposit should be used for prepared inputs that may legally
be redistributed. No script silently downloads or substitutes missing data.
