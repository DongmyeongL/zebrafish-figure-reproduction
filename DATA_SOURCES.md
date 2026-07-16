# Data inventory and provenance

All files under `derived_data/` are frozen analysis-level inputs copied from the
curated `final_figure_pack_1` workflow. They contain no imputed replacement for
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
