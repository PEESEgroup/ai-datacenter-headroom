# Current output map: capacity attributes and ERCOT load pools

Use the retained baseline and the planning-conditions extension together. The
extension is self-contained for current paired calculations: it does not require
running or modifying the large baseline first.

| Result | Current calculation and output |
| --- | --- |
| Seven-region official-demand annual balances | `reproduction_20261009/run_reproduction.py`; unchanged |
| Current MISO/ERCOT paired results | `planning_conditions_20261023/run_pool_consistency.py`, `pool_results/annual_pairs.csv.gz` and `complete_paths.csv.gz` |
| Matched ERCOT growth and response realization | `planning_conditions_20261023/run_paired_realization.py`, `paired_realization_results/annual_balances.csv`, `response_thresholds.csv` and `complete_paths.csv` |
| Current primary ranges and complete-path limits | `pool_results/primary_annual.csv.gz`, `primary_complete_paths.csv`, `annual_core_ranges.csv` |
| Current timing, lower-endpoint, spatial-scope and assignment sensitivities | `pool_results/timing_*`, `lower_*`, `scope_*`, `geographic_assignment_results.csv` |
| Current Matrix/Frontier retention sensitivity | `planning_conditions_20261023/run_campus_retention.py`; `pool_results/remaining_campus_retention_cases.csv`, `campus_retention_annual.csv` and `campus_retention_complete_paths.csv` |
| ERCOT association bounds and legacy failing cases | `pool_results/annual_pool_bounds.csv`, `legacy_association_check.csv` |
| Local MISO binding-year check | `pool_results/miso_2034_resource_perturbation.csv` |
| Joint annual boundaries and PJM vintages | `planning_conditions_20261023/run_analysis.py`, `results/`; MISO boundaries updated, official-demand ERCOT and PJM results unchanged |
| PJM operating loads and ERCOT same-table scale ratios | `planning_conditions_20261023/run_empirical.py`, `empirical_results/`; unchanged |
| Paired-result and joint-condition diagnostic (not the main Fig. 4c,d heatmaps) | `planning_conditions_20261023/plot_pool_conditions.py`, `figures/Fig4_pool_consistent.png` |
| Supplementary Fig. 12 joint boundaries | `planning_conditions_20261023/plot_conditions.py`, `figures/` |
| Physical-hour PV, history and other supporting results | Retained baseline and its follow-up; unchanged |

ERCOT published-response paired quantities in the earlier baseline's `results`,
`source_data`, `plotting_tables`, `display_tables`, follow-up paired extracts and
diagnostic summaries reproduce the preceding zero/half/full-association version.
They are archived numerical history, **not the current manuscript's defaults**.
This also applies to the ERCOT published-response rows of the baseline's
`default_weight_update/remaining_campus_retention_cases.csv`. Use the extension's
same-named table above for the current retention sensitivity.
The pool-corrected extension supersedes all of those affected ERCOT quantities;
it does not supersede the baseline's official-demand balances or PV calculations.

The actual associated fraction is `linked_response_fraction`. The separate
`association_grid_position` indexes the lower/midpoint/upper feasible settings;
zero grid position does not mean zero ERCOT association. The extension README
explains the source time mapping, upper/lower bounds and necessary-condition scope.

The 24 October capacity correction excludes Joule, GW Ranch and Ridgeline supply
attributes from default MW weights, retaining their site-count evidence. New Era's
separately advertised campus scale remains eligible. Both the national denominator
and regional weights are rebuilt. See `reproduction_20261009/CAPACITY_CORRECTION.md`.
The extension includes corrected standardized inputs and can run independently.
To regenerate its inputs from a fresh baseline calculation, use
`refresh_allocation_inputs.py --stage candidates`, run the pool entry, then use
`refresh_allocation_inputs.py --stage conditions` before the other entries.
