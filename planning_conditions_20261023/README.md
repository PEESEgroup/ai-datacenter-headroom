# Annual planning conditions and PJM load-evidence extension

This extension uses retained standardized inputs, official PJM forecast workbooks and PJM's operational-load accuracy report. It also provides the current paired comparisons with ERCOT load-pool consistency. It does not refit or change the official-demand regional model.

## Run

```sh
python -B run_analysis.py
python -B -m unittest -v test_analysis.py
python -B plot_conditions.py
python -B run_empirical.py
python -B -m unittest -v test_analysis.py test_empirical.py
python -B plot_empirical.py
python -B run_pool_consistency.py
python -B -m unittest -v test_pool_consistency.py
python -B plot_pool_conditions.py
```

Python, NumPy, pandas, openpyxl, PyMuPDF and Matplotlib are required. No network connection is needed to run the packaged calculations. Third-party source workbooks and reports retain provider terms; author-written analysis code follows the baseline repository's MIT license.

## Inputs and conventions

- `inputs/native_inputs.csv`: unchanged NERC 2025 native demand, response, resource and reference-margin inputs from the baseline package.
- `inputs/miso_display_timing_paths.csv`: the current displayed reference candidate and its four previously specified source-arrival rules, with anticipated resources, published response and zero response association. These are existing results, not a new fit.
- `inputs/source_growth_and_response_anchors.csv`: the source-audited official DC-growth and ERCOT response quantities.
- `sources/2025-load-report-tables.xlsx` and `sources/2026-load-report-tables.xlsx`: official PJM workbooks. Table B1's unique PJM RTO row supplies each summer year, with selected row/column coordinates retained in the output.
- `sources/2025-load-report.pdf` and `sources/2026-load-report.pdf`: provider definitions and planning context.

`EXPERIMENT_PROTOCOL.json` specifies the controls, range and interpretation. `results/source_register.json` records the downloaded source URLs and hashes.

## Questions

**MISO growth and resources.** For a matched no-new-growth background, the balance is `B0_A - lambda * P_DC + u * (R_P - R_A)`. One growth factor `lambda` and one native-resource fraction `u` apply to the entire ten-year path. The all-year growth boundary is the smallest annual ratio, not a choice of different controls in each year. Four pre-existing source-arrival rules give a deterministic timing range, not a probability interval.

**ERCOT resources and response.** Official demand and published ordinary response remain fixed. Provisional large-load curtailment is initially excluded, then credited once through a common realization fraction. One resource fraction also applies to every year. This background differs explicitly from the MISO candidate-growth comparison.

**PJM forecast vintages.** All ten 2025-vintage summer peaks must match the retained native gross-demand inputs before the 2026-vintage differences are applied. Only gross demand changes: resources, response and each year's reference margin stay fixed. The result is a one-factor vintage stress, not an updated full PJM adequacy forecast, a DC-only forecast change, or a test against realized future demand.

**ERCOT magnitude.** The 2035 required effective provisional reduction is compared with gross demand and the source's 25-GW new-DC increment. The subtraction assumes a common peak-demand basis and full reducibility of that increment. It does not identify the remaining participating loads, imply that all data centers have only 25 GW of demand, or establish delivery.

## Operational load evidence update

`run_empirical.py` adds three source-based comparisons without changing the baseline inputs:

- Exact July 2025 forecast/actual pairs from PJM's 2026 accuracy workbook. Only AEP, COMED, DOM and PS have both parent-zone records. Their covered-zone sum is not a coincident system peak. Annual workbook values are broader large-load adjustments, not automatically DC-only values.
- DC-only monthly peaks from the report's top charts. Vector markers and forecast polylines are recovered with axis/date calibration; one PDF point is allowed per ordinate. The primary common window is January-August 2025. Monthly values are approximate and are never substituted for exact workbook quantities. Parent zones and RTO/subzone values are not added together. Missing or confidential data are not zero.
- The same NERC summer table's 69.4639-GW large-load total, 58.3656-GW provisional curtailment and required 51.1541-GW reduction. This is the revised paper's primary scale comparison: 73.6% of total large loads is required, while the source credit is 84.0%. The earlier conditional 25-GW increment comparison remains an auxiliary output of `run_analysis.py`.

Original sources are in `sources/accuracy`. Full source-cell records, coverage, monthly coordinates and comparisons are in `empirical_results`. The 2024/2025 forecast comparison uses identical target months. No historical error is extrapolated into 2035, no bias correction is fitted, and no main-model parameter is changed. Nineteen focused tests cover the original boundaries and the additional evidence processing.

## Outputs

Full-precision CSVs report the frontiers, exact binding-year segments, selected resource/growth conditions, all 80 PJM annual condition-vintage rows and the eight complete-path summaries. JSON reports the ERCOT magnitude comparison and calculation checks. Rounded publication tables are downstream displays; the numerical results retain full precision. No scenario counts are interpreted as probabilities, no reliability risk is calculated, and no regional native-capacity totals are added together.

## Current ERCOT paired outputs

`run_pool_consistency.py` is the current paired-calculation entry. It reads compact standardized inputs in `pool_inputs` and writes `pool_results`. It supersedes **ERCOT published-response paired** balances, pressure, growth limits, lower-demand and geographic sensitivities in the earlier `reproduction_20261009` outputs. Those older outputs reproduce the earlier zero/half/full-association version, not an alternative current default. MISO results, ERCOT current-only response and all official-demand regional balances remain unchanged.

NERC printed p. 131 labels DC additions as end-of-year and its large-load/curtailment table as summer. The scenario maps each DC anchor to the corresponding summer planning year, within the total large-load pool, without claiming an observed monthly profile. It removes associated response whenever retaining it would exceed remaining large-load demand. Both background and candidate worlds must pass this necessary bound. Under linear source timing the minimum common association is 66.1330%, set by 2030; the 2035-only minimum, 55.6068%, is insufficient for a common ten-year setting. Other source timing rules recompute the bound.

`association_grid_position` is 0, 0.5 or 1 within the permitted association interval; it is **not** the associated fraction itself. `linked_response_fraction` is the actual fraction of the available source-associated response bound. MISO's interval remains [0,1]. ERCOT published-response pairs use [chi_min,1]; current-only excludes provisional association. `official_linked_response_GW` and `candidate_linked_response_GW` contain the quantities actually used. `background_pool_slack_GW`, `candidate_pool_slack_GW` and `fixed_budget_pool_slack_GW` are nonnegative in applicable current ERCOT pairs; NaN means that provisional pool check is not applicable, not missing demand. The two 2025 inputs are comparison anchors, not annual outcome rows.

`annual_pairs.csv.gz` retains 388,800 annual outcomes and 38,880 complete paths. The primary grid has 51,840 points, 648 paths per region/resource/response condition. Timing, lower-endpoint and scope outputs retain all existing cases under the revised rule. `legacy_association_check.csv` explicitly identifies the old failing backgrounds. `miso_2034_resource_perturbation.csv` reports a resource-only local break-even check: +0.5034376 native GW in 2034 removes the displayed endpoint omission. This is not a refit or an estimated uncertainty distribution.

`plot_pool_conditions.py` plots current paired results together with the unchanged joint frontiers from `run_analysis.py`. Its lower panels show joint resource conditions, not the annual growth/response heatmaps retained in main Fig. 4c,d. `plot_conditions.py` reproduces the joint boundaries shown in Supplementary Fig. 12. MISO uses a matched candidate background; ERCOT uses official demand. The upper scales of the joint-condition panels report 2035 native resource GW, not nameplate capacity. Twelve additional tests cover the pool correction, alongside the nineteen retained tests.
