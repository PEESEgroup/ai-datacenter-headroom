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
python -B run_paired_realization.py
python -B -m unittest -v test_paired_realization.py
python -B run_campus_retention.py
python -B -m unittest -v test_campus_retention.py
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

## Matched ERCOT growth and effective response

`run_paired_realization.py` connects the retained reference-growth pairs to provisional-response realization without changing the regional model. For each timing rule and nominal association setting, it holds the association fixed, retains published ordinary response, and applies one common realization fraction to the provisional credit remaining in each world and every year. One resource fraction selects the annual anticipated-to-prospective native contribution difference. Zero provisional realization therefore differs from current-only response.

`PAIRED_REALIZATION_PROTOCOL.json` specifies the controls; its input manifest verifies retained inputs. `paired_realization_results` supplies full-precision annual balances, complete-path summaries and response thresholds. With anticipated resources and the minimum linear-timing association, the background requires 62.5% realization and reference growth 77.2%. At 70%, only the candidate path falls below requirement, adding 2.21 GW in 2034 and 4.09 GW in 2035. A 25% share of resource uplift lowers the thresholds to 10.9% and 34.1%. The three association settings give candidate thresholds of 77.2%, 80.6% and 84.3% at anticipated resources; these are conditional sensitivity results, not uncertainty probabilities. Ten new tests cover direct reconstruction, both worlds, nominal pool bounds, threshold necessity and the distinction from current-only response. The four preceding test suites contain 41 focused tests. This entry adds no main figure or model refit.

## Current campus-retention sensitivity

`run_campus_retention.py` applies the current paired accounting to the four
Matrix/Frontier keep/remove combinations. It uses the existing reviewed MW
weights, removes the selected campus weights and renormalizes shares within the
seven regions. National growth, footprint capture, peak mapping, regional
resources and the source-growth path stay fixed. One weight multiplier applies
to the whole candidate path. The associated-response fraction is selected from
the full-horizon feasible interval, not the earlier zero-association ERCOT default.

`pool_inputs/campus_retention_weights.csv` gives the regional and seven-region
weight totals; `CAMPUS_RETENTION_INPUT_MANIFEST.json` identifies the 267-record
source and hashes all calculation inputs. `pool_results/remaining_campus_retention_cases.csv`
reports the 32 primary endpoint cases. The companion annual and complete-path
files retain 960 annual rows and 96 paths across all three association settings.
Both-retained cases reproduce the current displayed pairs. The anticipated,
published-response ERCOT endpoint balances span +13.01 to +13.47 GW and all have
zero added requirement. The baseline table remains archived; it is not used for
the current ERCOT sensitivity. Eight regression tests cover these results,
unaffected MISO/current-only values, annual pool bounds and identical reruns,
bringing the extension total to 49 tests. No main model is refitted or rerun.

## Capacity correction

The standardized candidate paths include the 24 October capacity-attribute correction. See `../reproduction_20261009/CAPACITY_CORRECTION.md` for source decisions and the narrow correction scope. The extension remains self-contained. To refresh its inputs from a rebuilt adjacent baseline, run `python refresh_allocation_inputs.py --stage candidates`, `python run_pool_consistency.py`, then `python refresh_allocation_inputs.py --stage conditions` before the remaining entries.
