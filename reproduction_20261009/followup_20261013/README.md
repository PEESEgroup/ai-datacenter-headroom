# Targeted regional and PV follow-up

This entry reconciles regional inputs, checks the retained equal-label PV
fixtures, then rebuilds the source-corrected physical-hour PV grid and windows.
It extracts the current reviewed-capacity paired results and checks their annual
limits independently. It does not run the regional annual model or redraw figures.

## Run only the added checks

Use the Python environment in the parent README. From `reproduction_20261009`:

```sh
python -B followup_20261013/run_followup.py
```

No download, unpublished companion directory or additional Python dependencies
are required. A C compiler is required, as in the core package. Generated outputs
go to `followup_20261013/results/`, or the directory selected by `--output-root`.
`checks.json` reports `status: passed` only after comparing complete keys and
values with the retained results. The run does not modify core inputs or outputs.

## Data and results

| Comparison | Inputs | Recomputed output |
| --- | --- | --- |
| Regional input reconciliation | Core NERC inputs; `inputs/pjm_rrs_table4.csv`; `inputs/miso_oms_summer_components.csv` | `regional_input_reconciliation.csv`; `tables/regional_inputs.tex` (Table 69) |
| Native-ledger comparisons | Reconciliation table and the same PJM Table 4 inputs | `regional_native_comparison.csv`; `nerc_response_side_identity.csv`; `pjm_within_report_mapping.csv` |
| Source timing metadata | `inputs/power_header_snapshots.json`; `inputs/ceres_hourbox.json` | `power_export_header_check.csv` |
| Peak windows | Core standardized 2016-2025 weather, reported EIA demand and unchanged physics/dispatch | `pv_peak_windows_full.csv` (108 cases); `pv_peak_window_summary.csv`; `tables/pv_windows.tex` (Table 70) |
| Contribution evidence | Existing full-precision paired results and annual boundary tables | `contribution_paired_comparison.csv`; `contribution_conditions.json` |

`expected/` contains retained comparison fixtures, not inputs to PV dispatch.
Contribution evidence is extracted from existing core results, not estimated
again. Regional-report inputs are factual transcriptions; source dates, pages,
URLs and archived-document hashes are in `SOURCE_MANIFEST.json`.
`inputs/regional_study_comparison.csv` also preserves the three-row display
underlying Table 68 and Response Table 2. Original PDFs and bulk weather archives
are not included or needed by this entry point.

## Interpretation

The reconciliation preserves ICAP, UCAP and seasonal accredited-capacity
conventions; their differences are not prediction errors. MISO's approximately
142.0-GW requirement is inferred from rounded components and published balances,
not an independently extracted full-precision requirement.

The three native-ledger checks can also be run with
`python -B followup_20261013/regional_checks.py`. They report each study's balance
relative to its own requirement, verify the equivalent placement of the NERC
response term, and apply the PJM pool factors only within that report's portfolio.
They do not transfer accreditation factors between studies. All three checks
are included in both the annual runner and the targeted follow-up.

The public source check reads retained POWER header snapshots and matches their
recorded archive hashes to the core source audit. The authors checked the twenty
original archives when preparing these snapshots. This run does not re-read or
download those bulk archives, certify POWER integration bounds or establish
physical alignment with EIA hour-ending demand. CERES hour boxes document an
upstream convention, not the complete export mapping.

The legacy `pv_peak_windows_full.csv` and `tables/pv_windows.tex` reproduce the
old equal-label convention for regression only. Current results are
`pv_peak_windows_corrected.csv` and `pv_peak_window_summary_corrected.csv`, also
available in `physical_intervals/interval_results`. The provider FAQ establishes
POWER hour-start labels; EIA demand uses hour-ending labels. The current join is
EIA end t to POWER start t-1 hour, with adjacent checks centered on that match.

The 50/100/200 highest-load-hour windows are nested samples of the same complete
2025 summer, checked at residual-label offsets -1, 0 and +1 hour. They are not
independent experiments, confidence intervals or ten-year adequacy estimates.
No offset is selected for better performance. Annual energy is unchanged by the
interval correction. Current top-100-hour results differ from the old equal-label
values; neither calculation establishes ELCC or capacity credit.

The independent-directory test used Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3
on macOS/Apple silicon. It tested this entry point, not a new full-core run or a
fresh installation of every pinned dependency.
