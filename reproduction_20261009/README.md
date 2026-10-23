# Data-center growth and regional planning scenarios

Current reproduction package: **reviewed load-capacity weights and physical-hour
PV matching, 22 October 2026**, building on the WECC-California-aligned core.
This folder contains code, standardized inputs and figure/table data. It does not
contain the paper, Supplementary Information, reviewer reports or response letters.
The earlier repository files remain available separately.

## Run the analysis

From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B run_reproduction.py
```

The runner recomputes the annual 2026-2035 regional scenarios, matched data-center
growth comparisons, resource/response conditions, historical comparisons and
chronological PV-storage results. It also runs the supporting inventory and price
analyses and exports the numerical data for the figures and tables. No companion
download, GPU or proprietary solver is required. The standard run starts from
packaged inputs rather than repeating all upstream provider downloads.

The tested environment is Python 3.12.14 on macOS 15.3/Apple silicon, with a C
compiler. Allow several minutes, 4 GB RAM and 5 GB disk for dependencies and
generated results. Other operating systems have not been tested. A successful
run ends with `status: passed` in `package_reproduction_verification.json`.

## Find the data

| Material | Location |
| --- | --- |
| Planning components and scenario rules | `numerical/inputs`, `numerical/config`, `numerical/sources` |
| Project records and frozen geographic crosswalk | `numerical/spatial`, `alignment` |
| Source-matched weather, demand and historical panels | `numerical/intervention`, `numerical/historical`, `support/inputs` |
| Recomputed full-precision results | `numerical/results` and the corresponding support output directories, created by the runner |
| Figure/table data | `plotting_tables`, `source_data`, `display_tables`, `SOURCE_DATA_INDEX.csv` |
| Same-year 2025 PV comparison | `plotting_tables/pv_2025_configuration_comparison.csv` |
| Regional input reconciliation and peak windows (Tables 69-70) | `followup_20261013/inputs`, `followup_20261013/expected` |
| Native-ledger and within-PJM comparisons | `followup_20261013/regional_checks.py`, with three small tables in `followup_20261013/expected` |
| Project-attribute decisions and source locators | `numerical/spatial/capacity_attribute_review.csv`, `default_weight_update/primary_source_register.csv` |

Large fixtures are stored losslessly compressed and restored automatically.
This includes the pre-alignment comparison cube. Generated CEC source hashes
are written to `numerical/results/cec_source_hashes.csv`, leaving the frozen
source files unchanged on repeat runs. Packaging regression tests can be run
with `python -B -m unittest discover -s tests -v`.
Retained earlier results serve as comparison fixtures, not inputs to the model.
The eight geographic weighting rules and California-aligned regional outputs
are used in this version. California denotes the WECC-California planning area,
not CAISO alone. The default run uses the frozen geographic crosswalk; it does
not rebuild every upstream GIS asset or render the publication figures.

The default capacity ledger excludes five generation-only quantities and uses
separately stated load scales for PORTS and Monarch. These values feed both
the national capture fraction and regional shares. Raw records, source decisions
and the complete old-input comparison are retained in
`numerical/spatial/capacity_attribute_review.csv` and `default_weight_update`.
Count-based alternatives retain the sites. Existing-load county shares are
archived inputs; subsequent county growth follows reviewed project-load weights.

Current PV peak results use the documented interval join: EIA demand ending at
time t is paired with POWER weather starting at t minus one hour.
`physical_intervals` rebuilds 676 configurations and 50/100/200-hour windows.
Annual energy uses 2016-2025 weather; peak-hour summaries use summer 2025.
Older equal-label outputs are regression fixtures, not current peak results.

## Run the added checks without rerunning the main model

```sh
python -B followup_20261013/run_followup.py
```

This entry rebuilds regional input reconciliation, checks legacy PV fixtures,
then recomputes the corrected physical-hour grid and peak windows. It does not
run the regional annual builder. It extracts current paired contributions and
independently checks the MISO annual limits.
See `followup_20261013/README.md` for outputs, source metadata and the limits of
the time-alignment checks. `default_weight_update/audit_current.py` separately
checks the reviewed inputs, 388,800 annual pairs and current PV reader exports.

## Interpretation

The input, accounting, sensitivity and claim rules are recorded together in
FRAMEWORK_RULES.json. National energy, project-scale location weights,
grid-supplied candidate increments and official planning demand have distinct
roles. Unknown project delivery or an unclear reported capacity basis is not,
by itself, a reason to delete a record. The existing generation-attribute
corrections remain in place; this documentation alignment excludes no further
projects and changes no numerical results.

These are conditional summer-peak planning comparisons, not forecasts of outages
or calibrated scenario probabilities. Regions retain native capacity conventions;
their balances are not added into a national shortfall. Source-supported matched
growth-path comparisons are available for MISO/ERCOT, with narrower comparisons
elsewhere. PV energy savings and sampled peak-hour relief do not establish firm
capacity credit. See `INPUTS_AND_SCOPE.md` for input definitions and scope.

Author-written code is MIT-licensed. Third-party data are not relicensed under
MIT; provider attribution and use conditions apply as explained in
`THIRD_PARTY_NOTICES.md`. Cite the repository version or commit used.
