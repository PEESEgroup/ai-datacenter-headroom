# Data-center growth and regional planning scenarios

Current reproduction package: **WECC-California-aligned revision, 9 October 2026**.
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

Large fixtures are stored losslessly compressed and restored automatically.
Retained earlier results serve as comparison fixtures, not inputs to the model.
The eight geographic weighting rules and California-aligned regional outputs
are used in this version. California denotes the WECC-California planning area,
not CAISO alone. The default run uses the frozen geographic crosswalk; it does
not rebuild every upstream GIS asset or render the publication figures.

## Interpretation

These are conditional summer-peak planning comparisons, not forecasts of outages
or calibrated scenario probabilities. Regions retain native capacity conventions;
their balances are not added into a national shortfall. Source-supported matched
growth-path comparisons are available for MISO/ERCOT, with narrower comparisons
elsewhere. PV energy savings and sampled peak-hour relief do not establish firm
capacity credit. See `INPUTS_AND_SCOPE.md` for input definitions and scope.

Author-written code is MIT-licensed. Third-party data are not relicensed under
MIT; provider attribution and use conditions apply as explained in
`THIRD_PARTY_NOTICES.md`. Cite the repository version or commit used.
