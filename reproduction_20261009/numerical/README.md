# Annual Data-Center Planning Scenarios: 2026-2035

Version: `contribution-polish-20261005`. This independent numerical package does
not modify or supersede the earlier submitted-vintage repository. It contains no
manuscript, cover letter, referee report or account credential.

The framework links specified data-center growth, regional resource development
and load response over annual paths. Matched comparisons distinguish cushion use
from added contribution requirements and conditions changing that result. The
reserve-balance identity is established accounting, not a new estimator;
scenario results do not demonstrate improved procurement or predict reliability
events. Geography informs conditional load allocation and siting context, not a
county-level network hosting model.

## Requirements and Installation

Python 3.12, an ordinary CPU, a C compiler, 4 GB available RAM and 1 GB free disk space
are recommended. The exact Python dependencies are in `requirements.txt`.
Tested operating system: macOS 15.3, Apple silicon. Other operating systems have
not been tested. The hourly dispatch kernel uses the existing C implementation;
Apple clang 14.0.3 was tested. No GPU or proprietary solver is required. A typical installation
with a working internet connection takes approximately 1-3 minutes.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B scripts/run_all.py
```

The supplied real datasets are also the demonstration inputs. `run_all.py`
verifies their hashes before running. Allow roughly 1-5 minutes on an ordinary
desktop; the accompanying verification file records the actual tested runtime
and environment. The package runs offline after dependency installation.

## Expected Outputs

- National energy paths, spatial/peak mappings and the 292,650-row annual
  additional-load scenario cube: `results/101_*` through `results/110_*`.
- Seven-area native conditions and 388,800 paired MISO/ERCOT annual points:
  `seasonal_resource_ledger_20261004/03_*` and `14_*`.
- Complete paths, paired pressure, added requirements and full-period limits:
  `results/contribution/`. This includes all 38,880 ten-year paths and the 5,184
  zero-association core paths behind the new SI range tables.
- A fresh rolling-origin fit of every declared historical method and genuine
  region-heldout refits: `results/historical_refit/`. On the 150 matched cases,
  the fixed resource-drift/calibrated-demand-response method gives about 6.38 pp
  gap MAE, versus 7.13 pp for same-reference persistence. The pooled advantage
  is not universal across regions; the full declared comparison is retained.
- `reproduction_verification.json`: success requires calculation stages,
  formula checks and the final `run_frozen_outputs.py` comparison to pass.
  That stage checks complete keys, schemas, missing/infinite-value states,
  numeric values and text in all 40 new-release tables across `intervention`,
  `source_quality`, `targeted` and `spatial`. Immutable fixtures and explicit
  keys are in `regression/contribution_polish_20261005/`; absolute numeric tolerance
  is 1e-10, with no relative tolerance. Details are recorded in
  `frozen_output_verification.json`. PASS means reproducible declared outputs,
  not independent scientific validation of inputs or future outcomes.
- Four monotone source-DC timing rules retaining every official anchor:
  `results/targeted/` contains 20,736 core ten-year paths, annual values,
  conditional resource-development thresholds and response-budget checks.
- Reparsed public project MW and all eight fixed-total allocation rules:
  `results/spatial/`. Raw records, the source workbook, frozen county crosswalk,
  fallback ledger, size-proxy and service-territory diagnostics are included.
- All 676 PV-storage results recomputed from ten years of source-matched hourly
  weather at 20 sites, with high-load hours reselected from archived EIA-reported load:
  `results/intervention/`. Energy conversion and chronological dispatch are rerun.
- A complete local summer is required before ranking: all 2,928 observations
  must be unique, finite, positive and pass a gross-outlier screen (>10 times
  the region-year summer median; 5x/20x diagnostics are archived). Only 2025
  qualifies across all four areas (100 selected hours per area). The unresolved
  SPP 2023 value of 3,621,097 MW and 2024 MISO/SPP gaps invalidate those common
  years. An API recheck still reports the outlier. No clipping, author filling
  or forecast substitution is used. Point exclusion plus reranking of SPP
  2023 is a separate incomplete-season diagnostic, not the primary sample.
  Original extracts and earlier regression fixtures remain archived.
- Lower 390/430 TWh endpoint author scenarios and five scope alternatives:
  `results/source_quality/`. These reconstruct 138,240 and 259,200 annual
  matched MISO/ERCOT points respectively. The endpoints are informed by the
  LBNL comparison of EPRI/IEA; they are not full agency forecasts. Scope tests
  do not certify every campus identity or electrical connection.
- Scope classification retains notes and all saved source fields; mixed,
  transition and uncertain references do not establish a mining MW fraction.
  Campus representatives are selected after each allocation rule's eligibility
  filters. Shared keys require a leading premise number and matching
  state/county/address. Unknown, road-only, road-vicinity and distance descriptions are
  record-specific; this does not certify that retained records are distinct projects.
  `15_targeted_output_field_dictionary.csv` separates the unchanged reference
  candidate input from `candidate_GW`, the actual increment used in each case.

These are conditional planning accounts, not probabilities of outage, calibrated
scenario probabilities or a measurement of all seven areas' DC-only attribution.
Path counts describe the declared grid, not independent observations.

## Data Lineage and Limits

`sources/` contains archived public NERC, LBNL, CEC and EPRI evidence and derived
scenario inputs; `SOURCE_LEDGER.csv` identifies source links and file hashes.
`source_audit/` records a separate numerical check of 560 NERC table values.
Historical training/outcome panels are preserved source-derived data; the runner
refits from these panels but does not re-extract all historical PDFs.

The original annual backbone, identified source DC paths, earlier decomposition,
peak-window definitions and county mapping crosswalks are frozen inputs. The runner
rebuilds updated energy/peak scenarios, re-extracts the
current NERC native panel, reconstructs paired contributions and re-estimates
historical calibration. It also reparses supplied public project records and
reruns the complete hourly PV-storage grid. The main command reads source-matched
weather CSVs; archived NASA/ERA5 files support the additional extraction check
below. It does not regenerate the original county/utility polygons, price series
or EIA inventory experiment, or certify actual electric-service assignment.
It does not regenerate artwork. Those separate components must not
be described as reproduced by this command.

Nominal peak-weather matching uses archived UTC labels. The adjacent-hour
diagnostic reports substantial interval-alignment sensitivity; it is not used
to select a favorable coincidence value. These results are conditional physical
screens, not certified capacity accreditation. Complete provider interval
sign-off and upstream measurement accuracy remain separate from coverage QA.
`quality/hour_interval_source_check.json` records the official-source check:
EIA hour-ending integration and ERA5 validity instants are established, but the
checked NASA POWER/CERES documentation does not establish the integration bounds
of the archived irradiance labels. No alignment is chosen from PV scores and no
stable regional peak-capacity ranking is inferred.

For a provider-file weather re-extraction, use an environment with `xarray` and
a compatible NetCDF/HDF5 backend, then run:

```sh
python -B scripts/run_pv_hourly.py --from-provider-files
```

This check compares every matched weather value with the archived hourly inputs
before rerunning all 676 points. Its separate verification record identifies
whether provider files were re-extracted. The core pinned requirements suffice
for the default hourly run; the optional reader has a separately recorded tested
environment, not a claim of a newly tested locked installation.

Use `scripts/run_all.py` as the supported entry point. Archived builder modules
also retain functions used during development; their original standalone `main`
entry points can require inputs outside this curated package and are not supported.

### Current-Attempt Status and Frozen Comparison

`reproduction_verification.json` and `frozen_output_verification.json` describe
the latest invocation, with a unique attempt ID, UTC timestamps and input/code
manifest hashes. The full entry point first sets its status to `running` and the
comparison status to `not_run`. `passed` becomes true only after the relevant
checks finish; a handled failure records `failed` and the exception. If a process
is forcibly terminated before it can finish, `running`/`not_run` remains false,
not a stale success. A previous success may appear as `last_success`, explicitly
historical and not evidence that the latest attempt passed. Check the exit code
and latest status, not the historical record.

The current immutable fixtures are in `regression/contribution_polish_20261005`.
Earlier fixtures are retained unchanged. The comparator checks parsed table
schemas, complete keys, all columns and missing/infinite-value states at the
declared tolerance. It does not compare CSV bytes or distinguish every textual
NA spelling or serialized numeric type. Do not update fixtures automatically
after a discrepancy; record and review each intended change before freezing a
new version.

## Running Alternative Inputs

Work on a copy and retain this version unchanged. Replace files only with matching
column schemas and compatible region, season and capacity definitions. Edit the
scenario configuration only with a separately recorded rule change. Expected
files are regression tests for this frozen release, not targets for altered data.
When evaluating another vintage, run the calculation stages individually in order and
record new input hashes rather than changing expected files to hide differences:

```sh
python -B scripts/run_review_package.py
python -B scripts/run_contributions.py
python -B scripts/run_historical_refit.py
python -B scripts/run_timing_sensitivity.py
python -B scripts/run_spatial_weights.py
python -B scripts/seasonal_load_quality.py
python -B scripts/run_pv_hourly.py
python -B scripts/run_source_quality.py
```

Remove comparisons with this release's expected outputs only in an explicitly
versioned new-vintage runner. Historical method weights use matured outcomes
available to each origin; do not tune on future outcomes or the display scenario.

## License and Review Status

PEESE-authored code is distributed under the MIT license in `LICENSE`. MIT does
not relicense third-party documents or data. Provider rights and access terms
continue to apply; links are included for provenance, not a claim that all source
material is newly open-licensed. Check those terms before further redistribution.

The clean-environment test is an automated author-side reproduction, not an
independent colleague certification. `INDEPENDENT_REVIEW.csv` is a blank sign-off
form for that remaining check. No public deployment is implied by this local
release. A repository branch or permanent archival DOI must be verified before
the manuscript can state that this exact version is publicly accessible.

The preceding private independent-agent audit reran annual accounts, paired
contributions, historical refits, timing, spatial weights, provider-file weather
extraction and hourly dispatch. It found finite defects addressed by this release;
it did not certify all upstream records, physical interval alignment or these
final closure changes. The earlier author-side integrity test installed pinned
dependencies into a new isolated Python 3.12 environment. This closure's
extracted-copy test reuses that documented environment; it is not a second fresh
installation or an independent colleague installation. Anonymous retrieval of
this exact release remains untested because it has not been published.
