# Public reproducibility package

This repository provides a public reproducibility package for the analysis of AI data-center growth and regional supply-headroom alignment in the United States.

The package is intentionally built around **derived source-data tables and reproducibility scripts**, not a dump of the full research workspace. It supports three checks:

1. Trace headline calculations to machine-readable source tables.
2. Recompute selected headline metrics from the included tables.
3. Inspect which inputs are public-derived, restricted, redacted or available only through source-specific terms.

The public package includes a lightweight public evidence layer, derived source-data tables and provenance scripts. It excludes manuscript text, supplementary-information text, final figure PDFs, exact site-coordinate logs, complete raw project-announcement fields and large/restricted raw third-party data.

## Quick start

```bash
cd ai-datacenter-headroom-repro
python code/validate_package.py
```

The validation script checks file hashes and writes:

- `outputs/headline_metric_check.csv`
- `outputs/manifest_check.txt`

A full figure rebuild may require the optional geospatial stack listed in `environment.yml`, local map assets and restricted third-party source data. The public package therefore prioritizes canonical figure-source tables and selected validation checks.

## Directory guide

- `source_data/derived_tables/`: canonical figure-source, shared-demand, audit and robustness CSV/TeX tables.
- `source_data/public_evidence/`: representative public source-evidence samples and source-coverage manifests for project records, LMP/load panels, NASA POWER queries and third-party downloads.
- `source_data/redacted_site_level/`: redacted site-level NASA POWER and solar-resource outputs with exact coordinates, names, addresses and request URLs removed.
- `source_data/restricted_file_manifest.csv`: exact files intentionally omitted from the public-style package and why.
- `code/provenance_scripts/`: scripts used during the analysis workflow. Some are provenance scripts that require local raw inputs; see `code/README.md`.
- `docs/REPRODUCIBILITY_CHECKLIST.md`: checklist for the public package contents.
- `checksums/SHA256SUMS.txt`: SHA-256 checksums for package files.

## What this package does not include

The package does not redistribute restricted raw datasets, very large raw LMP parquet files, exact full site-coordinate logs, complete raw project-announcement fields, manuscript text, supplementary-information text or final figure PDFs. Those materials should be handled through source-specific permissions or controlled private access where appropriate.
