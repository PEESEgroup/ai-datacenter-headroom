# Source-data directory

`derived_tables/` contains the machine-readable source tables used for headline values, figure panels, audits and robustness checks.

`public_evidence/` contains representative public source-evidence samples and source-coverage manifests. This layer is meant to make the upstream data chain auditable without publishing the complete raw project-record, coordinate or large parquet archives.

`redacted_site_level/` contains reduced site-level solar and NASA POWER hourly files. Exact coordinates, URLs, names, addresses and raw project-identifying fields were removed from the public-style package. The retained fields are sufficient to audit aggregate ISO-level solar-resource and PV-window calculations.

`restricted_file_manifest.csv` lists exact files excluded from the public-style archive and the reason for exclusion.
