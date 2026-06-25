# Public Evidence Layer

This directory adds a lightweight public evidence layer to the reproducibility package. It is intended to show the upstream data chain behind the derived source tables without publishing the full private/raw working archive.

Included files:

- `raw_input_manifest.csv`: high-level map from source groups to public artifacts and redistribution notes.
- `project_record_public_evidence_sample.csv`: representative public project-evidence rows with state/county/status/capacity fields and public source URLs. Exact addresses, coordinates, facility names, operators and tenants are omitted.
- `project_record_geography_status_summary.csv`: full model-facing project-inventory summary by state, ISO/RTO, status class and confidence tier.
- `lmp_load_source_coverage_manifest.csv`: ISO/RTO load and LMP audit coverage, with row counts, time ranges and sanitized file-name examples rather than raw parquet files.
- `third_party_download_manifest.csv`: public third-party data sources used for costs, fuel, emissions, water context, solar and market/load screens.
- `nasa_power_redacted_request_log_summary.csv`: redacted NASA POWER request summary by ISO/status, without exact coordinates or request URLs.

This layer does not restore the full raw archive. The public GitHub package remains a derived-table and audit package; exact coordinates, full raw project records, raw LMP/load parquet panels, large third-party downloads, manuscript text, SI text and final figure PDFs remain outside the Git-tracked public package.
