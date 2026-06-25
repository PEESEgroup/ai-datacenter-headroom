# Data availability and redistribution notes

This package prioritizes reproducibility while respecting third-party data restrictions.

## Included

- Canonical county, state and grid-region AI load allocation tables.
- Canonical supply-headroom, generator-pipeline, nuclear-recovery, on-site generation and price-stress source tables.
- Robustness and audit tables.
- A lightweight public evidence layer with representative project-record source URLs, project geography/status summaries, LMP/load coverage manifests, third-party source manifests and redacted NASA POWER request summaries.
- Redacted NASA POWER site-level and hourly PV-window outputs with exact coordinates and request URLs removed.

## Redacted or omitted from the public-style package

- Exact data-center site coordinates and raw site-identifying fields where source terms or privacy concerns may restrict redistribution.
- Full raw ISO/RTO hourly LMP/load parquet records. The package includes harmonized derived summaries and source manifests rather than the 131.5-million-record raw panel.
- Complete raw FracTracker/Piedmont/partner fields and pair-level deduplication evidence that may contain names, addresses, operator/source text or source-specific license terms. The public package includes a representative source-evidence sample and aggregate summaries instead.
- External raw generator, interconnection queue, market, cost and fuel datasets when their public providers require users to download them directly.
- Manuscript text, supplementary-information text and final figure PDFs.

## Controlled access

Restricted raw inputs, API logs with exact coordinates and full raw LMP/load panels should remain under controlled private access where source-data terms permit. The public GitHub release should keep this redacted structure unless explicit redistribution permission is confirmed.
