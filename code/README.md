# Code directory

`validate_package.py` is self-contained and can be run from the package root using the Python standard library.

`provenance_scripts/` contains analysis workflow scripts copied from the working repository. These scripts document the calculation and figure-building path, but not all are intended to be public one-click scripts because several require local raw inputs, geospatial helper files or source-restricted data. The canonical source tables in `source_data/derived_tables/` are therefore the primary reproducibility interface.

Recommended sequence:

1. Run `python code/validate_package.py`.
2. Inspect `source_data/file_manifest.csv` and `source_data/restricted_file_manifest.csv`.
3. Inspect figure-source tables under `source_data/derived_tables/tables/fig*_canonical_*`.
4. Use the provenance scripts as audit trails for how derived tables and figures were generated.
