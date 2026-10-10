# Zero-Load Quality Diagnostic

The main price regression retains reported zero load because the archived
harmonized panels do not identify whether it is a measurement or a missing-value
code. This diagnostic does not change the main regression or annual balances.

From the release root, run:

```sh
python support/price/audit_zero_load.py
```

The script writes source-hashed zero-record counts, the flagged observations,
and common/zone-slope comparisons under `support/price/results/`. It archives
both exclusion at the original load scale and exclusion followed by zone-year
restandardization. Their standardized coefficients have different units and
must not be conflated. Price winsorization remains the baseline transformation.
The reported percentage change uses the restandardized exclusion diagnostic.

Neither a positive value nor a zero value is certified by this diagnostic as
correct or missing. Resolve that question with provider records before changing
the primary cleaning rule.
