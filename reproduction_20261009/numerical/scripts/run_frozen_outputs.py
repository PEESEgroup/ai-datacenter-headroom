"""Compare every new-release table with an immutable, versioned regression fixture."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from run_status import RunStatus

ROOT = Path(__file__).resolve().parents[1]


def compare_frames(actual, expected, keys, atol=1e-10):
    if list(actual.columns) != list(expected.columns):
        raise ValueError("Column schema or order changed")
    if not keys or not set(keys).issubset(actual.columns):
        raise ValueError("A declared complete key is required")
    for frame in (actual, expected):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Null or duplicate complete key")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError("Missing, extra or altered keys")
    largest = 0.0
    for column in left:
        a, b = left[column], right[column]
        if not a.isna().equals(b.isna()):
            raise ValueError("Missing-value pattern changed: " + column)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            if not np.isposinf(av).tolist() == np.isposinf(bv).tolist() or not np.isneginf(av).tolist() == np.isneginf(bv).tolist():
                raise ValueError("Infinity state changed: " + column)
            finite = np.isfinite(av) & np.isfinite(bv)
            difference = np.abs(av[finite] - bv[finite])
            limit = 0 if (pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b)) else atol
            if (difference > limit).any():
                raise ValueError("Numeric values changed: " + column)
            largest = max(largest, float(difference.max()) if len(difference) else 0)
        else:
            pd.testing.assert_series_equal(a, b, check_dtype=False, check_exact=True)
    return largest


def compare_outputs(base):
    manifest = json.loads((base / "manifest.json").read_text())
    actual_set = {p.relative_to(ROOT).as_posix() for folder in manifest["folders"]
                  for p in (ROOT / folder).glob("*") if p.suffix in {".csv", ".gz"}}
    expected_set = {entry["output"] for entry in manifest["tables"]}
    if actual_set != expected_set:
        raise ValueError("Declared output set changed: " + str(sorted(actual_set ^ expected_set)))
    comparisons = []
    for entry in manifest["tables"]:
        fixture = base / entry["fixture"]
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen fixture changed: " + entry["fixture"])
        actual = pd.read_csv(ROOT / entry["output"])
        expected = pd.read_csv(fixture)
        if len(expected) != entry["rows"]:
            raise ValueError("Manifest row count disagrees with fixture")
        delta = compare_frames(actual, expected, entry["keys"], manifest["absolute_tolerance"])
        comparisons.append({"output": entry["output"], "rows": len(actual),
                            "complete_keys_and_all_columns_match": True, "maximum_absolute_difference": delta})
    record = {"passed": True, "tables_compared": len(comparisons),
              "absolute_tolerance": manifest["absolute_tolerance"], "relative_tolerance": 0,
              "checks": comparisons, "scientific_validation_implied": False}
    return record


def main():
    base = ROOT / "regression/contribution_polish_20261005"
    with RunStatus(ROOT / "frozen_output_verification.json",
                   {"input_and_code": ROOT / "SHA256SUMS.txt", "regression": base / "manifest.json"}) as attempt:
        record = compare_outputs(base)
        attempt.succeed(record)
    print(json.dumps({k: v for k, v in attempt.record.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
