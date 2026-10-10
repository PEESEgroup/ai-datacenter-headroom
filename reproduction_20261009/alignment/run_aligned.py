"""Recalculate a versioned alignment, retaining old fixtures as comparisons.

Intentional spatial changes are measured rather than asserted equal to old
results. Native planning, historical and PV regressions remain strict.
"""
from pathlib import Path
import importlib
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "numerical"
OUT = ROOT / "alignment"
sys.path.insert(0, str(NUM / "scripts"))
import run_review_package as review

STRICT_COMPARE = review.compare
CHANGED = {"121_fig4_paired_contribution_display.csv", "06_displayed_DC_thresholds.csv",
           "02_complete_path_summary.csv", "01_core_contribution_ranges.csv"}
DIFFERENCES = []


def compare(expected, actual, keys, values):
    if expected.name not in CHANGED:
        return STRICT_COMPARE(expected, actual, keys, values)
    old, new = pd.read_csv(expected), pd.read_csv(actual)
    for frame in [old, new]:
        assert not frame[keys].isna().any().any() and not frame.duplicated(keys).any()
    joined = old[keys + values].merge(new[keys + values], on=keys, how="outer", indicator=True,
                                      suffixes=("_old", "_new"), validate="one_to_one")
    assert len(joined) == len(old) == len(new) and joined._merge.eq("both").all()
    for field in values:
        a, b = joined[field + "_old"], joined[field + "_new"]
        finite = np.isfinite(a) & np.isfinite(b)
        DIFFERENCES.append({"table": actual.name, "field": field, "rows": len(new),
            "maximum_absolute_change": float((a[finite] - b[finite]).abs().max()) if finite.any() else 0,
            "missing_state_changes": int(a.isna().ne(b.isna()).sum()),
            "comparison_role": "intentional geography update; not an old-fixture PASS"})


def main():
    start = time.perf_counter()
    status = OUT / "aligned_run_verification.json"
    record = {"passed": False, "status": "running", "stages": [], "scientific_validation_implied": False}
    status.write_text(json.dumps(record, indent=2) + "\n")
    review.compare = compare
    try:
        for name in ["run_review_package", "run_contributions", "run_historical_refit",
                     "run_timing_sensitivity", "run_spatial_weights", "seasonal_load_quality",
                     "run_pv_hourly", "run_source_quality"]:
            module = importlib.import_module(name)
            module.main()
            record["stages"].append(name)
            status.write_text(json.dumps(record, indent=2) + "\n")
        pd.DataFrame(DIFFERENCES).to_csv(OUT / "intentional_recalculation_differences.csv", index=False)
        subprocess.run([sys.executable, "-B", str(ROOT / "build_figure_data.py")], cwd=ROOT, check=True)
        record.update(passed=True, status="passed", runtime_seconds=time.perf_counter() - start,
                      old_fixtures_replaced=False, native_source_numerics_changed=False,
                      geometry_is_2025_connection_verification=False)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        status.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
