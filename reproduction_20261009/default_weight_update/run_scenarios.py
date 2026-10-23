"""Recalculate allocation-dependent results and record intentional differences."""
from pathlib import Path
import importlib
import gzip
import json
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG / "default_weight_update"
sys.path.insert(0, str(PKG / "numerical/scripts"))
import run_review_package as review

STRICT = review.compare
CHANGED = {"121_fig4_paired_contribution_display.csv", "06_displayed_DC_thresholds.csv",
           "02_complete_path_summary.csv", "01_core_contribution_ranges.csv"}
DIFF = []


def compare(expected, actual, keys, values):
    if expected.name not in CHANGED:
        return STRICT(expected, actual, keys, values)
    a, b = pd.read_csv(expected), pd.read_csv(actual)
    assert len(a) == len(b) and not b.duplicated(keys).any()
    paired = a[keys+values].merge(b[keys+values], on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(paired) == len(a)
    for field in values:
        old, new = paired[field+"_old"], paired[field+"_new"]
        finite = np.isfinite(old) & np.isfinite(new)
        DIFF.append(dict(table=actual.name, field=field,
                         changed_rows=int((abs(old-new)>1e-9).sum()),
                         max_change=float(abs(old[finite]-new[finite]).max()) if finite.any() else 0,
                         comparison="intentional reviewed-input change; not old-fixture agreement"))


def main():
    start = time.perf_counter()
    for entry in json.loads((PKG / "COMPRESSED_FILES.json").read_text()):
        target = PKG / entry["target"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(PKG / entry["stored"], "rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    subprocess.run([sys.executable, str(PKG / "rebuild_backbone.py")], check=True)
    spec = importlib.util.spec_from_file_location("revised_inputs", PKG/"default_weight_update/rebuild_inputs.py")
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    inputs.main()
    review.compare = compare
    stages = ["run_review_package", "run_contributions", "run_historical_refit",
              "run_timing_sensitivity", "run_spatial_weights", "seasonal_load_quality",
              "run_pv_hourly", "run_source_quality"]
    status = ROOT / "calculation_status.json"
    result = dict(passed=False, completed=[])
    try:
        for name in stages:
            importlib.import_module(name).main()
            result["completed"].append(name)
            status.write_text(json.dumps(result, indent=2)+"\n")
        pd.DataFrame(DIFF).to_csv(PKG/"default_weight_update/intentional_differences.csv", index=False)
        result.update(passed=True, seconds=time.perf_counter()-start, parameters_retuned=False)
    finally:
        status.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
