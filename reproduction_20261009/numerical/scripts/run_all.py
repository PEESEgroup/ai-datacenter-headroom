"""Run annual reconstruction, complete-path contributions and historical refitting."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

from run_status import RunStatus, mark_not_run

ROOT = Path(__file__).resolve().parents[1]


def run_stages():
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Input or code hash mismatch: " + name)
    for name in ("test_reproduction_compare.py", "test_seasonal_load_quality.py", "test_scope_quality.py", "test_frozen_outputs.py", "test_run_status.py", "run_review_package.py", "run_contributions.py",
                 "run_historical_refit.py", "run_timing_sensitivity.py", "run_spatial_weights.py",
                 "seasonal_load_quality.py", "run_pv_hourly.py", "run_source_quality.py", "run_frozen_outputs.py"):
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / name)], check=True)
    record = {"passed": True,
              "python": sys.version, "platform": platform.platform(),
              "isolated_virtual_environment": sys.prefix != sys.base_prefix,
              "annual_rebuilt": True, "contribution_rebuilt": True, "historical_refitted": True,
              "anchor_timing_sensitivity_recomputed": True, "project_weights_recomputed": True,
              "hourly_PV_grid_recomputed": True, "raw_provider_weather_reextracted": False,
              "complete_season_gate_tested": True, "low_demand_and_scope_checks_recomputed": True,
              "new_output_complete_key_and_all_column_regression_passed": True,
              "new_output_tables_compared": json.loads((ROOT / "frozen_output_verification.json").read_text())["tables_compared"],
              "frozen_comparison_attempt_id": json.loads((ROOT / "frozen_output_verification.json").read_text())["attempt_id"],
              "regression_success_is_scientific_validation": False,
              "independent_colleague_reproduction": False, "full_raw_map_weather_inventory_reingestion": False}
    return record


def main():
    with RunStatus(ROOT / "reproduction_verification.json",
                   {"input_and_code": ROOT / "SHA256SUMS.txt",
                    "regression": ROOT / "regression/contribution_polish_20261005/manifest.json"}) as attempt:
        mark_not_run(ROOT / "frozen_output_verification.json", attempt.record["attempt_id"])
        attempt.succeed(run_stages())
    print(json.dumps(attempt.record, indent=2))


if __name__ == "__main__":
    main()
