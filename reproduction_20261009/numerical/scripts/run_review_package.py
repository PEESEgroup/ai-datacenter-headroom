"""Portable numerical reproduction; uses only files inside the package."""
from pathlib import Path
import importlib.util
import json
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def compare(expected, actual, keys, values):
    old, new = pd.read_csv(expected), pd.read_csv(actual)
    if old.empty:
        raise AssertionError(f"Expected output is empty: {expected.name}")
    if len(old) != len(new):
        raise AssertionError(f"Coverage differs: {expected.name}: {len(old)} vs {len(new)}")
    for label, frame in [("expected", old), ("actual", new)]:
        missing = set(keys + values) - set(frame.columns)
        if missing or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise AssertionError(f"Invalid {label} columns or keys: {expected.name}: {sorted(missing)}")
    joined = old[keys + values].merge(new[keys + values], on=keys, how="outer",
                                    indicator=True, suffixes=("_expected", "_actual"), validate="one_to_one")
    if len(joined) != len(old) or not joined["_merge"].eq("both").all():
        raise AssertionError(f"Key sets differ: {expected.name}")
    for col in values:
        if not np.allclose(joined[col + "_expected"], joined[col + "_actual"], atol=1e-8, rtol=1e-10, equal_nan=True):
            raise AssertionError(f"Numerical difference: {expected.name}: {col}")


def main():
    start = time.perf_counter()
    for folder in [ROOT / "results", ROOT / "docs", ROOT / "expected"]:
        folder.mkdir(parents=True, exist_ok=True)
    for name in ["build_fixed_energy_mapping.py", "build_lbnl2025_scenario_update.py",
                 "build_core_envelope_acceptance.py", "build_main_figure_tables.py"]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)
    ledger_path = ROOT / "seasonal_resource_ledger_20261004"
    ledger = module("native_ledger", ledger_path / "build_seasonal_ledger.py")
    native = pd.read_csv(ROOT / "inputs/native_summer_components.csv")
    ercot = pd.read_csv(ROOT / "inputs/ercot_reported_response.csv")
    native = ledger.add_response(native, ercot)
    panel = ledger.condition_panel(native)
    panel.to_csv(ledger_path / "03_corrected_condition_panel.csv", index=False)
    comparison = panel.copy()
    comparison["legacy_response_condition"] = comparison.response_condition.replace({"current_cap": "current_only"})
    # Only undelayed rows define the reference background used by this function.
    comparison["old_balance_GW"] = comparison.margin_GW
    pairs, _ = ledger.paired_replacement_panel(comparison)
    pairs.to_csv(ledger_path / "14_paired_DC_replacement_new_background.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "decision_thresholds_20261004/prepare_inputs.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "decision_thresholds_20261004/build_thresholds.py")], check=True)
    metrics = pd.read_csv(ROOT / "historical/02_metrics.csv")
    predictions = pd.read_csv(ROOT / "historical/01_predictions.csv")
    variables = {"balance_GW": ("predicted_margin_MW", "outcome_margin_MW", 1000),
                 "gap_pp": ("predicted_gap_to_RML_pp", "outcome_gap_to_RML_pp", 1),
                 "demand_GW": ("predicted_demand_MW", "outcome_total_internal_demand_MW", 1000),
                 "resources_GW": ("predicted_resources_MW", "outcome_anticipated_resources_MW", 1000),
                 "response_GW": ("predicted_response_MW", "outcome_reported_demand_response_MW", 1000)}
    verified = []
    for row in metrics[metrics.horizon.eq("all") & metrics.scope.isin(["all_origins", "fitted_origins"])].itertuples():
        selected = predictions[predictions.model.eq(row.model)]
        if row.scope == "fitted_origins":
            selected = selected[selected.fit_supported]
        prediction, outcome, scale = variables[row.variable]
        mae = ((selected[prediction] - selected[outcome]) / scale).abs().mean()
        assert len(selected) == row.n and np.isclose(mae, row.mae, atol=1e-10)
        verified.append({"model": row.model, "scope": row.scope, "variable": row.variable, "mae": mae})
    pd.DataFrame(verified).to_csv(ROOT / "results/143_current_historical_score_reproduction.csv", index=False)
    compare(ROOT / "expected/112_core_total_system_trajectories.csv", ROOT / "results/112_core_total_system_trajectories.csv",
            ["planning_region", "resource_condition", "response_condition"], ["balance_2035_GW", "minimum_balance_GW", "years_below_reference"])
    compare(ROOT / "expected/121_fig4_paired_contribution_display.csv", ROOT / "results/121_fig4_paired_contribution_display.csv",
            ["planning_region", "year", "resource_condition", "response_condition", "linked_response_fraction"],
            ["candidate_gross_path_GW", "replacement_balance_GW", "dc_path_pressure_contribution_GW", "dc_added_required_contribution_GW"])
    compare(ROOT / "expected/native_panel.csv", ledger_path / "03_corrected_condition_panel.csv",
            ["planning_region", "year", "resource_condition", "response_condition"],
            ["gross_peak_demand_MW", "capacity_equivalent_resources_MW", "credited_DR_MW", "margin_GW"])
    threshold_result = ROOT / "decision_thresholds_20261004/results/06_displayed_DC_thresholds.csv"
    compare(ROOT / "expected/06_displayed_DC_thresholds.csv", threshold_result,
            ["planning_region", "year", "resource_condition", "response_condition", "linked_response_fraction"],
            ["replacement_balance_GW", "dc_path_pressure_contribution_GW", "dc_added_required_contribution_GW"])
    facts = {"passed": True, "isolated_virtual_environment": sys.prefix != sys.base_prefix,
             "native_rows": len(panel), "paired_rows": len(pairs), "historical_score_cells": len(verified),
             "runtime_seconds": time.perf_counter() - start, "python": sys.version,
             "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
             "independent_human_reproduction": False, "hourly_PV_recomputed": False,
             "historical_calibration_refitted": False, "figure_assets_regenerated": False}
    (ROOT / "clean_run_verification.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
