"""Compare aligned results with the untouched pre-alignment calculation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "numerical"
OUT = ROOT / "alignment"
OLD = OUT / "baseline"


def unchanged(relative):
    a, b = pd.read_csv(OLD / relative), pd.read_csv(NUM / relative)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10, rtol=1e-12)
    return {"table": relative, "rows": len(b), "unchanged": True}


def compare_ranges(relative, keys):
    a, b = pd.read_csv(OLD / relative), pd.read_csv(NUM / relative)
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    fields = [c for c in a.select_dtypes(include="number").columns if c not in keys and c in b]
    both = a[keys + fields].merge(b[keys + fields], on=keys, how="outer", indicator=True,
                                  suffixes=("_old", "_new"), validate="one_to_one")
    assert both._merge.eq("both").all()
    rows = []
    for field in fields:
        delta = both[field + "_new"] - both[field + "_old"]
        rows.append({"table": relative, "field": field, "rows": len(b),
                     "changed_rows": int((delta.abs() > 1e-10).sum()),
                     "maximum_absolute_change": float(delta.abs().max())})
    return rows


def main():
    strict = [unchanged(p) for p in [
        "seasonal_resource_ledger_20261004/03_corrected_condition_panel.csv",
        "results/101_lbnl2025_national_energy_paths.csv",
        "results/112_core_total_system_trajectories.csv",
        "results/118_fig2_source_annual_plotting.csv",
        "results/intervention/pv_storage_fine_grid.csv",
        "results/historical_refit/02_metrics.csv",
    ]]
    pairs0 = pd.read_csv(OLD / "results/121_fig4_paired_contribution_display.csv")
    pairs1 = pd.read_csv(NUM / "results/121_fig4_paired_contribution_display.csv")
    pd.testing.assert_frame_equal(pairs0, pairs1, check_exact=False, atol=1e-10, rtol=1e-12)
    strict.append({"table": "results/121_fig4_paired_contribution_display.csv", "rows": len(pairs1), "unchanged": True})
    spatial = pd.read_csv(OUT / "eight_rule_weight_comparison.csv")
    primary = spatial[spatial.allocation_case.eq("central_reported_mid")]
    primary_joint = primary.loc[primary.model_region.ne("WECC-California"),
                                "core_active_reported_mw_share_joint_change"]
    assert primary_joint.abs().max() < 1e-12
    records = pd.read_csv(OUT / "all_project_assignments.csv")
    assert len(records) == 1680 and not records.record_key.duplicated().any()
    crosswalk = pd.read_csv(OUT / "project_footprint_crosswalk.csv")
    assert not crosswalk.actual_electrical_connection_verified.any()
    assert not crosswalk.exact_2025_NERC_GIS_verified.any()
    cube = pd.read_csv(NUM / "results/104_annual_scenario_cube_v1_2.csv")
    california = cube[cube.planning_region.eq("WECC-California")]
    old_cube = pd.read_csv(OLD / "results/104_annual_scenario_cube_v1_2.csv", usecols=["planning_region"])
    old_ca_rows = int(old_cube.planning_region.eq("WECC-California").sum())
    assert len(cube) == 341250 and len(california) - old_ca_rows == 48600
    keys = ["planning_region", "year", "resource_condition", "response_condition"]
    changed = compare_ranges("results/116_core_envelope_region_labels.csv", ["planning_region"])
    pd.DataFrame(changed).to_csv(OUT / "core_envelope_numerical_differences.csv", index=False)
    report = {"passed": True, "strict_unchanged_tables": strict, "project_records_retained": len(records),
              "annual_scenario_rows": len(cube), "california_rows": len(california),
              "new_california_additional_load_rows": len(california) - old_ca_rows,
              "other_six_primary_joint_weights_unchanged": True,
              "independent_external_scientific_validation": False,
              "exact_2025_boundary_or_connection_certified": False,
              "California_shape_is_transferred_scenario_not_native_peak_observation": True}
    (OUT / "differential_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
