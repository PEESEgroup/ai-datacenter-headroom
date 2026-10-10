"""Rebuild complete annual pairs and path limits from freshly reconstructed inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_review_package import compare

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["planning_region", "resource_condition", "response_condition", "data_center_scenario_id",
        "post2030_extension_id", "footprint_capture_case", "peak_mapping_case", "allocation_case", "linked_response_fraction"]
TOL = 1e-7


def main():
    native = pd.read_csv(ROOT / "seasonal_resource_ledger_20261004/03_corrected_condition_panel.csv")
    dc = pd.read_csv(ROOT / "seasonal_resource_ledger_20261004/14_paired_DC_replacement_new_background.csv")
    nkeys = ["planning_region", "year", "resource_condition", "response_condition"]
    dc = dc.merge(native[nkeys + ["reference_margin_level_pct", "margin_GW"]], on=nkeys,
                  suffixes=("", "_native"), validate="many_to_one")
    factor = 1 + dc.reference_margin_level_pct / 100
    dc["fresh_no_new_DC_balance_GW"] = dc.margin_GW + factor * (dc.official_gross_path_GW - dc.official_linked_response_GW)
    dc["fresh_with_DC_balance_GW"] = dc.fresh_no_new_DC_balance_GW - factor * (dc.candidate_gross_path_GW - dc.candidate_linked_response_GW)
    dc["fresh_DC_pressure_GW"] = dc.fresh_no_new_DC_balance_GW - dc.fresh_with_DC_balance_GW
    dc["fresh_DC_added_requirement_GW"] = np.maximum(-dc.fresh_with_DC_balance_GW, 0) - np.maximum(-dc.fresh_no_new_DC_balance_GW, 0)
    dc["cushion_consumed_GW"] = dc.fresh_DC_pressure_GW - dc.fresh_DC_added_requirement_GW
    p, b0, bs = dc.fresh_DC_pressure_GW, dc.fresh_no_new_DC_balance_GW, dc.fresh_with_DC_balance_GW
    dc["annual_regime"] = np.select([p <= TOL, b0 < -TOL, bs < -TOL],
        ["no_net_pressure", "amplifies_existing_requirement", "triggers_requirement"], default="absorbed_by_cushion")
    dc["annual_growth_scale_limit"] = np.where((b0 >= -TOL) & (p > TOL), np.maximum(b0, 0) / p.where(p > TOL), np.nan)
    summary = dc.groupby(KEYS).agg(
        annual_points=("year", "size"), pressure_sum_GW_year=("fresh_DC_pressure_GW", "sum"),
        added_requirement_sum_GW_year=("fresh_DC_added_requirement_GW", "sum"),
        cushion_consumed_sum_GW_year=("cushion_consumed_GW", "sum"),
        minimum_candidate_balance_GW=("fresh_with_DC_balance_GW", "min"),
        candidate_below_years=("fresh_with_DC_balance_GW", lambda x: int((x < -TOL).sum())),
        background_below_years=("fresh_no_new_DC_balance_GW", lambda x: int((x < -TOL).sum())),
        all_year_growth_scale=("annual_growth_scale_limit", "min")).reset_index()
    summary["all_year_scale_status"] = np.select([summary.background_below_years > 0, summary.all_year_growth_scale.isna()],
        ["background_below_reference", "no_finite_algebraic_bound"], default="finite")
    summary.loc[summary.all_year_scale_status.ne("finite"), "all_year_growth_scale"] = np.nan
    end = dc[dc.year.eq(2035)][KEYS + ["annual_growth_scale_limit", "fresh_no_new_DC_balance_GW",
        "fresh_with_DC_balance_GW", "fresh_DC_pressure_GW", "fresh_DC_added_requirement_GW", "cushion_consumed_GW"]]
    summary = summary.merge(end.rename(columns={"annual_growth_scale_limit": "endpoint_growth_scale"}), on=KEYS, validate="one_to_one")
    paths = dc.merge(summary[KEYS + ["endpoint_growth_scale", "all_year_growth_scale", "all_year_scale_status"]], on=KEYS, validate="many_to_one")
    paths["at_endpoint"] = paths.fresh_no_new_DC_balance_GW - paths.endpoint_growth_scale * paths.fresh_DC_pressure_GW
    min_end = paths.groupby(KEYS).at_endpoint.min().rename("min_balance_at_endpoint_scale_GW").reset_index()
    summary = summary.merge(min_end, on=KEYS, validate="one_to_one")
    summary["endpoint_test_misses_earlier_pressure"] = summary.endpoint_growth_scale.notna() & summary.min_balance_at_endpoint_scale_GW.lt(-TOL)
    finite = paths[paths.all_year_scale_status.eq("finite")].copy()
    finite["at_full_limit"] = finite.fresh_no_new_DC_balance_GW - finite.all_year_growth_scale * finite.fresh_DC_pressure_GW
    assert finite.at_full_limit.min() >= -TOL
    assert finite.groupby(KEYS).at_full_limit.min().abs().max() < TOL
    out = ROOT / "results/contribution"
    out.mkdir(parents=True, exist_ok=True)
    dc.to_csv(out / "01_annual_paired_outcomes.csv.gz", index=False)
    summary.to_csv(out / "02_complete_path_summary.csv", index=False)
    compare(ROOT / "expected/02_complete_path_summary.csv", out / "02_complete_path_summary.csv", KEYS,
            ["pressure_sum_GW_year", "added_requirement_sum_GW_year", "cushion_consumed_sum_GW_year",
             "fresh_DC_pressure_GW", "fresh_DC_added_requirement_GW", "cushion_consumed_GW", "all_year_growth_scale",
             "endpoint_growth_scale", "min_balance_at_endpoint_scale_GW"])
    expected = pd.read_csv(ROOT / "expected/02_complete_path_summary.csv")
    status = summary.merge(expected[KEYS + ["all_year_scale_status", "endpoint_test_misses_earlier_pressure"]],
                           on=KEYS, suffixes=("", "_expected"), validate="one_to_one")
    status["all_year_status_changed"] = status.all_year_scale_status.ne(status.all_year_scale_status_expected)
    status["endpoint_miss_status_changed"] = status.endpoint_test_misses_earlier_pressure.ne(status.endpoint_test_misses_earlier_pressure_expected)
    status.to_csv(out / "old_new_path_status_comparison.csv", index=False)
    assert len(dc) == 388800 and len(summary) == 38880 and summary.annual_points.eq(10).all()
    core = summary[summary.resource_condition.isin(["anticipated", "prospective"])
                   & summary.linked_response_fraction.eq(0)]
    quantities = ["fresh_DC_pressure_GW", "fresh_DC_added_requirement_GW", "cushion_consumed_GW",
                  "pressure_sum_GW_year", "added_requirement_sum_GW_year", "all_year_growth_scale",
                  "endpoint_growth_scale", "min_balance_at_endpoint_scale_GW"]
    records = []
    group_keys = ["planning_region", "resource_condition", "response_condition"]
    for keys, group in core.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        row.update(paths=len(group), finite_all_year_paths=int(group.all_year_scale_status.eq("finite").sum()),
                   infeasible_backgrounds=int(group.all_year_scale_status.eq("background_below_reference").sum()),
                   endpoint_misses=int(group.endpoint_test_misses_earlier_pressure.sum()),
                   absorbed_2035=int(group.fresh_DC_added_requirement_GW.le(TOL).sum()),
                   positive_added_requirement_2035=int(group.fresh_DC_added_requirement_GW.gt(TOL).sum()))
        for column in quantities:
            row[column + "_min"] = group[column].min()
            row[column + "_max"] = group[column].max()
        records.append(row)
    ranges = pd.DataFrame(records)
    ranges.to_csv(out / "03_core_contribution_ranges.csv", index=False)
    compare(ROOT / "expected/01_core_contribution_ranges.csv", out / "03_core_contribution_ranges.csv",
            group_keys, [column for column in ranges if column not in group_keys])
    record = {"all_passed": True, "annual_points": len(dc), "complete_paths": len(summary),
              "core_paths": len(core), "core_range_cells_matched": False, "old_new_ranges_audited": True,
              "changed_path_status_count": int((status.all_year_status_changed | status.endpoint_miss_status_changed).sum()),
              "inputs_refitted": False}
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
