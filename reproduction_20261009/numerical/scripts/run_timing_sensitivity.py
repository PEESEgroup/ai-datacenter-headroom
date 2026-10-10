"""Anchor-preserving annual timing and matched resource/response conditions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "targeted/protocol.json").read_text())
TOL = CONFIG["tolerance_GW"]
KEYS = ["planning_region", "resource_condition", "response_condition", "data_center_scenario_id",
        "post2030_extension_id", "footprint_capture_case", "peak_mapping_case", "allocation_case"]
DISPLAY = json.loads((ROOT / "contribution/protocol.json").read_text())["display"]


def source_path(region, rule, years):
    anchors = CONFIG["official_DC_anchors_GW"][region]
    xs = np.array(sorted(map(int, anchors)))
    ys = np.array([anchors[str(x)] for x in xs], float)
    out = []
    for year in years:
        index = min(np.searchsorted(xs, year, side="right") - 1, len(xs) - 2)
        x = (year - xs[index]) / (xs[index + 1] - xs[index])
        if rule == "earlier_arrival":
            x = np.sqrt(x)
        elif rule == "later_arrival":
            x = x * x
        elif rule == "last_interval_early_plateau" and index == len(xs) - 2:
            x = min(1, (year - xs[index]) / (xs[index + 1] - xs[index] - 1))
        out.append(ys[index] + x * (ys[index + 1] - ys[index]))
    return np.array(out)


def summarize(annual):
    keys = KEYS + ["source_DC_timing_rule"]
    a = annual.copy()
    a["required_GW"] = (-a.candidate_balance_GW).clip(lower=0)
    a["background_required_GW"] = (-a.background_balance_GW).clip(lower=0)
    a["added_requirement_GW"] = a.required_GW - a.background_required_GW
    a["annual_scale"] = np.where((a.background_balance_GW >= -TOL) & (a.pressure_GW > TOL),
                                 a.background_balance_GW.clip(lower=0) / a.pressure_GW, np.nan)
    out = a.groupby(keys).agg(annual_points=("year", "size"),
        minimum_balance_GW=("candidate_balance_GW", "min"),
        below_years=("candidate_balance_GW", lambda x: int((x < -TOL).sum())),
        background_below_years=("background_balance_GW", lambda x: int((x < -TOL).sum())),
        annual_requirement_sum_GW_year=("required_GW", "sum"),
        annual_added_requirement_sum_GW_year=("added_requirement_GW", "sum"),
        all_year_growth_scale=("annual_scale", "min")).reset_index()
    out["all_year_status"] = np.select([out.background_below_years.gt(0), out.all_year_growth_scale.isna()],
        ["background_below_reference", "no_finite_bound"], default="finite")
    out.loc[out.all_year_status.ne("finite"), "all_year_growth_scale"] = np.nan
    end = a[a.year.eq(2035)][keys + ["annual_scale", "candidate_balance_GW", "pressure_GW", "added_requirement_GW"]]
    out = out.merge(end.rename(columns={"annual_scale": "endpoint_growth_scale"}), on=keys, validate="one_to_one")
    paired = a.merge(out[keys + ["endpoint_growth_scale", "all_year_growth_scale"]], on=keys, validate="many_to_one")
    paired["endpoint_scaled_balance_GW"] = paired.background_balance_GW - paired.endpoint_growth_scale * paired.pressure_GW
    out = out.merge(paired.groupby(keys).endpoint_scaled_balance_GW.min().rename("minimum_endpoint_scaled_balance_GW").reset_index(), on=keys, validate="one_to_one")
    out["endpoint_miss"] = out.endpoint_growth_scale.notna() & out.minimum_endpoint_scaled_balance_GW.lt(-TOL)
    finite = paired[paired.all_year_growth_scale.notna()].copy()
    finite["full_limit_balance_GW"] = finite.background_balance_GW - finite.all_year_growth_scale * finite.pressure_GW
    assert finite.full_limit_balance_GW.min() >= -TOL
    assert finite.groupby(keys).full_limit_balance_GW.min().abs().max() < TOL
    limiting = a[a.annual_scale.notna()].sort_values(["annual_scale", "year"]).drop_duplicates(keys)
    out = out.merge(limiting[keys + ["year"]].rename(columns={"year": "limiting_year"}), on=keys, how="left", validate="one_to_one")
    out.loc[out.all_year_status.ne("finite"), "limiting_year"] = np.nan
    return a, out


def resource_frontier(a):
    keys = [k for k in KEYS if k != "resource_condition"] + ["source_DC_timing_rule", "year"]
    lower = a[a.resource_condition.eq("anticipated")][keys + ["candidate_balance_GW"]]
    higher = a[a.resource_condition.eq("prospective")][keys + ["candidate_balance_GW"]]
    both = lower.merge(higher, on=keys, suffixes=("_A", "_P"), validate="one_to_one")
    both["available_native_uplift_GW"] = both.candidate_balance_GW_P - both.candidate_balance_GW_A
    assert both.available_native_uplift_GW.min() >= -TOL
    need = (-both.candidate_balance_GW_A).clip(lower=0)
    both["native_uplift_needed_GW"] = need
    both["required_fraction"] = np.where(need <= TOL, 0,
        np.divide(need, both.available_native_uplift_GW, out=np.full(len(both), np.inf), where=both.available_native_uplift_GW > TOL))
    path_keys = keys[:-1]
    paths = both.groupby(path_keys).required_fraction.max().rename("all_year_fraction").reset_index()
    end = both[both.year.eq(2035)].drop(columns="year").rename(columns={"required_fraction": "endpoint_fraction"})
    paths = paths.merge(end, on=path_keys, validate="one_to_one")
    paths["status"] = np.select([paths.all_year_fraction.eq(0), paths.all_year_fraction.gt(1 + TOL)],
        ["anticipated_sufficient", "listed_prospective_uplift_insufficient"], default="within_listed_uplift")
    check = both.merge(paths[path_keys + ["all_year_fraction"]], on=path_keys, validate="many_to_one")
    ok = check.all_year_fraction.between(0, 1 + TOL)
    assert (check.loc[ok, "candidate_balance_GW_A"] + check.loc[ok, "all_year_fraction"] * check.loc[ok, "available_native_uplift_GW"]).min() >= -TOL
    return both, paths


def display(frame):
    mask = pd.Series(True, index=frame.index)
    for key, value in DISPLAY.items():
        if key in frame.columns:
            mask &= frame[key].eq(value)
    return frame[mask].copy()


def main():
    out = ROOT / "results/targeted"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / "results/contribution/01_annual_paired_outcomes.csv.gz")
    core = original[original.resource_condition.isin(CONFIG["core"]["resources"]) & original.linked_response_fraction.eq(0)].copy()
    native = pd.read_csv(ROOT / "seasonal_resource_ledger_20261004/03_corrected_condition_panel.csv")
    nkeys = ["planning_region", "resource_condition", "response_condition", "year"]
    core = core.merge(native[nkeys + ["gross_peak_demand_MW"]], on=nkeys, validate="many_to_one")
    assert len(core) == 51840
    records, anchors = [], []
    for rule in CONFIG["timing_rules"]:
        changed = core.copy()
        changed["source_DC_timing_rule"] = rule
        for region in CONFIG["official_DC_anchors_GW"]:
            sel = changed.planning_region.eq(region)
            changed.loc[sel, "timed_official_DC_GW"] = source_path(region, rule, changed.loc[sel, "year"].to_numpy())
            anchors.extend({"planning_region": region, "year": int(t), "source_DC_timing_rule": rule, "official_DC_GW": v}
                           for t, v in zip(range(2025, 2036), source_path(region, rule, np.arange(2025, 2036))))
        k = 1 + changed.reference_margin_level_pct / 100
        changed["background_balance_GW"] = changed.margin_GW + k * changed.timed_official_DC_GW
        changed["pressure_GW"] = k * changed.candidate_gross_path_GW
        changed["candidate_balance_GW"] = changed.background_balance_GW - changed.pressure_GW
        changed["implied_non_DC_peak_GW"] = changed.gross_peak_demand_MW / 1000 - changed.timed_official_DC_GW
        records.append(changed)
    annual, paths = summarize(pd.concat(records, ignore_index=True))
    assert len(paths) == 20736 and paths.annual_points.eq(10).all()
    baseline = annual[annual.source_DC_timing_rule.eq("linear")]
    assert np.allclose(baseline.candidate_balance_GW, baseline.fresh_with_DC_balance_GW, atol=TOL, rtol=0)
    assert np.allclose(baseline.pressure_GW, baseline.fresh_DC_pressure_GW, atol=TOL, rtol=0)
    assert annual.implied_non_DC_peak_GW.min() >= -TOL
    endpoint = annual[annual.year.eq(2035)].groupby(KEYS).candidate_balance_GW.agg(["min", "max"])
    assert (endpoint["max"] - endpoint["min"]).abs().max() < TOL
    annual.to_csv(out / "01_anchor_timing_annual.csv.gz", index=False)
    paths.to_csv(out / "02_anchor_timing_path_summary.csv", index=False)
    pd.DataFrame(anchors).to_csv(out / "03_source_anchor_timing_rules.csv", index=False)
    grouping = ["planning_region", "resource_condition", "response_condition", "source_DC_timing_rule"]
    summary = paths.groupby(grouping).agg(paths=("annual_points", "size"), endpoint_misses=("endpoint_miss", "sum"),
        all_year_growth_scale_min=("all_year_growth_scale", "min"), all_year_growth_scale_max=("all_year_growth_scale", "max"),
        below_years_min=("below_years", "min"), below_years_max=("below_years", "max"),
        added_sum_min=("annual_added_requirement_sum_GW_year", "min"), added_sum_max=("annual_added_requirement_sum_GW_year", "max")).reset_index()
    summary.to_csv(out / "04_timing_core_summary.csv", index=False)
    display(paths).to_csv(out / "05_displayed_timing_comparison.csv", index=False)
    resource_annual, resource_paths = resource_frontier(annual)
    resource_annual.to_csv(out / "06_resource_frontier_annual.csv.gz", index=False)
    resource_paths.to_csv(out / "07_resource_frontier_paths.csv", index=False)
    display(resource_paths).to_csv(out / "08_displayed_resource_conditions.csv", index=False)
    linked = original[original.resource_condition.isin(CONFIG["core"]["resources"]) & original.year.eq(2035)].copy()
    linked["fixed_budget_response_GW"] = np.minimum(linked.candidate_linked_response_GW, linked.official_linked_response_GW)
    linked["response_expansion_GW"] = (linked.candidate_linked_response_GW - linked.official_linked_response_GW).clip(lower=0)
    linked["fixed_budget_candidate_balance_GW"] = linked.fresh_with_DC_balance_GW - (1 + linked.reference_margin_level_pct / 100) * linked.response_expansion_GW
    linked["fixed_budget_added_requirement_GW"] = (-linked.fixed_budget_candidate_balance_GW).clip(lower=0) - (-linked.fresh_no_new_DC_balance_GW).clip(lower=0)
    linked["added_requirement_change_GW"] = linked.fixed_budget_added_requirement_GW - linked.fresh_DC_added_requirement_GW
    assert (linked.added_requirement_change_GW >= -TOL).all()
    assert linked.loc[linked.linked_response_fraction.eq(0), "response_expansion_GW"].eq(0).all()
    linked.to_csv(out / "09_fixed_response_budget.csv", index=False)
    facts = {"passed": True, "annual_points": len(annual), "complete_paths": len(paths), "rules": list(CONFIG["timing_rules"]),
             "candidate_time_shapes_per_background": 9, "amplitude_combinations_per_shape": 72,
             "2035_endpoint_results_unchanged": True, "source_total_paths_unchanged": True,
             "timing_scenarios_are_not_probabilities": True, "baseline_physics_or_historical_models_changed": False}
    (out / "verification.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
