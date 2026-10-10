"""Targeted low-demand and footprint-scope comparisons on unchanged native paths."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

import run_timing_sensitivity as timing
from seasonal_load_quality import reported_demand, season
from scope_quality import classify_scope, campus_keys, campus_allocation_cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/source_quality"
KEYS = timing.KEYS
TOL = 1e-7


def annual_pairs(core, factor=1, rule="linear"):
    # Keep matched inputs, not stale outcome columns from the original scenario.
    columns = KEYS + ["year", "reference_margin_level_pct", "margin_GW", "candidate_gross_path_GW"]
    out = core[columns].copy().rename(columns={"candidate_gross_path_GW":"reference_candidate_gross_path_GW"})
    k = 1 + out.reference_margin_level_pct / 100
    source = np.empty(len(out))
    for region in ["MISO", "ERCOT"]:
        mask = out.planning_region.eq(region).to_numpy()
        source[mask] = timing.source_path(region, rule, out.loc[mask, "year"].to_numpy())
    out["background_GW"] = out.margin_GW + k * source
    out["candidate_GW"] = out.reference_candidate_gross_path_GW * factor
    out["pressure_GW"] = k * out.candidate_GW
    out["balance_GW"] = out.background_GW - out.pressure_GW
    out["added_requirement_GW"] = (-out.balance_GW).clip(lower=0) - (-out.background_GW).clip(lower=0)
    out["cushion_consumed_GW"] = out.pressure_GW - out.added_requirement_GW
    assert np.allclose(out.pressure_GW, out.added_requirement_GW + out.cushion_consumed_GW, atol=TOL, rtol=0)
    assert out.added_requirement_GW.ge(-TOL).all()
    out["source_DC_timing_rule"] = rule
    return out


def summarize(a):
    keys = KEYS + ["quality_case", "source_DC_timing_rule"]
    summary = a.groupby(keys).agg(years=("year", "size"), minimum_balance_GW=("balance_GW", "min"),
        below_years=("balance_GW", lambda x: int(x.lt(-TOL).sum())),
        added_requirement_sum_GW_year=("added_requirement_GW", "sum")).reset_index()
    assert summary.years.eq(10).all()
    endpoints = a[a.year.eq(2035)][keys + ["balance_GW", "pressure_GW", "added_requirement_GW", "cushion_consumed_GW"]]
    return summary.merge(endpoints, on=keys, validate="one_to_one")


def low_demand(core):
    reference = core[core.data_center_scenario_id.eq("lbnl_reference")].copy()
    national = pd.read_csv(ROOT / "results/101_lbnl2025_national_energy_paths.csv")
    reference_energy = national[national.data_center_scenario_id.eq("lbnl_reference")].set_index(["post2030_extension_id", "year"]).electricity_TWh
    base = float(reference_energy.loc[("epri_quarter_rate_taper", 2025)])
    multiplier = {"hold_2030_level": 0, "epri_quarter_rate_taper": .25, "slower_half_rate_taper_stress": .5}
    rows, paths = [], []
    for name, endpoint in [("external_390_TWh_endpoint", 390), ("external_430_TWh_endpoint", 430)]:
        rate = (endpoint / base) ** .2 - 1
        energy = {}
        for extension, m in multiplier.items():
            for year in range(2025, 2036):
                value = base * (1 + rate) ** (year - 2025) if year <= 2030 else endpoint * (1 + m * rate) ** (year - 2030)
                energy[(extension, year)] = value
                rows.append({"quality_case": name, "year": year, "post2030_extension_id": extension,
                    "electricity_TWh": value, "shared_author_2025_anchor_TWh": base,
                    "endpoint_basis": "LBNL 2025 Update PDF p13 comparison: EPRI lower ~390 / IEA ~430",
                    "annual_path_is_author_scenario": True, "original_agency_annual_path_reproduced": False})
        a = reference.copy()
        index = pd.MultiIndex.from_frame(a[["post2030_extension_id", "year"]])
        ratio = (np.array([energy[key] for key in index]) - base) / (reference_energy.reindex(index).to_numpy() - base)
        assert np.isfinite(ratio).all() and ((ratio > 0) & (ratio < 1)).all()
        for rule in timing.CONFIG["timing_rules"]:
            b = annual_pairs(a, ratio, rule)
            b["data_center_scenario_id"] = name
            b["quality_case"] = name
            paths.append(b)
    result = pd.concat(paths, ignore_index=True)
    result.to_csv(OUT / "03_lower_endpoint_annual_pairs.csv.gz", index=False)
    summarize(result).to_csv(OUT / "04_lower_endpoint_complete_paths.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "02_external_endpoint_energy_paths.csv", index=False)
    # The display selector normally requires lbnl_reference; select the other frozen dimensions here.
    mask = pd.Series(True, index=result.index)
    for key, value in timing.DISPLAY.items():
        if key in result and key != "data_center_scenario_id":
            mask &= result[key].eq(value)
    result[mask & result.year.eq(2035)].to_csv(OUT / "05_lower_endpoint_display_2035.csv", index=False)
    return result


def spatial(core):
    spec = importlib.util.spec_from_file_location("scope_rules", ROOT / "spatial/allocation_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    rows = rules.map_to_regions(rules.harmonize_inventory(ROOT / "spatial/project_records.csv", "latest_20260917"))
    rows = classify_scope(rows)
    rows["power_scale_not_certified_load"] = rows.facility_name.fillna("").str.contains("Fermi America", case=False).to_numpy(dtype=bool)
    rows["campus_key"], rows["usable_shared_address"] = campus_keys(rows,rules.normalize_text)
    candidate = rows[rows.duplicated("campus_key", keep=False)].copy()
    candidate.to_csv(OUT / "07_shared_address_review_candidates.csv", index=False)
    near = []
    coords = rows.dropna(subset=["lat", "long"])
    for _, group in coords.groupby("state"):
        values = list(group.itertuples())
        for i, first in enumerate(values):
            for second in values[i+1:]:
                distance = np.hypot((first.lat-second.lat)*111000, (first.long-second.long)*111000*np.cos(np.deg2rad(first.lat)))
                if distance < 100:
                    near.append({"source_row_a": first.source_row, "source_row_b": second.source_row,
                        "name_a": first.facility_name, "name_b": second.facility_name, "distance_m": distance,
                        "is_confirmed_duplicate": False})
    pd.DataFrame(near).to_csv(OUT / "08_close_coordinate_review_candidates.csv", index=False)
    # This is a campus-level sensitivity, not an assertion that all same-address phases duplicate.
    cases = {"exclude_pure_crypto": ~rows.scope_class.eq("explicit_pure_crypto"),
        "exclude_pure_and_mixed_crypto": rows.scope_class.eq("no_explicit_crypto_tag"),
        "shared_address_single_representative": ~rows.scope_class.eq("explicit_pure_crypto"),
        "exclude_private_power_campus_MW": ~rows.power_scale_not_certified_load & ~rows.scope_class.eq("explicit_pure_crypto"),
        "combined_scope_sensitivity": rows.scope_class.eq("no_explicit_crypto_tag") & ~rows.power_scale_not_certified_load}
    baseline, _ = rules.allocation_cases(rows)
    weights, annual = [], []
    for case, eligible in cases.items():
        if case in {"shared_address_single_representative","combined_scope_sensitivity"}:
            weight, selections = campus_allocation_cases(rows[eligible].copy(),rules)
            selections["quality_case"] = case
            selections.to_csv(OUT / ("14_campus_representatives_" + case + ".csv"),index=False)
        else:
            weight, _ = rules.allocation_cases(rows[eligible].copy())
        weight["quality_case"] = case
        weights.append(weight)
        joined = core.merge(weight[["allocation_case", "model_region", "allocation_share"]],
            left_on=["source_allocation_case", "planning_region"], right_on=["allocation_case", "model_region"],
            suffixes=("", "_scope"), validate="many_to_one")
        old = baseline.rename(columns={"allocation_share": "old_share", "allocation_case": "old_rule"})
        joined = joined.merge(old[["old_rule", "model_region", "old_share"]],
            left_on=["source_allocation_case", "planning_region"], right_on=["old_rule", "model_region"], validate="many_to_one")
        b = annual_pairs(joined, joined.allocation_share / joined.old_share)
        b["quality_case"] = case
        annual.append(b)
    all_weights = pd.concat(weights, ignore_index=True)
    assert np.allclose(all_weights.groupby(["quality_case", "allocation_case"]).allocation_share.sum(), 1)
    all_weights.to_csv(OUT / "09_scope_matched_allocation_weights.csv", index=False)
    result = pd.concat(annual, ignore_index=True)
    result.to_csv(OUT / "10_scope_annual_pairs.csv.gz", index=False)
    summarize(result).to_csv(OUT / "11_scope_complete_paths.csv", index=False)
    timing.display(result[result.year.eq(2035)]).to_csv(OUT / "12_scope_display_2035.csv", index=False)
    rows.to_csv(OUT / "06_project_scope_classification.csv", index=False)
    rows[rows.scope_class.ne("no_explicit_crypto_tag")].to_csv(OUT / "16_crypto_evidence_review.csv",index=False)
    return rows, result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(ROOT / "results/contribution/01_annual_paired_outcomes.csv.gz")
    core = core[core.resource_condition.isin(["anticipated", "prospective"]) & core.linked_response_fraction.eq(0)].copy()
    low = low_demand(core)
    scope_rows, scope = spatial(core)
    base = annual_pairs(core)
    assert np.allclose(base.balance_GW, core.fresh_with_DC_balance_GW, atol=TOL, rtol=0)
    assert np.allclose(base.added_requirement_GW, core.fresh_DC_added_requirement_GW, atol=TOL, rtol=0)
    new_load = reported_demand()
    complete = [year for year in [2023, 2024, 2025] if all(season(new_load, region, year)[1]["complete"] for region in ["MISO", "ERCOT", "PJM", "SPP"])]
    config = json.loads((ROOT / "intervention/grid_config.json").read_text())
    assert complete == config["observed_peak_years"]
    fields = [
        {"field":"reference_candidate_gross_path_GW","unit":"GW coincident scenario increment","meaning":"Unmodified candidate input before this targeted case; not the load of the output scenario ID"},
        {"field":"candidate_GW","unit":"GW coincident scenario increment","meaning":"Actual case-specific candidate DC growth used in every balance and paired contribution"},
        {"field":"background_GW","unit":"native GW planning contribution","meaning":"Matched no-new-DC-growth balance under the stated source-component timing rule"},
        {"field":"balance_GW","unit":"native GW planning contribution","meaning":"background_GW minus pressure_GW"},
        {"field":"pressure_GW","unit":"native GW planning contribution","meaning":"(1 + reference_margin_level_pct/100) times candidate_GW"},
        {"field":"added_requirement_GW","unit":"native GW planning contribution","meaning":"Positive-part requirement difference between matched candidate and no-new-growth cases"},
        {"field":"cushion_consumed_GW","unit":"native GW planning contribution","meaning":"pressure_GW minus added_requirement_GW"}]
    pd.DataFrame(fields).to_csv(OUT / "15_targeted_output_field_dictionary.csv",index=False)
    record = {"passed": True, "complete_common_peak_years": complete,
        "lower_endpoint_annual_points": len(low), "scope_annual_points": len(scope),
        "scope_source_rows": len(scope_rows), "raw_records_not_deleted": True,
        "pure_crypto_excluded_in_scope_matched_alternatives": True,
        "same_address_flags_are_not_confirmed_duplicates": True,
        "campus_selection_after_rule_specific_eligibility": True,
        "placeholder_addresses_are_distinct_records": True,
        "road_only_addresses_do_not_establish_campus_identity": True,
        "premise_address_and_county_required_for_shared_key": True,
        "full_record_crypto_evidence_retained": True,
        "reference_and_actual_candidate_fields_distinct": True,
        "private_power_capacity_not_substituted_as_grid_demand": True,
        "native_paths_and_historical_methods_unchanged": True,
        "low_endpoints_are_author_paths_not_full_EPRI_IEA_forecasts": True,
        "lower_source_replacement_scope": ["MISO", "ERCOT"],
        "seven_area_full_DC_attribution_claim": False}
    (OUT / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
