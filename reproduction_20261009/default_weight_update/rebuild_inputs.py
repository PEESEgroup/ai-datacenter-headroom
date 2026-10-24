"""Remove generation-scale MW from load weights without deleting source data."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUM = ROOT / "numerical"
OUT = ROOT / "default_weight_update"


def rules_module():
    spec = importlib.util.spec_from_file_location("reviewed_weights", NUM / "spatial/allocation_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    return rules


def capture_table(rows, rules, include_generation=False):
    eligible = ~rows.explicit_non_data_center
    valid = ~rules.generation_scale_mask(rows)
    definitions = {
        "core_active_reported_mw_share": (eligible & rows.status.isin(rules.CORE_ACTIVE), True),
        "broad_active_reported_mw_share": (eligible & rows.status.isin(rules.BROAD_ACTIVE), True),
        "core_active_site_count_share": (eligible & rows.status.isin(rules.CORE_ACTIVE), False),
    }
    output = []
    for name, (mask, capacity) in definitions.items():
        source = rows.reported_mw_mid if include_generation else rules.load_capacity_values(rows, "mid")
        weight = source.fillna(0).clip(lower=0) if capacity else pd.Series(1., index=rows.index)
        if capacity and not include_generation:
            weight = weight.where(valid, 0)
        denominator = weight[mask].sum()
        numerator = weight[mask & rows.in_seven_region_footprint].sum()
        output.append(dict(footprint_capture_case=name, numerator=numerator,
                           national_denominator=denominator, capture_share=numerator/denominator))
    return pd.DataFrame(output)


def main():
    rules = rules_module()
    rows = rules.map_to_regions(rules.harmonize_inventory(NUM / "spatial/project_records.csv", "latest_20260917"))
    rejected = rules.generation_scale_mask(rows)
    reviewed_exclusions = set(pd.read_csv(NUM / "spatial/capacity_attribute_review.csv")
                              .query("decision == 'exclude'").source_row)
    assert set(rows.loc[rejected, "source_row"]) == reviewed_exclusions
    assert rows.loc[rows.source_row.eq(941), "reported_mw_mid"].iloc[0] == 14000
    rows["generation_scale_excluded_from_capacity_weights"] = rejected
    rows["capacity_weight_basis"] = np.where(rejected, "generation scale; excluded", "reported project MW; conditional location evidence")
    rows["reviewed_load_weight_MW"] = rules.load_capacity_values(rows, "mid")
    rows.to_csv(ROOT / "alignment/all_project_assignments.csv", index=False)
    weights, subsets = rules.allocation_cases(rows)
    old_weights, _ = rules.allocation_cases(rows, include_generation_scale=True)
    capture = capture_table(rows, rules)
    legacy_capture = capture_table(rows, rules, include_generation=True)
    legacy_weights = pd.read_csv(OUT / "legacy_inputs/spatial_fixed_total_allocation_sensitivity.csv")
    pd.testing.assert_frame_equal(old_weights, legacy_weights, check_exact=False, atol=1e-12, rtol=1e-12)
    old = pd.read_csv(OUT / "legacy_inputs/new_footprint_capture.csv")
    pd.testing.assert_frame_equal(legacy_capture, old, check_exact=False, atol=1e-9, rtol=1e-12)
    for case, selection in subsets.items():
        if "site_count" in case:
            assert selection.source_row.eq(941).sum() == 1
        else:
            assert not rules.generation_scale_mask(selection).any(), case
    weights.to_csv(NUM / "sources/lbnl_2025_update/spatial_fixed_total_allocation_sensitivity.csv", index=False)
    capture.to_csv(ROOT / "alignment/new_footprint_capture.csv", index=False)
    capture.to_csv(OUT / "current_capture.csv", index=False)
    legacy_capture.to_csv(OUT / "legacy_capture.csv", index=False)
    comparison = old_weights.merge(weights, on=["allocation_case", "model_region"], suffixes=("_legacy", "_default"), validate="one_to_one")
    comparison["share_change_pp"] = 100*(comparison.allocation_share_default-comparison.allocation_share_legacy)
    comparison.to_csv(OUT / "weight_changes.csv", index=False)
    config_path = NUM / "config/lbnl2025_data_center_scenarios_v1_2.json"
    config = json.loads(config_path.read_text())
    values = capture.set_index("footprint_capture_case").capture_share
    for case in config["analyzed_footprint_capture_cases"]:
        if case["id"] in values:
            case["capture_share"] = float(values[case["id"]])
            note = "; reviewed generation-only scales excluded; separately identified load scales retained"
            if note not in case["role"]:
                case["role"] += note
    config["generation_scale_exclusion"] = {
        "source_row": 941, "reported_midpoint_MW": 14000,
        "source": "https://www.sec.gov/Archives/edgar/data/2071778/000207177826000010/frmi-20251231.htm",
        "reason": "Issuer identifies private generation, not equivalent data-center electrical demand",
        "raw_record_retained": True, "count_alternatives_retain_site": True,
        "size_proxy_replacement_allowed": False, "legacy_inclusion_is_sensitivity_only": True}
    config["capacity_attribute_review"] = "spatial/capacity_attribute_review.csv"
    config_path.write_text(json.dumps(config, indent=2)+"\n")
    # The membership alternative must use the same reviewed capacity definition.
    iid = rows.copy()
    crosswalk = pd.read_csv(ROOT / "alignment/project_footprint_crosswalk.csv")
    ids = crosswalk.loc[crosswalk.inside_IID_proxy, "source_row"]
    iid.loc[iid.source_row.isin(ids), "ISO_ASSIGNED"] = "WECC-California"
    iid["in_seven_region_footprint"] = iid.ISO_ASSIGNED.isin(rules.REGIONS)
    rules.allocation_cases(iid)[0].to_csv(ROOT / "alignment/iid_included_eight_rule_weights.csv", index=False)
    capture_table(iid, rules).to_csv(ROOT / "alignment/iid_included_capture.csv", index=False)
    # Coverage is an input description, not a source of scenario totals.
    output = []
    eligible = ~rows.explicit_non_data_center
    for name, mask in {
        "all_records": pd.Series(True, index=rows.index),
        "eligible_data_center_records": eligible,
        "core_active_all_sizes": eligible & rows.status.isin(rules.CORE_ACTIVE),
        "core_active_reported_mw": eligible & rows.status.isin(rules.CORE_ACTIVE) & rows.reported_mw_mid.notna() & ~rejected,
        "broad_active_reported_mw": eligible & rows.status.isin(rules.BROAD_ACTIVE) & rows.reported_mw_mid.notna() & ~rejected,
    }.items():
        chosen = rows[mask].copy()
        chosen["load_weight_MW"] = rules.load_capacity_values(chosen, "mid").fillna(0)
        for region, group in chosen.groupby("ISO_ASSIGNED"):
            output.append(dict(case=name, ISO_ASSIGNED=region, records=len(group),
                reported_mw_mid_GW=group.reported_mw_mid.sum()/1000,
                eligible_load_weight_GW=group.load_weight_MW.sum()/1000,
                record_share_pct=100*len(group)/len(chosen),
                reported_mw_share_pct=100*group.load_weight_MW.sum()/chosen.load_weight_MW.sum()))
    pd.DataFrame(output).to_csv(NUM / "sources/lbnl_2025_update/spatial_mapping_coverage.csv", index=False)
    (OUT / "input_checks.json").write_text(json.dumps({
        "passed": True, "raw_records": len(rows),
        "excluded_generation_records": int(rejected.sum()),
        "excluded_generation_midpoint_MW": float(rows.loc[rejected, "reported_mw_mid"].sum()),
        "capacity_review_registry": "numerical/spatial/capacity_attribute_review.csv",
        "capacity_rules_exclude_generation": True, "no_proxy_reintroduction": True,
        "count_cases_retain_site": True, "legacy_inputs_independently_reconstructed": True}, indent=2)+"\n")
    print(capture.to_string(index=False))


if __name__ == "__main__":
    main()
