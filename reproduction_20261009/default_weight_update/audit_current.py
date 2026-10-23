"""Check reviewed capacity semantics, paired identities and current reader exports."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NUM=ROOT/"numerical"


def main():
    spec=importlib.util.spec_from_file_location("capacity_audit",NUM/"spatial/allocation_rules.py")
    rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    raw=rules.map_to_regions(rules.harmonize_inventory(NUM/"spatial/project_records.csv","latest_20260917"))
    registry=pd.read_csv(NUM/"spatial/capacity_attribute_review.csv")
    weights=rules.load_capacity_values(raw,"mid")
    excluded=registry[registry.decision.eq("exclude")].source_row
    assert weights[raw.source_row.isin(excluded)].isna().all()
    for row in registry[registry.decision.eq("replace")].itertuples():
        np.testing.assert_allclose(weights[raw.source_row.eq(row.source_row)].iloc[0],row.load_mw_mid)
    core=~raw.explicit_non_data_center&raw.status.isin(rules.CORE_ACTIVE)
    numerator=weights[core&raw.in_seven_region_footprint].sum()
    denominator=weights[core].sum()
    capture=pd.read_csv(ROOT/"alignment/new_footprint_capture.csv").set_index("footprint_capture_case")
    np.testing.assert_allclose([numerator,denominator,numerator/denominator],capture.loc["core_active_reported_mw_share",["numerator","national_denominator","capture_share"]].astype(float),atol=1e-10)
    paired=pd.read_csv(NUM/"results/contribution/01_annual_paired_outcomes.csv.gz")
    b0=paired.fresh_no_new_DC_balance_GW;b=paired.replacement_balance_GW
    np.testing.assert_allclose(b0-b,paired.fresh_DC_pressure_GW,atol=1e-10)
    np.testing.assert_allclose(np.maximum(-b,0)-np.maximum(-b0,0),paired.fresh_DC_added_requirement_GW,atol=1e-10)
    grid=pd.read_csv(ROOT/"physical_intervals/interval_results/pv_storage_fine_grid.csv")
    for relative in ["numerical/results/intervention/pv_storage_fine_grid.csv","plotting_tables/pv_recomputed_full_grid.csv"]:
        pd.testing.assert_frame_equal(grid,pd.read_csv(ROOT/relative),atol=1e-12,rtol=1e-12)
    county=pd.read_csv(ROOT/"plotting_tables/Fig1_county_display_downscaling.csv")
    paths=pd.read_csv(ROOT/"plotting_tables/national_annual_energy_paths.csv")
    for year in [2025,2035]:
        selected=paths[paths.year.eq(year)&paths.data_center_scenario_id.eq("lbnl_reference")&paths.post2030_extension_id.eq("epri_quarter_rate_taper")]
        np.testing.assert_allclose(county[county.year.eq(year)].gw.sum(),selected.electricity_TWh.iloc[0]*1000/8760,atol=1e-10)
    result=dict(passed=True,excluded_generation_records=len(excluded),replaced_load_records=int(registry.decision.eq("replace").sum()),annual_pairs_checked=len(paired),physical_hour_grid_points=len(grid),county_national_totals_match=True)
    (ROOT/"default_weight_update/current_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
