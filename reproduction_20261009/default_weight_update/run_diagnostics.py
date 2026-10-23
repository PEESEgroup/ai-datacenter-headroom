"""Rebase the existing assignment and targeted campus checks on reviewed inputs."""
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT
NUM=PKG/"numerical"
OUT=PKG/"default_weight_update"


def main():
    previous=pd.read_csv(OUT/"legacy_inputs/geographic_assignment_candidates.csv")
    current=pd.read_csv(PKG/"alignment/all_project_assignments.csv")
    core=current[current.status.isin(["Proposed","Approved/Permitted/Under construction"])&~current.explicit_non_data_center&current.in_seven_region_footprint&current.reviewed_load_weight_MW.gt(0)].copy()
    cols=["source_row","candidate_service_regions","mapping_class"]
    core=core.merge(previous[cols],on="source_row",how="left",validate="one_to_one")
    assert core.mapping_class.notna().all() and len(core)==269
    core.to_csv(OUT/"geographic_records_used.csv",index=False)
    candidates=core.candidate_service_regions.fillna("").map(lambda x:set(x.split("|"))-{ "" })
    ambiguous=core.mapping_class.eq("ambiguous_current_supported")
    annual=pd.read_csv(NUM/"results/contribution/01_annual_paired_outcomes.csv.gz")
    for field,value in {"data_center_scenario_id":"lbnl_reference","post2030_extension_id":"epri_quarter_rate_taper","footprint_capture_case":"core_active_reported_mw_share","peak_mapping_case":"reference_coincidence","allocation_case":"reported_MW_reference_imputation","linked_response_fraction":0}.items():annual=annual[annual[field].eq(value)]
    annual=annual[annual.resource_condition.isin(["anticipated","prospective"])]
    end=annual[annual.year.eq(2035)]
    weights=core.groupby("ISO_ASSIGNED").reviewed_load_weight_MW.sum()
    out=[]
    for case in end.itertuples():
        region=case.planning_region
        outbound=core[ambiguous&core.ISO_ASSIGNED.eq(region)]
        inbound=core[ambiguous&core.ISO_ASSIGNED.ne(region)&candidates.map(lambda c:region in c)]
        for name,weight in [("retained",weights[region]),("minimum_assigned_weight",weights[region]-outbound.reviewed_load_weight_MW.sum()),("maximum_assigned_weight",weights[region]+inbound.reviewed_load_weight_MW.sum())]:
            factor=weight/weights[region]
            pressure=case.fresh_DC_pressure_GW*factor
            b0=case.fresh_no_new_DC_balance_GW
            balance=b0-pressure
            out.append(dict(region=region,resources=case.resource_condition,response=case.response_condition,assignment=name,assigned_MW=weight,seven_region_MW=weights.sum(),candidate_peak_increment_GW=case.candidate_gross_path_GW*factor,pressure_GW=pressure,background_GW=b0,with_growth_balance_GW=balance,growth_added_requirement_GW=max(-balance,0)-max(-b0,0),records_movable_out=len(outbound),records_movable_in=len(inbound)))
    geo=pd.DataFrame(out)
    geo.to_csv(OUT/"geographic_assignment_results.csv",index=False)
    # Fermi and the other generation scales no longer need a default keep/remove
    # justification. Test the two remaining source-supported campus weights.
    vertices=[]
    for matrix,frontier in product([0,1],repeat=2):
        reduced=weights.copy()
        for sid,keep in [(1120,matrix),(935,frontier)]:
            r=core[core.source_row.eq(sid)].iloc[0]
            reduced[r.ISO_ASSIGNED]-=(1-keep)*r.reviewed_load_weight_MW
        for case in end.itertuples():
            region=case.planning_region
            factor=(reduced[region]/reduced.sum())/(weights[region]/weights.sum())
            p=case.fresh_DC_pressure_GW*factor;b=case.fresh_no_new_DC_balance_GW-p
            vertices.append(dict(keep_matrix=matrix,keep_frontier=frontier,region=region,resources=case.resource_condition,response=case.response_condition,pressure_GW=p,balance_GW=b,added_GW=max(-b,0)-max(-case.fresh_no_new_DC_balance_GW,0)))
    pd.DataFrame(vertices).to_csv(OUT/"remaining_campus_retention_cases.csv",index=False)
    # Isolate the shape factor from all other scenario choices.
    mapping=pd.read_csv(NUM/"results/102_lbnl2025_peak_mapping_crosswalk.csv")
    ranges=[]
    for r in end.itertuples():
        shape=mapping[mapping.model_region.eq(r.planning_region)]
        reference=shape[shape.peak_mapping_case.eq("reference_coincidence")].gamma.iloc[0]
        ranges.append(dict(region=r.planning_region,resources=r.resource_condition,response=r.response_condition,reference_GW=r.fresh_DC_pressure_GW,shape_min_GW=r.fresh_DC_pressure_GW*shape.gamma.min()/reference,shape_max_GW=r.fresh_DC_pressure_GW*shape.gamma.max()/reference))
    pd.DataFrame(ranges).to_csv(OUT/"isolated_shape_ranges.csv",index=False)
    miso=annual[annual.planning_region.eq("MISO")&annual.resource_condition.eq("anticipated")&annual.response_condition.eq("published_all")]
    end_scale=miso[miso.year.eq(2035)].annual_growth_scale_limit.iloc[0]
    gap=(miso.fresh_no_new_DC_balance_GW-end_scale*miso.fresh_DC_pressure_GW).min()
    legacy=pd.read_csv(OUT/"legacy_inputs/121_fig4_paired_contribution_display.csv")
    legacy=legacy[legacy.planning_region.eq("MISO")&legacy.resource_condition.eq("anticipated")&legacy.response_condition.eq("published_all")&legacy.linked_response_fraction.eq(0)]
    terminal=legacy[legacy.year.eq(2035)].iloc[0]
    old_scale=terminal.source_path_removed_balance_GW/terminal.dc_path_pressure_contribution_GW
    old_gap=(legacy.source_path_removed_balance_GW-old_scale*legacy.dc_path_pressure_contribution_GW).min()
    np.testing.assert_allclose(gap,old_gap,atol=1e-10)
    report=dict(passed=True,reviewed_capacity_core_records=len(core),geographic_assignment_rows=len(geo),remaining_campus_cases=len(vertices),endpoint_scaled_MISO_gap_GW=float(gap),unchanged_endpoint_scaled_gap=True)
    (OUT/"additional_check_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
