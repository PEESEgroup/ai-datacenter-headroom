"""Refresh county and paired exports from current results, not stale fixtures."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"default_weight_update"


def main():
    parts=pd.read_csv(HERE/"legacy_inputs/county_display_regional_parts.csv",dtype={"GEOID":str})
    regional=pd.read_csv(ROOT/"numerical/results/103_lbnl2025_regional_load_paths.csv")
    for key,value in {"post2030_extension_id":"epri_quarter_rate_taper","footprint_capture_case":"core_active_reported_mw_share","peak_mapping_case":"reference_coincidence","allocation_case":"reported_MW_reference_imputation"}.items():regional=regional[regional[key].eq(value)]
    ref=regional[regional.data_center_scenario_id.eq("lbnl_reference")]
    targets={}
    for year in [2025,2035]:
        sub=ref[ref.year.eq(year)]
        targets[year]=sub.set_index("model_region").allocated_average_power_GW.to_dict()
        targets[year]["Non-ISO"]=sub.average_grid_power_GW.iloc[0]-sum(targets[year].values())
    existing=parts[parts.year.eq(2025)].copy()
    for region,target in targets[2025].items():
        mask=existing.iso.eq(region)
        existing.loc[mask,"gw"]*=target/existing.loc[mask,"gw"].sum()
    records=pd.read_csv(ROOT/"alignment/all_project_assignments.csv",dtype={"GEOID":str})
    records=records[records.status.isin(["Proposed","Approved/Permitted/Under construction"])&~records.explicit_non_data_center&records.reviewed_load_weight_MW.gt(0)].copy()
    records["display_region"]=records.ISO_ASSIGNED.where(records.in_seven_region_footprint,"Non-ISO")
    assert records.GEOID.notna().all()
    catalog=parts.sort_values("gw",ascending=False).drop_duplicates("GEOID").set_index("GEOID")
    names=pd.read_csv(ROOT/"numerical/spatial/census_county_reference_2023.csv",dtype={"GEOID":str}).set_index("GEOID")
    future=[]
    for region,target in targets[2035].items():
        base=existing[existing.iso.eq(region)].groupby("GEOID").gw.sum()
        shape=records[records.display_region.eq(region)].groupby("GEOID").reviewed_load_weight_MW.sum()
        growth=target-base.sum()
        value=base.add(shape/shape.sum()*growth,fill_value=0)
        for geoid,gw in value.items():
            metadata=catalog.loc[geoid].to_dict() if geoid in catalog.index else dict(county=names.loc[geoid,"NAME"],state_abb=names.loc[geoid,"STUSPS"],label=f"{names.loc[geoid,'NAME']}, {names.loc[geoid,'STUSPS']}")
            future.append({**metadata,"GEOID":geoid,"iso":region,"year":2035,"gw":gw})
    parts=pd.concat([existing,pd.DataFrame(future)],ignore_index=True)
    for year in [2025,2035]:
        for region,target in targets[year].items():np.testing.assert_allclose(parts.loc[parts.year.eq(year)&parts.iso.eq(region),"gw"].sum(),target,atol=1e-10)
    parts.to_csv(ROOT/"plotting_tables/county_display_regional_parts.csv",index=False)
    dominant=parts.sort_values("gw",ascending=False).drop_duplicates(["GEOID","year"])
    county=dominant.drop(columns=["gw","rank","allocated_average_MW"]).merge(parts.groupby(["GEOID","year"],as_index=False).gw.sum(),on=["GEOID","year"],validate="one_to_one")
    county["rank"]=county.groupby("year").gw.rank(method="first",ascending=False).astype(int)
    county["allocated_average_MW"]=county.gw*1000
    county.to_csv(ROOT/"plotting_tables/Fig1_county_display_downscaling.csv",index=False)
    regional[regional.year.isin([2025,2035])].to_csv(ROOT/"plotting_tables/Fig1e_regional_average_load.csv",index=False)
    for source,target in [("results/121_fig4_paired_contribution_display.csv","source_data/complete_displayed_paired_paths.csv"),("results/92_annual_balance_driver_decomposition.csv","source_data/annual_driver_decomposition.csv")]:
        pd.read_csv(ROOT/"numerical"/source).to_csv(ROOT/target,index=False)


if __name__=="__main__":main()
