"""Evaluate IID inclusion without treating its membership as established."""
from pathlib import Path
import json
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "alignment"


def main():
    cube = pd.read_csv(ROOT / "numerical/results/104_annual_scenario_cube_v1_2.csv")
    ca = cube[cube.planning_region.eq("WECC-California") & ~cube.data_center_scenario_id.eq("source_embedded")].copy()
    base_weights = pd.read_csv(ROOT / "numerical/results/124_fig1_spatial_allocation_plotting.csv")
    iid = pd.read_csv(OUT / "iid_included_eight_rule_weights.csv")
    names = base_weights[["source_allocation_case","allocation_case"]].drop_duplicates()
    iid = iid.rename(columns={"allocation_case":"source_allocation_case"}).merge(names,on="source_allocation_case",validate="many_to_one")
    iid = iid[iid.model_region.eq("WECC-California")].set_index("allocation_case").allocation_share
    capture = pd.read_csv(OUT / "iid_included_capture.csv").set_index("footprint_capture_case").capture_share
    ca["iid_included_allocation_share"] = ca.allocation_case.map(iid)
    ca["iid_included_capture_share"] = ca.footprint_capture_case.map(capture).fillna(1)
    factor = ca.iid_included_allocation_share * ca.iid_included_capture_share / (ca.allocation_share * ca.footprint_capture_share)
    ca["iid_included_incremental_load_GW"] = ca.incremental_data_center_load_GW * factor
    ca["iid_included_additional_load_GW"] = ca.iid_included_incremental_load_GW * ca.additional_share
    ca["iid_included_balance_GW"] = ca.base_balance_GW - (1+ca.reference_margin_level_pct/100)*ca.iid_included_additional_load_GW
    ca["iid_included_required_contribution_GW"] = (-ca.iid_included_balance_GW).clip(lower=0)
    assert np.isfinite(ca.iid_included_balance_GW).all()
    ca["interpretation"] = "IID membership sensitivity; not confirmed NERC 2025 membership or electrical connection"
    ca.to_csv(OUT / "iid_included_california_annual_stress.csv.gz",index=False)
    # The presentation cube has fewer allocations than the eight-rule core;
    # evaluate the boundary alternative against the complete core, not its subset.
    path = ROOT / "numerical/scripts/build_core_envelope_acceptance.py"
    spec = importlib.util.spec_from_file_location("boundary_core",path)
    acceptance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acceptance)
    builder = acceptance.load_builder()
    regional = acceptance.build_core_regional_paths(builder,builder.load_config())
    annual,_ = acceptance.build_generic_partial_additional_summary(
        acceptance.load_core_source_paths(),regional[regional.planning_region.eq("WECC-California")])
    core = annual.copy()
    aligned_capture = pd.read_csv(OUT / "new_footprint_capture.csv").set_index("footprint_capture_case").capture_share
    aligned_weights = base_weights[base_weights.planning_region.eq("WECC-California")].set_index("allocation_case").allocation_share
    factor = (core.allocation_case.map(iid) * core.footprint_capture_case.map(capture)
              / (core.allocation_case.map(aligned_weights) * core.footprint_capture_case.map(aligned_capture)))
    core["iid_included_additional_load_GW"] = core.additional_data_center_load_GW * factor
    core["iid_included_balance_GW"] = core.scenario_balance_GW - (1+core.reference_margin_level_pct/100)*core.iid_included_additional_load_GW
    core.to_csv(OUT / "iid_included_california_core_annual.csv.gz",index=False)
    summary = core.groupby(["year","resource_condition","response_condition"]).agg(
        base_proxy_min_GW=("core_partial_additional_balance_GW","min"),base_proxy_max_GW=("core_partial_additional_balance_GW","max"),
        iid_included_min_GW=("iid_included_balance_GW","min"),iid_included_max_GW=("iid_included_balance_GW","max"))
    assert len(summary) == 40
    summary.to_csv(OUT / "iid_boundary_core_annual_envelope.csv")
    report = {"passed":True,"annual_stress_rows":len(ca),"complete_eight_rule_core_annual_rows":len(core),"annual_core_summary_rows":len(summary),
              "membership_confirmed":False,"minimum_iid_included_core_balance_GW":float(core.iid_included_balance_GW.min())}
    (OUT / "iid_boundary_sensitivity_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
