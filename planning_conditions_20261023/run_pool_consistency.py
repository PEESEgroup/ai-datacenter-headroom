"""Current paired results subject to a common, annual ERCOT large-load pool.

This supersedes ERCOT published-response paired outputs of the baseline package.
Official-demand balances, all MISO inputs, weather and fitted models are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOL = 1e-7
IDS = ["data_center_scenario_id","post2030_extension_id","footprint_capture_case","peak_mapping_case","allocation_case"]
KEYS = ["planning_region","resource_condition","response_condition",*IDS,"association_grid_position"]
DISPLAY = dict(data_center_scenario_id="lbnl_reference",post2030_extension_id="epri_quarter_rate_taper",
               footprint_capture_case="core_active_reported_mw_share",peak_mapping_case="reference_coincidence",
               allocation_case="reported_MW_reference_imputation")
RULES = ["linear","earlier_arrival","later_arrival","last_interval_early_plateau"]

def source_path(region, rule, years):
    anchors = {2025:0, 2027:6 if region=="MISO" else 6.7,
               2030:14 if region=="MISO" else 23, 2035:18 if region=="MISO" else 25}
    years = np.asarray(years)
    result = np.zeros(len(years))
    for left,right in zip(list(anchors)[:-1],list(anchors)[1:]):
        mask=(years>left)&(years<=right)
        x=(years[mask]-left)/(right-left)
        if rule=="earlier_arrival": x=np.sqrt(x)
        elif rule=="later_arrival": x=x*x
        elif rule=="last_interval_early_plateau" and right==2035:
            x=np.minimum(1,(years[mask]-left)/(right-left-1))
        elif rule not in RULES: raise ValueError(rule)
        result[mask]=anchors[left]+(anchors[right]-anchors[left])*x
    return result

def association_bounds(pool, rule):
    p=pool[pool.year.between(2026,2035)].sort_values("year").copy()
    assert p.year.tolist()==list(range(2026,2036))
    p["official_GW"]=source_path("ERCOT",rule,p.year)
    p["large_load_GW"]=p.total_large_load_MW/1000
    p["provisional_GW"]=p.provisional_curtailment_MW/1000
    base=pool.loc[pool.year.eq(2025),"provisional_curtailment_MW"].item()/1000
    p["available_GW"]=np.minimum(p.official_GW,np.maximum(p.provisional_GW-base,0))
    p["minimum_removed_GW"]=np.maximum(p.provisional_GW-p.large_load_GW+p.official_GW,0)
    assert (p.available_GW+TOL>=p.minimum_removed_GW).all()
    p["annual_minimum_fraction"]=np.divide(p.minimum_removed_GW,p.available_GW,
        out=np.zeros(len(p)),where=p.available_GW.to_numpy()>TOL)
    chi=float(p.annual_minimum_fraction.max())
    assert chi<=1+TOL
    p["common_minimum_fraction"]=chi
    p["binding_year"]=int(p.loc[p.annual_minimum_fraction.idxmax(),"year"])
    p["source_DC_timing_rule"]=rule
    return p

def evaluate(candidates, native, pool, rule="linear", positions=(0,.5,1)):
    assert all(0<=position<=1 for position in positions)
    candidates=candidates[candidates.planning_region.isin(["MISO","ERCOT"])].copy()
    n=native[native.planning_region.isin(["MISO","ERCOT"])
             & native.response_condition.isin(["current_cap","published_all"])].copy()
    a=candidates.merge(n,on=["planning_region","year"],validate="many_to_many")
    a=a.merge(pd.DataFrame({"association_grid_position":positions}),how="cross")
    a["source_DC_timing_rule"]=rule
    a["official_gross_path_GW"]=0.0
    for region in ["MISO","ERCOT"]:
        mask=a.planning_region.eq(region)
        a.loc[mask,"official_gross_path_GW"]=source_path(region,rule,a.loc[mask,"year"])
    bounds=association_bounds(pool,rule)
    a=a.merge(bounds[["year","large_load_GW","provisional_GW","minimum_removed_GW","common_minimum_fraction"]],on="year",validate="many_to_one")
    ercot=a.planning_region.eq("ERCOT")&a.response_condition.eq("published_all")
    a["minimum_association_fraction"]=np.where(ercot,a.common_minimum_fraction,0)
    a["linked_response_fraction"]=a.minimum_association_fraction+(1-a.minimum_association_fraction)*a.association_grid_position
    base=pool.loc[pool.year.eq(2025),"provisional_curtailment_MW"].item()/1000
    a["available_linked_response_GW"]=np.minimum(a.official_gross_path_GW,a.credited_DR_MW/1000)
    ec=a.planning_region.eq("ERCOT")
    a.loc[ec,"available_linked_response_GW"]=np.where(ercot[ec],
        np.minimum(a.loc[ec,"official_gross_path_GW"],np.maximum(a.loc[ec,"provisional_GW"]-base,0)),0)
    a["official_linked_response_GW"]=a.linked_response_fraction*a.available_linked_response_GW
    a["response_per_incremental_GW"]=np.divide(a.official_linked_response_GW,a.official_gross_path_GW,
        out=np.zeros(len(a)),where=a.official_gross_path_GW.to_numpy()>TOL)
    a["candidate_linked_response_GW"]=a.response_per_incremental_GW*a.candidate_GW
    k=1+a.reference_margin_level_pct/100
    a["background_GW"]=a.margin_GW+k*(a.official_gross_path_GW-a.official_linked_response_GW)
    a["pressure_GW"]=k*(a.candidate_GW-a.candidate_linked_response_GW)
    a["balance_GW"]=a.background_GW-a.pressure_GW
    a["added_requirement_GW"]=np.maximum(-a.balance_GW,0)-np.maximum(-a.background_GW,0)
    a["cushion_consumed_GW"]=a.pressure_GW-a.added_requirement_GW
    a["annual_scale"]=np.divide(np.maximum(a.background_GW,0),a.pressure_GW,
        out=np.full(len(a),np.nan),where=(a.background_GW.to_numpy()>=-TOL)&(a.pressure_GW.to_numpy()>TOL))
    a["background_pool_slack_GW"]=np.where(ercot,a.large_load_GW-a.official_gross_path_GW-a.provisional_GW+a.official_linked_response_GW,np.nan)
    a["candidate_pool_slack_GW"]=np.where(ercot,a.background_pool_slack_GW+a.candidate_GW-a.candidate_linked_response_GW,np.nan)
    a["fixed_budget_response_GW"]=np.minimum(a.official_linked_response_GW,a.candidate_linked_response_GW)
    a["fixed_budget_balance_GW"]=a.balance_GW-k*(a.candidate_linked_response_GW-a.fixed_budget_response_GW)
    a["fixed_budget_pool_slack_GW"]=np.where(ercot,a.background_pool_slack_GW+a.candidate_GW-a.fixed_budget_response_GW,np.nan)
    for field in ["background_pool_slack_GW","candidate_pool_slack_GW","fixed_budget_pool_slack_GW"]:
        assert a.loc[ercot,field].min()>=-TOL,field
    assert (a.official_gross_path_GW[ec]<=a.large_load_GW[ec]+TOL).all()
    np.testing.assert_allclose(a.background_GW-a.balance_GW,a.pressure_GW,atol=TOL,rtol=0)
    return a

def summary(a, extra=()):
    keys=KEYS+["source_DC_timing_rule",*extra]
    s=a.groupby(keys,dropna=False).agg(annual_points=("year","size"),
        background_below_years=("background_GW",lambda x:int((x<-TOL).sum())),
        candidate_below_years=("balance_GW",lambda x:int((x<-TOL).sum())),
        minimum_candidate_balance_GW=("balance_GW","min"),
        pressure_sum_GW_year=("pressure_GW","sum"),
        added_requirement_sum_GW_year=("added_requirement_GW","sum"),
        all_year_growth_scale=("annual_scale","min")).reset_index()
    assert s.annual_points.eq(10).all()
    s["all_year_scale_status"]=np.select([s.background_below_years.gt(0),s.all_year_growth_scale.isna()],
        ["background_below_reference","no_finite_algebraic_bound"],default="finite")
    s.loc[s.all_year_scale_status.ne("finite"),"all_year_growth_scale"]=np.nan
    end=a[a.year.eq(2035)][keys+["background_GW","balance_GW","pressure_GW","added_requirement_GW","cushion_consumed_GW","annual_scale","linked_response_fraction"]]
    s=s.merge(end.rename(columns={"annual_scale":"endpoint_growth_scale"}),on=keys,validate="one_to_one")
    check=a.merge(s[keys+["endpoint_growth_scale","all_year_growth_scale"]],on=keys,validate="many_to_one")
    check["endpoint_scaled_balance_GW"]=check.background_GW-check.endpoint_growth_scale*check.pressure_GW
    s=s.merge(check.groupby(keys,dropna=False).endpoint_scaled_balance_GW.min().rename("min_balance_at_endpoint_scale_GW").reset_index(),on=keys,validate="one_to_one")
    s["endpoint_miss"]=s.endpoint_growth_scale.notna()&s.min_balance_at_endpoint_scale_GW.lt(-TOL)
    finite=check[check.all_year_growth_scale.notna()]
    assert (finite.background_GW-finite.all_year_growth_scale*finite.pressure_GW).min()>=-TOL
    return s

def display(a, include_scenario=True):
    mask=pd.Series(True,index=a.index)
    for key,value in DISPLAY.items():
        if key=="data_center_scenario_id" and not include_scenario: continue
        if key in a: mask &=a[key].eq(value)
    return a[mask].copy()

def save(frame,path):
    frame.to_csv(path,index=False,compression={"method":"gzip","mtime":0} if path.suffix==".gz" else None)

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    inputs=HERE/"pool_inputs"
    manifest=HERE/"POOL_INPUT_MANIFEST.json"
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text())["files"].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,"Input changed: "+name
    candidates=pd.read_csv(inputs/"candidate_paths.csv.gz")
    native=pd.read_csv(inputs/"native_conditions.csv")
    pool=pd.read_csv(inputs/"ercot_large_load_source.csv")
    ncheck=native[native.planning_region.eq("ERCOT")].groupby("year").provisional_large_load_response_MW
    assert ncheck.nunique().eq(1).all()
    np.testing.assert_allclose(ncheck.first().sort_index(),pool[pool.year.between(2026,2035)].sort_values("year").provisional_curtailment_MW,atol=TOL,rtol=0)
    bounds=pd.concat([association_bounds(pool,r) for r in RULES],ignore_index=True)
    save(bounds,out/"annual_pool_bounds.csv")
    core=evaluate(candidates,native,pool)
    save(core,out/"annual_pairs.csv.gz")
    paths=summary(core)
    save(paths,out/"complete_paths.csv.gz")
    save(display(core),out/"displayed_pairs.csv")
    primary=core[core.association_grid_position.eq(0)&core.resource_condition.isin(["anticipated","prospective"])]
    save(primary,out/"primary_annual.csv.gz")
    primary_paths=summary(primary)
    save(primary_paths,out/"primary_complete_paths.csv")
    save(display(primary_paths),out/"displayed_complete_paths.csv")
    geo=pd.read_csv(inputs/"geographic_candidates.csv")
    reference=display(primary[primary.year.eq(2035)])
    geo=geo.merge(reference[["planning_region","resource_condition","response_condition","candidate_GW","pressure_GW","background_GW"]],
        left_on=["region","resources","response"],right_on=["planning_region","resource_condition","response_condition"],validate="many_to_one")
    geo["pressure_GW"]=geo.pressure_GW*geo.candidate_peak_increment_GW/geo.candidate_GW
    geo["with_growth_balance_GW"]=geo.background_GW-geo.pressure_GW
    geo["growth_added_requirement_GW"]=np.maximum(-geo.with_growth_balance_GW,0)-np.maximum(-geo.background_GW,0)
    save(geo,out/"geographic_assignment_results.csv")
    ranges=[]
    for key,g in primary.groupby(["planning_region","resource_condition","response_condition","year"]):
        row=dict(zip(["planning_region","resource_condition","response_condition","year"],key),paths=len(g))
        for field in ["pressure_GW","balance_GW","added_requirement_GW","cushion_consumed_GW"]:
            row[field+"_min"],row[field+"_max"]=g[field].min(),g[field].max()
        ranges.append(row)
    save(pd.DataFrame(ranges),out/"annual_core_ranges.csv")
    timed=[]
    core_native=native[native.resource_condition.isin(["anticipated","prospective"])]
    for rule in RULES:
        timed.append(evaluate(candidates,core_native,pool,rule,positions=(0,)))
    timing=pd.concat(timed,ignore_index=True)
    save(timing,out/"timing_annual.csv.gz")
    save(summary(timing),out/"timing_complete_paths.csv")
    for name in ["lower","scope"]:
        c=pd.read_csv(inputs/(name+"_candidates.csv.gz"))
        rules=RULES if name=="lower" else ["linear"]
        a=pd.concat([evaluate(c,core_native,pool,r,positions=(0,)) for r in rules],ignore_index=True)
        assert len(a)==(138240 if name=="lower" else 259200)
        save(a,out/(name+"_annual.csv.gz"))
        save(summary(a,extra=("quality_case",)),out/(name+"_complete_paths.csv.gz"))
        shown=display(a[a.year.eq(2035)],include_scenario=name!="lower")
        assert len(shown)>0
        save(shown,out/(name+"_display_2035.csv"))
    # Diagnose, rather than silently retain, the previous inadmissible backgrounds.
    linear=bounds[bounds.source_DC_timing_rule.eq("linear")].copy()
    rejected=[]
    for chi in [0,.5,1]:
        for r in linear.itertuples():
            slack=r.large_load_GW-r.official_GW-r.provisional_GW+chi*r.available_GW
            rejected.append(dict(year=r.year,legacy_association_fraction=chi,background_pool_slack_GW=slack,
                                 passes_pool_bound=bool(slack>=-TOL)))
    save(pd.DataFrame(rejected),out/"legacy_association_check.csv")
    # A resource-only perturbation of 2034; every other annual input stays fixed.
    m=display(primary)
    m=m[m.planning_region.eq("MISO")&m.resource_condition.eq("anticipated")&m.response_condition.eq("published_all")].sort_values("year")
    assert len(m)==10
    endpoint=m.iloc[-1].background_GW/m.iloc[-1].pressure_GW
    gap=float(-(m.background_GW-endpoint*m.pressure_GW).min())
    perturb=[]
    for delta in [-1,-.5,0,.25,.5,gap,.75,1]:
        b=m.background_GW.to_numpy().copy(); b[m.year.to_numpy()==2034]+=delta
        ratios=b/m.pressure_GW.to_numpy()
        perturb.append(dict(resource_change_2034_GW=delta,endpoint_growth_scale=endpoint,
            all_year_growth_scale=float(ratios.min()),binding_year=int(m.year.to_numpy()[ratios.argmin()]),
            minimum_balance_at_endpoint_scale_GW=float((b-endpoint*m.pressure_GW.to_numpy()).min())))
    save(pd.DataFrame(perturb),out/"miso_2034_resource_perturbation.csv")
    facts=dict(passed=True,annual_pairs=len(core),complete_paths=len(paths),
        primary_annual_points=len(primary),primary_complete_paths=len(primary_paths),
        ERCOT_linear_common_minimum_fraction=float(linear.common_minimum_fraction.iloc[0]),
        ERCOT_linear_binding_year=int(linear.binding_year.iloc[0]),
        minimum_background_pool_slack_GW=float(core.background_pool_slack_GW.min()),
        minimum_candidate_pool_slack_GW=float(core.candidate_pool_slack_GW.min()),
        miso_endpoint_omission_GW=gap,main_model_refitted=False,
        source_time_mapping="End-of-year DC anchors used as summer planning equivalents; annual source total and provisional response retain summer basis",
        primary_association="Minimum common full-horizon fraction satisfying both load-pool sides; sensitivity at feasible-interval midpoint and upper endpoint")
    (out/"verification.json").write_text(json.dumps(facts,indent=2)+"\n")
    print(json.dumps(facts,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"pool_results")
    main(parser.parse_args().output)
