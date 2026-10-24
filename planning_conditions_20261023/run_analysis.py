"""Reproduce joint annual planning conditions and a fixed-background PJM vintage stress."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
TOL = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summer_forecast(path):
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book["Table B1"]
    rows = list(sheet.values)
    header_number, header = next((i, row) for i, row in enumerate(rows, 1)
                                 if 2026 in row and 2035 in row)
    row_number, values = next((i, row) for i, row in enumerate(rows, 1) if row[0] == "PJM RTO")
    result = []
    for column, year in enumerate(header, 1):
        if isinstance(year, int) and 2026 <= year <= 2035:
            result.append(dict(year=year, gross_MW=float(values[column-1]),
                               sheet="Table B1", row=row_number, column=column,
                               header_row=header_number))
    book.close()
    frame = pd.DataFrame(result).sort_values("year")
    assert frame.year.tolist() == list(range(2026, 2036))
    return frame


def native_vectors(native, region):
    frame = native[native.planning_region.eq(region)].sort_values("year").copy()
    assert frame.year.tolist() == list(range(2026, 2036))
    return frame


def minimum_growth_limit(background, pressure):
    background, pressure = np.asarray(background), np.asarray(pressure)
    if np.any(background < -TOL):
        return np.nan, -1
    positive = pressure > TOL
    if not positive.any():
        return np.inf, -1
    ratios = np.where(positive, np.maximum(background, 0) / np.where(positive, pressure, 1), np.inf)
    index = int(np.argmin(ratios))
    return float(ratios[index]), index


def required_fraction(need, available):
    need, available = np.asarray(need), np.asarray(available)
    if np.all(need <= TOL):
        return 0.0, -1
    ratios = np.where(need <= TOL, 0.0,
                      np.divide(need, available, out=np.full(need.shape, np.inf), where=available > TOL))
    index = int(np.argmax(ratios))
    return float(ratios[index]), index


def envelope_breakpoints(intercepts, slopes, years, envelope):
    candidates = [0.0, 1.0]
    for a, b in itertools.combinations(range(len(intercepts)), 2):
        denominator = slopes[a] - slopes[b]
        if abs(denominator) > TOL:
            point = (intercepts[b] - intercepts[a]) / denominator
            if TOL < point < 1 - TOL:
                candidates.append(float(point))
    candidates = sorted(set(candidates))
    choose = np.argmin if envelope == "minimum" else np.argmax
    output = []
    for left, right in zip(candidates[:-1], candidates[1:]):
        midpoint = (left + right) / 2
        index = int(choose(intercepts + midpoint * slopes))
        if output and output[-1]["binding_year"] == years[index]:
            output[-1]["resource_share_end"] = right
        else:
            output.append(dict(resource_share_start=left, resource_share_end=right,
                               binding_year=int(years[index])))
    return output


def miso_conditions(paths, native, protocol):
    n = native_vectors(native, "MISO")
    years = n.year.to_numpy()
    uplift = (n.prospective_native_MW - n.anticipated_native_MW).to_numpy() / 1000
    assert uplift.min() >= -TOL
    u_values = np.linspace(0, 1, protocol["resource_share_plot_points"])
    curves, anchors, segments, needs = [], [], [], []
    for rule in protocol["MISO"]["source_timing_rules"]:
        p = paths[paths.source_DC_timing_rule.eq(rule)].sort_values("year")
        assert p.year.tolist() == years.tolist()
        b0, pressure = p.background_balance_GW.to_numpy(), p.pressure_GW.to_numpy()
        assert pressure.min() > TOL
        np.testing.assert_allclose(b0-pressure, p.candidate_balance_GW, atol=TOL, rtol=0)
        for u in u_values:
            background = b0 + u * uplift
            limit, index = minimum_growth_limit(background, pressure)
            endpoint = background[-1] / pressure[-1]
            residual = background - limit * pressure
            assert residual.min() >= -TOL and abs(residual[index]) < TOL
            curves.append(dict(source_timing_rule=rule, resource_share=float(u),
                               all_year_growth_scale=limit, endpoint_growth_scale=float(endpoint),
                               binding_year=int(years[index]),
                               min_balance_at_endpoint_scale_GW=float((background-endpoint*pressure).min())))
        for u in protocol["reported_resource_shares"]:
            limit, index = minimum_growth_limit(b0 + u*uplift, pressure)
            anchors.append(dict(source_timing_rule=rule, resource_share=u,
                                all_year_growth_scale=limit,
                                endpoint_growth_scale=float((b0[-1]+u*uplift[-1])/pressure[-1]),
                                binding_year=int(years[index])))
        for scale in protocol["reported_MISO_growth_scales"]:
            u, index = required_fraction(scale*pressure-b0, uplift)
            residual = b0 - scale*pressure + u*uplift
            assert residual.min() >= -TOL
            needs.append(dict(source_timing_rule=rule, growth_scale=scale,
                              required_resource_share=u, binding_year=int(years[index]) if u > TOL else 0,
                              endpoint_resource_share=max(0.0, float((scale*pressure[-1]-b0[-1])/uplift[-1]))))
        for row in envelope_breakpoints(b0/pressure, uplift/pressure, years, "minimum"):
            segments.append(dict(source_timing_rule=rule, **row))
    return pd.DataFrame(curves), pd.DataFrame(anchors), pd.DataFrame(segments), pd.DataFrame(needs)


def ercot_conditions(native, protocol):
    n = native_vectors(native, "ERCOT")
    years = n.year.to_numpy()
    k = 1 + n.reference_margin_level_pct.to_numpy() / 100
    lower = n.anticipated_native_MW.to_numpy() / 1000
    uplift = (n.prospective_native_MW - n.anticipated_native_MW).to_numpy() / 1000
    gross = n.gross_peak_demand_MW.to_numpy() / 1000
    ordinary = n.ordinary_response_MW.to_numpy() / 1000
    flex = n.provisional_large_load_response_MW.to_numpy() / 1000
    b0 = lower - k*(gross-ordinary)
    np.testing.assert_allclose(b0+k*flex,
        lower-k*n.published_net_demand_MW.to_numpy()/1000, atol=TOL, rtol=0)
    assert uplift.min() >= -TOL and flex.min() > TOL
    curves = []
    for u in np.linspace(0, 1, protocol["resource_share_plot_points"]):
        fraction, index = required_fraction(-b0-u*uplift, k*flex)
        residual = b0+u*uplift+k*flex*fraction
        assert residual.min() >= -TOL
        if fraction > TOL:
            assert (b0+u*uplift+k*flex*(fraction-1e-6)).min() < -TOL
        curves.append(dict(resource_share=float(u), all_year_provisional_fraction=fraction,
                           endpoint_provisional_fraction=max(0.0, float((-b0[-1]-u*uplift[-1])/(k[-1]*flex[-1]))),
                           binding_year=int(years[index]) if fraction > TOL else 0,
                           minimum_balance_GW=float(residual.min())))
    segments = envelope_breakpoints(np.r_[-b0/(k*flex), 0],
                                    np.r_[-uplift/(k*flex), 0], np.r_[years, 0], "maximum")
    u_zero, index = required_fraction(-b0, uplift)
    end_need = max(0.0, float(-b0[-1]/k[-1]))
    anchors = [dict(resource_share=u, all_year_provisional_fraction=required_fraction(-b0-u*uplift,k*flex)[0])
               for u in protocol["reported_resource_shares"]]
    facts = dict(required_effective_provisional_GW_2035=end_need,
                 provisional_credit_GW_2035=float(flex[-1]),
                 required_provisional_fraction=required_fraction(-b0,k*flex)[0],
                 resource_share_without_provisional=u_zero,
                 no_provisional_resource_binding_year=int(years[index]),
                 gross_peak_GW_2035=float(gross[-1]),
                 effective_reduction_share_of_gross_peak=end_need/float(gross[-1]))
    return pd.DataFrame(curves), pd.DataFrame(anchors), pd.DataFrame(segments), facts


def pjm_stress(native, old, new):
    n = native_vectors(native, "PJM")
    forecast = old.merge(new, on="year", suffixes=("_2025", "_2026"), validate="one_to_one")
    np.testing.assert_array_equal(n.gross_peak_demand_MW, forecast.gross_MW_2025)
    forecast["delta_gross_GW"] = (forecast.gross_MW_2026-forecast.gross_MW_2025)/1000
    forecast["reference_margin_pct"] = n.reference_margin_level_pct.to_numpy()
    k = 1+n.reference_margin_level_pct/100
    forecast["delta_balance_GW"] = -(1+forecast.reference_margin_pct/100)*forecast.delta_gross_GW
    annual, summaries = [], []
    for resource in ["anticipated", "prospective"]:
        for response in ["current_only", "published"]:
            r = n[resource+"_native_MW"].to_numpy()/1000
            f = n["current_cap_response_MW" if response=="current_only" else "published_response_MW"].to_numpy()/1000
            for vintage in [2025, 2026]:
                gross = forecast["gross_MW_"+str(vintage)].to_numpy()/1000
                balance = r-k.to_numpy()*(gross-f)
                for year,g,rr,ff,kk,b in zip(n.year,gross,r,f,k,balance):
                    annual.append(dict(resource_condition=resource,response_condition=response,
                        demand_vintage=vintage,year=int(year),gross_GW=float(g),resource_GW=float(rr),
                        response_GW=float(ff),requirement_multiplier=float(kk),balance_GW=float(b),
                        required_contribution_GW=float(max(0,-b)),demand_change_to_zero_GW=float(b/kk)))
                below=balance < -TOL
                summaries.append(dict(resource_condition=resource,response_condition=response,
                    demand_vintage=vintage,first_below_year=int(n.year.to_numpy()[below][0]) if below.any() else 0,
                    years_below=int(below.sum()),balance_2035_GW=float(balance[-1]),
                    maximum_requirement_GW=float(np.maximum(-balance,0).max())))
    annual = pd.DataFrame(annual)
    for _,group in annual.groupby(["resource_condition","response_condition"]):
        a=group[group.demand_vintage.eq(2025)].sort_values("year")
        b=group[group.demand_vintage.eq(2026)].sort_values("year")
        np.testing.assert_allclose(b.balance_GW.to_numpy()-a.balance_GW.to_numpy(),forecast.delta_balance_GW,atol=TOL,rtol=0)
    a=annual[annual.resource_condition.eq("anticipated")&annual.response_condition.eq("published")]
    forecast=forecast.merge(a.pivot(index="year",columns="demand_vintage",values="balance_GW").rename(
        columns={2025:"anticipated_published_balance_2025_GW",2026:"anticipated_published_balance_2026_GW"}),on="year")
    return forecast, annual, pd.DataFrame(summaries)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest=HERE/"INPUT_MANIFEST.json"
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text())["files"].items():
            assert sha(HERE/name)==digest,"Input bytes differ: "+name
    protocol=json.loads((HERE/"EXPERIMENT_PROTOCOL.json").read_text())
    native=pd.read_csv(HERE/"inputs/native_inputs.csv")
    paths=pd.read_csv(HERE/"inputs/miso_display_timing_paths.csv")
    old=summer_forecast(HERE/"sources/2025-load-report-tables.xlsx")
    new=summer_forecast(HERE/"sources/2026-load-report-tables.xlsx")
    mc,ma,ms,mn=miso_conditions(paths,native,protocol)
    ec,ea,es,ef=ercot_conditions(native,protocol)
    vf,va,vs=pjm_stress(native,old,new)
    anchors=pd.read_csv(HERE/"inputs/source_growth_and_response_anchors.csv")
    dc=anchors[anchors.region.eq("ERCOT")&anchors.year.eq(2035)&anchors.field.eq("source_DC_growth_anchor_GW")]
    assert len(dc)==1 and dc.source_value.iloc[0]==dc.saved_value.iloc[0]
    ef["identified_new_DC_increment_GW_2035"]=float(dc.source_value.iloc[0])
    ef["conditional_minimum_relief_outside_that_increment_GW"]=max(0,ef["required_effective_provisional_GW_2035"]-float(dc.source_value.iloc[0]))
    ef["comparison_assumption"]="Same 2035 peak-demand basis; full 25-GW new-DC increment assumed reducible; not a claim about all existing data-center load or actual delivery"
    for name,frame in [("miso_joint_frontier",mc),("miso_frontier_anchors",ma),("miso_binding_segments",ms),
                       ("miso_resource_requirements",mn),("ercot_joint_frontier",ec),("ercot_frontier_anchors",ea),
                       ("ercot_binding_segments",es),("pjm_forecast_vintages",vf),
                       ("pjm_vintage_annual_conditions",va),("pjm_vintage_summary",vs)]:
        frame.to_csv(output/(name+".csv"),index=False)
    write_json(output/"ercot_scale_comparison.json",ef)
    source_urls={
        "2025-load-report-tables.xlsx":"https://www.pjm.com/-/media/DotCom/planning/res-adeq/load-forecast/2025-load-report-tables.xlsx",
        "2026-load-report-tables.xlsx":"https://www.pjm.com/-/media/DotCom/planning/res-adeq/load-forecast/2026-load-report-tables.xlsx",
        "2025-load-report.pdf":"https://www.pjm.com/-/media/DotCom/library/reports-notices/load-forecast/2025-load-report.pdf",
        "2026-load-report.pdf":"https://www.pjm.com/-/media/DotCom/library/reports-notices/load-forecast/2026-load-report.pdf"}
    write_json(output/"source_register.json",{"retrieved":"2026-10-23","sources":[dict(file="sources/"+name,url=url,sha256=sha(HERE/"sources"/name)) for name,url in source_urls.items()],
        "workbook_selection":"Table B1, year columns, unique PJM RTO row; not digitized from charts",
        "native_input_origin":"Unchanged NERC 2025 source-audit panel in public baseline commit 3c0e7c6de99d11c3127ce4c4e6d9da993184de58"})
    linear=ma[ma.source_timing_rule.eq("linear")&ma.resource_share.eq(0)].iloc[0]
    check = paths[paths.source_DC_timing_rule.eq("linear")].sort_values("year")
    ratios = check.background_balance_GW / check.pressure_GW
    np.testing.assert_allclose([linear.endpoint_growth_scale, linear.all_year_growth_scale],
                              [ratios.iloc[-1], ratios.min()], atol=1e-12, rtol=0)
    reference=mn[mn.growth_scale.eq(1)]
    facts=dict(passed=True,baseline_2025_PJM_matches_all_ten_native_gross_peaks=True,
        MISO_anticipated_endpoint_growth_scale=float(linear.endpoint_growth_scale),
        MISO_anticipated_all_year_growth_scale=float(linear.all_year_growth_scale),
        MISO_reference_required_resource_share_min=float(reference.required_resource_share.min()),
        MISO_reference_required_resource_share_max=float(reference.required_resource_share.max()),
        ERCOT=ef,PJM_summary=vs.to_dict("records"),
        probabilistic_validation=False,main_model_rerun=False)
    write_json(output/"verification.json",facts)
    print(json.dumps(facts,indent=2,allow_nan=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    main(parser.parse_args().output)
