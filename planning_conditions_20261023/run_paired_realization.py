"""Connect retained ERCOT matched growth paths to effective provisional response."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from run_analysis import required_fraction
import run_pool_consistency as pool_model

HERE = Path(__file__).resolve().parent
TOL = 1e-7


def inputs():
    manifest = json.loads((HERE / "PAIRED_REALIZATION_INPUT_MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    protocol = json.loads((HERE / "PAIRED_REALIZATION_PROTOCOL.json").read_text())
    native = pd.read_csv(HERE / "inputs/native_inputs.csv")
    native = native[native.planning_region.eq("ERCOT")].sort_values("year")
    assert native.year.tolist() == list(range(2026, 2036))
    candidates = pool_model.display(pd.read_csv(HERE / "pool_inputs/candidate_paths.csv.gz"))
    candidates = candidates[candidates.planning_region.eq("ERCOT")]
    conditions = pd.read_csv(HERE / "pool_inputs/native_conditions.csv")
    pool = pd.read_csv(HERE / "pool_inputs/ercot_large_load_source.csv")
    pairs = []
    for rule in protocol["source_timing_rules"]:
        frame = pool_model.evaluate(candidates, conditions, pool, rule,
                                    positions=tuple(protocol["association_grid_positions"]))
        frame = frame[frame.resource_condition.eq("anticipated") & frame.response_condition.eq("published_all")]
        pairs.append(frame)
    return pd.concat(pairs, ignore_index=True), native, protocol


def vectors(frame, native):
    frame = frame.sort_values("year")
    assert frame.year.tolist() == native.year.tolist()
    k = 1 + native.reference_margin_level_pct.to_numpy() / 100
    np.testing.assert_allclose(k, 1 + frame.reference_margin_level_pct.to_numpy()/100, atol=TOL, rtol=0)
    h0, hs = frame.official_linked_response_GW.to_numpy(), frame.candidate_linked_response_GW.to_numpy()
    credit = native.provisional_large_load_response_MW.to_numpy()/1000
    c0, cs = credit-h0, credit-h0+hs
    uplift = (native.prospective_native_MW-native.anticipated_native_MW).to_numpy()/1000
    gross = native.gross_peak_demand_MW.to_numpy()/1000
    ordinary = native.ordinary_response_MW.to_numpy()/1000
    resource = native.anticipated_native_MW.to_numpy()/1000
    gross0 = gross-frame.official_gross_path_GW.to_numpy()
    gross_s = gross0+frame.candidate_GW.to_numpy()
    full0, full_s = frame.background_GW.to_numpy(), frame.balance_GW.to_numpy()
    np.testing.assert_allclose(full0, resource-k*(gross0-ordinary-c0), atol=TOL, rtol=0)
    np.testing.assert_allclose(full_s, resource-k*(gross_s-ordinary-cs), atol=TOL, rtol=0)
    large = frame.large_load_GW.to_numpy()
    assert np.min(c0) >= -TOL and np.min(cs) >= -TOL and np.min(uplift) >= -TOL
    assert np.min(large-frame.official_gross_path_GW.to_numpy()-c0) >= -TOL
    assert np.min(large-frame.official_gross_path_GW.to_numpy()+frame.candidate_GW.to_numpy()-cs) >= -TOL
    return dict(years=native.year.to_numpy(), k=k, uplift=uplift, resource=resource, ordinary=ordinary,
                gross0=gross0, gross_s=gross_s, full0=full0, full_s=full_s, c0=c0, cs=cs,
                full_official=resource-k*(gross-ordinary-credit), official_credit=credit)


def paired_balances(v, resource_share, realization):
    assert 0 <= resource_share <= 1 and 0 <= realization <= 1
    b0 = v["full0"]+resource_share*v["uplift"]-v["k"]*(1-realization)*v["c0"]
    bs = v["full_s"]+resource_share*v["uplift"]-v["k"]*(1-realization)*v["cs"]
    added = np.maximum(-bs, 0)-np.maximum(-b0, 0)
    pressure = b0-bs
    assert np.min(added) >= -TOL and np.min(pressure-added) >= -TOL
    return b0, bs, pressure, added


def response_limit(full_balance, credit, k, uplift, years, resource_share):
    need = k*credit-full_balance-resource_share*uplift
    fraction, index = required_fraction(need, k*credit)
    endpoint, _ = required_fraction(need[-1:], (k*credit)[-1:])
    return dict(all_year_fraction=fraction, endpoint_fraction=endpoint,
                binding_year=int(years[index]) if index >= 0 else 0,
                feasible_fraction_le_one=bool(fraction <= 1+TOL))


def calculate(pairs, native, protocol):
    annual, complete, thresholds = [], [], []
    keys = ["source_DC_timing_rule", "association_grid_position"]
    for (rule, position), frame in pairs.groupby(keys, sort=True):
        frame = frame.sort_values("year")
        v = vectors(frame, native)
        chi = float(frame.linked_response_fraction.iloc[0])
        np.testing.assert_allclose(frame.linked_response_fraction, chi, atol=TOL, rtol=0)
        setting = dict(source_timing_rule=rule, association_grid_position=float(position),
                       nominal_association_fraction=chi)
        for u in np.linspace(0, 1, protocol["resource_share_points"]):
            for world, full, credit in [("no_new_growth", v["full0"], v["c0"]),
                                        ("reference_growth", v["full_s"], v["cs"]),
                                        ("official_demand", v["full_official"], v["official_credit"])]:
                limit = response_limit(full, credit, v["k"], v["uplift"], v["years"], float(u))
                if limit["feasible_fraction_le_one"]:
                    residual = full+u*v["uplift"]-v["k"]*(1-limit["all_year_fraction"])*credit
                    assert residual.min() >= -TOL
                thresholds.append(dict(**setting, resource_share=float(u), world=world, **limit))
        for u in protocol["reported_resource_shares"]:
            for zeta in protocol["reported_realization_fractions"]:
                b0, bs, pressure, added = paired_balances(v, u, zeta)
                for i, year in enumerate(v["years"]):
                    annual.append(dict(**setting, resource_share=u, realization_fraction=zeta, year=int(year),
                        background_balance_GW=float(b0[i]), candidate_balance_GW=float(bs[i]),
                        growth_pressure_GW=float(pressure[i]), additional_requirement_GW=float(added[i]),
                        background_nominal_provisional_GW=float(v["c0"][i]),
                        candidate_nominal_provisional_GW=float(v["cs"][i]),
                        background_effective_provisional_GW=float(zeta*v["c0"][i]),
                        candidate_effective_provisional_GW=float(zeta*v["cs"][i]),
                        resource_uplift_GW=float(u*v["uplift"][i])))
                complete.append(dict(**setting, resource_share=u, realization_fraction=zeta,
                    background_below_years=int((b0 < -TOL).sum()), candidate_below_years=int((bs < -TOL).sum()),
                    minimum_background_balance_GW=float(b0.min()), minimum_candidate_balance_GW=float(bs.min()),
                    maximum_added_requirement_GW=float(added.max()),
                    added_requirement_sum_GW_year=float(added.sum()),
                    balance_2035_background_GW=float(b0[-1]), balance_2035_candidate_GW=float(bs[-1]),
                    additional_requirement_2035_GW=float(added[-1])))
    return pd.DataFrame(annual), pd.DataFrame(complete), pd.DataFrame(thresholds)


def main(output):
    pairs, native, protocol = inputs()
    annual, complete, thresholds = calculate(pairs, native, protocol)
    reported = thresholds[thresholds.resource_share.isin(protocol["reported_resource_shares"])].copy()
    display = annual[annual.source_timing_rule.eq("linear") & annual.association_grid_position.eq(0)
                     & annual.resource_share.eq(0)]
    limits = reported[reported.source_timing_rule.eq("linear") & reported.association_grid_position.eq(0)]
    exact = lambda world, u: float(limits[limits.world.eq(world) & limits.resource_share.eq(u)].all_year_fraction.item())
    outcome = display[display.realization_fraction.eq(.7)].set_index("year")
    facts = dict(passed=True, annual_points=len(annual), complete_settings=len(complete),
        threshold_rows=len(thresholds), input_years=10, source_timing_rules=4, association_settings=3,
        anticipated_no_new_growth_fraction=exact("no_new_growth", 0),
        anticipated_reference_fraction=exact("reference_growth", 0),
        quarter_uplift_no_new_growth_fraction=exact("no_new_growth", .25),
        quarter_uplift_reference_fraction=exact("reference_growth", .25),
        official_demand_fraction=exact("official_demand", 0),
        realization_70_added_2034_GW=float(outcome.loc[2034,"additional_requirement_GW"]),
        realization_70_added_2035_GW=float(outcome.loc[2035,"additional_requirement_GW"]),
        realization_70_background_below_years=int((outcome.background_balance_GW < -TOL).sum()),
        positive_threshold_binding_years=sorted(thresholds.loc[thresholds.all_year_fraction.gt(TOL),"binding_year"].unique().tolist()),
        nominal_association_fixed_when_realization_changes=True, published_ordinary_response_retained=True,
        official_model_refitted=False, main_figures_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("annual_balances.csv", annual), ("complete_paths.csv", complete),
                        ("response_thresholds.csv", thresholds), ("reported_thresholds.csv", reported)]:
        frame.to_csv(output/name, index=False, float_format="%.17g")
    (output/"verification.json").write_text(json.dumps(facts, indent=2, allow_nan=False)+"\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"paired_realization_results")
    main(parser.parse_args().output)
