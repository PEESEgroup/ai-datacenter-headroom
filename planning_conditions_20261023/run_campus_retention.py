"""Matrix/Frontier retention sensitivity using the current paired-response rules."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import run_pool_consistency as pool

HERE = Path(__file__).resolve().parent
CASE = ["keep_matrix", "keep_frontier"]


def inputs():
    manifest = json.loads((HERE / "CAMPUS_RETENTION_INPUT_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    weights = pd.read_csv(HERE / "pool_inputs/campus_retention_weights.csv")
    assert len(weights) == 8 and not weights.duplicated(CASE + ["planning_region"]).any()
    weights["weight_scale"] = (
        weights.retained_region_weight_MW / weights.retained_seven_region_weight_MW
        / (weights.reference_region_weight_MW / weights.reference_seven_region_weight_MW))
    candidates = pool.display(pd.read_csv(HERE / "pool_inputs/candidate_paths.csv.gz"))
    native = pd.read_csv(HERE / "pool_inputs/native_conditions.csv")
    native = native[native.resource_condition.isin(["anticipated", "prospective"])]
    source = pd.read_csv(HERE / "pool_inputs/ercot_large_load_source.csv")
    return weights, candidates, native, source


def calculate(weights, candidates, native, source):
    annual = []
    for flags, rows in weights.groupby(CASE, sort=True):
        candidate = candidates.merge(rows[["planning_region", "weight_scale"]],
                                     on="planning_region", validate="many_to_one")
        candidate["candidate_GW"] *= candidate.weight_scale
        result = pool.evaluate(candidate, native, source)
        for name, value in zip(CASE, flags):
            result[name] = value
        annual.append(result)
    annual = pd.concat(annual, ignore_index=True)
    annual = annual.sort_values(CASE + pool.KEYS + ["year"]).reset_index(drop=True)
    complete = pool.summary(annual, extra=tuple(CASE))
    primary = annual[annual.year.eq(2035) & annual.association_grid_position.eq(0)].copy()
    primary = primary.rename(columns={"planning_region": "region", "resource_condition": "resources",
        "response_condition": "response", "candidate_GW": "candidate_peak_increment_GW",
        "balance_GW": "balance_GW", "added_requirement_GW": "added_GW"})
    primary = primary[CASE + ["region", "resources", "response", "pressure_GW", "balance_GW", "added_GW",
        "candidate_peak_increment_GW", "background_GW", "linked_response_fraction", "weight_scale",
        "background_pool_slack_GW", "candidate_pool_slack_GW"]]
    assert (len(annual), len(complete), len(primary)) == (960, 96, 32)
    return annual, complete, primary


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    weights, candidates, native, source = inputs()
    annual, complete, primary = calculate(weights, candidates, native, source)
    pool.save(annual, out / "campus_retention_annual.csv")
    pool.save(complete, out / "campus_retention_complete_paths.csv")
    pool.save(primary, out / "remaining_campus_retention_cases.csv")
    current = pool.display(pd.read_csv(HERE / "pool_results/displayed_pairs.csv"))
    current = current[current.resource_condition.isin(["anticipated", "prospective"])].sort_values(pool.KEYS + ["year"])
    retained = annual[annual.keep_matrix.eq(1) & annual.keep_frontier.eq(1)].sort_values(pool.KEYS + ["year"])
    for field in ["candidate_GW", "background_GW", "pressure_GW", "balance_GW", "added_requirement_GW"]:
        np.testing.assert_allclose(retained[field], current[field], atol=pool.TOL, rtol=0)
    ercot = primary[primary.region.eq("ERCOT") & primary.resources.eq("anticipated")
                    & primary.response.eq("published_all")]
    report = dict(passed=True, annual_rows=len(annual), complete_paths=len(complete), primary_endpoint_rows=len(primary),
        both_retained_reproduce_current_pairs=True,
        ERCOT_anticipated_published_balance_min_GW=float(ercot.balance_GW.min()),
        ERCOT_anticipated_published_balance_max_GW=float(ercot.balance_GW.max()),
        minimum_background_pool_slack_GW=float(annual.background_pool_slack_GW.min()),
        minimum_candidate_pool_slack_GW=float(annual.candidate_pool_slack_GW.min()),
        official_regional_model_changed=False, parameters_refitted=False)
    (out / "campus_retention_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "pool_results")
    main(parser.parse_args().output)
