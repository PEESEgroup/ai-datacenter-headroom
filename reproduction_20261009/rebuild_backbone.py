"""Rebuild native annual inputs from components, never from saved balances."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "numerical"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    annual = module("public_annual", ROOT / "scripts/build_annual_scenario_cube.py")
    data = pd.read_csv(ROOT / "inputs/source_planning_components.csv")
    assert not any("balance" in col or "required" in col for col in data)
    matrix = data.rename(columns={"resource_condition": "resource_path",
        "response_condition": "dr_treatment", "base_gross_peak_demand_MW": "gross_peak_demand_MW",
        "credited_response_MW": "credited_DR_MW"})
    matrix["net_peak_demand_MW"] = matrix.gross_peak_demand_MW - matrix.credited_DR_MW
    matrix = annual.recompute_balance(matrix)
    config = annual.load_config()
    panel = annual.build_resource_response_panel(matrix, config)
    source_only = json.loads(json.dumps(config))
    source_only["dc_conditions"]["modeled_scenarios"] = []
    empty = pd.DataFrame(columns=["planning_region", "year", "scenario", "modeled_incremental_DC_GW"])
    cube = annual.build_cube(panel, empty, source_only)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    cube.to_csv(results / "81_annual_scenario_cube.csv", index=False)
    identification = pd.read_csv(ROOT / "inputs/embedded_DC_paths.csv")
    ercot = pd.read_csv(ROOT / "inputs/ercot_reported_response.csv").rename(
        columns={"provisional_large_load_response_MW": "source_large_load_curtailment_MW"})
    annual.build_identified_counterfactuals(panel, identification, ercot).to_csv(
        results / "84_source_identified_dc_counterfactuals.csv", index=False)

    rows = []
    for _, group in cube.groupby(["planning_region", "resource_condition", "response_condition"]):
        group = group.sort_values("year")
        first = group.iloc[0]
        factor = 1 + first.reference_margin_level_pct / 100
        for r in group.itertuples():
            terms = {"resource_stock_term_GW": (r.capacity_equivalent_resources_MW - first.capacity_equivalent_resources_MW) / 1000,
                "source_gross_demand_term_GW": -factor * (r.base_gross_peak_demand_MW - first.base_gross_peak_demand_MW) / 1000,
                "additional_load_stress_term_GW": 0.0,
                "credited_response_term_GW": factor * (r.credited_response_MW - first.credited_response_MW) / 1000,
                "reference_margin_change_term_GW": -r.base_net_peak_demand_MW * (r.reference_margin_level_pct - first.reference_margin_level_pct) / 100000}
            change = r.base_balance_GW - first.base_balance_GW
            np.testing.assert_allclose(sum(terms.values()), change, atol=1e-10)
            rows.append({"planning_region": r.planning_region, "year": r.year, "base_year": 2026,
                "resource_condition": r.resource_condition, "response_condition": r.response_condition,
                "dc_condition_id": "source_embedded", "overlap_case": "not_applicable",
                "balance_change_from_2026_GW": change, **terms,
                "reconstructed_balance_change_GW": sum(terms.values()),
                "identity_residual_GW": sum(terms.values()) - change,
                "decomposition_type": "exact_accounting_not_causal",
                "causal_interpretation_allowed": False, "source_gross_demand_contains_embedded_DC": True,
                "additional_load_term_is_identified_DC_attribution": False})
    pd.DataFrame(rows).to_csv(results / "92_annual_balance_driver_decomposition.csv", index=False)
    print(json.dumps({"backbone_rows": len(cube), "balance_outputs_read_as_inputs": False}))


if __name__ == "__main__":
    main()
