"""Export a frozen, minimal input set; never modify the source ledger."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEDGER = ROOT / "seasonal_resource_ledger_20261004"
OUT = HERE / "inputs"
OUT.mkdir(exist_ok=True)
panel = pd.read_csv(LEDGER / "03_corrected_condition_panel.csv")
panel.to_csv(OUT / "native_panel.csv", index=False, float_format="%.17g")
pairs = pd.read_csv(LEDGER / "14_paired_DC_replacement_new_background.csv")
columns = ["planning_region", "year", "resource_condition", "response_condition", "data_center_scenario_id",
    "post2030_extension_id", "footprint_capture_case", "peak_mapping_case", "allocation_case", "linked_response_fraction",
    "reference_margin_level_pct", "source_path_removed_balance_GW", "replacement_balance_GW", "candidate_gross_path_GW",
    "candidate_linked_response_GW", "official_gross_path_GW", "official_linked_response_GW",
    "dc_path_pressure_contribution_GW", "dc_added_required_contribution_GW"]
pairs[columns].to_csv(OUT / "paired_paths.csv.gz", index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
rows = [
    ["ERCOT DC path", "NERC LTRA 2025, p. 131; 6.7/23/25 GW at 2027/2030/2035", "Source already applies a 49.8% observed peak/request adjustment. No second adjustment; not raw requested MW."],
    ["ERCOT provisional response", "NERC LTRA 2025, pp. 128-131; 58.37 GW in 2035", "Reported provisional large-load credit, not demonstrated DC-only delivery. Realization fraction is a scenario."],
    ["MISO response association", "Bound min(official DC increment, total credited response); LTRA p. 42", "Feasibility bound, not an observed DC-only increment. Full association may remove existing ordinary response in the counterfactual."],
    ["ERCOT response association", "Bound min(DC increment, provisional-response increase above 2025)", "The provisional increase is source-based; its DC-associated fraction is not identified. Separate from response realization."],
    ["Prospective resources", "NERC prospective margin reconstructed on matching source net demand", "Broader development condition, not all queued nameplate or a promised completion forecast."],
    ["PJM reference rule", "NERC LTRA 2025, pp. 90-94; changing RRS/ELCC basis", "Keep capacity and reference conventions matched. Native GW gap is not a nameplate construction requirement."],
    ["Candidate peak shapes", "CEC November 2025 profiles pp. 17-18; April 2026 methodology section 5", "Own-mean normalized complete templates; 5/7 and 2/7 day weights; fixed 2025 JJA diagnostic windows. Transferred scenarios, not seven-region measurements."],
    ["Spatial allocation", "Eight fixed-total allocation rules and reported-MW footprint evidence", "Conditional siting weights, not a regional large-load forecast. Missing MW is not automatically assigned reported-MW weight."],
    ["Post-2030 load", "LBNL source through 2030; author-defined taper alternatives thereafter", "Scenario continuation, not a published LBNL 2035 forecast or empirical confidence interval."]
]
pd.DataFrame(rows, columns=["input", "evidence", "interpretation"]).to_csv(OUT / "source_evidence.csv", index=False)
print(f"Exported {len(panel)} native rows and {len(pairs)} paired rows without changing sources")
