#!/usr/bin/env python3
"""Build the additive LBNL 2025 data-center scenario update (v1.2).

The source-period national energy paths are anchored to the LBNL 2025 Update.
Post-2030 values, energy-to-peak mapping, overlap, and spatial allocation remain
explicit scenario or sensitivity dimensions. Existing results 80-99 are inputs
for lineage only and are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config/lbnl2025_data_center_scenarios_v1_2.json"
ANCHOR_PATH = (
    ROOT
    / "sources/lbnl_2025_update/lbnl_2025_update_figure1_anchors.csv"
)
LBNL_PDF_PATH = (
    ROOT
    / "sources/lbnl_2025_update/lbnl_us_data_center_energy_usage_2025_update.pdf"
)
PEAK_MAPPING_PATH = (
    ROOT
    / "sources/lbnl_2025_update/peak_mapping_factors_fixed_energy_20261004.csv"
)
EPRI_HTML_PATH = ROOT / "sources/epri_2026/load-growth.html"
EPRI_PAGE_PATH = ROOT / "sources/epri_2026/load-growth-page.js"
BASE_CUBE_PATH = ROOT / "results/81_annual_scenario_cube.csv"
SPATIAL_WEIGHTS_PATH = ROOT / "sources/lbnl_2025_update/spatial_fixed_total_allocation_sensitivity.csv"
SPATIAL_COVERAGE_PATH = ROOT / "sources/lbnl_2025_update/spatial_mapping_coverage.csv"

ANCHOR_OUTPUT = ROOT / "results/100_lbnl2025_anchor_audit.csv"
NATIONAL_OUTPUT = ROOT / "results/101_lbnl2025_national_energy_paths.csv"
MAPPING_OUTPUT = ROOT / "results/102_lbnl2025_peak_mapping_crosswalk.csv"
REGIONAL_OUTPUT = ROOT / "results/103_lbnl2025_regional_load_paths.csv"
CUBE_OUTPUT = ROOT / "results/104_annual_scenario_cube_v1_2.csv"
SUMMARY_OUTPUT = ROOT / "results/105_annual_scenario_trajectory_summary_v1_2.csv"
NAMED_OUTPUT = ROOT / "results/106_named_scenario_years_v1_2.csv"
UNCERTAINTY_OUTPUT = ROOT / "results/107_uncertainty_component_ranges_v1_2.csv"
SPATIAL_OUTPUT = ROOT / "results/108_spatial_allocation_sensitivity_v1_2.csv"
MANIFEST_OUTPUT = ROOT / "results/109_lbnl2025_source_manifest.csv"
QA_OUTPUT = ROOT / "results/110_lbnl2025_scenario_update_qa.csv"
MANUSCRIPT_DISPLAY_OUTPUT = (
    ROOT / "results/111_manuscript_display_scenario_envelope_v1_2.csv"
)
NOTE_OUTPUT = ROOT / "docs/20_lbnl2025_scenario_results.md"

TOL = 1e-8

REGION_MAP = {
    "PJM": "PJM",
    "MISO": "MISO",
    "SPP": "SPP",
    "ERCOT": "ERCOT",
    "NYISO": "NYISO",
    "ISO-NE": "ISO-NE",
    "WECC-California": "WECC-California",
}

ALLOCATION_LABELS = {
    "central_reported_low": "reported_MW_low_imputation",
    "central_reported_mid": "reported_MW_reference_imputation",
    "central_reported_high": "reported_MW_high_imputation",
    "central_reported_plus_size_proxy_sensitivity": "reported_plus_size_proxy",
    "central_site_count": "core_site_count",
    "broad_reported_mid": "broad_reported_MW",
    "broad_site_count": "broad_site_count",
    "high_evidence_reported_mid": "high_evidence_reported_MW",
}


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[wrote] {path} ({len(frame):,} rows)")


def scenario_metadata(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in config["national_energy_scenarios"]  # type: ignore[index]
    }


def extension_metadata(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in config["post_2030_extensions"]  # type: ignore[index]
    }


def mapping_metadata(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in config["peak_mapping_cases"]  # type: ignore[index]
    }


def footprint_metadata(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in config["analyzed_footprint_capture_cases"]  # type: ignore[index]
    }


def build_anchor_audit(anchors: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    out = anchors.copy()
    scenarios = scenario_metadata(config)
    out["scenario_label"] = out["scenario_id"].map(
        {key: value["display_label"] for key, value in scenarios.items()}
    )
    out["scenario_role"] = out["scenario_id"].map(
        {key: value["role"] for key, value in scenarios.items()}
    )
    out["source_document"] = "United States Data Center Energy Usage Report: 2025 Update"
    out["source_url"] = "https://doi.org/10.71468/P1RP4F"
    out["source_period"] = True
    out["value_is_exact_reported_number"] = out["value_basis"].isin(
        ["reported historical value", "reported reference value", "reported compounded lower value", "reported compounded upper value"]
    )
    out["value_is_figure_digitized"] = out["value_basis"].str.contains(
        "figure digitization", case=False
    )
    out["value_is_reported_growth_derived"] = out["value_basis"].str.contains(
        "reported .* percent annual increase", case=False, regex=True
    )
    return out.sort_values(["scenario_id", "year"]).reset_index(drop=True)


def build_national_paths(
    anchors: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    scenarios = scenario_metadata(config)
    extensions = extension_metadata(config)
    last_year = int(config["analysis_horizon"]["planning_last_year"])  # type: ignore[index]
    rows: list[dict[str, object]] = []

    for scenario_id, scenario_spec in scenarios.items():
        source = anchors[anchors["scenario_id"].eq(scenario_id)].set_index("year")
        value_2025 = float(source.loc[2025, "electricity_TWh"])
        value_2030 = float(source.loc[2030, "electricity_TWh"])
        source_cagr = (value_2030 / value_2025) ** (1.0 / 5.0) - 1.0

        for extension_id, extension_spec in extensions.items():
            multiplier = float(extension_spec["growth_rate_multiplier"])
            post_rate = source_cagr * multiplier
            for year in range(2024, last_year + 1):
                if year <= 2030:
                    record = source.loc[year]
                    electricity = float(record["electricity_TWh"])
                    basis = str(record["value_basis"])
                    location = str(record["source_location"])
                    digitization_uncertainty = float(
                        record["digitization_uncertainty_TWh"]
                    )
                    source_period = True
                else:
                    electricity = value_2030 * (1.0 + post_rate) ** (year - 2030)
                    basis = "explicit post-2030 continuation scenario"
                    location = (
                        "EPRI 2026 continuation rule"
                        if extension_id == "epri_quarter_rate_taper"
                        else "author-defined structural sensitivity"
                    )
                    digitization_uncertainty = 0.0
                    source_period = False

                rows.append(
                    {
                        "year": year,
                        "data_center_scenario_id": scenario_id,
                        "data_center_scenario_label": scenario_spec["display_label"],
                        "data_center_scenario_role": scenario_spec["role"],
                        "post2030_extension_id": extension_id,
                        "post2030_extension_role": extension_spec["role"],
                        "electricity_TWh": electricity,
                        "average_grid_power_GW": electricity / 8.76,
                        "source_2025_2030_CAGR_pct": source_cagr * 100.0,
                        "post2030_annual_growth_rate_pct": post_rate * 100.0,
                        "value_basis": basis,
                        "source_location_or_extension_basis": location,
                        "digitization_uncertainty_TWh": digitization_uncertainty,
                        "is_LBNL_source_period": source_period,
                        "LBNL_authorship_after_2030": False if year > 2030 else True,
                    }
                )

    out = pd.DataFrame(rows)
    baseline = out[out["year"].eq(2025)][
        [
            "data_center_scenario_id",
            "post2030_extension_id",
            "average_grid_power_GW",
        ]
    ].rename(columns={"average_grid_power_GW": "average_grid_power_2025_GW"})
    out = out.merge(
        baseline,
        on=["data_center_scenario_id", "post2030_extension_id"],
        how="left",
        validate="many_to_one",
    )
    out["incremental_average_power_vs_2025_GW"] = (
        out["average_grid_power_GW"] - out["average_grid_power_2025_GW"]
    ).clip(lower=0.0)
    return out.sort_values(
        ["data_center_scenario_id", "post2030_extension_id", "year"]
    ).reset_index(drop=True)


def build_mapping_crosswalk(
    mapping: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    cases = mapping_metadata(config)
    reverse = {str(spec["source_shape"]): case for case, spec in cases.items()}
    out = mapping.copy()
    out["peak_mapping_case"] = out["source_shape"].map(reverse)
    out["peak_mapping_role"] = out["peak_mapping_case"].map(
        {case: spec["role"] for case, spec in cases.items()}
    )
    out["gamma_recomputed"] = (
        out["peak_window_coincidence"] / out["shape_mean_load_factor"]
    )
    out["gamma_identity_residual"] = out["gamma"] - out["gamma_recomputed"]
    out["mapping_interpretation"] = (
        "region-specific diagnostic peak-window factor divided by the selected complete-template mean; "
        "5/7 weekday and 2/7 weekend weights; fixed annual energy, not measured seven-region coincidence"
    )
    return out.sort_values(["model_region", "peak_mapping_case"]).reset_index(
        drop=True
    )


def prepare_weights(weights: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    allowed = set(config["spatial_allocation"]["sensitivity_cases"])  # type: ignore[index]
    out = weights[weights["allocation_case"].isin(allowed)].copy()
    out = out.rename(columns={"allocation_case": "source_allocation_case"})
    out["allocation_case"] = out["source_allocation_case"].map(ALLOCATION_LABELS)
    out["planning_region"] = out["model_region"].map(REGION_MAP)
    out["footprint_match"] = np.where(
        out["model_region"].eq("WECC-California"),
        "NERC_listed_component_geographic_proxy_not_connection_certification",
        "direct_or_near_direct",
    )
    return out


def build_regional_paths(
    national: pd.DataFrame,
    mapping: pd.DataFrame,
    weights: pd.DataFrame,
    allocation_cases: list[str],
    config: dict[str, object],
) -> pd.DataFrame:
    selected = weights[weights["source_allocation_case"].isin(allocation_cases)].copy()
    footprint = pd.DataFrame(config["analyzed_footprint_capture_cases"]).rename(  # type: ignore[arg-type]
        columns={"id": "footprint_capture_case", "role": "footprint_capture_role"}
    )
    path = national[national["year"].between(2025, 2035)].merge(
        selected[
            [
                "source_allocation_case",
                "allocation_case",
                "model_region",
                "planning_region",
                "footprint_match",
                "allocation_share",
                "weight_interpretation",
            ]
        ],
        how="cross",
    )
    path = path.merge(footprint, how="cross")
    path = path.merge(
        mapping[
            [
                "model_region",
                "peak_mapping_case",
                "shape_mean_load_factor",
                "peak_window_coincidence",
                "gamma",
                "peak_mapping_role",
            ]
        ],
        on="model_region",
        how="left",
        validate="many_to_many",
    )
    path["average_power_assigned_to_analyzed_footprint_GW"] = (
        path["average_grid_power_GW"] * path["capture_share"]
    )
    path["allocated_average_power_GW"] = (
        path["average_power_assigned_to_analyzed_footprint_GW"]
        * path["allocation_share"]
    )
    path["regional_peak_equivalent_load_GW"] = (
        path["allocated_average_power_GW"] * path["gamma"]
    )
    keys = [
        "data_center_scenario_id",
        "post2030_extension_id",
        "source_allocation_case",
        "footprint_capture_case",
        "model_region",
        "peak_mapping_case",
    ]
    baseline = path[path["year"].eq(2025)][
        keys + ["regional_peak_equivalent_load_GW"]
    ].rename(
        columns={
            "regional_peak_equivalent_load_GW": "regional_peak_equivalent_2025_GW"
        }
    )
    path = path.merge(baseline, on=keys, how="left", validate="many_to_one")
    path["incremental_data_center_load_vs_2025_GW"] = (
        path["regional_peak_equivalent_load_GW"]
        - path["regional_peak_equivalent_2025_GW"]
    ).clip(lower=0.0)
    path["load_category"] = "data-center load scenario"
    path["spatial_interpretation"] = (
        "national path set first; explicit analyzed-footprint capture applied before conditional project-evidence allocation"
    )
    return path.sort_values(
        [
            "source_allocation_case",
            "footprint_capture_case",
            "data_center_scenario_id",
            "post2030_extension_id",
            "peak_mapping_case",
            "model_region",
            "year",
        ]
    ).reset_index(drop=True)


def build_cube(
    base_cube: pd.DataFrame,
    regional: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    base = base_cube[base_cube["dc_condition_id"].eq("source_embedded")].copy()
    base_rows = pd.DataFrame(
        {
            "planning_region": base["planning_region"],
            "nerc_assessment_area": base["nerc_assessment_area"],
            "year": base["year"].astype(int),
            "resource_condition": base["resource_condition"],
            "resource_condition_type": base["resource_condition_type"],
            "response_condition": base["response_condition"],
            "data_center_scenario_id": "source_embedded",
            "data_center_scenario_label": "source-embedded regional demand path",
            "post2030_extension_id": "not_applicable",
            "peak_mapping_case": "not_applicable",
            "footprint_capture_case": "not_applicable",
            "footprint_capture_share": np.nan,
            "overlap_case": "not_applicable",
            "assumed_embedded_share": 1.0,
            "additional_share": 0.0,
            "national_electricity_TWh": np.nan,
            "national_average_grid_power_GW": np.nan,
            "allocation_case": "not_applicable",
            "allocation_share": np.nan,
            "gamma": np.nan,
            "incremental_data_center_load_GW": 0.0,
            "additional_data_center_load_GW": 0.0,
            "base_gross_peak_demand_MW": base["base_gross_peak_demand_MW"],
            "credited_response_MW": base["credited_response_MW"],
            "base_net_peak_demand_MW": base["base_net_peak_demand_MW"],
            "capacity_equivalent_resources_MW": base[
                "capacity_equivalent_resources_MW"
            ],
            "reference_margin_level_pct": base["reference_margin_level_pct"],
            "base_balance_GW": base["base_balance_GW"],
            "scenario_gross_peak_demand_MW": base["base_gross_peak_demand_MW"],
            "scenario_net_peak_demand_MW": base["base_net_peak_demand_MW"],
            "scenario_reference_requirement_MW": base[
                "base_reference_requirement_MW"
            ],
            "scenario_balance_GW": base["base_balance_GW"],
            "scenario_required_contribution_GW": base[
                "base_required_contribution_GW"
            ],
            "planning_pressure_status": np.where(
                base["base_balance_GW"] >= -TOL,
                "at_or_above_reference",
                "below_reference",
            ),
            "scenario_applicable": True,
            "not_applicable_reason": "",
            "generic_scenario_not_identified_attribution": True,
            "source_identified_path_preferred": base["planning_region"].isin(
                ["MISO", "ERCOT"]
            ),
            "causal_attribution_allowed": False,
            "reliability_event_interpretation_allowed": False,
        }
    )

    cube_case = str(config["spatial_allocation"]["cube_case"])  # type: ignore[index]
    modeled = regional[
        regional["source_allocation_case"].eq(cube_case)
        & regional["planning_region"].notna()
    ].copy()
    modeled = modeled.rename(
        columns={
            "electricity_TWh": "national_electricity_TWh",
            "average_grid_power_GW": "national_average_grid_power_GW",
            "incremental_data_center_load_vs_2025_GW": "incremental_data_center_load_GW",
        }
    )
    merge_base = base.drop(
        columns=[
            "dc_condition_id",
            "dc_model_scenario",
            "overlap_case",
            "assumed_embedded_share",
            "additional_share",
            "modeled_incremental_DC_GW",
            "additional_DC_stress_GW",
        ],
        errors="ignore",
    )
    merged = merge_base.merge(
        modeled,
        on=["planning_region", "year"],
        how="inner",
        validate="many_to_many",
        suffixes=("_base", ""),
    )
    overlap = pd.DataFrame(config["overlap_cases"]).rename(  # type: ignore[arg-type]
        columns={"id": "overlap_case"}
    )
    merged = merged.merge(overlap, how="cross")
    merged["additional_data_center_load_GW"] = (
        merged["incremental_data_center_load_GW"] * merged["additional_share"]
    )
    merged["scenario_gross_peak_demand_MW"] = (
        merged["base_gross_peak_demand_MW"]
        + merged["additional_data_center_load_GW"] * 1000.0
    )
    merged["scenario_net_peak_demand_MW"] = (
        merged["scenario_gross_peak_demand_MW"] - merged["credited_response_MW"]
    )
    mu = merged["reference_margin_level_pct"] / 100.0
    merged["scenario_reference_requirement_MW"] = (
        merged["scenario_net_peak_demand_MW"] * (1.0 + mu)
    )
    merged["scenario_balance_GW"] = (
        merged["capacity_equivalent_resources_MW"]
        - merged["scenario_reference_requirement_MW"]
    ) / 1000.0
    merged["scenario_required_contribution_GW"] = (
        -merged["scenario_balance_GW"]
    ).clip(lower=0.0)
    merged["planning_pressure_status"] = np.where(
        merged["scenario_balance_GW"] >= -TOL,
        "at_or_above_reference",
        "below_reference",
    )
    merged["scenario_applicable"] = True
    merged["not_applicable_reason"] = ""
    merged["generic_scenario_not_identified_attribution"] = True
    merged["source_identified_path_preferred"] = merged["planning_region"].isin(
        ["MISO", "ERCOT"]
    )
    merged["causal_attribution_allowed"] = False
    merged["reliability_event_interpretation_allowed"] = False

    modeled_rows = merged[
        [
            "planning_region",
            "nerc_assessment_area",
            "year",
            "resource_condition",
            "resource_condition_type",
            "response_condition",
            "data_center_scenario_id",
            "data_center_scenario_label",
            "post2030_extension_id",
            "peak_mapping_case",
            "footprint_capture_case",
            "capture_share",
            "overlap_case",
            "assumed_embedded_share",
            "additional_share",
            "national_electricity_TWh",
            "national_average_grid_power_GW",
            "allocation_case",
            "allocation_share",
            "gamma",
            "incremental_data_center_load_GW",
            "additional_data_center_load_GW",
            "base_gross_peak_demand_MW",
            "credited_response_MW",
            "base_net_peak_demand_MW",
            "capacity_equivalent_resources_MW",
            "reference_margin_level_pct",
            "base_balance_GW",
            "scenario_gross_peak_demand_MW",
            "scenario_net_peak_demand_MW",
            "scenario_reference_requirement_MW",
            "scenario_balance_GW",
            "scenario_required_contribution_GW",
            "planning_pressure_status",
            "scenario_applicable",
            "not_applicable_reason",
            "generic_scenario_not_identified_attribution",
            "source_identified_path_preferred",
            "causal_attribution_allowed",
            "reliability_event_interpretation_allowed",
        ]
    ].copy()
    modeled_rows = modeled_rows.rename(
        columns={"capture_share": "footprint_capture_share"}
    )

    out = pd.concat([base_rows, modeled_rows], ignore_index=True)
    out["scenario_semantics"] = np.where(
        out["data_center_scenario_id"].eq("source_embedded"),
        "official regional demand path with embedded data-center load unresolved",
        "national data-center load scenario applied through an explicit overlap case",
    )
    out["result_status"] = "computed_requires_independent_review"
    return out.sort_values(
        [
            "planning_region",
            "resource_condition",
            "response_condition",
            "data_center_scenario_id",
            "post2030_extension_id",
            "peak_mapping_case",
            "footprint_capture_case",
            "overlap_case",
            "year",
        ]
    ).reset_index(drop=True)


def build_trajectory_summary(cube: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "planning_region",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "post2030_extension_id",
        "peak_mapping_case",
        "footprint_capture_case",
        "overlap_case",
    ]
    rows: list[dict[str, object]] = []
    for values, group in cube.groupby(keys, dropna=False, sort=True):
        ordered = group.sort_values("year")
        below = ordered[ordered["scenario_balance_GW"] < -TOL]
        minimum = ordered.loc[ordered["scenario_balance_GW"].idxmin()]
        y2030 = ordered[ordered["year"].eq(2030)]
        y2035 = ordered[ordered["year"].eq(2035)]
        rows.append(
            {
                **dict(zip(keys, values)),
                "first_below_reference_year": (
                    int(below["year"].min()) if not below.empty else np.nan
                ),
                "years_below_reference": int(len(below)),
                "minimum_balance_GW": float(minimum["scenario_balance_GW"]),
                "minimum_balance_year": int(minimum["year"]),
                "balance_2030_GW": (
                    float(y2030["scenario_balance_GW"].iloc[0])
                    if len(y2030) == 1
                    else np.nan
                ),
                "balance_2035_GW": (
                    float(y2035["scenario_balance_GW"].iloc[0])
                    if len(y2035) == 1
                    else np.nan
                ),
                "required_contribution_2035_GW": (
                    float(y2035["scenario_required_contribution_GW"].iloc[0])
                    if len(y2035) == 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_named_years(cube: pd.DataFrame) -> pd.DataFrame:
    return cube[
        cube["year"].isin([2030, 2035])
        & cube["data_center_scenario_id"].ne("source_embedded")
        & cube["post2030_extension_id"].eq("epri_quarter_rate_taper")
        & cube["peak_mapping_case"].eq("reference_coincidence")
        & cube["resource_condition"].isin(["anticipated", "prospective"])
        & cube["response_condition"].isin(["current_only", "published_all"])
    ].copy().sort_values(
        [
            "planning_region",
            "year",
            "resource_condition",
            "response_condition",
            "data_center_scenario_id",
            "footprint_capture_case",
            "overlap_case",
        ]
    )


def build_manuscript_display(
    cube: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    spec = config["manuscript_display"]  # type: ignore[index]
    out = cube[
        cube["year"].between(
            int(config["analysis_horizon"]["planning_first_year"]),  # type: ignore[index]
            int(config["analysis_horizon"]["planning_last_year"]),  # type: ignore[index]
        )
        & cube["data_center_scenario_id"].eq(spec["national_energy_scenario"])
        & cube["post2030_extension_id"].eq(spec["post_2030_extension"])
        & cube["footprint_capture_case"].eq(spec["footprint_capture_case"])
        & cube["peak_mapping_case"].eq(spec["peak_mapping_case"])
        & cube["overlap_case"].isin(spec["overlap_cases"])
    ].copy()
    pairs = pd.DataFrame(spec["resource_response_pairs"]).rename(
        columns={"id": "scenario_pair_id", "role": "scenario_pair_role"}
    )
    out = out.merge(
        pairs,
        on=["resource_condition", "response_condition"],
        how="inner",
        validate="many_to_one",
    )
    out["display_family_id"] = "lbnl_reference_reported_mw_footprint"
    out["display_role"] = "illustrative named-scenario family"
    out["central_forecast_interpretation_allowed"] = False
    out["load_source_status"] = np.where(
        out["year"].le(2030),
        "LBNL source-period scenario",
        "conditional EPRI taper-rule continuation scenario",
    )
    return out.sort_values(
        [
            "planning_region",
            "scenario_pair_id",
            "overlap_case",
            "year",
        ]
    ).reset_index(drop=True)


def component_range(
    selected: pd.DataFrame, component: str, values: list[str]
) -> dict[str, object]:
    subset = selected[selected[component].isin(values)]
    balances = subset.groupby(component)["scenario_balance_GW"].first()
    return {
        "component": component,
        "cases_compared": " | ".join(values),
        "minimum_balance_GW": float(balances.min()),
        "maximum_balance_GW": float(balances.max()),
        "balance_range_GW": float(balances.max() - balances.min()),
    }


def build_uncertainty_ranges(
    cube: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    modeled = cube[
        cube["year"].eq(2035)
        & cube["resource_condition"].eq("anticipated")
        & cube["response_condition"].eq("published_all")
        & cube["data_center_scenario_id"].ne("source_embedded")
    ].copy()
    rows: list[dict[str, object]] = []
    reported_footprint_case = str(config["reported_footprint_case"])
    for region in sorted(modeled["planning_region"].unique()):
        region_data = modeled[modeled["planning_region"].eq(region)]

        national = region_data[
            region_data["post2030_extension_id"].eq("epri_quarter_rate_taper")
            & region_data["peak_mapping_case"].eq("reference_coincidence")
            & region_data["footprint_capture_case"].eq(reported_footprint_case)
            & region_data["overlap_case"].eq("half_embedded_diagnostic")
        ]
        extensions = region_data[
            region_data["data_center_scenario_id"].eq("lbnl_reference")
            & region_data["peak_mapping_case"].eq("reference_coincidence")
            & region_data["footprint_capture_case"].eq(reported_footprint_case)
            & region_data["overlap_case"].eq("half_embedded_diagnostic")
        ]
        mappings = region_data[
            region_data["data_center_scenario_id"].eq("lbnl_reference")
            & region_data["post2030_extension_id"].eq("epri_quarter_rate_taper")
            & region_data["footprint_capture_case"].eq(reported_footprint_case)
            & region_data["overlap_case"].eq("half_embedded_diagnostic")
        ]
        overlaps = region_data[
            region_data["data_center_scenario_id"].eq("lbnl_reference")
            & region_data["post2030_extension_id"].eq("epri_quarter_rate_taper")
            & region_data["peak_mapping_case"].eq("reference_coincidence")
            & region_data["footprint_capture_case"].eq(reported_footprint_case)
        ]
        footprints = region_data[
            region_data["data_center_scenario_id"].eq("lbnl_reference")
            & region_data["post2030_extension_id"].eq("epri_quarter_rate_taper")
            & region_data["peak_mapping_case"].eq("reference_coincidence")
            & region_data["overlap_case"].eq("half_embedded_diagnostic")
        ]

        comparisons = [
            component_range(
                national,
                "data_center_scenario_id",
                [
                    "lbnl_compounded_lower",
                    "lbnl_reference",
                    "lbnl_compounded_upper",
                ],
            ),
            component_range(
                extensions,
                "post2030_extension_id",
                [
                    "hold_2030_level",
                    "epri_quarter_rate_taper",
                    "slower_half_rate_taper_stress",
                ],
            ),
            component_range(
                mappings,
                "peak_mapping_case",
                ["low_coincidence", "reference_coincidence", "high_coincidence"],
            ),
            component_range(
                overlaps,
                "overlap_case",
                ["fully_embedded", "half_embedded_diagnostic", "fully_additional"],
            ),
            component_range(
                footprints,
                "footprint_capture_case",
                [
                    "core_active_reported_mw_share",
                    "broad_active_reported_mw_share",
                    "core_active_site_count_share",
                    "full_national_assignment_upper_stress",
                ],
            ),
        ]
        for comparison in comparisons:
            rows.append({"planning_region": region, "year": 2035, **comparison})

    out = pd.DataFrame(rows)
    out["dominant_component_within_reported_set"] = out.groupby(
        "planning_region"
    )["balance_range_GW"].transform(lambda x: x.eq(x.max()))
    out["interpretation"] = (
        "one-at-a-time accounting range; not a confidence interval and not additive across components"
    )
    return out.sort_values(["planning_region", "component"]).reset_index(drop=True)


def build_spatial_sensitivity(
    base_cube: pd.DataFrame,
    regional_all_cases: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    base = base_cube[
        base_cube["dc_condition_id"].eq("source_embedded")
        & base_cube["resource_condition"].eq("anticipated")
        & base_cube["response_condition"].eq("published_all")
    ].copy().drop(
        columns=[
            "overlap_case",
            "assumed_embedded_share",
            "additional_share",
            "modeled_incremental_DC_GW",
            "additional_DC_stress_GW",
        ],
        errors="ignore",
    )
    paths = regional_all_cases[
        regional_all_cases["data_center_scenario_id"].eq("lbnl_reference")
        & regional_all_cases["post2030_extension_id"].eq(
            "epri_quarter_rate_taper"
        )
        & regional_all_cases["peak_mapping_case"].eq("reference_coincidence")
        & regional_all_cases["footprint_capture_case"].eq(
            str(config["reported_footprint_case"])
        )
        & regional_all_cases["planning_region"].notna()
        & regional_all_cases["year"].between(2026, 2035)
    ].copy()
    merged = base.merge(
        paths,
        on=["planning_region", "year"],
        how="inner",
        validate="many_to_many",
    ).merge(
        pd.DataFrame(config["overlap_cases"]).rename(  # type: ignore[arg-type]
            columns={"id": "overlap_case"}
        ),
        how="cross",
    )
    merged["additional_data_center_load_GW"] = (
        merged["incremental_data_center_load_vs_2025_GW"]
        * merged["additional_share"]
    )
    gross = (
        merged["base_gross_peak_demand_MW"]
        + merged["additional_data_center_load_GW"] * 1000.0
    )
    net = gross - merged["credited_response_MW"]
    requirement = net * (1.0 + merged["reference_margin_level_pct"] / 100.0)
    merged["scenario_balance_GW"] = (
        merged["capacity_equivalent_resources_MW"] - requirement
    ) / 1000.0
    merged["planning_pressure_status"] = np.where(
        merged["scenario_balance_GW"] >= -TOL,
        "at_or_above_reference",
        "below_reference",
    )
    return merged[
        [
            "planning_region",
            "year",
            "source_allocation_case",
            "allocation_case",
            "allocation_share",
            "footprint_capture_case",
            "capture_share",
            "overlap_case",
            "incremental_data_center_load_vs_2025_GW",
            "additional_data_center_load_GW",
            "base_balance_GW",
            "scenario_balance_GW",
            "planning_pressure_status",
            "weight_interpretation",
            "spatial_interpretation",
        ]
    ].sort_values(
        ["planning_region", "allocation_case", "overlap_case", "year"]
    ).reset_index(drop=True)


def build_manifest(paths: list[tuple[str, Path, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": source_id,
                "role": role,
                "path": path.relative_to(ROOT).as_posix(),
                "record_count": count,
                "sha256": sha256(path),
            }
            for source_id, path, role, count in paths
        ]
    )


def build_qa(
    anchors: pd.DataFrame,
    national: pd.DataFrame,
    mapping: pd.DataFrame,
    weights: pd.DataFrame,
    regional: pd.DataFrame,
    cube: pd.DataFrame,
    manuscript_display: pd.DataFrame,
    epri_text: str,
) -> pd.DataFrame:
    modeled = cube[cube["data_center_scenario_id"].ne("source_embedded")]
    identity = (
        modeled["capacity_equivalent_resources_MW"]
        - modeled["scenario_reference_requirement_MW"]
    ) / 1000.0 - modeled["scenario_balance_GW"]
    source_after_2030 = national[
        (national["year"] > 2030) & national["is_LBNL_source_period"]
    ]
    share_sums = weights.groupby("source_allocation_case")["allocation_share"].sum()
    footprint_cases = regional[
        ["footprint_capture_case", "capture_share"]
    ].drop_duplicates()
    expected_source_rows = 3 * 7
    checks = [
        (
            "complete_LBNL_source_anchor_grid",
            len(anchors) == expected_source_rows,
            f"rows={len(anchors)} expected={expected_source_rows}",
        ),
        (
            "reported_2028_reference_anchor",
            np.isclose(
                anchors.loc[
                    anchors["scenario_id"].eq("lbnl_reference")
                    & anchors["year"].eq(2028),
                    "electricity_TWh",
                ].iloc[0],
                464.0,
            ),
            "expected 464 TWh",
        ),
        (
            "reported_2030_anchor_triplet",
            set(
                anchors.loc[anchors["year"].eq(2030), "electricity_TWh"].tolist()
            )
            == {521.0, 649.0, 843.0},
            "expected 521/649/843 TWh",
        ),
        (
            "source_paths_monotone",
            bool(
                anchors.groupby("scenario_id")["electricity_TWh"]
                .apply(lambda x: x.is_monotonic_increasing)
                .all()
            ),
            "all 2024-2030 source-period paths nondecreasing",
        ),
        (
            "no_LBNL_authorship_after_2030",
            source_after_2030.empty,
            f"incorrectly attributed rows={len(source_after_2030)}",
        ),
        (
            "EPRI_factor_four_rule_archived",
            "decline by a factor of 4 for each subsequent 5-year period" in epri_text,
            "official EPRI page asset contains the frozen extension rule",
        ),
        (
            "peak_mapping_identity",
            float(mapping["gamma_identity_residual"].abs().max()) < 1e-8,
            f"max residual={mapping['gamma_identity_residual'].abs().max():.3e}",
        ),
        (
            "spatial_share_closure",
            float((share_sums - 1.0).abs().max()) < 1e-8,
            f"max residual={(share_sums - 1.0).abs().max():.3e}",
        ),
        (
            "footprint_capture_is_explicit_and_bounded",
            set(footprint_cases["footprint_capture_case"])
            == {
                "core_active_reported_mw_share",
                "broad_active_reported_mw_share",
                "core_active_site_count_share",
                "full_national_assignment_upper_stress",
            }
            and footprint_cases["capture_share"].between(0.0, 1.0).all(),
            "national load is not silently assigned in full to the analyzed footprint",
        ),
        (
            "no_negative_incremental_load",
            float(regional["incremental_data_center_load_vs_2025_GW"].min())
            >= -TOL,
            f"minimum={regional['incremental_data_center_load_vs_2025_GW'].min():.6f}",
        ),
        (
            "California_proxy_has_explicit_planning_crosswalk",
            regional.loc[regional["model_region"].eq("WECC-California"), "planning_region"].eq("WECC-California").all()
            and not regional["model_region"].eq("CAISO").any(),
            "NERC-listed component proxy; transferred peak-shape scenarios; not DC attribution",
        ),
        (
            "annual_cube_balance_identity",
            float(identity.abs().max()) < 1e-8,
            f"max residual={identity.abs().max():.3e}",
        ),
        (
            "no_central_scenario_label",
            not cube["data_center_scenario_label"]
            .str.contains("central", case=False, na=False)
            .any(),
            "all modeled paths are named scenarios",
        ),
        (
            "no_AI_data_center_category_label",
            not cube["data_center_scenario_label"]
            .str.contains("AI data", case=False, na=False)
            .any(),
            "data-center load scenario terminology used",
        ),
        (
            "manuscript_display_is_named_scenario_family",
            not manuscript_display.empty
            and set(manuscript_display["overlap_case"])
            == {
                "fully_embedded",
                "half_embedded_diagnostic",
                "fully_additional",
            }
            and set(manuscript_display["scenario_pair_id"])
            == {
                "anticipated_published_response",
                "anticipated_current_response",
                "prospective_published_response",
                "two_year_supply_delay_published_response",
                "joint_two_year_delay_current_response",
            }
            and not manuscript_display[
                "central_forecast_interpretation_allowed"
            ].any(),
            "LBNL Reference display retains all overlap scenarios and is not labelled central",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "detail": detail}
            for check, passed, detail in checks
        ]
    )


def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    table = frame.reset_index() if include_index else frame.reset_index(drop=True)
    columns = [str(column) for column in table.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_results_note(
    national: pd.DataFrame,
    named: pd.DataFrame,
    uncertainty: pd.DataFrame,
    qa: pd.DataFrame,
) -> None:
    epri = national[
        national["post2030_extension_id"].eq("epri_quarter_rate_taper")
        & national["year"].isin([2030, 2035])
    ][
        ["data_center_scenario_id", "year", "electricity_TWh", "average_grid_power_GW"]
    ]
    epri_table = epri.pivot(
        index="data_center_scenario_id",
        columns="year",
        values="electricity_TWh",
    )
    dominant = uncertainty[uncertainty["dominant_component_within_reported_set"]][
        ["planning_region", "component", "balance_range_GW"]
    ]
    negative = named[named["scenario_balance_GW"] < -TOL]
    energy_markdown = markdown_table(epri_table.round(1), include_index=True)
    dominant_markdown = markdown_table(dominant.round(2), include_index=False)
    text = f"""# LBNL 2025 scenario update: computed results

## What changed

- The national load layer now uses the LBNL 2025 Update for total U.S. data-center electricity use through 2030.
- The three national paths are the LBNL compounded lower boundary, Reference Case, and compounded upper boundary. They are scenarios, not probabilities and not a unique central forecast.
- Values for 2031-2035 are explicit continuation scenarios. The cross-source taper-rule case applies EPRI's factor-of-four rule to each LBNL path's 2025-2030 compound annual growth rate; it does not transfer EPRI load levels.
- Annual energy, analyzed-footprint capture, average grid power, regional peak-equivalent load, embedded-load overlap, and planning balance remain separate quantities.

## National energy anchors and continuation

Under the EPRI quarter-rate continuation, the national paths are:

{energy_markdown}

The 2035 values are not LBNL forecasts. They are conditional extensions of the LBNL 2030 endpoints.

## Frozen main-text display family

The compact display table uses the LBNL Reference Case, the core-active reported-MW footprint share, the reference coincidence mapping, and the named EPRI taper-rule continuation. It retains five resource-response pairs and all three overlap conditions as distinct scenarios. This is an illustrative scenario family, not a central, most-likely or probabilistic forecast.

## Planning-pressure interpretation

The named 2030/2035 output contains {len(named):,} region-scenario-condition records; {len(negative):,} are below their region-native reference requirement. This count is descriptive only because it spans deliberately different resource, response, national-load, and overlap conditions.

The largest one-at-a-time 2035 uncertainty component in each region is:

{dominant_markdown}

These component ranges are not confidence intervals and must not be added together.

## Claim boundary

The update supports statements of the form: under a named data-center load, footprint-capture, resource, response, and overlap scenario, a region's planning balance follows the reported annual path. It does not support a claim that the model predicts outages or uniquely identifies the amount of future regional pressure caused by data centers.

## QA

All {len(qa)} automated checks passed: {bool(qa['passed'].all())}.
"""
    NOTE_OUTPUT.write_text(text, encoding="utf-8")
    print(f"[wrote] {NOTE_OUTPUT}")


def main() -> None:
    config = load_config()
    anchors_raw = pd.read_csv(ANCHOR_PATH)
    peak_raw = pd.read_csv(PEAK_MAPPING_PATH)
    weights_raw = pd.read_csv(SPATIAL_WEIGHTS_PATH)
    coverage_raw = pd.read_csv(SPATIAL_COVERAGE_PATH)
    base_cube = pd.read_csv(BASE_CUBE_PATH)
    epri_text = EPRI_PAGE_PATH.read_text(encoding="utf-8")

    anchors = build_anchor_audit(anchors_raw, config)
    national = build_national_paths(anchors_raw, config)
    mapping = build_mapping_crosswalk(peak_raw, config)
    weights = prepare_weights(weights_raw, config)
    cube_case = str(config["spatial_allocation"]["cube_case"])  # type: ignore[index]
    regional = build_regional_paths(
        national, mapping, weights, [cube_case], config
    )
    regional_all = build_regional_paths(
        national,
        mapping,
        weights,
        list(config["spatial_allocation"]["sensitivity_cases"]),  # type: ignore[index]
        config,
    )
    cube = build_cube(base_cube, regional, config)
    summary = build_trajectory_summary(cube)
    named = build_named_years(cube)
    manuscript_display = build_manuscript_display(cube, config)
    uncertainty = build_uncertainty_ranges(cube, config)
    spatial = build_spatial_sensitivity(base_cube, regional_all, config)

    manifest = build_manifest(
        [
            ("scenario_protocol", CONFIG_PATH, "frozen v1.2 scenario rules", 1),
            ("lbnl_2025_update_pdf", LBNL_PDF_PATH, "national source-period energy scenarios", 36),
            ("lbnl_anchor_audit", ANCHOR_PATH, "exact, derived and digitized source anchors", len(anchors_raw)),
            ("epri_load_growth_page", EPRI_HTML_PATH, "official post-2030 continuation source page", 1),
            ("epri_load_growth_page_asset", EPRI_PAGE_PATH, "archived factor-of-four continuation rule", 1),
            ("peak_mapping_factors", PEAK_MAPPING_PATH, "energy-to-peak mapping audit", len(peak_raw)),
            ("spatial_allocation_weights", SPATIAL_WEIGHTS_PATH, "conditional project-evidence location weights", len(weights_raw)),
            ("spatial_footprint_coverage", SPATIAL_COVERAGE_PATH, "evidence-based analyzed-footprint capture cases", len(coverage_raw)),
            ("annual_scenario_cube_v1_1", BASE_CUBE_PATH, "region-native demand-resource-response backbone", len(base_cube)),
        ]
    )
    qa = build_qa(
        anchors,
        national,
        mapping,
        weights,
        regional_all,
        cube,
        manuscript_display,
        epri_text,
    )

    write(anchors, ANCHOR_OUTPUT)
    write(national, NATIONAL_OUTPUT)
    write(mapping, MAPPING_OUTPUT)
    write(regional, REGIONAL_OUTPUT)
    write(cube, CUBE_OUTPUT)
    write(summary, SUMMARY_OUTPUT)
    write(named, NAMED_OUTPUT)
    write(uncertainty, UNCERTAINTY_OUTPUT)
    write(spatial, SPATIAL_OUTPUT)
    write(manifest, MANIFEST_OUTPUT)
    write(qa, QA_OUTPUT)
    write(manuscript_display, MANUSCRIPT_DISPLAY_OUTPUT)
    write_results_note(national, named, uncertainty, qa)

    failures = qa[~qa["passed"]]
    if not failures.empty:
        raise RuntimeError(
            "LBNL 2025 scenario update QA failed:\n"
            + failures.to_string(index=False)
        )


if __name__ == "__main__":
    main()
