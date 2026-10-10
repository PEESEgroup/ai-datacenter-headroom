#!/usr/bin/env python3
"""Build the frozen 2026-2035 regional planning-pressure scenario cube.

The builder adds new outputs only. It does not modify the earlier WP3-WP6
results or any submitted-manuscript artifact.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config/annual_scenarios_v1.json"
MATRIX_PATH = (
    WORKSPACE
    / "natcomm_core_method_audit_batch10_20260917/results/02_condition_matrix_region_year.csv"
)
DC_STRESS_PATH = (
    WORKSPACE
    / "natcomm_major_revision_batch7_20260917/results/29_bottom_up_dc_scenarios_spatial_refresh.csv"
)
DC_IDENTIFICATION_PATH = (
    WORKSPACE
    / "natcomm_dc_partial_identification_batch11_20260917/results/02_matched_DC_identification_region_year.csv"
)
ERCOT_RESPONSE_PATH = ROOT / "results/04_ercot_large_load_response_audit.csv"

DEFINITION_OUTPUT = ROOT / "results/80_annual_scenario_definition.csv"
CUBE_OUTPUT = ROOT / "results/81_annual_scenario_cube.csv"
SUMMARY_OUTPUT = ROOT / "results/82_annual_scenario_trajectory_summary.csv"
HEADLINE_OUTPUT = ROOT / "results/83_headline_scenario_trajectories.csv"
IDENTIFIED_OUTPUT = ROOT / "results/84_source_identified_dc_counterfactuals.csv"
IDENTIFIED_SUMMARY_OUTPUT = ROOT / "results/85_source_identified_dc_summary.csv"
FINDINGS_OUTPUT = ROOT / "results/86_annual_scenario_key_findings.csv"
SOURCE_MANIFEST_OUTPUT = ROOT / "results/87_annual_scenario_source_manifest.csv"
QA_OUTPUT = ROOT / "results/88_annual_scenario_cube_qa.csv"
NOTE_OUTPUT = ROOT / "docs/16_annual_scenario_results.md"

TOL = 1e-8


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(
    matrix: pd.DataFrame,
    dc_stress: pd.DataFrame,
    identification: pd.DataFrame,
    ercot_response: pd.DataFrame,
) -> pd.DataFrame:
    inputs = [
        ("annual_scenario_freeze", CONFIG_PATH, "frozen scenario rules", 1),
        ("seven_region_condition_matrix", MATRIX_PATH, "annual demand-resource-response backbone", len(matrix)),
        ("fixed_total_dc_spatial_paths", DC_STRESS_PATH, "low/mid/high data-center stress paths", len(dc_stress)),
        ("matched_dc_identification", DC_IDENTIFICATION_PATH, "source-path attribution classes and annual bounds", len(identification)),
        ("ercot_large_load_response", ERCOT_RESPONSE_PATH, "explicit large-load response linkage audit", len(ercot_response)),
    ]
    return pd.DataFrame(
        [
            {
                "source_id": source_id,
                "role": role,
                "path": path.name,
                "record_count": count,
                "sha256": sha256(path),
            }
            for source_id, path, role, count in inputs
        ]
    )


def resource_metadata(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in config["resource_conditions"]  # type: ignore[index]
    }


def recompute_balance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["reference_requirement_MW"] = out["net_peak_demand_MW"] * (
        1.0 + out["reference_margin_level_pct"] / 100.0
    )
    out["margin_MW"] = (
        out["capacity_equivalent_resources_MW"] - out["reference_requirement_MW"]
    )
    out["margin_GW"] = out["margin_MW"] / 1000.0
    out["reserve_margin_pct"] = (
        out["capacity_equivalent_resources_MW"] / out["net_peak_demand_MW"] - 1.0
    ) * 100.0
    out["gap_to_RML_pp"] = (
        out["reserve_margin_pct"] - out["reference_margin_level_pct"]
    )
    out["condition_status"] = np.where(
        out["margin_GW"] >= -TOL, "at_or_above_RML", "below_RML"
    )
    return out


def delayed_positive_net_path(
    group: pd.DataFrame, lag_years: int, condition_id: str
) -> pd.DataFrame:
    ordered = group.sort_values("year").copy()
    years = ordered["year"].astype(int).tolist()
    resources = ordered["capacity_equivalent_resources_MW"].astype(float).tolist()
    scheduled: dict[int, float] = {year: 0.0 for year in years}

    for index in range(1, len(years)):
        delta = resources[index] - resources[index - 1]
        target_year = years[index] + lag_years if delta > 0 else years[index]
        if target_year in scheduled:
            scheduled[target_year] += delta

    level = resources[0]
    delayed_values: list[float] = []
    for year in years:
        if year != years[0]:
            level += scheduled[year]
        delayed_values.append(level)

    ordered["resource_path"] = condition_id
    ordered["capacity_equivalent_resources_MW"] = delayed_values
    ordered["resource_condition_type"] = "timing_stress"
    ordered["resource_delay_years"] = lag_years
    ordered["resource_rule"] = (
        f"positive annual anticipated-path net changes delayed {lag_years} year(s); "
        "negative annual net changes applied on source schedule"
    )
    return recompute_balance(ordered)


def build_resource_response_panel(
    matrix: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    metadata = resource_metadata(config)
    source = matrix[matrix["resource_path"].isin(["anticipated", "prospective"])].copy()
    source["resource_condition_type"] = "source_condition"
    source["resource_delay_years"] = 0
    source["resource_rule"] = source["resource_path"].map(
        {condition: str(spec["rule"]) for condition, spec in metadata.items()}
    )

    delayed_frames: list[pd.DataFrame] = []
    anticipated = matrix[matrix["resource_path"].eq("anticipated")]
    for lag in (1, 2, 3):
        condition_id = f"anticipated_positive_net_additions_delayed_{lag}y"
        for _, group in anticipated.groupby(
            ["planning_region", "dr_treatment"], sort=True
        ):
            delayed_frames.append(delayed_positive_net_path(group, lag, condition_id))

    panel = pd.concat([source, *delayed_frames], ignore_index=True)
    panel["cross_region_GW_aggregation_allowed"] = False
    panel["reliability_event_interpretation_allowed"] = False
    return panel.sort_values(
        ["planning_region", "resource_path", "dr_treatment", "year"]
    ).reset_index(drop=True)


def dc_condition_records(config: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "dc_condition_id": "source_embedded",
            "dc_model_scenario": "source_embedded",
            "overlap_case": "not_applicable",
            "assumed_embedded_share": 1.0,
            "additional_share": 0.0,
        }
    ]
    dc_config = config["dc_conditions"]  # type: ignore[index]
    for scenario in dc_config["modeled_scenarios"]:
        for overlap in dc_config["overlap_cases"]:
            records.append(
                {
                    "dc_condition_id": f"modeled_{scenario}__{overlap['id']}",
                    "dc_model_scenario": f"modeled_{scenario}",
                    "overlap_case": overlap["id"],
                    "assumed_embedded_share": float(
                        overlap["assumed_embedded_share"]
                    ),
                    "additional_share": float(overlap["additional_share"]),
                }
            )
    return records


def build_cube(
    panel: pd.DataFrame, dc_stress: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    stress = dc_stress[
        dc_stress["planning_region"].notna()
        & dc_stress["year"].between(2026, 2035)
    ].copy()
    stress["dc_model_scenario"] = "modeled_" + stress["scenario"].astype(str)
    stress_lookup = stress.set_index(
        ["planning_region", "year", "dc_model_scenario"]
    )
    matched_regions = set(
        config["dc_conditions"]["matched_planning_regions"]  # type: ignore[index]
    )

    records: list[dict[str, object]] = []
    for base in panel.to_dict(orient="records"):
        region = str(base["planning_region"])
        year = int(base["year"])
        mu = float(base["reference_margin_level_pct"]) / 100.0
        base_balance = float(base["margin_GW"])
        base_required = max(0.0, -base_balance)

        for condition in dc_condition_records(config):
            modeled = condition["dc_model_scenario"] != "source_embedded"
            applicable = True
            reason = ""
            modeled_increment = 0.0
            if modeled:
                key = (region, year, condition["dc_model_scenario"])
                if region not in matched_regions or key not in stress_lookup.index:
                    applicable = False
                    reason = (
                        "author-modeled DC footprint is not matched to this planning footprint"
                        if region == "WECC-California"
                        else "matched annual modeled DC path unavailable"
                    )
                    modeled_increment = np.nan
                else:
                    modeled_increment = float(
                        stress_lookup.loc[key, "modeled_incremental_DC_GW"]
                    )

            if applicable:
                additional_dc = modeled_increment * float(condition["additional_share"])
                scenario_gross = float(base["gross_peak_demand_MW"]) + additional_dc * 1000.0
                scenario_net = scenario_gross - float(base["credited_DR_MW"])
                scenario_requirement = scenario_net * (1.0 + mu)
                scenario_balance = (
                    float(base["capacity_equivalent_resources_MW"])
                    - scenario_requirement
                ) / 1000.0
                reserve_margin = (
                    float(base["capacity_equivalent_resources_MW"]) / scenario_net - 1.0
                ) * 100.0
                gap_pp = reserve_margin - float(base["reference_margin_level_pct"])
                required = max(0.0, -scenario_balance)
                pressure = additional_dc * (1.0 + mu)
                required_increment = max(0.0, required - base_required)
                status = (
                    "at_or_above_RML" if scenario_balance >= -TOL else "below_RML"
                )
            else:
                additional_dc = np.nan
                scenario_gross = np.nan
                scenario_net = np.nan
                scenario_requirement = np.nan
                scenario_balance = np.nan
                reserve_margin = np.nan
                gap_pp = np.nan
                required = np.nan
                pressure = np.nan
                required_increment = np.nan
                status = "not_applicable"

            records.append(
                {
                    "planning_region": region,
                    "nerc_assessment_area": base["nerc_assessment_area"],
                    "year": year,
                    "resource_condition": base["resource_path"],
                    "resource_condition_type": base["resource_condition_type"],
                    "resource_delay_years": int(base["resource_delay_years"]),
                    "resource_rule": base["resource_rule"],
                    "response_condition": base["dr_treatment"],
                    **condition,
                    "scenario_applicable": applicable,
                    "not_applicable_reason": reason,
                    "base_gross_peak_demand_MW": float(base["gross_peak_demand_MW"]),
                    "credited_response_MW": float(base["credited_DR_MW"]),
                    "base_net_peak_demand_MW": float(base["net_peak_demand_MW"]),
                    "capacity_equivalent_resources_MW": float(
                        base["capacity_equivalent_resources_MW"]
                    ),
                    "reference_margin_level_pct": float(
                        base["reference_margin_level_pct"]
                    ),
                    "base_reference_requirement_MW": float(
                        base["reference_requirement_MW"]
                    ),
                    "base_balance_GW": base_balance,
                    "base_required_contribution_GW": base_required,
                    "modeled_incremental_DC_GW": modeled_increment,
                    "additional_DC_stress_GW": additional_dc,
                    "scenario_gross_peak_demand_MW": scenario_gross,
                    "scenario_net_peak_demand_MW": scenario_net,
                    "scenario_reference_requirement_MW": scenario_requirement,
                    "scenario_reserve_margin_pct": reserve_margin,
                    "scenario_gap_to_reference_pp": gap_pp,
                    "scenario_balance_GW": scenario_balance,
                    "scenario_required_contribution_GW": required,
                    "dc_additional_pressure_GW": pressure,
                    "dc_associated_required_increment_GW": required_increment,
                    "condition_status": status,
                    "dc_stress_interpretation": (
                        "source demand path; no author-modeled increment added"
                        if not modeled
                        else "deterministic additional-load stress; not an identified DC counterfactual"
                    ),
                    "preferred_dc_attribution_output": (
                        "source_identified_path_counterfactual"
                        if region in {"MISO", "ERCOT"}
                        else (
                            "generic_overlap_stress"
                            if region in matched_regions
                            else "not_identified_for_matched_footprint"
                        )
                    ),
                    "generic_dc_stress_primary_reporting_allowed": (
                        (not modeled) or region not in {"MISO", "ERCOT"}
                    ) and applicable,
                    "cross_region_GW_aggregation_allowed": False,
                    "reliability_event_interpretation_allowed": False,
                    "causal_attribution_allowed": False,
                    "result_status": "computed_requires_independent_review",
                }
            )
    return pd.DataFrame(records).sort_values(
        [
            "planning_region",
            "resource_condition",
            "response_condition",
            "dc_condition_id",
            "year",
        ]
    ).reset_index(drop=True)


def trajectory_metrics(group: pd.DataFrame) -> dict[str, object]:
    ordered = group.sort_values("year")
    balance = ordered["scenario_balance_GW"].astype(float)
    years = ordered["year"].astype(int)
    below = balance < -TOL
    below_years = years[below].tolist()
    first_below = min(below_years) if below_years else np.nan
    last_below = max(below_years) if below_years else np.nan
    recovers = bool(
        below_years
        and ((years > int(first_below)) & (balance >= -TOL)).any()
    )
    if not below_years:
        category = "above_reference_through_horizon"
    elif len(below_years) == len(ordered):
        category = "below_reference_throughout_horizon"
    elif recovers:
        category = "non_monotonic_or_recovers"
    else:
        category = "crosses_and_remains_below_through_horizon"

    minimum_index = balance.idxmin()
    return {
        "first_year": int(years.iloc[0]),
        "last_year": int(years.iloc[-1]),
        "first_balance_GW": float(balance.iloc[0]),
        "last_balance_GW": float(balance.iloc[-1]),
        "change_in_balance_GW": float(balance.iloc[-1] - balance.iloc[0]),
        "minimum_balance_GW": float(balance.loc[minimum_index]),
        "minimum_balance_year": int(ordered.loc[minimum_index, "year"]),
        "first_below_reference_year": first_below,
        "last_below_reference_year": last_below,
        "years_below_reference": int(below.sum()),
        "recovers_after_first_below": recovers,
        "cumulative_negative_balance_GW_year": float((-balance.clip(upper=0)).sum()),
        "trajectory_category": category,
    }


def build_summary(cube: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "planning_region",
        "resource_condition",
        "response_condition",
        "dc_condition_id",
        "dc_model_scenario",
        "overlap_case",
        "assumed_embedded_share",
        "additional_share",
    ]
    rows: list[dict[str, object]] = []
    applicable = cube[cube["scenario_applicable"]].copy()
    for values, group in applicable.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update(trajectory_metrics(group))
        row["latest_required_contribution_GW"] = float(
            group.sort_values("year")["scenario_required_contribution_GW"].iloc[-1]
        )
        row["latest_dc_associated_required_increment_GW"] = float(
            group.sort_values("year")["dc_associated_required_increment_GW"].iloc[-1]
        )
        row["result_status"] = "computed_requires_independent_review"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_definition(config: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(config["headline_scenarios"]).assign(  # type: ignore[arg-type]
        protocol_id=config["protocol_id"],
        frozen_on=config["frozen_on"],
        probabilistic_forecast=False,
        unique_most_likely_future=False,
    )


def build_headline(cube: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in config["headline_scenarios"]:  # type: ignore[index]
        subset = cube[
            cube["resource_condition"].eq(spec["resource_condition"])
            & cube["response_condition"].eq(spec["response_condition"])
            & cube["dc_model_scenario"].eq(spec["dc_condition"])
        ].copy()
        if spec["overlap_case"] != "all_frozen_overlap_cases":
            subset = subset[subset["overlap_case"].eq(spec["overlap_case"])]
        subset["headline_scenario_id"] = spec["id"]
        subset["headline_role"] = spec["role"]
        frames.append(subset)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["headline_scenario_id", "planning_region", "overlap_case", "year"]
    ).reset_index(drop=True)


def source_identification_role(region: str, row: pd.Series) -> str:
    if region == "MISO":
        return "source_reported_DC_path_with_linked_response_interval"
    if region == "ERCOT":
        return "source_reported_DC_path_with_explicit_large_load_response_interval"
    if region == "PJM":
        return "cross_source_strict_DC_floor_diagnostic"
    if region == "NYISO":
        return "mixed_large_load_upper_bound_not_DC_point_estimate"
    return "not_identified"


def source_path_value_status(region: str, year: int, applicable: bool) -> str:
    if not applicable:
        return "not_identified"
    if region in {"MISO", "ERCOT"}:
        if year in {2027, 2030, 2035}:
            return "source_reported_anchor"
        return "transparent_interpolation_between_source_anchors"
    if region == "PJM":
        return "raw_table_reconstructed_annual_strict_floor"
    if region == "NYISO":
        return "source_reported_annual_mixed_large_load_bound"
    return "not_identified"


def build_identified_counterfactuals(
    panel: pd.DataFrame,
    identification: pd.DataFrame,
    ercot_response: pd.DataFrame,
) -> pd.DataFrame:
    identification = identification.copy()
    identification["year"] = identification["year"].astype(int)
    identified_lookup = identification.set_index(["planning_region", "year"])
    ercot_response = ercot_response.copy()
    ercot_response["year"] = ercot_response["year"].astype(int)
    ercot_lookup = ercot_response.set_index("year")
    ercot_base = float(
        ercot_lookup.loc[2025, "source_large_load_curtailment_MW"]
    ) / 1000.0

    records: list[dict[str, object]] = []
    for base in panel.to_dict(orient="records"):
        region = str(base["planning_region"])
        year = int(base["year"])
        key = (region, year)
        identified = key in identified_lookup.index
        if identified:
            evidence = identified_lookup.loc[key]
            q_lower = float(evidence["q_lower_GW"]) if pd.notna(evidence["q_lower_GW"]) else np.nan
            q_upper = float(evidence["q_upper_GW"]) if pd.notna(evidence["q_upper_GW"]) else np.nan
            identification_class = str(evidence["identification_class"])
        else:
            evidence = pd.Series(dtype=object)
            q_lower = np.nan
            q_upper = np.nan
            identification_class = "not_identified"

        applicable = region in {"MISO", "ERCOT", "PJM", "NYISO"} and (
            pd.notna(q_lower) or pd.notna(q_upper)
        )
        mu = float(base["reference_margin_level_pct"]) / 100.0
        with_balance = float(base["margin_GW"])
        with_required = max(0.0, -with_balance)

        if applicable and region in {"MISO", "ERCOT"}:
            q = q_upper
            if region == "ERCOT" and base["dr_treatment"] == "published_all":
                explicit_total = float(
                    ercot_lookup.loc[year, "source_large_load_curtailment_MW"]
                ) / 1000.0
                linked_upper = min(q, max(0.0, explicit_total - ercot_base))
            elif region == "MISO" and base["dr_treatment"] != "none":
                linked_upper = min(q, float(base["credited_DR_MW"]) / 1000.0)
            else:
                linked_upper = 0.0
            e2_lower = (1.0 + mu) * (q - linked_upper)
            e2_upper = (1.0 + mu) * q
            complete_path_upper_identified = True
        elif applicable and region == "PJM":
            q = q_lower
            linked_upper = (
                0.0
                if base["dr_treatment"] == "none"
                else min(q, float(base["credited_DR_MW"]) / 1000.0)
            )
            e2_lower = (1.0 + mu) * (q - linked_upper)
            e2_upper = (1.0 + mu) * q
            complete_path_upper_identified = False
        elif applicable and region == "NYISO":
            linked_upper = np.nan
            e2_lower = 0.0
            e2_upper = (1.0 + mu) * q_upper
            complete_path_upper_identified = False
        else:
            linked_upper = np.nan
            e2_lower = np.nan
            e2_upper = np.nan
            complete_path_upper_identified = False

        if applicable:
            no_new_balance_lower = with_balance + e2_lower
            no_new_balance_upper = with_balance + e2_upper
            no_new_required_lower = max(0.0, -no_new_balance_upper)
            no_new_required_upper = max(0.0, -no_new_balance_lower)
            e3b_lower = max(0.0, with_required - no_new_required_upper)
            e3b_upper = max(0.0, with_required - no_new_required_lower)
        else:
            no_new_balance_lower = np.nan
            no_new_balance_upper = np.nan
            no_new_required_lower = np.nan
            no_new_required_upper = np.nan
            e3b_lower = np.nan
            e3b_upper = np.nan

        records.append(
            {
                "planning_region": region,
                "year": year,
                "resource_condition": base["resource_path"],
                "response_condition": base["dr_treatment"],
                "identification_applicable": applicable,
                "identification_class": identification_class,
                "counterfactual_role": source_identification_role(region, evidence),
                "path_value_status": source_path_value_status(region, year, applicable),
                "path_lower_GW": q_lower,
                "path_upper_GW": q_upper,
                "linked_response_lower_GW": 0.0 if applicable and region != "NYISO" else np.nan,
                "linked_response_upper_GW": linked_upper,
                "complete_path_upper_identified": complete_path_upper_identified,
                "with_path_balance_GW": with_balance,
                "with_path_required_contribution_GW": with_required,
                "path_pressure_contribution_lower_GW": e2_lower,
                "path_pressure_contribution_upper_GW": e2_upper,
                "no_new_path_balance_lower_GW": no_new_balance_lower,
                "no_new_path_balance_upper_GW": no_new_balance_upper,
                "no_new_path_required_contribution_lower_GW": no_new_required_lower,
                "no_new_path_required_contribution_upper_GW": no_new_required_upper,
                "path_associated_required_increment_lower_GW": e3b_lower,
                "path_associated_required_increment_upper_GW": e3b_upper,
                "causal_interpretation_allowed": False,
                "result_status": "computed_requires_independent_review",
            }
        )
    return pd.DataFrame(records).sort_values(
        ["planning_region", "resource_condition", "response_condition", "year"]
    ).reset_index(drop=True)


def first_below(year: pd.Series, balance: pd.Series) -> float:
    selected = year[balance < -TOL]
    return float(selected.min()) if len(selected) else np.nan


def build_identified_summary(identified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    applicable = identified[identified["identification_applicable"]]
    for keys, group in applicable.groupby(
        ["planning_region", "resource_condition", "response_condition"], sort=True
    ):
        ordered = group.sort_values("year")
        latest = ordered.iloc[-1]
        rows.append(
            {
                "planning_region": keys[0],
                "resource_condition": keys[1],
                "response_condition": keys[2],
                "counterfactual_role": latest["counterfactual_role"],
                "complete_path_upper_identified": bool(
                    latest["complete_path_upper_identified"]
                ),
                "first_below_with_path_year": first_below(
                    ordered["year"], ordered["with_path_balance_GW"]
                ),
                "first_below_no_new_path_lower_balance_year": first_below(
                    ordered["year"], ordered["no_new_path_balance_lower_GW"]
                ),
                "first_below_no_new_path_upper_balance_year": first_below(
                    ordered["year"], ordered["no_new_path_balance_upper_GW"]
                ),
                "latest_with_path_balance_GW": float(latest["with_path_balance_GW"]),
                "latest_pressure_lower_GW": float(
                    latest["path_pressure_contribution_lower_GW"]
                ),
                "latest_pressure_upper_GW": float(
                    latest["path_pressure_contribution_upper_GW"]
                ),
                "latest_required_increment_lower_GW": float(
                    latest["path_associated_required_increment_lower_GW"]
                ),
                "latest_required_increment_upper_GW": float(
                    latest["path_associated_required_increment_upper_GW"]
                ),
                "maximum_required_increment_upper_GW": float(
                    ordered["path_associated_required_increment_upper_GW"].max()
                ),
                "result_status": "computed_requires_independent_review",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["planning_region", "resource_condition", "response_condition"]
    ).reset_index(drop=True)


def build_findings(headline: pd.DataFrame) -> pd.DataFrame:
    keys = ["headline_scenario_id", "planning_region", "overlap_case"]
    rows: list[dict[str, object]] = []
    applicable = headline[headline["scenario_applicable"]]
    for values, group in applicable.groupby(keys, sort=True):
        metrics = trajectory_metrics(group)
        latest = group.sort_values("year").iloc[-1]
        rows.append(
            {
                "headline_scenario_id": values[0],
                "planning_region": values[1],
                "overlap_case": values[2],
                **metrics,
                "latest_required_contribution_GW": float(
                    latest["scenario_required_contribution_GW"]
                ),
                "latest_dc_associated_required_increment_GW": float(
                    latest["dc_associated_required_increment_GW"]
                ),
                "generic_dc_stress_primary_reporting_allowed": bool(
                    latest["generic_dc_stress_primary_reporting_allowed"]
                ),
                "interpretation": "conditional planning path; not a reliability forecast",
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_qa(
    panel: pd.DataFrame,
    cube: pd.DataFrame,
    summary: pd.DataFrame,
    headline: pd.DataFrame,
    identified: pd.DataFrame,
    source_manifest: pd.DataFrame,
    config: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("CUBE_QA01_seven_regions", cube["planning_region"].nunique() == 7, "blocking"))
    checks.append(("CUBE_QA02_complete_horizon", set(cube["year"]) == set(range(2026, 2036)), "blocking"))
    checks.append(("CUBE_QA03_resource_response_panel_rows", len(panel) == 1050, "blocking"))
    checks.append(("CUBE_QA04_cube_rows", len(cube) == 10500, "blocking"))
    checks.append(("CUBE_QA05_unique_cube_keys", not cube.duplicated(["planning_region", "year", "resource_condition", "response_condition", "dc_condition_id"]).any(), "blocking"))
    checks.append(("CUBE_QA06_applicable_rows", int(cube["scenario_applicable"].sum()) == 9150, "blocking"))

    applicable_counts = cube[cube["scenario_applicable"]].groupby(
        ["planning_region", "resource_condition", "response_condition", "dc_condition_id"]
    )["year"].nunique()
    checks.append(("CUBE_QA07_each_applicable_trajectory_has_ten_years", applicable_counts.eq(10).all(), "blocking"))

    source_ref = cube[
        cube["resource_condition"].eq("anticipated")
        & cube["response_condition"].eq("published_all")
        & cube["dc_condition_id"].eq("source_embedded")
    ].sort_values(["planning_region", "year"])
    expected_ref = matrix[
        matrix["resource_path"].eq("anticipated")
        & matrix["dr_treatment"].eq("published_all")
    ].sort_values(["planning_region", "year"])
    checks.append(("CUBE_QA08_source_reference_reproduced", np.allclose(source_ref["scenario_balance_GW"], expected_ref["margin_GW"], atol=TOL), "blocking"))

    applicable = cube[cube["scenario_applicable"]]
    checks.append(("CUBE_QA09_positive_part_identity", np.allclose(applicable["scenario_required_contribution_GW"], np.maximum(0.0, -applicable["scenario_balance_GW"]), atol=TOL), "blocking"))
    checks.append(("CUBE_QA10_dc_pressure_identity", np.allclose(applicable["base_balance_GW"] - applicable["scenario_balance_GW"], applicable["dc_additional_pressure_GW"], atol=TOL), "blocking"))

    source_only = cube[cube["dc_condition_id"].eq("source_embedded")]
    checks.append(("CUBE_QA11_source_embedded_has_zero_added_stress", np.allclose(source_only["additional_DC_stress_GW"], 0.0, atol=TOL), "blocking"))
    wecc_modeled = cube[
        cube["planning_region"].eq("WECC-California")
        & ~cube["dc_condition_id"].eq("source_embedded")
    ]
    checks.append(("CUBE_QA12_wecc_modeled_cells_explicitly_not_applicable", (~wecc_modeled["scenario_applicable"]).all() and wecc_modeled["scenario_balance_GW"].isna().all(), "blocking"))

    pivot = panel.pivot_table(
        index=["planning_region", "dr_treatment", "year"],
        columns="resource_path",
        values="capacity_equivalent_resources_MW",
    )
    delayed_columns = [
        "anticipated_positive_net_additions_delayed_1y",
        "anticipated_positive_net_additions_delayed_2y",
        "anticipated_positive_net_additions_delayed_3y",
    ]
    checks.append(("CUBE_QA13_delay_stresses_do_not_exceed_anticipated", all((pivot[column] <= pivot["anticipated"] + TOL).all() for column in delayed_columns), "blocking"))

    checks.append(("CUBE_QA14_summary_rows", len(summary) == 915, "blocking"))
    expected_headlines = {str(item["id"]) for item in config["headline_scenarios"]}  # type: ignore[index]
    checks.append(("CUBE_QA15_all_headline_scenarios_present", set(headline["headline_scenario_id"]) == expected_headlines, "blocking"))
    checks.append(("CUBE_QA16_no_causal_or_reliability_flags", (not cube["causal_attribution_allowed"].astype(bool).any()) and (not cube["reliability_event_interpretation_allowed"].astype(bool).any()), "blocking"))
    checks.append(("CUBE_QA17_no_cross_region_GW_aggregation", not cube["cross_region_GW_aggregation_allowed"].astype(bool).any(), "blocking"))

    identified_applicable = identified[identified["identification_applicable"]]
    checks.append(("CUBE_QA18_identified_rows_cover_four_evidence_regions", set(identified_applicable["planning_region"]) == {"MISO", "ERCOT", "PJM", "NYISO"}, "blocking"))
    checks.append(("CUBE_QA19_identified_bounds_ordered", (identified_applicable["path_pressure_contribution_lower_GW"] <= identified_applicable["path_pressure_contribution_upper_GW"] + TOL).all() and (identified_applicable["path_associated_required_increment_lower_GW"] <= identified_applicable["path_associated_required_increment_upper_GW"] + TOL).all(), "blocking"))
    checks.append(("CUBE_QA20_e3b_not_larger_than_e2", (identified_applicable["path_associated_required_increment_upper_GW"] <= identified_applicable["path_pressure_contribution_upper_GW"] + TOL).all(), "blocking"))
    unidentified = identified[~identified["identification_applicable"]]
    checks.append(("CUBE_QA21_unknown_identification_remains_null", unidentified["path_pressure_contribution_lower_GW"].isna().all() and unidentified["path_pressure_contribution_upper_GW"].isna().all(), "blocking"))
    ercot_nonpublished = identified[
        identified["planning_region"].eq("ERCOT")
        & identified["identification_applicable"]
        & ~identified["response_condition"].eq("published_all")
    ]
    checks.append(("CUBE_QA22_ercot_explicit_linkage_only_in_published_response", np.allclose(ercot_nonpublished["linked_response_upper_GW"], 0.0, atol=TOL), "blocking"))
    anchor_rows = identified[
        identified["planning_region"].isin(["MISO", "ERCOT"])
        & identified["identification_applicable"]
        & identified["path_value_status"].eq("source_reported_anchor")
    ]
    checks.append(("CUBE_QA23_miso_ercot_anchor_status_only_at_reported_years", set(anchor_rows["year"]) == {2027, 2030, 2035}, "blocking"))
    modeled_deep = cube[
        cube["planning_region"].isin(["MISO", "ERCOT"])
        & ~cube["dc_model_scenario"].eq("source_embedded")
    ]
    checks.append(("CUBE_QA24_generic_deep_case_stress_not_primary_attribution", (~modeled_deep["generic_dc_stress_primary_reporting_allowed"]).all(), "blocking"))
    checks.append(("CUBE_QA25_source_manifest_complete", len(source_manifest) == 5 and source_manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all(), "blocking"))
    checks.append(("CUBE_QA26_no_unique_prediction_label", not bool(config.get("unique_most_likely_future", False)), "blocking"))
    return pd.DataFrame(checks, columns=["check_id", "passed", "severity"])


def fmt_year(value: object) -> str:
    return "none through 2035" if pd.isna(value) else str(int(float(value)))


def build_note(findings: pd.DataFrame, identified_summary: pd.DataFrame) -> str:
    reference = findings[
        findings["headline_scenario_id"].eq("source_published_reference")
    ].set_index("planning_region")
    delay = findings[
        findings["headline_scenario_id"].eq("two_year_supply_timing_stress")
    ].set_index("planning_region")
    joint = findings[
        findings["headline_scenario_id"].eq("joint_pressure_envelope")
        & findings["overlap_case"].eq("fully_additional")
    ].set_index("planning_region")

    region_lines: list[str] = []
    for region in sorted(reference.index):
        row = reference.loc[region]
        region_lines.append(
            f"- **{region}:** source-reference first below = "
            f"{fmt_year(row['first_below_reference_year'])}; 2035 balance = "
            f"{row['last_balance_GW']:.1f} GW; minimum = "
            f"{row['minimum_balance_GW']:.1f} GW ({int(row['minimum_balance_year'])})."
        )

    delay_lines: list[str] = []
    for region in sorted(delay.index):
        source_year = reference.loc[region, "first_below_reference_year"]
        delay_year = delay.loc[region, "first_below_reference_year"]
        delay_lines.append(
            f"- **{region}:** source reference {fmt_year(source_year)}; "
            f"two-year timing stress {fmt_year(delay_year)}."
        )

    joint_lines: list[str] = []
    for region in sorted(joint.index):
        row = joint.loc[region]
        qualifier = (
            " Consistency stress only; the source-path result governs attribution."
            if not bool(row["generic_dc_stress_primary_reporting_allowed"])
            else ""
        )
        joint_lines.append(
            f"- **{region}:** fully-additional high-DC joint stress first below = "
            f"{fmt_year(row['first_below_reference_year'])}; 2035 balance = "
            f"{row['last_balance_GW']:.1f} GW.{qualifier}"
        )

    identified_lines: list[str] = []
    selected = identified_summary[
        identified_summary["resource_condition"].eq("anticipated")
        & identified_summary["response_condition"].eq("published_all")
    ]
    for _, row in selected.sort_values("planning_region").iterrows():
        identified_lines.append(
            f"- **{row['planning_region']}:** 2035 source-evidence pressure interval "
            f"{row['latest_pressure_lower_GW']:.1f} to "
            f"{row['latest_pressure_upper_GW']:.1f} GW; associated required-"
            f"contribution increment {row['latest_required_increment_lower_GW']:.1f} "
            f"to {row['latest_required_increment_upper_GW']:.1f} GW."
        )

    return f"""# Annual scenario-cube results

Status: computed under the 2026-09-30 frozen rules; independent source and
implementation review remains required before manuscript use.

## Source-published reference trajectory

{chr(10).join(region_lines)}

## Two-year positive-net-addition timing stress

This diagnostic shifts positive annual net changes in the anticipated resource
path and leaves negative changes on schedule. It is not a project-delivery
forecast.

{chr(10).join(delay_lines)}

## Joint pressure envelope

The values below show the fully-additional endpoint of the high-DC overlap
envelope with two-year delayed positive net additions and current-only
response. The fully embedded and half-embedded endpoints remain in the output.

{chr(10).join(joint_lines)}

WECC-California has no joint DC-stress result because the author-modeled
California path is CAISO-based and does not match the broader planning
footprint.

## Source-identifiable pathway diagnostics

{chr(10).join(identified_lines)}

MISO and ERCOT use source-reported DC paths with explicit linkage intervals.
PJM is a strict cross-source floor, not a complete regional path. NYISO is a
mixed-large-load upper bound, not a DC point estimate. SPP, ISO-NE and
WECC-California remain unidentified for source-path removal.

## Interpretation boundary

All crossing years are conditional on the stated ten-year scenario. Negative
balances indicate a shortfall relative to a native planning reference under
that condition; they are not outage forecasts. The full cube is retained so
the headline subset cannot conceal response, overlap or timing dependence.
"""


def main() -> None:
    config = load_config()
    matrix = pd.read_csv(MATRIX_PATH)
    dc_stress = pd.read_csv(DC_STRESS_PATH)
    identification = pd.read_csv(DC_IDENTIFICATION_PATH)
    ercot_response = pd.read_csv(ERCOT_RESPONSE_PATH)

    panel = build_resource_response_panel(matrix, config)
    cube = build_cube(panel, dc_stress, config)
    summary = build_summary(cube)
    definition = build_definition(config)
    headline = build_headline(cube, config)
    identified = build_identified_counterfactuals(
        panel, identification, ercot_response
    )
    identified_summary = build_identified_summary(identified)
    findings = build_findings(headline)
    source_manifest = build_source_manifest(
        matrix, dc_stress, identification, ercot_response
    )
    qa = build_qa(
        panel,
        cube,
        summary,
        headline,
        identified,
        source_manifest,
        config,
        matrix,
    )

    DEFINITION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    definition.to_csv(DEFINITION_OUTPUT, index=False)
    cube.to_csv(CUBE_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    headline.to_csv(HEADLINE_OUTPUT, index=False)
    identified.to_csv(IDENTIFIED_OUTPUT, index=False)
    identified_summary.to_csv(IDENTIFIED_SUMMARY_OUTPUT, index=False)
    findings.to_csv(FINDINGS_OUTPUT, index=False)
    source_manifest.to_csv(SOURCE_MANIFEST_OUTPUT, index=False)
    qa.to_csv(QA_OUTPUT, index=False)
    NOTE_OUTPUT.write_text(build_note(findings, identified_summary), encoding="utf-8")

    failed = qa.loc[~qa["passed"], "check_id"].tolist()
    if failed:
        raise SystemExit(f"Blocking annual-cube QA failures: {failed}")

    print(f"Built {len(cube)} annual scenario-cube rows")
    print(f"Applicable annual rows: {int(cube['scenario_applicable'].sum())}")
    print(f"Trajectory summaries: {len(summary)}")
    print(f"Source-identification rows: {len(identified)}")
    print(f"Blocking QA passed: {len(qa)}/{len(qa)}")


if __name__ == "__main__":
    main()
