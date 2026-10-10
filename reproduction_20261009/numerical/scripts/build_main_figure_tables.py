#!/usr/bin/env python3
"""Build compact plotting tables for the frozen main-figure design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = 1e-8
CORE_RESOURCES = ["anticipated", "prospective"]
CORE_RESPONSES = ["current_only", "published_all"]
FOCUS_REGIONS = ["MISO", "PJM", "SPP", "ERCOT"]
DIAGNOSTIC_REGIONS = ["SPP", "NYISO", "ISO-NE"]


def write(frame: pd.DataFrame, name: str) -> None:
    path = RESULTS / name
    frame.to_csv(path, index=False)
    print(f"[wrote] {path} ({len(frame):,} rows)")


def build_fig2() -> pd.DataFrame:
    columns = [
        "planning_region",
        "year",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "scenario_balance_GW",
        "planning_pressure_status",
        "reference_margin_level_pct",
    ]
    cube = pd.read_csv(RESULTS / "104_annual_scenario_cube_v1_2.csv", usecols=columns)
    out = cube[
        cube["data_center_scenario_id"].eq("source_embedded")
        & cube["resource_condition"].isin(CORE_RESOURCES)
        & cube["response_condition"].isin(CORE_RESPONSES)
    ].copy()
    keys = ["planning_region", "year", "resource_condition", "response_condition"]
    if out.duplicated(keys).any():
        raise ValueError("Figure 2 source trajectories are not unique")
    out["below_reference"] = out["scenario_balance_GW"] < -TOL
    out = out.sort_values(keys).reset_index(drop=True)
    return out


def build_fig3_decomposition() -> pd.DataFrame:
    decomp = pd.read_csv(RESULTS / "92_annual_balance_driver_decomposition.csv")
    out = decomp[
        decomp["planning_region"].isin(FOCUS_REGIONS)
        & decomp["dc_condition_id"].eq("source_embedded")
        & decomp["resource_condition"].isin(CORE_RESOURCES)
        & decomp["response_condition"].isin(CORE_RESPONSES)
    ].copy()
    return out.sort_values(
        ["planning_region", "resource_condition", "response_condition", "year"]
    ).reset_index(drop=True)


def build_fig3_additional_ranges() -> pd.DataFrame:
    source = pd.read_csv(RESULTS / "112_core_total_system_trajectories.csv")
    partial = pd.read_csv(RESULTS / "113_core_partial_additional_trajectory_summary.csv")
    rows: list[dict[str, object]] = []
    for region in DIAGNOSTIC_REGIONS:
        s = source[source["planning_region"].eq(region)]
        p = partial[partial["planning_region"].eq(region)]
        rows.extend(
            [
                {
                    "planning_region": region,
                    "evidence_condition": "source path",
                    "balance_2035_min_GW": float(s["balance_2035_GW"].min()),
                    "balance_2035_max_GW": float(s["balance_2035_GW"].max()),
                    "interpretation": "source-embedded total-system planning balance",
                },
                {
                    "planning_region": region,
                    "evidence_condition": "50% additional-load diagnostic",
                    "balance_2035_min_GW": float(p["balance_2035_GW"].min()),
                    "balance_2035_max_GW": float(p["balance_2035_GW"].max()),
                    "interpretation": "deterministic core diagnostic; not a probability or best estimate",
                },
            ]
        )
    return pd.DataFrame(rows)


def add_paired_estimands(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["dc_path_pressure_contribution_GW"] = (
        out["source_path_removed_balance_GW"] - out["replacement_balance_GW"]
    )
    out["no_new_required_contribution_GW"] = (
        -out["source_path_removed_balance_GW"]
    ).clip(lower=0.0)
    out["dc_added_required_contribution_GW"] = (
        out["replacement_required_contribution_GW"]
        - out["no_new_required_contribution_GW"]
    )
    return out


def build_fig4() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replacement = add_paired_estimands(
        pd.read_csv(RESULTS / "114_core_source_path_replacement_annual.csv")
    )
    display = replacement[
        replacement["data_center_scenario_id"].eq("lbnl_reference")
        & replacement["post2030_extension_id"].eq("epri_quarter_rate_taper")
        & replacement["footprint_capture_case"].eq("core_active_reported_mw_share")
        & replacement["peak_mapping_case"].eq("reference_coincidence")
        & replacement["allocation_case"].eq("reported_MW_reference_imputation")
    ].copy()
    display["display_scenario_label"] = (
        "LBNL Reference; reported-MW footprint; reference coincidence"
    )

    endpoints = replacement[replacement["year"].eq(2035)].copy()
    envelope = (
        endpoints.groupby(
            [
                "planning_region",
                "resource_condition",
                "response_condition",
                "linked_response_fraction",
            ],
            as_index=False,
        )
        .agg(
            no_new_balance_min_GW=("source_path_removed_balance_GW", "min"),
            no_new_balance_max_GW=("source_path_removed_balance_GW", "max"),
            replacement_balance_min_GW=("replacement_balance_GW", "min"),
            replacement_balance_max_GW=("replacement_balance_GW", "max"),
            dc_pressure_min_GW=("dc_path_pressure_contribution_GW", "min"),
            dc_pressure_max_GW=("dc_path_pressure_contribution_GW", "max"),
            dc_added_requirement_min_GW=("dc_added_required_contribution_GW", "min"),
            dc_added_requirement_max_GW=("dc_added_required_contribution_GW", "max"),
        )
        .sort_values(
            [
                "planning_region",
                "resource_condition",
                "response_condition",
                "linked_response_fraction",
            ]
        )
        .reset_index(drop=True)
    )

    floor = pd.read_csv(RESULTS / "84_source_identified_dc_counterfactuals.csv")
    floor = floor[
        floor["planning_region"].eq("PJM")
        & floor["year"].eq(2035)
        & floor["resource_condition"].isin(CORE_RESOURCES)
        & floor["response_condition"].isin(CORE_RESPONSES)
    ].copy()
    return display, envelope, floor


def build_fig1_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial = pd.read_csv(RESULTS / "108_spatial_allocation_sensitivity_v1_2.csv")
    spatial = spatial[
        [
            "planning_region",
            "source_allocation_case",
            "allocation_case",
            "allocation_share",
        ]
    ].drop_duplicates()
    spatial["is_display_allocation"] = spatial["allocation_case"].eq(
        "reported_MW_reference_imputation"
    )

    regional = pd.read_csv(RESULTS / "103_lbnl2025_regional_load_paths.csv")
    capture = regional[
        ["footprint_capture_case", "capture_share", "footprint_capture_role"]
    ].drop_duplicates()
    capture = capture[
        ~capture["footprint_capture_case"].eq("full_national_assignment_upper_stress")
    ].copy()
    capture["mapping_type"] = "footprint capture"
    capture = capture.rename(
        columns={
            "footprint_capture_case": "mapping_case",
            "capture_share": "mapping_value",
            "footprint_capture_role": "mapping_role",
        }
    )

    peak = pd.read_csv(RESULTS / "102_lbnl2025_peak_mapping_crosswalk.csv")
    peak = (
        peak.groupby(["peak_mapping_case", "peak_mapping_role"], as_index=False)
        .agg(mapping_value=("gamma", "median"), range_min=("gamma", "min"), range_max=("gamma", "max"))
        .rename(
            columns={
                "peak_mapping_case": "mapping_case",
                "peak_mapping_role": "mapping_role",
            }
        )
    )
    peak["mapping_type"] = "peak mapping gamma"
    capture["range_min"] = capture["mapping_value"]
    capture["range_max"] = capture["mapping_value"]
    mapping = pd.concat(
        [
            capture[
                [
                    "mapping_type",
                    "mapping_case",
                    "mapping_value",
                    "range_min",
                    "range_max",
                    "mapping_role",
                ]
            ],
            peak[
                [
                    "mapping_type",
                    "mapping_case",
                    "mapping_value",
                    "range_min",
                    "range_max",
                    "mapping_role",
                ]
            ],
        ],
        ignore_index=True,
    )
    return spatial.sort_values(["planning_region", "allocation_case"]), mapping


def build_qa(
    fig2: pd.DataFrame,
    decomp: pd.DataFrame,
    display: pd.DataFrame,
    spatial: pd.DataFrame,
) -> pd.DataFrame:
    e2_resid = (
        display["source_path_removed_balance_GW"]
        - display["replacement_balance_GW"]
        - display["dc_path_pressure_contribution_GW"]
    ).abs().max()
    e3b_resid = (
        (-display["replacement_balance_GW"]).clip(lower=0.0)
        - (-display["source_path_removed_balance_GW"]).clip(lower=0.0)
        - display["dc_added_required_contribution_GW"]
    ).abs().max()
    tests = [
        ("fig2_has_280_unique_region_year_conditions", len(fig2) == 280),
        (
            "fig2_all_regions_have_40_rows",
            bool((fig2.groupby("planning_region").size() == 40).all()),
        ),
        (
            "fig3_decomposition_identity_closes",
            float(decomp["identity_residual_GW"].abs().max()) <= TOL,
        ),
        ("fig4_display_has_240_paired_rows", len(display) == 240),
        ("fig4_e2_identity_closes", float(e2_resid) <= TOL),
        ("fig4_e3b_identity_closes", float(e3b_resid) <= TOL),
        (
            "fig1_has_eight_allocation_cases_per_region",
            bool((spatial.groupby("planning_region").size() == 8).all()),
        ),
    ]
    return pd.DataFrame(
        {
            "test": [name for name, _ in tests],
            "passed": [passed for _, passed in tests],
        }
    )


def main() -> None:
    fig2 = build_fig2()
    decomp = build_fig3_decomposition()
    additional = build_fig3_additional_ranges()
    display, envelope, floor = build_fig4()
    spatial, mapping = build_fig1_tables()
    qa = build_qa(fig2, decomp, display, spatial)
    if not qa["passed"].all():
        raise RuntimeError(f"Main-figure plotting QA failed:\n{qa}")

    write(fig2, "118_fig2_source_annual_plotting.csv")
    write(decomp, "119_fig3_driver_decomposition_plotting.csv")
    write(additional, "120_fig3_partial_additional_ranges.csv")
    write(display, "121_fig4_paired_contribution_display.csv")
    write(envelope, "122_fig4_paired_contribution_envelope_2035.csv")
    write(floor, "123_fig4_pjm_floor_2035.csv")
    write(spatial, "124_fig1_spatial_allocation_plotting.csv")
    write(mapping, "125_fig1_mapping_crosswalk_plotting.csv")
    write(qa, "126_main_figure_plotting_qa.csv")


if __name__ == "__main__":
    main()
