#!/usr/bin/env python3
"""Validate the frozen core envelope and assign evidence-matched region labels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LBNL_BUILDER_PATH = ROOT / "scripts/build_lbnl2025_scenario_update.py"
BASE_CUBE_PATH = ROOT / "results/104_annual_scenario_cube_v1_2.csv"
SOURCE_PATH_PATH = ROOT / "results/84_source_identified_dc_counterfactuals.csv"

TOTAL_SYSTEM_OUTPUT = ROOT / "results/112_core_total_system_trajectories.csv"
GENERIC_OUTPUT = ROOT / "results/113_core_partial_additional_trajectory_summary.csv"
REPLACEMENT_ANNUAL_OUTPUT = ROOT / "results/114_core_source_path_replacement_annual.csv"
REPLACEMENT_SUMMARY_OUTPUT = ROOT / "results/115_core_source_path_replacement_summary.csv"
LABEL_OUTPUT = ROOT / "results/116_core_envelope_region_labels.csv"
QA_OUTPUT = ROOT / "results/117_core_envelope_acceptance_qa.csv"
NOTE_OUTPUT = ROOT / "docs/23_core_envelope_acceptance.md"

TOL = 1e-8
CORE_RESOURCES = ["anticipated", "prospective"]
CORE_RESPONSES = ["current_only", "published_all"]
LINKED_RESPONSE_FRACTIONS = [0.0, 0.5, 1.0]


def load_builder():
    spec = importlib.util.spec_from_file_location("lbnl2025_builder", LBNL_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import the LBNL 2025 scenario builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[wrote] {path} ({len(frame):,} rows)")


def trajectory_summary(
    annual: pd.DataFrame,
    keys: list[str],
    balance_col: str,
    extra_year_columns: list[str] | None = None,
) -> pd.DataFrame:
    data = annual.copy()
    data["_below"] = data[balance_col] < -TOL
    grouped = data.groupby(keys, dropna=False, sort=True)

    summary = grouped.agg(
        years_below_reference=("_below", "sum"),
        minimum_balance_GW=(balance_col, "min"),
    ).reset_index()

    first = (
        data[data["_below"]]
        .groupby(keys, dropna=False)["year"]
        .min()
        .rename("first_below_reference_year")
        .reset_index()
    )
    summary = summary.merge(first, on=keys, how="left", validate="one_to_one")

    minimum_rows = data.loc[grouped[balance_col].idxmin(), keys + ["year"]].rename(
        columns={"year": "minimum_balance_year"}
    )
    summary = summary.merge(
        minimum_rows, on=keys, how="left", validate="one_to_one"
    )

    extra_year_columns = extra_year_columns or []
    for year in [2030, 2035]:
        selected = data[data["year"].eq(year)][
            keys + [balance_col] + extra_year_columns
        ].copy()
        selected = selected.rename(
            columns={
                balance_col: f"balance_{year}_GW",
                **{column: f"{column}_{year}" for column in extra_year_columns},
            }
        )
        summary = summary.merge(
            selected, on=keys, how="left", validate="one_to_one"
        )

    summary["years_below_reference"] = summary["years_below_reference"].astype(int)
    return summary.sort_values(keys).reset_index(drop=True)


def build_core_regional_paths(builder, config: dict[str, object]) -> pd.DataFrame:
    anchors = pd.read_csv(builder.ANCHOR_PATH)
    peak_mapping = pd.read_csv(builder.PEAK_MAPPING_PATH)
    spatial_weights = pd.read_csv(builder.SPATIAL_WEIGHTS_PATH)

    national = builder.build_national_paths(anchors, config)
    mapping = builder.build_mapping_crosswalk(peak_mapping, config)
    weights = builder.prepare_weights(spatial_weights, config)
    regional = builder.build_regional_paths(
        national,
        mapping,
        weights,
        list(config["spatial_allocation"]["sensitivity_cases"]),
        config,
    )

    envelope = config["scientific_claim_freeze"]["core_envelope"]
    return regional[
        regional["planning_region"].notna()
        & regional["year"].between(2026, 2035)
        & regional["data_center_scenario_id"].isin(envelope["national_paths"])
        & regional["post2030_extension_id"].isin(
            envelope["post_2030_extensions"]
        )
        & regional["footprint_capture_case"].isin(
            envelope["footprint_capture_cases"]
        )
        & regional["peak_mapping_case"].isin(envelope["peak_mapping_cases"])
    ].copy()


def load_core_source_paths() -> pd.DataFrame:
    columns = [
        "planning_region",
        "year",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "reference_margin_level_pct",
        "scenario_balance_GW",
    ]
    base = pd.read_csv(BASE_CUBE_PATH, usecols=columns)
    return base[
        base["data_center_scenario_id"].eq("source_embedded")
        & base["resource_condition"].isin(CORE_RESOURCES)
        & base["response_condition"].isin(CORE_RESPONSES)
    ].copy()


def build_total_system_summary(base: pd.DataFrame) -> pd.DataFrame:
    keys = ["planning_region", "resource_condition", "response_condition"]
    out = trajectory_summary(base, keys, "scenario_balance_GW")
    out["evidence_stream"] = "source-embedded total-system planning path"
    out["interpretation"] = (
        "conditional source-path balance; not a data-center attribution result"
    )
    out["result_status"] = "computed_requires_independent_review"
    return out


def build_generic_partial_additional_summary(
    base: pd.DataFrame, regional: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_fields = [
        "planning_region",
        "year",
        "resource_condition",
        "response_condition",
        "scenario_balance_GW",
        "reference_margin_level_pct",
    ]
    annual = base[base_fields].merge(
        regional, on=["planning_region", "year"], how="inner", validate="many_to_many"
    )
    annual["additional_share"] = 0.5
    annual["additional_data_center_load_GW"] = (
        annual["incremental_data_center_load_vs_2025_GW"]
        * annual["additional_share"]
    )
    multiplier = 1.0 + annual["reference_margin_level_pct"] / 100.0
    annual["core_partial_additional_balance_GW"] = (
        annual["scenario_balance_GW"]
        - multiplier * annual["additional_data_center_load_GW"]
    )
    annual["core_partial_additional_required_contribution_GW"] = (
        -annual["core_partial_additional_balance_GW"]
    ).clip(lower=0.0)

    keys = [
        "planning_region",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "post2030_extension_id",
        "footprint_capture_case",
        "peak_mapping_case",
        "source_allocation_case",
        "allocation_case",
    ]
    summary = trajectory_summary(
        annual,
        keys,
        "core_partial_additional_balance_GW",
        ["incremental_data_center_load_vs_2025_GW"],
    )
    summary["diagnostic_role"] = (
        "50 percent additional-load accounting diagnostic; not a probability or best estimate"
    )
    summary["result_status"] = "computed_requires_independent_review"
    return annual, summary


def build_source_path_replacement(
    base: pd.DataFrame, regional: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identified = pd.read_csv(SOURCE_PATH_PATH)
    identified = identified[
        identified["planning_region"].isin(["MISO", "ERCOT"])
        & identified["resource_condition"].isin(CORE_RESOURCES)
        & identified["response_condition"].isin(CORE_RESPONSES)
    ].copy()

    base_check = base[
        [
            "planning_region",
            "year",
            "resource_condition",
            "response_condition",
            "reference_margin_level_pct",
            "scenario_balance_GW",
        ]
    ].rename(columns={"scenario_balance_GW": "source_balance_check_GW"})
    identified = identified.merge(
        base_check,
        on=["planning_region", "year", "resource_condition", "response_condition"],
        how="inner",
        validate="one_to_one",
    )
    annual = identified.merge(
        regional,
        on=["planning_region", "year"],
        how="inner",
        validate="many_to_many",
    )
    annual = annual.merge(
        pd.DataFrame({"linked_response_fraction": LINKED_RESPONSE_FRACTIONS}),
        how="cross",
    )

    annual["official_gross_path_GW"] = annual["path_upper_GW"].fillna(0.0).clip(lower=0.0)
    annual["available_linked_response_GW"] = (
        annual["linked_response_upper_GW"].fillna(0.0).clip(lower=0.0)
    )
    annual["official_linked_response_GW"] = (
        annual["linked_response_fraction"]
        * annual["available_linked_response_GW"]
    )
    annual["linked_response_rate"] = np.where(
        annual["official_gross_path_GW"] > 0.0,
        np.minimum(
            1.0,
            annual["official_linked_response_GW"]
            / annual["official_gross_path_GW"],
        ),
        0.0,
    )
    annual["candidate_gross_path_GW"] = annual[
        "incremental_data_center_load_vs_2025_GW"
    ].clip(lower=0.0)
    annual["candidate_linked_response_GW"] = (
        annual["candidate_gross_path_GW"] * annual["linked_response_rate"]
    )
    annual["official_net_path_GW"] = (
        annual["official_gross_path_GW"]
        - annual["official_linked_response_GW"]
    )
    annual["candidate_net_path_GW"] = (
        annual["candidate_gross_path_GW"]
        - annual["candidate_linked_response_GW"]
    )
    multiplier = 1.0 + annual["reference_margin_level_pct"] / 100.0
    annual["source_path_removed_balance_GW"] = (
        annual["with_path_balance_GW"]
        + multiplier * annual["official_net_path_GW"]
    )
    annual["replacement_balance_GW"] = (
        annual["source_path_removed_balance_GW"]
        - multiplier * annual["candidate_net_path_GW"]
    )
    annual["replacement_change_from_source_GW"] = (
        annual["replacement_balance_GW"] - annual["with_path_balance_GW"]
    )
    annual["replacement_required_contribution_GW"] = (
        -annual["replacement_balance_GW"]
    ).clip(lower=0.0)
    annual["linked_response_transfer_rule"] = (
        "same feasible linked-response rate applied to official and replacement paths"
    )
    annual["interpretation"] = (
        "conditional source-path replacement; linked-response fractions are structural scenarios, not probabilities"
    )
    annual["response_evidence_role"] = np.where(
        annual["planning_region"].eq("MISO") & annual["linked_response_fraction"].gt(0),
        "outer structural association bound; ordinary response not identified as DC-induced",
        "controlled paired condition; response association remains conditional",
    )

    keys = [
        "planning_region",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "post2030_extension_id",
        "footprint_capture_case",
        "peak_mapping_case",
        "source_allocation_case",
        "allocation_case",
        "linked_response_fraction",
    ]
    summary = trajectory_summary(
        annual,
        keys,
        "replacement_balance_GW",
        ["official_gross_path_GW", "candidate_gross_path_GW"],
    )
    summary["interpretation"] = (
        "conditional source-path replacement; not a causal or probabilistic forecast"
    )
    annual["result_status"] = "computed_requires_independent_review"
    summary["result_status"] = "computed_requires_independent_review"
    return annual, summary


def sign_status(below: int, total: int, all_years_nonnegative: bool = False) -> str:
    if below == total:
        return "invariant_below_reference_in_2035"
    if below == 0:
        if all_years_nonnegative:
            return "invariant_at_or_above_reference_through_2035"
        return "invariant_at_or_above_reference_in_2035"
    return "sign_sensitive_across_core_conditions"


def build_region_labels(
    base: pd.DataFrame,
    total_system: pd.DataFrame,
    generic: pd.DataFrame,
    replacement: pd.DataFrame,
) -> pd.DataFrame:
    labels: list[dict[str, object]] = []
    floor = pd.read_csv(SOURCE_PATH_PATH)
    floor = floor[
        floor["planning_region"].eq("PJM")
        & floor["year"].eq(2035)
        & floor["resource_condition"].isin(CORE_RESOURCES)
        & floor["response_condition"].isin(CORE_RESPONSES)
    ].copy()

    final_labels = {
        "MISO": "2035 total-system pressure across source conditions; data-center replacement is load, resource and linked-response sensitive",
        "PJM": "2035 source-path pressure; classified-subset removal remains below reference as a cross-vintage diagnostic, conditional on inclusion",
        "SPP": "resource-path-sensitive planning pressure; additional data-center load deepens anticipated-path exposure",
        "ERCOT": "resource- and response-sensitive planning balance; no unique deficit classification",
        "NYISO": "positive source balance, but the 2035 sign is sensitive to additional-load magnitude and geography",
        "ISO-NE": "buffered across source conditions and the pre-specified partial-additional core diagnostic",
        "WECC-California": "positive source balance; additional-load stress uses a NERC-listed-component geographic proxy, not identified DC attribution",
    }
    permitted = {
        "MISO": "MISO is below its native reference by 2035 under all four core source conditions; the DC-specific replacement outcome remains conditional.",
        "PJM": "PJM is below its native reference by 2035 under all four core source conditions; the classified-subset removal diagnostic also remains below reference if its inclusion in the later forecast is assumed.",
        "SPP": "SPP pressure appears under anticipated resources and is relieved under the prospective-resource path; added DC load increases the anticipated-path requirement.",
        "ERCOT": "ERCOT's balance changes sign across named resource and response conditions, so results must be reported as a ladder.",
        "NYISO": "NYISO remains above reference on the source path, while additional-load scenarios can reverse the 2035 sign under some allocations.",
        "ISO-NE": "ISO-NE remains at or above its native reference across the tested core source and partial-additional conditions.",
        "WECC-California": "WECC-California remains above reference across the tested source conditions; DC-specific attribution is left unidentified.",
    }
    prohibited = {
        "MISO": "Do not describe one exact shortage year or a source-independent DC-caused deficit.",
        "PJM": "Do not certify inclusion of the older classified subset in the later forecast, attribute the full gap to data centers, or treat the subset as a complete regional path.",
        "SPP": "Do not call SPP unconditionally constrained or report the partial-additional diagnostic as a forecast.",
        "ERCOT": "Do not select one central deficit value or assume all large-load curtailment is data-center response.",
        "NYISO": "Do not call NYISO either robustly constrained or robustly buffered under all DC allocations.",
        "ISO-NE": "Do not generalize beyond the tested footprint, load-shape and allocation envelope.",
        "WECC-California": "Do not call the geographic proxy an exact 2025 NERC boundary, certify project connections, or equate transferred peak shapes with observed noncoincident peaks.",
    }

    for region in sorted(base["planning_region"].unique()):
        source_region = total_system[total_system["planning_region"].eq(region)]
        source_below = int((source_region["balance_2035_GW"] < -TOL).sum())
        source_total = int(len(source_region))
        source_annual = base[base["planning_region"].eq(region)]
        source_all_years_nonnegative = bool(
            (source_annual["scenario_balance_GW"] >= -TOL).all()
        )
        first_years = source_region["first_below_reference_year"].dropna()

        dc_class = "not_identified_for_numeric_replacement"
        dc_status = "not_applicable"
        dc_below = np.nan
        dc_total = np.nan
        dc_min = np.nan
        dc_max = np.nan
        if region in ["MISO", "ERCOT"]:
            dc = replacement[replacement["planning_region"].eq(region)]
            outer_min = float(dc["balance_2035_GW"].min())
            outer_max = float(dc["balance_2035_GW"].max())
            if region == "MISO":
                dc = dc[dc["linked_response_fraction"].eq(0.0)]
            dc_below = int((dc["balance_2035_GW"] < -TOL).sum())
            dc_total = int(len(dc))
            dc_min = float(dc["balance_2035_GW"].min())
            dc_max = float(dc["balance_2035_GW"].max())
            dc_class = "source_path_replacement_with_linked_response_ladder"
            dc_status = sign_status(dc_below, dc_total)
        elif region == "PJM":
            endpoints = pd.concat(
                [
                    floor["no_new_path_balance_lower_GW"],
                    floor["no_new_path_balance_upper_GW"],
                ],
                ignore_index=True,
            )
            dc_below = int((endpoints < -TOL).sum())
            dc_total = int(len(endpoints))
            dc_min = float(endpoints.min())
            dc_max = float(endpoints.max())
            dc_class = "strict_floor_partial_identification"
            dc_status = "cross_vintage_subset_removal_conditional_on_inclusion"
        elif region in ["SPP", "NYISO", "ISO-NE", "WECC-California"]:
            dc = generic[generic["planning_region"].eq(region)]
            dc_below = int((dc["balance_2035_GW"] < -TOL).sum())
            dc_total = int(len(dc))
            dc_min = float(dc["balance_2035_GW"].min())
            dc_max = float(dc["balance_2035_GW"].max())
            dc_class = "partial_additional_load_diagnostic"
            dc_status = sign_status(dc_below, dc_total)

        labels.append(
            {
                "planning_region": region,
                "total_system_evidence_class": "source-embedded total-system planning path",
                "total_system_core_trajectory_count": source_total,
                "total_system_2035_below_count": source_below,
                "total_system_2035_min_GW": float(source_region["balance_2035_GW"].min()),
                "total_system_2035_max_GW": float(source_region["balance_2035_GW"].max()),
                "total_system_first_below_earliest": (
                    int(first_years.min()) if not first_years.empty else np.nan
                ),
                "total_system_first_below_latest": (
                    int(first_years.max()) if not first_years.empty else np.nan
                ),
                "total_system_direction_status": sign_status(
                    source_below, source_total, source_all_years_nonnegative
                ),
                "dc_evidence_class": dc_class,
                "dc_core_trajectory_or_endpoint_count": dc_total,
                "dc_2035_below_count": dc_below,
                "dc_2035_min_GW": dc_min,
                "dc_2035_max_GW": dc_max,
                "dc_direction_status": dc_status,
                "dc_label_response_scope": "no-linked-response controlled comparison" if region == "MISO" else "applicable conditional response range",
                "outer_association_2035_min_GW": outer_min if region in ["MISO", "ERCOT"] else np.nan,
                "outer_association_2035_max_GW": outer_max if region in ["MISO", "ERCOT"] else np.nan,
                "final_region_label": final_labels[region],
                "permitted_claim": permitted[region],
                "prohibited_claim": prohibited[region],
                "result_status": "computed_requires_independent_review",
            }
        )
    return pd.DataFrame(labels)


def build_qa(
    config: dict[str, object],
    regional: pd.DataFrame,
    base: pd.DataFrame,
    total_system: pd.DataFrame,
    generic_annual: pd.DataFrame,
    generic_summary: pd.DataFrame,
    replacement_annual: pd.DataFrame,
    replacement_summary: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    envelope = config["scientific_claim_freeze"]["core_envelope"]
    generic_identity = (
        generic_annual["scenario_balance_GW"]
        - (1.0 + generic_annual["reference_margin_level_pct"] / 100.0)
        * generic_annual["additional_data_center_load_GW"]
        - generic_annual["core_partial_additional_balance_GW"]
    ).abs().max()
    replacement_identity = (
        replacement_annual["with_path_balance_GW"]
        + (1.0 + replacement_annual["reference_margin_level_pct"] / 100.0)
        * (
            replacement_annual["official_net_path_GW"]
            - replacement_annual["candidate_net_path_GW"]
        )
        - replacement_annual["replacement_balance_GW"]
    ).abs().max()
    source_residual = (
        replacement_annual["with_path_balance_GW"]
        - replacement_annual["source_balance_check_GW"]
    ).abs().max()
    alpha_zero = replacement_annual[
        replacement_annual["linked_response_fraction"].eq(0.0)
    ]
    alpha_one = replacement_annual[
        replacement_annual["linked_response_fraction"].eq(1.0)
    ]
    removal_upper_residual = (
        alpha_zero["source_path_removed_balance_GW"]
        - alpha_zero["no_new_path_balance_upper_GW"]
    ).abs().max()
    removal_lower_residual = (
        alpha_one["source_path_removed_balance_GW"]
        - alpha_one["no_new_path_balance_lower_GW"]
    ).abs().max()

    checks = [
        ("core regional path row count", len(regional) == 45360, f"rows={len(regional)}"),
        ("total-system trajectory count", len(total_system) == 28, f"rows={len(total_system)}"),
        ("generic core trajectory count", len(generic_summary) == 18144, f"rows={len(generic_summary)}"),
        ("replacement core annual count", len(replacement_annual) == 155520, f"rows={len(replacement_annual)}"),
        ("replacement trajectory count", len(replacement_summary) == 15552, f"rows={len(replacement_summary)}"),
        ("seven region labels", len(labels) == 7, f"rows={len(labels)}"),
        (
            "all eight fixed-total allocations retained",
            replacement_annual["source_allocation_case"].nunique() == 8,
            f"count={replacement_annual['source_allocation_case'].nunique()}",
        ),
        (
            "upper footprint stress excluded from core",
            "full_national_assignment_upper_stress"
            not in set(regional["footprint_capture_case"]),
            "upper stress absent",
        ),
        (
            "core paths match freeze",
            set(regional["data_center_scenario_id"]) == set(envelope["national_paths"]),
            str(sorted(regional["data_center_scenario_id"].unique())),
        ),
        (
            "core extensions match freeze",
            set(regional["post2030_extension_id"])
            == set(envelope["post_2030_extensions"]),
            str(sorted(regional["post2030_extension_id"].unique())),
        ),
        (
            "core peak mappings match freeze",
            set(regional["peak_mapping_case"]) == set(envelope["peak_mapping_cases"]),
            str(sorted(regional["peak_mapping_case"].unique())),
        ),
        (
            "generic half-additional identity closes",
            generic_identity < TOL,
            f"max_abs_residual={generic_identity:.3e}",
        ),
        (
            "source balance matches identified-path table",
            source_residual < TOL,
            f"max_abs_residual={source_residual:.3e}",
        ),
        (
            "replacement identity closes",
            replacement_identity < TOL,
            f"max_abs_residual={replacement_identity:.3e}",
        ),
        (
            "zero-linked removal matches identified upper endpoint",
            removal_upper_residual < TOL,
            f"max_abs_residual={removal_upper_residual:.3e}",
        ),
        (
            "fully-linked removal matches identified lower endpoint",
            removal_lower_residual < TOL,
            f"max_abs_residual={removal_lower_residual:.3e}",
        ),
        (
            "candidate linked response never exceeds candidate path",
            (
                replacement_annual["candidate_linked_response_GW"]
                <= replacement_annual["candidate_gross_path_GW"] + TOL
            ).all(),
            "bounded structural response transfer",
        ),
        (
            "replacement restricted to MISO and ERCOT",
            set(replacement_annual["planning_region"]) == {"MISO", "ERCOT"},
            str(sorted(replacement_annual["planning_region"].unique())),
        ),
        (
            "linked-response ladder complete",
            set(replacement_annual["linked_response_fraction"])
            == set(LINKED_RESPONSE_FRACTIONS),
            str(sorted(replacement_annual["linked_response_fraction"].unique())),
        ),
        (
            "MISO and PJM total-system 2035 direction invariant below",
            labels.set_index("planning_region")
            .loc[["MISO", "PJM"], "total_system_direction_status"]
            .eq("invariant_below_reference_in_2035")
            .all(),
            "expected evidence-matched labels",
        ),
        (
            "ISO-NE core partial-additional direction invariant above",
            labels.set_index("planning_region")
            .loc["ISO-NE", "dc_direction_status"]
            == "invariant_at_or_above_reference_in_2035",
            str(labels.set_index("planning_region").loc["ISO-NE", "dc_direction_status"]),
        ),
        (
            "ERCOT and SPP total-system signs remain conditional",
            labels.set_index("planning_region")
            .loc[["ERCOT", "SPP"], "total_system_direction_status"]
            .eq("sign_sensitive_across_core_conditions")
            .all(),
            "expected mixed signs",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in checks
        ]
    )


def write_note(labels: pd.DataFrame, qa: pd.DataFrame) -> None:
    lines = [
        "# Core-envelope result acceptance",
        "",
        "Date evaluated: 2026-09-30",
        "",
        "This audit applies the frozen claim hierarchy to the pre-specified core",
        "envelope. Upper stresses do not vote on robustness. Total-system pressure,",
        "source-path replacement and generic partial-additional diagnostics remain",
        "separate evidence streams.",
        "",
        "## Region labels",
        "",
        "| Region | Total-system status | DC evidence status | Accepted label |",
        "|---|---|---|---|",
    ]
    for row in labels.itertuples(index=False):
        lines.append(
            f"| {row.planning_region} | {row.total_system_direction_status} | "
            f"{row.dc_direction_status} | {row.final_region_label} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MISO and PJM are below their native references in 2035 under all four",
            "  core source resource-response conditions. This is a total-system result,",
            "  not a seven-region DC-causal claim.",
            "- MISO replacement outcomes vary with the load path, resource path and",
            "  linked-response rule. PJM subset removal is a cross-vintage diagnostic conditional on inclusion.",
            "- ERCOT and SPP change sign across named resource or response conditions.",
            "  They must remain scenario ladders rather than single deficit estimates.",
            "- NYISO is positive on the source paths but can cross below the reference",
            "  under some partial-additional load allocations. ISO-NE remains positive",
            "  across the tested core source and partial-additional conditions.",
            "- WECC-California retains its source balance and adds a geographic-proxy",
            "  additional-load diagnostic, not DC attribution or connection certification.",
            "",
            "## Acceptance status",
            "",
            f"All {len(qa)} blocking checks passed: **{bool(qa['passed'].all())}**.",
            "Numerical outputs retain `computed_requires_independent_review` until the",
            "submission-stage source-page and independent reproduction sign-off.",
        ]
    )
    NOTE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {NOTE_OUTPUT}")


def main() -> None:
    builder = load_builder()
    config = builder.load_config()
    regional = build_core_regional_paths(builder, config)
    base = load_core_source_paths()

    total_system = build_total_system_summary(base)
    generic_annual, generic_summary = build_generic_partial_additional_summary(
        base, regional
    )
    replacement_annual, replacement_summary = build_source_path_replacement(
        base, regional
    )
    labels = build_region_labels(
        base, total_system, generic_summary, replacement_summary
    )
    qa = build_qa(
        config,
        regional,
        base,
        total_system,
        generic_annual,
        generic_summary,
        replacement_annual,
        replacement_summary,
        labels,
    )
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"]].to_dict("records")
        raise RuntimeError(f"Core-envelope acceptance failed: {failed}")

    replacement_columns = [
        "planning_region",
        "year",
        "resource_condition",
        "response_condition",
        "data_center_scenario_id",
        "post2030_extension_id",
        "footprint_capture_case",
        "peak_mapping_case",
        "source_allocation_case",
        "allocation_case",
        "linked_response_fraction",
        "reference_margin_level_pct",
        "with_path_balance_GW",
        "source_balance_check_GW",
        "official_gross_path_GW",
        "available_linked_response_GW",
        "official_linked_response_GW",
        "official_net_path_GW",
        "no_new_path_balance_lower_GW",
        "no_new_path_balance_upper_GW",
        "candidate_gross_path_GW",
        "candidate_linked_response_GW",
        "candidate_net_path_GW",
        "source_path_removed_balance_GW",
        "replacement_balance_GW",
        "replacement_change_from_source_GW",
        "replacement_required_contribution_GW",
        "linked_response_transfer_rule",
        "interpretation",
        "result_status",
    ]
    write(total_system, TOTAL_SYSTEM_OUTPUT)
    write(generic_summary, GENERIC_OUTPUT)
    write(replacement_annual[replacement_columns], REPLACEMENT_ANNUAL_OUTPUT)
    write(replacement_summary, REPLACEMENT_SUMMARY_OUTPUT)
    write(labels, LABEL_OUTPUT)
    write(qa, QA_OUTPUT)
    write_note(labels, qa)


if __name__ == "__main__":
    main()
