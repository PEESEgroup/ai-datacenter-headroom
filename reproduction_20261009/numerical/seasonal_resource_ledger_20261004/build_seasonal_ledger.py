#!/usr/bin/env python3
"""Build an independent native summer ledger and component-timing comparison."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
B4 = WORKSPACE / "natcomm_major_revision_batch4_20260916"
B8 = WORKSPACE / "natcomm_external_validation_batch8_20260917"
B10 = WORKSPACE / "natcomm_core_method_audit_batch10_20260917"
SOURCE = B4 / "sources/nerc_ltra_2025.pdf"
PAGE_MAP = {"MISO": 42, "SPP": 59, "ISO-NE": 68, "NYISO": 74, "PJM": 90, "ERCOT": 128, "WECC-California": 147}
SOURCE_URL = "https://www.nerc.com/globalassets/our-work/assessments/nerc_ltra_2025.pdf"
spec = importlib.util.spec_from_file_location("annual_accounting", ROOT / "scripts/build_annual_scenario_cube.py")
annual = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(annual)

FIELDS = {
    "gross_peak_demand_MW": r"^Total Internal Demand\s+(.+)$",
    "published_response_MW": r"^Demand Response\s+(.+)$",
    "published_net_demand_MW": r"^Net Internal Demand\s+(.+)$",
    "tier1_contribution_MW": r"^Additions: Tier 1\s+(.+)$",
    "tier2_contribution_MW": r"^Additions: Tier 2\s+(.+)$",
    "tier3_contribution_MW": r"^Additions: Tier 3\s+(.+)$",
    "reported_transfer_MW": r"^(?:Net Firm Capacity Transfers|Net Capacity Transfers \(WECC Model\))\s+(.+)$",
    "existing_and_transfer_MW": r"^Existing-Certain and (?:Net Firm Transfers|Net Transfers)\s+(.+)$",
    "published_ARM_pct": r"^Anticipated Reserve Margin \(%\)\s+(.+)$",
    "published_PRM_pct": r"^Prospective Reserve Margin \(%\)\s+(.+)$",
    "reference_margin_level_pct": r"^Reference Margin Level \(%\)\s+(.+)$",
}


def digest(path: Path) -> str:
    obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            obj.update(chunk)
    return obj.hexdigest()


def numbers(text: str) -> list[float]:
    return [float(token.replace(",", "")) for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def extract_native(reader: PdfReader, config: dict) -> pd.DataFrame:
    rows = []
    for region, page in PAGE_MAP.items():
        lines = [re.sub(r"\s+", " ", line.strip()) for line in reader.pages[page - 1].extract_text().splitlines()]
        parsed = {}
        for field, pattern in FIELDS.items():
            matches = [numbers(match.group(1)) for line in lines if (match := re.match(pattern, line))]
            valid = [values for values in matches if len(values) == len(config["years"])]
            if len(valid) != 1:
                raise ValueError(f"{region} p.{page}: {field} has {len(valid)} matching rows")
            parsed[field] = valid[0]
        for index, year in enumerate(config["years"]):
            rows.append({"planning_region": region, "year": year, "source_pdf_page": page,
                         **{field: values[index] for field, values in parsed.items()}})
    out = pd.DataFrame(rows).sort_values(["planning_region", "year"]).reset_index(drop=True)
    out["existing_certain_native_MW"] = out.existing_and_transfer_MW - out.reported_transfer_MW
    out["anticipated_native_MW"] = out.existing_certain_native_MW + out.reported_transfer_MW + out.tier1_contribution_MW
    out["calculated_net_demand_MW"] = out.gross_peak_demand_MW - out.published_response_MW
    out["net_display_rounding_difference_MW"] = out.calculated_net_demand_MW - out.published_net_demand_MW
    out["calculated_ARM_pct"] = (out.anticipated_native_MW / out.calculated_net_demand_MW - 1) * 100
    out["ARM_display_difference_pp"] = out.calculated_ARM_pct - out.published_ARM_pct
    out["prospective_native_MW"] = out.published_net_demand_MW * (1 + out.published_PRM_pct / 100)
    out["prospective_other_net_contribution_MW"] = out.prospective_native_MW - out.anticipated_native_MW
    out["prospective_beyond_reported_tier2_MW"] = out.prospective_other_net_contribution_MW - out.tier2_contribution_MW
    out["season"] = "summer_June_September"
    out["peak_aggregation"] = np.where(out.planning_region.isin(["SPP", "WECC-California"]), "noncoincident", "coincident")
    out["capacity_basis"] = "source_native_summer_capacity_contribution_not_universal_UCAP"
    out["retirement_rule"] = "confirmed_retirements_embedded_do_not_subtract_twice"
    out["generic_extra_accreditation_factor"] = 1.0
    return out


def extract_ercot_response(reader: PdfReader) -> pd.DataFrame:
    text = reader.pages[130].extract_text()
    pattern = r"(202[5-9]|203[0-5]) \(S\)\s+([\d,.]+)\s+([\d,.]+)"
    rows = [{"year": int(year), "total_large_load_MW": numbers(load)[0], "provisional_large_load_response_MW": numbers(response)[0]}
            for year, load, response in re.findall(pattern, text)]
    out = pd.DataFrame(rows)
    if len(out) != 11 or out.year.duplicated().any():
        raise ValueError("ERCOT summer curtailment table was not uniquely extracted")
    out["source_pdf_page"] = 131
    return out


def add_response(native: pd.DataFrame, ercot: pd.DataFrame) -> pd.DataFrame:
    out = native.merge(ercot[["year", "provisional_large_load_response_MW"]], on="year", validate="many_to_one")
    out.loc[out.planning_region != "ERCOT", "provisional_large_load_response_MW"] = 0
    out["ordinary_response_MW"] = out.published_response_MW - out.provisional_large_load_response_MW
    out["first_forecast_ordinary_response_MW"] = out.groupby("planning_region").ordinary_response_MW.transform("first")
    out["current_cap_response_MW"] = np.minimum(out.ordinary_response_MW, out.first_forecast_ordinary_response_MW)
    out["first_year_response_decline_MW"] = np.maximum(0, out.first_forecast_ordinary_response_MW - out.ordinary_response_MW)
    out["future_or_provisional_response_MW"] = out.published_response_MW - out.current_cap_response_MW
    return out


def delayed_tier1(values: np.ndarray, lag: int) -> np.ndarray:
    if lag < 1:
        raise ValueError("Lag must be positive")
    shifted = np.r_[np.zeros(min(lag, len(values))), values[:-lag]] if lag < len(values) else np.zeros(len(values))
    # Retain a source-year ceiling if the aggregate category loses contribution.
    return np.minimum(shifted, values)


def condition_panel(native: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for _, group in native.groupby("planning_region", sort=True):
        group = group.sort_values("year").copy()
        for resource in ["anticipated", "prospective", *[f"anticipated_tier1_arrival_delayed_{lag}y" for lag in (1, 2, 3)]]:
            part = group.copy()
            if resource == "anticipated":
                supply = part.anticipated_native_MW.to_numpy()
                part["counted_tier1_MW"] = part.tier1_contribution_MW
            elif resource == "prospective":
                supply = part.prospective_native_MW.to_numpy()
                part["counted_tier1_MW"] = part.tier1_contribution_MW
            else:
                lag = int(resource[-2])
                part["counted_tier1_MW"] = delayed_tier1(part.tier1_contribution_MW.to_numpy(), lag)
                supply = (part.existing_certain_native_MW + part.reported_transfer_MW + part.counted_tier1_MW).to_numpy()
            part["capacity_equivalent_resources_MW"] = supply
            part["resource_condition"] = resource
            part["resource_condition_type"] = "source_condition" if resource in ["anticipated", "prospective"] else "component_timing_stress"
            part["stock_component_unchanged"] = True
            part["transfer_component_unchanged"] = True
            for response, column in [("none", None), ("current_cap", "current_cap_response_MW"), ("published_all", "published_response_MW")]:
                out = part.copy()
                out["response_condition"] = response
                out["credited_DR_MW"] = 0 if column is None else out[column]
                out["net_peak_demand_MW"] = out.gross_peak_demand_MW - out.credited_DR_MW
                tables.append(annual.recompute_balance(out))
    return pd.concat(tables, ignore_index=True).sort_values(["planning_region", "resource_condition", "response_condition", "year"]).reset_index(drop=True)


def compare_conditions(panel: pd.DataFrame) -> pd.DataFrame:
    legacy_matrix = pd.read_csv(B10 / "results/02_condition_matrix_region_year.csv")
    legacy = annual.build_resource_response_panel(legacy_matrix, annual.load_config())
    legacy["resource_condition"] = legacy.resource_path
    legacy["response_condition"] = legacy.dr_treatment
    new = panel.copy()
    new["legacy_resource_condition"] = new.resource_condition.str.replace("tier1_arrival", "positive_net_additions", regex=False)
    new["legacy_response_condition"] = new.response_condition.replace({"current_cap": "current_only"})
    old = legacy[["planning_region", "year", "resource_condition", "response_condition", "capacity_equivalent_resources_MW", "margin_GW"]]
    old = old.rename(columns={"resource_condition": "legacy_resource_condition", "response_condition": "legacy_response_condition",
                              "capacity_equivalent_resources_MW": "old_resources_MW", "margin_GW": "old_balance_GW"})
    out = new.merge(old, on=["planning_region", "year", "legacy_resource_condition", "legacy_response_condition"], validate="one_to_one")
    out["resource_change_MW"] = out.capacity_equivalent_resources_MW - out.old_resources_MW
    out["balance_change_GW"] = out.margin_GW - out.old_balance_GW
    out["below_reference_sign_changed"] = out.margin_GW.lt(-1e-8) != out.old_balance_GW.lt(-1e-8)
    out["change_type"] = np.where(out.resource_condition_type == "source_condition", "no_substantive_input_change", "planned_contribution_timing_redesign_not_baseline_forecast_correction")
    return out


def annual_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    keys = ["planning_region", "resource_condition", "response_condition"]
    fields = ["existing_certain_native_MW", "reported_transfer_MW", "counted_tier1_MW", "capacity_equivalent_resources_MW",
              "gross_peak_demand_MW", "credited_DR_MW", "reference_margin_level_pct", "margin_GW"]
    for field in fields:
        out[f"base_{field}"] = out.groupby(keys)[field].transform("first")
    factor0 = 1 + out.base_reference_margin_level_pct / 100
    out["existing_native_change_GW"] = (out.existing_certain_native_MW - out.base_existing_certain_native_MW) / 1000
    out["transfer_change_GW"] = (out.reported_transfer_MW - out.base_reported_transfer_MW) / 1000
    out["planned_contribution_change_GW"] = (out.counted_tier1_MW - out.base_counted_tier1_MW) / 1000
    resource_change = (out.capacity_equivalent_resources_MW - out.base_capacity_equivalent_resources_MW) / 1000
    out["other_prospective_net_change_GW"] = resource_change - out.existing_native_change_GW - out.transfer_change_GW - out.planned_contribution_change_GW
    out["gross_demand_term_GW"] = -(out.gross_peak_demand_MW - out.base_gross_peak_demand_MW) * factor0 / 1000
    out["response_term_GW"] = (out.credited_DR_MW - out.base_credited_DR_MW) * factor0 / 1000
    out["reference_requirement_term_GW"] = -out.net_peak_demand_MW * (out.reference_margin_level_pct - out.base_reference_margin_level_pct) / 100000
    terms = ["existing_native_change_GW", "transfer_change_GW", "planned_contribution_change_GW", "other_prospective_net_change_GW",
             "gross_demand_term_GW", "response_term_GW", "reference_requirement_term_GW"]
    out["reconstructed_balance_change_GW"] = out[terms].sum(axis=1)
    out["actual_balance_change_GW"] = out.margin_GW - out.base_margin_GW
    out["residual_GW"] = out.reconstructed_balance_change_GW - out.actual_balance_change_GW
    out["existing_change_is_retirement_only"] = False
    return out[[*keys, "year", *terms, "reconstructed_balance_change_GW", "actual_balance_change_GW", "residual_GW", "existing_change_is_retirement_only"]]


def update_cube(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(ROOT / "results/104_annual_scenario_cube_v1_2.csv")
    keys = ["planning_region", "year", "resource_condition", "response_condition"]
    lookup = comparison[["planning_region", "year", "legacy_resource_condition", "legacy_response_condition", "resource_condition", "response_condition", "resource_change_MW"]]
    lookup = lookup.rename(columns={"resource_condition": "new_resource_condition", "response_condition": "new_response_condition",
                                    "legacy_resource_condition": "resource_condition", "legacy_response_condition": "response_condition"})
    out = old.merge(lookup, on=keys, validate="many_to_one", how="left")
    if out.resource_change_MW.isna().any():
        raise ValueError("Incomplete scenario-to-native-ledger join")
    out["legacy_resource_condition"] = out.resource_condition
    out["legacy_response_condition"] = out.response_condition
    out["resource_condition"] = out.pop("new_resource_condition")
    out["response_condition"] = out.pop("new_response_condition")
    out.loc[out.resource_condition.str.contains("tier1_arrival"), "resource_condition_type"] = "component_timing_stress"
    out["legacy_scenario_balance_GW"] = out.scenario_balance_GW
    out["capacity_equivalent_resources_MW"] += out.resource_change_MW
    out["base_balance_GW"] = (out.capacity_equivalent_resources_MW - out.base_net_peak_demand_MW * (1 + out.reference_margin_level_pct / 100)) / 1000
    out["scenario_balance_GW"] = (out.capacity_equivalent_resources_MW - out.scenario_reference_requirement_MW) / 1000
    out["scenario_required_contribution_GW"] = np.maximum(0, -out.scenario_balance_GW)
    out["planning_pressure_status"] = np.where(out.scenario_balance_GW >= -1e-8, "at_or_above_reference", "below_reference")
    out.loc[~out.scenario_applicable, "planning_pressure_status"] = "not_applicable"
    out["season"] = "summer_June_September"
    out["input_revision_id"] = "seasonal_native_ledger_20261004"
    out["claim_role"] = "additional_load_diagnostic_not_complete_path_replacement"
    out["predictive_skill_newly_established"] = False
    old_sign = out.legacy_scenario_balance_GW < -1e-8
    new_sign = out.scenario_balance_GW < -1e-8
    out["scenario_sign_changed"] = (old_sign != new_sign) & out.scenario_applicable
    summary = out.groupby(["planning_region", "resource_condition", "response_condition", "year"], as_index=False).agg(
        scenario_rows=("scenario_balance_GW", "size"),
        applicable_rows=("scenario_applicable", "sum"),
        sign_changed_rows=("scenario_sign_changed", "sum"),
        maximum_resource_change_MW=("resource_change_MW", "max"),
        minimum_resource_change_MW=("resource_change_MW", "min"))
    return out, summary


def historical_stock_audit() -> pd.DataFrame:
    sra = pd.read_csv(B8 / "results/02_sra_outcome_panel.csv")
    out = sra.copy()
    out["operating_and_transfer_native_MW"] = out.existing_certain_capacity_MW + out.net_firm_capacity_transfers_MW
    out["reconstructed_anticipated_MW"] = out.operating_and_transfer_native_MW + out.tier1_planned_capacity_MW
    out["resource_identity_difference_MW"] = out.reconstructed_anticipated_MW - out.anticipated_resources_MW
    out["planned_fraction_in_anticipated_anchor"] = out.tier1_planned_capacity_MW / out.anticipated_resources_MW
    out["anticipated_anchor_is_operating_stock"] = False
    out["category_change_is_project_delivery_truth"] = False
    return out


def annual_condition_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in comparison.groupby(["planning_region", "resource_condition", "response_condition"]):
        group = group.sort_values("year")
        row = dict(zip(["planning_region", "resource_condition", "response_condition"], keys))
        for label, column in [("old", "old_balance_GW"), ("new", "margin_GW")]:
            balance = group[column]
            negative = balance.lt(-1e-8)
            row[f"{label}_first_below_reference_year"] = int(group.loc[negative, "year"].min()) if negative.any() else pd.NA
            row[f"{label}_years_below_reference"] = int(negative.sum())
            row[f"{label}_maximum_required_contribution_GW"] = float(np.maximum(0, -balance).max())
            row[f"{label}_required_contribution_year_sum_GW_year"] = float(np.maximum(0, -balance).sum())
            row[f"{label}_balance_2030_GW"] = float(group.loc[group.year.eq(2030), column].iloc[0])
            row[f"{label}_balance_2035_GW"] = float(group.loc[group.year.eq(2035), column].iloc[0])
        row["first_below_reference_changed"] = row["old_first_below_reference_year"] != row["new_first_below_reference_year"] if pd.notna(row["old_first_below_reference_year"]) and pd.notna(row["new_first_below_reference_year"]) else pd.notna(row["old_first_below_reference_year"]) != pd.notna(row["new_first_below_reference_year"])
        row["additional_required_contribution_year_sum_GW_year"] = row["new_required_contribution_year_sum_GW_year"] - row["old_required_contribution_year_sum_GW_year"]
        row["interpretation"] = "conditional annual capacity-contribution sum, not unserved energy or event prediction"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_replacement_panel(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(ROOT / "results/114_core_source_path_replacement_annual.csv")
    tables = []
    balance_columns = ["with_path_balance_GW", "source_balance_check_GW", "no_new_path_balance_lower_GW",
                       "no_new_path_balance_upper_GW", "source_path_removed_balance_GW", "replacement_balance_GW"]
    for resource in ["anticipated", "prospective", *[f"anticipated_tier1_arrival_delayed_{lag}y" for lag in (1, 2, 3)]]:
        source_resource = "prospective" if resource == "prospective" else "anticipated"
        part = old[old.resource_condition.eq(source_resource)].copy()
        lookup = comparison[comparison.resource_condition.eq(resource) & comparison.response_condition.ne("none")][
            ["planning_region", "year", "legacy_response_condition", "margin_GW", "old_balance_GW"]].copy()
        # Use the undelayed source background for both members of each old pair.
        lookup = lookup.merge(comparison[comparison.resource_condition.eq(source_resource) & comparison.response_condition.ne("none")][
            ["planning_region", "year", "legacy_response_condition", "old_balance_GW"]].rename(columns={"old_balance_GW": "paired_source_background_GW"}),
            on=["planning_region", "year", "legacy_response_condition"], validate="one_to_one")
        lookup["paired_resource_shift_GW"] = lookup.margin_GW - lookup.paired_source_background_GW
        part = part.merge(lookup[["planning_region", "year", "legacy_response_condition", "paired_resource_shift_GW"]],
            left_on=["planning_region", "year", "response_condition"],
            right_on=["planning_region", "year", "legacy_response_condition"], validate="many_to_one", how="left")
        if part.paired_resource_shift_GW.isna().any():
            raise ValueError("Incomplete paired DC resource-background join")
        old_pressure = part.source_path_removed_balance_GW - part.replacement_balance_GW
        part["undelayed_added_required_contribution_GW"] = np.maximum(0, -part.replacement_balance_GW) - np.maximum(0, -part.source_path_removed_balance_GW)
        for column in balance_columns:
            part[column] += part.paired_resource_shift_GW
        part["replacement_required_contribution_GW"] = np.maximum(0, -part.replacement_balance_GW)
        part["dc_path_pressure_contribution_GW"] = part.source_path_removed_balance_GW - part.replacement_balance_GW
        part["dc_added_required_contribution_GW"] = np.maximum(0, -part.replacement_balance_GW) - np.maximum(0, -part.source_path_removed_balance_GW)
        part["dc_pressure_change_from_undelayed_GW"] = part.dc_path_pressure_contribution_GW - old_pressure
        part["dc_added_requirement_change_from_undelayed_GW"] = part.dc_added_required_contribution_GW - part.undelayed_added_required_contribution_GW
        part["legacy_resource_condition"] = part.resource_condition
        part["resource_condition"] = resource
        part["response_condition"] = part.response_condition.replace({"current_only": "current_cap"})
        part["season"] = "summer_June_September"
        part["claim_role"] = "paired_DC_path_replacement_not_generic_additional_load"
        part["resource_condition_type"] = "source_condition" if resource in ["anticipated", "prospective"] else "component_timing_stress"
        tables.append(part.drop(columns="legacy_response_condition"))
    out = pd.concat(tables, ignore_index=True)
    summary = out.groupby(["planning_region", "year", "resource_condition", "response_condition"], as_index=False).agg(
        paired_cases=("replacement_balance_GW", "size"),
        replacement_balance_min_GW=("replacement_balance_GW", "min"),
        replacement_balance_max_GW=("replacement_balance_GW", "max"),
        dc_pressure_min_GW=("dc_path_pressure_contribution_GW", "min"),
        dc_pressure_max_GW=("dc_path_pressure_contribution_GW", "max"),
        dc_added_requirement_min_GW=("dc_added_required_contribution_GW", "min"),
        dc_added_requirement_max_GW=("dc_added_required_contribution_GW", "max"),
        maximum_added_requirement_change_GW=("dc_added_requirement_change_from_undelayed_GW", "max"))
    return out, summary


def evidence_registry(reader: PdfReader) -> pd.DataFrame:
    evidence = [
        ("common_season", 167, "June", "summer scope, not all-season adequacy"),
        ("common_response", 167, "reduced by the amount", "explicit response deducted once"),
        ("common_stock_and_retirements", 168, "Less confirmed retirements", "no second blanket retirement subtraction"),
        ("common_prospective", 168, "Less unconfirmed retirements", "not simply anticipated plus Tier 2"),
        ("MISO_stock", 47, "confirmed seasonal accredited capacity", "already-accredited source path"),
        ("MISO_existing_delay", 47, "three", "additional lag is sensitivity beyond source ramp"),
        ("PJM_reference", 92, "Installed Reserve Requirement", "retain native reference/capacity pairing"),
        ("NYISO_response", 78, "derated to 989 MW", "do not derate the credited response twice"),
        ("ERCOT_accreditation", 130, "average ELCC", "native contribution, not a second generic factor"),
        ("ERCOT_provisional_response", 131, "Controllable and Dispatchable Demand Response", "explicit provisional category"),
        ("WECC_distinct_footprint", 147, "new assessment area", "not CAISO or former CA/MX"),
        ("WECC_reference", 149, "not used", "do not substitute CPUC reference"),
    ]
    rows = []
    for key, page, phrase, decision in evidence:
        text = re.sub(r"\s+", " ", reader.pages[page-1].extract_text())
        rows.append({"check": key, "source_pdf_page": page, "matched": phrase.lower() in text.lower(),
                     "decision": decision, "source_url": SOURCE_URL})
    return pd.DataFrame(rows)


def retirement_differential(reader: PdfReader) -> pd.DataFrame:
    text = reader.pages[128].extract_text()
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    original = next(numbers(line.split("Total MW ", 1)[1]) for line in lines if line.startswith("Total MW "))
    alternative = next(numbers(line.split("Total MW* ", 1)[1]) for line in lines if line.startswith("Total MW* "))
    if len(original) != 5 or len(alternative) != 5:
        raise ValueError("ERCOT retirement alternative parse failed")
    return pd.DataFrame({"planning_region": "ERCOT", "year": range(2026, 2031), "generation_table_total_MW": original,
                         "generation_table_additional_retirement_MW": alternative,
                         "additional_retirement_differential_MW": np.array(original) - np.array(alternative),
                         "source_pdf_page": 129, "applied_as_extra_headline_deduction": False,
                         "reason": "different generation-table baseline; no unit-level baseline exclusion, so no automatic subtraction"})


def write_report(native: pd.DataFrame, comparison: pd.DataFrame, cube_summary: pd.DataFrame, hist: pd.DataFrame, qa: pd.DataFrame,
                 trajectories: pd.DataFrame, paired_summary: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    core = comparison[comparison.resource_condition.isin(["anticipated", "prospective"]) & comparison.response_condition.ne("none")]
    endpoints = comparison[(comparison.year == 2035) & (comparison.response_condition == "published_all")]
    rows = ["季节性资源与响应口径修正：2026-10-04", "",
        "本轮不将失败的统计校准升级为主预测，也不为保持旧结果而更改参数。旧结果、论文和 SI 未覆盖。", "",
        "一、核查与实质处理", "",
        "重新从 NERC 原始表提取七区 2026-2035 年共 70 个 region-year；拆分 existing-certain、报告的转移及 Tier 1。",
        "七区保持来源原生容量和对应参考要求，不统一贴 UCAP 标签，不再次乘 ELCC，不再扣一次来源已包含的确认退役。",
        "SPP 与 WECC-California 的需求为非同时峰值聚合，其余五区为同时峰值；不能将七区统一称作 coincident peak。",
        "明确这是夏季 June-September 的年度路径，不声称已构造完整冬夏双季或风险小时可靠性模型。",
        "current_only 的真实规则是首年普通响应上限，且保留来源后续下降；新输出名称 current_cap，数值不变。ISO-NE 的首年 623 MW 在后续降为 544 MW，不应在文字中说始终冻结为 623 MW。",
        "新时序压力只延迟 Tier 1 贡献的到达，包括第一预测年的计划新增；现有资源变化与转移继续按来源日程。旧净增长延迟规则保留作对照。",
        "MISO 来源本身已含 GIA 延迟，因此新 lag 是额外压力情景，不是再次估计其最可能完成率。固定延迟后按原来源贡献计入，并不重估后年 ELCC 或逐项目退役。", "",
        "二、对主结果的影响", "",
        f"四条 source-based resource-response 条件共 {len(core)} 个年度值：最大变化 {core.balance_change_GW.abs().max():.10f} GW；符号变化 {int(core.below_reference_sign_changed.sum())} 个。",
        "主要来源情景没有发现需要重新打折或统一扣退役的数值错误，所以不人为改变这些余额。",
        "改变的是时序压力情景的构造。对照表保留每个区域、年份、资源/响应条件的旧值、新值和原因。", "",
        "2035 published-response 条件下的旧/新余额（原生 GW，不可跨区物理求和）："]
    for region, group in endpoints.groupby("planning_region"):
        rows.append(f"  {region}:")
        for _, row in group.iterrows():
            rows.append(f"    {row.resource_condition}: {row.old_balance_GW:.3f} -> {row.margin_GW:.3f} (变化 {row.balance_change_GW:+.3f})")
    rows += ["", f"完整追加负荷 cube 保持原 national/spatial/peak/response 输入，重新计算受影响的资源与余额，共 {int(cube_summary.scenario_rows.sum()):,} 行。",
        f"其中条件内余额符号变化 {int(cube_summary.sign_changed_rows.sum()):,} 行；计数是情景计数，不是发生概率。",
        "主 source 条件对应的已配对 DC 贡献无需改变；本轮已在新延迟资源背景下重算 MISO/ERCOT 的移除与替换配对。",
        "同一负荷与响应路径下，资源背景共同变化不会改变 E2 的配对压力差值；但会改变 E3b，即 DC 增长增加的补充需求。这两个量不能混称。",
        "旧干预输出尚未升级到这些新延迟背景，不可直接粘贴。", "",
        "年度时序对照（anticipated，published_all，无额外 DC 追加）："]
    selected = trajectories[trajectories.resource_condition.eq("anticipated_tier1_arrival_delayed_2y") & trajectories.response_condition.eq("published_all")]
    for row in selected.itertuples():
        old_first = str(int(row.old_first_below_reference_year)) if pd.notna(row.old_first_below_reference_year) else "期间无"
        new_first = str(int(row.new_first_below_reference_year)) if pd.notna(row.new_first_below_reference_year) else "期间无"
        rows.append(f"  {row.planning_region}, 2 年延迟：情景首次低于参考 {old_first} -> {new_first}；十年补充贡献逐年总和变化 {row.additional_required_contribution_year_sum_GW_year:+.2f} GW-year。")
    rows += ["GW-year 是逐年容量补充贡献之和，不是缺电电量，也不是实际可靠性事件预测。", "",
        f"新背景的 MISO/ERCOT 配对共 {int(paired_summary.paired_cases.sum()):,} 行；核心来源背景不变。新延迟背景下，同路径 DC 额外补充需求的最大增量为 {paired_summary.maximum_added_requirement_change_GW.max():.2f} GW。", "",
        "2035 相对 2026 的精确余额分解（anticipated，published_all，原生 GW）：",
        "区域 | existing 变化 | 转移 | Tier 1 | gross demand 项 | 响应项 | 参考要求项 | 余额净变化"]
    endpoints_decomp = decomposition[decomposition.year.eq(2035) & decomposition.resource_condition.eq("anticipated") & decomposition.response_condition.eq("published_all")]
    for row in endpoints_decomp.itertuples():
        rows.append(f"  {row.planning_region}: {row.existing_native_change_GW:+.2f} | {row.transfer_change_GW:+.2f} | {row.planned_contribution_change_GW:+.2f} | {row.gross_demand_term_GW:+.2f} | {row.response_term_GW:+.2f} | {row.reference_requirement_term_GW:+.2f} | {row.actual_balance_change_GW:+.2f}")
    rows += ["分解以 2026 年参考比例计算需求/响应变化，把参考比例交互项归入参考要求项；这是指定顺序下的精确会计分解，不是各项的独立因果估计。",
        "PJM 的参考要求项不可被误写成 DC 负荷或退役；ERCOT 的响应项不可被当成已实现的发电供给。", "",
        "三、历史起点与预测边界", "",
        f"历史 SRA 资源恒等式核查 {len(hist)} 个 region-year；planned Tier 1 占 anticipated 起点的最高比例为 {hist.planned_fraction_in_anticipated_anchor.max():.1%}。",
        "因此 anticipated-resource persistence 是规划资源起点，不是实际 operating stock。前一轮校准的权重不能解释成物理交付概率。",
        "前期计划项目到目标季节可能转入 existing 类别；类别差异不能直接当作交付/未交付真值。",
        "本轮没有修复跨年度全部 accreditation 差异，也没有取得独立的实际 accredited-resource 观测。不能声称完整 margin 已超过 persistence，或删掉不利回测。", "",
        "四、仍保留的科学边界", "",
        "Existing-certain 的下降可能来自退役、暂停或容量认可变化，不能从聚合表单独识别每个来源。",
        "ERCOT generation 表给出额外退役差额，但表基线与 dashboard 不完全一致；未确认单位级排除关系前，不追加扣除。",
        "Prospective 仍是包含其他容量、转移及潜在退役的原生条件，不能逐项当作已验证的可交付供给。",
        "这是输入和情景处理的可审计改进，不是项目交付模型、七区 ELCC 重估或独立预测验证。", "",
        f"内部 QA: {int(qa.passed.sum())}/{len(qa)} 项通过。原文件 hash 检查已完成。未 push。", ""]
    (HERE / "RESULTS_CN.txt").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    config = json.loads((HERE / "protocol.json").read_text())
    protected = [SOURCE, ROOT / "scripts/build_annual_scenario_cube.py", ROOT / "results/104_annual_scenario_cube_v1_2.csv",
        ROOT / "results/114_core_source_path_replacement_annual.csv", ROOT / "manuscript_reviewfix_20261004/main.tex",
        ROOT / "manuscript_reviewfix_20261004/supplementary_information.tex", B10 / "results/02_condition_matrix_region_year.csv",
        ROOT / "calibrated_reference_20261004/01_predictions.csv", ROOT / "calibrated_reference_20261004/02_metrics.csv"]
    before = {str(path): digest(path) for path in protected}
    reader = PdfReader(SOURCE)
    native = extract_native(reader, config)
    ercot = extract_ercot_response(reader)
    native = add_response(native, ercot)
    panel = condition_panel(native)
    comparison = compare_conditions(panel)
    decomposition = annual_decomposition(panel)
    cube, cube_summary = update_cube(comparison)
    historical = historical_stock_audit()
    evidence = evidence_registry(reader)
    retirement = retirement_differential(reader)
    trajectories = annual_condition_summary(comparison)
    paired, paired_summary = paired_replacement_panel(comparison)
    core = comparison[comparison.resource_condition.isin(["anticipated", "prospective"])]
    lagged = comparison[comparison.resource_condition.str.contains("tier1_arrival")]
    qa_items = [
        ("native_seven_by_ten", len(native) == 70 and not native.duplicated(["planning_region", "year"]).any()),
        ("full_condition_panel_1050", len(panel) == 1050),
        ("native_resource_components_reproduce", np.allclose(native.anticipated_native_MW, native.existing_and_transfer_MW + native.tier1_contribution_MW)),
        ("display_net_rounding_within_1_MW", native.net_display_rounding_difference_MW.abs().max() <= config["input_rounding_tolerance_MW"]),
        ("display_ARM_within_rounded_0_1_pp", native.ARM_display_difference_pp.abs().max() <= .11),
        ("no_extra_capacity_discount", native.generic_extra_accreditation_factor.eq(1).all()),
        ("ordinary_response_nonnegative", native.ordinary_response_MW.ge(0).all()),
        ("current_cap_respects_source_availability", native.current_cap_response_MW.le(native.ordinary_response_MW).all()),
        ("first_year_response_cap_respected", native.current_cap_response_MW.le(native.first_forecast_ordinary_response_MW).all()),
        ("core_source_values_unchanged_within_csv_rounding", core.balance_change_GW.abs().max() < 1e-6),
        ("core_source_signs_unchanged", not core.below_reference_sign_changed.any()),
        ("timing_never_exceeds_anticipated", (lagged.capacity_equivalent_resources_MW <= lagged.anticipated_native_MW + 1e-8).all()),
        ("timing_first_year_tier1_zero", lagged.loc[lagged.year.eq(2026), "counted_tier1_MW"].eq(0).all()),
        ("exact_decomposition", decomposition.residual_GW.abs().max() < 1e-8),
        ("historical_SRA_resource_identity_within_rounding", historical.resource_identity_difference_MW.abs().max() <= 1.01),
        ("source_evidence_matches", evidence.matched.all()),
        ("cube_same_case_count", len(cube) == len(pd.read_csv(ROOT / "results/104_annual_scenario_cube_v1_2.csv", usecols=["year"]))),
        ("cube_resource_delta_equals_balance_delta", np.allclose((cube.scenario_balance_GW - cube.legacy_scenario_balance_GW)[cube.scenario_applicable], (cube.resource_change_MW / 1000)[cube.scenario_applicable], atol=1e-7)),
        ("paired_full_case_count", len(paired) == 388800),
        ("paired_pressure_preserved_under_same_resource_shift", paired.dc_pressure_change_from_undelayed_GW.abs().max() < 1e-7),
        ("paired_pressure_matches_candidate_net_path", np.allclose(paired.dc_path_pressure_contribution_GW, paired.candidate_net_path_GW * (1 + paired.reference_margin_level_pct / 100), atol=1e-7)),
        ("paired_requirement_positive_part_identity", np.allclose(paired.dc_added_required_contribution_GW, np.maximum(0, -paired.replacement_balance_GW) - np.maximum(0, -paired.source_path_removed_balance_GW))),
        ("paired_source_background_preserved", paired.loc[paired.resource_condition_type.eq("source_condition"), "paired_resource_shift_GW"].abs().max() < 1e-7),
    ]
    for path, sha in before.items():
        qa_items.append((f"preserved_{Path(path).name}", digest(Path(path)) == sha))
    qa = pd.DataFrame([{"check": name, "passed": bool(value)} for name, value in qa_items])
    qa.to_csv(HERE / "12_qa.csv", index=False)
    if not qa.passed.all():
        raise AssertionError(qa[~qa.passed].to_string(index=False))
    outputs = {
        "01_native_stock_additions_transfer_ledger.csv": native,
        "02_ercot_summer_linked_response_source.csv": ercot,
        "03_corrected_condition_panel.csv": panel,
        "04_old_new_condition_comparison.csv": comparison,
        "05_separated_annual_decomposition.csv": decomposition,
        "06_annual_scenario_cube_corrected.csv": cube,
        "07_cube_change_summary.csv": cube_summary,
        "08_historical_SRA_stock_anchor_audit.csv": historical,
        "09_source_evidence_checks.csv": evidence,
        "10_retirement_baseline_exclusion_audit.csv": retirement,
        "11_preservation_manifest.csv": pd.DataFrame([{"path": path, "sha256": sha} for path, sha in before.items()]),
        "13_annual_condition_trajectory_comparison.csv": trajectories,
        "14_paired_DC_replacement_new_background.csv": paired,
        "15_paired_DC_background_impact_summary.csv": paired_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False)
        print(f"Saved {name}: {len(frame):,} rows", flush=True)
    write_report(native, comparison, cube_summary, historical, qa, trajectories, paired_summary, decomposition)
    print(f"QA {int(qa.passed.sum())}/{len(qa)}; no legacy files modified", flush=True)


if __name__ == "__main__":
    main()
