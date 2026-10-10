#!/usr/bin/env python3
"""Audit small, rolling-origin calibrations of regional planning paths."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
HISTORY = ROOT / "historical_main_scenario_20261004"
BATCH = WORKSPACE / "natcomm_external_validation_batch8_20260917"
spec = importlib.util.spec_from_file_location("historical_accounting", HISTORY / "build_historical_test.py")
history = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(history)

COMPONENTS = {
    "demand": ("origin_total_internal_demand_MW", "total_internal_demand_MW", "outcome_total_internal_demand_MW"),
    "response": ("origin_reported_demand_response_MW", "reported_demand_response_MW", "outcome_reported_demand_response_MW"),
    "resources": ("origin_anticipated_resources_MW", "forecast_resources_MW", "outcome_anticipated_resources_MW"),
}
CASES = ["fully_frozen_persistence", "common_reference_persistence", "raw_official", "anchored_components", "anchored_split_resources", "rolling_selected"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows(panel: pd.DataFrame, vintage: int) -> pd.DataFrame:
    return panel[(panel.vintage < vintage) & (panel.target_year <= vintage)].copy()


def enough_training(train: pd.DataFrame, config: dict) -> bool:
    return (len(train[["planning_region", "target_year"]].drop_duplicates()) >= config["minimum_unique_region_targets"]
            and train.target_year.nunique() >= config["minimum_training_target_years"])


def weights(data: pd.DataFrame) -> np.ndarray:
    counts = data.groupby(["planning_region", "target_year"]).target_year.transform("size")
    w = 1 / counts.to_numpy(dtype=float)
    return w / w.sum()


def fit_lad(features: np.ndarray, change: np.ndarray, scale: np.ndarray, w: np.ndarray, config: dict) -> np.ndarray:
    """Bounded LAD with a fixed, negligible penalty to resolve flat minima."""
    x = np.asarray(features, dtype=float) * (100 / scale)[:, None]
    y = np.asarray(change, dtype=float) * 100 / scale
    n, p = x.shape
    objective = np.r_[np.full(p, config["tie_regularization_pp"]), w]
    constraints = np.r_[np.c_[x, -np.eye(n)], np.c_[-x, -np.eye(n)]]
    rhs = np.r_[y, -y]
    answer = linprog(objective, A_ub=constraints, b_ub=rhs,
                     bounds=[tuple(config["coefficient_bounds"])] * p + [(0, None)] * n,
                     method="highs")
    if not answer.success:
        raise RuntimeError(answer.message)
    return answer.x[:p]


def fit_model(train: pd.DataFrame, model: str, config: dict) -> dict[str, float]:
    if model == "common_reference_persistence":
        return {"demand": 0.0, "response": 0.0, "resources": 0.0}
    w = weights(train)
    scale = (train.origin_total_internal_demand_MW - train.origin_reported_demand_response_MW).to_numpy()
    coefficients = {}
    for component, (base, official, outcome) in COMPONENTS.items():
        if component == "resources" and model == "anchored_split_resources":
            x = np.c_[train.existing_certain_and_transfers_MW - train[base], train.tier1_additions_MW]
            alpha = fit_lad(x, (train[outcome] - train[base]).to_numpy(), scale, w, config)
            coefficients["resources_existing"] = float(alpha[0])
            coefficients["resources_tier1"] = float(alpha[1])
        else:
            alpha = fit_lad((train[official] - train[base]).to_numpy()[:, None],
                            (train[outcome] - train[base]).to_numpy(), scale, w, config)
            coefficients[component] = float(alpha[0])
    return coefficients


def predict(frame: pd.DataFrame, model: str, coefficients: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for component, (base, official, _) in COMPONENTS.items():
        if model == "raw_official":
            out[f"predicted_{component}_MW"] = out[official]
        elif model in ["fully_frozen_persistence", "common_reference_persistence"]:
            out[f"predicted_{component}_MW"] = out[base]
        elif component == "resources" and model == "anchored_split_resources":
            out[f"predicted_{component}_MW"] = (out[base]
                + coefficients["resources_existing"] * (out.existing_certain_and_transfers_MW - out[base])
                + coefficients["resources_tier1"] * out.tier1_additions_MW)
        else:
            out[f"predicted_{component}_MW"] = out[base] + coefficients[component] * (out[official] - out[base])
    rml = out.origin_reference_margin_level_pct if model == "fully_frozen_persistence" else out.reference_margin_level_pct
    out["predicted_RML_pct"] = rml
    calculated = history.account(out.predicted_demand_MW, out.predicted_response_MW, out.predicted_resources_MW, rml)
    for field in ["margin_MW", "gap_to_RML_pp", "reference_requirement_MW", "net_peak_demand_MW"]:
        out[f"predicted_{field}"] = calculated[field].to_numpy()
    out["model"] = model
    return out


def build_rolling(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, parameter_rows, selections = [], [], []
    cache = []
    for vintage in config["forecast_vintages"]:
        test = panel[panel.vintage == vintage]
        train = training_rows(panel, vintage)
        supported = enough_training(train, config)
        fit_models = {}
        for model in config["candidate_order"]:
            effective = model if supported else config["cold_start"]
            coefficients = fit_model(train, effective, config)
            forecast = predict(test, effective, coefficients)
            forecast["model"] = model
            forecast["effective_model"] = effective
            forecast["fit_supported"] = supported
            fit_models[model] = forecast
            for name, value in coefficients.items():
                parameter_rows.append({"vintage": vintage, "candidate": model, "effective_model": effective,
                    "coefficient": name, "value": value, "train_rows": len(train),
                    "train_unique_region_targets": len(train[["planning_region", "target_year"]].drop_duplicates()),
                    "maximum_train_origin": train.vintage.max() if len(train) else np.nan,
                    "maximum_train_outcome_year": train.target_year.max() if len(train) else np.nan,
                    "fit_supported": supported})
            predictions.append(forecast)
        matured = pd.concat(cache, ignore_index=True) if cache else pd.DataFrame()
        if len(matured):
            matured = matured[(matured.target_year <= vintage) & matured.fit_supported].copy()
        selection_supported = (len(matured) > 0
            and len(matured[["planning_region", "target_year"]].drop_duplicates()) >= config["selection_minimum_unique_region_targets"]
            and matured.vintage.nunique() >= config["selection_minimum_origins"])
        chosen = "anchored_components" if supported else config["cold_start"]
        losses = {}
        if selection_supported:
            for model in config["candidate_order"]:
                inner = matured[matured.model == model]
                origin_net = inner.origin_total_internal_demand_MW - inner.origin_reported_demand_response_MW
                loss = abs(inner.predicted_margin_MW - inner.outcome_margin_MW) / origin_net * 100
                losses[model] = float(np.dot(weights(inner), loss))
            best = min(losses.values())
            chosen = next(model for model in config["candidate_order"] if losses[model] <= best + config["selection_tie_tolerance_pp"])
        for model in config["candidate_order"]:
            selections.append({"vintage": vintage, "candidate": model, "selected": model == chosen,
                "selection_supported": selection_supported, "inner_loss_pp": losses.get(model, np.nan),
                "inner_rows_per_candidate": len(matured[matured.model == model]) if len(matured) else 0,
                "maximum_inner_outcome_year": matured.target_year.max() if len(matured) else np.nan})
        for forecast in fit_models.values():
            forecast["selection_supported"] = selection_supported
        selected = fit_models[chosen].copy()
        selected["model"] = "rolling_selected"
        selected["selected_candidate"] = chosen
        selected["selection_supported"] = selection_supported
        predictions.append(selected)
        for comparator in ["fully_frozen_persistence", "raw_official"]:
            forecast = predict(test, comparator, {})
            forecast["fit_supported"] = supported
            forecast["selection_supported"] = selection_supported
            predictions.append(forecast)
        cache.extend(fit_models.values())
        print(f"Origin {vintage}: {len(train)} historical rows, fit={supported}, selection={chosen}", flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameter_rows), pd.DataFrame(selections)


VARIABLES = {
    "balance_GW": ("predicted_margin_MW", "outcome_margin_MW", 1000),
    "gap_pp": ("predicted_gap_to_RML_pp", "outcome_gap_to_RML_pp", 1),
    "demand_GW": ("predicted_demand_MW", "outcome_total_internal_demand_MW", 1000),
    "resources_GW": ("predicted_resources_MW", "outcome_anticipated_resources_MW", 1000),
    "response_GW": ("predicted_response_MW", "outcome_reported_demand_response_MW", 1000),
}


def metrics(frame: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(grouping, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for variable, (prediction, outcome, scale) in VARIABLES.items():
            error = (subset[prediction] - subset[outcome]) / scale
            rows.append({**dict(zip(grouping, keys)), "variable": variable, "n": len(subset),
                "mae": float(abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean())})
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for scope, data in [("all_origins", predictions), ("fitted_origins", predictions[predictions.fit_supported]),
                        ("nested_selection_origins", predictions[predictions.selection_supported.eq(True)])]:
        for horizon in ["all", "1", "2", "3", "4", "5", "3-5"]:
            subset = data if horizon == "all" else data[data.lead_years.between(3, 5)] if horizon == "3-5" else data[data.lead_years == int(horizon)]
            table = metrics(subset, ["model"])
            table["scope"] = scope
            table["horizon"] = horizon
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def robustness(predictions: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, boots = [], []
    rng = np.random.default_rng(config["bootstrap_seed"])
    for scope, pool in [("all_origins", predictions), ("fitted_origins", predictions[predictions.fit_supported])]:
        for model in ["anchored_components", "anchored_split_resources", "rolling_selected"]:
            data = pool[pool.model == model].copy()
            keys = ["vintage", "target_year", "planning_region"]
            for baseline in ["fully_frozen_persistence", "common_reference_persistence", "raw_official"]:
                comparison = pool[pool.model == baseline]
                for variable in VARIABLES:
                    prediction, outcome, scale = VARIABLES[variable]
                    match = data.merge(comparison[keys + [prediction]].rename(columns={prediction: "baseline_prediction"}),
                                       on=keys, validate="one_to_one")
                    match["improvement"] = (abs(match.baseline_prediction - match[outcome]) - abs(match[prediction] - match[outcome])) / scale
                    for dimension in ["planning_region", "vintage"]:
                        for omitted in sorted(match[dimension].unique()):
                            subset = match[match[dimension] != omitted]
                            rows.append({"scope": scope, "model": model, "baseline": baseline, "variable": variable,
                                "omit_dimension": dimension, "omitted": omitted, "n": len(subset),
                                "diagnostic_type": "evaluation_case_deletion_not_heldout_region_refitting",
                                "improvement": float(subset.improvement.mean())})
                        blocks = match.groupby(dimension).improvement.agg(["sum", "count"])
                        draws = rng.integers(0, len(blocks), (config["bootstrap_draws"], len(blocks)))
                        samples = blocks["sum"].to_numpy()[draws].sum(axis=1) / blocks["count"].to_numpy()[draws].sum(axis=1)
                        boots.append({"scope": scope, "model": model, "baseline": baseline, "variable": variable,
                            "block_dimension": dimension, "improvement": float(match.improvement.mean()),
                            "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))})
    return pd.DataFrame(rows), pd.DataFrame(boots)


def error_decomposition(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = predictions.copy()
    factor = 1 + out.predicted_RML_pct / 100
    out["resource_error_MW"] = out.predicted_resources_MW - out.outcome_anticipated_resources_MW
    out["demand_error_MW"] = -(out.predicted_demand_MW - out.outcome_total_internal_demand_MW) * factor
    out["response_error_MW"] = (out.predicted_response_MW - out.outcome_reported_demand_response_MW) * factor
    observed_net = out.outcome_total_internal_demand_MW - out.outcome_reported_demand_response_MW
    out["RML_error_MW"] = -observed_net * (out.predicted_RML_pct - out.outcome_reference_margin_level_pct) / 100
    fields = ["resource_error_MW", "demand_error_MW", "response_error_MW", "RML_error_MW"]
    out["error_sum_MW"] = out[fields].sum(axis=1)
    out["absolute_component_sum_MW"] = out[fields].abs().sum(axis=1)
    out["balance_absolute_error_MW"] = abs(out.predicted_margin_MW - out.outcome_margin_MW)
    out["cancellation_MW"] = out.absolute_component_sum_MW - out.balance_absolute_error_MW
    rows = []
    for scope, data in [("all_origins", out), ("fitted_origins", out[out.fit_supported])]:
        for name, group in data.groupby("model"):
            rows.append({"scope": scope, "model": name, "n": len(group),
                "sum_component_MAE_GW": group.absolute_component_sum_MW.mean() / 1000,
                "balance_MAE_GW": group.balance_absolute_error_MW.mean() / 1000,
                "mean_cancellation_GW": group.cancellation_MW.mean() / 1000,
                **{field.replace("_MW", "_MAE_GW"): group[field].abs().mean() / 1000 for field in fields}})
    keep = ["vintage", "target_year", "planning_region", "model", "fit_supported", *fields,
            "error_sum_MW", "absolute_component_sum_MW", "balance_absolute_error_MW", "cancellation_MW"]
    return out[keep], pd.DataFrame(rows)


def annual_paths(primary: pd.DataFrame, config: dict, selections: pd.DataFrame) -> pd.DataFrame:
    ltra = pd.read_csv(BATCH / "results/01_ltra_forecast_panel.csv")
    california = pd.DataFrame(history.extract_ltra_region(BATCH / "sources/nerc_ltra_2025.pdf", 147, 2025, "WECC-California"))
    source = pd.concat([ltra[ltra.vintage == 2025], california], ignore_index=True)
    origin_columns = [value[0] for value in COMPONENTS.values()] + ["origin_reference_margin_level_pct"]
    origin = pd.read_csv(HISTORY / "01_current_main_historical_panel.csv")
    origin = origin[origin.vintage == 2025][["planning_region", *origin_columns]].drop_duplicates()
    source = source.merge(origin, on="planning_region", validate="many_to_one")
    source["forecast_resources_MW"] = source.existing_certain_and_transfers_MW + source.tier1_additions_MW
    train = training_rows(primary, 2025)
    selected = selections[(selections.vintage == 2025) & selections.selected].iloc[0].candidate
    tables = []
    for model in ["raw_official", "common_reference_persistence", "anchored_components", "anchored_split_resources"]:
        parameters = fit_model(train, model, config) if model not in ["raw_official", "common_reference_persistence"] else {}
        table = predict(source, model, parameters)
        table["selected_at_2025"] = model == selected
        table["interpretation"] = "retrospective_candidate_planning_path_not_physical_delivery_model"
        table["horizon_status"] = np.where(table.lead_years <= 5, "historical_lead_range_not_region_specific_acceptance", "unvalidated_scenario_transport")
        table["region_status"] = np.where(table.planning_region == "WECC-California", "pooled_transport_no_historical_training", "six_region_training_domain")
        table["fit_outcome_cutoff"] = 2025
        table["promotion_status"] = "audit_candidate_only_not_promoted_to_manuscript"
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def write_report(summary: pd.DataFrame, params: pd.DataFrame, selection: pd.DataFrame, loo: pd.DataFrame,
                 boots: pd.DataFrame, qa: pd.DataFrame, cancellation: pd.DataFrame) -> None:
    lines = ["# Calibrated Reference-Path Audit", "", "Date: 2026-10-04", "",
        "## What Was Tested", "",
        "This is retrospective method development with time-respecting fitting and nested selection. The historical outcomes were already inspected during earlier work, so it is not a fresh, blinded or prospectively preregistered validation.", "",
        "Every origin uses only earlier forecast origins with outcomes published by that origin. Future labels do not fit coefficients or select candidates. A small fixed family is compared with both fully frozen persistence and component persistence using the same forecast-source RML. Cold starts are explicitly shown and are not presented as forecast skill.", "",
        "The 150 rows contain only 42 distinct region-target assessments, repeated across origins and horizons; the fitted-origin 90 rows contain 30 distinct assessments. These are not 150 or 90 independent observed outcomes. Omission diagnostics delete evaluation cases to test dominance; they do not refit with a genuinely held-out region. Cluster intervals have only six regional blocks and are approximate.", "",
        "Dhat = D0 + alpha_D * (Dofficial - D0); DRhat = DR0 + alpha_DR * (DRofficial - DR0).", "",
        "Simple resource calibration: Rhat = R0 + alpha_R * (Rofficial - R0). Split calibration: Rhat = R0 + alpha_E * (Eofficial - R0) + alpha_T * Tier1official. All coefficients are pooled and bounded in [0,1].", "",
        "R0 is the origin SRA anticipated planning-resource quantity, not a verified operating inventory. Alpha_T is not a physical completion probability. These are statistical calibrations of later planning-assessment quantities and cannot replace retirement, ELCC or firm-transfer ledgers solely because a score improves.", "",
        "## Matched Results", "",
        "Scores below use identical cases for each comparator. GW denotes native region-level balance error, not a physical sum across regions.", "",
        "| Sample | Model | n | Balance MAE (GW) | Gap MAE (pp) |", "| --- | --- | ---: | ---: | ---: |"]
    for scope in ["all_origins", "fitted_origins", "nested_selection_origins"]:
        sub = summary[(summary.scope == scope) & (summary.horizon == "all")]
        for model in CASES:
            a = sub[(sub.model == model) & (sub.variable == "balance_GW")].iloc[0]
            b = sub[(sub.model == model) & (sub.variable == "gap_pp")].iloc[0]
            lines.append(f"| {scope} | {model} | {int(a.n)} | {a.mae:.4f} | {b.mae:.4f} |")
    lines += ["", "## Selection and Coefficients", "", "| Origin | Selected | Selection supported |", "| --- | --- | --- |"]
    for _, row in selection[selection.selected].iterrows():
        lines.append(f"| {int(row.vintage)} | {row.candidate} | {bool(row.selection_supported)} |")
    latest = params[(params.vintage == 2025) & params.fit_supported]
    lines += ["", "2025 fitted coefficients (only outcomes through 2025):", "", latest[["candidate", "coefficient", "value"]].to_string(index=False), "",
        "## Frozen Acceptance Gate", ""]
    for model in ["anchored_components", "anchored_split_resources", "rolling_selected"]:
        b = boots[(boots.scope == "fitted_origins") & (boots.model == model) & boots.variable.isin(["balance_GW", "gap_pp"]) & boots.baseline.isin(["fully_frozen_persistence", "common_reference_persistence"])]
        l = loo[(loo.scope == "fitted_origins") & (loo.model == model) & loo.variable.isin(["balance_GW", "gap_pp"]) & loo.baseline.isin(["fully_frozen_persistence", "common_reference_persistence"])]
        passed = bool((b.improvement > 0).all() and (b.ci_low > 0).all() and (l.improvement >= 0).all())
        lines.append(f"- {model}: {'PASS' if passed else 'NOT PASSED'}; the gate is not relaxed after seeing results.")
    lines += ["", "A better pooled score alone does not establish robustness or justify replacing the paper's main path. The component improvements, excluded-region reversals, uncertainty and source-definition changes must be visible.", "",
        "## Component Information and Error Cancellation", "",
        "Fitted-origin component scores use the same 90 cases; do not compare them directly with a different 150-case result.", "",
        "| Model | Demand MAE (GW) | Resources MAE (GW) | Response MAE (GW) |", "| --- | ---: | ---: | ---: |"]
    sub = summary[(summary.scope == "fitted_origins") & (summary.horizon == "all")]
    for model in CASES:
        values = [float(sub[(sub.model == model) & (sub.variable == variable)].iloc[0].mae)
                  for variable in ["demand_GW", "resources_GW", "response_GW"]]
        lines.append(f"| {model} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines += ["", "Component skill is not identical to balance skill: persistence can underestimate both supply and demand, with opposite-signed errors cancelling. A more accurate component can remove some accidental cancellation. This is an explanation to audit, not permission to ignore an unfavorable balance score.", "",
        cancellation[cancellation.scope == "fitted_origins"].to_string(index=False), "",
        "Component improvement must also be checked across regions. The following exclusions/intervals summarize each calibrated component versus persistence:", ""]
    for variable in ["demand_GW", "resources_GW", "response_GW"]:
        b = boots[(boots.scope == "fitted_origins") & (boots.model == "anchored_split_resources") & (boots.variable == variable) & (boots.baseline == "fully_frozen_persistence")]
        l = loo[(loo.scope == "fitted_origins") & (loo.model == "anchored_split_resources") & (loo.variable == variable) & (loo.baseline == "fully_frozen_persistence")]
        values = "; ".join(f"{row.block_dimension} CI [{row.ci_low:.4f}, {row.ci_high:.4f}]" for _, row in b.iterrows())
        lines.append(f"- {variable}: worst leave-one-out improvement {l.improvement.min():.4f} GW; {values}.")
    lines += ["", "## Practical Decision", "",
        "Do not adopt the calibrated candidate as a validated reference forecast: the full-balance gate did not pass. Keep these failed trials and coefficients in the audit trail. Demand contains forward information, and resource calibration can improve its own pooled score; neither implies that all scenarios are inaccurate nor that a reference path is predictively validated.", "",
        "The next substantive route is a matched seasonal resource-stock/accreditation/response ledger, together with independent demand evidence. Additional searches over horizons, baselines or parameterizations on these same outcomes would be exploratory development, not a fresh validation. No retrospective winning metric may be silently substituted for the full-balance score.", "",
        "## Boundaries and Next Decision", "",
        "The target is later SRA planning quantities, not realized outage risk or independently observed seasonal resources. Accreditation and response definitions can change; the unfavorable ERCOT 2025->2026 case is retained. All regions use the same statistical rules, but their native MW quantities are not identical physical estimands.", "",
        "The EIA demand-anchor result from the previous audit remains separate: it evaluates independently reported load, not the full resource balance. It is not substituted into the gap score.", "",
        "The 2025 candidate annual paths cover 2026-2035. The same coefficients apply consistently to the whole path. Leads 6-10 and California coefficient transport have not passed historical acceptance; the output is a scenario candidate, not a validated ten-year forecast. No DC-specific paired counterfactual or intervention is relabelled as recalculated by this run.", "",
        "Unknown future projects, explicit retirements and ELCC changes cannot be resolved by statistical shrinkage. Even a passing retrospective calibration would require physical-ledger review and genuinely new-vintage testing before strong predictive claims.", "",
        "Old results, current manuscript/SI and source inputs were preserved. No Overleaf/GitHub push was performed.", "",
        f"QA: {int(qa.passed.sum())}/{len(qa)} checks passed.", "",
        "## Reproduction", "",
        "`python build_calibrated_reference.py`", "",
        "`python -m unittest discover -s tests -v`", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = json.loads((HERE / "protocol.json").read_text())
    protected = list(HISTORY.glob("*.csv")) + [HISTORY / "REPORT.txt", HISTORY / "build_historical_test.py",
        ROOT / "manuscript_revision_20261003/main.tex", ROOT / "manuscript_revision_20261003/supplementary_information.tex",
        ROOT / "scripts/build_annual_scenario_cube.py", BATCH / "results/01_ltra_forecast_panel.csv", BATCH / "results/02_sra_outcome_panel.csv"]
    before = {str(path): digest(path) for path in protected}
    panel = pd.read_csv(HISTORY / "01_current_main_historical_panel.csv")
    primary = panel[panel["sample"] == "matched_six_regions"].copy()
    predictions, parameters, selections = build_rolling(primary, config)
    summary = summarize(predictions)
    loo, boots = robustness(predictions, config)
    decomposition, cancellation = error_decomposition(predictions)
    paths = annual_paths(primary, config, selections)
    latest_train = training_rows(primary, 2025)
    extension = panel[panel.planning_region == "WECC-California"]
    california = pd.concat([predict(extension, model, fit_model(latest_train, model, config))
        for model in ["common_reference_persistence", "anchored_components", "anchored_split_resources"]], ignore_index=True)
    outputs = {
        "01_predictions.csv": predictions,
        "02_metrics.csv": summary,
        "03_coefficients.csv": parameters,
        "04_nested_selection.csv": selections,
        "05_leave_one_out.csv": loo,
        "06_cluster_intervals.csv": boots,
        "07_metrics_by_region.csv": metrics(predictions, ["planning_region", "model"]),
        "08_candidate_annual_paths_2025_vintage.csv": paths,
        "09_california_extension.csv": california,
        "13_balance_error_decomposition.csv": decomposition,
        "14_error_cancellation_summary.csv": cancellation,
    }
    qa_rows = [
        ("primary_case_count_150", len(primary) == 150),
        ("each_model_identical_sample", all(len(predictions[predictions.model == name]) == 150 for name in CASES)),
        ("training_origins_strictly_earlier", (parameters.maximum_train_origin.dropna() < parameters.loc[parameters.maximum_train_origin.notna(), "vintage"]).all()),
        ("training_outcomes_available_at_origin", (parameters.maximum_train_outcome_year.dropna() <= parameters.loc[parameters.maximum_train_outcome_year.notna(), "vintage"]).all()),
        ("selection_outcomes_available_at_origin", (selections.maximum_inner_outcome_year.dropna() <= selections.loc[selections.maximum_inner_outcome_year.notna(), "vintage"]).all()),
        ("bounded_coefficients", parameters.value.between(0, 1).all()),
        ("complete_annual_candidate_paths", len(paths) == 280 and not paths.duplicated(["model", "planning_region", "target_year"]).any()),
        ("annual_training_cutoff_2025", paths.fit_outcome_cutoff.eq(2025).all()),
        ("finite_predictions", np.isfinite(predictions[["predicted_margin_MW", "predicted_gap_to_RML_pp"]].to_numpy()).all()),
        ("raw_official_reproduces_previous", np.allclose(predictions.loc[predictions.model == "raw_official", "predicted_margin_MW"], predictions.loc[predictions.model == "raw_official", "forecast_margin_MW"])),
        ("exact_signed_error_decomposition", np.allclose(decomposition.error_sum_MW.to_numpy(), (predictions.predicted_margin_MW - predictions.outcome_margin_MW).to_numpy(), atol=1e-7)),
    ]
    for path, original in before.items():
        qa_rows.append((f"preserved_{Path(path).name}", digest(Path(path)) == original))
    qa = pd.DataFrame([{"check": name, "passed": bool(passed)} for name, passed in qa_rows])
    if not qa.passed.all():
        raise AssertionError(qa[~qa.passed].to_string(index=False))
    outputs["10_qa.csv"] = qa
    outputs["11_preserved_input_manifest.csv"] = pd.DataFrame([{"path": path, "sha256": value} for path, value in before.items()])
    outputs["12_metrics_by_vintage.csv"] = metrics(predictions, ["vintage", "model"])
    for name, data in outputs.items():
        data.to_csv(HERE / name, index=False)
    import scipy
    import pypdf
    environment = {"python": platform.python_version(), "executable": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "pypdf": pypdf.__version__, "pypdf_module_path": pypdf.__file__,
        "protocol_sha256": digest(HERE / "protocol.json"), "builder_sha256": digest(Path(__file__))}
    (HERE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    write_report(summary, parameters, selections, loo, boots, qa, cancellation)
    print(f"Saved independent calibration audit; QA {int(qa.passed.sum())}/{len(qa)}", flush=True)


if __name__ == "__main__":
    main()
