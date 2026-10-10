"""Retrospective stock-history and source-increment bridge experiment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parent.parent
BATCH = ROOT / "historical"
PANEL = ROOT / "historical_main_scenario_20261004/01_current_main_historical_panel.csv"
SRA = BATCH / "02_sra_outcome_panel.csv"
LTRA = BATCH / "01_ltra_forecast_panel.csv"
spec = importlib.util.spec_from_file_location(
    "prior_component_rules", ROOT / "calibrated_reference_20261004/build_calibrated_reference.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
KEYS = ["vintage", "target_year", "planning_region"]
CANDIDATES = ["trend_level_resources", "change_only_bridge"]
PRIMARY = ["balance_GW", "gap_pp"]
FIELDS = {
    "demand": "total_internal_demand_MW",
    "response": "reported_demand_response_MW",
    "resources": "anticipated_resources_MW",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_origin_features(panel, sra, ltra):
    """Use only records published by each row's forecast origin."""
    out = panel.copy()
    first = ltra[ltra.lead_years.eq(1)][[
        "vintage", "planning_region", "existing_certain_and_transfers_MW", "tier1_additions_MW"]].copy()
    first["first_forward_resources_MW"] = first.existing_certain_and_transfers_MW + first.tier1_additions_MW
    out = out.merge(first[["vintage", "planning_region", "first_forward_resources_MW"]],
                    on=["vintage", "planning_region"], validate="many_to_one")
    features = []
    for vintage, region in out[["vintage", "planning_region"]].drop_duplicates().itertuples(index=False):
        history = sra[sra.planning_region.eq(region) & sra.target_year.le(vintage)].sort_values("target_year")
        row = dict(vintage=vintage, planning_region=region,
                   maximum_stock_history_year=int(history.target_year.max()))
        consecutive = history.target_year.diff().eq(1)
        for component, field in FIELDS.items():
            changes = history[field].diff()[consecutive].tail(3)
            row[f"drift_{component}_MW_per_year"] = float(changes.median()) if len(changes) >= 2 else 0.0
        row["stock_history_changes"] = int(consecutive.tail(3).sum())
        features.append(row)
    out = out.merge(pd.DataFrame(features), on=["vintage", "planning_region"], validate="many_to_one")
    out["resource_trend_MW"] = np.maximum(0.0,
        out.origin_anticipated_resources_MW + out.lead_years * out.drift_resources_MW_per_year)
    out["level_resource_feature_MW"] = out.forecast_resources_MW - out.resource_trend_MW
    out["bridge_resource_feature_MW"] = (out.forecast_resources_MW - out.first_forward_resources_MW
        - (out.lead_years - 1) * out.drift_resources_MW_per_year)
    return out


def load_coefficients(train, supported, config):
    if not supported:
        return dict(demand=0.0, response=0.0)
    scale = (train.origin_total_internal_demand_MW - train.origin_reported_demand_response_MW).to_numpy()
    result = {}
    for component in ["demand", "response"]:
        base, official, outcome = prior.COMPONENTS[component]
        result[component] = float(prior.fit_lad(
            (train[official] - train[base]).to_numpy()[:, None],
            (train[outcome] - train[base]).to_numpy(), scale, prior.weights(train), config)[0])
    return result


def resource_coefficient(train, model, config):
    feature = "level_resource_feature_MW" if model == "trend_level_resources" else "bridge_resource_feature_MW"
    scale = (train.origin_total_internal_demand_MW - train.origin_reported_demand_response_MW).to_numpy()
    return float(prior.fit_lad(train[feature].to_numpy()[:, None],
        (train.outcome_anticipated_resources_MW - train.resource_trend_MW).to_numpy(),
        scale, prior.weights(train), config)[0])


def predict(data, model, coeff, supported):
    if model in ["common_reference_persistence", "fully_frozen_persistence", "raw_official"]:
        return prior.predict(data, model, {})
    if model == "anchored_split_resources":
        return prior.predict(data, model, coeff)
    out = data.copy()
    for component in ["demand", "response"]:
        base, official, _ = prior.COMPONENTS[component]
        if model == "historical_component_drift":
            value = out[base] + out.lead_years * out[f"drift_{component}_MW_per_year"]
        else:
            value = out[base] + coeff[component] * (out[official] - out[base])
        out[f"predicted_{component}_MW"] = np.maximum(0.0, value)
    if model == "resource_persistence_calibrated_load":
        out["predicted_resources_MW"] = out.origin_anticipated_resources_MW
    else:
        out["predicted_resources_MW"] = out.resource_trend_MW
        if model in CANDIDATES and supported:
            feature = "level_resource_feature_MW" if model == "trend_level_resources" else "bridge_resource_feature_MW"
            out["predicted_resources_MW"] += coeff["resource_alpha"] * out[feature]
            out["predicted_resources_MW"] = np.maximum(0.0, out.predicted_resources_MW)
    if model in CANDIDATES and not supported:
        for component, (base, _, _) in prior.COMPONENTS.items():
            out[f"predicted_{component}_MW"] = out[base]
    out["predicted_RML_pct"] = out.reference_margin_level_pct
    accounting = prior.history.account(out.predicted_demand_MW, out.predicted_response_MW,
                                        out.predicted_resources_MW, out.predicted_RML_pct)
    for field in ["margin_MW", "gap_to_RML_pp", "reference_requirement_MW", "net_peak_demand_MW"]:
        out[f"predicted_{field}"] = accounting[field].to_numpy()
    out["model"] = model
    return out


def run_rolling(panel, config, heldout=None, quiet=False):
    forecasts, parameters = [], []
    for vintage in config["forecast_vintages"]:
        test = panel[panel.vintage.eq(vintage)].copy()
        train = prior.training_rows(panel, vintage)
        if heldout is not None:
            train = train[train.planning_region.ne(heldout)]
        supported = prior.enough_training(train, config)
        load = load_coefficients(train, supported, config)
        previous = prior.fit_model(train, "anchored_split_resources" if supported else
                                  "common_reference_persistence", config)
        if not supported:
            previous = dict(demand=0.0, response=0.0, resources_existing=0.0, resources_tier1=0.0)
        for model in config["comparators"] + CANDIDATES:
            coeff = previous if model == "anchored_split_resources" else load.copy()
            if model in CANDIDATES:
                coeff["resource_alpha"] = resource_coefficient(train, model, config) if supported else 0.0
            forecast = predict(test, model, coeff, supported)
            forecast["model"] = model
            forecast["fit_supported"] = supported
            forecast["selection_supported"] = False
            forecast["heldout_region"] = heldout
            forecasts.append(forecast)
            if model in CANDIDATES:
                for name, value in coeff.items():
                    parameters.append(dict(vintage=vintage, model=model, coefficient=name, value=value,
                        train_cases=len(train), train_regions="|".join(sorted(train.planning_region.unique())),
                        max_train_origin=train.vintage.max() if len(train) else np.nan,
                        max_train_outcome=train.target_year.max() if len(train) else np.nan,
                        fit_supported=supported, heldout_region=heldout))
        if not quiet:
            print(f"Origin {vintage}: {len(train)} matured cases, supported={supported}", flush=True)
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(parameters)


def improvement_tables(predictions, config):
    deletions, intervals = [], []
    rng = np.random.default_rng(config["bootstrap_seed"])
    for scope, pool in [("all_origins", predictions), ("fitted_origins", predictions[predictions.fit_supported])]:
        for model in CANDIDATES:
            for baseline in ["common_reference_persistence", "fully_frozen_persistence", "resource_drift_calibrated_load"]:
                data = pool[pool.model.eq(model)]
                comparator = pool[pool.model.eq(baseline)]
                for variable in PRIMARY:
                    prediction, outcome, scale = prior.VARIABLES[variable]
                    matched = data.merge(comparator[KEYS + [prediction]].rename(
                        columns={prediction: "baseline_prediction"}), on=KEYS, validate="one_to_one")
                    matched["improvement"] = ((matched.baseline_prediction - matched[outcome]).abs()
                        - (matched[prediction] - matched[outcome]).abs()) / scale
                    for dimension in ["planning_region", "vintage"]:
                        for omitted in sorted(matched[dimension].unique()):
                            subset = matched[matched[dimension].ne(omitted)]
                            deletions.append(dict(scope=scope, model=model, baseline=baseline,
                                variable=variable, dimension=dimension, omitted=omitted,
                                improvement=subset.improvement.mean(), diagnostic="evaluation_deletion_not_refitting"))
                    for dimension in ["planning_region", "target_year"]:
                        block = matched.groupby(dimension).improvement.agg(["sum", "count"])
                        draw = rng.integers(0, len(block), (config["bootstrap_draws"], len(block)))
                        values = block["sum"].to_numpy()[draw].sum(axis=1) / block["count"].to_numpy()[draw].sum(axis=1)
                        intervals.append(dict(scope=scope, model=model, baseline=baseline, variable=variable,
                            dimension=dimension, blocks=len(block), improvement=matched.improvement.mean(),
                            ci_low=np.quantile(values, .025), ci_high=np.quantile(values, .975)))
    return pd.DataFrame(deletions), pd.DataFrame(intervals)


def audit_source_definitions(sra):
    from pypdf import PdfReader
    events = [
        ("ERCOT", "nerc_ltra_2025.pdf", 130, "average ELCC", "LTRA changes IBR contribution from historical on-peak factors to average ELCC", "capacity_basis_change_not_reconciled_by_generic_scaling"),
        ("ERCOT", "nerc_sra_2026.pdf", 34, "67.9%", "SRA peak-hour ARM is 67.9%; peak-hour adequacy and 21:00 risk remain distinct", "on_peak_not_risk_hour_or_LTRA_ELCC"),
        ("all", "nerc_sra_2026.pdf", 44, "Tier 1", "Anticipated includes existing certain, Tier1 planned and firm transfers", "not_operating_inventory"),
        ("all", "nerc_sra_2026.pdf", 43, "controllable", "Net internal demand subtracts controllable and dispatchable peak-hour DR", "DR_not_arbitrary_future_curtailment"),
        ("SPP", "nerc_sra_2026.pdf", 49, "10.0%", "SPP SRA RML changes 19% to 10%; table does not identify a capacity-currency conversion", "unresolved_require_primary_rule_crosswalk"),
    ]
    rows = []
    for region, file, page, needle, interpretation, boundary in events:
        path = BATCH / "sources" / file
        text = PdfReader(path).pages[page-1].extract_text() or ""
        if needle not in text:
            raise AssertionError((file, page, needle))
        position = text.index(needle)
        rows.append(dict(region=region, file=file, pdf_page=page, sha256=sha(path),
            matched_text=text[max(0, position-160):position+280].replace("\n", " "),
            finding=interpretation, boundary=boundary, evidence_verified=True))
    stock = sra.copy()
    stock["stock_identity_error_MW"] = stock.anticipated_resources_MW - (
        stock.existing_certain_capacity_MW + stock.tier1_planned_capacity_MW + stock.net_firm_capacity_transfers_MW)
    stock["planned_share_of_anticipated"] = stock.tier1_planned_capacity_MW / stock.anticipated_resources_MW
    manifest = pd.read_csv(WORKSPACE / "natcomm_revision_execution_20260929/results/78_eia860m_target_season_source_manifest.csv")
    manifest["LTRA_origin_eligibility"] = np.where(manifest.role.eq("forecast_origin"),
        "December month-end state is not certified available at December LTRA origin; not used as a predictor here",
        "Outcome-month snapshot; not an origin predictor for that same year")
    manifest["native_capacity_mapping"] = "Not established; physical EIA capacities are not NERC accredited contributions"
    manifest["archival_revision_boundary"] = "Downloaded archive is not independently certified as the bytes available on historical publication date"
    return pd.DataFrame(rows), stock, manifest


def main():
    config = json.loads((HERE / "protocol.json").read_text())
    protected = [PANEL, SRA, LTRA,
        ROOT / "calibrated_reference_20261004/01_predictions.csv",
        ROOT / "joint_balance_calibration_20261005/01_predictions.csv"]
    protected += [ROOT / "manuscript_reviewfix_20261004" / name for name in
        ["main.tex", "main.pdf", "supplementary_information.tex", "supplementary_information.pdf"]]
    manifest = {str(path): sha(path) for path in protected}
    protocol_hash = sha(HERE / "protocol.json")
    full = pd.read_csv(PANEL)
    raw = full[full["sample"].eq("matched_six_regions")].copy()
    sra, ltra = pd.read_csv(SRA), pd.read_csv(LTRA)
    source_audit, stock, eia_timing = audit_source_definitions(sra)
    panel = add_origin_features(raw, sra, ltra)
    forecasts, parameters = run_rolling(panel, config)
    summary = prior.summarize(forecasts)
    deletions, intervals = improvement_tables(forecasts, config)
    heldout, heldout_parameters = [], []
    for region in config["regions"]:
        fold, params = run_rolling(panel, config, heldout=region, quiet=True)
        heldout.append(fold[fold.planning_region.eq(region)])
        heldout_parameters.append(params)
    heldout = pd.concat(heldout, ignore_index=True)
    heldout_metrics = prior.metrics(heldout, ["model"])
    metrics = summary[summary.scope.eq("all_origins") & summary.horizon.eq("all")]
    fitted = summary[summary.scope.eq("fitted_origins") & summary.horizon.eq("all")]
    def mae(frame, model, variable):
        return float(frame[frame.model.eq(model) & frame.variable.eq(variable)].iloc[0].mae)
    gates = []
    for model in CANDIDATES:
        baseline = ["common_reference_persistence", "fully_frozen_persistence"]
        pooled = all(mae(frame, model, variable) < mae(frame, compare, variable)
            for frame in [metrics, fitted] for variable in PRIMARY for compare in baseline)
        deleted = deletions[deletions.model.eq(model) & deletions.baseline.isin(baseline)]
        ci = intervals[intervals.model.eq(model) & intervals.baseline.isin(baseline)]
        transfer = all(mae(heldout_metrics, model, variable) < mae(heldout_metrics, compare, variable)
            for variable in PRIMARY for compare in baseline)
        source_info = all(mae(frame, model, variable) < mae(frame, "resource_drift_calibrated_load", variable)
            for frame in [metrics, fitted] for variable in PRIMARY)
        gates.append(dict(model=model, both_metrics_both_persistence_baselines=pooled,
            no_deletion_reversal=bool(deleted.improvement.ge(-1e-8).all()),
            true_region_heldout_improvement=transfer,
            approximate_block_CIs_positive=bool(ci.ci_low.gt(0).all()),
            source_information_beyond_stock_drift=source_info,
            gate_passed=bool(pooled and deleted.improvement.ge(-1e-8).all()
                and transfer and ci.ci_low.gt(0).all() and source_info)))
    gates = pd.DataFrame(gates)
    old = pd.read_csv(ROOT / "calibrated_reference_20261004/01_predictions.csv")
    unchanged_models = ["common_reference_persistence", "fully_frozen_persistence", "raw_official", "anchored_split_resources"]
    matched = forecasts[forecasts.model.isin(unchanged_models)].merge(
        old[KEYS + ["model", "predicted_margin_MW"]], on=KEYS + ["model"], suffixes=("", "_old"), validate="one_to_one")
    decomposition, cancellation = prior.error_decomposition(forecasts)
    qa = [
        dict(check="unchanged_150_cases_42_unique_targets", passed=len(raw) == 150 and len(raw[["planning_region", "target_year"]].drop_duplicates()) == 42),
        dict(check="identical_cases_no_duplicates", passed=all(len(data) == 150 and not data.duplicated(KEYS).any() for _, data in forecasts.groupby("model"))),
        dict(check="origin_stock_history_no_future", passed=panel.maximum_stock_history_year.le(panel.vintage).all()),
        dict(check="training_origin_no_future", passed=parameters.max_train_origin.dropna().lt(parameters.loc[parameters.max_train_origin.notna(), "vintage"]).all()),
        dict(check="training_outcome_matured", passed=parameters.max_train_outcome.dropna().le(parameters.loc[parameters.max_train_outcome.notna(), "vintage"]).all()),
        dict(check="coefficients_bounded", passed=parameters.value.between(-1e-8, 1+1e-8).all()),
        dict(check="same_first_forward_year_feature_zero", passed=panel.loc[panel.lead_years.eq(1), "bridge_resource_feature_MW"].abs().max() < 1e-8),
        dict(check="anticipated_resource_identity", passed=stock.stock_identity_error_MW.abs().max() <= 1.01),
        dict(check="all_source_audit_snippets_found", passed=source_audit.evidence_verified.all()),
        dict(check="preserved_prior_comparators", passed=np.allclose(matched.predicted_margin_MW, matched.predicted_margin_MW_old, atol=1e-7)),
        dict(check="six_true_heldout_folds", passed=len(heldout) == len(forecasts) and heldout.heldout_region.eq(heldout.planning_region).all()),
        dict(check="exact_error_decomposition", passed=np.allclose(decomposition.error_sum_MW, forecasts.predicted_margin_MW - forecasts.outcome_margin_MW, atol=1e-7)),
        dict(check="finite_positive_components", passed=np.isfinite(forecasts.predicted_margin_MW).all() and forecasts.predicted_resources_MW.ge(0).all() and forecasts.predicted_net_peak_demand_MW.gt(0).all()),
        dict(check="protocol_unchanged", passed=protocol_hash == sha(HERE / "protocol.json")),
    ]
    qa += [dict(check="preserved:" + path, passed=sha(path) == digest) for path, digest in manifest.items()]
    qa = pd.DataFrame(qa)
    if not qa.passed.all():
        raise AssertionError(qa[~qa.passed].to_string(index=False))
    outputs = {
        "01_predictions.csv": forecasts, "02_metrics.csv": summary,
        "03_coefficients.csv": parameters, "04_origin_features.csv": panel,
        "05_evaluation_deletions.csv": deletions, "06_block_intervals.csv": intervals,
        "07_true_region_heldout_predictions.csv": heldout,
        "08_true_region_heldout_metrics.csv": heldout_metrics,
        "09_true_region_heldout_coefficients.csv": pd.concat(heldout_parameters, ignore_index=True),
        "10_source_definition_audit.csv": source_audit, "11_anticipated_stock_identity.csv": stock,
        "12_EIA_snapshot_timing_audit.csv": eia_timing, "13_acceptance_gates.csv": gates,
        "14_QA.csv": qa, "15_by_region.csv": prior.metrics(forecasts, ["planning_region", "model"]),
        "16_by_vintage.csv": prior.metrics(forecasts, ["vintage", "model"]),
        "17_error_decomposition.csv": decomposition, "18_error_cancellation.csv": cancellation,
        "19_preservation_manifest.csv": pd.DataFrame([dict(path=path, sha256=digest) for path, digest in manifest.items()]),
    }
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False)
    record = dict(protocol_sha256=protocol_hash, builder_sha256=sha(__file__),
        python=platform.python_version(), executable=sys.executable,
        numpy=np.__version__, pandas=pd.__version__, retrospective_not_blind=True,
        manuscript_promoted=False, protected_files_unchanged=True)
    (HERE / "run_record.json").write_text(json.dumps(record, indent=2))
    print(metrics[metrics.variable.isin(PRIMARY)][["model", "variable", "n", "mae"]].to_string(index=False))
    print(gates.to_string(index=False))
    print(f"QA {qa.passed.sum()}/{len(qa)}", flush=True)


if __name__ == "__main__":
    main()
