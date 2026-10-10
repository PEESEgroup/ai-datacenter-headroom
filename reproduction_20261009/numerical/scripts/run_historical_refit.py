"""Refit all nine declared methods and six heldout-region folds locally."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_review_package import module, compare

ROOT = Path(__file__).resolve().parents[1]


def main():
    bridge = module("portable_supply_bridge", ROOT / "historical_supply_bridge_20261005/build_supply_bridge.py")
    config = json.loads((ROOT / "historical_supply_bridge_20261005/protocol.json").read_text())
    raw = pd.read_csv(bridge.PANEL)
    raw = raw[raw["sample"].eq("matched_six_regions")].copy()
    panel = bridge.add_origin_features(raw, pd.read_csv(bridge.SRA), pd.read_csv(bridge.LTRA))
    assert len(panel) == 150
    assert panel.maximum_stock_history_year.le(panel.vintage).all()
    predictions, parameters = bridge.run_rolling(panel, config)
    metrics = bridge.prior.summarize(predictions)
    out = ROOT / "results/historical_refit"
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "01_predictions.csv", index=False)
    parameters.to_csv(out / "03_coefficients.csv", index=False)
    metrics.to_csv(out / "02_metrics.csv", index=False)
    compare(ROOT / "expected/01_predictions.csv", out / "01_predictions.csv",
            ["vintage", "target_year", "planning_region", "model"],
            ["predicted_margin_MW", "predicted_gap_to_RML_pp", "predicted_demand_MW", "predicted_response_MW", "predicted_resources_MW"])
    heldout = []
    for region in config["regions"]:
        fold, _ = bridge.run_rolling(panel, config, heldout=region, quiet=True)
        heldout.append(fold[fold.planning_region.eq(region)])
    heldout = pd.concat(heldout, ignore_index=True)
    heldout.to_csv(out / "07_true_region_heldout_predictions.csv", index=False)
    compare(ROOT / "expected/07_true_region_heldout_predictions.csv", out / "07_true_region_heldout_predictions.csv",
            ["vintage", "target_year", "planning_region", "model"], ["predicted_margin_MW", "predicted_gap_to_RML_pp"])
    assert parameters.max_train_origin.dropna().lt(parameters.loc[parameters.max_train_origin.notna(), "vintage"]).all()
    assert parameters.max_train_outcome.dropna().le(parameters.loc[parameters.max_train_outcome.notna(), "vintage"]).all()
    assert parameters.value.between(-1e-8, 1+1e-8).all()
    record = {"all_passed": True, "historical_refitted": True, "methods": int(predictions.model.nunique()),
              "cases_per_method": 150, "heldout_regions": config["regions"], "prospective_blind_validation": False}
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
