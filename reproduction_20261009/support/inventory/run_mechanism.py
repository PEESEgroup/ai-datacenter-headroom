"""Rerun the existing stock-mechanism comparisons after the raw inventory fit."""
from pathlib import Path
import json
import pandas as pd
import build_stock_flow as stock
import mechanism_functions as mechanism

HERE=Path(__file__).resolve().parent


def main():
    config=json.loads((HERE/'protocol.json').read_text())
    cohorts=pd.read_csv(HERE/'04_target_active_cohorts.csv')
    units=pd.read_csv(HERE/'10_unit_predictions.csv')
    weights=pd.read_csv(HERE/'03_source_peak_contribution_weights.csv')
    old=pd.read_csv(HERE/'08_predictions.csv')
    combined=mechanism.hybrid_units(cohorts,units,weights,stock)
    forecasts=mechanism.hybrid_predictions(combined,old)
    output=HERE/'mechanism'
    output.mkdir(exist_ok=True)
    forecasts.to_csv(output/'01_all_supply_predictions.csv',index=False)
    stock.metrics(forecasts).to_csv(output/'02_supply_scores.csv',index=False)
    _,deletions,intervals=mechanism.diagnostics(forecasts,config)
    deletions.to_csv(output/'04_supply_deletion_checks.csv',index=False)
    intervals.to_csv(output/'05_supply_block_intervals.csv',index=False)
    mechanism.gates(intervals,deletions).to_csv(output/'06_supply_evidence_gates.csv',index=False)
    print(stock.metrics(forecasts).query("scope=='all' and split=='overall'").to_string(index=False))


if __name__=='__main__':
    main()
