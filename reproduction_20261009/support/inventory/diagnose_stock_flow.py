"""Fixed diagnostics, source anomalies, held-out refits and evidence gates."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stock',HERE/'build_stock_flow.py')
stock=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stock)


def main():
    config=json.loads((HERE/'protocol.json').read_text())
    ledger=pd.read_csv(HERE/'01_full_inventory_ledger.csv')
    cohorts=pd.read_csv(HERE/'04_target_active_cohorts.csv')
    unseen=pd.read_csv(HERE/'05_unseen_entry_cohorts.csv')
    weights=pd.read_csv(HERE/'03_source_peak_contribution_weights.csv')
    predictions=pd.read_csv(HERE/'08_predictions.csv')
    parameters=pd.read_csv(HERE/'09_probability_parameters.csv')
    unit_predictions=pd.read_csv(HERE/'10_unit_predictions.csv')
    flow=pd.read_csv(HERE/'06_exact_observed_stock_flow.csv')
    intervals=pd.read_csv(HERE/'14_block_intervals.csv')
    deletions=pd.read_csv(HERE/'13_deletion_diagnostics.csv')
    anomalies=ledger[ledger.conflicting_sheets| (ledger.sheet.eq('Planned') & ledger.status.eq('OP'))
        | (ledger.sheet.eq('Operating') & ~ledger.status.isin(['OP','SB','OA','OS']))]
    anomalies.to_csv(HERE/'18_raw_status_anomalies.csv',index=False)
    heldout_predictions,heldout_parameters=[],[]
    for region in config['regions']:
        p,par,_=stock.aggregate_predictions(cohorts,unseen,ledger,weights,config,heldout=region)
        heldout_predictions.append(p[p.region.eq(region)])
        heldout_parameters.append(par)
    heldout=pd.concat(heldout_predictions,ignore_index=True)
    heldout.to_csv(HERE/'19_true_region_heldout_predictions.csv',index=False)
    pd.concat(heldout_parameters,ignore_index=True).to_csv(HERE/'20_true_region_heldout_parameters.csv',index=False)
    stock.metrics(heldout).to_csv(HERE/'21_true_region_heldout_metrics.csv',index=False)
    status=ledger[ledger.region.notna() & ledger.sheet.eq('Operating')].groupby(['snapshot_year','region','status']).agg(
        units=('unit_key','size'),summer_MW=('summer_MW','sum')).reset_index()
    unit_summary=[]
    for (origin,target,region),g in cohorts.groupby(['vintage','target_year','region']):
        unit_summary.append(dict(vintage=origin,target_year=target,region=region,
            unknown_units=int(g.target_active.isna().sum()),
            unknown_origin_capacity_MW=g.loc[g.target_active.isna()].capacity_for_fit_MW.sum(),
            imputed_origin_units=int(g.forecast_capacity_imputed.sum()),
            imputed_origin_capacity_MW=g.loc[g.forecast_capacity_imputed].forecast_unit_MW.sum(),
            origin_OP_scheduled_retirement_MW=g.loc[g.status.eq('OP')&g.retirement_due].forecast_unit_MW.sum(),
            surviving_scheduled_retirement_MW=g.loc[g.status.eq('OP')&g.retirement_due&g.target_active.eq(1)].forecast_unit_MW.sum(),
            origin_OP_not_active_at_target_MW=g.loc[g.status.eq('OP')&g.target_active.eq(0)].forecast_unit_MW.sum(),
            origin_OP_active_BA_moved_MW=g.loc[g.status.eq('OP')&g.target_active.eq(1)&~g.same_BA_region].forecast_unit_MW.sum(),
            realized_visible_project_summer_MW=g.loc[g.sheet.eq('Planned')&g.observed_active_in_origin_region.eq(1)].target_summer_MW.sum()))
    unit_summary=pd.DataFrame(unit_summary)
    unit_summary.to_csv(HERE/'22_survival_retirement_visibility_audit.csv',index=False)
    native=pd.read_csv(HERE.parent/'inputs/02_sra_outcome_panel.csv')
    comparison=[]
    for (year,region),g in ledger[ledger.is_active_OP & ledger.region.notna()].groupby(['snapshot_year','region']):
        ns=g.summer_MW.sum()
        same=native[native.target_year.eq(year)&native.planning_region.eq(region)]
        comparison.append(dict(year=year,region=region,OP_summer_physical_MW=ns,
            native_existing_certain_MW=same.existing_certain_capacity_MW.item() if len(same) else np.nan,
            interpretation='different capacity basis/footprint; difference is not forecast error or a fitted conversion'))
    pd.DataFrame(comparison).to_csv(HERE/'23_native_capacity_non_equivalence.csv',index=False)
    gate=[]
    for scope in intervals.scope.unique():
        for model in ['learned_visible_stock','learned_open_stock']:
            for quantity in intervals.quantity.unique():
                sub=intervals[intervals.scope.eq(scope)&intervals.model.eq(model)&intervals.quantity.eq(quantity)]
                deletion=deletions[deletions.scope.eq(scope)&deletions.model.eq(model)&deletions.quantity.eq(quantity)]
                gate.append(dict(scope=scope,model=model,quantity=quantity,
                    lower_MAE_than_both_comparators=bool(sub.improvement_GW.gt(0).all()),
                    all_evaluation_deletions_improve=bool(deletion.improvement_GW.gt(0).all()),
                    region_block_intervals_positive=bool(sub.loc[sub.dimension.eq('region'),'ci_low_GW'].gt(0).all()),
                    component_evidence_gate_passed=bool(sub.improvement_GW.gt(0).all()&deletion.improvement_GW.gt(0).all()
                        &sub.loc[sub.dimension.eq('region'),'ci_low_GW'].gt(0).all()),
                    native_full_balance_adoption_allowed=False))
    pd.DataFrame(gate).to_csv(HERE/'24_component_evidence_gates.csv',index=False)
    # Differences between successive reports' peak weights are capacity-definition diagnostics.
    w=weights.sort_values(['region','technology','year']).copy()
    w['prior_year_weight']=w.groupby(['region','technology']).weight.shift(1)
    w['weight_change']=w.weight-w.prior_year_weight
    w.to_csv(HERE/'25_peak_contribution_rule_changes.csv',index=False)
    status.to_csv(HERE/'26_OP_vs_other_operating_statuses.csv',index=False)
    unknown=cohorts[cohorts.target_active.isna()]
    unknown[['vintage','target_year','region','unit_key','origin_state','capacity_for_fit_MW','target_status']].to_csv(HERE/'27_unknown_outcome_records.csv',index=False)
    error_parts=[]
    for row in predictions[predictions.model.eq('learned_open_stock')].itertuples(index=False):
        g=cohorts[cohorts.vintage.eq(row.vintage)&cohorts.target_year.eq(row.target_year)&cohorts.region.eq(row.region)].copy()
        probability=unit_predictions[unit_predictions.vintage.eq(row.vintage)&unit_predictions.target_year.eq(row.target_year)&unit_predictions.region.eq(row.region)].set_index('unit_key').learned_probability
        p=g.unit_key.map(probability)
        actual_active=g.observed_active_in_origin_region.fillna(0)
        for quantity in ['physical_summer_stock','origin_weighted_capacity_index']:
            if quantity=='origin_weighted_capacity_index' and not row.index_available:
                continue
            if quantity=='physical_summer_stock':
                forecast_value=g.forecast_unit_MW
                actual_value=g.target_summer_MW
                total_forecast,total_actual=row.predicted_summer_MW,row.observed_summer_MW
                background=row.expected_unseen_summer_MW
            else:
                forecast_value=stock.peak_unit_values(g,weights,row.vintage)
                actual_value=stock.peak_unit_values(g,weights,row.vintage,use_target=True)
                total_forecast,total_actual=row.predicted_index_MW,row.observed_index_MW
                background=total_forecast-(forecast_value*p).sum()
            terms={}
            for name,mask in [('origin_OP',g.sheet.eq('Operating')&g.status.eq('OP')),
                             ('other_operating_status',g.sheet.eq('Operating')&g.status.ne('OP')),
                             ('visible_planned',g.sheet.eq('Planned'))]:
                terms[name+'_error_MW']=(forecast_value[mask]*p[mask]).sum()-(actual_value[mask]*actual_active[mask]).sum()
            all_matched_actual=(actual_value*actual_active).sum()
            terms['unmodeled_entries_and_background_error_MW']=background-(total_actual-all_matched_actual)
            reconstructed=sum(terms.values())
            error_parts.append(dict(vintage=row.vintage,target_year=row.target_year,region=row.region,
                quantity=quantity,**terms,total_error_MW=total_forecast-total_actual,
                reconstruction_residual_MW=reconstructed-(total_forecast-total_actual)))
    errors=pd.DataFrame(error_parts)
    assert errors.reconstruction_residual_MW.abs().max()<1e-6
    errors.to_csv(HERE/'28_exact_component_error_decomposition.csv',index=False)
    worst=[]
    for quantity,g in errors.groupby('quantity'):
        ranked=g.assign(absolute_error_GW=g.total_error_MW.abs()/1000).sort_values('absolute_error_GW',ascending=False).head(20)
        ranked['rank']=range(1,len(ranked)+1)
        worst.append(ranked)
    pd.concat(worst,ignore_index=True).to_csv(HERE/'29_largest_20_component_errors.csv',index=False)
    sectors=ledger[ledger.region.notna() & ledger.is_active_OP].groupby(['snapshot_year','region','sector']).agg(
        OP_units=('unit_key','size'),reported_summer_MW=('summer_MW','sum'),nameplate_MW=('nameplate_MW','sum')).reset_index()
    sectors['scope_warning']='BA-assigned EIA inventory includes commercial/industrial sectors; not an ISO firm-supply membership ledger'
    sectors.to_csv(HERE/'30_sector_scope_audit.csv',index=False)
    (HERE/'diagnostic_record.json').write_text(json.dumps(dict(
        true_heldout_prediction_rows=len(heldout),status_anomaly_rows=len(anomalies),
        unknown_cohort_rows=len(unknown),probability_coefficient_rows=len(parameters),
        maximum_stock_flow_residual_MW=float(flow.residual_MW.abs().max()),
        maximum_error_decomposition_residual_MW=float(errors.reconstruction_residual_MW.abs().max()),
        full_balance_accepted=False,
        reason='physical stock and fixed-origin index are not an audited accredited, tier/transfer-matched NERC balance'),indent=2))
    print(pd.DataFrame(gate).to_string(index=False))


if __name__=='__main__':
    main()
