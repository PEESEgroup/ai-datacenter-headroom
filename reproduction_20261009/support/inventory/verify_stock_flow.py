"""Independent output verification; does not import fitting or builder code."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    ledger=pd.read_csv(HERE/'01_full_inventory_ledger.csv')
    cohorts=pd.read_csv(HERE/'04_target_active_cohorts.csv')
    predictions=pd.read_csv(HERE/'08_predictions.csv')
    scores=pd.read_csv(HERE/'11_metrics.csv')
    weights=pd.read_csv(HERE/'03_source_peak_contribution_weights.csv')
    flow=pd.read_csv(HERE/'06_exact_observed_stock_flow.csv')
    parameters=pd.read_csv(HERE/'09_probability_parameters.csv')
    shared=pd.read_csv(HERE/'32_shared_capacity_reporting_audit.csv')
    for row in shared[shared.identified_reporting_pattern].itertuples(index=False):
        members=ledger[ledger.snapshot_year.eq(row.snapshot_year)&ledger.plant_id.eq(row.plant_id)
            &ledger.unit_code.fillna('').astype(str).eq(str(row.unit_code))]
        assert len(members)==row.members
        assert abs(members.summer_MW.sum()-row.reported_summer_MW)<1e-8
        assert members.loc[members.shared_capacity_member,'summer_MW'].isna().all()
        assert members.loc[members.shared_capacity_member,'capacity_for_fit_MW'].eq(0).all()
    covered=cohorts[cohorts.shared_capacity_member]
    assert covered.forecast_unit_MW.eq(0).all()
    assert not covered.forecast_capacity_imputed.any()
    checks.append('shared-block reported sums preserved; blank members have no duplicate forecast or fit credit')
    for file in ['02_EIA_source_manifest.csv','03_source_peak_contribution_weights.csv']:
        manifest=pd.read_csv(HERE/file)
        path_column='file' if file.startswith('02') else 'source_file'
        for _,row in manifest[[path_column,'sha256']].drop_duplicates().iterrows():
            assert hashlib.sha256((HERE.parent/row[path_column]).read_bytes()).hexdigest()==row.sha256
    checks.append('all EIA and NERC raw-source hashes verified')
    resolved=ledger[~ledger.conflicting_sheets].copy()
    for (vintage,target),g in cohorts.groupby(['vintage','target_year']):
        target_rows=resolved[resolved.snapshot_year.eq(target)].set_index('unit_key')
        status=g.unit_key.map(target_rows.status)
        tab=g.unit_key.map(target_rows.sheet)
        expected=pd.Series(np.nan,index=g.index)
        expected.loc[tab.eq('Operating') & status.eq('OP')]=1
        expected.loc[tab.eq('Operating') & status.isin(['SB','OA','OS'])]=0
        expected.loc[tab.isin(['Planned','Retired','Canceled or Postponed'])]=0
        expected.loc[tab.eq('Planned') & status.eq('OP')]=np.nan
        assert np.allclose(expected,g.target_active,equal_nan=True)
    checks.append('all raw-sheet target labels independently reconstructed')
    for row in predictions.itertuples(index=False):
        target=ledger[ledger.snapshot_year.eq(row.target_year)&ledger.is_active_OP&ledger.region.eq(row.region)]
        assert abs(target.summer_MW.sum()-row.observed_summer_MW)<1e-6
        if row.model=='inventory_persistence':
            origin=ledger[ledger.snapshot_year.eq(row.vintage)&ledger.is_active_OP&ledger.region.eq(row.region)]
            assert abs(origin.summer_MW.sum()-row.predicted_summer_MW)<1e-6
        if row.index_available:
            value=target.summer_MW.copy()
            for tech in ['wind','solar']:
                w=weights.loc[weights.year.eq(row.vintage)&weights.region.eq(row.region)&weights.technology.eq(tech),'weight'].item()
                mask=target.technology.eq(tech)
                value.loc[mask]=target.loc[mask,'nameplate_MW']*w
            value.loc[target.technology.eq('storage')]=0
            assert abs(value.sum()-row.observed_index_MW)<1e-6
    checks.append('all observed stock/index and persistence values independently reconstructed')
    for row in scores.itertuples(index=False):
        pool=predictions.copy()
        if row.scope=='short_horizon_supported_origins':
            pool=pool[(pool.vintage>=2022)&(pool.horizon<=2)]
        pool=pool[pool.model.eq(row.model)]
        if row.split!='overall':
            label=int(row.subgroup) if row.split in ['horizon','vintage'] else row.subgroup
            pool=pool[pool[row.split].eq(label)]
        p,o=('predicted_summer_MW','observed_summer_MW') if row.quantity=='physical_summer_stock' else ('predicted_index_MW','observed_index_MW')
        g=pool[pool[p].notna()&pool[o].notna()]
        assert len(g)==row.cases
        if not len(g):
            assert pd.isna(row.MAE_GW)
            continue
        assert abs(((g[p]-g[o]).abs()/1000).mean()-row.MAE_GW)<1e-9
    checks.append(f'all {len(scores)} score rows independently recalculated')
    cell_audit=[]
    for row in parameters.itertuples(index=False):
        train=cohorts[(cohorts.vintage<row.vintage)&(cohorts.target_year<=row.vintage)
            &cohorts.horizon.eq(row.horizon)&cohorts.origin_state.eq(row.origin_state)
            &cohorts.schedule_due.eq(row.schedule_due)]
        if row.fit_level=='technology_state_horizon_schedule':
            train=train[train.technology.eq(row.technology)]
        known=train[train.target_active.notna()&train.capacity_for_fit_MW.gt(0)]
        if pd.notna(row.probability):
            repeats=known.groupby(['unit_key','horizon','target_year']).unit_key.transform('size')
            w=known.capacity_for_fit_MW/repeats
            probability=float(np.clip((w*known.target_active).sum()/w.sum(),0,1))
            assert abs(probability-row.probability)<1e-10
        else:
            probability=np.nan
        cell_audit.append(dict(vintage=row.vintage,horizon=row.horizon,origin_state=row.origin_state,
            technology=row.technology,schedule_due=row.schedule_due,fit_level=row.fit_level,
            raw_pool_units=train.unit_key.nunique(),resolved_units=known.unit_key.nunique(),
            resolved_plants=known.plant_id.nunique(),resolved_origin_capacity_MW=known.capacity_for_fit_MW.sum(),
            independently_reconstructed_probability=probability))
    pd.DataFrame(cell_audit).to_csv(HERE/'31_resolved_training_cell_audit.csv',index=False)
    checks.append('all nonfallback delivery/survival frequencies independently reconstructed')
    assert flow.residual_MW.abs().max()<1e-6
    assert (parameters.max_training_target.dropna()<=parameters.loc[parameters.max_training_target.notna(),'vintage']).all()
    assert (parameters.max_training_origin.dropna()<parameters.loc[parameters.max_training_origin.notna(),'vintage']).all()
    checks.append('stock accounting and fit cutoffs checked')
    manifest=pd.read_csv(HERE/'17_preservation_manifest.csv')
    for row in manifest.itertuples(index=False):
        assert hashlib.sha256((HERE.parent/row.file).read_bytes()).hexdigest()==row.before_sha256==row.after_sha256
    checks.append(f'all {len(manifest)} earlier files preserved')
    (HERE/'verification.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
    print('\n'.join(checks))


if __name__=='__main__':
    main()
