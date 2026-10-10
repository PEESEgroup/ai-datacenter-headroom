"""Rebuild retained SI components without changing the annual scenario model."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def compare(relative, keys, columns, tolerance=1e-8):
    actual = pd.read_csv(HERE/relative, float_precision='round_trip')
    expected = pd.read_csv(HERE/'expected'/relative.replace('/results/','/'), float_precision='round_trip')
    for frame in [actual, expected]:
        assert not frame[keys].isna().any().any(), relative
        assert not frame.duplicated(keys).any(), relative
    joined = actual.merge(expected, on=keys, how='outer', suffixes=('_run','_frozen'),
                          indicator=True, validate='one_to_one')
    assert joined._merge.eq('both').all() and len(joined)>0, relative
    for column in columns:
        a,b = joined[column+'_run'],joined[column+'_frozen']
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.allclose(a,b,atol=tolerance,rtol=0,equal_nan=True), (relative,column)
        else:
            assert a.fillna('<NA>').equals(b.fillna('<NA>')), (relative,column)
    return {'file':relative,'rows':len(joined),'keys':keys,'columns':columns,'passed':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component',choices=['all','price','inventory'],default='all')
    args=parser.parse_args()
    runs=[]
    commands=[]
    if args.component in ['all','price']:
        commands.append(['price/build_price_fixed_effects.py'])
    if args.component in ['all','inventory']:
        commands.extend([['-m','unittest','discover','-s','inventory/tests'],
                         ['inventory/build_stock_flow.py'], ['inventory/verify_stock_flow.py'],
                         ['inventory/diagnose_stock_flow.py'], ['inventory/run_mechanism.py']])
    record={'status':'running','passed':False,'runs':runs,'comparisons':[]}
    destination=HERE/'support_verification.json'
    destination.write_text(json.dumps(record,indent=2)+'\n')
    try:
        for command in commands:
            start=time.perf_counter()
            result=subprocess.run([sys.executable,'-B',*command],cwd=HERE,check=False)
            runs.append({'command':command,'seconds':time.perf_counter()-start,'exit_code':result.returncode})
            destination.write_text(json.dumps(record,indent=2)+'\n')
            if result.returncode:
                raise RuntimeError('Support command failed: '+str(command))
        comparisons=[]
        if args.component in ['all','price']:
            comparisons.extend([
              compare('price/results/04_price_zone_timestamp_fe_slopes.csv',['ISO','zone'],
                      ['coefficient_dollars_per_mwh_per_within_zone_year_load_sd','clustered_standard_error','observations','p_value']),
              compare('price/results/04_price_fe_robustness.csv',['ISO','sample'],
                      ['coefficient_dollars_per_mwh_per_within_zone_year_load_sd','clustered_standard_error','observations','p_value']),
              compare('price/results/04_price_fe_region_summary.csv',['ISO'],
                      ['zones','median_conditional_slope','positive_zone_share','positive_significant_zone_share'])])
        if args.component in ['all','inventory']:
            comparisons.extend([
              compare('inventory/08_predictions.csv',['vintage','target_year','region','model'],
                      ['predicted_summer_MW','observed_summer_MW','predicted_index_MW','observed_index_MW']),
              compare('inventory/11_metrics.csv',['scope','model','quantity','split','subgroup'],
                      ['cases','MAE_GW','RMSE_GW','bias_GW']),
              compare('inventory/19_true_region_heldout_predictions.csv',['vintage','target_year','region','model'],
                      ['predicted_summer_MW','observed_summer_MW','predicted_index_MW','observed_index_MW']),
              compare('inventory/21_true_region_heldout_metrics.csv',['scope','model','quantity','split','subgroup'],
                      ['cases','MAE_GW','RMSE_GW','bias_GW']),
              compare('inventory/mechanism/01_all_supply_predictions.csv',['vintage','target_year','region','model'],
                      ['predicted_summer_MW','observed_summer_MW','predicted_index_MW','observed_index_MW']),
              compare('inventory/mechanism/02_supply_scores.csv',['scope','model','quantity','split','subgroup'],
                      ['cases','MAE_GW','RMSE_GW','bias_GW'])])
        record.update(status='passed',passed=True,comparisons=comparisons)
    except Exception as error:
        record.update(status='failed',passed=False,error=repr(error))
        raise
    finally:
        destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
