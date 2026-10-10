"""Recompute retained SI annual physical sensitivities, not new peak credit."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'numerical/intervention'
EXPECTED=ROOT/'source_data/restored_support_20261004'


def main():
    spec=importlib.util.spec_from_file_location('physics',DATA/'physics.py')
    physics=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    config=json.loads((DATA/'physics_config.json').read_text())
    sites=pd.read_csv(DATA/'sites.csv')
    architecture=pd.read_csv(EXPECTED/'42_pv_storage_architecture_results.csv')
    efficiency=pd.read_csv(EXPECTED/'42_pv_storage_efficiency_sensitivity.csv')
    requested=pd.concat([architecture,efficiency],ignore_index=True)
    requested=requested[requested.pv_nameplate_per_mw_load.eq(2)&requested.storage_duration_h.eq(4)]
    results=[]
    checks=[]
    for region,region_sites in sites.groupby('ISO'):
        arrays=[]
        for row in region_sites.itertuples(index=False):
            weather=pd.read_csv(DATA/'hourly_inputs'/(row.representative_id+'.csv.gz'))
            timestamps=pd.DatetimeIndex(pd.to_datetime(weather.timestamp_utc,utc=True))
            converted=physics.solar_geometry_and_pv(timestamps,row.latitude,row.longitude,
                weather.ghi_w_m2.to_numpy(),weather.air_temperature_c.to_numpy(),weather.wind_speed_m_s.to_numpy(),config)
            arrays.append(converted.pv_ac_kw_per_kwdc.to_numpy())
        arrays=np.array(arrays)
        primary=[]
        for row in requested[requested.ISO.eq(region)].itertuples(index=False):
            pooled=row.architecture=='pooled_regional_storage_sensitivity'
            inputs=[arrays.mean(axis=0)] if pooled else arrays
            dispatch=[physics.simulate_storage(pv,2,4,row.round_trip_efficiency,config) for pv in inputs]
            residual=np.mean([item['residual'] for item in dispatch],axis=0)
            curtailed=np.mean([item['curtailed'] for item in dispatch],axis=0)
            loss=np.mean([item['charged_from_pv'].sum()-item['discharged_to_load'].sum()
                -(item['analysis_end_soc']-item['analysis_start_soc']) for item in dispatch])/len(residual)
            results.append(dict(ISO=region,architecture=row.architecture,
                round_trip_efficiency=row.round_trip_efficiency,annual_energy_displacement_share=1-residual.mean(),
                pv_curtailment_share=curtailed.sum()/(2*arrays.mean(axis=0)).sum(),
                maximum_residual_mw_per_mw_load=residual.max(),battery_conversion_loss_share=loss))
            assert abs(1-residual.mean()-row.annual_energy_displacement_share)<1e-8
            assert abs(residual.max()-row.maximum_residual_mw_per_mw_load)<1e-8
            if hasattr(row,'battery_conversion_loss_share') and pd.notna(row.battery_conversion_loss_share):
                assert abs(loss-row.battery_conversion_loss_share)<1e-8
            if not pooled and abs(row.round_trip_efficiency-.85)<1e-9:
                primary=[item['residual'] for item in dispatch]
        assert len(primary)==5
        leaves=[1-np.delete(np.array(primary),index,axis=0).mean() for index in range(5)]
        old=pd.read_csv(EXPECTED/'42_pv_storage_leave_one_site_out.csv')
        old=old[old.ISO.eq(region)&old.pv_nameplate_per_mw_load.eq(2)&old.storage_duration_h.eq(4)]
        assert np.allclose(sorted(leaves),sorted(old.annual_energy_displacement_share),atol=1e-8,rtol=0)
        differences=[]
        for pv in arrays:
            a=physics.simulate_storage(pv,2,24,.85,config,warmup_days=30)['residual']
            b=physics.simulate_storage(pv,2,24,.85,config,warmup_days=60)['residual']
            differences.append(float(np.max(np.abs(a-b))))
        assert max(differences)<1e-10
        checks.append(dict(region=region,leave_one_site_out_passed=True,warmup_maximum_difference=max(differences)))
    gas=pd.read_csv(EXPECTED/'gas_utilization_consequences_per_credited_GW.csv')
    nameplate=gas.assumed_accredited_contribution_GW/gas.assumed_accreditation
    generation=8.76*nameplate*gas.capacity_factor
    co2=generation*1e6*gas.heat_rate_MMBtu_per_MWh*53.06/1e9
    methane=generation*1e6*gas.heat_rate_MMBtu_per_MWh*19.294545*gas.methane_leakage_fraction/(1-gas.methane_leakage_fraction)*29.8/1e9
    for actual,expected in [(nameplate,gas.nameplate_GW),(generation,gas.generation_TWh),
                           (co2,gas.combustion_CO2_Mt),(methane,gas.upstream_CH4_CO2e_Mt),(co2+methane,gas.CO2e_Mt)]:
        assert np.allclose(actual,expected,atol=1e-8,rtol=0)
    output=HERE/'physical_results'
    output.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(output/'annual_PV_sensitivities.csv',index=False)
    (output/'verification.json').write_text(json.dumps(dict(passed=True,checks=checks,
        gas_rows=len(gas),PV_cases=len(results),scope='annual physical sensitivities only; no new peak or accreditation result'),indent=2)+'\n')
    print(json.dumps({'passed':True,'PV_cases':len(results),'gas_rows':len(gas)},indent=2))


if __name__=='__main__':
    main()
