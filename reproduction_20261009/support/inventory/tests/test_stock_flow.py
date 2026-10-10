import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('stock', Path(__file__).resolve().parents[1] / 'build_stock_flow.py')
stock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stock)


class StockRules(unittest.TestCase):
    def cc_frame(self):
        return pd.DataFrame(dict(snapshot_year=[2026]*3, plant_id=[1004]*3,
            unit_code=['0001']*3, prime_mover=['CT','CT','CA'],
            summer_MW=[np.nan,np.nan,555.0], nameplate_MW=[240.6,240.6,331.5],
            sheet=['Operating']*3, BA=['MISO']*3, status=['OP']*3,
            conflicting_sheets=[False]*3, region=['MISO']*3,
            technology=['gas']*3,
            plant_name=['Edwardsport']*3, unit_key=['1004_CT1','1004_CT2','1004_ST']))

    def test_shared_capacity_is_not_three_independent_ratings(self):
        original = self.cc_frame()
        result, audit = stock.identify_shared_capacity(original)
        self.assertEqual(result.shared_capacity_member.tolist(), [True,True,False])
        self.assertEqual(result.aggregate_capacity_carrier.tolist(), [False,False,True])
        self.assertEqual(result.summer_MW.sum(), 555)
        self.assertTrue(audit.identified_reporting_pattern.all())

    def test_unit_codes_do_not_join_different_plants(self):
        frame = self.cc_frame()
        frame.loc[2, 'plant_id'] = 9999
        result, _ = stock.identify_shared_capacity(frame)
        self.assertFalse(result.shared_capacity_member.any())

    def test_blank_unit_code_never_guesses_block(self):
        frame = self.cc_frame()
        frame['unit_code'] = ''
        result, _ = stock.identify_shared_capacity(frame)
        self.assertFalse(result.shared_capacity_member.any())

    def test_mixed_block_status_is_ambiguous_not_resolved(self):
        frame = self.cc_frame()
        frame.loc[0, 'status'] = 'SB'
        result, audit = stock.identify_shared_capacity(frame)
        self.assertFalse(result.shared_capacity_member.any())
        self.assertTrue(result.shared_capacity_ambiguous.all())
        self.assertFalse(audit.identified_reporting_pattern.any())

    def test_all_missing_block_has_no_reported_carrier(self):
        frame = self.cc_frame()
        frame['summer_MW'] = np.nan
        result, _ = stock.identify_shared_capacity(frame)
        self.assertFalse(result.shared_capacity_member.any())
        self.assertTrue(result.summer_MW.isna().all())

    def test_shared_hydro_uses_the_same_reporting_rule(self):
        frame = self.cc_frame()
        frame['technology'] = 'hydro'
        frame['prime_mover'] = 'HY'
        frame['summer_MW'] = [9.5, np.nan, np.nan]
        frame['nameplate_MW'] = [2.5, 2.2, 2.5]
        result, _ = stock.identify_shared_capacity(frame)
        self.assertEqual(result.shared_capacity_member.tolist(), [False,True,True])
        self.assertEqual(result.summer_MW.sum(), 9.5)

    def test_reported_zero_member_stays_an_explicit_zero_record(self):
        frame = self.cc_frame()
        frame.loc[1, 'summer_MW'] = 0.0
        result, audit = stock.identify_shared_capacity(frame)
        self.assertEqual(result.shared_capacity_member.tolist(), [True,False,False])
        self.assertEqual(audit.members.iloc[0], 3)
        self.assertEqual(result.summer_MW.sum(), 555)

    def test_missing_month_is_not_a_verified_due_date(self):
        actual = stock.due(pd.Series([2025, 2026, 2026, 2026, np.nan]),
                           pd.Series([np.nan, np.nan, 6, 7, np.nan]), 2026)
        self.assertEqual(actual.tolist(), [True, False, True, False, False])

    def test_active_means_OP_not_operating_sheet(self):
        frame = pd.DataFrame(dict(unit_key=['a', 'b', 'c', 'd', 'e', 'f'],
            sheet=['Operating']*5+['Retired'], status=['OP','SB','OA','OS','UNKNOWN','UNKNOWN'],
            conflicting_sheets=[False]*6))
        target = stock.resolve_target(frame)
        self.assertEqual(target.loc[['a','b','c','d','f'],'target_active'].tolist(),[1,0,0,0,0])
        self.assertTrue(pd.isna(target.loc['e','target_active']))

    def test_pumped_storage_is_storage_not_conventional_hydro(self):
        row=pd.Series({'Energy Source Code':'WAT','Prime Mover Code':'PS','Technology':'Hydroelectric Pumped Storage'})
        self.assertEqual(stock.technology(row),'storage')

    def test_conflicting_status_is_unknown(self):
        frame = pd.DataFrame(dict(unit_key=['a','a'],sheet=['Operating','Retired'],
                                 status=['OP','UNKNOWN'],conflicting_sheets=[True,True]))
        self.assertTrue(stock.resolve_target(frame).empty)

    def test_OP_on_Planned_sheet_is_not_a_zero_label(self):
        frame=pd.DataFrame(dict(unit_key=['a'],sheet=['Planned'],status=['OP'],conflicting_sheets=[False]))
        self.assertTrue(pd.isna(stock.resolve_target(frame).loc['a','target_active']))

    def test_unknown_does_not_lower_completion_probability(self):
        frame = pd.DataFrame(dict(unit_key=['a','b','c'],plant_id=[1,2,3],horizon=1,
            target_year=2020,capacity_for_fit_MW=100,target_active=[1,0,np.nan]))
        config = dict(minimum_cell_units=2,minimum_cell_plants=2,minimum_cell_capacity_MW=100)
        self.assertEqual(stock.cell_probability(frame,config), .5)

    def test_train_only_matured_earlier_origins(self):
        frame = pd.DataFrame(dict(vintage=[2019,2020,2020,2021],target_year=[2020,2021,2022,2022],
            horizon=[1,1,2,1],region='MISO',origin_state='OP',technology='gas',schedule_due=False,
            unit_key=['a','b','c','d'],plant_id=[1,2,3,4],target_active=[1,0,0,0],
            capacity_for_fit_MW=100,scheduled_probability=1))
        config = dict(minimum_cell_units=1,minimum_cell_plants=1,minimum_cell_capacity_MW=1)
        before, params = stock.predict_visible(frame,2021,config)
        modified = frame.copy()
        modified.loc[modified.target_year>2021,'target_active']=1
        after, _ = stock.predict_visible(modified,2021,config)
        self.assertEqual(before.learned_probability.tolist(),after.learned_probability.tolist())
        self.assertTrue((params.max_training_target.dropna()<=2021).all())
        self.assertTrue((params.max_training_origin.dropna()<2021).all())

    def test_capacity_fit_weight_is_origin_not_target_size(self):
        frame = pd.DataFrame(dict(unit_key=['a','b'],plant_id=[1,2],horizon=1,target_year=2020,
            capacity_for_fit_MW=[100,100],target_active=[1,0],target_summer_MW=[100000,0]))
        config = dict(minimum_cell_units=1,minimum_cell_plants=1,minimum_cell_capacity_MW=1)
        self.assertEqual(stock.cell_probability(frame,config), .5)

    def test_source_weights_are_origin_not_future(self):
        weights=pd.DataFrame(dict(year=[2020,2020,2021,2021],region='MISO',
            technology=['wind','solar','wind','solar'],weight=[.2,.5,.9,.9]))
        frame=pd.DataFrame(dict(region='MISO',technology=['wind','solar','storage','gas'],
            forecast_unit_MW=[100,100,100,80],nameplate_MW=[100,100,100,100]))
        values=stock.peak_unit_values(frame,weights,2020)
        self.assertEqual(values.tolist(),[20,50,0,80])

    def test_CAISO_has_no_automatic_WECC_weight(self):
        weights=pd.DataFrame(dict(year=[2020,2020],region='MISO',technology=['wind','solar'],weight=[.2,.5]))
        frame=pd.DataFrame(dict(region=['CAISO'],technology=['gas'],forecast_unit_MW=[100],nameplate_MW=[100]))
        self.assertTrue(stock.peak_unit_values(frame,weights,2020).isna().all())

    def test_heldout_region_labels_do_not_fit(self):
        frame=pd.DataFrame(dict(vintage=[2019,2019,2020],target_year=[2020,2020,2021],horizon=1,
            region=['MISO','PJM','PJM'],origin_state='OP',technology='gas',schedule_due=False,
            unit_key=['a','b','c'],plant_id=[1,2,3],target_active=[1,0,1],capacity_for_fit_MW=100,
            scheduled_probability=1))
        config=dict(minimum_cell_units=1,minimum_cell_plants=1,minimum_cell_capacity_MW=1)
        a,_=stock.predict_visible(frame,2020,config,heldout='PJM')
        frame.loc[frame.region.eq('PJM'),'target_active']=1
        b,_=stock.predict_visible(frame,2020,config,heldout='PJM')
        self.assertEqual(a.learned_probability.tolist(),b.learned_probability.tolist())


if __name__=='__main__':
    unittest.main()
