"""Focused checks for source reconciliation and affine annual conditions."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_analysis import HERE, summer_forecast, minimum_growth_limit, required_fraction, envelope_breakpoints


class PlanningConditionsTests(unittest.TestCase):
    def test_all_year_limit_is_not_endpoint_only(self):
        value,index=minimum_growth_limit(np.array([1.,3.]),np.array([2.,2.]))
        self.assertEqual((value,index),(0.5,0))

    def test_negative_background_is_infeasible_not_zero_capacity(self):
        value,index=minimum_growth_limit(np.array([-1.,3.]),np.array([2.,2.]))
        self.assertTrue(np.isnan(value))
        self.assertEqual(index,-1)

    def test_no_pressure_has_no_finite_growth_bound(self):
        value,index=minimum_growth_limit(np.array([1.,3.]),np.zeros(2))
        self.assertTrue(np.isinf(value))

    def test_zero_requirement_has_no_binding_year(self):
        self.assertEqual(required_fraction(np.array([-1.,0.]),np.array([0.,2.])),(0.,-1))

    def test_no_available_uplift_can_be_infeasible(self):
        value,_=required_fraction(np.array([1.,1.]),np.array([0.,2.]))
        self.assertTrue(np.isinf(value))

    def test_exact_binding_year_switch(self):
        segments=envelope_breakpoints(np.array([0.,0.5]),np.array([2.,1.]),np.array([2034,2035]),"minimum")
        self.assertEqual([s["binding_year"] for s in segments],[2034,2035])
        self.assertAlmostEqual(segments[0]["resource_share_end"],0.5)

    def test_PJM_workbook_matches_native_baseline(self):
        old=summer_forecast(HERE/"sources/2025-load-report-tables.xlsx")
        n=pd.read_csv(HERE/"inputs/native_inputs.csv")
        n=n[n.planning_region.eq("PJM")].sort_values("year")
        np.testing.assert_array_equal(old.gross_MW,n.gross_peak_demand_MW)

    def test_provider_revision_anchors(self):
        a=summer_forecast(HERE/"sources/2025-load-report-tables.xlsx").set_index("year")
        b=summer_forecast(HERE/"sources/2026-load-report-tables.xlsx").set_index("year")
        self.assertEqual(b.loc[2028,"gross_MW"]-a.loc[2028,"gross_MW"],-4414)
        self.assertEqual(b.loc[2035,"gross_MW"]-a.loc[2035,"gross_MW"],6949)

    def test_ERCOT_response_is_not_credited_twice(self):
        n=pd.read_csv(HERE/"inputs/native_inputs.csv")
        n=n[(n.planning_region=="ERCOT")&(n.year==2035)].iloc[0]
        k=1+n.reference_margin_level_pct/100
        baseline=(n.anticipated_native_MW-k*(n.gross_peak_demand_MW-n.ordinary_response_MW))/1000
        with_credit=baseline+k*n.provisional_large_load_response_MW/1000
        direct=(n.anticipated_native_MW-k*n.published_net_demand_MW)/1000
        self.assertAlmostEqual(with_credit,direct,places=10)
        self.assertAlmostEqual(-baseline/k,51.15412747252748,places=9)

    def test_control_fraction_is_common_across_years(self):
        need=np.array([2.,6.])
        available=np.array([4.,8.])
        value,index=required_fraction(need,available)
        self.assertEqual((value,index),(0.75,1))
        self.assertTrue(np.all(-need+value*available>=0))


if __name__=="__main__":
    unittest.main(verbosity=2)
