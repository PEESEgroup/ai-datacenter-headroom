"""Focused checks of paired realization, with direct load-response reconstruction."""
import unittest
import numpy as np
import pandas as pd
import run_paired_realization as p


class PairedRealizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs, cls.native, cls.protocol = p.inputs()
        cls.groups = [(key, frame.sort_values("year")) for key, frame in
                      cls.pairs.groupby(["source_DC_timing_rule", "association_grid_position"])]

    def test_retains_full_credit_pairs(self):
        old = pd.read_csv(p.HERE/"pool_results/displayed_pairs.csv")
        old = old[old.planning_region.eq("ERCOT") & old.resource_condition.eq("anticipated")
                  & old.response_condition.eq("published_all")].sort_values(["association_grid_position","year"])
        new = self.pairs[self.pairs.source_DC_timing_rule.eq("linear")].sort_values(["association_grid_position","year"])
        np.testing.assert_allclose(new.background_GW, old.background_GW, atol=p.TOL, rtol=0)
        np.testing.assert_allclose(new.balance_GW, old.balance_GW, atol=p.TOL, rtol=0)
        for _, frame in self.groups:
            v = p.vectors(frame, self.native)
            b0, bs, _, _ = p.paired_balances(v, 0, 1)
            np.testing.assert_allclose(b0, frame.background_GW, atol=p.TOL, rtol=0)
            np.testing.assert_allclose(bs, frame.balance_GW, atol=p.TOL, rtol=0)

    def test_direct_reconstruction_of_both_worlds(self):
        for _, frame in self.groups:
            v = p.vectors(frame, self.native)
            for u, zeta in [(0,.7), (.25,0), (.5,.8), (1,1)]:
                b0, bs, _, _ = p.paired_balances(v, u, zeta)
                r = v["resource"]+u*v["uplift"]
                np.testing.assert_allclose(b0, r-v["k"]*(v["gross0"]-v["ordinary"]-zeta*v["c0"]), atol=p.TOL, rtol=0)
                np.testing.assert_allclose(bs, r-v["k"]*(v["gross_s"]-v["ordinary"]-zeta*v["cs"]), atol=p.TOL, rtol=0)

    def test_nominal_association_and_pools_stay_fixed(self):
        for _, frame in self.groups:
            self.assertEqual(frame.linked_response_fraction.nunique(), 1)
            v = p.vectors(frame, self.native)
            for zeta in [0,.5,.7,.8,1]:
                self.assertTrue((zeta*v["c0"] <= v["c0"]+p.TOL).all())
                self.assertTrue((zeta*v["cs"] <= v["cs"]+p.TOL).all())

    def test_added_requirement_preserves_background_gap(self):
        for _, frame in self.groups:
            v = p.vectors(frame, self.native)
            for zeta in [0,.5,.7,.8,1]:
                b0, bs, pressure, added = p.paired_balances(v, 0, zeta)
                np.testing.assert_allclose(added, np.maximum(-bs,0)-np.maximum(-b0,0), atol=p.TOL, rtol=0)
                self.assertGreaterEqual(float(added.min()), -p.TOL)
                self.assertTrue((added <= pressure+p.TOL).all())

    def test_threshold_sufficiency_and_necessity(self):
        for _, frame in self.groups:
            v = p.vectors(frame, self.native)
            for u in [0,.25,.5,1]:
                for full, credit in [(v["full0"],v["c0"]),(v["full_s"],v["cs"])]:
                    d = p.response_limit(full, credit, v["k"], v["uplift"], v["years"], u)
                    z = d["all_year_fraction"]
                    self.assertGreaterEqual(float((full+u*v["uplift"]-v["k"]*(1-z)*credit).min()), -p.TOL)
                    if z > 1e-6:
                        self.assertLess(float((full+u*v["uplift"]-v["k"]*(1-z+1e-6)*credit).min()), -p.TOL)
                        self.assertEqual(d["binding_year"], 2035)

    def test_selected_full_path_values(self):
        frame = dict(self.groups)[("linear",0.0)]
        v = p.vectors(frame,self.native)
        for u, expected0, expected_s in [(0,.6252131249696062,.7699535297399912),
                                        (.25,.1092425769743235,.33818932376512584)]:
            for full, credit, expected in [(v["full0"],v["c0"],expected0),(v["full_s"],v["cs"],expected_s)]:
                result = p.response_limit(full,credit,v["k"],v["uplift"],v["years"],u)
                self.assertAlmostEqual(result["all_year_fraction"],expected)
                self.assertEqual(result["binding_year"],2035)
        b0,bs,_,added = p.paired_balances(v,0,.7)
        self.assertTrue((b0 >= 0).all())
        self.assertEqual(v["years"][bs < -p.TOL].tolist(),[2034,2035])
        np.testing.assert_allclose(added[-2:],[2.1048107284383732,3.9778780506273037],atol=p.TOL,rtol=0)

    def test_zero_realization_is_not_current_only(self):
        frame = dict(self.groups)[("linear",0.0)]
        v = p.vectors(frame,self.native)
        b0,bs,_,_ = p.paired_balances(v,0,0)
        current = pd.read_csv(p.HERE/"pool_results/displayed_pairs.csv")
        current = current[current.planning_region.eq("ERCOT") & current.resource_condition.eq("anticipated")
                          & current.response_condition.eq("current_cap") & current.association_grid_position.eq(0)].sort_values("year")
        ordinary_gain = (self.native.ordinary_response_MW-self.native.current_cap_response_MW).to_numpy()/1000
        np.testing.assert_allclose(bs-current.balance_GW.to_numpy(),v["k"]*ordinary_gain,atol=p.TOL,rtol=0)
        self.assertGreater(ordinary_gain[-1],0)

    def test_monotone_in_resources_and_realization(self):
        for _, frame in self.groups:
            v = p.vectors(frame,self.native)
            a = p.paired_balances(v,0,.5)
            b = p.paired_balances(v,.25,.5)
            c = p.paired_balances(v,0,.8)
            for i in [0,1]:
                self.assertTrue((b[i]>=a[i]-p.TOL).all())
                self.assertTrue((c[i]>=a[i]-p.TOL).all())

    def test_complete_setting_keys_and_ten_years(self):
        annual,complete,thresholds = p.calculate(self.pairs,self.native,self.protocol)
        keys=["source_timing_rule","association_grid_position","resource_share","realization_fraction"]
        self.assertFalse(annual.duplicated(keys+["year"]).any())
        self.assertTrue(annual.groupby(keys).size().eq(10).all())
        self.assertEqual((len(annual),len(complete),len(thresholds)),(2400,240,3636))
        self.assertEqual(sorted(thresholds.loc[thresholds.all_year_fraction.gt(p.TOL),"binding_year"].unique().tolist()),[2035])

    def test_zero_credit_edge_cases(self):
        v = np.array([0.0])
        result = p.response_limit(np.array([1.0]),v,np.ones(1),v,np.array([2035]),0)
        self.assertEqual(result["all_year_fraction"],0)
        self.assertEqual(result["binding_year"],0)
        result = p.response_limit(np.array([-1.0]),v,np.ones(1),v,np.array([2035]),0)
        self.assertFalse(result["feasible_fraction_le_one"])
        self.assertTrue(np.isinf(result["all_year_fraction"]))


if __name__ == "__main__":
    unittest.main()
