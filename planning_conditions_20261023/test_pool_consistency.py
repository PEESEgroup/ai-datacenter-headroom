"""Independent, focused tests of the revised ERCOT paired construction."""
import unittest
import numpy as np
import pandas as pd
import run_pool_consistency as p

class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool=pd.read_csv(p.HERE/"pool_inputs/ercot_large_load_source.csv")
        cls.native=pd.read_csv(p.HERE/"pool_inputs/native_conditions.csv")
        candidates=pd.read_csv(p.HERE/"pool_inputs/candidate_paths.csv.gz")
        cls.c=p.display(candidates)
        cls.a=p.evaluate(cls.c,cls.native,cls.pool)

    def test_exact_source_bound(self):
        b=p.association_bounds(self.pool,"linear").set_index("year")
        self.assertAlmostEqual(b.loc[2035,"minimum_removed_GW"],13.9017)
        self.assertAlmostEqual(b.loc[2030,"minimum_removed_GW"],15.2106)
        self.assertAlmostEqual(b.common_minimum_fraction.iloc[0],15.2106/23)
        self.assertEqual(b.binding_year.iloc[0],2030)

    def test_both_sides_for_all_timing_rules(self):
        for rule in p.RULES:
            a=p.evaluate(self.c,self.native,self.pool,rule)
            e=a[a.planning_region.eq("ERCOT")&a.response_condition.eq("published_all")]
            self.assertGreaterEqual(e.background_pool_slack_GW.min(),-p.TOL)
            self.assertGreaterEqual(e.candidate_pool_slack_GW.min(),-p.TOL)
            self.assertGreaterEqual(e.fixed_budget_pool_slack_GW.min(),-p.TOL)

    def test_minimum_is_necessary_for_common_fraction(self):
        for rule in p.RULES:
            b=p.association_bounds(self.pool,rule)
            chi=b.common_minimum_fraction.iloc[0]-1e-6
            slack=b.large_load_GW-b.official_GW-b.provisional_GW+chi*b.available_GW
            self.assertLess(slack.min(),-p.TOL)

    def test_legacy_zero_and_half_rejected(self):
        b=p.association_bounds(self.pool,"linear")
        for fraction in [0,.5]:
            self.assertLess((b.large_load_GW-b.official_GW-b.provisional_GW+fraction*b.available_GW).min(),0)

    def test_source_anchors_preserved(self):
        for rule in p.RULES:
            np.testing.assert_allclose(p.source_path("ERCOT",rule,[2025,2027,2030,2035]),[0,6.7,23,25])
            np.testing.assert_allclose(p.source_path("MISO",rule,[2025,2027,2030,2035]),[0,6,14,18])

    def test_miso_matches_previous_display(self):
        old=pd.read_csv(p.HERE/"inputs/miso_display_timing_paths.csv")
        new=self.a[self.a.planning_region.eq("MISO")&self.a.resource_condition.eq("anticipated")
            &self.a.response_condition.eq("published_all")&self.a.association_grid_position.eq(0)].sort_values("year")
        old=old[old.source_DC_timing_rule.eq("linear")].sort_values("year")
        np.testing.assert_allclose(new.background_GW,old.background_balance_GW,atol=p.TOL,rtol=0)
        np.testing.assert_allclose(new.balance_GW,old.candidate_balance_GW,atol=p.TOL,rtol=0)

    def test_main_formula_independently(self):
        for r in self.a.itertuples():
            k=1+r.reference_margin_level_pct/100
            b0=r.margin_GW+k*(r.official_gross_path_GW-r.official_linked_response_GW)
            bs=b0-k*(r.candidate_GW-r.candidate_linked_response_GW)
            self.assertAlmostEqual(r.balance_GW,bs)
            self.assertAlmostEqual(r.added_requirement_GW,max(0,-bs)-max(0,-b0))

    def test_fixed_budget_never_adds_credit(self):
        a=self.a
        self.assertTrue((a.fixed_budget_response_GW<=a.official_linked_response_GW+p.TOL).all())
        self.assertTrue((a.fixed_budget_balance_GW<=a.balance_GW+p.TOL).all())

    def test_unique_ten_year_paths(self):
        a=self.a
        self.assertFalse(a.duplicated(p.KEYS+["year"]).any())
        self.assertTrue(a.groupby(p.KEYS).size().eq(10).all())

    def test_full_association_has_zero_incremental_pressure_ercot(self):
        a=self.a
        a=a[a.planning_region.eq("ERCOT")&a.response_condition.eq("published_all")&a.association_grid_position.eq(1)]
        np.testing.assert_allclose(a.pressure_GW,0,atol=p.TOL,rtol=0)

    def test_current_only_keeps_all_candidate_load(self):
        a=self.a[self.a.planning_region.eq("ERCOT")&self.a.response_condition.eq("current_cap")]
        np.testing.assert_allclose(a.candidate_linked_response_GW,0,atol=p.TOL,rtol=0)
        np.testing.assert_allclose(a.pressure_GW,(1+a.reference_margin_level_pct/100)*a.candidate_GW,atol=p.TOL,rtol=0)

    def test_feasibility_preserved_when_candidate_scaled(self):
        e=self.a[self.a.planning_region.eq("ERCOT")&self.a.response_condition.eq("published_all")]
        for scale in [0,.5,1,2,10]:
            slack=e.background_pool_slack_GW+scale*e.candidate_GW*(1-e.response_per_incremental_GW)
            self.assertGreaterEqual(slack.min(),-p.TOL)

if __name__=="__main__": unittest.main()
