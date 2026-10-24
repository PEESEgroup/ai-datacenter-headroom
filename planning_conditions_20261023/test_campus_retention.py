"""Regression and accounting checks for the current campus-retention sensitivity."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

import run_campus_retention as c
import run_pool_consistency as p


class CampusRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights, candidates, native, source = c.inputs()
        cls.annual, cls.complete, cls.primary = c.calculate(cls.weights, candidates, native, source)

    def test_weights_and_all_keep_identity(self):
        w = self.weights
        self.assertEqual(len(w), 8)
        self.assertTrue((w.retained_region_weight_MW > 0).all())
        both = w[w.keep_matrix.eq(1) & w.keep_frontier.eq(1)]
        np.testing.assert_allclose(both.weight_scale, 1, atol=p.TOL, rtol=0)
        np.testing.assert_allclose(w.weight_scale,
            w.retained_region_weight_MW / w.retained_seven_region_weight_MW
            / (w.reference_region_weight_MW / w.reference_seven_region_weight_MW), atol=p.TOL, rtol=0)

    def test_both_retained_match_current_annual_pairs(self):
        new = self.annual[self.annual.keep_matrix.eq(1) & self.annual.keep_frontier.eq(1)]
        old = pd.read_csv(c.HERE / "pool_results/displayed_pairs.csv")
        old = old[old.resource_condition.isin(["anticipated", "prospective"])]
        keys = p.KEYS + ["year"]
        new, old = new.sort_values(keys), old.sort_values(keys)
        for name in ["candidate_GW", "background_GW", "pressure_GW", "balance_GW", "added_requirement_GW"]:
            np.testing.assert_allclose(new[name], old[name], atol=p.TOL, rtol=0)

    def test_unaffected_conditions_retain_legacy_values(self):
        old = pd.read_csv(c.HERE.parent / "reproduction_20261009/default_weight_update/remaining_campus_retention_cases.csv")
        keys = c.CASE + ["region", "resources", "response"]
        new = self.primary.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        new = new[new.region.eq("MISO") | new.response.eq("current_cap")]
        for name in ["pressure_GW", "balance_GW", "added_GW"]:
            np.testing.assert_allclose(new[name + "_new"], new[name + "_old"], atol=p.TOL, rtol=0)

    def test_pool_bounds_and_paired_identity(self):
        a = self.annual
        np.testing.assert_allclose(a.background_GW - a.balance_GW, a.pressure_GW, atol=p.TOL, rtol=0)
        np.testing.assert_allclose(a.added_requirement_GW,
            np.maximum(-a.balance_GW, 0) - np.maximum(-a.background_GW, 0), atol=p.TOL, rtol=0)
        self.assertGreaterEqual(float(a.background_pool_slack_GW.min()), -p.TOL)
        self.assertGreaterEqual(float(a.candidate_pool_slack_GW.min()), -p.TOL)

    def test_ERCOT_current_endpoint_values(self):
        a = self.primary
        a = a[a.region.eq("ERCOT") & a.resources.eq("anticipated") & a.response.eq("published_all")]
        a = a.sort_values(c.CASE)
        np.testing.assert_allclose(a.balance_GW,
            [13.522047821828634, 13.379046983224839, 13.218876289206719, 13.081495788316587],
            atol=p.TOL, rtol=0)
        np.testing.assert_allclose(a.linked_response_fraction, .6613304347826087, atol=p.TOL, rtol=0)
        self.assertTrue(a.added_GW.eq(0).all())

    def test_one_weight_scale_applies_to_whole_path(self):
        for _, frame in self.annual.groupby(c.CASE + ["planning_region"]):
            self.assertEqual(frame.weight_scale.nunique(), 1)
        self.assertTrue(self.annual.groupby(c.CASE + p.KEYS).size().eq(10).all())

    def test_complete_settings_and_association_bounds(self):
        self.assertEqual((len(self.annual), len(self.complete), len(self.primary)), (960, 96, 32))
        a = self.annual
        applicable = a[a.planning_region.eq("ERCOT") & a.response_condition.eq("published_all")]
        self.assertTrue((applicable.linked_response_fraction >= applicable.minimum_association_fraction - p.TOL).all())
        self.assertEqual(sorted(a.association_grid_position.unique().tolist()), [0, .5, 1])

    def test_first_run_and_rerun_are_identical(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            out = Path(directory)
            c.main(out)
            first = {p.name: p.read_bytes() for p in out.iterdir()}
            c.main(out)
            self.assertEqual(first, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
