"""Regression checks for reviewed load weights and raw-data preservation."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class CapacitySemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location("capacity_rules_tests",ROOT/"numerical/spatial/allocation_rules.py")
        cls.rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.rules)
        cls.rows=cls.rules.map_to_regions(cls.rules.harmonize_inventory(ROOT/"numerical/spatial/project_records.csv","latest_20260917"))
        cls.registry=pd.read_csv(ROOT/"numerical/spatial/capacity_attribute_review.csv")

    def test_review_does_not_mutate_raw_mw(self):
        before=self.rows.reported_mw_mid.copy(deep=True)
        self.rules.load_capacity_values(self.rows,"mid")
        pd.testing.assert_series_equal(before,self.rows.reported_mw_mid)
        self.assertEqual(self.rows.loc[self.rows.source_row.eq(941),"reported_mw_mid"].iloc[0],14000)

    def test_exclusions_cannot_return_through_size_proxy(self):
        _,subsets=self.rules.allocation_cases(self.rows)
        excluded=set(self.registry[self.registry.decision.eq("exclude")].source_row)
        for case,rows in subsets.items():
            if "site_count" not in case:self.assertFalse(set(rows.source_row)&excluded,case)
            else:self.assertIn(941,set(rows.source_row))

    def test_separate_load_values_replace_generation_values(self):
        reviewed=self.rules.load_capacity_values(self.rows,"mid")
        for row in self.registry[self.registry.decision.eq("replace")].itertuples():
            self.assertAlmostEqual(reviewed[self.rows.source_row.eq(row.source_row)].iloc[0],row.load_mw_mid)


if __name__=="__main__":unittest.main()
