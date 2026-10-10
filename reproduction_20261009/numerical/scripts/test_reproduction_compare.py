"""Regression tests for complete-key numerical verification."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from run_review_package import compare
from run_timing_sensitivity import source_path


class ReproductionChecks(unittest.TestCase):
    def check_tables(self, old, new, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            expected, actual = Path(directory) / "expected.csv", Path(directory) / "actual.csv"
            pd.DataFrame(old).to_csv(expected, index=False)
            pd.DataFrame(new).to_csv(actual, index=False)
            if fail:
                with self.assertRaises(AssertionError):
                    compare(expected, actual, ["id"], ["value"])
            else:
                compare(expected, actual, ["id"], ["value"])

    def test_disjoint_equal_size_does_not_pass(self):
        self.check_tables({"id": [1], "value": [123]}, {"id": [2], "value": [999]}, True)

    def test_missing_and_extra_keys_do_not_cancel(self):
        self.check_tables({"id": [1, 2], "value": [1, 2]}, {"id": [1, 3], "value": [1, 2]}, True)

    def test_row_order_is_irrelevant(self):
        self.check_tables({"id": [1, 2], "value": [1, 2]}, {"id": [2, 1], "value": [2, 1]})

    def test_duplicate_keys_rejected(self):
        self.check_tables({"id": [1, 2], "value": [1, 2]}, {"id": [1, 1], "value": [1, 2]}, True)

    def test_null_keys_rejected(self):
        self.check_tables({"id": [1, 2], "value": [1, 2]}, {"id": [1, None], "value": [1, 2]}, True)

    def test_numerical_difference_rejected(self):
        self.check_tables({"id": [1], "value": [123]}, {"id": [1], "value": [999]}, True)

    def test_row_count_difference_rejected(self):
        self.check_tables({"id": [1, 2], "value": [1, 2]}, {"id": [1], "value": [1]}, True)

    def test_anchor_rules_preserve_values_and_monotonicity(self):
        import numpy as np
        from run_timing_sensitivity import CONFIG
        for region, anchors in CONFIG["official_DC_anchors_GW"].items():
            for rule in CONFIG["timing_rules"]:
                years = np.arange(2025, 2036)
                values = source_path(region, rule, years)
                self.assertTrue((np.diff(values) >= 0).all())
                for year, value in anchors.items():
                    self.assertAlmostEqual(values[int(year) - 2025], value)


if __name__ == "__main__":
    unittest.main()
