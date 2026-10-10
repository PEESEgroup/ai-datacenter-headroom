"""Regression comparisons must fail closed on keys, schema, text and numeric state."""
import unittest

import numpy as np
import pandas as pd

from run_frozen_outputs import compare_frames


class FrozenOutputTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"id": [1,2], "value": [1.,np.nan], "text": ["a","b"]})

    def check_failure(self, changed):
        with self.assertRaises((ValueError, AssertionError)):
            compare_frames(changed, self.reference, ["id"])

    def test_reordering_is_allowed(self):
        self.assertEqual(compare_frames(self.reference.iloc[::-1], self.reference, ["id"]), 0)

    def test_disjoint_equal_length_keys_fail(self):
        self.check_failure(self.reference.assign(id=[3,4]))

    def test_missing_and_extra_keys_fail(self):
        self.check_failure(self.reference.iloc[:1])
        self.check_failure(pd.concat([self.reference,self.reference.iloc[:1].assign(id=3)]))

    def test_null_and_duplicate_keys_fail(self):
        self.check_failure(self.reference.assign(id=[1,1]))
        self.check_failure(self.reference.assign(id=[1,np.nan]))

    def test_schema_and_text_fail(self):
        self.check_failure(self.reference.drop(columns="text"))
        self.check_failure(self.reference.assign(text=["a","changed"]))

    def test_changed_numeric_and_missing_states_fail(self):
        self.check_failure(self.reference.assign(value=[1.001,np.nan]))
        self.check_failure(self.reference.assign(value=[1.,0.]))

    def test_infinity_sign_and_finiteness_fail(self):
        reference = self.reference.assign(value=[np.inf,-np.inf])
        for values in ([np.inf,np.inf],[np.inf,0.]):
            with self.assertRaises(ValueError):
                compare_frames(reference.assign(value=values),reference,["id"])

    def test_roundoff_tolerance_only(self):
        self.assertLess(compare_frames(self.reference.assign(value=[1.+1e-12,np.nan]),self.reference,["id"]),1e-10)

    def test_empty_tables_require_schema(self):
        empty = self.reference.iloc[:0]
        self.assertEqual(compare_frames(empty,empty,["id"]),0)
        with self.assertRaises(ValueError):
            compare_frames(empty.drop(columns="text"),empty,["id"])


if __name__ == "__main__":
    unittest.main()
