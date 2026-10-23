"""Check that the documented rule ledger describes the retained calculation."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "numerical/spatial"))
sys.path.insert(0, str(ROOT / "numerical/scripts"))
import allocation_rules as rules
from scope_quality import classify_scope, campus_keys, campus_allocation_cases


class FrameworkRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((ROOT / "FRAMEWORK_RULES.json").read_text())
        cls.rows = rules.map_to_regions(rules.harmonize_inventory(
            ROOT / "numerical/spatial/project_records.csv", "latest_20260917"))

    def test_uncertain_record_not_automatically_excluded(self):
        joule = self.rows[self.rows.source_row.eq(1152)]
        self.assertEqual(len(joule), 1)
        self.assertFalse(rules.generation_scale_mask(joule).any())
        self.assertAlmostEqual(rules.load_capacity_values(joule, "mid").iloc[0], 6227.5)
        self.assertEqual(self.policy["input_decisions"]["additional_project_exclusions_in_this_alignment"], [])

    def test_existing_generation_corrections_and_count_retention(self):
        excluded = set(self.rows.loc[rules.generation_scale_mask(self.rows), "source_row"])
        self.assertEqual(excluded, {941, 835, 891, 1667, 1673})
        _, subsets = rules.allocation_cases(self.rows)
        for case, chosen in subsets.items():
            selected = set(chosen.source_row)
            if "site_count" in case:
                self.assertTrue(excluded <= selected)
            else:
                self.assertFalse(excluded & selected)

    def test_documented_broader_inventory_scope(self):
        rows = classify_scope(self.rows)
        core = rows[~rows.explicit_non_data_center & rows.status.isin(rules.CORE_ACTIVE)]
        pure = core[core.scope_class.eq("explicit_pure_crypto")]
        inside = pure[pure.in_seven_region_footprint]
        outside = pure[~pure.in_seven_region_footprint]
        self.assertEqual(len(inside), 3)
        self.assertEqual(inside.reported_mw_mid.sum(), 0)
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside.reported_mw_mid.sum(), 30)

    def test_scope_families_are_fixed_capture_comparisons(self):
        rows = classify_scope(self.rows)
        rows["campus_key"], _ = campus_keys(rows, rules.normalize_text)
        pure = ~rows.scope_class.eq("explicit_pure_crypto")
        non_crypto = rows.scope_class.eq("no_explicit_crypto_tag")
        cases = {
            "exclude_pure_crypto": pure,
            "exclude_pure_and_mixed_crypto": non_crypto,
            "shared_address_single_representative": pure,
            "legacy_generation_scale_retained": pure,
            "combined_scope_sensitivity": non_crypto & ~rows.facility_name.fillna("").str.contains("Fermi America", case=False),
        }
        self.assertEqual(len(cases), 5)
        self.assertFalse(self.policy["quantities"]["candidate_increment"]["county_display_is_an_input"])
        self.assertFalse(self.policy["pv_storage"]["regional_capacity_credit_calculated"])
        for case, eligible in cases.items():
            if case in {"shared_address_single_representative", "combined_scope_sensitivity"}:
                weights, _ = campus_allocation_cases(rows[eligible].copy(), rules)
            else:
                weights, _ = rules.allocation_cases(rows[eligible].copy(),
                    include_generation_scale=case == "legacy_generation_scale_retained")
            for _, group in weights.groupby("allocation_case"):
                self.assertAlmostEqual(group.allocation_share.sum(), 1)


if __name__ == "__main__":
    unittest.main()
