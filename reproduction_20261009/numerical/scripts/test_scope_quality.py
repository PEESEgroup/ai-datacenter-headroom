"""Regress the exact full-record and campus eligibility defects found in review."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np
import pandas as pd

from scope_quality import classify_scope,campus_keys,campus_allocation_cases

ROOT = Path(__file__).resolve().parents[1]


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("allocation",ROOT / "spatial/allocation_rules.py")
        cls.rules = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.rules)
        rows = cls.rules.map_to_regions(cls.rules.harmonize_inventory(ROOT / "spatial/project_records.csv","latest_20260917"))
        cls.rows = classify_scope(rows)
        cls.rows["campus_key"],cls.rows["usable_shared_address"] = campus_keys(cls.rows,cls.rules.normalize_text)

    def test_other_information_retained(self):
        a = self.rows.set_index("source_row")
        for i in [687,856,890,1121]:
            self.assertNotEqual(a.loc[i,"scope_class"],"no_explicit_crypto_tag")
            self.assertIn("other_info",a.loc[i,"crypto_evidence_fields"])

    def test_transitions_not_called_pure(self):
        a = self.rows.set_index("source_row")
        for i in [856,890,1121]:
            self.assertEqual(a.loc[i,"scope_class"],"mixed_or_transition_crypto")
        self.assertEqual(a.loc[687,"scope_class"],"uncertain_crypto_reference")
        self.assertFalse(a.crypto_MW_fraction_identified.any())

    def test_nullable_conditions(self):
        sample = pd.DataFrame({"purpose":pd.Series([pd.NA,"AI","bitcoin"],dtype="string"),
                               "other_info":pd.Series([pd.NA,"Bitcoin and AI",pd.NA],dtype="string")})
        a = classify_scope(sample)
        self.assertEqual(list(a.scope_class),["no_explicit_crypto_tag","mixed_or_transition_crypto","explicit_pure_crypto"])

    def test_unknown_addresses_are_distinct(self):
        a = self.rows.set_index("source_row")
        self.assertNotEqual(a.loc[1020,"campus_key"],a.loc[1056,"campus_key"])
        self.assertFalse(a.loc[1020,"usable_shared_address"])
        self.assertFalse(a.loc[1056,"usable_shared_address"])

    def test_road_only_and_vicinity_addresses_do_not_establish_identity(self):
        a = self.rows.set_index("source_row")
        ids = [411,412,1134,1135,1136,1137,1138,1139]
        self.assertFalse(a.loc[ids,"usable_shared_address"].any())
        self.assertEqual(a.loc[ids,"campus_key"].nunique(),len(ids))
        _, selected = campus_allocation_cases(self.rows,self.rules)
        for case in ["central_site_count","broad_site_count"]:
            self.assertTrue(set(range(1134,1140)).issubset(set(selected[selected.allocation_case.eq(case)].source_row)))

    def test_premise_address_requires_matching_county(self):
        rows = pd.DataFrame({"source_row":[1,2,3],"state":["TX"]*3,
                             "county":["Ellis","Ellis","Johnson"],
                             "address":["123 Main Street"]*3})
        keys, usable = campus_keys(rows,self.rules.normalize_text)
        self.assertTrue(usable.all())
        self.assertEqual(keys.iloc[0],keys.iloc[1])
        self.assertNotEqual(keys.iloc[0],keys.iloc[2])

    def test_distance_descriptions_are_not_premise_numbers(self):
        a = self.rows.set_index("source_row")
        self.assertFalse(a.loc[946,"usable_shared_address"])
        self.assertEqual(a.loc[946,"campus_key"],"row|946")
        addresses = ["3 miles outside of Sealy", "2.5 km north of town", "500 feet west of Main Street",
                     "3 Main Street", "123 Mile Road", "123 Main Street"]
        rows = pd.DataFrame({"source_row":range(6),"state":["TX"]*6,
                             "county":["Austin"]*6,"address":addresses})
        _, usable = campus_keys(rows,self.rules.normalize_text)
        self.assertEqual(list(usable),[False,False,False,True,True,True])

    def test_crypto_word_forms_and_current_eligibility(self):
        sample = pd.DataFrame({"purpose":["crypto mining","crypto-mining","cryptomining","cryptomines"],
                               "other_info":[""]*4})
        self.assertTrue(classify_scope(sample).scope_class.eq("explicit_pure_crypto").all())
        a = self.rows.set_index("source_row")
        self.assertEqual(a.loc[492,"scope_class"],"mixed_or_transition_crypto")
        self.assertNotEqual(a.loc[697,"scope_class"],"no_explicit_crypto_tag")
        _, subsets = self.rules.allocation_cases(self.rows)
        self.assertFalse(any({492,697} & set(x.source_row) for x in subsets.values()))

    def test_cancelled_record_cannot_displace_proposed(self):
        _,a = campus_allocation_cases(self.rows,self.rules)
        core = a[a.allocation_case.eq("central_reported_mid")]
        self.assertIn(711,set(core.source_row))
        self.assertNotIn(710,set(core.source_row))
        self.assertTrue({1020,1056}.issubset(set(core.source_row)))

    def test_each_rule_has_own_eligibility_and_unique_campus(self):
        weights,a = campus_allocation_cases(self.rows,self.rules)
        self.assertFalse(a.duplicated(["allocation_case","campus_key"]).any())
        self.assertTrue(np.allclose(weights.groupby("allocation_case").allocation_share.sum(),1))
        high = a[a.allocation_case.eq("high_evidence_reported_mid")]
        self.assertTrue(high.status.isin(self.rules.HIGH_EVIDENCE).all())


if __name__ == "__main__":
    unittest.main()
