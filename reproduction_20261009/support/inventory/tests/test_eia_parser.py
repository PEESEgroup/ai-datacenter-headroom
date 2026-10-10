from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eia_parser
from build_stock_flow import numeric


class LiteralIdentifierTests(unittest.TestCase):
    def read_example(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.xlsx"
            frame = pd.DataFrame({
                "Plant ID": [62844, 100, 101, None, 102],
                "Generator ID": ["NA", "", "N/A", "A", "B"],
                "Net Summer Capacity (MW)": [150, 20, "", 30, "N/A"],
            })
            frame.to_excel(path, sheet_name="Operating", index=False, startrow=2)
            return eia_parser.read_860m(path, "Operating")

    def test_literal_NA_generator_is_retained(self):
        frame = self.read_example()
        self.assertIn("62844::NA", frame.unit_key.tolist())
        self.assertEqual(frame.loc[frame.unit_key.eq("62844::NA"),
                                   "Net Summer Capacity (MW)"].iloc[0], 150)

    def test_other_literal_identifier_is_not_a_missing_token(self):
        self.assertIn("101::N/A", self.read_example().unit_key.tolist())

    def test_blank_plant_or_generator_is_not_a_valid_key(self):
        self.assertEqual(set(self.read_example().unit_key), {"62844::NA", "101::N/A", "102::B"})

    def test_missing_numeric_cells_remain_unknown_not_zero(self):
        frame = self.read_example()
        capacities = numeric(frame, "Net Summer Capacity (MW)")
        self.assertTrue(capacities[frame.unit_key.isin(["101::N/A", "102::B"])].isna().all())
