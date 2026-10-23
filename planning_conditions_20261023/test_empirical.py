"""Focused checks on evidence coverage, load definitions and source recovery."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from run_empirical import HERE,read_workbook,july_comparisons,digitize_dc_months,extract_ercot_scale


class EmpiricalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=read_workbook(HERE/"sources/accuracy/2026-accuracy-report.xlsx")
        cls.pairs,cls.primary,cls.coverage=july_comparisons(cls.raw)
        cls.months,cls.summary,cls.calibrations,cls.month_coverage=digitize_dc_months(
            HERE/"sources/accuracy/2026-load-forecast-accuracy-report.pdf")

    def test_exact_workbook_cells_are_retained(self):
        self.assertEqual(len(self.raw),1124)
        r=self.primary.set_index("sheet")
        self.assertEqual(r.loc["AEP","forecast_MW"],1761)
        self.assertAlmostEqual(r.loc["DOM","actual_MW"],5563.1688136)

    def test_missing_actuals_are_not_zero(self):
        r=self.coverage.set_index("zone")
        self.assertFalse(r.loc["BGE","actual_present"])
        self.assertFalse(r.loc["BGE","matched"])
        self.assertTrue(pd.isna(r.loc["BGE","actual_MW"]))

    def test_no_parent_subzone_or_RTO_double_count(self):
        self.assertEqual(set(self.primary.sheet),{"AEP","COMED","DOM","PS"})
        self.assertEqual(len(self.primary),4)
        self.assertAlmostEqual(self.primary.forecast_MW.sum(),7566)

    def test_matched_coverage_not_full_forecast_total(self):
        full=self.raw[self.raw.sheet.eq("RTO")&self.raw.vintage.eq(2025)&self.raw.year.eq(2025)].MW.iloc[0]
        self.assertEqual(full,7735)
        self.assertEqual(full-self.primary.forecast_MW.sum(),169)

    def test_zonal_deviations_have_both_directions(self):
        r=self.primary.set_index("sheet").actual_minus_forecast_MW
        self.assertTrue(r["DOM"]>0 and r["COMED"]>0)
        self.assertTrue(r["AEP"]<0 and r["PS"]<0)
        self.assertAlmostEqual(r.sum(),601.1738136)

    def test_monthly_records_are_source_filtered_DC_graphs(self):
        primary=self.months[self.months.vintage.eq(2025)&self.months.zone.ne("RTO")]
        self.assertEqual(len(primary),32)
        self.assertEqual(primary.month.nunique(),8)
        self.assertTrue(primary.digitized_from_vector_graph.all())
        self.assertEqual(set(primary.zone),{"AEP","COMED","DOM","PS"})

    def test_common_vintage_comparison_uses_same_months(self):
        self.assertTrue(self.summary.months.eq(8).all())
        self.assertTrue(self.summary.first_month.eq("2025-01").all())
        self.assertTrue(self.summary.last_month.eq("2025-08").all())

    def test_graph_reading_allowance_does_not_change_2025_signs(self):
        primary=self.months[self.months.vintage.eq(2025)&self.months.zone.ne("RTO")]
        for zone,group in primary.groupby("zone"):
            expected="below" if zone=="PS" else "above"
            self.assertTrue(group.sign_with_allowance.eq(expected).all())
        self.assertTrue(all(c["max_tick_fit_error_points"]<1 for c in self.calibrations))

    def test_same_source_ERCOT_scale(self):
        source=extract_ercot_scale(HERE/"sources/accuracy/NERC_2025_ERCOT_p131.pdf")
        r=source.set_index("year").loc[2035]
        self.assertEqual(r.total_large_load_MW,69463.9)
        self.assertEqual(r.provisional_curtailment_MW,58365.6)
        self.assertAlmostEqual(100*51154.12747252748/r.total_large_load_MW,73.6413122104)


if __name__=="__main__":
    unittest.main(verbosity=2)
