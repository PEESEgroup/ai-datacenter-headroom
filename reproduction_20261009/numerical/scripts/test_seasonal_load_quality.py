"""A rankable top 100 is not evidence of a complete source season."""
import unittest

import numpy as np
import pandas as pd

from seasonal_load_quality import select_peaks, ZONES


class CompleteSeasonTests(unittest.TestCase):
    def setUp(self):
        hours = pd.date_range("2023-06-01 01:00", "2023-10-01 00:00", freq="h", tz=ZONES["MISO"]).tz_convert("UTC")
        self.frame = pd.DataFrame({"timestamp_utc": hours, "iso": "MISO", "iso_total_load_mw": np.arange(len(hours)) + 1000.0})

    def test_complete_season(self):
        peaks, audits = select_peaks(self.frame, "MISO", [2023])
        self.assertEqual(len(peaks), 100)
        self.assertEqual(audits[0]["valid_hours"], 2928)

    def test_missing_timestamp(self):
        with self.assertRaises(ValueError):
            select_peaks(self.frame.iloc[1:], "MISO", [2023])

    def test_nonfinite_record(self):
        self.frame.loc[0, "iso_total_load_mw"] = np.nan
        with self.assertRaises(ValueError):
            select_peaks(self.frame, "MISO", [2023])

    def test_zero_record(self):
        self.frame.loc[0, "iso_total_load_mw"] = 0
        with self.assertRaises(ValueError):
            select_peaks(self.frame, "MISO", [2023])

    def test_duplicate_record(self):
        with self.assertRaises(ValueError):
            select_peaks(pd.concat([self.frame, self.frame.iloc[:1]]), "MISO", [2023])

    def test_hour_ending_boundaries(self):
        self.frame.loc[self.frame.index[-1], "iso_total_load_mw"] = 5000
        outside = self.frame.iloc[:1].copy()
        outside["timestamp_utc"] -= pd.Timedelta(hours=1)
        outside["iso_total_load_mw"] = 2000000
        peaks, _ = select_peaks(pd.concat([outside,self.frame]), "MISO", [2023])
        self.assertEqual(peaks.iso_total_load_mw.max(),5000)

    def test_finite_positive_gross_outlier(self):
        self.frame.loc[100, "iso_total_load_mw"] = 3621097
        with self.assertRaises(ValueError):
            select_peaks(self.frame, "MISO", [2023])

    def test_outlier_not_replaced_or_ranked(self):
        from seasonal_load_quality import season
        self.frame.loc[100, "iso_total_load_mw"] = 3621097
        _, audit = season(self.frame, "MISO", 2023)
        self.assertEqual(audit["finite_positive_hours"],2928)
        self.assertEqual(audit["gross_outlier_hours"],1)
        self.assertEqual(audit["valid_hours"],2927)
        self.assertFalse(audit["complete"])
        self.assertEqual(self.frame.loc[100,"iso_total_load_mw"],3621097)


if __name__ == "__main__":
    unittest.main()
