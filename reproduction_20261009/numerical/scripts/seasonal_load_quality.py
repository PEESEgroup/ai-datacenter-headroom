"""Reconstruct reported demand and require a complete local season before ranking."""
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {"ERCO": "ERCOT", "MISO": "MISO", "PJM": "PJM", "SWPP": "SPP"}
ZONES = {"PJM": "America/New_York", "MISO": "America/Chicago", "SPP": "America/Chicago", "ERCOT": "America/Chicago"}
GROSS_OUTLIER_MULTIPLE = 10.0


def reported_demand():
    records = []
    for code, region in REGIONS.items():
        path = ROOT / "intervention/sources/eia930" / (code + "_reported_demand_2023_2025.json.gz")
        with gzip.open(path, "rt") as handle:
            source = json.load(handle)
        assert source["series_id"] == "EBA." + code + "-ALL.D.H"
        frame = pd.DataFrame(source["data"], columns=["timestamp", "iso_total_load_mw"])
        frame["timestamp_utc"] = pd.to_datetime(frame.timestamp, format="%Y%m%dT%H", utc=True)
        frame["iso_total_load_mw"] = pd.to_numeric(frame.iso_total_load_mw, errors="coerce")
        frame["iso"] = region
        records.append(frame[["timestamp_utc", "iso", "iso_total_load_mw"]])
    result = pd.concat(records).sort_values(["iso", "timestamp_utc"])
    assert not result.duplicated(["iso", "timestamp_utc"]).any()
    return result


def season(frame, region, year):
    start = pd.Timestamp(f"{year}-06-01", tz=ZONES[region]).tz_convert("UTC")
    stop = pd.Timestamp(f"{year}-10-01", tz=ZONES[region]).tz_convert("UTC")
    # EIA reports the ending label of each hourly interval in the local season.
    expected = pd.date_range(start + pd.Timedelta(hours=1), stop, freq="h")
    part = frame[frame.iso.eq(region) & frame.timestamp_utc.between(start, stop, inclusive="right")].copy()
    duplicate = part.timestamp_utc.duplicated().sum()
    values = part.set_index("timestamp_utc").iso_total_load_mw
    matched = values.reindex(expected) if not duplicate else pd.Series(np.nan, index=expected)
    finite_positive = np.isfinite(matched) & matched.gt(0)
    median = matched[finite_positive].median()
    gross_outlier = finite_positive & matched.gt(GROSS_OUTLIER_MULTIPLE * median)
    valid = finite_positive & ~gross_outlier
    quality = {"ISO": region, "year": year, "expected_hours": len(expected), "recorded_hours": len(part),
        "missing_timestamps": len(expected.difference(part.timestamp_utc)), "duplicate_timestamps": int(duplicate),
        "nonfinite_recorded": int((~np.isfinite(part.iso_total_load_mw)).sum()),
        "nonpositive_recorded": int(part.iso_total_load_mw.le(0).sum()), "valid_hours": int(valid.sum()),
        "finite_positive_hours": int(finite_positive.sum()), "gross_outlier_hours": int(gross_outlier.sum()),
        "gross_outlier_threshold_MW": float(GROSS_OUTLIER_MULTIPLE * median),
        "complete": bool(valid.all() and not duplicate), "season_boundary_basis": "EIA_hour_ending"}
    return part, quality


def select_peaks(frame, region, years, top=100):
    selected, audits = [], []
    for year in years:
        part, quality = season(frame, region, year)
        if not quality["complete"]:
            raise ValueError("Incomplete season must not be ranked as full summer: " + json.dumps(quality))
        quality["selected_hours"] = top
        part["year"] = year
        selected.append(part.sort_values(["iso_total_load_mw", "timestamp_utc"], ascending=[False, True]).head(top))
        audits.append(quality)
    peaks = pd.concat(selected).sort_values(["year", "iso_total_load_mw"], ascending=[True, False])
    assert len(peaks) == top * len(years)
    return peaks, audits


def main():
    out = ROOT / "results/source_quality"
    out.mkdir(parents=True, exist_ok=True)
    new = reported_demand()
    old = pd.read_csv(ROOT / "intervention/observed_hourly_load_2023_2025.csv.gz")
    old.timestamp_utc = pd.to_datetime(old.timestamp_utc, utc=True)
    rows = []
    for source, frame in [("previous_mixed_ISO_extract", old), ("reported_EIA930_D_H", new)]:
        for region in REGIONS.values():
            for year in [2023, 2024, 2025]:
                _, row = season(frame, region, year)
                row["source"] = source
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / "01_load_season_coverage.csv", index=False)
    new.to_csv(out / "reported_hourly_demand_reconstructed.csv.gz", index=False)
    flagged = []
    for region in REGIONS.values():
        for year in [2023, 2024, 2025]:
            part, _ = season(new, region, year)
            positive = part.loc[np.isfinite(part.iso_total_load_mw) & part.iso_total_load_mw.gt(0), "iso_total_load_mw"]
            for multiple in [5.0, 10.0, 20.0]:
                suspicious = part[part.iso_total_load_mw.gt(multiple * positive.median())]
                for r in suspicious.itertuples():
                    flagged.append({"ISO":region,"year":year,"timestamp_utc":r.timestamp_utc,
                        "reported_demand_MW":r.iso_total_load_mw,"median_multiple":multiple,
                        "automatic_replacement":False,"main_season_excluded":True})
    pd.DataFrame(flagged).to_csv(out / "13_gross_load_outlier_audit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
