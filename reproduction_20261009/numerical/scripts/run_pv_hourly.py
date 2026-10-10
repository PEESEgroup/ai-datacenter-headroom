"""Recompute all 676 PV-storage points from archived hourly weather and load."""
from pathlib import Path
import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import shutil

import numpy as np
import pandas as pd
from run_review_package import compare
from seasonal_load_quality import reported_demand, select_peaks, season

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "intervention"
OUT = ROOT / "results/intervention"


def source_weather(row, audit, config):
    import xarray as xr
    item = audit.set_index("representative_id").loc[row.representative_id]
    for field, hash_field in [("nasa_cache_path", "nasa_cache_sha256"), ("era5_path", "era5_sha256")]:
        path = DATA / item[field]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item[hash_field]
    with gzip.open(DATA / item.nasa_cache_path, "rt") as handle:
        values = json.load(handle)["properties"]["parameter"][config["weather"]["nasa_power_variable"]]
    nasa = pd.DataFrame({"timestamp_utc": pd.to_datetime(list(values), format="%Y%m%d%H", utc=True),
                         "ghi_w_m2": pd.to_numeric(list(values.values()))})
    with xr.open_dataset(DATA / item.era5_path) as raw:
        era = pd.DataFrame({"timestamp_utc": pd.to_datetime(raw.valid_time.values, utc=True),
                            "air_temperature_c": raw.t2m.values.astype(float) - 273.15,
                            "wind_speed_m_s": np.hypot(raw.u10.values.astype(float), raw.v10.values.astype(float))})
    era = era[era.timestamp_utc.between(pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:00", tz="UTC"))]
    merged = nasa.merge(era, on="timestamp_utc", validate="one_to_one").sort_values("timestamp_utc")
    archived = pd.read_csv(DATA / "hourly_inputs" / (row.representative_id + ".csv.gz"))
    archived.timestamp_utc = pd.to_datetime(archived.timestamp_utc, utc=True)
    assert merged.timestamp_utc.reset_index(drop=True).equals(archived.timestamp_utc)
    assert np.allclose(merged[["ghi_w_m2", "air_temperature_c", "wind_speed_m_s"]],
                       archived[["ghi_w_m2", "air_temperature_c", "wind_speed_m_s"]], atol=1e-10, rtol=1e-12)
    return merged


def kernel():
    library = OUT / ("storage_kernel.dylib" if sys.platform == "darwin" else "storage_kernel.so")
    command = ["cc", "-O3", "-dynamiclib"] if sys.platform == "darwin" else ["cc", "-O3", "-shared", "-fPIC"]
    subprocess.run(command + [str(DATA / "storage_kernel.c"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.dispatch.argtypes = [pointer, ctypes.c_size_t, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_size_t, pointer]
    lib.dispatch.restype = None

    def run(pv, ratio, hours):
        pv = np.ascontiguousarray(pv, dtype=np.float64)
        residual = np.empty(len(pv), dtype=np.float64)
        lib.dispatch(pv, len(pv), ratio, hours, 0.85, 720, residual)
        return residual
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-provider-files", action="store_true",
                        help="Also re-extract NASA/ERA5 provider archives; requires the optional weather environment")
    parser.add_argument("--write-quality-reference", action="store_true",
                        help="Create separate corrected regression fixtures; never replace the old peak fixtures")
    args = parser.parse_args()
    def regression(expected, actual, keys, columns):
        if args.write_quality_reference:
            assert not expected.exists(), "Corrected reference already exists; do not silently refreeze it"
            shutil.copy2(actual, expected)
        compare(expected, actual, keys, columns)
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("wp6_physics", DATA / "physics.py")
    physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    config = json.loads((DATA / "physics_config.json").read_text())
    rules = json.loads((DATA / "grid_config.json").read_text())
    sites = pd.read_csv(DATA / "sites.csv")
    audit = pd.read_csv(DATA / "weather_source_audit.csv")
    load = reported_demand()
    assert not load.duplicated(["iso", "timestamp_utc"]).any()
    dispatch = kernel()
    random = np.random.default_rng(48).uniform(0, 0.85, 2000)
    for ratio, hours in [(0, 0), (0.75, 4), (2, 4), (3, 24)]:
        expected = physics.simulate_storage(random, ratio, hours, 0.85, config)["residual"]
        assert np.max(np.abs(expected - dispatch(random, ratio, hours))) < 1e-12
    records, site_year_records, coverage, fixed_checks, anomaly_diagnostics = [], [], [], [], []
    prefix = rules["regression_reference_prefix"]
    for region in rules["regions"]:
        selected = sites[sites.ISO.eq(region)].sort_values("representative_id")
        assert len(selected) == 5
        arrays = []
        timestamps = None
        for site in selected.itertuples():
            if args.from_provider_files:
                weather = source_weather(site, audit, config)
            else:
                weather = pd.read_csv(DATA / "hourly_inputs" / (site.representative_id + ".csv.gz"))
                weather.timestamp_utc = pd.to_datetime(weather.timestamp_utc, utc=True)
            times = pd.DatetimeIndex(weather.timestamp_utc)
            expected_times = pd.date_range("2016-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
            assert len(weather) == 87672 and times.equals(expected_times)
            values = weather[["ghi_w_m2", "air_temperature_c", "wind_speed_m_s"]]
            assert np.isfinite(values).all().all()
            converted = physics.solar_geometry_and_pv(times, site.latitude, site.longitude,
                weather.ghi_w_m2.to_numpy(), weather.air_temperature_c.to_numpy(), weather.wind_speed_m_s.to_numpy(), config)
            arrays.append(converted.pv_ac_kw_per_kwdc.to_numpy())
            timestamps = times
        peaks, audits = select_peaks(load, region, rules["observed_peak_years"], rules["top_load_hours_per_year"])
        coverage.extend(audits)
        destination = OUT / ("observed_summer_peak_hours_" + region + ".csv")
        peaks.to_csv(destination, index=False)
        regression(DATA / (prefix + destination.name), destination,
                   ["timestamp_utc", "iso"], ["iso_total_load_mw", "year"])
        indices = timestamps.get_indexer(pd.DatetimeIndex(peaks.timestamp_utc))
        assert (indices >= 0).all()
        diagnostic_indices = None
        if region == "SPP":
            partial, gate = season(load,region,2023)
            assert gate["gross_outlier_hours"] == 1 and not gate["complete"]
            partial = partial[partial.iso_total_load_mw.gt(0) & partial.iso_total_load_mw.le(gate["gross_outlier_threshold_MW"])]
            partial = partial.sort_values(["iso_total_load_mw","timestamp_utc"],ascending=[False,True]).head(100)
            diagnostic_times = pd.concat([partial,peaks]).timestamp_utc
            diagnostic_indices = timestamps.get_indexer(pd.DatetimeIndex(diagnostic_times))
            assert len(diagnostic_indices) == 200 and (diagnostic_indices >= 0).all()
        for ratio in rules["pv_ratios"]:
            for hours in rules["storage_hours"]:
                site_residual = np.stack([dispatch(pv, ratio, hours) for pv in arrays])
                residual = site_residual.mean(axis=0)
                tail = residual[indices]
                records.append({"ISO": region, "pv_ratio": ratio, "storage_hours": hours,
                    "annual_displacement_share": 1 - residual.mean(), "summer_peak_mean_residual": tail.mean(),
                    "summer_peak_p95_residual": np.quantile(tail, 0.95), "full_year_max_residual": residual.max(),
                    "full_year_p99_residual": np.quantile(residual, 0.99)})
                if diagnostic_indices is not None:
                    d = residual[diagnostic_indices]
                    anomaly_diagnostics.append({"ISO":region,"pv_ratio":ratio,"storage_hours":hours,
                        "summer_peak_mean_residual":d.mean(),"summer_peak_p95_residual":np.quantile(d,.95),
                        "sample":"2023 invalid point excluded and reranked plus complete 2025; incomplete-season diagnostic only",
                        "primary_result":False})
                if [ratio, hours] in rules["fixed_cases"]:
                    for site_index, site in enumerate(selected.itertuples()):
                        for year in rules["observed_peak_years"]:
                            values = site_residual[site_index, indices[peaks.year.eq(year).to_numpy()]]
                            annual = site_residual[site_index, timestamps.year == year]
                            site_year_records.append({"ISO": region, "representative_id": site.representative_id,
                                "year": year, "pv_ratio": ratio, "storage_hours": hours,
                                "annual_displacement_share": 1 - annual.mean(),
                                "summer_peak_mean_residual": values.mean(), "summer_peak_p95_residual": np.quantile(values, .95),
                                "full_year_max_residual": annual.max()})
                    for shift in [-1, 0, 1]:
                        sampled = residual[indices + shift]
                        fixed_checks.append({"ISO": region, "pv_ratio": ratio, "storage_hours": hours,
                            "weather_alignment_shift_h": shift, "mean_residual": sampled.mean(),
                            "p95_residual": np.quantile(sampled, .95)})
        print("[hourly PV] " + region + ": 169 configurations recalculated", flush=True)
    fine = pd.DataFrame(records)
    keys = ["ISO", "pv_ratio", "storage_hours"]
    assert len(fine) == 676 and not fine.duplicated(keys).any()
    path = OUT / "pv_storage_fine_grid.csv"
    fine.to_csv(path, index=False)
    assert fine.iloc[:, 3:].ge(-1e-10).all().all() and fine.iloc[:, 3:].le(1 + 1e-10).all().all()
    regression(DATA / (prefix + "pv_storage_fine_grid.csv"), path, keys, list(fine.columns[3:]))
    pd.DataFrame(site_year_records).to_csv(OUT / "pv_storage_site_year_cases.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "complete_peak_season_coverage.csv", index=False)
    pd.DataFrame(fixed_checks).to_csv(OUT / "peak_timestamp_alignment_sensitivity.csv", index=False)
    pd.DataFrame(anomaly_diagnostics).to_csv(OUT / "SPP_anomaly_exclusion_diagnostic.csv",index=False)
    original = pd.read_csv(DATA / "expected_pv_storage_fine_grid.csv")
    stable = fine.merge(original, on=keys, validate="one_to_one", suffixes=("", "_original"))
    for column in ["annual_displacement_share", "full_year_max_residual", "full_year_p99_residual"]:
        assert np.allclose(stable[column], stable[column + "_original"], atol=1e-10, rtol=0)
    record = {"passed": True, "grid_points_recomputed": len(fine), "weather_sites": len(sites),
              "weather_hours_per_site": 87672, "selected_peak_hours_per_region": 100 * len(rules["observed_peak_years"]),
              "complete_common_peak_years": rules["observed_peak_years"], "full_season_checked_before_selection": True,
              "gross_outlier_screen_before_selection": True,"SPP_partial_season_diagnostic_not_primary":True,
              "author_load_imputation": False, "annual_energy_results_unchanged": True,
              "PV_conversion_recomputed": True, "chronological_dispatch_recomputed": True,
              "peak_hour_selection_recomputed": True, "raw_provider_weather_reextracted": args.from_provider_files,
              "unchanged_physical_model": True, "native_capacity_credit_estimated": False}
    name = "verification_raw_sources.json" if args.from_provider_files else "verification.json"
    (OUT / name).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
