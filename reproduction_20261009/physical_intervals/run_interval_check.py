"""Recompute only PV dispatch summaries using documented physical intervals."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT = (ROOT / "numeric_inputs" if (ROOT / "numeric_inputs").exists()
           else ROOT.parent / "wecc_california_alignment_20261009/numerical")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demand_to_weather_labels(demand_end, relative_shift=0):
    """POWER starts the interval that EIA ends one hour later."""
    return pd.DatetimeIndex(demand_end) - pd.Timedelta(hours=1) + pd.Timedelta(hours=relative_shift)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    data = args.numeric_root / "intervention"
    out = ROOT / "interval_results"
    out.mkdir(exist_ok=True)
    protected = [data / n for n in ["physics.py", "physics_config.json", "storage_kernel.c", "grid_config.json", "sites.csv"]]
    protected += sorted((data / "hourly_inputs").glob("*.csv.gz"))
    protected += sorted((data / "sources/eia930").glob("*.json.gz"))
    protected += [args.numeric_root / "source_audit/native_inputs.csv"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    physics = load_module("interval_physics", data / "physics.py")
    quality = load_module("interval_quality", args.numeric_root / "scripts/seasonal_load_quality.py")
    config = json.loads((data / "physics_config.json").read_text())
    rules = json.loads((data / "grid_config.json").read_text())
    sites = pd.read_csv(data / "sites.csv")
    original = pd.read_csv(data / (rules["regression_reference_prefix"] + "pv_storage_fine_grid.csv"))
    load = quality.reported_demand()
    library = out / ("storage_kernel.dylib" if sys.platform == "darwin" else "storage_kernel.so")
    flags = ["cc", "-O3", "-dynamiclib"] if sys.platform == "darwin" else ["cc", "-O3", "-shared", "-fPIC"]
    subprocess.run(flags + [str(data / "storage_kernel.c"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.dispatch.argtypes = [pointer, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_size_t, pointer]
    lib.dispatch.restype = None

    def dispatch(pv, ratio, hours):
        pv = np.ascontiguousarray(pv, dtype=np.float64)
        result = np.empty(len(pv), dtype=np.float64)
        lib.dispatch(pv, len(pv), ratio, hours, .85, 720, result)
        return result

    # Test the direction and date rollover, independently of array positions.
    ends = pd.DatetimeIndex(["2025-07-02T00:00:00Z", "2025-07-02T13:00:00Z"])
    assert list(demand_to_weather_labels(ends)) == [pd.Timestamp("2025-07-01T23:00:00Z"), pd.Timestamp("2025-07-02T12:00:00Z")]
    random = np.random.default_rng(48).uniform(0, .85, 2000)
    for ratio, hours in [(0, 0), (1, 0), (2, 4), (3, 8), (3, 24)]:
        independent = physics.simulate_storage(random, ratio, hours, .85, config)["residual"]
        assert np.max(np.abs(independent - dispatch(random, ratio, hours))) < 1e-12

    grid, fixed, annual, windows, coverage, mappings = [], [], [], [], [], []
    legacy_error = 0.0
    annual_error = 0.0
    for region in ["PJM", "MISO", "SPP", "ERCOT"]:
        chosen = sites[sites.ISO.eq(region)].sort_values("representative_id")
        assert len(chosen) == 5
        arrays = []
        for site in chosen.itertuples():
            weather = pd.read_csv(data / "hourly_inputs" / (site.representative_id + ".csv.gz"))
            times = pd.DatetimeIndex(pd.to_datetime(weather.timestamp_utc, utc=True))
            assert times.equals(pd.date_range("2016-01-01", "2025-12-31 23:00", freq="h", tz="UTC"))
            converted = physics.solar_geometry_and_pv(times, site.latitude, site.longitude,
                weather.ghi_w_m2.to_numpy(), weather.air_temperature_c.to_numpy(), weather.wind_speed_m_s.to_numpy(), config)
            arrays.append(converted.pv_ac_kw_per_kwdc.to_numpy())
        selections = {}
        for n in [50, 100, 200]:
            peaks, audit = quality.select_peaks(load, region, [2025], n)
            selections[n] = peaks
            if n == 100:
                coverage.extend(audit)
                old = pd.read_csv(data / (rules["regression_reference_prefix"] + "observed_summer_peak_hours_" + region + ".csv"))
                assert list(pd.to_datetime(old.timestamp_utc, utc=True)) == list(peaks.timestamp_utc)
                mapping = peaks.rename(columns={"timestamp_utc": "demand_hour_end_utc"}).copy()
                mapping["weather_interval_start_utc"] = demand_to_weather_labels(peaks.timestamp_utc)
                mapping["weather_interval_end_utc"] = pd.DatetimeIndex(peaks.timestamp_utc)
                assert (mapping.weather_interval_end_utc - mapping.weather_interval_start_utc).eq(pd.Timedelta(hours=1)).all()
                mappings.append(mapping)
        for n in [50, 100]:
            assert set(selections[n].timestamp_utc).issubset(set(selections[200].timestamp_utc))
        correct_indices = times.get_indexer(demand_to_weather_labels(selections[100].timestamp_utc))
        legacy_indices = times.get_indexer(pd.DatetimeIndex(selections[100].timestamp_utc))
        assert (correct_indices >= 0).all() and np.array_equal(correct_indices + 1, legacy_indices)
        for ratio in rules["pv_ratios"]:
            for hours in rules["storage_hours"]:
                residuals = np.stack([dispatch(pv, ratio, hours) for pv in arrays])
                mean = residuals.mean(axis=0)
                tail = mean[correct_indices]
                row = dict(ISO=region, pv_ratio=ratio, storage_hours=hours,
                    annual_displacement_share=1-mean.mean(), summer_peak_mean_residual=tail.mean(),
                    summer_peak_p95_residual=np.quantile(tail, .95), full_year_max_residual=mean.max(),
                    full_year_p99_residual=np.quantile(mean, .99))
                old = original[original.ISO.eq(region) & original.pv_ratio.eq(ratio) & original.storage_hours.eq(hours)].iloc[0]
                annual_error = max(annual_error, *(abs(row[k]-old[k]) for k in ["annual_displacement_share", "full_year_max_residual", "full_year_p99_residual"]))
                legacy_error = max(legacy_error, abs(np.quantile(mean[legacy_indices],.95)-old.summer_peak_p95_residual), abs(mean[legacy_indices].mean()-old.summer_peak_mean_residual))
                grid.append(row)
                if [ratio, hours] not in rules["fixed_cases"]:
                    continue
                for year in range(2016,2026):
                    annual.append(dict(ISO=region, year=year, pv_ratio=ratio, storage_hours=hours,
                                       annual_displacement_share=1-mean[times.year == year].mean()))
                for i, site in enumerate(chosen.itertuples()):
                    fixed.append(dict(ISO=region, representative_id=site.representative_id, year=2025,
                        pv_ratio=ratio, storage_hours=hours,
                        summer_peak_mean_residual=residuals[i,correct_indices].mean(),
                        summer_peak_p95_residual=np.quantile(residuals[i,correct_indices],.95)))
                for n, selection in selections.items():
                    # Adjacent tests are centred on the corrected match, not the old equal-label match.
                    for shift in [-1,0,1]:
                        indices = times.get_indexer(demand_to_weather_labels(selection.timestamp_utc, shift))
                        assert (indices >= 0).all()
                        sample = mean[indices]
                        windows.append(dict(region=region, pv_ratio=ratio, storage_hours=hours, top_hours=n,
                            relative_shift_h=shift, absolute_weather_minus_demand_label_h=shift-1,
                            physical_primary=(shift == 0), p95_residual_pct=100*np.quantile(sample,.95),
                            mean_residual_pct=100*sample.mean()))
        print(region + ": 169 configurations; physical interval join and nested windows checked", flush=True)
    assert annual_error < 1e-12 and legacy_error < 1e-12
    fine = pd.DataFrame(grid)
    assert len(fine) == 676 and not fine.duplicated(["ISO","pv_ratio","storage_hours"]).any()
    assert fine.iloc[:,3:].ge(-1e-10).all().all() and fine.iloc[:,3:].le(1+1e-10).all().all()
    fine.to_csv(out / "pv_storage_fine_grid.csv", index=False)
    pd.DataFrame(fixed).to_csv(out / "pv_storage_site_cases.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "pv_annual_energy.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "complete_peak_season_coverage.csv", index=False)
    pd.concat(mappings,ignore_index=True).to_csv(out / "physical_peak_hour_mapping.csv",index=False)
    win = pd.DataFrame(windows)
    win.to_csv(out / "pv_peak_windows_corrected.csv",index=False)
    summary = []
    for (region,ratio,hours), group in win.groupby(["region","pv_ratio","storage_hours"]):
        correct = group[group.relative_shift_h.eq(0)].set_index("top_hours")
        summary.append(dict(region=region,pv_ratio=ratio,storage_hours=hours,
            p95_50=correct.loc[50,"p95_residual_pct"],p95_100=correct.loc[100,"p95_residual_pct"],
            p95_200=correct.loc[200,"p95_residual_pct"],
            corrected_window_min=correct.p95_residual_pct.min(),corrected_window_max=correct.p95_residual_pct.max(),
            all_checks_min=group.p95_residual_pct.min(),all_checks_max=group.p95_residual_pct.max()))
    pd.DataFrame(summary).to_csv(out / "pv_peak_window_summary_corrected.csv",index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    record = dict(passed=True, grid_points=676, weather_sites=20, hours_per_site=87672,
        selected_hours_per_region=100, common_peak_year=2025, mapping="EIA end t -> POWER start t-1h",
        inputs_unchanged=True, main_annual_model_rerun=False, physical_parameters_changed=False,
        peak_hours_selected_unchanged=True, annual_energy_max_error=annual_error,
        legacy_equal_label_replication_max_error=legacy_error,
        storage_independent_kernel_checks=5, midnight_rollover_test=True,
        sources={"POWER":"https://power.larc.nasa.gov/docs/faqs/other/",
                 "EIA":"https://www.eia.gov/survey/form/eia_930/instructions.pdf"},
        ERA5_temperature_and_wind="instantaneous at interval start; retained meteorological approximation",
        capacity_credit_estimated=False, input_sha256=hashes)
    (out/"verification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
