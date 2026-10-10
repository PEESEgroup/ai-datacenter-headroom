"""Export main-figure numerical tables from freshly rebuilt results."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NUM = ROOT / "numerical"
OUT = ROOT / "plotting_tables"
KEYS = ["planning_region", "resource_condition", "response_condition"]
IDS = ["data_center_scenario_id", "post2030_extension_id", "footprint_capture_case", "peak_mapping_case", "allocation_case"]


def write(name, frame):
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def pv_variability():
    sys.path.insert(0, str(NUM / "scripts"))
    from run_pv_hourly import kernel
    dispatch = kernel()
    spec = importlib.util.spec_from_file_location("figure_physics", NUM / "intervention/physics.py")
    physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    config = json.loads((NUM / "intervention/physics_config.json").read_text())
    sites = pd.read_csv(NUM / "intervention/sites.csv")
    grid = pd.read_csv(NUM / "results/intervention/pv_storage_fine_grid.csv")
    years, peaks_out, display = [], [], []
    for region, group in sites.groupby("ISO"):
        group = group.sort_values("representative_id")
        arrays = []
        for r in group.itertuples():
            weather = pd.read_csv(NUM / "intervention/hourly_inputs" / (r.representative_id + ".csv.gz"))
            times = pd.DatetimeIndex(pd.to_datetime(weather.timestamp_utc, utc=True))
            converted = physics.solar_geometry_and_pv(times, r.latitude, r.longitude,
                weather.ghi_w_m2.to_numpy(), weather.air_temperature_c.to_numpy(), weather.wind_speed_m_s.to_numpy(), config)
            arrays.append(converted.pv_ac_kw_per_kwdc.to_numpy())
        peaks = pd.read_csv(NUM / "results/intervention" / ("observed_summer_peak_hours_" + region + ".csv"))
        indices = times.get_indexer(pd.DatetimeIndex(pd.to_datetime(peaks.timestamp_utc, utc=True)))
        assert (indices >= 0).all()
        for ratio, hours in [(1, 0), (2, 4), (3, 8)]:
            site_residual = np.stack([dispatch(pv, ratio, hours) for pv in arrays])
            residual = site_residual.mean(axis=0)
            frozen = grid[grid.ISO.eq(region) & grid.pv_ratio.eq(ratio) & grid.storage_hours.eq(hours)].iloc[0]
            np.testing.assert_allclose(1 - residual.mean(), frozen.annual_displacement_share, atol=1e-12)
            yearly = []
            for year in range(2016, 2026):
                value = 1 - residual[times.year == year].mean()
                yearly.append(value)
                years.append({"ISO": region, "year": year, "pv_ratio": ratio, "storage_hours": hours,
                    "annual_displacement_share": value, "sites": 5, "sample": "regional equal-site average"})
            for index, r in enumerate(group.itertuples()):
                peaks_out.append({"ISO": region, "representative_id": r.representative_id, "year": 2025,
                    "pv_ratio": ratio, "storage_hours": hours,
                    "summer_peak_p95_residual": np.quantile(site_residual[index, indices], .95)})
            display.append({"ISO": region, "pv_ratio": ratio, "storage_hours": hours,
                "annual_displacement_share": 1 - residual.mean(), "yearly_displacement_min": min(yearly),
                "yearly_displacement_max": max(yearly), "summer_peak_p95_residual": np.quantile(residual[indices], .95)})
    write("pv_annual_weather_years.csv", pd.DataFrame(years))
    write("pv_2025_site_peak_variability.csv", pd.DataFrame(peaks_out))
    write("pv_display_with_variability.csv", pd.DataFrame(display))


def main():
    pairs = pd.read_csv(NUM / "results/contribution/01_annual_paired_outcomes.csv.gz")
    core = pairs[pairs.resource_condition.isin(["anticipated", "prospective"]) & pairs.linked_response_fraction.eq(0)]
    selected = {"post2030_extension_id": "epri_quarter_rate_taper", "footprint_capture_case": "core_active_reported_mw_share",
                "peak_mapping_case": "reference_coincidence", "allocation_case": "reported_MW_reference_imputation"}
    named = core.copy()
    for field, value in selected.items():
        named = named[named[field].eq(value)]
    assert len(named) == 240
    write("named_national_paths.csv", named)
    fields = ["fresh_DC_pressure_GW", "fresh_DC_added_requirement_GW", "cushion_consumed_GW", "fresh_with_DC_balance_GW"]
    rows = []
    for key, group in core.groupby(["planning_region", "year", "resource_condition", "response_condition"]):
        row = dict(zip(["planning_region", "year", "resource_condition", "response_condition"], key))
        row["paths"] = len(group)
        for field in fields:
            row[field + "_min"], row[field + "_max"] = group[field].min(), group[field].max()
        rows.append(row)
    write("annual_core_ranges.csv", pd.DataFrame(rows))
    lower = pd.read_csv(NUM / "results/source_quality/03_lower_endpoint_annual_pairs.csv.gz")
    lower = lower[lower.source_DC_timing_rule.eq("linear")]
    for field, value in selected.items():
        lower = lower[lower[field].eq(value)]
    write("additional_low_endpoint_paths.csv", lower)

    native = pd.read_csv(NUM / "seasonal_resource_ledger_20261004/03_corrected_condition_panel.csv")
    drivers = []
    for region, group in native[native.resource_condition.eq("anticipated") & native.response_condition.eq("published_all")].groupby("planning_region"):
        first = group.sort_values("year").iloc[0]
        factor = 1 + first.reference_margin_level_pct / 100
        for r in group.itertuples():
            terms = {"resource_stock_term_GW": (r.capacity_equivalent_resources_MW - first.capacity_equivalent_resources_MW) / 1000,
                "source_gross_demand_term_GW": -factor * (r.gross_peak_demand_MW - first.gross_peak_demand_MW) / 1000,
                "credited_response_term_GW": factor * (r.credited_DR_MW - first.credited_DR_MW) / 1000,
                "reference_margin_change_term_GW": -r.net_peak_demand_MW * (r.reference_margin_level_pct - first.reference_margin_level_pct) / 100000}
            delta = r.margin_GW - first.margin_GW
            np.testing.assert_allclose(sum(terms.values()), delta, atol=1e-10)
            drivers.append({"planning_region": region, "year": r.year, "resource_condition": "anticipated",
                "response_condition": "published_all", "balance_2026_GW": first.margin_GW, "balance_GW": r.margin_GW,
                **terms, "balance_change_from_2026_GW": delta, "identity_residual_GW": delta - sum(terms.values())})
    write("seven_region_annual_drivers.csv", pd.DataFrame(drivers))
    reference = named[named.data_center_scenario_id.eq("lbnl_reference")]
    miso = reference[reference.planning_region.eq("MISO") & reference.resource_condition.eq("anticipated") & reference.response_condition.eq("published_all")].sort_values("year")
    limits = miso.fresh_no_new_DC_balance_GW / miso.fresh_DC_pressure_GW
    write("miso_annual_growth_boundaries.csv", pd.DataFrame({"year": miso.year, "annual_limit_pct": 100 * limits}))
    surface = [{"year": r.year, "growth_scale_pct": scale * 100,
                "balance_GW": r.fresh_no_new_DC_balance_GW - scale * r.fresh_DC_pressure_GW}
               for scale in np.linspace(0, 1.5, 61) for r in miso.itertuples()]
    write("miso_annual_growth_conditions.csv", pd.DataFrame(surface))
    bounds, surface = [], []
    ercot = native[native.planning_region.eq("ERCOT") & native.response_condition.eq("published_all") & native.resource_condition.isin(["anticipated", "prospective"])]
    for r in ercot.itertuples():
        leverage = (1 + r.reference_margin_level_pct / 100) * r.provisional_large_load_response_MW / 1000
        without = r.margin_GW - leverage
        needed = max(-without, 0) / leverage if leverage > 0 else (0 if without >= 0 else np.nan)
        bounds.append({"year": r.year, "resource_condition": r.resource_condition, "response_limit_pct": 100 * needed,
                       "balance_without_provisional_GW": without, "balance_published_GW": r.margin_GW})
        surface.extend({"year": r.year, "resource_condition": r.resource_condition,
                        "response_realized_pct": 100 * scale, "balance_GW": without + scale * leverage}
                       for scale in np.linspace(0, 1, 41))
    write("ercot_annual_response_boundaries.csv", pd.DataFrame(bounds))
    write("ercot_annual_response_conditions.csv", pd.DataFrame(surface))

    pv = pd.read_csv(NUM / "results/intervention/pv_storage_fine_grid.csv")
    write("pv_recomputed_full_grid.csv", pv)
    write("pv_recomputed_site_years.csv", pd.read_csv(NUM / "results/intervention/pv_storage_site_year_cases.csv"))
    write("national_annual_energy_paths.csv", pd.read_csv(NUM / "results/101_lbnl2025_national_energy_paths.csv"))
    write("seven_region_source_annual_paths.csv", pd.read_csv(NUM / "results/118_fig2_source_annual_plotting.csv"))
    write("fixed_total_spatial_weights.csv", pd.read_csv(NUM / "results/spatial/fixed_total_weights.csv"))
    pv_variability()
    (OUT / "recomputed_data_verification.json").write_text(json.dumps({"passed": True, "annual_core_pairs": len(core),
        "named_points": len(named), "PV_configurations": len(pv), "inputs_are_recomputed_results": True}, indent=2) + "\n")
    print(json.dumps({"figure_tables_recomputed": True, "named_points": len(named), "PV_configurations": len(pv)}))


if __name__ == "__main__":
    main()
