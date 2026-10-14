"""Reconcile reports and test fixed PV peak windows; never run the annual builder."""
import ctypes
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root", type=Path, default=HERE.parent,
                    help="Root of the accompanying reproduction_20261009 package.")
parser.add_argument("--output-root", type=Path, default=HERE / "results",
                    help="Directory for recomputed checks and supplementary tables.")
args = parser.parse_args()
PUBLIC = args.package_root.resolve()
NUMERIC = PUBLIC / "numerical"
DATA = NUMERIC / "intervention"
OUT = args.output_root.resolve()
OUT.mkdir(parents=True, exist_ok=True)
TABLES = OUT / "tables"
TABLES.mkdir(exist_ok=True)
INPUTS = HERE / "inputs"
EXPECTED = HERE / "expected"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, caption, label, columns, header, rows, note):
    lines = [r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}",
             r"\ifrevisionmarkup\color{revisionblue}\captionsetup{font+={color=revisionblue},labelfont+={color=revisionblue}}\fi",
             r"\begin{longtable}{" + columns + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule " + " & ".join(header) + r" \\\midrule\endfirsthead",
             r"\toprule " + " & ".join(header) + r" \\\midrule\endhead",
             r"\bottomrule\endfoot"]
    lines.extend((row if isinstance(row, str) else " & ".join(row)) + r" \\" for row in rows)
    lines += [r"\end{longtable}", r"\tabnote{" + note + "}", r"\endgroup"]
    path.write_text("\n".join(lines) + "\n")


protected = [NUMERIC / "source_audit/native_inputs.csv", DATA / "physics.py",
             DATA / "physics_config.json", DATA / "storage_kernel.c", DATA / "grid_config.json"]
protected += sorted((DATA / "hourly_inputs").glob("*.csv.gz"))
protected += sorted((PUBLIC / "plotting_tables").glob("*.csv"))
protected.append(NUMERIC / "expected/121_fig4_paired_contribution_display.csv")
before = {str(p): sha(p) for p in protected}
module("followup_package_integrity", PUBLIC / "verify_package.py").verify()
for item in json.loads((HERE / "SOURCE_MANIFEST.json").read_text())["packaged_inputs"]:
    assert sha(HERE / item["path"]) == item["sha256"], item["path"]
native = pd.read_csv(NUMERIC / "source_audit/native_inputs.csv")
rows = []
for region, years in [("MISO", [2027]), ("PJM", [2028, 2035])]:
    for year in years:
        n = native.loc[native.planning_region.eq(region) & native.year.eq(year)].iloc[0]
        q = (1 + n.reference_margin_level_pct / 100) * n.published_net_demand_MW / 1000
        for condition in ["anticipated", "prospective"]:
            resource = n[condition + "_native_MW"] / 1000
            rows.append(dict(region=region, summer_year=year, case="NERC-" + condition,
                             gross_GW=n.gross_peak_demand_MW / 1000,
                             separate_response_GW=n.published_response_MW / 1000,
                             resource_GW=resource, requirement_GW=q, balance_GW=resource-q,
                             rule=str(n.reference_margin_level_pct) + "% RML", rounded_source=False))

pjm = pd.read_csv(INPUTS / "pjm_rrs_table4.csv").set_index("summer_year")
for year in [2028, 2035]:
    row = pjm.loc[year]
    icap, ucap, irm, factor, fpr, gross, qi, qu, bi, bu = row[
        ["icap_MW", "ucap_MW", "irm_pct", "pool_accreditation_pct", "fpr", "gross_peak_MW",
         "icap_requirement_MW", "ucap_requirement_MW", "icap_surplus_MW", "ucap_surplus_MW"]]
    assert abs(icap - qi - bi) < 1e-9 and abs(ucap - qu - bu) < 1e-9
    assert abs(gross - native.loc[native.planning_region.eq("PJM") & native.year.eq(year), "gross_peak_demand_MW"].iloc[0]) < 1e-9
    for basis, resource, requirement, balance, rule in [
        ("ICAP", icap, qi, bi, f"{irm:.1f}% IRM"),
        ("UCAP", ucap, qu, bu, f"{fpr:.4f} FPR")]:
        rows.append(dict(region="PJM", summer_year=year, case="RRS-" + basis,
                         gross_GW=gross / 1000, separate_response_GW=np.nan,
                         resource_GW=resource / 1000, requirement_GW=requirement / 1000,
                         balance_GW=balance / 1000, rule=rule, rounded_source=False))

# The standardized chart values exclude unavailable resources; PRMR is inferred at display precision.
miso = pd.read_csv(INPUTS / "miso_oms_summer_components.csv")
for row in miso.itertuples():
    name, balance = row.resource_bookend, row.published_balance_GW
    resource = row.committed_GW + row.potential_new_GW + row.replacement_GW
    rows.append(dict(region="MISO", summer_year=2027, case="OMS-" + name,
                     gross_GW=np.nan, separate_response_GW=np.nan,
                     resource_GW=resource, requirement_GW=resource - balance,
                     balance_GW=balance, rule="Published PRMR", rounded_source=True))
reconciliation = pd.DataFrame(rows).sort_values(["region", "summer_year", "case"])
reconciliation.to_csv(OUT / "regional_input_reconciliation.csv", index=False)
assert np.max(np.abs(reconciliation.resource_GW - reconciliation.requirement_GW - reconciliation.balance_GW)) < 1e-10

formatted = []
for (region, year), group in reconciliation.groupby(["region", "summer_year"], sort=False):
    formatted.append(r"\multicolumn{7}{l}{\textbf{" + region + f", summer {year}; regional report {year}/{str(year+1)[-2:]}" + "}}")
    for r in group.itertuples():
        num = lambda value: "--" if pd.isna(value) else (f"{value:.1f}" if r.rounded_source else f"{value:.3f}")
        formatted.append([r.case, num(r.gross_GW), num(r.separate_response_GW), num(r.resource_GW),
                          r.rule.replace("%", r"\%"), num(r.requirement_GW), num(r.balance_GW)])
table(TABLES / "regional_inputs.tex", "Regional input reconciliation with native conventions retained.",
      "tab:regional-input-reconciliation", "L{0.19\\linewidth}rrrL{0.15\\linewidth}rr",
      ["Case", "$G$", "DR", "$R$", "Reference", "$Q$", "$B$"], formatted,
      "Values are native GW. NERC prospective resources are reconstructed from reported margins. "
      "PJM rows reproduce RRS Table 4; a dash is not zero response. MISO PRMR is inferred from rounded "
      "survey components and balances. Capacity conventions are not harmonized.")

print("Regional input reconciliation: 12 rows; PJM shared demand verified.", flush=True)

physics = module("followup_physics", DATA / "physics.py")
quality = module("followup_quality", NUMERIC / "scripts/seasonal_load_quality.py")
config = json.loads((DATA / "physics_config.json").read_text())
sites = pd.read_csv(DATA / "sites.csv")
audit = pd.read_csv(DATA / "weather_source_audit.csv")
provider_headers = []
snapshots = json.loads((INPUTS / "power_header_snapshots.json").read_text())["sites"]
hourbox = json.loads((INPUTS / "ceres_hourbox.json").read_text())
assert hourbox["time_basis"] == "GMT" and hourbox["first_hourbox_start"] == "00:00"
assert hourbox["first_hourbox_end"] == "01:00" and not hourbox["power_export_interval_mapping_certified"]
assert len(snapshots) == 20 and {s["site"] for s in snapshots} == set(sites.representative_id)
for raw in snapshots:
    item = audit.loc[audit.representative_id.eq(raw["site"])].iloc[0]
    header, parameter = raw["header"], raw["irradiance_parameter"]
    assert raw["archive_sha256"] == item.nasa_cache_sha256
    assert header["time_standard"] == "UTC" and "SYN1DEG" in header["sources"]
    assert parameter["units"] == "Wh/m^2"
    assert not {"time_bounds", "interval_start", "interval_end", "hour_interval"}.intersection(header)
    provider_headers.append(dict(site=raw["site"], time_standard=header["time_standard"],
                                 source=";".join(header["sources"]), units=parameter["units"],
                                 sha256=raw["archive_sha256"], hour_interval_bounds_in_header=False))
pd.DataFrame(provider_headers).to_csv(OUT / "power_export_header_check.csv", index=False)

library = OUT / ("storage_kernel.dylib" if sys.platform == "darwin" else "storage_kernel.so")
flags = ["cc", "-O3", "-dynamiclib"] if sys.platform == "darwin" else ["cc", "-O3", "-shared", "-fPIC"]
subprocess.run(flags + [str(DATA / "storage_kernel.c"), "-o", str(library)], check=True)
lib = ctypes.CDLL(str(library))
pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
lib.dispatch.argtypes = [pointer, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_size_t, pointer]
lib.dispatch.restype = None


def dispatch(pv, ratio, hours):
    pv = np.ascontiguousarray(pv, dtype=np.float64)
    residual = np.empty(len(pv), dtype=np.float64)
    lib.dispatch(pv, len(pv), ratio, hours, 0.85, 720, residual)
    return residual


random = np.random.default_rng(48).uniform(0, .85, 2000)
for ratio, hours in [(1, 0), (2, 4), (3, 8)]:
    assert np.max(np.abs(dispatch(random, ratio, hours) - physics.simulate_storage(random, ratio, hours, .85, config)["residual"])) < 1e-12

load = quality.reported_demand()
expected = pd.read_csv(DATA / "integrity_expected_pv_storage_fine_grid.csv")
window_rows = []
max_nominal_difference = 0.0
for region in ["PJM", "MISO", "SPP", "ERCOT"]:
    selected = sites.loc[sites.ISO.eq(region)].sort_values("representative_id")
    assert len(selected) == 5
    arrays = []
    for site in selected.itertuples():
        weather = pd.read_csv(DATA / "hourly_inputs" / f"{site.representative_id}.csv.gz")
        times = pd.DatetimeIndex(pd.to_datetime(weather.timestamp_utc, utc=True))
        assert len(times) == 87672 and times.equals(pd.date_range("2016-01-01", "2025-12-31 23:00", freq="h", tz="UTC"))
        converted = physics.solar_geometry_and_pv(times, site.latitude, site.longitude,
            weather.ghi_w_m2.to_numpy(), weather.air_temperature_c.to_numpy(), weather.wind_speed_m_s.to_numpy(), config)
        arrays.append(converted.pv_ac_kw_per_kwdc.to_numpy())
    windows = {n: quality.select_peaks(load, region, [2025], n)[0] for n in [50, 100, 200]}
    for n in [50, 100]:
        assert set(windows[n].timestamp_utc).issubset(set(windows[200].timestamp_utc))
    for ratio, hours in [(1, 0), (2, 4), (3, 8)]:
        site_residual = np.stack([dispatch(pv, ratio, hours) for pv in arrays])
        residual = site_residual.mean(axis=0)
        reference = expected.loc[expected.ISO.eq(region) & expected.pv_ratio.eq(ratio) & expected.storage_hours.eq(hours)].iloc[0]
        assert abs((1-residual.mean()) - reference.annual_displacement_share) < 1e-12
        for n, peaks in windows.items():
            indices = times.get_indexer(pd.DatetimeIndex(peaks.timestamp_utc))
            assert (indices >= 1).all() and (indices < len(times) - 1).all()
            days = pd.DatetimeIndex(peaks.timestamp_utc).tz_convert(quality.ZONES[region]).normalize().nunique()
            for shift in [-1, 0, 1]:
                tail = residual[indices + shift]
                p95 = float(np.quantile(tail, .95))
                if n == 100 and shift == 0:
                    delta = abs(p95 - reference.summer_peak_p95_residual)
                    max_nominal_difference = max(max_nominal_difference, delta)
                    assert delta < 1e-12
                window_rows.append(dict(region=region, pv_ratio=ratio, storage_hours=hours,
                    peak_hours=n, shift_h=shift, local_days=int(days), year=2025,
                    mean_residual_pct=100 * float(tail.mean()), p95_residual_pct=100 * p95,
                    sample="complete common 2025 summer", interval_certified=False))
    print(f"PV {region}: three fixed configurations, 50/100/200-hour windows and -1/0/+1 labels.", flush=True)
windows = pd.DataFrame(window_rows)
windows.to_csv(OUT / "pv_peak_windows_full.csv", index=False)
summary = []
for (region, ratio, hours), group in windows.groupby(["region", "pv_ratio", "storage_hours"], sort=False):
    nominal = group.loc[group.shift_h.eq(0)].set_index("peak_hours")
    summary.append(dict(region=region, pv_ratio=ratio, storage_hours=hours,
                        p95_50=nominal.loc[50, "p95_residual_pct"], p95_100=nominal.loc[100, "p95_residual_pct"],
                        p95_200=nominal.loc[200, "p95_residual_pct"], all_checks_min=group.p95_residual_pct.min(),
                        all_checks_max=group.p95_residual_pct.max()))
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "pv_peak_window_summary.csv", index=False)
table(TABLES / "pv_windows.tex", "Peak-window and label sensitivity for the displayed PV--storage configurations.",
      "tab:pv-peak-window-followup", "llrrrrr",
      ["Region", "PV/storage", "Top 50", "Top 100", "Top 200", "All checks min", "All checks max"],
      [[r.region, f"{int(r.pv_ratio)}x/{int(r.storage_hours)}h"] + [f"{v:.1f}" for v in
        [r.p95_50, r.p95_100, r.p95_200, r.all_checks_min, r.all_checks_max]] for r in summary.itertuples()],
      "Entries are P95 residual demand as a percentage of the constant load, after hourly averaging of five independently "
      "dispatched sites. Top-50/100/200 columns use nominal labels; the final columns span all three windows and all three "
      "label offsets (-1, 0, +1 h). Windows are nested selections from the same qualifying 2025 summer, not independent "
      "experiments or statistical confidence intervals. The physical conversion, dispatch and annual energy outputs are unchanged. "
      "The 100-hour nominal column reproduces the existing main-figure values. No offset is selected on performance.")

comparisons = []
for (ratio, hours, n, shift), group in windows.groupby(["pv_ratio", "storage_hours", "peak_hours", "shift_h"]):
    values = group.set_index("region").p95_residual_pct
    comparisons.append(dict(pv_ratio=ratio, storage_hours=hours, peak_hours=n, shift_h=shift,
                            ercot_p95=values["ERCOT"], spp_p95=values["SPP"], ercot_below_spp=bool(values["ERCOT"] < values["SPP"])))
pd.DataFrame(comparisons).to_csv(OUT / "pv_common_rule_comparisons.csv", index=False)
common = windows.pivot(index=["region", "peak_hours", "shift_h"], columns="pv_ratio", values="p95_residual_pct")
assert (common[3] <= common[2] + 1e-12).all() and (common[2] <= common[1] + 1e-12).all()

# Extract the contribution comparison from frozen full-precision outputs.
paired = pd.read_csv(NUMERIC / "expected/121_fig4_paired_contribution_display.csv")
paired = paired.loc[paired.year.eq(2035) & paired.linked_response_fraction.eq(0)]
mask = (paired.planning_region.eq("MISO") & paired.response_condition.eq("published_all")) | (
    paired.planning_region.eq("ERCOT") & paired.resource_condition.eq("anticipated"))
paired = paired.loc[mask, ["planning_region", "resource_condition", "response_condition",
    "source_path_removed_balance_GW", "replacement_balance_GW", "dc_path_pressure_contribution_GW",
    "dc_added_required_contribution_GW"]]
assert len(paired) == 4
assert np.max(np.abs(paired.source_path_removed_balance_GW - paired.replacement_balance_GW - paired.dc_path_pressure_contribution_GW)) < 1e-10
paired.to_csv(OUT / "contribution_paired_comparison.csv", index=False)
growth = pd.read_csv(PUBLIC / "plotting_tables/miso_annual_growth_boundaries.csv")
response = pd.read_csv(PUBLIC / "plotting_tables/ercot_annual_response_boundaries.csv")
limits = dict(miso_endpoint_pct=float(growth.loc[growth.year.eq(2035), "annual_limit_pct"].iloc[0]),
              miso_all_year_pct=float(growth.annual_limit_pct.min()),
              miso_limiting_year=int(growth.loc[growth.annual_limit_pct.idxmin(), "year"]),
              ercot_anticipated_all_year_response_pct=float(response.loc[response.resource_condition.eq("anticipated"), "response_limit_pct"].max()),
              contribution="Data-grounded regional--annual supply--demand scenario framework and conditional planning judgments",
              not_claimed=["new balance identity", "nodal power flow", "optimized investment portfolio", "validated future reliability prediction"])
(OUT / "contribution_conditions.json").write_text(json.dumps(limits, indent=2) + "\n")

comparisons = {
    "regional_input_reconciliation.csv": ["region", "summer_year", "case"],
    "power_export_header_check.csv": ["site"],
    "pv_peak_windows_full.csv": ["region", "pv_ratio", "storage_hours", "peak_hours", "shift_h"],
    "pv_peak_window_summary.csv": ["region", "pv_ratio", "storage_hours"],
    "pv_common_rule_comparisons.csv": ["pv_ratio", "storage_hours", "peak_hours", "shift_h"],
    "contribution_paired_comparison.csv": ["planning_region", "resource_condition", "response_condition"],
}
for name, keys in comparisons.items():
    actual, expected = pd.read_csv(OUT / name), pd.read_csv(EXPECTED / name)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), name
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  expected.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, rtol=1e-10, atol=1e-10)
assert limits == json.loads((EXPECTED / "contribution_conditions.json").read_text())
for name in ["regional_inputs.tex", "pv_windows.tex"]:
    assert (TABLES / name).read_bytes() == (EXPECTED / name).read_bytes(), name
assert all(sha(Path(p)) == value for p, value in before.items())
report = dict(status="passed", regional_input_rows=len(reconciliation),
              retained_header_snapshots_checked=len(provider_headers), raw_provider_archives_rechecked=0,
              pv_fixed_configurations=12, pv_site_dispatches=60, window_checks=len(windows),
              nominal_100_hour_max_difference=max_nominal_difference, protected_files_unchanged=len(before),
              main_annual_builder_run=False, full_676_grid_rebuilt=False, figures_redrawn=False,
              physical_interval_certified=False, expected_tables_matched=True,
              header_check_scope="Retained metadata and recorded archive hashes; no raw-provider download or reread.")
(OUT / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(summary.to_string(index=False), flush=True)
print(json.dumps(report, indent=2), flush=True)
