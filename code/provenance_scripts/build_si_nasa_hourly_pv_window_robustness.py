from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT / "out_std" / "ANALYSIS"
CANON = ROOT / "tables" / "fig2_5_canonical_20260514"
OUT = ROOT / "tables" / "si_round4_nasa_hourly"
OUT.mkdir(parents=True, exist_ok=True)

ISO4 = ["PJM", "MISO", "SPP", "ERCOT"]
SCENARIOS = ["low", "mid", "high"]
PV_GRID = np.round(np.arange(0.0, 2.0001, 0.05), 4)
STORAGE_H = 4.0
START = "20250601"
END = "20250831"
NASA_PARAMETER = "ALLSKY_SFC_SW_DWN"
NASA_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/hourly/point"
TIME_STANDARD = "LST"
COMMUNITY = "RE"
RATE_LIMIT_SECONDS = 0.08

PANEL_PATHS = {
    "PJM": ANALYSIS / "PJM_HC_D" / "panel_hourly.parquet",
    "MISO": ANALYSIS / "MISO_HC_D" / "panel_hourly.parquet",
    "SPP": ANALYSIS / "SPP_HC_D" / "panel_hourly_zoneproxy.parquet",
    "ERCOT": ANALYSIS / "ERCOT_HC_D" / "panel_hourly_4zone.parquet",
}
ISO_TZ = {
    "PJM": "America/New_York",
    "MISO": "America/Chicago",
    "SPP": "America/Chicago",
    "ERCOT": "America/Chicago",
}


def tex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def fmt_threshold(row: pd.Series) -> str:
    if bool(row.get("threshold_reached", False)):
        return f"{float(row['pv_ratio_for_1pct_residual']):.2f}"
    return ">2.00"


def load_onsite_module():
    path = PROJECT / "scripts" / "make_fig_onsite_panel_d.py"
    spec = importlib.util.spec_from_file_location("onsite_panel_d_canonical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nasa_url(lat: float, lon: float) -> str:
    params = {
        "parameters": NASA_PARAMETER,
        "community": COMMUNITY,
        "longitude": f"{float(lon):.4f}",
        "latitude": f"{float(lat):.4f}",
        "start": START,
        "end": END,
        "format": "JSON",
        "time-standard": TIME_STANDARD,
    }
    return NASA_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_site(site: pd.Series, timeout: int = 45) -> tuple[list[dict[str, object]], dict[str, object]]:
    url = nasa_url(site["lat_sent"], site["lon_sent"])
    status = "ok"
    status_code = None
    payload = b""
    error = ""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-datacenter-si-nasa-hourly-audit/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = getattr(resp, "status", None)
            payload = resp.read()
        data = json.loads(payload.decode("utf-8"))
        vals = data["properties"]["parameter"][NASA_PARAMETER]
        records = [
            {
                "site_id": site["site_id"],
                "ISO": site["ISO"],
                "site_weight_mw": float(site["site_weight_mw"]),
                "lat_sent": float(site["lat_sent"]),
                "lon_sent": float(site["lon_sent"]),
                "hour_key_lst": key,
                "ghi_wh_m2_proxy": float(value) if value is not None else np.nan,
            }
            for key, value in vals.items()
        ]
    except Exception as exc:
        status = "error"
        error = repr(exc)
        records = []
    log = {
        "site_id": site["site_id"],
        "ISO": site["ISO"],
        "lat_sent": float(site["lat_sent"]),
        "lon_sent": float(site["lon_sent"]),
        "site_weight_mw": float(site["site_weight_mw"]),
        "api_endpoint": NASA_ENDPOINT,
        "parameter": NASA_PARAMETER,
        "community": COMMUNITY,
        "start": START,
        "end": END,
        "time_standard": TIME_STANDARD,
        "request_url": url,
        "http_status": status_code,
        "status": status,
        "n_hourly_values": len(records),
        "response_sha256": hashlib.sha256(payload).hexdigest() if payload else "",
        "error": error,
    }
    return records, log


def build_or_load_site_hourly(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_path = OUT / "nasa_power_hourly_site_irradiance_2025_jja.csv.gz"
    log_path = OUT / "nasa_power_hourly_request_log.csv"
    if hourly_path.exists() and log_path.exists() and not refresh:
        hourly = pd.read_csv(hourly_path)
        log = pd.read_csv(log_path)
        return hourly, log

    sites = pd.read_csv(CANON / "fig5" / "fig5a_site_level_solar_resource_source.csv")
    sites = sites[sites["ISO"].isin(ISO4)].copy()
    sites["site_weight_mw"] = pd.to_numeric(sites["site_weight_mw"], errors="coerce").fillna(
        pd.to_numeric(sites["mw_peak_mw"], errors="coerce")
    )
    sites = sites.dropna(subset=["site_id", "ISO", "lat_sent", "lon_sent", "site_weight_mw"])
    sites = sites[sites["site_weight_mw"] > 0].sort_values(["ISO", "site_id"]).reset_index(drop=True)

    all_records: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    for idx, site in sites.iterrows():
        records, log = fetch_site(site)
        logs.append(log)
        all_records.extend(records)
        print(f"[NASA] {idx + 1:03d}/{len(sites):03d} {site['ISO']} {site['site_id']} {log['status']} {log['n_hourly_values']} hours")
        time.sleep(RATE_LIMIT_SECONDS)

    hourly = pd.DataFrame(all_records)
    log = pd.DataFrame(logs)
    if hourly.empty:
        raise RuntimeError("No NASA POWER hourly records were retrieved.")
    hourly.to_csv(hourly_path, index=False, compression="gzip")
    log.to_csv(log_path, index=False)
    print(f"[wrote] {hourly_path.relative_to(ROOT)} ({len(hourly):,} rows)")
    print(f"[wrote] {log_path.relative_to(ROOT)} ({len(log):,} rows)")
    return hourly, log


def weighted_iso_profiles(site_hourly: pd.DataFrame) -> pd.DataFrame:
    df = site_hourly.copy()
    df["datetime_lst"] = pd.to_datetime(df["hour_key_lst"], format="%Y%m%d%H", errors="coerce")
    df["date_lst"] = df["datetime_lst"].dt.date.astype(str)
    df["hour_1_24"] = df["datetime_lst"].dt.hour + 1
    df["ghi_wh_m2_proxy"] = pd.to_numeric(df["ghi_wh_m2_proxy"], errors="coerce").clip(lower=0)
    df["weighted_ghi"] = df["ghi_wh_m2_proxy"] * df["site_weight_mw"]
    agg = (
        df.groupby(["ISO", "hour_key_lst", "datetime_lst", "date_lst", "hour_1_24"], as_index=False)
        .agg(weighted_ghi_sum=("weighted_ghi", "sum"), weight_mw=("site_weight_mw", "sum"), n_sites=("site_id", "nunique"))
    )
    agg["site_weighted_ghi_wh_m2_proxy"] = agg["weighted_ghi_sum"] / agg["weight_mw"]

    canonical = pd.read_csv(ANALYSIS / "iso7_summer2025_eia930_mean_mw_by_local_hour_jja.csv")
    canonical = canonical[canonical["iso"].isin(ISO4)].copy()
    stats = canonical.groupby("iso")["pv_nonneg_per_mwac"].agg(canonical_jja_mean_phi="mean", canonical_jja_max_phi="max")
    raw_stats = agg.groupby("ISO")["site_weighted_ghi_wh_m2_proxy"].agg(raw_mean="mean", raw_max="max")
    scale = stats.join(raw_stats, how="inner")
    scale["raw_pu_mean"] = scale["raw_mean"] / 1000.0
    scale["hourly_scale_to_canonical_mean"] = scale["canonical_jja_mean_phi"] / scale["raw_pu_mean"].replace(0, np.nan)
    agg = agg.merge(scale[["hourly_scale_to_canonical_mean", "canonical_jja_mean_phi", "canonical_jja_max_phi"]], left_on="ISO", right_index=True, how="left")
    agg["raw_ghi_pu"] = agg["site_weighted_ghi_wh_m2_proxy"] / 1000.0
    agg["pv_pu_scaled_to_canonical_jja_mean"] = (
        agg["raw_ghi_pu"] * agg["hourly_scale_to_canonical_mean"]
    ).clip(lower=0, upper=1.0)
    out = agg[
        [
            "ISO",
            "hour_key_lst",
            "datetime_lst",
            "date_lst",
            "hour_1_24",
            "n_sites",
            "weight_mw",
            "site_weighted_ghi_wh_m2_proxy",
            "raw_ghi_pu",
            "hourly_scale_to_canonical_mean",
            "canonical_jja_mean_phi",
            "canonical_jja_max_phi",
            "pv_pu_scaled_to_canonical_jja_mean",
        ]
    ].sort_values(["ISO", "datetime_lst"])
    path = OUT / "nasa_power_hourly_iso_profiles_2025_jja.csv"
    out.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(out):,} rows)")
    return out


def peak_windows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for iso, path in PANEL_PATHS.items():
        df = pd.read_parquet(path)
        if "zone" in df.columns:
            df = df[~df["zone"].astype(str).str.upper().str.contains("TOTAL", na=False)].copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
        df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
        # Convert UTC market timestamps to local clock time for selecting the
        # summer peak week.  Dropping the timezone before filtering avoids DST
        # ambiguity in non-summer months that are present in the source panels.
        local = df["ts_utc"].dt.tz_convert(ISO_TZ[iso])
        df["local_ts"] = local.dt.tz_localize(None)
        df = df[(df["local_ts"].dt.date >= date(2025, 6, 1)) & (df["local_ts"].dt.date <= date(2025, 8, 31))]
        hourly = df.groupby("local_ts", as_index=False)["load_mw"].sum().dropna()
        peak = hourly.loc[hourly["load_mw"].idxmax()]
        peak_date = peak["local_ts"].date()
        for case, shift_days in [
            ("one_week_earlier", -7),
            ("peak_load_week", 0),
            ("one_week_later", 7),
        ]:
            center = peak_date + timedelta(days=shift_days)
            start = center - timedelta(days=3)
            end = center + timedelta(days=3)
            rows.append(
                {
                    "ISO": iso,
                    "window_case": case,
                    "peak_load_local_timestamp": peak["local_ts"].isoformat(),
                    "peak_load_mw": float(peak["load_mw"]),
                    "window_center_date": center.isoformat(),
                    "window_start_date": start.isoformat(),
                    "window_end_date": end.isoformat(),
                    "load_panel_source": str(path),
                }
            )
    out = pd.DataFrame(rows).sort_values(["ISO", "window_case"])
    path = OUT / "nasa_power_hourly_peak_windows.csv"
    out.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(out):,} rows)")
    return out


def load_peak_hour() -> pd.Series:
    pc = pd.read_csv(ANALYSIS / "iso_coincidence_peak_window_3h.csv")
    pc = pc[(pc["dc_shape_name"].eq("avg")) & (pc["ISO"].isin(ISO4))].copy()
    return pc.set_index("ISO")["iso_peak_hour"].astype(int)


def week_phi(profile: pd.DataFrame, iso: str, start_date: str, end_date: str) -> np.ndarray:
    sub = profile[
        (profile["ISO"].eq(iso))
        & (profile["date_lst"] >= start_date)
        & (profile["date_lst"] <= end_date)
    ].sort_values("datetime_lst")
    if len(sub) != 168:
        raise ValueError(f"Expected 168 hourly NASA records for {iso} {start_date}--{end_date}; found {len(sub)}")
    return sub["pv_pu_scaled_to_canonical_jja_mean"].to_numpy(dtype=float)


def solve_screen(profile: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mod = load_onsite_module()
    peak_hour = load_peak_hour()
    source = pd.read_csv(CANON / "fig5" / "fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")
    source = source[
        source["year"].eq(2035)
        & source["ISO_RTO"].isin(ISO4)
        & np.isclose(source["storage_duration_h"], STORAGE_H)
        & source["scenario"].isin(SCENARIOS)
        & np.isclose(source["storage_power_ratio_to_pv_capacity"], mod.R_SP)
    ].copy()
    base = source.drop_duplicates(["ISO_RTO", "scenario"])[["ISO_RTO", "scenario", "HC_MW", "incremental_ai_peak_MW", "nonAI_growth_MW"]]
    wlookup = windows.set_index(["ISO", "window_case"])

    rows: list[dict[str, object]] = []
    for _, b in base.iterrows():
        iso = str(b["ISO_RTO"])
        scenario = str(b["scenario"])
        hc_mw = float(b["HC_MW"])
        ai_mw = float(b["incremental_ai_peak_MW"])
        nonai_mw = float(b["nonAI_growth_MW"])
        for window_case in ["one_week_earlier", "peak_load_week", "one_week_later"]:
            win = wlookup.loc[(iso, window_case)]
            phi = week_phi(profile, iso, str(win["window_start_date"]), str(win["window_end_date"]))
            w0 = mod.w_idx0_from_peak_multi(int(peak_hour.loc[iso]), mod.N_DAYS)
            for pv_ratio in PV_GRID:
                pv_cap_mw = max(0.0, pv_ratio * ai_mw)
                p_sto_mw = mod.R_SP * pv_cap_mw
                e_mwh = STORAGE_H * p_sto_mw
                g = mod.g_hourly_mw(ai_mw, pv_ratio, phi)
                pv = pv_ratio * ai_mw * phi
                if STORAGE_H <= 0.0 or p_sto_mw <= 0.0:
                    residual_peak = mod.b_no_storage_mw(g, w0)
                else:
                    residual_peak = mod.b_with_storage_mw(g, pv, w0, p_sto_mw, e_mwh)
                available_margin = hc_mw - residual_peak
                residual_firm = max(0.0, -available_margin)
                residual_share = residual_firm / ai_mw if ai_mw > 0 else 0.0
                rows.append(
                    {
                        "ISO": iso,
                        "scenario": scenario,
                        "year": 2035,
                        "window_case": window_case,
                        "window_start_date": win["window_start_date"],
                        "window_end_date": win["window_end_date"],
                        "peak_load_local_timestamp": win["peak_load_local_timestamp"],
                        "storage_duration_h": STORAGE_H,
                        "storage_power_ratio_to_pv_capacity": float(mod.R_SP),
                        "pv_nameplate_ratio_to_incremental_ai_peak": float(pv_ratio),
                        "HC_MW": hc_mw,
                        "nonAI_growth_MW": nonai_mw,
                        "incremental_ai_peak_MW": ai_mw,
                        "PV_cap_MW": pv_cap_mw,
                        "P_sto_MW": p_sto_mw,
                        "E_MWh": e_mwh,
                        "residual_ai_peak_after_pv_storage_MW": residual_peak,
                        "available_margin_MW_after_pv_storage": available_margin,
                        "residual_firm_backstop_required_MW": residual_firm,
                        "residual_firm_share_of_incremental_ai_peak": residual_share,
                        "profile_source": "NASA POWER hourly ALLSKY_SFC_SW_DWN, site-weighted by AI site size, scaled to canonical JJA mean PV profile",
                        "script_path": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
                    }
                )

    detailed = pd.DataFrame(rows).sort_values(["scenario", "ISO", "window_case", "pv_nameplate_ratio_to_incremental_ai_peak"])
    detailed_path = OUT / "nasa_power_hourly_pv_storage_window_results.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"[wrote] {detailed_path.relative_to(ROOT)} ({len(detailed):,} rows)")

    summary_rows: list[dict[str, object]] = []
    for (iso, scenario, window_case), g in detailed.groupby(["ISO", "scenario", "window_case"]):
        good = g[g["residual_firm_share_of_incremental_ai_peak"] <= 0.01].sort_values("pv_nameplate_ratio_to_incremental_ai_peak")
        if good.empty:
            sel = g.sort_values("pv_nameplate_ratio_to_incremental_ai_peak").iloc[-1]
            threshold = np.nan
            reached = False
        else:
            sel = good.iloc[0]
            threshold = float(sel["pv_nameplate_ratio_to_incremental_ai_peak"])
            reached = True
        summary_rows.append(
            {
                "ISO": iso,
                "scenario": scenario,
                "year": 2035,
                "window_case": window_case,
                "window_start_date": sel["window_start_date"],
                "window_end_date": sel["window_end_date"],
                "peak_load_local_timestamp": sel["peak_load_local_timestamp"],
                "storage_duration_h": STORAGE_H,
                "pv_ratio_for_1pct_residual": threshold,
                "threshold_reached": reached,
                "selected_residual_firm_gap_percent": float(sel["residual_firm_share_of_incremental_ai_peak"]) * 100.0,
                "selected_residual_firm_gap_GW": float(sel["residual_firm_backstop_required_MW"]) / 1000.0,
                "selected_available_margin_GW": float(sel["available_margin_MW_after_pv_storage"]) / 1000.0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "ISO", "window_case"])
    summary_path = OUT / "nasa_power_hourly_pv_storage_window_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[wrote] {summary_path.relative_to(ROOT)} ({len(summary):,} rows)")
    return detailed, summary


def write_tex_table(summary: pd.DataFrame) -> None:
    order = ["PJM", "MISO", "SPP", "ERCOT"]
    scenario_order = ["low", "mid", "high"]
    window_order = ["one_week_earlier", "peak_load_week", "one_week_later"]
    pivot = summary.set_index(["ISO", "scenario", "window_case"])
    rows = []
    for iso in order:
        for scenario in scenario_order:
            vals = []
            reached_flags = []
            for window in window_order:
                r = pivot.loc[(iso, scenario, window)]
                vals.append(fmt_threshold(r))
                reached_flags.append(bool(r["threshold_reached"]))
            finite = [
                float(pivot.loc[(iso, scenario, w)]["pv_ratio_for_1pct_residual"])
                for w in window_order
                if bool(pivot.loc[(iso, scenario, w)]["threshold_reached"])
            ]
            spread = "--" if len(finite) < 2 else f"{max(finite) - min(finite):.2f}"
            flag = (
                "all windows reach threshold"
                if all(reached_flags)
                else r"threshold not reached by 200\% PV in at least one window"
            )
            rows.append(
                f"{tex_escape(iso)} & {tex_escape(scenario)} & {vals[0]} & {vals[1]} & {vals[2]} & {spread} & {flag} \\\\"
            )
    rows_tex = "\n".join(rows)
    tex = rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Date-resolved NASA POWER hourly PV-window robustness screen.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l l r r r r L{{0.34\linewidth}}}}
\toprule
Grid region & Scenario & Earlier week & Peak-load week & Later week & Max--min & Interpretation \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{Entries report the minimum PV nameplate ratio, relative to incremental AI peak load, required to reduce the AI-attributable residual firm requirement to at most 1\% with 4 h storage and storage power equal to 50\% of PV capacity. The peak-load week is the seven-day window centered on the 2025 summer peak-load date in the ISO/RTO hourly load panel; earlier and later weeks shift that calendar window by seven days. Hourly PV profiles use NASA POWER hourly \texttt{{ALLSKY\_SFC\_SW\_DWN}} at the AI-site coordinates, site-weighted by modeled site size and scaled to the canonical JJA mean PV profile used in the main Fig. 5 screen. This is a date-resolved screening robustness check, not site-specific PV engineering design. Values shown as \(>2.00\) do not reach the 1\% threshold within the evaluated 0--200\% PV grid.}}
\end{{table}}
\endgroup
"""
    path = OUT / "nasa_power_hourly_pv_window_summary_table.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"[wrote] {path.relative_to(ROOT)}")


def write_manifest(log: pd.DataFrame, profile: pd.DataFrame, detailed: pd.DataFrame, summary: pd.DataFrame) -> None:
    manifest = pd.DataFrame(
        [
            {
                "artifact": "nasa_power_hourly_site_irradiance_2025_jja.csv.gz",
                "rows": len(pd.read_csv(OUT / "nasa_power_hourly_site_irradiance_2025_jja.csv.gz")),
                "description": "site-hour NASA POWER hourly ALLSKY_SFC_SW_DWN records for AI-site coordinates in PJM, MISO, SPP and ERCOT",
                "source": NASA_ENDPOINT,
                "script": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
            },
            {
                "artifact": "nasa_power_hourly_request_log.csv",
                "rows": len(log),
                "description": "request URL, coordinate, response hash and status for each NASA POWER hourly API query",
                "source": NASA_ENDPOINT,
                "script": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
            },
            {
                "artifact": "nasa_power_hourly_iso_profiles_2025_jja.csv",
                "rows": len(profile),
                "description": "site-weighted ISO hourly irradiance proxy and scaled PV profile",
                "source": "site-hour NASA POWER output plus canonical JJA mean PV profile",
                "script": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
            },
            {
                "artifact": "nasa_power_hourly_pv_storage_window_results.csv",
                "rows": len(detailed),
                "description": "PV-ratio sweep for date-resolved peak-load, earlier and later windows",
                "source": "canonical Fig. 5 PV-storage grid assumptions plus NASA hourly profiles",
                "script": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
            },
            {
                "artifact": "nasa_power_hourly_pv_storage_window_summary.csv",
                "rows": len(summary),
                "description": "threshold summary printed in SI",
                "source": "nasa_power_hourly_pv_storage_window_results.csv",
                "script": "scripts/build_si_nasa_hourly_pv_window_robustness.py",
            },
        ]
    )
    path = OUT / "nasa_power_hourly_source_manifest.csv"
    manifest.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-query NASA POWER even if cached site-hour file exists")
    args = parser.parse_args()
    hourly, log = build_or_load_site_hourly(refresh=args.refresh)
    if (log["status"] != "ok").any():
        bad = log[log["status"] != "ok"]
        raise RuntimeError(f"NASA POWER request failures: {bad[['site_id','ISO','error']].head().to_dict('records')}")
    profile = weighted_iso_profiles(hourly)
    windows = peak_windows()
    detailed, summary = solve_screen(profile, windows)
    write_tex_table(summary)
    write_manifest(log, profile, detailed, summary)


if __name__ == "__main__":
    main()
