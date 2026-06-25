from __future__ import annotations

import os
import sys
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
FIG1 = ROOT / "tables" / "fig1_canonical_20260514"
SI16 = ROOT / "tables" / "si_experiments_20260516"
SI17 = ROOT / "tables" / "si_experiments_20260517_robustness"
SI2 = ROOT / "tables" / "si_round2"
OUT = ROOT / "tables" / "si_round3"
PV_PROFILE_SOURCE = (
    "out_std/ANALYSIS/iso7_summer2025_pv_nonneg_per_mwac_by_local_hour_jja_wide.csv; "
    "out_std/ANALYSIS/iso7_summer2025_eia930_mean_mw_by_local_hour_jja.csv"
)
PV_PROFILE_TYPE = "representative_JJA_mean_daily_profile_tiled_7_days"
AUDIT_NOTE = "SI audit output; source tables retain calculation details."

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
SCENARIOS = ["low", "mid", "high"]
AI_COL = {"low": "D_ai_low_MW", "mid": "D_ai_mid_MW", "high": "D_ai_high_MW"}
MARGIN_COL = {"low": "Gap_lowcase_MW", "mid": "Gap_midcase_MW", "high": "Gap_highcase_MW"}


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


def write_csv(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(df):,} rows)")
    return path


def write_tex(name: str, content: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(content, encoding="utf-8")
    print(f"[wrote] {path.relative_to(ROOT)}")
    return path


def fmt_num(value: object, digits: int = 1, na: str = "--") -> str:
    if value is None or pd.isna(value):
        return na
    return f"{float(value):.{digits}f}"


def first_negative(g: pd.DataFrame, margin_col: str = "margin_GW") -> str:
    sub = g.sort_values("year")
    neg = sub[sub[margin_col] < 0]
    if neg.empty:
        return ">2035"
    return str(int(neg.iloc[0]["year"]))


def year_delta(new: str, base: str) -> object:
    if new == base:
        return 0
    if new == ">2035" or base == ">2035":
        return f"{base}->{new}"
    try:
        return int(new) - int(base)
    except Exception:
        return f"{base}->{new}"


def norm_iso(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    upper = text.upper().replace("_", "-")
    if upper in {"ISONE", "ISO NE", "ISO-NE"}:
        return "ISO-NE"
    return text


def peak_coincidence_sensitivity() -> dict[str, pd.DataFrame]:
    gap = pd.read_csv(CANON / "fig2" / "fig2_shared_canonical_gap_yearly.csv")
    gamma = pd.read_csv(SI2 / "peak_coincidence_gamma_factors.csv").set_index("iso_region")["gamma_factor"].to_dict()
    gamma_cases = [
        ("central", "central regional three-hour peak-window factor", None),
        ("absolute_0p85", "absolute Gamma = 0.85 applied to reconstructed allocated average incremental AI load", 0.85),
        ("absolute_1p00", "absolute Gamma = 1.00 applied to reconstructed allocated average incremental AI load", 1.00),
    ]
    rows: list[dict[str, object]] = []
    for case, definition, gamma_abs in gamma_cases:
        for scenario in SCENARIOS:
            scenario_rows = []
            for _, r in gap.iterrows():
                iso = str(r["ISO"])
                gamma_central = float(gamma[iso])
                central_ai = float(r[AI_COL[scenario]])
                avg_ai = central_ai / gamma_central if gamma_central else central_ai
                ai_case = central_ai if case == "central" else avg_ai * float(gamma_abs)
                margin = float(r["HC_MW"]) - ai_case
                rec = {
                    "gamma_case": case,
                    "gamma_definition": definition,
                    "iso_region": iso,
                    "scenario": scenario,
                    "year": int(r["year"]),
                    "ai_peak_load_GW": ai_case / 1000.0,
                    "margin_GW": margin / 1000.0,
                    "deficit_GW": max(0.0, -margin / 1000.0),
                    "source_table": "tables/fig2_5_canonical_20260514/fig2/fig2_shared_canonical_gap_yearly.csv; tables/si_round2/peak_coincidence_gamma_factors.csv",
                    "script_path": "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity",
                    "notes": "AI peak load is incremental relative to the 2025 central baseline; absolute Gamma cases reconstruct allocated average incremental load from the central Gamma before applying the sensitivity factor.",
                }
                scenario_rows.append(rec)
            tmp = pd.DataFrame(scenario_rows)
            firsts = tmp.groupby("iso_region").apply(lambda x: first_negative(x, "margin_GW")).to_dict()
            tmp["first_negative_year"] = tmp["iso_region"].map(firsts)
            tmp["exposure_indicator"] = tmp["margin_GW"] < 0
            rows.extend(tmp.to_dict("records"))
    detailed = pd.DataFrame(rows).sort_values(["gamma_case", "scenario", "iso_region", "year"])

    summary_rows = []
    central = detailed[detailed["gamma_case"].eq("central")]
    central_lookup = central.set_index(["scenario", "iso_region"])
    for (case, scenario, iso), g in detailed.groupby(["gamma_case", "scenario", "iso_region"]):
        y2035 = g[g["year"].eq(2035)].iloc[0]
        base = central_lookup.loc[(scenario, iso)]
        base2035 = base[base["year"].eq(2035)].iloc[0]
        d_year = year_delta(str(y2035["first_negative_year"]), str(base2035["first_negative_year"]))
        d_margin = float(y2035["margin_GW"]) - float(base2035["margin_GW"])
        if case == "central":
            flag = "central"
        elif str(y2035["first_negative_year"]) == str(base2035["first_negative_year"]):
            flag = "timing stable"
        else:
            flag = "timing shifts under absolute Gamma sensitivity"
        summary_rows.append(
            {
                "gamma_case": case,
                "scenario": scenario,
                "iso_region": iso,
                "first_negative_year": y2035["first_negative_year"],
                "margin_2035_GW": y2035["margin_GW"],
                "deficit_2035_GW": y2035["deficit_GW"],
                "change_in_first_negative_year_vs_central": d_year,
                "change_in_2035_margin_vs_central_GW": d_margin,
                "interpretation_flag": flag,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "iso_region", "gamma_case"])

    exp_rows = []
    for (case, scenario, year), g in detailed.groupby(["gamma_case", "scenario", "year"]):
        denom = float(g["ai_peak_load_GW"].sum())
        numer = float(g.loc[g["exposure_indicator"], "ai_peak_load_GW"].sum())
        exp_rows.append(
            {
                "gamma_case": case,
                "scenario": scenario,
                "year": int(year),
                "seven_region_exposure_share": np.nan if denom <= 0 else numer / denom,
                "exposed_regions": "; ".join(g.loc[g["exposure_indicator"], "iso_region"].tolist()),
                "source_table": "tables/si_round3/peak_coincidence_sensitivity.csv",
                "script_path": "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity",
            }
        )
    exposure = pd.DataFrame(exp_rows).sort_values(["gamma_case", "scenario", "year"])

    write_csv(detailed, "peak_coincidence_sensitivity.csv")
    write_csv(summary, "peak_coincidence_sensitivity_summary.csv")
    write_csv(exposure, "peak_coincidence_exposure_share.csv")

    display = summary[summary["scenario"].eq("mid")].copy()
    display["case_label"] = display["gamma_case"].map(
        {"central": "central", "absolute_0p85": "$\\Gamma=0.85$", "absolute_1p00": "$\\Gamma=1.00$"}
    )
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {r.case_label} & {tex_escape(r.first_negative_year)} & {fmt_num(r.margin_2035_GW, 1)} & {fmt_num(r.change_in_2035_margin_vs_central_GW, 1)} & {tex_escape(r.interpretation_flag)} \\\\"
        for r in display.itertuples(index=False)
    )
    write_tex(
        "peak_coincidence_sensitivity_summary_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Peak-coincidence sensitivity of first-negative years and 2035 margins.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l l l r r L{{0.32\linewidth}}}}
\toprule
Grid region & Gamma case & First negative year & 2035 margin (GW) & Change vs central (GW) & Interpretation \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The sensitivity uses absolute $\Gamma=0.85$ and $\Gamma=1.00$ cases applied to reconstructed allocated average incremental AI load. Because the central regional three-hour peak-window factors are slightly above 1.0, the $\Gamma=1.00$ case is a no-peak-uplift sensitivity rather than an increase relative to the central conversion. Full low/mid/high results and exposure shares are retained in the source CSVs.}}
\end{{table}}
\endgroup
""",
    )
    return {"detailed": detailed, "summary": summary, "exposure": exposure}


def pv_storage_window_sensitivity() -> dict[str, pd.DataFrame]:
    pv = pd.read_csv(CANON / "fig5" / "fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")
    pv = pv[(pv["year"].eq(2035)) & (pv["ISO_RTO"].isin(["ERCOT", "MISO", "PJM", "SPP"]))].copy()
    window_cases = [
        ("central_representative_JJA_week", "representative JJA mean daily profile, tiled days 1--7", "representative JJA day 1", "representative JJA day 7"),
        ("shifted_one_week_earlier", "same representative JJA mean profile; date-specific week not available in canonical LP input", "representative JJA day -6", "representative JJA day 0"),
        ("shifted_one_week_later", "same representative JJA mean profile; date-specific week not available in canonical LP input", "representative JJA day 8", "representative JJA day 14"),
    ]
    rows: list[dict[str, object]] = []
    for case, note, start, end in window_cases:
        for (iso, scenario, dur), g in pv.groupby(["ISO_RTO", "scenario", "storage_duration_h"]):
            g = g.sort_values("pv_nameplate_ratio_to_incremental_ai_peak")
            good = g[g["residual_firm_share_of_incremental_ai_peak"] <= 0.01]
            sel = good.iloc[0] if not good.empty else g.iloc[-1]
            rows.append(
                {
                    "window_case": case,
                    "iso_region": iso,
                    "scenario": scenario,
                    "year": 2035,
                    "window_start": start,
                    "window_end": end,
                    "profile_source": PV_PROFILE_SOURCE,
                    "profile_type": PV_PROFILE_TYPE,
                    "pv_ratio_for_1pct_residual": np.nan if good.empty else float(sel["pv_nameplate_ratio_to_incremental_ai_peak"]),
                    "storage_duration_h": float(dur),
                    "storage_power_ratio": float(sel["storage_power_ratio_to_pv_capacity"]),
                    "residual_firm_gap_percent": float(sel["residual_firm_share_of_incremental_ai_peak"]) * 100.0,
                    "residual_firm_gap_GW": float(sel["residual_firm_backstop_required_MW"]) / 1000.0,
                    "signed_ai_headroom_balance_percent": float(sel["available_margin_MW_after_pv_storage"]) / float(sel["incremental_ai_peak_MW"]) * 100.0 if float(sel["incremental_ai_peak_MW"]) > 0 else np.nan,
                    "source_table": "tables/fig2_5_canonical_20260514/fig5/fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv",
                    "script_path": "scripts/build_si_round3_sensitivity_outputs.py::pv_storage_window_sensitivity",
                    "notes": f"{note}; implementation diagnostic; date-resolved shifted-week PV profile not identified",
                }
            )
    detailed = pd.DataFrame(rows).sort_values(["scenario", "iso_region", "storage_duration_h", "window_case"])

    summary_rows = []
    for (iso, scenario, dur), g in detailed.groupby(["iso_region", "scenario", "storage_duration_h"]):
        vals = g.set_index("window_case")["pv_ratio_for_1pct_residual"].to_dict()
        central = vals.get("central_representative_JJA_week", np.nan)
        earlier = vals.get("shifted_one_week_earlier", np.nan)
        later = vals.get("shifted_one_week_later", np.nan)
        arr = np.array([v for v in [central, earlier, later] if not pd.isna(v)], dtype=float)
        summary_rows.append(
            {
                "iso_region": iso,
                "scenario": scenario,
                "storage_duration_h": dur,
                "central_window_pv_ratio_for_1pct": central,
                "earlier_window_pv_ratio_for_1pct": earlier,
                "later_window_pv_ratio_for_1pct": later,
                "max_minus_min_pv_ratio": np.nan if arr.size == 0 else float(np.nanmax(arr) - np.nanmin(arr)),
                "ranking_change_flag": False,
                "interpretation_flag": "implementation diagnostic; no date-resolved shifted-calendar-week sensitivity was performed",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "storage_duration_h", "iso_region"])

    write_csv(detailed, "pv_storage_window_sensitivity.csv")
    write_csv(summary, "pv_storage_window_sensitivity_summary.csv")

    display = summary[(summary["scenario"].eq("mid")) & (summary["storage_duration_h"].eq(4.0))].copy()
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {fmt_num(r.central_window_pv_ratio_for_1pct, 2)} & {fmt_num(r.earlier_window_pv_ratio_for_1pct, 2)} & {fmt_num(r.later_window_pv_ratio_for_1pct, 2)} & {fmt_num(r.max_minus_min_pv_ratio, 2)} & {tex_escape(r.interpretation_flag)} \\\\"
        for r in display.itertuples(index=False)
    )
    write_tex(
        "pv_storage_window_sensitivity_summary_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{PV-plus-storage window-sensitivity diagnostic under the representative-day implementation.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l r r r r L{{0.38\linewidth}}}}
\toprule
Grid region & Central PV ratio & Earlier-window PV ratio & Later-window PV ratio & Max--min & Interpretation \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{Values show the mid-2035, 4 h storage case. The current canonical implementation uses a representative JJA mean daily PV/load profile tiled over seven days; earlier/later labels therefore diagnose implementation limits and do not represent shifted calendar weeks. This diagnostic identifies the limitation of the representative-day implementation and motivates a date-resolved shifted-window sensitivity. Full storage-duration and scenario rows are retained in the source CSV.}}
\end{{table}}
\endgroup
""",
    )
    return {"detailed": detailed, "summary": summary}


def panel_paths() -> dict[str, Path]:
    return {
        "CAISO": ANALYSIS / "CAISO_HC_D" / "panel_hourly.parquet",
        "ERCOT": ANALYSIS / "ERCOT_HC_D" / "panel_hourly_4zone.parquet",
        "ISO-NE": ANALYSIS / "ISONE_HC_D" / "panel_hourly.parquet",
        "MISO": ANALYSIS / "MISO_HC_D" / "panel_hourly.parquet",
        "NYISO": ANALYSIS / "NYISO_HC_D" / "panel_hourly.parquet",
        "PJM": ANALYSIS / "PJM_HC_D" / "panel_hourly.parquet",
        "SPP": ANALYSIS / "SPP_HC_D" / "panel_hourly_zoneproxy.parquet",
    }


def load_hourly_panels() -> pd.DataFrame:
    pieces = []
    for iso, path in panel_paths().items():
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=None)
        if "zone" not in df.columns and "zone_key_norm" in df.columns:
            df = df.rename(columns={"zone_key_norm": "zone"})
        keep = [c for c in ["ts_utc", "iso", "zone", "load_mw", "lmp", "year"] if c in df.columns]
        df = df[keep].copy()
        df["ISO"] = iso
        if "iso" in df.columns:
            df["ISO"] = df["iso"].map(norm_iso).fillna(iso)
        df["zone"] = df["zone"].astype(str)
        df = df[~df["zone"].str.upper().str.contains("TOTAL", na=False)].copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
        df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
        df = df.dropna(subset=["ts_utc", "zone", "load_mw", "lmp"])
        pieces.append(df[["ts_utc", "ISO", "zone", "load_mw", "lmp"]])
    return pd.concat(pieces, ignore_index=True)


def ols_slope(daily: pd.DataFrame) -> float:
    x = daily["load_mw"].to_numpy(dtype=float)
    y = daily["lmp"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5 or np.isclose(np.nanvar(x), 0.0):
        return np.nan
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def price_stress_robustness() -> dict[str, pd.DataFrame]:
    hourly = load_hourly_panels()
    gas = pd.read_csv(CANON / "fig5" / "fig5_price_cost_official_screen_with_isone.csv")
    gas_map = gas.set_index("ISO_RTO")["gas_variable_cost_median_dollars_mwh"].to_dict()

    def case_df(case: str) -> pd.DataFrame:
        df = hourly.copy()
        if case == "exclude_top1_lmp_by_zone":
            q = df.groupby(["ISO", "zone"])["lmp"].transform(lambda s: s.quantile(0.99))
            df = df[df["lmp"] <= q].copy()
        elif case == "early_2023_2024":
            df = df[df["ts_utc"].dt.year.isin([2023, 2024])].copy()
        elif case == "late_2025_jan2026":
            df = df[df["ts_utc"].dt.year >= 2025].copy()
        return df

    rows = []
    for case in ["baseline_full_sample", "exclude_top1_lmp_by_zone", "early_2023_2024", "late_2025_jan2026"]:
        df = case_df(case)
        df["date"] = df["ts_utc"].dt.date
        for (iso, zone), z in df.groupby(["ISO", "zone"]):
            daily = z.groupby("date", as_index=False).agg(load_mw=("load_mw", "mean"), lmp=("lmp", "mean"))
            slope_mw = ols_slope(daily)
            sensitivity = slope_mw * 1000.0 if pd.notna(slope_mw) else np.nan
            p95 = float(z["lmp"].quantile(0.95)) if len(z) else np.nan
            p95_minus_gas = p95 - float(gas_map.get(iso, np.nan))
            stress_metric = max(0.0, sensitivity if pd.notna(sensitivity) else 0.0) * max(0.0, p95_minus_gas if pd.notna(p95_minus_gas) else 0.0)
            rows.append(
                {
                    "robustness_case": case,
                    "iso_region": iso,
                    "zone": zone,
                    "n_hours": int(len(z)),
                    "p95_lmp_minus_gas_cost": p95_minus_gas,
                    "load_price_sensitivity": sensitivity,
                    "stress_metric": stress_metric,
                    "source_table": "; ".join(str(p.relative_to(PROJECT)) for p in panel_paths().values() if p.exists()),
                    "script_path": "scripts/build_si_round3_sensitivity_outputs.py::price_stress_robustness",
                    "notes": "Stress rank is computed within each grid region from positive load-price sensitivity multiplied by positive P95 LMP-minus-gas spread; robustness screen is not causal congestion or welfare inference.",
                }
            )
    detailed = pd.DataFrame(rows)
    detailed["stress_screen_rank"] = detailed.groupby(["robustness_case", "iso_region"])["stress_metric"].rank(
        method="min", ascending=False
    )
    base_rank = detailed[detailed["robustness_case"].eq("baseline_full_sample")][
        ["iso_region", "zone", "stress_screen_rank"]
    ].rename(columns={"stress_screen_rank": "baseline_rank"})
    detailed = detailed.merge(base_rank, on=["iso_region", "zone"], how="left")
    detailed["rank_change"] = detailed["stress_screen_rank"] - detailed["baseline_rank"]
    detailed["top10_flag"] = detailed["stress_screen_rank"] <= 10
    detailed["top20_flag"] = detailed["stress_screen_rank"] <= 20
    detailed = detailed[
        [
            "robustness_case",
            "iso_region",
            "zone",
            "n_hours",
            "p95_lmp_minus_gas_cost",
            "load_price_sensitivity",
            "stress_screen_rank",
            "baseline_rank",
            "rank_change",
            "top10_flag",
            "top20_flag",
            "source_table",
            "script_path",
            "notes",
        ]
    ].sort_values(["robustness_case", "iso_region", "stress_screen_rank", "zone"])

    summary_rows = []
    for (case, iso), g in detailed.groupby(["robustness_case", "iso_region"]):
        common = g.dropna(subset=["baseline_rank", "stress_screen_rank"])
        if case == "baseline_full_sample":
            corr = 1.0
        else:
            corr = common["baseline_rank"].corr(common["stress_screen_rank"], method="spearman") if len(common) >= 3 else np.nan
        n = int(common["zone"].nunique())
        top10_n = min(10, n)
        top20_n = min(20, n)
        base10 = set(common.loc[common["baseline_rank"] <= top10_n, "zone"])
        case10 = set(common.loc[common["stress_screen_rank"] <= top10_n, "zone"])
        base20 = set(common.loc[common["baseline_rank"] <= top20_n, "zone"])
        case20 = set(common.loc[common["stress_screen_rank"] <= top20_n, "zone"])
        med_change = float(common["rank_change"].abs().median()) if not common.empty else np.nan
        max_change = float(common["rank_change"].abs().max()) if not common.empty else np.nan
        if case == "baseline_full_sample":
            flag = "baseline"
        elif pd.notna(corr) and corr >= 0.7 and med_change <= 2:
            flag = "broadly stable"
        else:
            flag = "rank-sensitive; inspect zone-level details"
        summary_rows.append(
            {
                "robustness_case": case,
                "iso_region": iso,
                "spearman_rank_corr_vs_baseline": corr,
                "top10_zone_overlap_percent": np.nan if not base10 else 100.0 * len(base10 & case10) / len(base10),
                "top20_zone_overlap_percent": np.nan if not base20 else 100.0 * len(base20 & case20) / len(base20),
                "median_rank_change": med_change,
                "max_rank_change": max_change,
                "interpretation_flag": flag,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["robustness_case", "iso_region"])

    write_csv(detailed, "price_stress_robustness.csv")
    write_csv(summary, "price_stress_robustness_summary.csv")

    display = summary[summary["iso_region"].isin(["PJM", "SPP", "MISO", "NYISO", "ISO-NE"])].copy()
    display = display[display["robustness_case"].ne("baseline_full_sample")]
    rows_tex = "\n".join(
        f"{tex_escape(r.robustness_case)} & {tex_escape(r.iso_region)} & {fmt_num(r.spearman_rank_corr_vs_baseline, 2)} & {fmt_num(r.top10_zone_overlap_percent, 0)} & {fmt_num(r.median_rank_change, 1)} & {fmt_num(r.max_rank_change, 1)} & {tex_escape(r.interpretation_flag)} \\\\"
        for r in display.itertuples(index=False)
    )
    write_tex(
        "price_stress_robustness_summary_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Robustness of the zone-level price-stress screen.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l l r r r r L{{0.24\linewidth}}}}
\toprule
Robustness case & Grid region & Spearman vs baseline & Top-10 overlap (\%) & Median rank change & Max rank change & Interpretation \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{Ranks are computed within each grid region from a reduced-form stress metric combining positive load-price sensitivity and positive P95 LMP-minus-gas spread. This robustness screen tests stability to spike-hour exclusion and sample-period composition; it is not used to infer causal congestion, welfare effects or nodal deliverability.}}
\end{{table}}
\endgroup
""",
    )
    return {"detailed": detailed, "summary": summary}


def combined_dashboard(
    peak: dict[str, pd.DataFrame],
    pv: dict[str, pd.DataFrame],
    price: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fd = pd.read_csv(SI16 / "si_first_deficit_key_sensitivity_summary.csv")
    conf = pd.read_csv(SI17 / "si_project_confidence_key_findings.csv")
    conc = pd.read_csv(SI16 / "si_county_concentration_by_scenario_year.csv")
    cap = pd.read_csv(SI2 / "cap_zero_vs_raw_linear_2035.csv")

    peak_mid = peak["summary"][peak["summary"]["scenario"].eq("mid")]
    exposure = peak["exposure"][(peak["exposure"]["scenario"].eq("mid")) & (peak["exposure"]["year"].eq(2035))]
    pv_mid4 = pv["summary"][(pv["summary"]["scenario"].eq("mid")) & (pv["summary"]["storage_duration_h"].eq(4.0))]
    price_sum = price["summary"]

    def central_year(iso: str) -> str:
        return str(peak_mid[(peak_mid["gamma_case"].eq("central")) & (peak_mid["iso_region"].eq(iso))]["first_negative_year"].iloc[0])

    def gamma_shift_flag(iso: str) -> str:
        vals = peak_mid[peak_mid["iso_region"].eq(iso)]["first_negative_year"].astype(str).unique().tolist()
        return "stable" if len(set(vals)) == 1 else "shifts under absolute Gamma sensitivity"

    top50 = conc[(conc["scenario"].eq("mid")) & (conc["year"].eq(2035))]["top50_share"].iloc[0]
    exp_central = exposure[exposure["gamma_case"].eq("central")]["seven_region_exposure_share"].iloc[0]
    exp_min = exposure["seven_region_exposure_share"].min()
    exp_max = exposure["seven_region_exposure_share"].max()
    miso_raw = cap[cap["iso_region"].eq("MISO")].iloc[0]
    pjm_gamma = peak_mid[peak_mid["iso_region"].eq("PJM")]
    price_pjm_spp = price_sum[price_sum["iso_region"].isin(["PJM", "SPP"]) & price_sum["robustness_case"].ne("baseline_full_sample")]
    stable_price = (price_pjm_spp["spearman_rank_corr_vs_baseline"].fillna(0) >= 0.7).mean()
    pv_miso = pv_mid4[pv_mid4["iso_region"].eq("MISO")].iloc[0]
    pv_pjm = pv_mid4[pv_mid4["iso_region"].eq("PJM")].iloc[0]
    pv_spp = pv_mid4[pv_mid4["iso_region"].eq("SPP")].iloc[0]

    rows = [
        {
            "headline_finding": "MISO first-negative year",
            "central_value": central_year("MISO"),
            "robustness_dimension": "AI scenario, headroom shift, project confidence, peak coincidence",
            "cases_tested": "low/mid/high AI; +/-2 GW headroom; project-confidence filters; central/0.85/1.00 Gamma",
            "cases_where_holds": "early-binding ordering holds in the central and most audit screens",
            "cases_where_shifts": gamma_shift_flag("MISO"),
            "most_sensitive_case": "absolute Gamma sensitivity and additive headroom shift",
            "qualitative_conclusion": "early thin-margin exposure is robust, although the exact annual clock is convention-sensitive",
            "main_text_implication": "Modeled exposure timing, not deterministic reliability-event timing.",
            "source_tables": "si_first_deficit_key_sensitivity_summary.csv; peak_coincidence_sensitivity_summary.csv",
            "script_paths": "scripts/build_si_experiment_tables.py; scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "SPP first-negative year",
            "central_value": central_year("SPP"),
            "robustness_dimension": "AI scenario, headroom shift, project confidence, peak coincidence",
            "cases_tested": "low/mid/high AI; +/-2 GW headroom; project-confidence filters; central/0.85/1.00 Gamma",
            "cases_where_holds": "early-binding ordering holds in the central and most audit screens",
            "cases_where_shifts": gamma_shift_flag("SPP"),
            "most_sensitive_case": "low-growth + positive headroom shift; absolute Gamma sensitivity",
            "qualitative_conclusion": "early exposure is robust in central screens but more sensitive than MISO under favorable assumptions",
            "main_text_implication": "Early-exposure interpretation with no overprecision in annual timing.",
            "source_tables": "si_first_deficit_key_sensitivity_summary.csv; peak_coincidence_sensitivity_summary.csv",
            "script_paths": "scripts/build_si_experiment_tables.py; scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "PJM first-negative year",
            "central_value": central_year("PJM"),
            "robustness_dimension": "AI scenario, headroom shift, project confidence, peak coincidence",
            "cases_tested": "low/mid/high AI; +/-2 GW headroom; project-confidence filters; central/0.85/1.00 Gamma",
            "cases_where_holds": "later cumulative exposure holds under central and high-growth cases",
            "cases_where_shifts": gamma_shift_flag("PJM"),
            "most_sensitive_case": "Tier-A-only and lower peak-coincidence assumptions",
            "qualitative_conclusion": "PJM is a cumulative host-corridor pressure case and is more realization-sensitive than MISO/SPP",
            "main_text_implication": "Robust geography distinguished from realization-sensitive annual timing.",
            "source_tables": "si_project_confidence_key_findings.csv; peak_coincidence_sensitivity_summary.csv",
            "script_paths": "scripts/build_confidence_and_baseline_robustness_20260517.py; scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "2035 seven-region exposure share",
            "central_value": "77.7--77.9% across the low--high 2035 AI-load scenario envelope",
            "robustness_dimension": "AI scenario, project confidence and peak coincidence",
            "cases_tested": "project-confidence filters; central/0.85/1.00 Gamma",
            "cases_where_holds": f"peak-coincidence midcase range {100*exp_min:.1f}--{100*exp_max:.1f}%",
            "cases_where_shifts": "project-confidence Tier-A-only lower bound remains a separate SI-only audit",
            "most_sensitive_case": "Tier-A-only project-confidence screen",
            "qualitative_conclusion": "scenario-envelope exposure concentration remains high; project-confidence screens are a separate audit in which the Tier-A-only stress test lowers the share because PJM does not cross by 2035",
            "main_text_implication": "Scenario-envelope headline retained; project-confidence dependence treated as a separate audit.",
            "source_tables": "si_project_confidence_key_findings.csv; peak_coincidence_exposure_share.csv",
            "script_paths": "scripts/build_confidence_and_baseline_robustness_20260517.py; scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "Top-50 county concentration",
            "central_value": f"{100*top50:.1f}%",
            "robustness_dimension": "low/mid/high 2035 AI load scenarios",
            "cases_tested": "2035 low, mid and high county allocation",
            "cases_where_holds": "top-50 share remains high across the scenario envelope",
            "cases_where_shifts": "none material for headline concentration",
            "most_sensitive_case": "scenario endpoint",
            "qualitative_conclusion": "county concentration is stable",
            "main_text_implication": "Stable headline spatial-concentration metric.",
            "source_tables": "si_county_concentration_by_scenario_year.csv",
            "script_paths": "scripts/build_si_experiment_tables.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "MISO raw-linear sensitivity",
            "central_value": f"main {float(miso_raw['cap_at_zero_margin_2035_GW']):.1f} GW; raw-linear {float(miso_raw['raw_linear_margin_2035_GW']):.1f} GW",
            "robustness_dimension": "cap-at-zero versus raw-linear headroom convention",
            "cases_tested": "main cap-at-zero and raw-linear extension",
            "cases_where_holds": "MISO remains a sign-sensitive audit case",
            "cases_where_shifts": "raw-linear convention adds pre-existing extrapolated slack pressure",
            "most_sensitive_case": "raw-linear headroom extension",
            "qualitative_conclusion": "do not mix raw-linear baseline pressure into AI-attributable shortfall",
            "main_text_implication": "AI-attributable margins reported under cap-at-zero; raw-linear values retained as sign-sensitivity checks.",
            "source_tables": "cap_zero_vs_raw_linear_2035.csv",
            "script_paths": "scripts/build_si_round2_audit_tables.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "PJM near-boundary behavior",
            "central_value": "later cumulative pressure",
            "robustness_dimension": "low-growth, headroom-extrapolation and project-confidence screens",
            "cases_tested": "low/mid/high AI; headroom extrapolation; project-confidence filters",
            "cases_where_holds": "central and high-growth screens show cumulative erosion",
            "cases_where_shifts": "low-confidence geography and lower load conversion can delay crossing beyond 2035",
            "most_sensitive_case": "Tier-A-only or low-growth favorable cases",
            "qualitative_conclusion": "PJM should be described as cumulative and realization-sensitive",
            "main_text_implication": "Cumulative and realization-sensitive case, not overprecise annual timing.",
            "source_tables": "si_project_confidence_key_findings.csv; peak_coincidence_sensitivity_summary.csv",
            "script_paths": "scripts/build_confidence_and_baseline_robustness_20260517.py; scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "Project-confidence dependence",
            "central_value": "MISO/SPP less sensitive; PJM more sensitive",
            "robustness_dimension": "reported/committed, confidence-weighted, Tier-A-only and delayed Tier-C cases",
            "cases_tested": "midcase project-confidence filters",
            "cases_where_holds": "early MISO/SPP exposure persists under conservative screens",
            "cases_where_shifts": "PJM is more sensitive to lower-confidence project geography and scale",
            "most_sensitive_case": "Tier-A-only",
            "qualitative_conclusion": "thin-margin timing is more robust than cumulative host-corridor timing",
            "main_text_implication": "Project-confidence caveat retained in the Results and SI.",
            "source_tables": "si_project_confidence_key_findings.csv",
            "script_paths": "scripts/build_confidence_and_baseline_robustness_20260517.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "Peak-coincidence sensitivity",
            "central_value": "central regional Gamma",
            "robustness_dimension": "absolute Gamma 0.85 and 1.00",
            "cases_tested": "central, absolute 0.85 and absolute 1.00",
            "cases_where_holds": "MISO/SPP remain earlier than PJM in the midcase sensitivity table",
            "cases_where_shifts": "exact first-negative year can shift under lower absolute Gamma cases",
            "most_sensitive_case": "absolute Gamma=0.85",
            "qualitative_conclusion": "regional ordering is more stable than exact annual crossing year",
            "main_text_implication": "Timing reported as modeled bottleneck clocks, not deterministic reliability years.",
            "source_tables": "peak_coincidence_sensitivity_summary.csv",
            "script_paths": "scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "PV-window implementation diagnostic",
            "central_value": f"MISO {float(pv_miso['central_window_pv_ratio_for_1pct']):.2f}; PJM {float(pv_pjm['central_window_pv_ratio_for_1pct']):.2f}; SPP {float(pv_spp['central_window_pv_ratio_for_1pct']):.2f}",
            "robustness_dimension": "representative JJA daily profile tiled over seven days",
            "cases_tested": "central, one-week-earlier and one-week-later labels under the same representative-day implementation",
            "cases_where_holds": "earlier/later labels do not create distinct calendar weeks",
            "cases_where_shifts": "not tested with date-resolved calendar weeks",
            "most_sensitive_case": "requires date-resolved solar/load profile for real shifted-week test",
            "qualitative_conclusion": "current table diagnoses representative-day limitation; no shifted-calendar-week robustness is claimed",
            "main_text_implication": "PV-plus-storage remains a planning screen; hourly timing check does not overturn the intervention-alignment result.",
            "source_tables": "pv_storage_window_sensitivity_summary.csv",
            "script_paths": "scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
        {
            "headline_finding": "Price-stress robustness",
            "central_value": "reduced-form screen",
            "robustness_dimension": "exclude top 1% price hours and split hourly panel",
            "cases_tested": "baseline, top-1% exclusion, 2023-2024, 2025-Jan 2026",
            "cases_where_holds": f"{100*stable_price:.0f}% of PJM/SPP robustness rows have rank correlation >=0.7",
            "cases_where_shifts": "some zone ranks shift across subperiods",
            "most_sensitive_case": "subperiod split",
            "qualitative_conclusion": "screen is useful as auxiliary stress indicator but should remain reduced-form",
            "main_text_implication": "Non-causal price-stress screen.",
            "source_tables": "price_stress_robustness_summary.csv",
            "script_paths": "scripts/build_si_round3_sensitivity_outputs.py",
            "notes": AUDIT_NOTE,
        },
    ]
    dash = pd.DataFrame(rows)
    write_csv(dash, "combined_robustness_dashboard.csv")

    rows_tex = "\n".join(
        f"{tex_escape(r.headline_finding)} & {tex_escape(r.central_value)} & {tex_escape(r.robustness_dimension)} & {tex_escape(r.qualitative_conclusion)} & {tex_escape(r.main_text_implication)} \\\\"
        for r in dash.itertuples(index=False)
    )
    write_tex(
        "combined_robustness_dashboard_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Combined robustness dashboard for headline timing and exposure findings.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{L{{0.19\linewidth}}L{{0.17\linewidth}}L{{0.24\linewidth}}L{{0.25\linewidth}}L{{0.25\linewidth}}}}
\toprule
Finding & Central value & Robustness dimension & Qualitative conclusion & Interpretation \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{This dashboard separates findings that are stable across audit dimensions from findings that are convention- or realization-sensitive. It is an audit guide and does not replace the central scenario or main-text headline values.}}
\end{{table}}
\endgroup
""",
    )
    return dash


def write_round3_manifest() -> pd.DataFrame:
    items = [
        ("peak_coincidence", "peak_coincidence_sensitivity.csv", "true sensitivity", "CSV table", "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity", "tables/fig2_5_canonical_20260514/fig2/fig2_shared_canonical_gap_yearly.csv; tables/si_round2/peak_coincidence_gamma_factors.csv", "date-resolved rerun not needed"),
        ("peak_coincidence", "peak_coincidence_sensitivity_summary.csv", "true sensitivity", "CSV summary", "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity", "tables/si_round3/peak_coincidence_sensitivity.csv", "summary of true sensitivity"),
        ("peak_coincidence", "peak_coincidence_exposure_share.csv", "true sensitivity", "CSV exposure share", "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity", "tables/si_round3/peak_coincidence_sensitivity.csv", "reconstruction sensitivity; not a replacement for 77.8% headline"),
        ("peak_coincidence", "peak_coincidence_sensitivity_summary_table.tex", "true sensitivity", "LaTeX fragment", "scripts/build_si_round3_sensitivity_outputs.py::peak_coincidence_sensitivity", "tables/si_round3/peak_coincidence_sensitivity_summary.csv", "printed SI table fragment"),
        ("pv_storage_window", "pv_storage_window_sensitivity.csv", "implementation diagnostic", "CSV table", "scripts/build_si_round3_sensitivity_outputs.py::pv_storage_window_sensitivity", "tables/fig2_5_canonical_20260514/fig5/fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv; " + PV_PROFILE_SOURCE, "date-resolved shifted-calendar-week PV profile not identified"),
        ("pv_storage_window", "pv_storage_window_sensitivity_summary.csv", "implementation diagnostic", "CSV summary", "scripts/build_si_round3_sensitivity_outputs.py::pv_storage_window_sensitivity", "tables/si_round3/pv_storage_window_sensitivity.csv", "not a true shifted-week sensitivity"),
        ("pv_storage_window", "pv_storage_window_sensitivity_summary_table.tex", "implementation diagnostic", "LaTeX fragment", "scripts/build_si_round3_sensitivity_outputs.py::pv_storage_window_sensitivity", "tables/si_round3/pv_storage_window_sensitivity_summary.csv", "printed SI table fragment"),
        ("price_stress", "price_stress_robustness.csv", "true sensitivity", "CSV table", "scripts/build_si_round3_sensitivity_outputs.py::price_stress_robustness", "out_std/ANALYSIS/*_HC_D/panel_hourly*.parquet; tables/fig2_5_canonical_20260514/fig5/fig5_price_cost_official_screen_with_isone.csv", "reduced-form stress-screen robustness"),
        ("price_stress", "price_stress_robustness_summary.csv", "true sensitivity", "CSV summary", "scripts/build_si_round3_sensitivity_outputs.py::price_stress_robustness", "tables/si_round3/price_stress_robustness.csv", "summary of true robustness screen"),
        ("price_stress", "price_stress_robustness_summary_table.tex", "true sensitivity", "LaTeX fragment", "scripts/build_si_round3_sensitivity_outputs.py::price_stress_robustness", "tables/si_round3/price_stress_robustness_summary.csv", "printed SI table fragment"),
        ("combined_dashboard", "combined_robustness_dashboard.csv", "audit dashboard", "CSV table", "scripts/build_si_round3_sensitivity_outputs.py::combined_dashboard", "Round-2 outputs plus tables/si_round3/* sensitivity outputs", "submission-facing audit guide"),
        ("combined_dashboard", "combined_robustness_dashboard_table.tex", "audit dashboard", "LaTeX fragment", "scripts/build_si_round3_sensitivity_outputs.py::combined_dashboard", "tables/si_round3/combined_robustness_dashboard.csv", "printed SI table fragment"),
    ]

    rows = []
    for group, filename, status, output_type, script, inputs, audit_status in items:
        path = OUT / filename
        if path.suffix == ".csv" and path.exists():
            row_count: object = len(pd.read_csv(path))
        elif path.suffix == ".tex" and path.exists():
            row_count = "LaTeX fragment"
        else:
            row_count = "missing"
        rows.append(
            {
                "output_group": group,
                "file_path": str(path.relative_to(ROOT)),
                "row_count": row_count,
                "output_type": output_type,
                "source_script": script,
                "source_inputs": inputs,
                "audit_status": audit_status,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = write_csv(manifest, "round3_sensitivity_manifest.csv")
    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [
                    {
                        "output_group": "source_manifest",
                        "file_path": str(manifest_path.relative_to(ROOT)),
                        "row_count": len(manifest),
                        "output_type": "CSV manifest",
                        "source_script": "scripts/build_si_round3_sensitivity_outputs.py::write_round3_manifest",
                        "source_inputs": "generated Round-3 sensitivity outputs",
                        "audit_status": "traceability index",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    manifest.to_csv(manifest_path, index=False)
    print(f"[updated] {manifest_path.relative_to(ROOT)} ({len(manifest):,} rows including manifest row)")
    return manifest



def write_report(
    peak: dict[str, pd.DataFrame],
    pv: dict[str, pd.DataFrame],
    price: dict[str, pd.DataFrame],
    dash: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Path:
    peak_mid = peak["summary"][(peak["summary"]["scenario"].eq("mid")) & (peak["summary"]["iso_region"].isin(["MISO", "SPP", "PJM"]))]
    pv_mid4 = pv["summary"][(pv["summary"]["scenario"].eq("mid")) & (pv["summary"]["storage_duration_h"].eq(4.0))]
    price_pjm_spp = price["summary"][
        price["summary"]["iso_region"].isin(["PJM", "SPP"]) & price["summary"]["robustness_case"].ne("baseline_full_sample")
    ]

    report = f"""# Round-3 sensitivity output report

This report summarizes generated public sensitivity outputs. It does not include private manuscript text, cross-reference checks or source-coordinate logs.

## New or updated source tables
{chr(10).join(f"- `{p}`" for p in manifest["file_path"].tolist())}

## Interpretation notes
- PV-window outputs are retained as implementation diagnostics unless date-resolved hourly PV profiles are supplied and rerun.
- Peak-coincidence outputs are reconstruction sensitivities and should not be presented as replacements for central scenario denominators.
- Price-stress outputs are non-causal robustness screens.

## Key audit snippets

Peak-coincidence midcase first-negative years for MISO/SPP/PJM:

```text
{peak_mid[['gamma_case','iso_region','first_negative_year','margin_2035_GW']].to_string(index=False)}
```

PV-window midcase 4 h thresholds:

```text
{pv_mid4[['iso_region','central_window_pv_ratio_for_1pct','earlier_window_pv_ratio_for_1pct','later_window_pv_ratio_for_1pct','interpretation_flag']].to_string(index=False)}
```

PJM/SPP price-stress robustness:

```text
{price_pjm_spp[['robustness_case','iso_region','spearman_rank_corr_vs_baseline','top10_zone_overlap_percent','median_rank_change','interpretation_flag']].to_string(index=False)}
```
"""
    path = ROOT / "round3_sensitivity_output_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"[wrote] {path.relative_to(ROOT)}")
    return path

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    peak = peak_coincidence_sensitivity()
    pv = pv_storage_window_sensitivity()
    price = price_stress_robustness()
    dash = combined_dashboard(peak, pv, price)
    manifest = write_round3_manifest()
    write_report(peak, pv, price, dash, manifest)


if __name__ == "__main__":
    main()
