from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parents[2]
TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
ANALYSIS = PROJECT / "out_std" / "ANALYSIS"
GENERATOR = PROJECT / "out_std" / "GENERATOR"
FIG5_DATA = WORK / "fig5_data_experiments_20260513"
FIG5_SUPP = FIG5_DATA / "supplementary_realdata_results"
FIG5_NASA = FIG5_DATA / "nasa_power_site_solar_20260513"
OUT = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"

SCENARIOS = ("low", "mid", "high")
YEARS = tuple(range(2025, 2036))
ISO7 = ("CAISO", "ERCOT", "MISO", "PJM", "NYISO", "ISO-NE", "SPP")
ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
FOCUS_ISOS = ["MISO", "SPP", "PJM", "ERCOT"]
DEMAND_TAG = "20260407_152405"
PV_GRID_FINE = np.round(np.arange(0.0, 2.0001, 0.05), 4)
STORAGE_GRID_FINE = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 12.0],
    dtype=float,
)


def norm_iso(value: object) -> str:
    if pd.isna(value):
        return "Non-ISO"
    text = str(value).strip()
    upper = text.upper().replace("_", "-")
    if upper in {"ISONE", "ISO NE", "ISO-NE"}:
        return "ISO-NE"
    if upper in {"NONISO", "NON-ISO", "NON ISO", ""}:
        return "Non-ISO"
    return text if text in set(ISO7) else text


def write(df: pd.DataFrame, rel: str) -> Path:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[wrote] {path} ({len(df):,} rows)")
    return path


def demand_path(scenario: str) -> Path:
    folder = PROJECT / "out_std" / "DC" / "demand_dynamic"
    path = folder / f"dc_demand_county_year_MWpeak_2025_2035_LF0.8_AeffBaseline_{DEMAND_TAG}_{scenario}.csv"
    if path.exists():
        return path
    files = sorted(
        folder.glob(f"dc_demand_county_year_MWpeak_2025_2035_LF0.8_AeffBaseline_*_{scenario}.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No county demand file found for {scenario}")
    return files[0]


def national_average_mw() -> pd.DataFrame:
    curve = pd.read_csv(ANALYSIS / "national_dc_twh_high_mid_low_2025_2035.csv")
    rows = []
    for _, row in curve.iterrows():
        year = int(row["year"])
        if year not in YEARS:
            continue
        for scenario in SCENARIOS:
            twh = float(row[f"TWh_{scenario}"])
            rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "national_TWh": twh,
                    "national_average_GW": twh / 8.76,
                    "national_average_MW": twh * 1_000_000.0 / 8760.0,
                }
            )
    return pd.DataFrame(rows)


def gamma_map() -> dict[str, float]:
    gamma = pd.read_csv(ANALYSIS / "iso_peak_conversion_factors.csv")
    gamma = gamma[gamma["dc_shape_name"].astype(str).str.lower().eq("avg")].copy()
    out = {norm_iso(row["ISO"]): float(row["Gamma"]) for _, row in gamma.iterrows()}
    out["Non-ISO"] = 1.0
    return out


def county_meta_from_fig1() -> pd.DataFrame:
    canon = TABLE_ARCHIVE / "tables" / "fig1_canonical_20260514" / "fig1_canonical_county_all_scenarios.csv"
    if canon.exists():
        meta = pd.read_csv(canon, dtype={"GEOID": str})
        meta["GEOID"] = meta["GEOID"].astype(str).str.zfill(5)
        meta["iso"] = meta["iso"].map(norm_iso)
        keep = ["GEOID", "iso", "state_abb", "label", "gamma"]
        return meta[keep].drop_duplicates("GEOID")

    # Fallback is deliberately weaker: it preserves non-ISO where known, then treats
    # remaining unmatched counties as non-ISO rather than silently renormalizing ISO7.
    assign = pd.read_csv(ANALYSIS / "county_potential_with_noniso.csv", dtype={"GEOID": str})
    assign["GEOID"] = assign["GEOID"].astype(str).str.zfill(5)
    assign["iso"] = assign["ISO_ASSIGNED"].map(norm_iso)
    assign["state_abb"] = np.nan
    assign["label"] = assign["CTYNAME"].astype(str)
    assign["gamma"] = assign["iso"].map(gamma_map()).fillna(1.0)
    return assign[["GEOID", "iso", "state_abb", "label", "gamma"]].drop_duplicates("GEOID")


def build_canonical_county_yearly() -> tuple[pd.DataFrame, pd.DataFrame]:
    nat = national_average_mw()
    meta = county_meta_from_fig1()
    gammas = gamma_map()
    pieces: list[pd.DataFrame] = []
    audit_rows = []
    for scenario in SCENARIOS:
        raw = pd.read_csv(demand_path(scenario), dtype={"GEOID": str})
        raw["GEOID"] = raw["GEOID"].astype(str).str.zfill(5)
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype(int)
        raw["MW_peak_total"] = pd.to_numeric(raw["MW_peak_total"], errors="coerce").fillna(0.0)
        raw = raw[raw["year"].isin(YEARS)].copy()
        raw = raw.merge(meta, on="GEOID", how="left")
        raw["iso"] = raw["iso"].map(norm_iso).fillna("Non-ISO")
        raw["gamma"] = raw["gamma"].fillna(raw["iso"].map(gammas)).fillna(1.0)
        raw["state_abb"] = raw["state_abb"].fillna("")
        raw["label"] = raw["label"].fillna(raw["GEOID"])
        for year in YEARS:
            sub = raw[raw["year"].eq(year)].copy()
            total_old = float(sub["MW_peak_total"].sum())
            if total_old <= 0:
                raise ValueError(f"Zero source demand total for {scenario} {year}")
            nat_row = nat[(nat["scenario"].eq(scenario)) & (nat["year"].eq(year))].iloc[0]
            sub["raw_spatial_share"] = sub["MW_peak_total"] / total_old
            sub["MW_canonical"] = sub["raw_spatial_share"] * float(nat_row["national_average_MW"]) * sub["gamma"]
            sub["GW_canonical"] = sub["MW_canonical"] / 1000.0
            sub["scenario"] = scenario
            pieces.append(
                sub[
                    [
                        "GEOID",
                        "year",
                        "scenario",
                        "iso",
                        "state_abb",
                        "label",
                        "raw_spatial_share",
                        "MW_peak_total",
                        "MW_canonical",
                        "GW_canonical",
                        "gamma",
                    ]
                ]
            )
            audit_rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "source_file": demand_path(scenario).name,
                    "source_peak_GW_total": total_old / 1000.0,
                    "raw_ISO7_share": float(sub[sub["iso"].isin(ISO7)]["MW_peak_total"].sum() / total_old),
                    "national_average_GW": float(nat_row["national_average_GW"]),
                    "ISO7_coincident_GW": float(sub[sub["iso"].isin(ISO7)]["MW_canonical"].sum() / 1000.0),
                    "nonISO_coincident_GW": float(sub[~sub["iso"].isin(ISO7)]["MW_canonical"].sum() / 1000.0),
                    "unassigned_county_rows": int(raw["iso"].eq("Non-ISO").sum()),
                }
            )
    county = pd.concat(pieces, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    write(county, "shared/canonical_county_ai_demand_yearly.csv")
    write(audit, "shared/canonical_county_ai_demand_audit.csv")
    return county, audit


def build_iso_ai_demand(county: pd.DataFrame) -> pd.DataFrame:
    iso = (
        county[county["iso"].isin(ISO7)]
        .groupby(["scenario", "year", "iso"], as_index=False)["MW_canonical"]
        .sum()
        .rename(columns={"iso": "ISO", "MW_canonical": "ai_demand_MW"})
    )
    baseline = (
        iso[iso["scenario"].eq("mid") & iso["year"].eq(2025)]
        .set_index("ISO")["ai_demand_MW"]
        .to_dict()
    )
    iso["ai_baseline_mid2025_MW"] = iso["ISO"].map(baseline).fillna(0.0)
    iso["ai_growth_vs_mid2025_MW"] = (iso["ai_demand_MW"] - iso["ai_baseline_mid2025_MW"]).clip(lower=0.0)
    iso["ai_demand_GW"] = iso["ai_demand_MW"] / 1000.0
    iso["ai_growth_vs_mid2025_GW"] = iso["ai_growth_vs_mid2025_MW"] / 1000.0
    order = {iso_name: i for i, iso_name in enumerate(ISO_ORDER)}
    iso["iso_order"] = iso["ISO"].map(order).fillna(99).astype(int)
    iso = iso.sort_values(["scenario", "year", "iso_order"]).reset_index(drop=True)
    write(iso, "shared/canonical_iso_ai_demand_yearly.csv")
    return iso


def build_canonical_gap(iso_ai: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(ANALYSIS / "fig1b_gap_iso7_slow_mid_fast_2025_2035.csv")
    old["ISO"] = old["ISO_RTO"].map(norm_iso)
    old["year"] = pd.to_numeric(old["year"], errors="coerce").astype(int)
    old = old[old["ISO"].isin(ISO7) & old["year"].isin(YEARS)].copy()
    old = old.rename(columns={"HC_raw_MW": "HC_raw_legacy_cap0_MW", "HC_MW": "HC_legacy_cap0_MW"})
    old = old.drop(columns=[c for c in old.columns if c.startswith("D_ai_")], errors="ignore")

    raw_headroom = pd.read_csv(ANALYSIS / "iso_potential_2025_2035_cap0_linear.csv")
    raw_headroom["ISO"] = raw_headroom["ISO_RTO"].map(norm_iso)
    raw_headroom["year"] = pd.to_numeric(raw_headroom["year"], errors="coerce").astype(int)
    raw_headroom = raw_headroom[
        raw_headroom["ISO"].isin(ISO7)
        & raw_headroom["year"].isin(YEARS)
        & raw_headroom["scenario_IA"].eq("EXEC")
    ].copy()
    raw_headroom = raw_headroom[
        [
            "ISO",
            "year",
            "slack_raw_lin_MW",
            "hosting_potential_cap0_MW",
            "shortfall_lin_MW",
            "trend_slope_MW_per_year",
        ]
    ].drop_duplicates(["ISO", "year"])
    old = old.merge(raw_headroom, on=["ISO", "year"], how="left")
    old["HC_signed_MW"] = pd.to_numeric(old["slack_raw_lin_MW"], errors="coerce").fillna(old["HC_raw_legacy_cap0_MW"])
    old["HC_cap0_MW"] = pd.to_numeric(old["hosting_potential_cap0_MW"], errors="coerce").fillna(old["HC_legacy_cap0_MW"])
    old["baseline_nonAI_shortfall_raw_MW"] = (-old["HC_signed_MW"]).clip(lower=0.0)
    old["HC_shortfall_raw_MW"] = pd.to_numeric(old["shortfall_lin_MW"], errors="coerce").fillna(old["baseline_nonAI_shortfall_raw_MW"])
    # Canonical HC_MW is signed raw slack. Use HC_cap0_MW only when a positive
    # visual headroom quantity is explicitly needed.
    old["HC_raw_MW"] = old["HC_signed_MW"]
    old["HC_MW"] = old["HC_signed_MW"]
    ai = iso_ai.pivot_table(index=["ISO", "year"], columns="scenario", values="ai_growth_vs_mid2025_MW", aggfunc="sum")
    ai = ai.rename(columns={s: f"D_ai_{s}_MW" for s in SCENARIOS}).reset_index()
    gap = old.merge(ai, on=["ISO", "year"], how="left")
    for scenario in SCENARIOS:
        col = f"D_ai_{scenario}_MW"
        gap[col] = gap[col].fillna(0.0)
    gap["D_nonAI_delta_MW"] = pd.to_numeric(gap["D_nonAI_delta_MW"], errors="coerce").fillna(0.0)
    gap["HC_MW"] = pd.to_numeric(gap["HC_MW"], errors="coerce")
    gap["HC_raw_MW"] = pd.to_numeric(gap["HC_raw_MW"], errors="coerce")
    scenario_name = {"low": "lowcase", "mid": "midcase", "high": "highcase"}
    legacy_name = {"low": "slow", "mid": "mid", "high": "fast"}
    for scenario in SCENARIOS:
        ai_col = f"D_ai_{scenario}_MW"
        # HC_MW is the signed modeled net slack available before incremental AI
        # demand. Non-AI load growth is part of the slack construction and must
        # not be subtracted again here.
        ai_demand = gap[ai_col]
        margin = gap["HC_MW"] - ai_demand
        gap[f"D_total_{legacy_name[scenario]}_MW"] = ai_demand
        gap[f"D_{legacy_name[scenario]}_MW"] = ai_demand
        gap[f"Gap_total_{legacy_name[scenario]}_MW"] = margin
        gap[f"Gap_{legacy_name[scenario]}_MW"] = margin
        gap[f"D_{scenario_name[scenario]}_MW"] = ai_demand
        gap[f"Gap_{scenario_name[scenario]}_MW"] = margin
    gap["ISO_RTO"] = gap["ISO"]
    gap = gap.sort_values(["ISO", "year"]).reset_index(drop=True)

    b2025 = gap[gap["year"].eq(2025)][["ISO", "HC_MW", "HC_cap0_MW"]].rename(
        columns={"HC_MW": "Carryover_2025_MW", "HC_cap0_MW": "Carryover_2025_cap0_MW"}
    )
    g2035 = gap[gap["year"].eq(2035)].merge(b2025, on="ISO", how="left")
    bridge = pd.DataFrame(
        {
            "ISO": g2035["ISO"],
            "Carryover_2025_MW": g2035["Carryover_2025_MW"],
            "Carryover_2025_cap0_MW": g2035["Carryover_2025_cap0_MW"],
            "New_supply_to_2035_MW": g2035["HC_MW"] - g2035["Carryover_2025_MW"],
            "Supply_total_MW": g2035["HC_MW"],
            "Signed_net_slack_2035_MW": g2035["HC_signed_MW"],
            "Positive_headroom_cap0_2035_MW": g2035["HC_cap0_MW"],
            "Baseline_nonAI_shortfall_raw_2035_MW": g2035["baseline_nonAI_shortfall_raw_MW"],
            "NonAI_growth_to_2035_MW": g2035["D_nonAI_delta_MW"],
            "NonAI_load_change_to_2035_MW": g2035["D_nonAI_delta_MW"],
            "AI_growth_to_2035_MW": g2035["D_ai_mid_MW"],
        }
    )
    bridge["Net_headroom_change_2025_2035_MW"] = bridge["New_supply_to_2035_MW"]
    bridge["Demand_total_MW"] = bridge["AI_growth_to_2035_MW"]
    bridge["Gap_MW"] = bridge["Supply_total_MW"] - bridge["Demand_total_MW"]
    bridge["Margin_2035_MW"] = bridge["Gap_MW"]
    bridge["Margin_2035_from_HC_MW"] = bridge["Gap_MW"]
    bridge["headroom_definition_note"] = "HC_MW is signed net supply slack before incremental AI; HC_cap0_MW/Positive_headroom_cap0_2035_MW is only the positive visual headroom."
    bridge["ISO"] = bridge["ISO"].map(norm_iso)
    bridge["iso_order"] = bridge["ISO"].map({iso: i for i, iso in enumerate(ISO_ORDER)})
    bridge = bridge.sort_values("iso_order").drop(columns=["iso_order"]).reset_index(drop=True)

    write(gap, "fig2/fig2_shared_canonical_gap_yearly.csv")
    write(bridge, "fig2/fig2b_iso_margin_bridge_2035_canonical_source.csv")
    return gap, bridge


def first_deficit_year(sub: pd.DataFrame, margin_col: str) -> float:
    g = sub.sort_values("year")
    years = g["year"].to_numpy(dtype=float)
    vals = g[margin_col].to_numpy(dtype=float)
    if len(vals) == 0 or np.all(vals > 0):
        return np.nan
    hit = np.where(vals <= 0)[0][0]
    if hit == 0:
        return years[hit]
    y0, y1 = years[hit - 1], years[hit]
    v0, v1 = vals[hit - 1], vals[hit]
    if v0 == v1:
        return y1
    return y0 + (0 - v0) * (y1 - y0) / (v1 - v0)


def headroom_delta_variants() -> pd.DataFrame:
    sens = pd.read_csv(ANALYSIS / "potential_extrapolation_sensitivity_full_2025_2035.csv")
    sens["ISO"] = sens["ISO_RTO"].map(norm_iso)
    sens = sens[sens["ISO"].isin(ISO7)].copy()
    sens["year"] = pd.to_numeric(sens["year"], errors="coerce").astype(int)
    base = sens[sens["scenario"].eq("Baseline_extension")][["ISO", "year", "raw_value", "hosting_potential"]].rename(
        columns={
            "raw_value": "baseline_raw_slack_MW",
            "hosting_potential": "baseline_hosting_potential_cap0_MW",
        }
    )
    variants = sens.merge(base, on=["ISO", "year"], how="left")
    variants["headroom_delta_vs_baseline_MW"] = variants["raw_value"] - variants["baseline_raw_slack_MW"]
    variants["headroom_cap0_delta_vs_baseline_MW"] = variants["hosting_potential"] - variants["baseline_hosting_potential_cap0_MW"]
    return variants[["scenario", "ISO", "year", "headroom_delta_vs_baseline_MW", "headroom_cap0_delta_vs_baseline_MW"]]


def build_fig2_panels(gap: pd.DataFrame, bridge: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Fig. 2a map source.
    a = (
        gap[gap["year"].eq(2025)][["ISO", "HC_MW"]]
        .rename(columns={"HC_MW": "headroom_2025_MW"})
        .sort_values("ISO")
        .reset_index(drop=True)
    )
    a["headroom_2025_GW"] = a["headroom_2025_MW"] / 1000.0
    write(a, "fig2/fig2a_iso_headroom_map_canonical_source.csv")

    # Fig. 2c generator pipeline credibility/filter data.
    active = pd.read_csv(ANALYSIS / "iso_queue_3yr_rate_2026_2028_active_vs_IAstage_typeBuckets.csv")
    active["ISO"] = active["ISO_RTO"].map(norm_iso)
    eff = pd.read_csv(ANALYSIS / "queue_eff_audit_iso_year_breakdown.csv")
    eff = eff[eff["scenario_ia"].eq("EXEC_only") & eff["alpha_case"].eq("baseline")].copy()
    eff["ISO"] = eff["ISO_RTO"].map(norm_iso)
    eff_sum = (
        eff.groupby("ISO", as_index=False)
        .agg(
            add_nameplate_gen_MW=("add_nameplate_gen_MW", "sum"),
            add_eff_MW=("add_eff_MW", "sum"),
            thermal_eff_MW=("thermal_eff", "sum"),
            wind_eff_MW=("wind_eff", "sum"),
            solar_eff_MW=("solar_eff", "sum"),
        )
    )
    storage = pd.read_csv(ANALYSIS / "queue_eff_audit_alpha_storage_sensitivity_focus_2028.csv")
    storage = storage[
        storage["ia_mode"].eq("EXEC")
        & storage["include_storage"].eq(True)
        & np.isclose(storage["alpha_storage"], 0.2)
    ].copy()
    storage["ISO"] = storage["ISO_RTO"].map(norm_iso)
    storage = storage[["ISO", "cum_eff_delta_vs_base"]].rename(columns={"cum_eff_delta_vs_base": "storage_20pct_credit_delta_MW"})
    top10 = pd.read_csv(ANALYSIS / "queue_eff_audit_top10_project_share_by_iso.csv")
    top10["ISO"] = top10["ISO_RTO"].map(norm_iso)
    if "top10_share_eff" not in top10.columns and "top10_share" in top10.columns:
        top10["top10_share_eff"] = top10["top10_share"]
    pipe = (
        active.merge(eff_sum, on="ISO", how="outer")
        .merge(storage, on="ISO", how="left")
        .merge(bridge[["ISO", "Margin_2035_MW"]], on="ISO", how="left")
        .merge(top10[["ISO", "top10_share_eff"]], on="ISO", how="left")
    )
    pipe["effective_over_active_queue"] = pipe["add_eff_MW"] / pipe["MW_sum_total_active_2026_2028"].replace(0, np.nan)
    pipe = pipe[pipe["ISO"].isin(ISO7)].copy()
    pipe["iso_order"] = pipe["ISO"].map({iso: i for i, iso in enumerate(ISO_ORDER)})
    pipe = pipe.sort_values("iso_order").drop(columns=["iso_order"]).reset_index(drop=True)
    write(pipe, "fig2/fig2c_generator_pipeline_canonical_source.csv")

    variants = headroom_delta_variants()
    rows = []
    summary_rows = []
    scenario_cols = {"low": "Gap_lowcase_MW", "mid": "Gap_midcase_MW", "high": "Gap_highcase_MW"}
    for scenario, margin_col in scenario_cols.items():
        for iso in ISO_ORDER:
            sub = gap[gap["ISO"].eq(iso)].copy().sort_values("year")
            fd = first_deficit_year(sub, margin_col)
            for _, row in sub.iterrows():
                year = int(row["year"])
                margin = float(row[margin_col])
                v = variants[(variants["ISO"].eq(iso)) & (variants["year"].eq(year))]
                sign_sensitive = False
                if not v.empty:
                    vals = margin + v["headroom_delta_vs_baseline_MW"].to_numpy(dtype=float)
                    sign_sensitive = bool(np.nanmin(vals) <= 0 <= np.nanmax(vals))
                rows.append(
                    {
                        "ISO": iso,
                        "scenario": scenario,
                        "year": year,
                        "margin_MW": margin,
                        "margin_GW": margin / 1000.0,
                        "first_deficit_year": fd,
                        "warning_0_to_2GW": bool(0 <= margin / 1000.0 <= 2),
                        "sign_sensitive_to_headroom_extrapolation": sign_sensitive,
                    }
                )
            m2035 = float(sub[sub["year"].eq(2035)][margin_col].iloc[0])
            summary_rows.append(
                {
                    "ISO": iso,
                    "scenario": scenario,
                    "first_deficit_year": fd,
                    "margin_2035_GW": m2035 / 1000.0,
                    "shortfall_2035_GW": max(0.0, -m2035 / 1000.0),
                }
            )
    clock = pd.DataFrame(rows)
    clock_summary = pd.DataFrame(summary_rows)
    write(clock, "fig2/fig2d_margin_clock_heatmap_canonical_source.csv")
    write(clock_summary, "fig2/fig2d_margin_clock_summary_canonical.csv")

    traj = []
    for iso in FOCUS_ISOS:
        sub = gap[gap["ISO"].eq(iso)].copy()
        for _, r in sub.iterrows():
            year = int(r["year"])
            rec = {
                "ISO": iso,
                "year": year,
                "low_growth_margin_GW": float(r["Gap_lowcase_MW"]) / 1000.0,
                "mid_growth_margin_GW": float(r["Gap_midcase_MW"]) / 1000.0,
                "high_growth_margin_GW": float(r["Gap_highcase_MW"]) / 1000.0,
            }
            v = variants[(variants["ISO"].eq(iso)) & (variants["year"].eq(year))]
            if v.empty:
                rec["headroom_extrapolation_low_GW"] = 0.0
                rec["headroom_extrapolation_high_GW"] = 0.0
            else:
                rec["headroom_extrapolation_low_GW"] = float(v["headroom_delta_vs_baseline_MW"].min()) / 1000.0
                rec["headroom_extrapolation_high_GW"] = float(v["headroom_delta_vs_baseline_MW"].max()) / 1000.0
            traj.append(rec)
    traj = pd.DataFrame(traj)
    write(traj, "fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv")
    return {"a": a, "pipe": pipe, "clock": clock, "clock_summary": clock_summary, "traj": traj}


def read_mix() -> pd.DataFrame:
    mix = pd.read_csv(PROJECT / "out_std" / "FIG" / "fig3b_iso_regional_hosting_risk_cards_wind_solar_split_debug.csv")
    first = mix.columns[0]
    mix = mix.rename(columns={first: "ISO"})
    mix["ISO"] = mix["ISO"].map(norm_iso)
    return mix


def build_fig3_data(bridge: pd.DataFrame, pipe: pd.DataFrame, fig2_traj: pd.DataFrame, iso_ai: pd.DataFrame, county: pd.DataFrame) -> None:
    mix = read_mix()
    core = bridge.merge(mix, on="ISO", how="left").merge(pipe, on="ISO", how="left", suffixes=("", "_pipe"))
    core["headroom_gw"] = core["Carryover_2025_MW"] / 1000.0
    core["new_supply_gw"] = core["New_supply_to_2035_MW"] / 1000.0
    core["nonai_gw"] = core["NonAI_growth_to_2035_MW"] / 1000.0
    core["ai_gw"] = core["AI_growth_to_2035_MW"] / 1000.0
    core["demand_total_gw"] = core["Demand_total_MW"] / 1000.0
    core["margin_gw"] = core["Margin_2035_MW"] / 1000.0
    core["future_pressure_gw"] = core["ai_gw"] + core["nonai_gw"] - core["new_supply_gw"]
    core["ai_to_headroom"] = core["ai_gw"] / core["headroom_gw"].replace(0, np.nan)
    core["nonai_to_headroom"] = core["nonai_gw"] / core["headroom_gw"].replace(0, np.nan)
    core["net_supply_to_headroom"] = core["new_supply_gw"] / core["headroom_gw"].replace(0, np.nan)
    core["firm_share"] = core[["gas_share", "coal_share", "nuclear_share", "hydro_share"]].sum(axis=1)
    core["vre_share"] = core[["wind_share", "solar_share"]].sum(axis=1)
    core["iso_order"] = core["ISO"].map({iso: i for i, iso in enumerate(ISO_ORDER)})
    core = core.sort_values("iso_order").drop(columns=["iso_order"]).reset_index(drop=True)
    write(core, "fig3/fig3_core_iso_metrics_canonical.csv")

    # Cluster labels are data-supported but should be treated as descriptive with n=7.
    features = ["headroom_ratio", "ai_to_headroom", "nonai_to_headroom", "net_supply_to_headroom", "margin_gw"]
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        z = StandardScaler().fit_transform(core[features].astype(float))
        labels = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(z)
        sil = float(silhouette_score(z, labels))
        cl = core.copy()
        cl["cluster_id_raw"] = labels
        names = {}
        for cid, g in cl.groupby("cluster_id_raw"):
            if g["margin_gw"].mean() < 0 and g["ai_to_headroom"].mean() > 1.5:
                name = "low-headroom growth"
            elif g["margin_gw"].mean() < 0:
                name = "deficit pressure"
            elif g["net_supply_to_headroom"].mean() > 1.0:
                name = "supply-buffered expansion"
            else:
                name = "high-margin limited growth"
            names[cid] = name
        cl["cluster"] = cl["cluster_id_raw"].map(names)
        centers = []
        for cid, g in cl.groupby("cluster_id_raw"):
            row = {"cluster_id_raw": cid, "cluster": names[cid], "members": ",".join(g["ISO"])}
            for f in features:
                row[f"{f}_mean"] = float(g[f].mean())
            centers.append(row)
        cl["cluster_silhouette_score_global"] = sil
        centers = pd.DataFrame(centers)
    except Exception as exc:
        cl = core.copy()
        cl["cluster_id_raw"] = np.nan
        cl["cluster"] = "not computed"
        cl["cluster_error"] = repr(exc)
        centers = pd.DataFrame([{"cluster_error": repr(exc)}])

    def rule_archetype(row: pd.Series) -> str:
        if row["margin_gw"] < 0 and row["ai_to_headroom"] > 1.5:
            return "low-headroom growth"
        if row["margin_gw"] < 0:
            return "deficit pressure"
        if row["net_supply_to_headroom"] > 1.0:
            return "supply-buffered expansion"
        return "high-margin limited growth"

    cl["statistical_cluster_label"] = cl["cluster"]
    cl["archetype_rule_label"] = cl.apply(rule_archetype, axis=1)
    # Use the rule label as the plotting label; keep the raw cluster columns to
    # document the descriptive clustering support and its small-n fragility.
    cl["cluster"] = cl["archetype_rule_label"]
    write(cl, "fig3/fig3a_clustered_archetype_source_canonical.csv")
    write(centers, "fig3/fig3a_cluster_centers_canonical.csv")

    lmp = pd.read_csv(ANALYSIS / "_SUMMARY" / "tables" / "zone_lmp_stats_all.csv")
    lmp["ISO"] = lmp["iso"].map(norm_iso)
    lmp = lmp[lmp["ISO"].isin(ISO7)].copy()
    lmp = lmp[~lmp["zone"].astype(str).str.upper().str.contains("TOTAL", na=False)].copy()
    lmp["beta_gw"] = pd.to_numeric(lmp["beta_lmp_per_mw"], errors="coerce") * 1000.0
    lmp["beta_stress_gw"] = lmp["beta_gw"].clip(lower=0.0)
    lmp["mean_lmp"] = pd.to_numeric(lmp["mean_lmp"], errors="coerce")
    lmp = lmp.dropna(subset=["beta_gw", "mean_lmp"])
    write(lmp, "fig3/fig3_zone_lmp_stress_distribution_source_canonical.csv")
    lmp_sum = (
        lmp.groupby("ISO", as_index=False)
        .agg(
            n_zones=("zone", "nunique"),
            median_beta_gw=("beta_stress_gw", "median"),
            p90_beta_gw=("beta_stress_gw", lambda x: float(np.nanquantile(x, 0.9))),
            max_beta_gw=("beta_stress_gw", "max"),
            median_lmp=("mean_lmp", "median"),
            p90_lmp=("mean_lmp", lambda x: float(np.nanquantile(x, 0.9))),
        )
    )
    write(lmp_sum, "fig3/fig3_zone_lmp_stress_summary_canonical.csv")

    risk = core.merge(lmp_sum, on="ISO", how="left")
    risk["first_deficit_mid"] = risk["ISO"].map(
        fig2_traj.groupby("ISO").apply(lambda g: first_deficit_year(g.rename(columns={"mid_growth_margin_GW": "m"}), "m")).to_dict()
    )
    risk["shortfall_2035_mid_GW"] = (-risk["margin_gw"]).clip(lower=0.0)
    fingerprint_metrics = [
        "headroom_gw",
        "ai_gw",
        "nonai_gw",
        "new_supply_gw",
        "margin_gw",
        "ai_to_headroom",
        "p90_beta_gw",
        "firm_share",
        "vre_share",
        "first_deficit_mid",
    ]
    fp_actual = risk[["ISO", *fingerprint_metrics]].copy()
    fp_norm = fp_actual.copy()
    for col in fingerprint_metrics:
        vals = pd.to_numeric(fp_norm[col], errors="coerce")
        if vals.notna().sum() <= 1 or vals.max() == vals.min():
            fp_norm[col] = 0.5
        else:
            fp_norm[col] = (vals - vals.min()) / (vals.max() - vals.min())
    write(fp_actual, "fig3/fig3b_fingerprint_actual_values_canonical.csv")
    write(fp_norm, "fig3/fig3b_fingerprint_normalized_canonical.csv")

    tx = pd.read_csv(ANALYSIS / "_SUMMARY" / "tables" / "zone_tx_potential_2025_daily.csv")
    tx["ISO"] = tx["iso"].map(norm_iso)
    tx = tx[tx["ISO"].isin(ISO7)].copy()
    tx = tx[~tx["zone"].astype(str).str.upper().str.contains("TOTAL", na=False)].copy()
    tx["beta_gw"] = pd.to_numeric(tx["beta_eff"], errors="coerce") * 1000.0
    tx = tx.merge(core[["ISO", "ai_to_headroom", "margin_gw", "ai_gw"]], on="ISO", how="left")
    tx["screen_index"] = tx["beta_gw"].clip(lower=0) * tx["ai_to_headroom"]
    write(tx, "fig3/fig3d_zone_pressure_screening_canonical.csv")

    growth = iso_ai[iso_ai["year"].eq(2035)][["ISO", "scenario", "ai_growth_vs_mid2025_GW"]]
    ns = tx.merge(growth, on="ISO", how="left")
    ns["national_zone_screen_index"] = ns["beta_gw"].clip(lower=0) * ns["ai_growth_vs_mid2025_GW"]
    write(ns, "fig3/fig3e_national_zone_screening_canonical.csv")

    pjm_weights = pd.read_csv(PROJECT / "out" / "county_to_pjm_zone_area.csv", dtype={"GEOID": str})
    pjm_weights["GEOID"] = pjm_weights["GEOID"].astype(str).str.zfill(5)
    pjm = county[county["iso"].eq("PJM")].merge(pjm_weights, on="GEOID", how="inner")
    pjm["MW_alloc"] = pjm["MW_canonical"] * pd.to_numeric(pjm["share"], errors="coerce").fillna(0.0)
    pjm_zone = (
        pjm.groupby(["scenario", "year", "PJM_ZONE"], as_index=False)["MW_alloc"]
        .sum()
        .rename(columns={"PJM_ZONE": "zone", "MW_alloc": "ai_demand_MW"})
    )
    base = (
        pjm_zone[pjm_zone["scenario"].eq("mid") & pjm_zone["year"].eq(2025)]
        .set_index("zone")["ai_demand_MW"]
        .to_dict()
    )
    pjm_zone["ai_growth_vs_mid2025_MW"] = (pjm_zone["ai_demand_MW"] - pjm_zone["zone"].map(base).fillna(0.0)).clip(lower=0.0)
    write(pjm_zone, "fig3/fig3f_pjm_zone_ai_exposure_canonical.csv")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm_text(value: object) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def build_fig4_data(gap: pd.DataFrame) -> None:
    rec = pd.read_csv(ANALYSIS / "fig4_panel_b_recoverable_nuclear_capacity_timeseries.csv")
    rec["ISO"] = rec["ISO"].map(norm_iso)
    write(rec, "fig4/fig4_recoverable_nuclear_capacity_source.csv")

    units = pd.read_csv(ANALYSIS / "fig4_panel_a_unit_level_table.csv")
    units["ISO"] = units["ISO"].map(norm_iso)
    units["plant_norm"] = units["plant_name"].map(norm_text)
    units["unit_norm"] = units["unit"].astype(str).map(norm_text)
    units["raw_capacity_GW"] = pd.to_numeric(units["capacity_GW"], errors="coerce")
    matched = PROJECT / "out" / "nuclear_retirement_candidates_matched.csv"
    if matched.exists():
        m = pd.read_csv(matched)
        m["plant_norm"] = m["Plant Name"].map(norm_text)
        m["unit_norm"] = m["Generator ID"].astype(str).map(norm_text)
        m["retire_year"] = pd.to_numeric(m["retire_year"], errors="coerce")
        units = units.merge(
            m.dropna(subset=["retire_year"])[["plant_norm", "unit_norm", "retire_year"]].drop_duplicates(),
            on=["plant_norm", "unit_norm"],
            how="left",
        )
    else:
        units["retire_year"] = np.nan
    units["available_year"] = np.where(units["pathway"].eq("restart"), 2028, units["retire_year"].fillna(2035)).astype(int)
    rec2035 = rec[rec["year"].eq(2035)].set_index("ISO")
    factors = {}
    for (iso, pathway), g in units.groupby(["ISO", "pathway"]):
        raw = float(g["raw_capacity_GW"].sum())
        if raw <= 0:
            factor = 0.0
        elif pathway == "restart":
            factor = float(rec2035.loc[iso, "restart_recoverable_GW"]) / raw if iso in rec2035.index else 0.0
        else:
            factor = float(rec2035.loc[iso, "retention_recoverable_GW"]) / raw if iso in rec2035.index else 0.0
        factors[(iso, pathway)] = max(0.0, factor)
    units["recoverability_factor"] = [factors.get((r.ISO, r.pathway), 0.0) for r in units.itertuples()]
    units["recoverable_GW"] = units["raw_capacity_GW"] * units["recoverability_factor"]
    for ratio_name, ratio in [("low", 0.8), ("mid", 0.9), ("high", 1.0)]:
        units[f"effective_{ratio_name}_GW"] = units["recoverable_GW"] * ratio
    write(units, "fig4/fig4_unit_level_nuclear_candidates_canonical.csv")

    gap_long = []
    for scenario, col in {"low": "Gap_lowcase_MW", "mid": "Gap_midcase_MW", "high": "Gap_highcase_MW"}.items():
        tmp = gap[["ISO", "year", col]].rename(columns={col: "baseline_margin_MW"}).copy()
        tmp["scenario"] = scenario
        tmp["baseline_margin_GW"] = tmp["baseline_margin_MW"] / 1000.0
        gap_long.append(tmp)
    gap_long = pd.concat(gap_long, ignore_index=True)
    write(gap_long, "fig4/fig4_baseline_margin_canonical_input.csv")

    cases = gap_long.merge(rec, on=["ISO", "year"], how="left").fillna(
        {"retention_recoverable_GW": 0.0, "restart_recoverable_GW": 0.0, "total_recoverable_GW": 0.0}
    )
    case_rows = []
    for _, row in cases.iterrows():
        for recovery_case, ratio in [("low_recovery", 0.8), ("mid_recovery", 0.9), ("high_recovery", 1.0)]:
            for policy in ["baseline", "retention", "retention+restart"]:
                add = 0.0
                if policy in {"retention", "retention+restart"}:
                    add += float(row["retention_recoverable_GW"]) * ratio
                if policy == "retention+restart":
                    add += float(row["restart_recoverable_GW"]) * ratio
                margin = float(row["baseline_margin_GW"]) + add
                case_rows.append(
                    {
                        "ISO": row["ISO"],
                        "year": int(row["year"]),
                        "scenario": row["scenario"],
                        "recovery_case": recovery_case,
                        "policy_case": policy,
                        "nuclear_effective_addition_GW": add,
                        "margin_GW": margin,
                        "shortfall_GW": max(0.0, -margin),
                    }
                )
    policy_cases = pd.DataFrame(case_rows)
    write(policy_cases, "fig4/fig4_policy_margin_cases_canonical.csv")

    # Unit marginal value: annual GW-years of deficit avoided, across AI and recovery uncertainty.
    unit_rows = []
    for u in units.itertuples():
        for scenario in SCENARIOS:
            sub = gap_long[(gap_long["ISO"].eq(u.ISO)) & (gap_long["scenario"].eq(scenario))].copy()
            for recovery_case, ratio in [("low_recovery", 0.8), ("mid_recovery", 0.9), ("high_recovery", 1.0)]:
                cap = float(u.recoverable_GW) * ratio
                active = sub["year"].ge(int(u.available_year))
                deficit = (-sub["baseline_margin_GW"]).clip(lower=0.0)
                avoided = np.where(active, np.minimum(deficit, cap), 0.0)
                unit_rows.append(
                    {
                        "plant_name": u.plant_name,
                        "unit": u.unit,
                        "ISO": u.ISO,
                        "pathway": u.pathway,
                        "available_year": int(u.available_year),
                        "raw_capacity_GW": float(u.raw_capacity_GW),
                        "recoverable_GW": float(u.recoverable_GW),
                        "scenario": scenario,
                        "recovery_case": recovery_case,
                        "deficit_GW_years_avoided": float(np.nansum(avoided)),
                        "max_single_year_avoided_GW": float(np.nanmax(avoided)) if len(avoided) else 0.0,
                    }
                )
    unit_value = pd.DataFrame(unit_rows)
    keys = ["plant_name", "unit", "ISO", "pathway", "available_year", "raw_capacity_GW", "recoverable_GW"]
    unit_summary = (
        unit_value.groupby(keys, as_index=False)
        .agg(
            value_low=("deficit_GW_years_avoided", "min"),
            value_mid=("deficit_GW_years_avoided", "median"),
            value_high=("deficit_GW_years_avoided", "max"),
            max_single_year_avoided_GW=("max_single_year_avoided_GW", "max"),
        )
        .sort_values("value_mid", ascending=False)
    )
    write(unit_value, "fig4/fig4_unit_marginal_value_long_canonical.csv")
    write(unit_summary, "fig4/fig4_unit_marginal_value_summary_canonical.csv")

    offset_rows = []
    for iso in ["MISO", "PJM", "SPP"]:
        for scenario in SCENARIOS:
            base = gap_long[(gap_long["ISO"].eq(iso)) & (gap_long["scenario"].eq(scenario)) & (gap_long["year"].eq(2035))]
            if base.empty:
                continue
            shortfall = max(0.0, -float(base["baseline_margin_GW"].iloc[0]))
            recrow = rec2035.loc[iso] if iso in rec2035.index else None
            ret = 0.0 if recrow is None else float(recrow["retention_recoverable_GW"]) * 0.9
            rest = 0.0 if recrow is None else float(recrow["restart_recoverable_GW"]) * 0.9
            used_ret = min(shortfall, ret)
            used_restart = min(max(0.0, shortfall - used_ret), rest)
            offset_rows.append(
                {
                    "ISO": iso,
                    "scenario": scenario,
                    "baseline_shortfall_2035_GW": shortfall,
                    "retention_effective_mid_GW": ret,
                    "restart_effective_mid_GW": rest,
                    "retention_used_mid_GW": used_ret,
                    "restart_used_mid_GW": used_restart,
                    "offset_mid_GW": used_ret + used_restart,
                    "residual_shortfall_mid_GW": max(0.0, shortfall - used_ret - used_restart),
                }
            )
    write(pd.DataFrame(offset_rows), "fig4/fig4_2035_offset_decomposition_canonical.csv")


def load_onsite_module():
    return load_module("onsite_panel_d_canonical", PROJECT / "scripts" / "make_fig_onsite_panel_d.py")


def load_peak_and_pv_profiles():
    pc = pd.read_csv(ANALYSIS / "iso_coincidence_peak_window_3h.csv")
    pc = pc[(pc["dc_shape_name"].astype(str).eq("avg")) & (pc["ISO"].isin(["ERCOT", "MISO", "PJM", "SPP"]))].copy()
    peak_hour = pc.set_index("ISO")["iso_peak_hour"].astype(int)
    pro = pd.read_csv(ANALYSIS / "iso7_summer2025_eia930_mean_mw_by_local_hour_jja.csv")
    phi_day = {
        iso: np.nan_to_num(
            pro[pro["iso"].eq(iso)]
            .set_index("hour_1_24")["pv_nonneg_per_mwac"]
            .reindex(range(1, 25))
            .to_numpy(),
            nan=0.0,
        )[:24]
        for iso in ["ERCOT", "MISO", "PJM", "SPP"]
    }
    return peak_hour, phi_day


def onsite_margin_with_storage(mod, peak_hour, phi_day, *, hc_mw: float, nonai_mw: float, ai_mw: float, iso: str, pv_ratio: float, storage_h: float) -> dict[str, float]:
    phi = np.tile(phi_day[iso], mod.N_DAYS)
    w0 = mod.w_idx0_from_peak_multi(int(peak_hour.loc[iso]), mod.N_DAYS)
    g24 = mod.g_hourly_mw(ai_mw, pv_ratio, phi)
    pv24 = pv_ratio * ai_mw * phi
    pv_cap_mw = max(0.0, pv_ratio * ai_mw)
    p_sto_mw = mod.R_SP * pv_cap_mw
    e_mwh = storage_h * p_sto_mw
    if storage_h <= 0 or p_sto_mw <= 0:
        residual_ai_peak = mod.b_no_storage_mw(g24, w0)
    else:
        residual_ai_peak = mod.b_with_storage_mw(g24, pv24, w0, p_sto_mw, e_mwh)
    return {
        "HC_MW": hc_mw,
        "nonAI_growth_MW": nonai_mw,
        "incremental_ai_peak_MW": ai_mw,
        "PV_cap_MW": pv_cap_mw,
        "P_sto_MW": p_sto_mw,
        "E_MWh": e_mwh,
        "residual_ai_peak_after_pv_storage_MW": residual_ai_peak,
        "available_margin_MW_after_pv_storage": hc_mw - residual_ai_peak,
    }


def atb_gas_tech_table(atb: pd.DataFrame, year: int = 2035) -> pd.DataFrame:
    gas = atb[
        (atb["technology"] == "NaturalGas_FE")
        & (atb["core_metric_variable"] == year)
        & (atb["scenario"] == "Moderate")
        & (atb["core_metric_parameter"].isin(["Heat Rate", "Variable O&M", "CAPEX", "Fixed O&M", "LCOE"]))
    ].copy()
    gas = gas.drop_duplicates(["techdetail", "display_name", "core_metric_parameter", "value"])
    return gas.pivot_table(
        index=["techdetail", "display_name"], columns="core_metric_parameter", values="value", aggfunc="median"
    ).reset_index()


def build_fig5_data(gap: pd.DataFrame) -> None:
    # Data sources already downloaded/cached; no external API call is made here.
    for src, rel in [
        (FIG5_NASA / "nasa_power_site_solar_resource_metrics.csv", "fig5/fig5a_site_level_solar_resource_source.csv"),
        (FIG5_NASA / "nasa_power_iso_solar_resource_summary.csv", "fig5/fig5a_iso_solar_resource_summary.csv"),
        (FIG5_NASA / "site_coordinates_sent_to_nasa_power.csv", "fig5/fig5a_site_coordinates_sent_to_nasa_power_audit.csv"),
        (FIG5_SUPP / "fig5_official_cost_value_screen_iso_with_isone.csv", "fig5/fig5_price_cost_official_screen_with_isone.csv"),
        (FIG5_SUPP / "fig5_zone_price_cost_value_screen_with_isone.csv", "fig5/fig5_zone_price_cost_value_screen_with_isone.csv"),
        (FIG5_SUPP / "fig5_data_quality_flags.csv", "fig5/fig5_data_quality_flags_existing.csv"),
    ]:
        if src.exists():
            write(pd.read_csv(src), rel)

    req_rows = []
    scenario_cols = {"low": ("D_ai_low_MW", "Gap_lowcase_MW"), "mid": ("D_ai_mid_MW", "Gap_midcase_MW"), "high": ("D_ai_high_MW", "Gap_highcase_MW")}
    for scenario, (ai_col, margin_col) in scenario_cols.items():
        for _, row in gap.iterrows():
            ai = float(row[ai_col])
            signed_headroom = float(row["HC_MW"])
            system_shortfall = max(0.0, -float(row[margin_col]))
            baseline_nonai_shortfall = max(0.0, -signed_headroom)
            ai_attributable_shortfall = max(0.0, system_shortfall - baseline_nonai_shortfall)
            req_rows.append(
                {
                    "ISO_RTO": row["ISO"],
                    "scenario": scenario,
                    "year": int(row["year"]),
                    "incremental_ai_peak_MW": ai,
                    "nonAI_growth_MW": float(row["D_nonAI_delta_MW"]),
                    "hosting_capacity_MW": signed_headroom,
                    "hosting_capacity_signed_MW": signed_headroom,
                    "positive_headroom_cap0_MW": float(row["HC_cap0_MW"]),
                    "baseline_nonAI_shortfall_before_ai_MW": baseline_nonai_shortfall,
                    "baseline_margin_MW": float(row[margin_col]),
                    "system_shortfall_after_incremental_ai_MW": system_shortfall,
                    "firm_onsite_capacity_required_MW": ai_attributable_shortfall,
                    "firm_onsite_share_of_incremental_ai": ai_attributable_shortfall / ai if ai > 0 else 0.0,
                    "firm_onsite_share_capped_at_100pct": min(1.0, ai_attributable_shortfall / ai) if ai > 0 else 0.0,
                    "residual_nonAI_shortfall_after_full_ai_offset_MW": baseline_nonai_shortfall,
                }
            )
    req = pd.DataFrame(req_rows)
    write(req, "fig5/fig5_firm_selfgen_requirement_canonical.csv")

    mod = load_onsite_module()
    peak_hour, phi_day = load_peak_and_pv_profiles()
    grid_rows = []
    for scenario, (ai_col, margin_col) in scenario_cols.items():
        for iso in ["ERCOT", "MISO", "PJM", "SPP"]:
            sub = gap[gap["ISO"].eq(iso)].copy()
            for _, row in sub.iterrows():
                for pv_ratio in PV_GRID_FINE:
                    for storage_h in STORAGE_GRID_FINE:
                        out = onsite_margin_with_storage(
                            mod,
                            peak_hour,
                            phi_day,
                            hc_mw=float(row["HC_MW"]),
                            nonai_mw=float(row["D_nonAI_delta_MW"]),
                            ai_mw=float(row[ai_col]),
                            iso=iso,
                            pv_ratio=float(pv_ratio),
                            storage_h=float(storage_h),
                        )
                        baseline_nonai_shortfall = max(0.0, -out["HC_MW"])
                        system_residual = max(0.0, -out["available_margin_MW_after_pv_storage"])
                        ai_attributable_residual = max(0.0, system_residual - baseline_nonai_shortfall)
                        grid_rows.append(
                            {
                                "ISO_RTO": iso,
                                "scenario": scenario,
                                "year": int(row["year"]),
                                "pv_nameplate_ratio_to_incremental_ai_peak": float(pv_ratio),
                                "storage_duration_h": float(storage_h),
                                "storage_power_ratio_to_pv_capacity": float(mod.R_SP),
                                **out,
                                "baseline_nonAI_shortfall_before_ai_MW": baseline_nonai_shortfall,
                                "system_residual_firm_backstop_required_MW": system_residual,
                                "residual_firm_backstop_required_MW": ai_attributable_residual,
                                "residual_firm_share_of_incremental_ai_peak": ai_attributable_residual / float(row[ai_col]) if float(row[ai_col]) > 0 else 0.0,
                                "interpretation": "PV/storage offsets modeled AI increment only; signed HC_MW can be negative, so system residual and AI-attributable residual are recorded separately.",
                            }
                        )
    hybrid = pd.DataFrame(grid_rows)
    write(hybrid, "fig5/fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")

    # Canonical fingerprint table for the current Fig. 5b design.
    solar = pd.read_csv(FIG5_NASA / "nasa_power_iso_solar_resource_summary.csv").set_index("ISO")
    cost = pd.read_csv(FIG5_SUPP / "fig5_official_cost_value_screen_iso_with_isone.csv").set_index("ISO_RTO")
    firm = pd.read_csv(FIG5_SUPP / "fig5_firm_selfgen_fuel_cost_emissions_tradeoff.csv")
    co2 = (
        firm[firm["scenario"].eq("mid") & np.isclose(firm["capacity_factor_sensitivity"], 0.5)]
        .groupby("ISO_RTO")["annual_co2_million_metric_tons"]
        .max()
    )
    fp_rows = []
    for iso in ISO_ORDER:
        hsub = hybrid[
            hybrid["ISO_RTO"].eq(iso)
            & hybrid["scenario"].eq("mid")
            & hybrid["year"].eq(2035)
            & np.isclose(hybrid["pv_nameplate_ratio_to_incremental_ai_peak"], 1.0)
            & np.isclose(hybrid["storage_duration_h"], 4.0)
        ]
        req2035 = req[req["ISO_RTO"].eq(iso) & req["scenario"].eq("mid") & req["year"].eq(2035)]
        firm_gap = float(req2035["firm_onsite_capacity_required_MW"].iloc[0] / 1000.0) if not req2035.empty else np.nan
        if not hsub.empty:
            pv_residual = float(hsub["residual_firm_share_of_incremental_ai_peak"].iloc[0] * 100.0)
        elif np.isfinite(firm_gap) and firm_gap <= 1e-9:
            pv_residual = 0.0
        else:
            pv_residual = np.nan
        fp_rows.append(
            {
                "ISO": iso,
                "solar_kwh_m2_day": float(solar.loc[iso, "tilted_latitude_annual_kwh_m2_day_mw_weighted_mean"]) if iso in solar.index else np.nan,
                "winter_summer_solar_ratio_pct": float(solar.loc[iso, "winter_to_summer_tilted_lat_ratio_mw_weighted_mean"] * 100.0) if iso in solar.index else np.nan,
                "retail_minus_pv_lcoe_dollars_mwh": float(cost.loc[iso, "commercial_retail_minus_comm_pv_lcoe_median"]) if iso in cost.index else np.nan,
                "p95_lmp_minus_gas_cost_dollars_mwh": float(cost.loc[iso, "p95_lmp_minus_gas_variable_cost_median"]) if iso in cost.index else np.nan,
                "firm_gap_2035_GW": firm_gap,
                "pv_100pct_4h_residual_pct": pv_residual,
                "gas_co2_mt_mid_cf50": float(co2.get(iso, np.nan)),
            }
        )
    write(pd.DataFrame(fp_rows), "fig5/fig5b_on_site_generation_fingerprint_canonical.csv")

    gas_iso = pd.read_csv(FIG5_DATA / "eia_2024_iso_state_proxy_gas_generation_cost.csv")
    atb = pd.read_csv(FIG5_DATA / "nrel_atb_2024_cost_screen_pv_storage_gas_long.csv", low_memory=False)
    gas_tech = atb_gas_tech_table(atb)
    gas_simple = gas_tech[~gas_tech["techdetail"].str.contains("CCS", case=False, na=False)].copy()
    trade_rows = []
    for _, r in req[req["year"].eq(2035) & req["scenario"].isin(SCENARIOS)].iterrows():
        gas_row = gas_iso[gas_iso["ISO_RTO"].eq(r["ISO_RTO"])]
        gas_cost = float(gas_row["gas_cost_2024_mean_dollars_per_mmbtu"].iloc[0]) if not gas_row.empty else np.nan
        for _, t in gas_simple.dropna(subset=["Heat Rate"]).iterrows():
            heat = float(t["Heat Rate"])
            vom = float(t["Variable O&M"]) if pd.notna(t.get("Variable O&M")) else 0.0
            for cf in [0.05, 0.10, 0.25, 0.50, 0.90]:
                cap_mw = float(r["firm_onsite_capacity_required_MW"])
                gen_mwh = cap_mw * 8760.0 * cf
                co2_rate = heat * 53.06
                trade_rows.append(
                    {
                        "ISO_RTO": r["ISO_RTO"],
                        "scenario": r["scenario"],
                        "firm_onsite_capacity_required_MW": cap_mw,
                        "firm_onsite_share_of_incremental_ai": r["firm_onsite_share_of_incremental_ai"],
                        "gas_techdetail_atb": t["techdetail"],
                        "gas_heat_rate_mmbtu_mwh_atb": heat,
                        "gas_variable_om_dollars_mwh_atb": vom,
                        "state_proxy_gas_cost_dollars_mmbtu_eia": gas_cost,
                        "variable_fuel_plus_vom_cost_dollars_mwh": heat * gas_cost + vom if pd.notna(gas_cost) else np.nan,
                        "epa_natural_gas_co2_kg_mmbtu": 53.06,
                        "co2_rate_kg_mwh": co2_rate,
                        "capacity_factor_sensitivity": cf,
                        "annual_generation_mwh": gen_mwh,
                        "annual_co2_million_metric_tons": gen_mwh * co2_rate / 1e9,
                    }
                )
    write(pd.DataFrame(trade_rows), "fig5/fig5_firm_selfgen_fuel_cost_emissions_canonical.csv")


def build_audit_tables(county_audit: pd.DataFrame, iso_ai: pd.DataFrame, gap: pd.DataFrame, bridge: pd.DataFrame) -> None:
    issues = []
    # 1. Scenario monotonicity of ISO demand.
    pivot = iso_ai.pivot_table(index=["ISO", "year"], columns="scenario", values="ai_demand_MW")
    bad = pivot[(pivot["low"] > pivot["mid"] + 1e-6) | (pivot["mid"] > pivot["high"] + 1e-6)].reset_index()
    if bad.empty:
        issues.append({"severity": "pass", "check": "AI scenario monotonicity", "finding": "low <= mid <= high for all ISO-years."})
    else:
        issues.append({"severity": "warning", "check": "AI scenario monotonicity", "finding": f"{len(bad)} ISO-year rows violate low <= mid <= high; likely due to scenario-specific spatial allocation shares."})
        write(bad, "audit/ai_scenario_monotonicity_violations.csv")

    # 2. Canonical accounting identity.
    err = (bridge["Supply_total_MW"] - bridge["AI_growth_to_2035_MW"] - bridge["Margin_2035_MW"]).abs().max()
    issues.append(
        {
            "severity": "pass" if err < 1e-6 else "error",
            "check": "2035 margin identity",
            "finding": f"max |signed net slack - AI - margin| = {err:.6g} MW. Non-AI load is not subtracted again because it is embodied in HC_MW.",
        }
    )

    # 2b. Signed slack is retained separately from cap-at-zero visualization headroom.
    neg_raw = gap[(gap["year"].eq(2035)) & (gap["HC_signed_MW"] < -1e-6)][
        ["ISO", "HC_signed_MW", "HC_cap0_MW", "baseline_nonAI_shortfall_raw_MW"]
    ].copy()
    if neg_raw.empty:
        issues.append(
            {
                "severity": "pass",
                "check": "signed slack vs cap-at-zero headroom",
                "finding": "No ISO has negative signed 2035 slack under the baseline headroom extension.",
            }
        )
    else:
        issues.append(
            {
                "severity": "warning",
                "check": "signed slack vs cap-at-zero headroom",
                "finding": "; ".join(
                    f"{r.ISO}: signed slack={r.HC_signed_MW/1000:.1f} GW, cap0 headroom={r.HC_cap0_MW/1000:.1f} GW"
                    for r in neg_raw.itertuples()
                )
                + ". Use signed values in data tables and reserve cap0 values for explicitly positive headroom displays.",
            }
        )
        write(neg_raw, "audit/signed_slack_negative_cap0_comparison.csv")

    # 3. Compare old vs canonical bridge to expose material conclusion changes.
    old_bridge = pd.read_csv(ANALYSIS / "iso7_pressure_headroom_vs_2035demand_mid.csv")
    old_bridge["ISO"] = old_bridge["ISO"].map(norm_iso)
    comp = bridge.merge(
        old_bridge[["ISO", "AI_growth_to_2035_MW", "Margin_2035_MW"]].rename(
            columns={"AI_growth_to_2035_MW": "old_AI_growth_to_2035_MW", "Margin_2035_MW": "old_Margin_2035_MW"}
        ),
        on="ISO",
        how="left",
    )
    comp["canonical_AI_growth_to_2035_MW"] = comp["AI_growth_to_2035_MW"]
    comp["canonical_Margin_2035_MW"] = comp["Margin_2035_MW"]
    comp["AI_growth_change_GW"] = (comp["canonical_AI_growth_to_2035_MW"] - comp["old_AI_growth_to_2035_MW"]) / 1000.0
    comp["margin_change_GW"] = (comp["canonical_Margin_2035_MW"] - comp["old_Margin_2035_MW"]) / 1000.0
    write(comp, "audit/old_vs_canonical_fig2_bridge_comparison.csv")
    material = comp[comp["margin_change_GW"].abs() >= 1.0]
    issues.append(
        {
            "severity": "warning" if not material.empty else "pass",
            "check": "material changes from old ISO-normalized/cap0 demand",
            "finding": f"{len(material)} ISOs change 2035 margin by >=1 GW after preserving non-ISO share, national total and signed slack.",
        }
    )

    # 4. Negative new supply is possible, but should be explicitly described as retirements/headroom decline.
    neg_supply = bridge[bridge["New_supply_to_2035_MW"] < -1e-6]
    issues.append(
        {
            "severity": "note" if not neg_supply.empty else "pass",
            "check": "negative net supply/headroom change",
            "finding": "Negative New_supply_to_2035_MW appears for " + ", ".join(neg_supply["ISO"]) + "; label as net headroom decline, not buildout."
            if not neg_supply.empty
            else "No negative New_supply_to_2035_MW values.",
        }
    )

    # 5. ISO7 is a subset, not the national total.
    ca = county_audit[county_audit["scenario"].eq("mid") & county_audit["year"].isin([2025, 2035])].copy()
    issues.append(
        {
            "severity": "pass",
            "check": "ISO7/non-ISO separation",
            "finding": "; ".join(
                f"{int(r.year)} mid raw ISO7 share={r.raw_ISO7_share:.1%}, nonISO coincident={r.nonISO_coincident_GW:.1f} GW"
                for r in ca.itertuples()
            ),
        }
    )
    write(pd.DataFrame(issues), "audit/canonical_result_audit_summary.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    county, county_audit = build_canonical_county_yearly()
    iso_ai = build_iso_ai_demand(county)
    gap, bridge = build_canonical_gap(iso_ai)
    fig2 = build_fig2_panels(gap, bridge)
    build_fig3_data(bridge, fig2["pipe"], fig2["traj"], iso_ai, county)
    build_fig4_data(gap)
    build_fig5_data(gap)
    build_audit_tables(county_audit, iso_ai, gap, bridge)
    print(f"\nDone. Canonical Fig.2-5 data written to {OUT}")


if __name__ == "__main__":
    main()
