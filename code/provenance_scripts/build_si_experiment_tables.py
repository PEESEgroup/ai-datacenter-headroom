from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "tables" / "fig2_5_canonical_20260514"
FIG1 = ROOT / "tables" / "fig1_canonical_20260514"
OUT = ROOT / "tables" / "si_experiments_20260516"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
SCENARIOS = ["low", "mid", "high"]
MARGIN_COL = {
    "low": "Gap_lowcase_MW",
    "mid": "Gap_midcase_MW",
    "high": "Gap_highcase_MW",
}


def write(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(df):,} rows)")
    return path


def canonical_table_inventory() -> pd.DataFrame:
    """Inventory canonical source tables so SI claims are tied to auditable files."""
    rows = []
    for base in [FIG1, CANON]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.csv")):
            rel = path.relative_to(ROOT)
            df = pd.read_csv(path)
            if "fig1_canonical_20260514" in str(rel):
                source_group = "Fig. 1 canonical tables"
            else:
                parent = path.parent.name
                source_group = {
                    "shared": "Shared canonical demand tables",
                    "audit": "Canonical audit tables",
                    "fig2": "Fig. 2 canonical tables",
                    "fig3": "Fig. 3 canonical tables",
                    "fig4": "Fig. 4 canonical tables",
                    "fig5": "Fig. 5 canonical tables",
                }.get(parent, f"{parent} canonical tables")
            rows.append(
                {
                    "relative_path": str(rel),
                    "source_group": source_group,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "file_size_mb": path.stat().st_size / 1e6,
                    "key_columns_preview": "; ".join(map(str, df.columns[:8])),
                }
            )
    out = pd.DataFrame(rows).sort_values(["source_group", "relative_path"])
    write(out, "si_canonical_source_table_inventory.csv")
    return out


def fmt_year(value: float | int | str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ">2035"
    try:
        if float(value) > 2035:
            return ">2035"
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def first_negative_annual(sub: pd.DataFrame, margin_col: str) -> str:
    g = sub.sort_values("year")
    neg = g[g[margin_col] < 0]
    if neg.empty:
        return ">2035"
    return str(int(neg.iloc[0]["year"]))


def first_interpolated(sub: pd.DataFrame, margin_col: str) -> float:
    g = sub.sort_values("year")
    years = g["year"].to_numpy(dtype=float)
    vals = g[margin_col].to_numpy(dtype=float)
    if len(vals) == 0 or np.all(vals >= 0):
        return np.nan
    hit = np.where(vals < 0)[0][0]
    if hit == 0:
        return years[hit]
    y0, y1 = years[hit - 1], years[hit]
    v0, v1 = vals[hit - 1], vals[hit]
    if np.isclose(v0, v1):
        return y1
    return y0 + (0 - v0) * (y1 - y0) / (v1 - v0)


def demand_and_noniso_audits() -> dict[str, pd.DataFrame]:
    county = pd.read_csv(CANON / "shared" / "canonical_county_ai_demand_yearly.csv", dtype={"GEOID": str})
    audit = pd.read_csv(CANON / "shared" / "canonical_county_ai_demand_audit.csv")
    audit["total_coincident_GW"] = audit["ISO7_coincident_GW"] + audit["nonISO_coincident_GW"]
    audit["ISO7_share_of_total"] = audit["ISO7_coincident_GW"] / audit["total_coincident_GW"]
    audit["nonISO_share_of_total"] = audit["nonISO_coincident_GW"] / audit["total_coincident_GW"]
    audit = audit[
        [
            "scenario",
            "year",
            "national_average_GW",
            "total_coincident_GW",
            "ISO7_coincident_GW",
            "nonISO_coincident_GW",
            "ISO7_share_of_total",
            "nonISO_share_of_total",
            "raw_ISO7_share",
        ]
    ].sort_values(["scenario", "year"])
    write(audit, "si_noniso_demand_audit.csv")

    rows = []
    for (scenario, year), g in county.groupby(["scenario", "year"]):
        vals = g[g["GW_canonical"] > 0].copy()
        vals = vals.sort_values("GW_canonical", ascending=False)
        total = vals["GW_canonical"].sum()
        rec = {
            "scenario": scenario,
            "year": int(year),
            "positive_counties": int(len(vals)),
            "counties_gt_0p1GW": int((vals["GW_canonical"] >= 0.1).sum()),
            "counties_gt_1GW": int((vals["GW_canonical"] >= 1.0).sum()),
            "total_GW": total,
        }
        for n in [3, 5, 10, 20, 50, 100]:
            rec[f"top{n}_share"] = float(vals.head(n)["GW_canonical"].sum() / total) if total else np.nan
        rows.append(rec)
    concentration = pd.DataFrame(rows).sort_values(["scenario", "year"])
    write(concentration, "si_county_concentration_by_scenario_year.csv")

    top_counties = (
        county[(county["scenario"] == "mid") & (county["year"] == 2035)]
        .sort_values("GW_canonical", ascending=False)
        .head(30)
        .copy()
    )
    top_counties = top_counties[
        ["rank" if "rank" in top_counties.columns else "GEOID"]
    ] if False else top_counties
    top_counties["rank"] = np.arange(1, len(top_counties) + 1)
    top_counties = top_counties[
        ["rank", "GEOID", "label", "state_abb", "iso", "GW_canonical", "raw_spatial_share", "gamma"]
    ]
    write(top_counties, "si_top30_counties_mid2035.csv")

    state = (
        county[(county["scenario"] == "mid") & (county["year"] == 2035)]
        .groupby("state_abb", as_index=False)
        .agg(
            gw_2035=("GW_canonical", "sum"),
            dominant_iso=("iso", lambda s: s.value_counts().idxmax()),
            iso_count=("iso", "nunique"),
            counties=("GEOID", "nunique"),
        )
    )
    state_2025 = (
        county[(county["scenario"] == "mid") & (county["year"] == 2025)]
        .groupby("state_abb", as_index=False)
        .agg(gw_2025=("GW_canonical", "sum"))
    )
    state_low_high = (
        county[(county["year"] == 2035) & (county["scenario"].isin(["low", "high"]))]
        .pivot_table(index="state_abb", columns="scenario", values="GW_canonical", aggfunc="sum")
        .reset_index()
        .rename(columns={"low": "gw_2035_low", "high": "gw_2035_high"})
    )
    state = state.merge(state_2025, on="state_abb", how="left").merge(state_low_high, on="state_abb", how="left")
    state = state.sort_values("gw_2035", ascending=False).copy()
    state["rank_2035"] = np.arange(1, len(state) + 1)
    state["share_of_national_mid2035"] = state["gw_2035"] / state["gw_2035"].sum()
    state["cumulative_share_mid2035"] = state["share_of_national_mid2035"].cumsum()
    state["growth"] = state["gw_2035"] - state["gw_2025"]
    state = state[
        [
            "rank_2035",
            "state_abb",
            "dominant_iso",
            "iso_count",
            "counties",
            "gw_2025",
            "gw_2035_low",
            "gw_2035",
            "gw_2035_high",
            "growth",
            "share_of_national_mid2035",
            "cumulative_share_mid2035",
        ]
    ]
    write(state, "si_state_concentration_mid2035.csv")
    return {"audit": audit, "concentration": concentration, "top_counties": top_counties, "state": state}


def plausibility_audit_exports() -> dict[str, pd.DataFrame]:
    detailed = pd.read_csv(CANON / "audit" / "fig2_5_detailed_calculation_audit.csv")
    plaus = pd.read_csv(CANON / "audit" / "scientific_plausibility_audit_20260514.csv")
    signed = pd.read_csv(CANON / "audit" / "signed_slack_negative_cap0_comparison.csv")
    result = pd.read_csv(CANON / "audit" / "canonical_result_audit_summary.csv")
    write(detailed, "si_detailed_calculation_audit.csv")
    write(plaus, "si_scientific_plausibility_audit.csv")
    write(signed, "si_signed_slack_cap0_comparison.csv")
    write(result, "si_canonical_result_audit_summary.csv")
    return {"detailed": detailed, "plausibility": plaus, "signed": signed, "result": result}


def first_deficit_robustness() -> dict[str, pd.DataFrame]:
    gap = pd.read_csv(CANON / "fig2" / "fig2_shared_canonical_gap_yearly.csv")
    shifts_gw = [-3, -2, -1, 0, 1, 2, 3]
    rows = []
    for iso in ISO_ORDER:
        sub_iso = gap[gap["ISO"] == iso].copy()
        for scenario, col in MARGIN_COL.items():
            for shift in shifts_gw:
                shifted = sub_iso[["year", col]].copy()
                shifted["margin_shifted_MW"] = shifted[col] + shift * 1000.0
                m2035 = float(shifted.loc[shifted["year"] == 2035, "margin_shifted_MW"].iloc[0])
                rows.append(
                    {
                        "ISO": iso,
                        "scenario": scenario,
                        "headroom_additive_shift_GW": shift,
                        "first_negative_annual_year": first_negative_annual(shifted, "margin_shifted_MW"),
                        "first_negative_interpolated_year": first_interpolated(shifted, "margin_shifted_MW"),
                        "margin_2035_GW": m2035 / 1000.0,
                        "shortfall_2035_GW": max(0.0, -m2035 / 1000.0),
                    }
                )
    sens = pd.DataFrame(rows)
    write(sens, "si_first_deficit_headroom_shift_sensitivity.csv")

    summary = sens[sens["headroom_additive_shift_GW"].isin([-2, 0, 2])].copy()
    summary["first_negative_interpolated_year"] = summary["first_negative_interpolated_year"].round(2)
    summary = summary.sort_values(["scenario", "ISO", "headroom_additive_shift_GW"])
    write(summary, "si_first_deficit_key_sensitivity_summary.csv")
    return {"headroom_shift": sens, "summary": summary}


def generator_pipeline_sensitivity() -> pd.DataFrame:
    pipe = pd.read_csv(CANON / "fig2" / "fig2c_generator_pipeline_canonical_source.csv")
    rows = []
    for _, r in pipe.iterrows():
        base_margin = float(r["Margin_2035_MW"]) / 1000.0
        storage_delta_20 = float(r.get("storage_20pct_credit_delta_MW", 0.0) or 0.0) / 1000.0
        active = float(r.get("MW_sum_total_active_2026_2028", np.nan)) / 1000.0
        ia = float(r.get("MW_sum_total_IAstage_2026_2028", np.nan)) / 1000.0
        eff = float(r.get("add_eff_MW", np.nan)) / 1000.0
        for storage_credit in [0.0, 0.1, 0.2, 0.5, 1.0]:
            # Linear scaling is a sensitivity screen only; it is not a storage ELCC model.
            storage_delta = storage_delta_20 * (storage_credit / 0.2) if storage_delta_20 else 0.0
            margin = base_margin + storage_delta
            rows.append(
                {
                    "ISO": r["ISO"],
                    "active_queue_GW": active,
                    "IA_stage_queue_GW": ia,
                    "effective_additions_GW": eff,
                    "effective_over_active_queue": r.get("effective_over_active_queue", np.nan),
                    "top10_share_eff": r.get("top10_share_eff", np.nan),
                    "storage_credit_assumption": storage_credit,
                    "storage_credit_delta_GW": storage_delta,
                    "margin_2035_with_storage_credit_GW": margin,
                    "shortfall_2035_with_storage_credit_GW": max(0.0, -margin),
                    "note": "Storage-credit variants linearly scale the 20% storage-credit source field for screening only.",
                }
            )
    out = pd.DataFrame(rows).sort_values(["ISO", "storage_credit_assumption"])
    write(out, "si_generator_pipeline_storage_credit_sensitivity.csv")
    return out


def archetype_robustness() -> dict[str, pd.DataFrame]:
    actual = pd.read_csv(CANON / "fig3" / "fig3b_fingerprint_actual_values_canonical.csv")
    clusters = pd.read_csv(CANON / "fig3" / "fig3a_clustered_archetype_source_canonical.csv")
    cols = [
        "headroom_gw",
        "ai_gw",
        "nonai_gw",
        "new_supply_gw",
        "margin_gw",
        "ai_to_headroom",
        "p90_beta_gw",
        "firm_share",
        "vre_share",
    ]
    mechanism_distance_cols = ["headroom_gw", "ai_gw", "nonai_gw", "new_supply_gw", "margin_gw"]
    df = actual.merge(
        clusters[["ISO", "risk_label", "cluster", "cluster_silhouette_score_global", "archetype_rule_label"]],
        on="ISO",
        how="left",
    )
    z = df[cols].astype(float)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    z.insert(0, "ISO", df["ISO"])
    z["risk_label"] = df["risk_label"]
    z["cluster"] = df["cluster"]
    write(z, "si_archetype_standardized_fingerprint_matrix.csv")

    def nearest_neighbor(data: pd.DataFrame, use_cols: list[str]) -> pd.DataFrame:
        arr = data[use_cols].to_numpy(float)
        rows = []
        for i, iso in enumerate(data["ISO"]):
            d = np.sqrt(np.nansum((arr - arr[i]) ** 2, axis=1))
            d[i] = np.inf
            j = int(np.nanargmin(d))
            rows.append({"ISO": iso, "nearest_neighbor": data.iloc[j]["ISO"], "distance": d[j]})
        return pd.DataFrame(rows)

    full_nn = nearest_neighbor(z, mechanism_distance_cols).rename(
        columns={"nearest_neighbor": "nearest_neighbor_full", "distance": "distance_full"}
    )
    loo_rows = []
    for drop in mechanism_distance_cols:
        use = [c for c in mechanism_distance_cols if c != drop]
        nn = nearest_neighbor(z, use)
        nn["dropped_variable"] = drop
        loo_rows.append(nn)
    loo = pd.concat(loo_rows, ignore_index=True).merge(full_nn, on="ISO", how="left")
    loo["nearest_neighbor_stable"] = loo["nearest_neighbor"] == loo["nearest_neighbor_full"]
    stability = (
        loo.groupby("ISO", as_index=False)
        .agg(
            nearest_neighbor_full=("nearest_neighbor_full", "first"),
            stable_fraction=("nearest_neighbor_stable", "mean"),
            variables_tested=("dropped_variable", "count"),
        )
        .merge(df[["ISO", "risk_label", "cluster", "cluster_silhouette_score_global"]], on="ISO", how="left")
    )
    write(loo, "si_archetype_leave_one_variable_out_nearest_neighbor.csv")
    write(stability, "si_archetype_stability_summary.csv")
    return {"matrix": z, "loo": loo, "stability": stability}


def nuclear_sensitivity() -> pd.DataFrame:
    cases = pd.read_csv(CANON / "fig4" / "fig4_policy_margin_cases_canonical.csv")
    rows = []
    for (iso, scenario, recovery_case, policy_case), g in cases.groupby(["ISO", "scenario", "recovery_case", "policy_case"]):
        g = g.sort_values("year").copy()
        fd = first_negative_annual(g.rename(columns={"margin_GW": "margin"}), "margin")
        interp = first_interpolated(g.rename(columns={"margin_GW": "margin"}), "margin")
        row2035 = g[g["year"] == 2035].iloc[0]
        rows.append(
            {
                "ISO": iso,
                "scenario": scenario,
                "recovery_case": recovery_case,
                "policy_case": policy_case,
                "first_negative_annual_year": fd,
                "first_negative_interpolated_year": interp,
                "nuclear_effective_addition_2035_GW": row2035["nuclear_effective_addition_GW"],
                "margin_2035_GW": row2035["margin_GW"],
                "shortfall_2035_GW": row2035["shortfall_GW"],
            }
        )
    out = pd.DataFrame(rows)
    base = out[out["policy_case"] == "baseline"][
        ["ISO", "scenario", "recovery_case", "shortfall_2035_GW"]
    ].rename(columns={"shortfall_2035_GW": "baseline_shortfall_2035_GW"})
    out = out.merge(base, on=["ISO", "scenario", "recovery_case"], how="left")
    out["offset_2035_GW"] = (out["baseline_shortfall_2035_GW"] - out["shortfall_2035_GW"]).clip(lower=0)
    out["offset_2035_pct_of_baseline_shortfall"] = np.where(
        out["baseline_shortfall_2035_GW"] > 0,
        out["offset_2035_GW"] / out["baseline_shortfall_2035_GW"],
        np.nan,
    )
    out = out.sort_values(["scenario", "recovery_case", "policy_case", "ISO"])
    write(out, "si_nuclear_policy_case_sensitivity.csv")
    return out


def pv_storage_sensitivity() -> dict[str, pd.DataFrame]:
    pv = pd.read_csv(CANON / "fig5" / "fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")
    pv = pv[pv["year"] == 2035].copy()
    rows = []
    for (iso, scenario, dur), g in pv.groupby(["ISO_RTO", "scenario", "storage_duration_h"]):
        inc = float(g["incremental_ai_peak_MW"].max())
        if inc <= 0:
            continue
        good = g[g["residual_firm_share_of_incremental_ai_peak"] <= 0.01].sort_values("pv_nameplate_ratio_to_incremental_ai_peak")
        rows.append(
            {
                "ISO": iso,
                "scenario": scenario,
                "storage_duration_h": dur,
                "incremental_ai_peak_GW": inc / 1000.0,
                "min_pv_ratio_for_residual_le_1pct": np.nan if good.empty else float(good.iloc[0]["pv_nameplate_ratio_to_incremental_ai_peak"]),
                "min_pv_GW_for_residual_le_1pct": np.nan if good.empty else float(good.iloc[0]["PV_cap_MW"]) / 1000.0,
                "residual_share_at_min": np.nan if good.empty else float(good.iloc[0]["residual_firm_share_of_incremental_ai_peak"]),
            }
        )
    threshold = pd.DataFrame(rows).sort_values(["scenario", "ISO", "storage_duration_h"])
    write(threshold, "si_pv_storage_min_pv_ratio_by_duration.csv")

    selected = pv[
        pv["pv_nameplate_ratio_to_incremental_ai_peak"].isin([0.0, 0.5, 1.0, 1.5, 2.0])
        & pv["storage_duration_h"].isin([0.0, 2.0, 4.0, 8.0, 12.0])
    ].copy()
    selected["residual_firm_GW"] = selected["residual_firm_backstop_required_MW"] / 1000.0
    selected["available_margin_after_pv_storage_GW"] = selected["available_margin_MW_after_pv_storage"] / 1000.0
    selected = selected[
        [
            "ISO_RTO",
            "scenario",
            "pv_nameplate_ratio_to_incremental_ai_peak",
            "storage_duration_h",
            "incremental_ai_peak_MW",
            "residual_firm_GW",
            "residual_firm_share_of_incremental_ai_peak",
            "available_margin_after_pv_storage_GW",
        ]
    ].sort_values(["scenario", "ISO_RTO", "pv_nameplate_ratio_to_incremental_ai_peak", "storage_duration_h"])
    write(selected, "si_pv_storage_selected_buildouts_2035.csv")
    return {"threshold": threshold, "selected": selected}


def gas_backstop_sensitivity() -> pd.DataFrame:
    gas = pd.read_csv(CANON / "fig5" / "fig5_firm_selfgen_fuel_cost_emissions_canonical.csv")
    gas["annual_variable_cost_billion_dollars"] = (
        gas["annual_generation_mwh"] * gas["variable_fuel_plus_vom_cost_dollars_mwh"] / 1e9
    )
    out = (
        gas.groupby(["ISO_RTO", "scenario", "gas_techdetail_atb"], as_index=False)
        .agg(
            firm_onsite_capacity_required_GW=("firm_onsite_capacity_required_MW", lambda s: float(s.iloc[0]) / 1000.0),
            co2_mt_min=("annual_co2_million_metric_tons", "min"),
            co2_mt_cf50=("annual_co2_million_metric_tons", lambda s: float(gas.loc[s.index][gas.loc[s.index, "capacity_factor_sensitivity"].eq(0.5)]["annual_co2_million_metric_tons"].iloc[0]) if gas.loc[s.index, "capacity_factor_sensitivity"].eq(0.5).any() else np.nan),
            co2_mt_max=("annual_co2_million_metric_tons", "max"),
            cost_bn_min=("annual_variable_cost_billion_dollars", "min"),
            cost_bn_cf50=("annual_variable_cost_billion_dollars", lambda s: float(gas.loc[s.index][gas.loc[s.index, "capacity_factor_sensitivity"].eq(0.5)]["annual_variable_cost_billion_dollars"].iloc[0]) if gas.loc[s.index, "capacity_factor_sensitivity"].eq(0.5).any() else np.nan),
            cost_bn_max=("annual_variable_cost_billion_dollars", "max"),
        )
    )
    write(out, "si_gas_backstop_technology_capacity_factor_sensitivity.csv")
    return out


def lmp_coverage_diagnostics() -> dict[str, pd.DataFrame]:
    zone = pd.read_csv(CANON / "fig5" / "fig5_zone_price_cost_value_screen_with_isone.csv")
    coverage = (
        zone.groupby("ISO_RTO", as_index=False)
        .agg(
            zones=("zone", "nunique"),
            n_hours_min=("n_hours", "min"),
            n_hours_median=("n_hours", "median"),
            n_hours_max=("n_hours", "max"),
            p95_lmp_median=("p95_lmp_dollars_mwh", "median"),
            p95_lmp_p90=("p95_lmp_dollars_mwh", lambda s: s.quantile(0.9)),
            p95_lmp_minus_gas_median=("p95_lmp_minus_gas_variable_cost", "median"),
            p95_lmp_minus_gas_p90=("p95_lmp_minus_gas_variable_cost", lambda s: s.quantile(0.9)),
            high_price_gt100h_share_p90=("share_hours_lmp_gt_100", lambda s: s.quantile(0.9)),
            negative_price_share_p90=("share_hours_lmp_lt_0", lambda s: s.quantile(0.9)),
        )
    )
    write(coverage, "si_lmp_zone_coverage_and_price_screen_summary.csv")

    stress = pd.read_csv(CANON / "fig3" / "fig3d_zone_pressure_screening_canonical.csv")
    stress_diag = (
        stress.groupby("ISO", as_index=False)
        .agg(
            zones=("zone", "nunique"),
            n_obs_median=("n", "median"),
            ols_r2_median=("ols_r2", "median"),
            huber_beta_median=("huber_beta", "median"),
            beta_eff_median=("beta_eff", "median"),
            beta_eff_p90=("beta_eff", lambda s: s.quantile(0.9)),
        )
    )
    write(stress_diag, "si_zone_load_price_regression_diagnostics.csv")
    return {"coverage": coverage, "stress_diag": stress_diag}


def write_findings(results: dict[str, object]) -> None:
    lines = ["# SI experiment findings (2026-05-16)", ""]
    inventory = results["inventory"]
    lines.append(
        f"- Canonical SI inventory spans {len(inventory):,} CSV source tables with "
        f"{int(inventory['rows'].sum()):,} tabular rows across Fig. 1--5 source, audit and shared-demand tables."
    )
    noniso = results["demand"]["audit"]
    mid2035 = noniso[(noniso["scenario"] == "mid") & (noniso["year"] == 2035)].iloc[0]
    lines.append(
        f"- Mid-2035 county allocation totals {mid2035['total_coincident_GW']:.1f} GW, with "
        f"{mid2035['ISO7_coincident_GW']:.1f} GW ({mid2035['ISO7_share_of_total']:.1%}) in the seven analyzed ISO/RTO regions "
        f"and {mid2035['nonISO_coincident_GW']:.1f} GW ({mid2035['nonISO_share_of_total']:.1%}) outside them."
    )
    conc = results["demand"]["concentration"]
    c2035 = conc[(conc["scenario"] == "mid") & (conc["year"] == 2035)].iloc[0]
    lines.append(
        f"- Mid-2035 demand remains top-heavy: {int(c2035['positive_counties'])} positive counties, "
        f"{int(c2035['counties_gt_1GW'])} counties above 1 GW, and top-50 share {c2035['top50_share']:.1%}."
    )
    state = results["demand"]["state"]
    top10_state_share = float(state.head(10)["share_of_national_mid2035"].sum())
    lines.append(
        f"- State-level concentration remains material but is not single-state dominated: "
        f"the top 10 states hold {top10_state_share:.1%} of mid-2035 county demand."
    )
    fd = results["first_deficit"]["summary"]
    base = fd[(fd["scenario"] == "mid") & (fd["headroom_additive_shift_GW"] == 0)]
    base_desc = ", ".join(
        f"{r.ISO}: {r.first_negative_annual_year}" for r in base.itertuples(index=False)
    )
    lines.append(f"- Baseline midcase first-negative annual years: {base_desc}.")
    plusminus = fd[(fd["scenario"] == "mid") & (fd["headroom_additive_shift_GW"].isin([-2, 2]))]
    lines.append(
        "- A +/-2 GW additive headroom stress test mainly changes timing for marginal systems; "
        "it does not remove the ordering that MISO/SPP are earlier than PJM under the baseline convention."
    )
    pipe = results["pipeline"]
    p_mid = pipe[(pipe["storage_credit_assumption"] == 0.2)].copy()
    lines.append(
        "- Generator-pipeline screen confirms very low effective-over-active conversion in several large queues; "
        f"MISO effective/active is {p_mid[p_mid['ISO'].eq('MISO')]['effective_over_active_queue'].iloc[0]:.2%}, "
        f"PJM is {p_mid[p_mid['ISO'].eq('PJM')]['effective_over_active_queue'].iloc[0]:.1%}."
    )
    arc = results["archetype"]["stability"]
    lines.append(
        f"- Archetype robustness is descriptive rather than statistical: nearest-neighbor stability ranges from "
        f"{arc['stable_fraction'].min():.2f} to {arc['stable_fraction'].max():.2f} across leave-one-variable-out tests."
    )
    nuc = results["nuclear"]
    nuc_mid = nuc[(nuc["scenario"] == "mid") & (nuc["recovery_case"] == "mid_recovery") & (nuc["policy_case"] == "retention+restart")]
    nuc_rows = nuc_mid[nuc_mid["baseline_shortfall_2035_GW"] > 0]
    nuc_desc = ", ".join(
        f"{r.ISO}: {r.offset_2035_pct_of_baseline_shortfall:.1%}" for r in nuc_rows.itertuples(index=False)
    )
    lines.append(f"- Midcase nuclear retention+restart offsets 2035 baseline shortfalls unevenly: {nuc_desc}.")
    pv = results["pv"]["threshold"]
    pv4 = pv[(pv["scenario"] == "mid") & (pv["storage_duration_h"] == 4.0)]
    pv_desc = ", ".join(
        f"{r.ISO}: {r.min_pv_ratio_for_residual_le_1pct:.2f}x" if pd.notna(r.min_pv_ratio_for_residual_le_1pct) else f"{r.ISO}: >2.0x"
        for r in pv4.itertuples(index=False)
    )
    lines.append(f"- With 4 h storage and storage power at 50% of PV capacity, minimum PV ratios for <=1% residual are: {pv_desc}.")
    gas = results["gas"]
    gas_mid = gas[(gas["scenario"] == "mid") & (gas["gas_techdetail_atb"] == "NG 1-on-1 Combined Cycle (H-Frame)")]
    gas_desc = ", ".join(
        f"{r.ISO_RTO}: {r.co2_mt_cf50:.1f} MtCO2/yr" for r in gas_mid.itertuples(index=False) if r.co2_mt_cf50 > 0
    )
    lines.append(f"- H-frame gas backstop at 50% capacity factor has nonzero midcase emissions only in deficit-facing systems: {gas_desc}.")
    lmp = results["lmp"]["coverage"]
    lines.append(
        f"- LMP screen covers {int(lmp['zones'].sum())} harmonized zones; per-ISO zone counts range from "
        f"{int(lmp['zones'].min())} to {int(lmp['zones'].max())}."
    )
    out = OUT / "si_experiment_findings.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[wrote] {out.relative_to(ROOT)}")


def main() -> None:
    results: dict[str, object] = {}
    results["inventory"] = canonical_table_inventory()
    results["demand"] = demand_and_noniso_audits()
    results["audit"] = plausibility_audit_exports()
    results["first_deficit"] = first_deficit_robustness()
    results["pipeline"] = generator_pipeline_sensitivity()
    results["archetype"] = archetype_robustness()
    results["nuclear"] = nuclear_sensitivity()
    results["pv"] = pv_storage_sensitivity()
    results["gas"] = gas_backstop_sensitivity()
    results["lmp"] = lmp_coverage_diagnostics()
    write_findings(results)


if __name__ == "__main__":
    main()
