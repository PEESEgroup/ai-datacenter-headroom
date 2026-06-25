from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables" / "si_round2"
CANON = ROOT / "tables" / "fig2_5_canonical_20260514"
SI16 = ROOT / "tables" / "si_experiments_20260516"
SI17 = ROOT / "tables" / "si_experiments_20260517_robustness"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
EXPOSED_ISOS = ["PJM", "MISO", "SPP"]
SCENARIO_COLS = {
    "low": "low_growth_margin_GW",
    "mid": "mid_growth_margin_GW",
    "high": "high_growth_margin_GW",
}
CASE_RENAME = {
    "main_inventory": "main_inventory",
    "reported_or_committed_only": "reported_committed",
    "confidence_weighted": "confidence_weighted",
    "high_confidence_only": "tierA_only",
    "inferred_delay_2yr": "delayed_tierC",
}
CASE_LABELS = {
    "main_inventory": "main inventory",
    "reported_committed": "reported/committed",
    "confidence_weighted": "confidence-weighted",
    "tierA_only": "Tier A only",
    "delayed_tierC": "delayed Tier C",
}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    ensure_out()
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(df):,} rows)")
    return path


def fmt(x: object, digits: int = 1) -> str:
    if x is None:
        return ""
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(val):
        return ""
    if abs(val) < 0.0005:
        val = 0.0
    return f"{val:.{digits}f}"


def fmt_pct(x: object, digits: int = 1) -> str:
    return fmt(x, digits) + r"\%"


def tex_escape(value: object) -> str:
    s = "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else str(value)
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
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def write_tex(name: str, body: str) -> Path:
    ensure_out()
    path = OUT / name
    path.write_text(body, encoding="utf-8")
    print(f"[wrote] {path.relative_to(ROOT)}")
    return path


def sign_label(x: float) -> str:
    return "negative" if x < 0 else "nonnegative"


def first_negative_year(df: pd.DataFrame, value_col: str) -> str:
    ordered = df.sort_values("year")
    neg = ordered[ordered[value_col] < 0]
    if neg.empty:
        return ">2035"
    return str(int(neg.iloc[0]["year"]))


def sign_sensitive_cells() -> pd.DataFrame:
    heat = pd.read_csv(CANON / "fig2" / "fig2d_margin_clock_heatmap_canonical_source.csv")
    traj = pd.read_csv(CANON / "fig2" / "fig2e_margin_trajectory_ribbons_canonical_source.csv")
    rows: list[dict[str, object]] = []
    for _, row in heat[heat["sign_sensitive_to_headroom_extrapolation"].astype(bool)].iterrows():
        iso, scenario, year = row["ISO"], row["scenario"], int(row["year"])
        main = float(row["margin_GW"])
        t = traj[(traj["ISO"].eq(iso)) & (traj["year"].eq(year))]
        if t.empty:
            continue
        col = SCENARIO_COLS[str(scenario)]
        base = float(t.iloc[0][col])
        sensitivity_candidates = [
            base + float(t.iloc[0]["headroom_extrapolation_low_GW"]),
            base + float(t.iloc[0]["headroom_extrapolation_high_GW"]),
        ]
        bound = sensitivity_candidates[0]
        for candidate in sensitivity_candidates:
            if sign_label(candidate) != sign_label(main):
                bound = candidate
                break
        rows.append(
            {
                "scenario": scenario,
                "iso_region": iso,
                "year": year,
                "main_cap_at_zero_margin_GW": main,
                "sensitivity_bound_margin_GW": bound,
                "headroom_delta_applied_GW": bound - base,
                "sign_main": sign_label(main),
                "sign_sensitivity_bound": sign_label(bound),
                "sign_sensitive_flag": True,
                "warning_margin_flag": bool(row["warning_0_to_2GW"]),
                "interpretation_note": "The headroom-extrapolation envelope changes the annual sign; retained as a robustness flag only. This is not the raw-linear baseline margin reported in the cap-at-zero/raw-linear 2035 audit.",
                "source_table": "tables/fig2_5_canonical_20260514/fig2/fig2d_margin_clock_heatmap_canonical_source.csv; tables/fig2_5_canonical_20260514/fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv",
            }
        )
    out = pd.DataFrame(rows).sort_values(["scenario", "iso_region", "year"])
    write_csv(out, "sign_sensitive_cells_fig2.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.scenario)} & {tex_escape(r.iso_region)} & {int(r.year)} & {fmt(r.main_cap_at_zero_margin_GW,2)} & {fmt(r.sensitivity_bound_margin_GW,2)} & {tex_escape(r.sign_main)} $\\rightarrow$ {tex_escape(r.sign_sensitivity_bound)} & {tex_escape('yes' if r.warning_margin_flag else 'no')} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "sign_sensitive_cells_fig2_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Sign-sensitive ISO-year cells in the Fig. 2 headroom-extrapolation envelope.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrll}}
\toprule
Scenario & Grid region & Year & Main margin & Sensitivity-bound margin & Sign change & 0--2 GW \\
 &  &  & (GW) & (GW) &  & warning \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{Sign-sensitive cells are annual ISO/RTO-scenario observations whose margin sign changes somewhere within the headroom-extrapolation envelope used for Fig. 2e. The sensitivity-bound margin is the main margin plus the relevant upper or lower envelope delta, not the raw-linear baseline margin reported in the cap-at-zero/raw-linear 2035 audit. These cells are retained as robustness flags and are not treated as deterministic reliability events.}}
\end{{table}}
\endgroup
""",
    )
    return out


def cap_zero_vs_raw_linear() -> pd.DataFrame:
    audit = pd.read_csv(CANON / "fig2" / "fig2_shared_canonical_gap_yearly.csv")
    sub = audit[audit["year"].eq(2035)].copy()
    rows: list[dict[str, object]] = []
    for iso in ISO_ORDER:
        r = sub[sub["ISO_RTO"].eq(iso)].iloc[0]
        cap_hc = float(r["HC_cap0_MW"]) / 1000
        raw_hc = float(r["HC_raw_linear_MW"]) / 1000
        ai_growth = float(r["D_ai_mid_MW"]) / 1000
        cap_margin = cap_hc - ai_growth
        raw_margin = raw_hc - ai_growth
        if cap_margin < 0:
            flag = "main exposed"
        elif raw_margin < 0 <= cap_margin:
            flag = "raw-linear sensitivity only"
        elif abs(cap_margin) <= 0.5:
            flag = "near-boundary"
        else:
            flag = "main positive"
        rows.append(
            {
                "iso_region": iso,
                "cap_at_zero_HC_2035_GW": cap_hc,
                "cap_at_zero_margin_2035_GW": cap_margin,
                "raw_linear_HC_2035_GW": raw_hc,
                "raw_linear_margin_2035_GW": raw_margin,
                "difference_raw_minus_cap_GW": raw_hc - cap_hc,
                "main_deficit_GW": max(0.0, -cap_margin),
                "raw_linear_deficit_GW": max(0.0, -raw_margin),
                "interpretation_flag": flag,
                "source_table": "tables/fig2_5_canonical_20260514/fig2/fig2_shared_canonical_gap_yearly.csv",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "cap_zero_vs_raw_linear_2035.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {fmt(r.cap_at_zero_HC_2035_GW,1)} & {fmt(r.cap_at_zero_margin_2035_GW,1)} & {fmt(r.raw_linear_HC_2035_GW,1)} & {fmt(r.raw_linear_margin_2035_GW,1)} & {tex_escape(r.interpretation_flag)} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "cap_zero_vs_raw_linear_2035_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Cap-at-zero and raw-linear 2035 margins by grid region.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrL{{0.23\linewidth}}}}
\toprule
Grid region & Cap-at-zero HC & Cap-at-zero margin & Raw-linear HC & Raw-linear margin & Interpretation \\
 & (GW) & (GW) & (GW) & (GW) &  \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{This side-by-side audit separates the main cap-at-zero AI-attributable convention from the raw-linear extrapolation sensitivity. Raw-linear negative slack is retained to diagnose sensitivity, but it is not mixed into the main AI-attributable shortfall calculation.}}
\end{{table}}
\endgroup
""",
    )
    return out


def queue_effective_assumptions() -> pd.DataFrame:
    pipe = pd.read_csv(CANON / "fig2" / "fig2c_generator_pipeline_canonical_source.csv")
    rows = []
    for _, r in pipe.iterrows():
        add_nameplate = float(r.get("add_nameplate_gen_MW", np.nan))
        add_eff = float(r.get("add_eff_MW", np.nan))
        rows.append(
            {
                "iso_region": r["ISO"],
                "queue_region_source": r.get("ISO_RTO", r["ISO"]),
                "screen_window_years": "2026-2028",
                "active_status_filter": "active or operational queue records",
                "ia_stage_filter": "interconnection-agreement-stage records retained as IA-stage subtotal",
                "operational_or_active_inclusion_rule": "active/operational records in the 2026-2028 screen",
                "withdrawn_exclusion_rule": "withdrawn, canceled or otherwise non-active records excluded",
                "technology": "non-storage effective additions; storage shown separately as 20% credit sensitivity",
                "nameplate_MW": float(r["MW_sum_total_active_2026_2028"]),
                "included_nameplate_MW": add_nameplate,
                "effective_capacity_factor": add_eff / add_nameplate if add_nameplate > 0 else np.nan,
                "effective_addition_MW": add_eff,
                "storage_base_case_treatment": "standalone storage excluded from base case",
                "storage_credit_sensitivity": float(r.get("storage_20pct_credit_delta_MW", np.nan)),
                "effective_over_active_percent": 100 * float(r.get("effective_over_active_queue", np.nan)),
                "top10_share_percent": 100 * float(r.get("top10_share_eff", np.nan)),
                "source_table": "tables/fig2_5_canonical_20260514/fig2/fig2c_generator_pipeline_canonical_source.csv",
                "script_path": "scripts/build_fig2_5_canonical_data.py",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "queue_effective_assumptions.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {fmt(r.nameplate_MW/1000,1)} & {fmt(r.included_nameplate_MW/1000,1)} & {fmt(r.effective_addition_MW/1000,2)} & {fmt_pct(r.effective_over_active_percent,1)} & {fmt(r.storage_credit_sensitivity/1000,1)} & {fmt_pct(r.top10_share_percent,0)} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "queue_effective_assumptions_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Generator-pipeline queue-to-effective-addition conversion summary.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
Grid region & Active queue & Included non-storage & Effective additions & Effective/active & Storage credit & Top-10 share \\
 & (GW) & nameplate (GW) & (GW) & (\%) & sensitivity (GW) & (\%) \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The screen is a near-term planning audit rather than a completion forecast. Gross active queue capacity is reduced by status, timing, generation-only and resource-type screens before being translated into effective additions; standalone storage is excluded in the base case and reported separately through the 20\% storage-credit sensitivity. The machine-readable CSV retains the screen window, active/operational status filter, interconnection-agreement-stage subtotal rule, withdrawn-project exclusion rule and storage treatment used for each grid region.}}
\end{{table}}
\endgroup
""",
    )
    return out


def queue_screen_rule_summary() -> pd.DataFrame:
    rows = [
        {
            "screen_component": "Screen window",
            "printed_rule_summary": "Near-term online-year window for additions entering the 2026--2028 generator-pipeline screen.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: screen_window_years; queue_eff_audit_iso_year_breakdown.csv: year",
            "author_review_flag": False,
        },
        {
            "screen_component": "Queue status inclusion",
            "printed_rule_summary": "Active or operational queue records are included in the gross active-queue denominator used for the screen.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: active_status_filter, operational_or_active_inclusion_rule",
            "author_review_flag": False,
        },
        {
            "screen_component": "Withdrawn/inactive exclusion",
            "printed_rule_summary": "Withdrawn, canceled or otherwise non-active records are excluded from the screened effective-addition calculation.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: withdrawn_exclusion_rule",
            "author_review_flag": False,
        },
        {
            "screen_component": "Interconnection-agreement subtotal",
            "printed_rule_summary": "Interconnection-agreement-stage records are retained as a subtotal used to distinguish later-stage projects from the gross active queue.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: ia_stage_filter; fig2c_generator_pipeline_canonical_source.csv: MW_sum_total_IAstage_2026_2028",
            "author_review_flag": False,
        },
        {
            "screen_component": "Generation-only screen",
            "printed_rule_summary": "Base-case effective additions count non-storage generation additions and decompose them into thermal, wind and solar components.",
            "repository_or_source_table_field": "fig2c_generator_pipeline_canonical_source.csv: add_nameplate_gen_MW, add_eff_MW, thermal_eff_MW, wind_eff_MW, solar_eff_MW",
            "author_review_flag": False,
        },
        {
            "screen_component": "Standalone storage base case",
            "printed_rule_summary": "Standalone storage is excluded from base-case effective additions.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: storage_base_case_treatment",
            "author_review_flag": False,
        },
        {
            "screen_component": "Storage-credit sensitivity",
            "printed_rule_summary": "Storage is shown separately as a 20% capacity-credit screening sensitivity and is not included in base-case margins.",
            "repository_or_source_table_field": "queue_effective_assumptions.csv: storage_credit_sensitivity; fig2c_generator_pipeline_canonical_source.csv: storage_20pct_credit_delta_MW",
            "author_review_flag": False,
        },
        {
            "screen_component": "Thermal effective factor",
            "printed_rule_summary": "Thermal effective capacity is retained as a resource-specific source-table field; factors are computed from effective capacity and nameplate by grid region and year rather than as one universal constant.",
            "repository_or_source_table_field": "queue_eff_audit_iso_year_breakdown.csv: thermal_nameplate, thermal_eff",
            "author_review_flag": True,
        },
        {
            "screen_component": "Wind effective factor",
            "printed_rule_summary": "Wind effective capacity is retained as a resource-specific source-table field; factors are computed from effective capacity and nameplate by grid region and year rather than as one universal constant.",
            "repository_or_source_table_field": "queue_eff_audit_iso_year_breakdown.csv: wind_nameplate, wind_eff",
            "author_review_flag": True,
        },
        {
            "screen_component": "Solar effective factor",
            "printed_rule_summary": "Solar effective capacity is retained as a resource-specific source-table field; factors are computed from effective capacity and nameplate by grid region and year rather than as one universal constant.",
            "repository_or_source_table_field": "queue_eff_audit_iso_year_breakdown.csv: solar_nameplate, solar_eff",
            "author_review_flag": True,
        },
    ]
    out = pd.DataFrame(rows)
    write_csv(out, "queue_screen_rule_summary.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.screen_component)} & {tex_escape(r.printed_rule_summary)} & {tex_escape(r.repository_or_source_table_field)} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "queue_screen_rule_summary_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Queue-screen rules used to construct near-term effective additions.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{L{{0.18\linewidth}}L{{0.43\linewidth}}L{{0.33\linewidth}}}}
\toprule
Screen component & Printed rule summary & Source-table field \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The printed conversion summary reports aggregate reductions from active queue capacity to screened effective additions. This rule summary records the status, timing, storage and resource-type screens used to produce those reductions. Project-level records and region/resource-specific factor fields are retained in the machine-readable queue-screen assumptions and source tables. The machine-readable tables retain the corresponding region-, resource- and year-specific factor fields used to construct the compact printed summary.}}
\end{{table}}
\endgroup
""",
    )
    return out


def project_tier_rules() -> pd.DataFrame:
    rules = [
        {
            "tier": "A",
            "tier_label": "A: committed, reported MW",
            "record_condition": "capacity-usable, committed status and reported MW",
            "mw_source": "reported MW",
            "mw_inference_rule": "direct reported MW retained; effective MW applies status realization factor",
            "status_requirement": "committed",
            "included_in_main_capacity_weighted_allocation": True,
            "included_in_reported_committed_case": True,
            "included_in_confidence_weighted_case": True,
            "included_in_tierA_only_case": True,
            "delayed_inferred_case_treatment": "no delay",
            "excluded_or_audit_only": False,
            "notes": "highest-evidence project-capacity tier",
            "script_path": "scripts/build_confidence_and_baseline_robustness_20260517.py",
        },
        {
            "tier": "B",
            "tier_label": "B: reported MW or committed inferred",
            "record_condition": "reported MW with non-committed status, or committed status with size-rank inferred MW",
            "mw_source": "reported MW or inferred MW",
            "mw_inference_rule": "reported MW retained where available; otherwise size-rank inferred MW for committed projects",
            "status_requirement": "reported MW or committed",
            "included_in_main_capacity_weighted_allocation": True,
            "included_in_reported_committed_case": True,
            "included_in_confidence_weighted_case": True,
            "included_in_tierA_only_case": False,
            "delayed_inferred_case_treatment": "no delay",
            "excluded_or_audit_only": False,
            "notes": "included in reported/committed and confidence-weighted screens",
            "script_path": "scripts/build_confidence_and_baseline_robustness_20260517.py",
        },
        {
            "tier": "C",
            "tier_label": "C: inferred size-rank MW",
            "record_condition": "capacity-usable record without reported MW and without committed status",
            "mw_source": "size-rank inferred MW",
            "mw_inference_rule": "MW inferred from project size-rank class and existing project-size distribution",
            "status_requirement": "not committed; capacity-inferable",
            "included_in_main_capacity_weighted_allocation": True,
            "included_in_reported_committed_case": False,
            "included_in_confidence_weighted_case": True,
            "included_in_tierA_only_case": False,
            "delayed_inferred_case_treatment": "online year shifted two years later in delayed Tier-C case",
            "excluded_or_audit_only": False,
            "notes": "used to test dependence on lower-confidence project scale and geography",
            "script_path": "scripts/build_confidence_and_baseline_robustness_20260517.py",
        },
        {
            "tier": "D",
            "tier_label": "D: unsized / excluded",
            "record_condition": "insufficient usable capacity information",
            "mw_source": "none",
            "mw_inference_rule": "no capacity assigned",
            "status_requirement": "not applicable",
            "included_in_main_capacity_weighted_allocation": False,
            "included_in_reported_committed_case": False,
            "included_in_confidence_weighted_case": False,
            "included_in_tierA_only_case": False,
            "delayed_inferred_case_treatment": "audit only",
            "excluded_or_audit_only": True,
            "notes": "retained in inventory audit but excluded from capacity-weighted allocation",
            "script_path": "scripts/build_confidence_and_baseline_robustness_20260517.py",
        },
    ]
    out = pd.DataFrame(rules)
    write_csv(out, "project_tier_inference_rules.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.tier)} & {tex_escape(r.record_condition)} & {tex_escape(r.mw_source)} & {tex_escape(r.mw_inference_rule)} & {tex_escape('yes' if r.included_in_main_capacity_weighted_allocation else 'no')} & {tex_escape(r.delayed_inferred_case_treatment)} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "project_tier_inference_rules_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Project-confidence tiers and MW inference rules.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lL{{0.23\linewidth}}L{{0.13\linewidth}}L{{0.28\linewidth}}lL{{0.18\linewidth}}}}
\toprule
Tier & Record condition & MW source & MW inference rule & Main allocation & Delayed-case treatment \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The confidence tiers are used to test dependence on project-level geography and capacity evidence; they do not replace the national load-growth scenario. Tier-specific screens are applied only to incremental 2025--2035 AI growth so that already-hosted 2025 load is not counted twice. Reported MW is used when available; otherwise the size-rank-inferred MW field from the project-inventory preprocessing is used only for records with enough information to enter the capacity-weighted allocation.}}
\end{{table}}
\endgroup
""",
    )
    return out


def topn_concentration_by_project_confidence() -> pd.DataFrame:
    county = pd.read_csv(CANON / "shared" / "canonical_county_ai_demand_yearly.csv", dtype={"GEOID": str})
    base = county[(county["scenario"].eq("mid")) & (county["year"].eq(2025))][
        ["GEOID", "iso", "GW_canonical"]
    ].rename(columns={"GW_canonical": "base_GW"})
    target = county[(county["scenario"].eq("mid")) & (county["year"].eq(2035))][
        ["GEOID", "iso", "GW_canonical"]
    ].rename(columns={"GW_canonical": "main_2035_GW"})
    data = base.merge(target, on=["GEOID", "iso"], how="outer").fillna(0.0)
    data["incremental_GW"] = (data["main_2035_GW"] - data["base_GW"]).clip(lower=0.0)

    project_total = pd.read_csv(SI17 / "si_project_pipeline_effective_capacity_by_confidence_case.csv")
    main = project_total[project_total["case"].eq("main_inventory")].rename(
        columns={"weighted_effective_MW": "main_effective_MW"}
    )
    rows = []
    for raw_case, out_case in CASE_RENAME.items():
        case = project_total[project_total["case"].eq(raw_case)].rename(
            columns={"weighted_effective_MW": "case_effective_MW"}
        )
        mult = (
            main[["year", "ISO", "main_effective_MW"]]
            .merge(case[["year", "ISO", "case_effective_MW"]], on=["year", "ISO"], how="left")
            .query("year == 2035")
            .copy()
        )
        mult["case_effective_MW"] = mult["case_effective_MW"].fillna(0.0)
        mult["multiplier"] = np.where(
            mult["main_effective_MW"] > 0,
            mult["case_effective_MW"] / mult["main_effective_MW"],
            1.0,
        )
        m = dict(zip(mult["ISO"], mult["multiplier"]))
        tmp = data.copy()
        tmp["multiplier"] = tmp["iso"].map(m).fillna(1.0)
        tmp["GW_case"] = tmp["base_GW"] + tmp["incremental_GW"] * tmp["multiplier"]
        g = tmp.sort_values("GW_case", ascending=False)
        total = float(g["GW_case"].sum())
        seven = float(g[g["iso"].isin(ISO_ORDER)]["GW_case"].sum())
        rows.append(
            {
                "filter_case": out_case,
                "year": 2035,
                "scenario": "mid",
                "top10_share_percent": 100 * float(g.head(10)["GW_case"].sum()) / total,
                "top20_share_percent": 100 * float(g.head(20)["GW_case"].sum()) / total,
                "top50_share_percent": 100 * float(g.head(50)["GW_case"].sum()) / total,
                "top100_share_percent": 100 * float(g.head(100)["GW_case"].sum()) / total,
                "positive_counties": int((g["GW_case"] > 0).sum()),
                "counties_above_0p1GW": int((g["GW_case"] > 0.1).sum()),
                "counties_above_0p5GW": int((g["GW_case"] > 0.5).sum()),
                "counties_above_1GW": int((g["GW_case"] > 1.0).sum()),
                "total_full_county_load_GW": total,
                "seven_region_load_GW": seven,
                "outside_region_load_GW": total - seven,
                "source_table": "tables/fig2_5_canonical_20260514/shared/canonical_county_ai_demand_yearly.csv; tables/si_experiments_20260517_robustness/si_project_pipeline_effective_capacity_by_confidence_case.csv",
                "script_path": "scripts/build_si_round2_audit_tables.py",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "topN_concentration_by_project_confidence.csv")
    rows_tex = "\n".join(
        f"{tex_escape(CASE_LABELS[r.filter_case])} & {fmt_pct(r.top10_share_percent,1)} & {fmt_pct(r.top20_share_percent,1)} & {fmt_pct(r.top50_share_percent,1)} & {fmt_pct(r.top100_share_percent,1)} & {int(r.positive_counties)} & {int(r.counties_above_1GW)} & {fmt(r.total_full_county_load_GW,1)} \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "topN_concentration_by_project_confidence_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{County concentration audit under grid-region project-confidence multipliers.}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Filter case & Top 10 & Top 20 & Top 50 & Top 100 & Positive counties & $>$1 GW counties & Total load \\
 & (\%) & (\%) & (\%) & (\%) &  &  & (GW) \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{This audit tests whether the spatial concentration result is driven by lower-confidence announced projects. It is separate from the seven-region exposure-share calculation and uses the full county allocation. Project-confidence filters are applied to county-level incremental load through grid-region multipliers derived from the same margin robustness pipeline, rather than by constructing an independent county-level national scenario or reassigning counties across grid regions.}}
\end{{table}}
\endgroup
""",
    )
    return out


def peak_coincidence_gamma() -> pd.DataFrame:
    county = pd.read_csv(CANON / "shared" / "canonical_county_ai_demand_yearly.csv")
    sub = county[county["iso"].isin(ISO_ORDER)].copy()
    grouped = (
        sub.groupby(["iso", "scenario"], as_index=False)
        .agg(
            gamma_min=("gamma", "min"),
            gamma_max=("gamma", "max"),
            years=("year", lambda x: f"{int(x.min())}-{int(x.max())}"),
        )
        .sort_values(["iso", "scenario"])
    )
    rows = []
    for iso in ISO_ORDER:
        g = grouped[grouped["iso"].eq(iso)]
        if g.empty:
            continue
        gamma_min, gamma_max = float(g["gamma_min"].min()), float(g["gamma_max"].max())
        rows.append(
            {
                "iso_region": iso,
                "scenario": "low/mid/high",
                "year_or_period": "2025-2035",
                "gamma_factor": gamma_min if np.isclose(gamma_min, gamma_max) else f"{gamma_min:.4f}-{gamma_max:.4f}",
                "peak_window_definition": "average three-hour ISO/RTO peak-window adjustment where available",
                "data_source": "canonical county-to-ISO allocation and peak-conversion table",
                "uses_three_hour_peak_window": True,
                "outside_region_treatment": "counties outside the seven analyzed grid regions remain in the national spatial inventory with no common ISO peak-window adjustment",
                "notes": "central factor used in the margin loop",
                "source_table": "tables/fig2_5_canonical_20260514/shared/canonical_county_ai_demand_yearly.csv",
                "script_path": "scripts/build_si_round2_audit_tables.py",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "peak_coincidence_gamma_factors.csv")
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {tex_escape(r.scenario)} & {tex_escape(r.year_or_period)} & {tex_escape(fmt(r.gamma_factor,3) if isinstance(r.gamma_factor, float) else r.gamma_factor)} & yes \\\\"
        for r in out.itertuples()
    )
    write_tex(
        "peak_coincidence_gamma_factors_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Regional peak-coincidence factors used to convert average AI load to peak-load-equivalent demand.}}
\begin{{tabular}}{{lllll}}
\toprule
Grid region & Scenario & Period & $\Gamma$ & Three-hour peak window \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}
\tabnote{{The peak-coincidence factor converts allocated average AI demand into the peak-load-equivalent quantity used in the annual margin loop. Counties outside the seven analyzed grid regions remain in the national spatial inventory but do not receive a common ISO peak-window adjustment.}}
\end{{table}}
\endgroup
""",
    )
    return out


def nuclear_unit_ledger() -> pd.DataFrame:
    units = pd.read_csv(CANON / "fig4" / "fig4_unit_level_nuclear_candidates_canonical.csv")
    rows = []
    for _, r in units.iterrows():
        pathway = str(r["pathway"])
        rows.append(
            {
                "unit_name": f"{r['plant_name']} {r['unit']}",
                "plant_name": r["plant_name"],
                "state": r["state"],
                "iso_region": r["ISO"],
                "pathway": pathway,
                "operating_status": "operating retention candidate" if pathway == "retention" else "restart or planned-opportunity candidate",
                "license_or_retirement_status_evidence": r.get("inclusion_basis", ""),
                "owner_or_developer_signal": "encoded in candidate screening source; not separately printed in compact ledger",
                "assumed_availability_year": int(r["available_year"]),
                "net_summer_capacity_MW": float(r["raw_capacity_GW"]) * 1000,
                "recoverability_factor": float(r["recoverability_factor"]),
                "credited_capacity_mid_MW": float(r["effective_mid_GW"]) * 1000,
                "credited_capacity_low_MW": float(r["effective_low_GW"]) * 1000,
                "credited_capacity_high_MW": float(r["effective_high_GW"]) * 1000,
                "included_in_retention_case": pathway == "retention",
                "included_in_restart_case": pathway == "restart",
                "included_in_planned_case": pathway == "planned",
                "screening_reason": r.get("inclusion_basis", ""),
                "exclusion_reason_if_any": "" if float(r["effective_mid_GW"]) > 0 else "zero recoverability under the current regional recovery screen",
                "source_reference_or_note": "EIA generator records and nuclear candidate screen",
                "source_table": "tables/fig2_5_canonical_20260514/fig4/fig4_unit_level_nuclear_candidates_canonical.csv",
                "script_path": "scripts/build_fig2_5_canonical_data.py",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "nuclear_unit_recovery_ledger.csv")
    display = out.sort_values(["iso_region", "pathway", "assumed_availability_year", "unit_name"]).copy()
    rows_tex = "\n".join(
        f"{tex_escape(r.unit_name)} & {tex_escape(r.iso_region)} & {tex_escape(r.pathway)} & {int(r.assumed_availability_year)} & {fmt(r.net_summer_capacity_MW/1000,2)} & {fmt(r.recoverability_factor,2)} & {fmt(r.credited_capacity_mid_MW/1000,2)} & {tex_escape(r.exclusion_reason_if_any)} \\\\"
        for r in display.itertuples()
    )
    write_tex(
        "nuclear_unit_recovery_ledger_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Unit-level nuclear recovery ledger.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{L{{0.20\linewidth}}llrrrrL{{0.22\linewidth}}}}
\toprule
Unit & Grid region & Pathway & Availability & Net capacity & Recoverability & Mid credited & Exclusion note \\
 &  &  & year & (GW) & factor & capacity (GW) &  \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The ledger makes the nuclear recovery screen traceable from candidate status to credited capacity. Retention, restart and planned opportunities are separated because they have different timing, siting and development-risk implications. The machine-readable CSV retains the license or retirement-status evidence field, owner/developer-signal placeholder, inclusion flags and exclusion note used for each unit; the compact printed table suppresses those fields for readability.}}
\end{{table}}
\endgroup
""",
    )
    return out


def nuclear_first_deficit_timing() -> pd.DataFrame:
    pol = pd.read_csv(CANON / "fig4" / "fig4_policy_margin_cases_canonical.csv")
    pol = pol[pol["recovery_case"].eq("mid_recovery")].copy()
    rows = []
    for (iso, scenario), g in pol.groupby(["ISO", "scenario"]):
        base = g[g["policy_case"].eq("baseline")].copy()
        base_first = first_negative_year(base, "margin_GW")
        base_2035 = float(base[base["year"].eq(2035)]["shortfall_GW"].iloc[0])
        for policy in ["baseline", "retention", "retention+restart"]:
            sub = g[g["policy_case"].eq(policy)].copy()
            first = first_negative_year(sub, "margin_GW")
            resid_2035 = float(sub[sub["year"].eq(2035)]["shortfall_GW"].iloc[0])
            avoided_2035 = base_2035 - resid_2035
            avoided_gwyr = float(
                (base.sort_values("year")["shortfall_GW"].to_numpy() - sub.sort_values("year")["shortfall_GW"].to_numpy()).clip(min=0).sum()
            )
            if base_first == first:
                interp = "severity reduction without delayed first-negative year" if avoided_2035 > 0 else "no timing change"
                delay = 0
            elif first == ">2035":
                interp = "avoided annual negative margin through 2035"
                delay = np.nan
            else:
                interp = "delayed binding year"
                delay = int(first) - int(base_first) if base_first != ">2035" else np.nan
            rows.append(
                {
                    "iso_region": iso,
                    "scenario": scenario,
                    "recovery_case": "mid_recovery",
                    "policy_case": policy,
                    "baseline_first_negative_year": base_first,
                    "first_negative_year_with_nuclear": first,
                    "delay_years": delay,
                    "baseline_2035_deficit_GW": base_2035,
                    "residual_2035_deficit_GW": resid_2035,
                    "avoided_2035_deficit_GW": avoided_2035,
                    "avoided_deficit_years_GWyr": avoided_gwyr,
                    "interpretation": interp,
                    "source_table": "tables/fig2_5_canonical_20260514/fig4/fig4_policy_margin_cases_canonical.csv",
                    "script_path": "scripts/build_si_round2_audit_tables.py",
                }
            )
    out = pd.DataFrame(rows)
    write_csv(out, "nuclear_first_deficit_timing.csv")
    show = out[out["iso_region"].isin(EXPOSED_ISOS)].copy()
    rows_tex = "\n".join(
        f"{tex_escape(r.iso_region)} & {tex_escape(r.scenario)} & {tex_escape(r.policy_case)} & {tex_escape(r.baseline_first_negative_year)} & {tex_escape(r.first_negative_year_with_nuclear)} & {fmt(r.baseline_2035_deficit_GW,1)} & {fmt(r.residual_2035_deficit_GW,1)} & {fmt(r.avoided_deficit_years_GWyr,1)} & {tex_escape(r.interpretation)} \\\\"
        for r in show.itertuples()
    )
    write_tex(
        "nuclear_first_deficit_timing_table.tex",
        rf"""\begingroup
\color{{blue}}
\begin{{table}}[H]
\centering
\caption{{Nuclear recovery and first-negative-margin timing.}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllllrrrL{{0.23\linewidth}}}}
\toprule
Grid region & Scenario & Nuclear case & Baseline first negative & With nuclear first negative & Baseline 2035 deficit & Residual 2035 deficit & Avoided GW-yr & Interpretation \\
 &  &  & year & year & (GW) & (GW) &  &  \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}%
}}
\tabnote{{The timing audit uses the same mid-recovery credited-capacity convention as the main nuclear screen. Baseline rows are retained as reference anchors. A case can reduce 2035 severity without delaying the first annual negative margin when the nuclear capacity arrives after the regional bottleneck clock has already crossed.}}
\end{{table}}
\endgroup
""",
    )
    return out


def main() -> None:
    outputs = {
        "SI-1": sign_sensitive_cells(),
        "SI-2": cap_zero_vs_raw_linear(),
        "SI-3A": queue_screen_rule_summary(),
        "SI-3": queue_effective_assumptions(),
        "SI-4": project_tier_rules(),
        "SI-5": topn_concentration_by_project_confidence(),
        "SI-7": peak_coincidence_gamma(),
        "SI-8": nuclear_unit_ledger(),
        "SI-9": nuclear_first_deficit_timing(),
    }
    report = OUT / "si_round2_generated_tables_manifest.md"
    manifest_rows = [
        (
            "SI-1",
            "sign_sensitive_cells_fig2.csv",
            "sign_sensitive_cells_fig2_table.tex",
            "tables/fig2_5_canonical_20260514/fig2/fig2d_margin_clock_heatmap_canonical_source.csv; tables/fig2_5_canonical_20260514/fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv",
            "scripts/build_si_round2_audit_tables.py::sign_sensitive_cells",
            "Headroom-extrapolation sign-sensitive cells; not raw-linear baseline margins.",
        ),
        (
            "SI-2",
            "cap_zero_vs_raw_linear_2035.csv",
            "cap_zero_vs_raw_linear_2035_table.tex",
            "tables/fig2_5_canonical_20260514/fig2/fig2_shared_canonical_gap_yearly.csv",
            "scripts/build_si_round2_audit_tables.py::cap_zero_vs_raw_linear",
            "Cap-at-zero versus raw-linear 2035 margin audit.",
        ),
        (
            "SI-3A",
            "queue_screen_rule_summary.csv",
            "queue_screen_rule_summary_table.tex",
            "tables/si_round2/queue_effective_assumptions.csv; tables/fig2_5_canonical_20260514/fig2/fig2c_generator_pipeline_canonical_source.csv; out_std/ANALYSIS/queue_eff_audit_iso_year_breakdown.csv",
            "scripts/build_si_round2_audit_tables.py::queue_screen_rule_summary",
            "Compact printed queue-screen rule summary; resource-specific factors remain source-table traced.",
        ),
        (
            "SI-3",
            "queue_effective_assumptions.csv",
            "queue_effective_assumptions_table.tex",
            "tables/fig2_5_canonical_20260514/fig2/fig2c_generator_pipeline_canonical_source.csv",
            "scripts/build_si_round2_audit_tables.py::queue_effective_assumptions",
            "Queue-to-effective conversion summary with machine-readable screen assumptions.",
        ),
        (
            "SI-4",
            "project_tier_inference_rules.csv",
            "project_tier_inference_rules_table.tex",
            "scripts/build_confidence_and_baseline_robustness_20260517.py",
            "scripts/build_si_round2_audit_tables.py::project_tier_rules",
            "Project-confidence tier definitions and inclusion logic.",
        ),
        (
            "SI-5",
            "topN_concentration_by_project_confidence.csv",
            "topN_concentration_by_project_confidence_table.tex",
            "tables/fig2_5_canonical_20260514/shared/canonical_county_ai_demand_yearly.csv; tables/si_experiments_20260517_robustness/si_project_pipeline_effective_capacity_by_confidence_case.csv",
            "scripts/build_si_round2_audit_tables.py::topn_concentration_by_project_confidence",
            "County concentration audit under grid-region project-confidence multipliers.",
        ),
        (
            "SI-7",
            "peak_coincidence_gamma_factors.csv",
            "peak_coincidence_gamma_factors_table.tex",
            "tables/fig2_5_canonical_20260514/shared/canonical_county_ai_demand_yearly.csv",
            "scripts/build_si_round2_audit_tables.py::peak_coincidence_gamma",
            "Peak-coincidence factors used to convert average load to peak-load-equivalent demand.",
        ),
        (
            "SI-8",
            "nuclear_unit_recovery_ledger.csv",
            "nuclear_unit_recovery_ledger_table.tex",
            "tables/fig2_5_canonical_20260514/fig4/fig4_unit_level_nuclear_candidates_canonical.csv",
            "scripts/build_si_round2_audit_tables.py::nuclear_unit_ledger",
            "Unit-level nuclear recovery ledger.",
        ),
        (
            "SI-9",
            "nuclear_first_deficit_timing.csv",
            "nuclear_first_deficit_timing_table.tex",
            "tables/fig2_5_canonical_20260514/fig4/fig4_policy_margin_cases_canonical.csv",
            "scripts/build_si_round2_audit_tables.py::nuclear_first_deficit_timing",
            "Nuclear recovery timing and severity audit.",
        ),
    ]
    lines = [
        "# SI round-2 generated audit tables",
        "",
        "Generated from existing canonical and supporting source tables.",
        "",
        "| Task | CSV | LaTeX fragment | Primary source table(s) | Builder | Scope note | Rows |",
        "|---|---|---|---|---|---|---:|",
    ]
    for task, csv_name, tex_name, source, builder, note in manifest_rows:
        df = outputs[task]
        lines.append(
            f"| {task} | `{csv_name}` | `{tex_name}` | `{source}` | `{builder}` | {note} | {len(df):,} |"
        )
    lines.extend(
        [
            "",
            "Unresolved: SI-6 high-density county deduplication audit was not generated because the private source archive contains county allocation weights but not raw pre/post-dedup record-count fields needed for a defensible table.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
