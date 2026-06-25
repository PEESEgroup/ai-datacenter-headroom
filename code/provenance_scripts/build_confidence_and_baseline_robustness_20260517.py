from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# The public package stores compact supporting source tables. The full canonical tables
# live in the analysis clone used to generate the current figures.
LOCAL_CANON = ROOT / "tables" / "fig2_5_canonical_20260514"
SOURCE_CANON = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables" / "tables" / "fig2_5_canonical_20260514"
CANON = LOCAL_CANON if LOCAL_CANON.exists() else SOURCE_CANON

DATA_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = DATA_ROOT / "out_std/DC/fractracker_processed/fractracker_pipeline_clean_20260305_164431.csv"
COUNTY_TO_ISO = DATA_ROOT / "out_std/GEO/county_to_iso_area.csv"
COUNTIES_2023 = DATA_ROOT / "out_std/GEO/cb2023_county_5m/cb_2023_us_county_5m.shp"

OUT = ROOT / "tables" / "si_experiments_20260517_robustness"
FIGOUT = ROOT / "figures" / "SI_experiments_20260517_robustness"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
ALL_ISO_ORDER = ISO_ORDER + ["Non-ISO"]
SCENARIOS = ["low", "mid", "high"]
YEARS = list(range(2025, 2036))

CASE_LABELS = {
    "main_inventory": "main inventory",
    "reported_or_committed_only": "reported/committed",
    "confidence_weighted": "confidence-weighted",
    "high_confidence_only": "high-confidence only",
    "inferred_delay_2yr": "inferred delayed 2 yr",
}

CASE_WEIGHTS = {
    "main_inventory": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.0},
    "reported_or_committed_only": {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0},
    "confidence_weighted": {"A": 1.0, "B": 0.75, "C": 0.50, "D": 0.0},
    "high_confidence_only": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
    "inferred_delay_2yr": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.0},
}

TIER_LABELS = {
    "A": "A: committed, reported MW",
    "B": "B: reported MW or committed inferred",
    "C": "C: inferred size-rank MW",
    "D": "D: unsized / excluded",
}


def write(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[wrote] {path.relative_to(ROOT)} ({len(df):,} rows)")
    return path


def clean_county_name(value: object) -> str:
    s = str(value or "").lower().strip()
    for token in [
        " county",
        " parish",
        " borough",
        " census area",
        " municipality",
        " city and borough",
        " city",
        ".",
        "'",
    ]:
        s = s.replace(token, "")
    s = " ".join(s.replace("-", " ").split())
    return s


STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def first_negative_year(g: pd.DataFrame, value_col: str = "margin_GW") -> float:
    ordered = g.sort_values("year")
    neg = ordered[ordered[value_col] < 0]
    if neg.empty:
        return np.nan
    return float(neg.iloc[0]["year"])


def first_negative_interpolated(g: pd.DataFrame, value_col: str = "margin_GW") -> float:
    ordered = g.sort_values("year")
    years = ordered["year"].to_numpy(dtype=float)
    vals = ordered[value_col].to_numpy(dtype=float)
    if len(vals) == 0 or np.all(vals >= 0):
        return np.nan
    idx = np.where(vals < 0)[0][0]
    if idx == 0:
        return years[idx]
    y0, y1 = years[idx - 1], years[idx]
    v0, v1 = vals[idx - 1], vals[idx]
    if np.isclose(v0, v1):
        return y1
    return y0 + (0 - v0) * (y1 - y0) / (v1 - v0)


def fmt_year(value: float) -> str:
    if value is None or pd.isna(value):
        return ">2035"
    return f"{value:.1f}" if abs(value - round(value)) > 1e-6 else f"{int(round(value))}"


def load_pipeline_with_iso() -> pd.DataFrame:
    import geopandas as gpd

    pipe = pd.read_csv(PIPELINE)
    pipe = pipe.reset_index().rename(columns={"index": "project_record_id"})
    pipe["project_record_id"] = pipe["project_record_id"] + 1

    # Assign county by 2023 Census polygons from project coordinates.
    valid_xy = pipe["lat"].notna() & pipe["long"].notna()
    pts = gpd.GeoDataFrame(
        pipe.loc[valid_xy].copy(),
        geometry=gpd.points_from_xy(pipe.loc[valid_xy, "long"], pipe.loc[valid_xy, "lat"]),
        crs="EPSG:4326",
    )
    counties = gpd.read_file(COUNTIES_2023)[["GEOID", "STATEFP", "NAME", "geometry"]]
    counties = counties.to_crs(pts.crs)
    joined = gpd.sjoin(pts, counties, how="left", predicate="within").drop(columns=["geometry", "index_right"])
    joined = joined.rename(columns={"GEOID": "GEOID_spatial", "NAME": "county_2023_spatial"})
    pipe = pipe.merge(
        joined[["project_record_id", "GEOID_spatial", "county_2023_spatial"]],
        on="project_record_id",
        how="left",
    )

    # Fallback by reported state/county names for points near boundaries.
    county_lookup = counties.drop(columns="geometry").copy()
    county_lookup["state"] = county_lookup["STATEFP"].map({v: k for k, v in STATE_ABBR_TO_FIPS.items()})
    county_lookup["county_norm"] = county_lookup["NAME"].map(clean_county_name)
    pipe["county_norm"] = pipe["county"].map(clean_county_name)
    pipe["state"] = pipe["state"].astype(str).str.upper().str.strip()
    fallback = pipe.merge(
        county_lookup[["GEOID", "state", "county_norm", "NAME"]].rename(
            columns={"GEOID": "GEOID_name", "NAME": "county_2023_name"}
        ),
        on=["state", "county_norm"],
        how="left",
    )
    pipe["GEOID"] = pipe["GEOID_spatial"].fillna(fallback["GEOID_name"]).astype("string")
    pipe["county_2023"] = pipe["county_2023_spatial"].fillna(fallback["county_2023_name"])

    c2i = pd.read_csv(COUNTY_TO_ISO, dtype={"GEOID": str})
    pipe = pipe.merge(c2i[["GEOID", "ISO_ASSIGNED", "share"]], on="GEOID", how="left")
    pipe["ISO_ASSIGNED"] = pipe["ISO_ASSIGNED"].fillna("unmatched")

    has_eff = pipe["mw_eff_mid"].notna()
    explicit_mw = pipe["mw_num"].notna()
    committed = pipe["status_class"].eq("committed")
    tier = np.select(
        [
            has_eff & explicit_mw & committed,
            has_eff & ((explicit_mw & ~committed) | (~explicit_mw & committed)),
            has_eff & ~explicit_mw & ~committed,
            ~has_eff,
        ],
        ["A", "B", "C", "D"],
        default="D",
    )
    pipe["confidence_tier"] = tier
    pipe["confidence_tier_label"] = pipe["confidence_tier"].map(TIER_LABELS)
    pipe["capacity_basis"] = np.select(
        [explicit_mw, has_eff, ~has_eff],
        ["reported_MW", "size_rank_inferred", "unsized"],
        default="unsized",
    )
    pipe["online_year_adj_base"] = pipe["online_year"].round().astype(int)
    pipe["mw_eff_mid"] = pipe["mw_eff_mid"].fillna(0.0)
    return pipe


def cumulative_project_capacity(pipe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for case, weights in CASE_WEIGHTS.items():
        for _, row in pipe.iterrows():
            tier = row["confidence_tier"]
            weight = weights.get(tier, 0.0)
            if weight <= 0 or row["mw_eff_mid"] <= 0:
                continue
            online = int(row["online_year_adj_base"])
            if case == "inferred_delay_2yr" and tier == "C":
                online += 2
            for year in YEARS:
                if online <= year:
                    rows.append(
                        {
                            "case": case,
                            "case_label": CASE_LABELS[case],
                            "year": year,
                            "ISO": row["ISO_ASSIGNED"],
                            "confidence_tier": tier,
                            "weighted_effective_MW": row["mw_eff_mid"] * weight,
                        }
                    )
    by_tier = pd.DataFrame(rows)
    if by_tier.empty:
        by_tier = pd.DataFrame(columns=["case", "case_label", "year", "ISO", "confidence_tier", "weighted_effective_MW"])
    by_tier = (
        by_tier.groupby(["case", "case_label", "year", "ISO", "confidence_tier"], as_index=False)
        .agg(weighted_effective_MW=("weighted_effective_MW", "sum"))
    )
    total = (
        by_tier.groupby(["case", "case_label", "year", "ISO"], as_index=False)
        .agg(weighted_effective_MW=("weighted_effective_MW", "sum"))
    )
    return by_tier, total


def build_confidence_sensitivity(project_total: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = pd.read_csv(CANON / "shared/canonical_iso_ai_demand_yearly.csv")
    gap = pd.read_csv(CANON / "fig2/fig2_shared_canonical_gap_yearly.csv")
    hc = gap[["ISO", "year", "HC_MW", "HC_raw_linear_MW"]].copy()
    base = demand.merge(hc, on=["ISO", "year"], how="left")

    main = project_total[project_total["case"].eq("main_inventory")][["year", "ISO", "weighted_effective_MW"]]
    main = main.rename(columns={"weighted_effective_MW": "main_project_effective_MW"})

    rows = []
    for case, label in CASE_LABELS.items():
        case_tot = project_total[project_total["case"].eq(case)][["year", "ISO", "weighted_effective_MW"]]
        case_tot = case_tot.rename(columns={"weighted_effective_MW": "case_project_effective_MW"})
        tmp = base.merge(main, on=["year", "ISO"], how="left").merge(case_tot, on=["year", "ISO"], how="left")
        tmp["main_project_effective_MW"] = tmp["main_project_effective_MW"].fillna(0.0)
        tmp["case_project_effective_MW"] = tmp["case_project_effective_MW"].fillna(0.0)
        tmp["confidence_multiplier"] = np.where(
            tmp["main_project_effective_MW"] > 0,
            tmp["case_project_effective_MW"] / tmp["main_project_effective_MW"],
            1.0,
        )
        tmp.loc[tmp["year"] <= 2025, "confidence_multiplier"] = 1.0
        tmp["case"] = case
        tmp["case_label"] = label
        # The headroom loop is imposed on incremental AI load relative to the
        # 2025 midcase baseline. Using total AI load here would double-count the
        # already-hosted 2025 stock and would not reproduce the canonical Fig. 2
        # bridge.
        tmp["ai_growth_confidence_MW"] = tmp["ai_growth_vs_mid2025_MW"] * tmp["confidence_multiplier"]
        tmp["ai_total_confidence_MW"] = tmp["ai_baseline_mid2025_MW"] + tmp["ai_growth_confidence_MW"]
        tmp["margin_confidence_MW"] = tmp["HC_MW"] - tmp["ai_growth_confidence_MW"]
        tmp["margin_confidence_GW"] = tmp["margin_confidence_MW"] / 1000
        tmp["shortfall_confidence_GW"] = np.maximum(0, -tmp["margin_confidence_GW"])
        rows.append(tmp)
    sens = pd.concat(rows, ignore_index=True)
    sens = sens[
        [
            "case",
            "case_label",
            "scenario",
            "year",
            "ISO",
            "ai_demand_MW",
            "ai_growth_vs_mid2025_MW",
            "ai_growth_confidence_MW",
            "ai_total_confidence_MW",
            "confidence_multiplier",
            "main_project_effective_MW",
            "case_project_effective_MW",
            "HC_MW",
            "margin_confidence_MW",
            "margin_confidence_GW",
            "shortfall_confidence_GW",
        ]
    ]

    summary_rows = []
    for (case, label, scenario, iso), g in sens.groupby(["case", "case_label", "scenario", "ISO"]):
        g2035 = g[g["year"].eq(2035)].iloc[0]
        summary_rows.append(
            {
                "case": case,
                "case_label": label,
                "scenario": scenario,
                "ISO": iso,
                "first_negative_annual_year": first_negative_year(g, "margin_confidence_GW"),
                "first_negative_interpolated_year": first_negative_interpolated(g, "margin_confidence_GW"),
                "margin_2035_GW": g2035["margin_confidence_GW"],
                "shortfall_2035_GW": g2035["shortfall_confidence_GW"],
                "ai_growth_2035_GW": g2035["ai_growth_confidence_MW"] / 1000,
                "ai_total_2035_GW": g2035["ai_total_confidence_MW"] / 1000,
                "confidence_multiplier_2035": g2035["confidence_multiplier"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    return sens, summary


def no_ai_baseline_check() -> pd.DataFrame:
    gap = pd.read_csv(CANON / "fig2/fig2_shared_canonical_gap_yearly.csv")
    demand = pd.read_csv(CANON / "shared/canonical_iso_ai_demand_yearly.csv")
    mid2035 = demand[(demand["scenario"].eq("mid")) & (demand["year"].eq(2035))][
        ["ISO", "ai_demand_GW", "ai_growth_vs_mid2025_GW"]
    ]
    mid2035 = mid2035.rename(
        columns={
            "ai_demand_GW": "mid_total_ai_demand_2035_GW",
            "ai_growth_vs_mid2025_GW": "mid_incremental_ai_growth_2035_GW",
        }
    )
    rows = []
    for iso, g in gap.groupby("ISO"):
        g = g.sort_values("year")
        cap0_neg = g[g["HC_MW"] < 0]
        raw_neg = g[g["HC_raw_linear_MW"] < 0]
        y2035 = g[g["year"].eq(2035)].iloc[0]
        y2025 = g[g["year"].eq(2025)].iloc[0]
        rows.append(
            {
                "ISO": iso,
                "no_ai_first_negative_year_cap0": np.nan if cap0_neg.empty else int(cap0_neg.iloc[0]["year"]),
                "no_ai_first_negative_year_raw_linear": np.nan if raw_neg.empty else int(raw_neg.iloc[0]["year"]),
                "headroom_2025_cap0_GW": y2025["HC_MW"] / 1000,
                "headroom_2035_cap0_GW": y2035["HC_MW"] / 1000,
                "headroom_2035_raw_linear_GW": y2035["HC_raw_linear_MW"] / 1000,
                "baseline_nonAI_shortfall_raw_2035_GW": y2035["baseline_nonAI_shortfall_raw_MW"] / 1000,
                "cap0_interpretation": "no negative no-AI margin under main cap-at-zero convention",
            }
        )
    out = pd.DataFrame(rows).merge(mid2035, on="ISO", how="left")
    out["mid_with_ai_margin_2035_GW"] = (
        out["headroom_2035_cap0_GW"] - out["mid_incremental_ai_growth_2035_GW"]
    )
    out["mid_ai_attributable_shortfall_2035_GW"] = np.maximum(0, -out["mid_with_ai_margin_2035_GW"])
    out["raw_linear_audit_flag"] = np.where(
        out["headroom_2035_raw_linear_GW"] < 0,
        "raw-linear no-AI slack is negative; retained as sensitivity, not counted as AI-attributable",
        "raw-linear no-AI slack remains non-negative",
    )
    return out.sort_values("ISO", key=lambda s: s.map({iso: i for i, iso in enumerate(ISO_ORDER)}))


def build_key_findings(summary: pd.DataFrame, baseline: pd.DataFrame, tier_summary: pd.DataFrame) -> pd.DataFrame:
    mid = summary[summary["scenario"].eq("mid")].copy()
    rows = []
    for case in CASE_LABELS:
        sub = mid[mid["case"].eq(case)].set_index("ISO")
        def y(iso: str) -> str:
            return fmt_year(sub.loc[iso, "first_negative_interpolated_year"])
        def m(iso: str) -> float:
            return float(sub.loc[iso, "margin_2035_GW"])
        rows.append(
            {
                "check": f"midcase timing under {CASE_LABELS[case]}",
                "finding": f"MISO first negative {y('MISO')}, SPP {y('SPP')}, PJM {y('PJM')}; 2035 margins are MISO {m('MISO'):.1f} GW, SPP {m('SPP'):.1f} GW, PJM {m('PJM'):.1f} GW.",
                "interpretation": "Use this as a data-confidence sensitivity, not as a replacement for the national demand pathway.",
            }
        )
    b = baseline.set_index("ISO")
    raw_flags = ", ".join(b.index[b["headroom_2035_raw_linear_GW"] < 0].tolist()) or "none"
    rows.append(
        {
            "check": "no-AI baseline convention",
            "finding": f"Under the cap-at-zero convention, no analyzed ISO has negative no-AI headroom; raw-linear audit flags: {raw_flags}.",
            "interpretation": "Main-text shortfalls are AI-attributable relative to cap-at-zero headroom; raw-linear slack is a sensitivity field.",
        }
    )
    main_tier = tier_summary[(tier_summary["case"].eq("main_inventory")) & (tier_summary["year"].eq(2035))]
    total = main_tier["weighted_effective_MW"].sum()
    tier_c = main_tier[main_tier["confidence_tier"].eq("C")]["weighted_effective_MW"].sum()
    rows.append(
        {
            "check": "inferred project component",
            "finding": f"In 2035, size-rank inferred records account for {tier_c/1000:.1f} GW of {total/1000:.1f} GW effective project-pipeline inventory ({100*tier_c/total:.1f}%).",
            "interpretation": "This is why the SI reports project-data confidence sensitivity instead of treating all project records as equally certain.",
        }
    )
    return pd.DataFrame(rows)


def make_figure(tier_summary: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGOUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {
        "A": "#2a9d8f",
        "B": "#e9c46a",
        "C": "#f4a261",
        "D": "#b8b0a6",
    }
    case_colors = {
        "main_inventory": "#30343f",
        "reported_or_committed_only": "#3a86ff",
        "confidence_weighted": "#9d4edd",
        "high_confidence_only": "#e76f51",
        "inferred_delay_2yr": "#6a994e",
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.85),
        gridspec_kw={"width_ratios": [1.15, 1.05, 1.1], "wspace": 0.38},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.34, wspace=0.52)

    ax = axes[0]
    data = tier_summary[(tier_summary["case"].eq("main_inventory")) & (tier_summary["year"].eq(2035))].copy()
    piv = data.pivot_table(index="ISO", columns="confidence_tier", values="weighted_effective_MW", aggfunc="sum", fill_value=0) / 1000
    piv = piv.reindex(ALL_ISO_ORDER).fillna(0)
    left = np.zeros(len(piv))
    y = np.arange(len(piv))
    for tier in ["A", "B", "C", "D"]:
        vals = piv[tier].to_numpy() if tier in piv else np.zeros(len(piv))
        ax.barh(y, vals, left=left, color=colors[tier], height=0.58, label=TIER_LABELS[tier].split(": ", 1)[1])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.invert_yaxis()
    ax.set_xlabel("effective project inventory, 2035 (GW)")
    ax.grid(axis="x", color="#e8e2d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Project tiers", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.legend(frameon=False, fontsize=5.7, ncol=1, loc="lower right", handlelength=1.0, handletextpad=0.35)

    ax = axes[1]
    mid = summary[(summary["scenario"].eq("mid")) & (summary["ISO"].isin(["MISO", "SPP", "PJM", "ERCOT"]))].copy()
    ypos = {"MISO": 0, "SPP": 1, "PJM": 2, "ERCOT": 3}
    for case, g in mid.groupby("case"):
        x = g["first_negative_interpolated_year"].fillna(2036.0).to_numpy()
        yv = g["ISO"].map(ypos).to_numpy()
        ax.scatter(x, yv, s=22, color=case_colors[case], label=CASE_LABELS[case], edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(2035, color="#aaa", lw=0.6, ls="--")
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.set_xlim(2027, 2036.25)
    ax.set_xticks([2028, 2030, 2032, 2034, 2036])
    ax.set_xticklabels(["2028", "2030", "2032", "2034", ">2035"])
    ax.grid(axis="x", color="#e8e2d9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("first negative year")
    ax.set_title("Timing screen", loc="left", fontsize=8, fontweight="bold", pad=3)

    ax = axes[2]
    focus = summary[(summary["scenario"].eq("mid")) & (summary["ISO"].isin(["PJM", "MISO", "SPP", "ERCOT", "CAISO"]))].copy()
    iso_y = {iso: i for i, iso in enumerate(["PJM", "MISO", "SPP", "ERCOT", "CAISO"])}
    for case, g in focus.groupby("case"):
        offset = {
            "main_inventory": -0.18,
            "reported_or_committed_only": -0.09,
            "confidence_weighted": 0.0,
            "high_confidence_only": 0.09,
            "inferred_delay_2yr": 0.18,
        }[case]
        ax.scatter(
            g["margin_2035_GW"],
            g["ISO"].map(iso_y) + offset,
            s=18,
            color=case_colors[case],
            edgecolor="white",
            linewidth=0.35,
            label=CASE_LABELS[case],
            zorder=3,
        )
    ax.axvline(0, color="#777", lw=0.7)
    ax.set_yticks(list(iso_y.values()))
    ax.set_yticklabels(list(iso_y.keys()))
    ax.invert_yaxis()
    ax.set_xlabel("2035 margin (GW)")
    ax.set_title("2035 severity", loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.grid(axis="x", color="#e8e2d9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    case_seen = {}
    for h, lab in zip(handles, labels):
        case_seen.setdefault(lab, h)
    fig.legend(
        case_seen.values(),
        case_seen.keys(),
        frameon=False,
        fontsize=5.7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.64, 0.02),
        handletextpad=0.25,
        columnspacing=1.0,
    )

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.10, 1.14, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    out_pdf = FIGOUT / "si_project_confidence_and_noai_robustness.pdf"
    out_png = FIGOUT / "si_project_confidence_and_noai_robustness.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_pdf.relative_to(ROOT)}")
    print(f"[wrote] {out_png.relative_to(ROOT)}")


def main() -> None:
    if not CANON.exists():
        raise FileNotFoundError(f"Canonical table directory not found: {CANON}")
    pipe = load_pipeline_with_iso()
    inventory_cols = [
        "project_record_id",
        "state",
        "county",
        "GEOID",
        "county_2023",
        "ISO_ASSIGNED",
        "status",
        "status_class",
        "capacity_basis",
        "confidence_tier",
        "confidence_tier_label",
        "p_realize",
        "mw_num",
        "sizerank",
        "mw_mid",
        "mw_eff_mid",
        "online_year",
    ]
    write(pipe[inventory_cols], "si_project_confidence_tier_inventory.csv")

    audit = pd.DataFrame(
        [
            {
                "check": "project records",
                "value": len(pipe),
                "note": "all processed project-inventory records",
            },
            {
                "check": "sized project records",
                "value": int((pipe["mw_eff_mid"] > 0).sum()),
                "note": "records with usable effective MW for sensitivity",
            },
            {
                "check": "records assigned to 2023 counties",
                "value": int(pipe["GEOID"].notna().sum()),
                "note": "spatial join plus state/county-name fallback",
            },
            {
                "check": "records not assigned to county",
                "value": int(pipe["GEOID"].isna().sum()),
                "note": "retained in project inventory but excluded from ISO sensitivity if unmatched",
            },
            {
                "check": "records assigned outside seven analyzed ISO/RTOs",
                "value": int(pipe["ISO_ASSIGNED"].eq("Non-ISO").sum()),
                "note": "retained in national inventory; not forced into ISO/RTO margin loop",
            },
        ]
    )
    write(audit, "si_project_confidence_geography_audit.csv")

    by_tier, project_total = cumulative_project_capacity(pipe)
    write(by_tier, "si_project_pipeline_effective_capacity_by_confidence_tier.csv")
    write(project_total, "si_project_pipeline_effective_capacity_by_confidence_case.csv")

    sens, summary = build_confidence_sensitivity(project_total)
    write(sens, "si_project_confidence_iso_year_margin_sensitivity.csv")
    write(summary.sort_values(["scenario", "case", "ISO"]), "si_project_confidence_margin_summary.csv")

    baseline = no_ai_baseline_check()
    write(baseline, "si_no_ai_baseline_headroom_check.csv")

    key = build_key_findings(summary, baseline, by_tier)
    write(key, "si_project_confidence_key_findings.csv")
    (OUT / "si_project_confidence_key_findings.md").write_text(
        "\n".join(f"- **{row.check}:** {row.finding} {row.interpretation}" for row in key.itertuples(index=False)),
        encoding="utf-8",
    )
    print(f"[wrote] {(OUT / 'si_project_confidence_key_findings.md').relative_to(ROOT)}")

    make_figure(by_tier, summary)


if __name__ == "__main__":
    main()
