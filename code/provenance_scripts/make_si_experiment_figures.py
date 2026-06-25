from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tables" / "si_experiments_20260516"
OUT = ROOT / "figures" / "SI_experiments_20260516"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
RISK_ORDER = ["MISO", "SPP", "PJM", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
ISO_COLORS = {
    "PJM": "#D8328A",
    "MISO": "#7B74B8",
    "SPP": "#C59600",
    "ERCOT": "#E46C0A",
    "CAISO": "#1B9E77",
    "NYISO": "#4E79A7",
    "ISO-NE": "#76A5C9",
    "Non-ISO": "#9A9A9A",
}
SCENARIO_COLORS = {"low": "#4E79A7", "mid": "#D9902F", "high": "#B83A34"}
GRID = "#E9E4DC"
TEXT = "#333333"
MUTED = "#777777"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[wrote] figures/SI_experiments_20260516/{stem}.pdf")


def clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


def scenario_sort_key(s: str) -> int:
    return {"low": 0, "mid": 1, "high": 2}.get(str(s), 99)


def year_to_float(v: object) -> float:
    if pd.isna(v):
        return 2036.2
    if str(v) == ">2035":
        return 2036.2
    return float(v)


def draw_scope_audit() -> None:
    df = pd.read_csv(DATA / "si_noniso_demand_audit.csv")
    mid = df[df["scenario"].eq("mid")].sort_values("year")
    low = df[df["scenario"].eq("low")].sort_values("year")
    high = df[df["scenario"].eq("high")].sort_values("year")

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    x = mid["year"].to_numpy()
    iso7 = mid["ISO7_coincident_GW"].to_numpy()
    noniso = mid["nonISO_coincident_GW"].to_numpy()
    ax.fill_between(x, 0, iso7, color="#4B9A77", alpha=0.78, lw=0, label="Seven analyzed grid regions")
    ax.fill_between(x, iso7, iso7 + noniso, color="#B8B8B8", alpha=0.68, lw=0, label="Outside analyzed regions")
    ax.plot(low["year"], low["total_coincident_GW"], color="#9E9E9E", lw=1.3, ls="--", label="Low total")
    ax.plot(high["year"], high["total_coincident_GW"], color="#555555", lw=1.3, ls=":", label="High total")
    ax.plot(x, mid["total_coincident_GW"], color="#222222", lw=1.5, label="Mid total")
    ax2 = ax.twinx()
    ax2.plot(x, mid["nonISO_share_of_total"] * 100, color="#8E5B2C", marker="o", ms=3.4, lw=1.4)
    ax2.set_ylabel("Outside-region share (%)", color="#8E5B2C")
    ax2.tick_params(axis="y", colors="#8E5B2C")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#8E5B2C")
    ax2.set_ylim(20, 40)
    ax.set_xlim(2025, 2035)
    ax.set_ylim(0, max(high["total_coincident_GW"]) * 1.08)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coincident AI demand (GW)")
    ax.set_title("Full county inventory versus seven-region subset", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False, columnspacing=1.0, handlelength=1.4)
    save(fig, "si_exp01_scope_audit")


def draw_county_concentration() -> None:
    df = pd.read_csv(DATA / "si_county_concentration_by_scenario_year.csv")
    mid = df[df["scenario"].eq("mid")].sort_values("year")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for col, label, color in [
        ("top10_share", "Top 10 counties", "#B75C2B"),
        ("top20_share", "Top 20 counties", "#D9902F"),
        ("top50_share", "Top 50 counties", "#2F7F7F"),
        ("top100_share", "Top 100 counties", "#6D8FC2"),
    ]:
        ax.plot(mid["year"], mid[col] * 100, marker="o", ms=3.5, lw=1.7, color=color, label=label)
    ax.set_ylim(30, 100)
    ax.set_xlim(2025, 2035)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative share of county demand (%)")
    ax.set_title("County concentration remains high as the inventory expands", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    ax.legend(loc="lower right", frameon=False)
    ax.annotate(
        "335 positive counties\n19 counties >1 GW",
        xy=(2035, float(mid[mid["year"].eq(2035)]["top50_share"].iloc[0]) * 100),
        xytext=(2031.5, 62),
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8),
        fontsize=8,
        color=MUTED,
    )
    save(fig, "si_exp02_county_concentration")


def draw_state_concentration() -> None:
    df = pd.read_csv(DATA / "si_state_concentration_mid2035.csv").sort_values("rank_2035").head(20)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    y = np.arange(len(df))[::-1]
    colors = [ISO_COLORS.get(x, "#999999") for x in df["dominant_iso"]]
    ax.barh(y, df["gw_2035"], color=colors, alpha=0.86, height=0.62)
    ax.errorbar(
        df["gw_2035"],
        y,
        xerr=[df["gw_2035"] - df["gw_2035_low"], df["gw_2035_high"] - df["gw_2035"]],
        fmt="none",
        ecolor="#555555",
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["state_abb"])
    ax.set_xlabel("Mid-2035 AI demand (GW)")
    ax.set_title("State concentration and scenario range", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    for yi, val, share in zip(y[:8], df["gw_2035"].head(8), df["share_of_national_mid2035"].head(8)):
        ax.text(val + 0.25, yi, f"{val:.1f}", va="center", fontsize=7.5, color=TEXT)
    legend_items = [Patch(facecolor=ISO_COLORS[k], label=k) for k in ["PJM", "ERCOT", "SPP", "MISO", "CAISO", "Non-ISO"] if k in set(df["dominant_iso"])]
    ax.legend(handles=legend_items, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, columnspacing=1.0)
    save(fig, "si_exp03_state_concentration")


def draw_first_deficit_sensitivity() -> None:
    df = pd.read_csv(DATA / "si_first_deficit_key_sensitivity_summary.csv")
    df = df[df["headroom_additive_shift_GW"].isin([-2, 0, 2])].copy()
    scenario_offsets = {"low": -0.22, "mid": 0.0, "high": 0.22}
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    ybase = {iso: i for i, iso in enumerate(RISK_ORDER[::-1])}
    for iso in RISK_ORDER:
        for scen in ["low", "mid", "high"]:
            sub = df[(df["ISO"].eq(iso)) & (df["scenario"].eq(scen))].copy()
            if sub.empty:
                continue
            xs = sub["first_negative_annual_year"].map(year_to_float).to_numpy()
            y = ybase[iso] + scenario_offsets[scen]
            ax.plot([xs.min(), xs.max()], [y, y], color=SCENARIO_COLORS[scen], lw=1.8, alpha=0.65)
            baseline = sub[sub["headroom_additive_shift_GW"].eq(0)]["first_negative_annual_year"].map(year_to_float).iloc[0]
            ax.scatter(baseline, y, s=32, color=SCENARIO_COLORS[scen], edgecolor="white", lw=0.6, zorder=4)
    ax.axvspan(2035.55, 2036.55, color="#F2F2F2", zorder=0)
    ax.set_yticks([ybase[i] for i in RISK_ORDER[::-1]])
    ax.set_yticklabels(RISK_ORDER[::-1])
    ax.set_xlim(2026.4, 2036.55)
    ax.set_xticks([2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036.2])
    ax.set_xticklabels(["2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035", ">2035"], rotation=0)
    ax.set_xlabel("First annual negative margin")
    ax.set_title("Deficit-clock sensitivity to AI growth and +/-2 GW headroom", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    ax.legend(
        handles=[Line2D([0], [0], color=SCENARIO_COLORS[s], marker="o", lw=1.8, label=s.capitalize()) for s in ["low", "mid", "high"]],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.5,
    )
    save(fig, "si_exp04_first_deficit_sensitivity")


def draw_generator_pipeline_conversion() -> None:
    df = pd.read_csv(DATA / "si_generator_pipeline_storage_credit_sensitivity.csv")
    df = df[df["storage_credit_assumption"].eq(0.2)].drop_duplicates("ISO").copy()
    df["order"] = df["ISO"].map({iso: i for i, iso in enumerate(RISK_ORDER)})
    df = df.sort_values("order")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(len(df))[::-1]
    for yi, r in zip(y, df.itertuples(index=False)):
        ax.plot([r.effective_additions_GW, r.active_queue_GW], [yi, yi], color="#CFC7BA", lw=1.2, zorder=1)
    ax.scatter(df["active_queue_GW"], y, s=52, color="#C8DAC8", edgecolor="white", lw=0.6, label="Active queue", zorder=3)
    ax.scatter(df["IA_stage_queue_GW"], y, s=48, color="#E7B649", marker="D", edgecolor="white", lw=0.6, label="IA-stage", zorder=3)
    ax.scatter(df["effective_additions_GW"], y, s=64, color="#239B88", edgecolor="white", lw=0.6, label="Effective additions", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 400)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ISO"])
    ax.set_xlabel("Capacity in 2026--2028 screen (GW, log scale)")
    ax.set_title("Queue-to-effective-addition screen", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    for yi, r in zip(y, df.itertuples(index=False)):
        pct = r.effective_over_active_queue * 100 if pd.notna(r.effective_over_active_queue) else np.nan
        label = "<0.1%" if pct < 0.05 and pct >= 0 else f"{pct:.1f}%"
        ax.text(330, yi, label, ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), columnspacing=1.0)
    save(fig, "si_exp05_generator_pipeline_conversion")


def draw_archetype_fingerprint() -> None:
    mat = pd.read_csv(DATA / "si_archetype_standardized_fingerprint_matrix.csv")
    stab = pd.read_csv(DATA / "si_archetype_stability_summary.csv")
    variables = ["headroom_gw", "ai_gw", "nonai_gw", "new_supply_gw", "margin_gw"]
    labels = ["2025\nheadroom", "AI\ngrowth", "non-AI\ngrowth", "net supply\nchange", "2035\nmargin"]
    mat["order"] = mat["ISO"].map({iso: i for i, iso in enumerate(RISK_ORDER)})
    mat = mat.sort_values("order")
    arr = mat[variables].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("fingerprint", ["#B83A34", "#F7F2EC", "#159A77"])
    fig = plt.figure(figsize=(5.5, 3.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1.25], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.2))
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat["ISO"])
    ax.set_xticks(np.arange(len(variables)))
    ax.set_xticklabels(labels)
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=7.2, color="#222222")
    ax.set_title("Mechanism profiles", loc="left", fontweight="bold", pad=6)
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    stab = stab.set_index("ISO").loc[mat["ISO"]].reset_index()
    ax2.barh(np.arange(len(stab)), stab["stable_fraction"], color="#8A8A8A", height=0.55)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("LOO\nstability")
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.tick_params(axis="y", left=False, labelleft=False)
    clean_axis(ax2)
    ax.text(
        0.0,
        1.02,
        "cell values are standardized z-scores",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    save(fig, "si_exp06_archetype_fingerprint")


def draw_nuclear_policy_sensitivity() -> None:
    df = pd.read_csv(DATA / "si_nuclear_policy_case_sensitivity.csv")
    df = df[
        df["scenario"].eq("mid")
        & df["policy_case"].isin(["baseline", "retention", "retention+restart"])
        & df["ISO"].isin(["PJM", "MISO", "SPP"])
    ].copy()
    rec_order = ["low_recovery", "mid_recovery", "high_recovery"]
    policy_order = ["baseline", "retention", "retention+restart"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xbase = {p: i for i, p in enumerate(policy_order)}
    offsets = {"low_recovery": -0.18, "mid_recovery": 0.0, "high_recovery": 0.18}
    markers = {"PJM": "o", "MISO": "s", "SPP": "^"}
    for iso in ["PJM", "MISO", "SPP"]:
        for rec in rec_order:
            sub = df[(df["ISO"].eq(iso)) & (df["recovery_case"].eq(rec))]
            xs = [xbase[p] + offsets[rec] for p in policy_order]
            ys = [float(sub[sub["policy_case"].eq(p)]["offset_2035_pct_of_baseline_shortfall"].iloc[0]) * 100 for p in policy_order]
            ax.plot(xs, ys, color=ISO_COLORS[iso], lw=1.0, alpha=0.35)
            ax.scatter(xs, ys, s=44, color=ISO_COLORS[iso], edgecolor="white", lw=0.7, marker=markers[iso], zorder=3)
    ax.set_xticks(range(len(policy_order)))
    ax.set_xticklabels(["Baseline", "Retention", "Retention\n+ restart"])
    ax.set_ylabel("2035 baseline shortfall offset (%)")
    ax.set_title("Nuclear recovery value depends on pathway and region", loc="left", fontweight="bold", pad=6)
    ax.set_ylim(-3, 70)
    clean_axis(ax)
    legend_iso = [Line2D([0], [0], color=ISO_COLORS[i], marker=markers[i], lw=0, label=i) for i in ["PJM", "MISO", "SPP"]]
    ax.legend(handles=legend_iso, frameon=False, loc="upper left", ncol=3)
    ax.text(0.98, 0.96, "points = low/mid/high\nrecovery sensitivity", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    save(fig, "si_exp07_nuclear_policy_sensitivity")


def draw_pv_storage_threshold() -> None:
    df = pd.read_csv(DATA / "si_pv_storage_min_pv_ratio_by_duration.csv")
    keep = ["PJM", "MISO", "SPP", "ERCOT"]
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    for iso in keep:
        sub_all = df[df["ISO"].eq(iso)]
        low = sub_all[sub_all["scenario"].eq("low")].sort_values("storage_duration_h")
        mid = sub_all[sub_all["scenario"].eq("mid")].sort_values("storage_duration_h")
        high = sub_all[sub_all["scenario"].eq("high")].sort_values("storage_duration_h")
        if mid.empty:
            continue
        x = mid["storage_duration_h"].to_numpy()
        y_mid = mid["min_pv_ratio_for_residual_le_1pct"].to_numpy(float)
        y_low = low["min_pv_ratio_for_residual_le_1pct"].to_numpy(float) if len(low) == len(mid) else y_mid
        y_high = high["min_pv_ratio_for_residual_le_1pct"].to_numpy(float) if len(high) == len(mid) else y_mid
        y_mid = np.nan_to_num(y_mid, nan=2.15)
        y_low = np.nan_to_num(y_low, nan=2.15)
        y_high = np.nan_to_num(y_high, nan=2.15)
        ax.fill_between(x, np.minimum(y_low, y_high), np.maximum(y_low, y_high), color=ISO_COLORS[iso], alpha=0.10, lw=0)
        ax.plot(x, y_mid, color=ISO_COLORS[iso], lw=1.8, label=iso)
    ax.axhline(1, color="#888888", lw=0.8, ls="--")
    ax.text(12.05, 1.02, "PV = AI peak", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.05, 2.2)
    ax.set_xlabel("Storage duration (h)")
    ax.set_ylabel("Minimum PV nameplate / AI peak")
    ax.set_title("PV-plus-storage threshold for <=1% residual firm gap", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=4, loc="upper right", columnspacing=0.9, handlelength=1.6)
    save(fig, "si_exp08_pv_storage_threshold")


def draw_gas_backstop_sensitivity() -> None:
    df = pd.read_csv(DATA / "si_gas_backstop_technology_capacity_factor_sensitivity.csv")
    df = df[df["gas_techdetail_atb"].eq("NG 1-on-1 Combined Cycle (H-Frame)")].copy()
    df = df[df["firm_onsite_capacity_required_GW"] > 0]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    markers = {"low": "o", "mid": "s", "high": "^"}
    for r in df.itertuples(index=False):
        color = ISO_COLORS.get(r.ISO_RTO, "#777777")
        ax.plot([r.cost_bn_min, r.cost_bn_max], [r.co2_mt_min, r.co2_mt_max], color=color, alpha=0.22, lw=5, solid_capstyle="round")
        ax.scatter(r.cost_bn_cf50, r.co2_mt_cf50, s=42 + 32 * r.firm_onsite_capacity_required_GW, color=color, marker=markers[r.scenario], edgecolor="white", lw=0.7, zorder=3)
    ax.set_xlabel("Annual variable cost (US$ billion/year)")
    ax.set_ylabel("Annual CO2 emissions (MtCO2/year)")
    ax.set_title("Gas-backstop burden is concentrated in deficit-facing systems", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    leg_iso = [Line2D([0], [0], color=ISO_COLORS[i], marker="o", lw=0, label=i) for i in ["PJM", "MISO", "SPP"]]
    leg_scen = [Line2D([0], [0], color="#555555", marker=markers[s], lw=0, label=s.capitalize()) for s in ["low", "mid", "high"]]
    ax.legend(handles=leg_iso + leg_scen, frameon=False, ncol=3, loc="upper left")
    ax.text(0.98, 0.05, "thick bands = capacity-factor range\nmarkers = 50% capacity factor", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUTED)
    save(fig, "si_exp09_gas_backstop_sensitivity")


def draw_lmp_price_screen() -> None:
    df = pd.read_csv(DATA / "si_lmp_zone_coverage_and_price_screen_summary.csv")
    df = df.set_index("ISO_RTO").loc[ISO_ORDER].reset_index()
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hlines(y, df["p95_lmp_minus_gas_median"], df["p95_lmp_minus_gas_p90"], color="#B8AFA3", lw=2.2, alpha=0.75)
    sizes = 28 + 5.5 * df["zones"]
    ax.scatter(df["p95_lmp_minus_gas_median"], y, s=sizes, color=[ISO_COLORS.get(i, "#777777") for i in df["ISO_RTO"]], edgecolor="white", lw=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ISO_RTO"])
    ax.set_xlabel("P95 LMP - gas variable cost ($/MWh)")
    ax.set_title("Zone-level price screen coverage and upper-tail spread", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    for yi, r in zip(y, df.itertuples(index=False)):
        ax.text(r.p95_lmp_minus_gas_p90 + 2, yi, f"n={int(r.zones)}", va="center", fontsize=7.5, color=MUTED)
    ax.text(0.98, 0.04, "point = median zone\nline = median to P90", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.3, color=MUTED)
    save(fig, "si_exp10_lmp_price_screen")


def draw_source_inventory() -> None:
    df = pd.read_csv(DATA / "si_canonical_source_table_inventory.csv")
    summary = df.groupby("source_group", as_index=False).agg(rows=("rows", "sum"), tables=("relative_path", "count"))
    summary = summary.sort_values("rows", ascending=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    y = np.arange(len(summary))
    ax.barh(y, summary["rows"], color="#7AA6A1", alpha=0.85, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["source_group"])
    ax.set_xlabel("Rows in canonical CSV source tables")
    ax.set_title("Canonical figure-source table inventory", loc="left", fontweight="bold", pad=6)
    clean_axis(ax)
    for yi, r in zip(y, summary.itertuples(index=False)):
        ax.text(r.rows * 1.02, yi, f"{int(r.tables)} tables", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, summary["rows"].max() * 1.25)
    save(fig, "si_exp11_source_table_inventory")


def main() -> None:
    setup_style()
    draw_scope_audit()
    draw_county_concentration()
    draw_state_concentration()
    draw_first_deficit_sensitivity()
    draw_generator_pipeline_conversion()
    draw_archetype_fingerprint()
    draw_nuclear_policy_sensitivity()
    draw_pv_storage_threshold()
    draw_gas_backstop_sensitivity()
    draw_lmp_price_screen()
    draw_source_inventory()


if __name__ == "__main__":
    main()
