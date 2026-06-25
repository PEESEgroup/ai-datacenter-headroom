from __future__ import annotations

import os
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514" / "fig4"
OUT = TABLE_ARCHIVE / "figures" / "Fig4_previous_design_canonical_20260514"

YEARS = list(range(2025, 2036))
SCENARIOS = ["low", "mid", "high"]
DEFICIT_ISOS = ["PJM", "MISO", "SPP"]
RECOVERY_CASES = [("80%", 0.80), ("90%", 0.90), ("100%", 1.00)]
ISO_COLORS = {"PJM": "#d83b8c", "MISO": "#7a73b7", "SPP": "#bc8c00"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 700,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "font.size": 6.8,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.2,
        }
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(TABLE / "fig4_baseline_margin_canonical_input.csv")
    rec = pd.read_csv(TABLE / "fig4_recoverable_nuclear_capacity_source.csv")
    return cases, rec


def short_unit_name(row: pd.Series) -> str:
    name = str(row["plant_name"])
    replacements = {
        "Clinton Power Station": "Clinton",
        "Crane Clean Energy Center": "Crane",
        "Palisades": "Palisades",
        "Monticello Nuclear Facility": "Monticello",
        "Quad Cities Generating Station": "Quad Cities",
        "Point Beach Nuclear Plant": "Point Beach",
        "Perry": "Perry",
        "Arkansas Nuclear One": "Arkansas",
        "Dresden Generating Station": "Dresden",
        "Prairie Island": "Prairie",
        "Donald C Cook": "D.C. Cook",
        "Cooper Nuclear Station": "Cooper",
        "Calvert Cliffs Nuclear Power Plant": "Calvert",
    }
    base = replacements.get(name, name.split(" Nuclear")[0].split(" Generating")[0])
    return f"{base} {row['unit']}"


def recoverable_gw(rec: pd.DataFrame, iso: str, year: int) -> float:
    row = rec[(rec["ISO"].eq(iso)) & (rec["year"].eq(year))]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    return float(r["retention_recoverable_GW"] + r["restart_recoverable_GW"])


def baseline_margin(cases: pd.DataFrame, iso: str, year: int, scenario: str = "mid") -> float:
    row = cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(year)) & (cases["scenario"].eq(scenario))]
    if row.empty:
        return np.nan
    return float(row["baseline_margin_GW"].iloc[0])


def deficit_after(cases: pd.DataFrame, rec: pd.DataFrame, iso: str, year: int, ratio: float, scenario: str = "mid") -> float:
    margin = baseline_margin(cases, iso, year, scenario)
    return max(0.0, -(margin + ratio * recoverable_gw(rec, iso, year)))


def first_deficit_year(cases: pd.DataFrame, rec: pd.DataFrame, iso: str, ratio: float, scenario: str = "mid") -> float:
    last_margin = None
    last_year = None
    for year in YEARS:
        margin = baseline_margin(cases, iso, year, scenario) + ratio * recoverable_gw(rec, iso, year)
        if margin < 0:
            if last_margin is not None and last_margin >= 0:
                frac = last_margin / (last_margin - margin)
                return last_year + frac * (year - last_year)
            return float(year)
        last_margin = margin
        last_year = year
    return 2036.0


def draw_panel_d(cases: pd.DataFrame, rec: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.15, 1.72), facecolor="white")
    years = np.asarray(YEARS)
    rows = []

    baseline_mid = {}
    residual_mid = {}
    offset_mid = {}
    for iso in DEFICIT_ISOS:
        base_vals = []
        resid_vals = []
        offset_vals = []
        for year in YEARS:
            base = max(0.0, -baseline_margin(cases, iso, year, "mid"))
            resid = deficit_after(cases, rec, iso, year, 0.90, "mid")
            off = max(0.0, base - resid)
            base_vals.append(base)
            resid_vals.append(resid)
            offset_vals.append(off)
            rows.append(
                {
                    "ISO": iso,
                    "year": year,
                    "baseline_mid_GW": base,
                    "residual_90pct_mid_GW": resid,
                    "offset_90pct_mid_GW": off,
                }
            )
        baseline_mid[iso] = np.asarray(base_vals)
        residual_mid[iso] = np.asarray(resid_vals)
        offset_mid[iso] = np.asarray(offset_vals)

    baseline_low = []
    baseline_high = []
    residual_low = []
    residual_high = []
    for year in YEARS:
        base_totals = []
        resid_totals = []
        for scenario in SCENARIOS:
            base_total = 0.0
            for iso in DEFICIT_ISOS:
                base_total += max(0.0, -baseline_margin(cases, iso, year, scenario))
            base_totals.append(base_total)

            for _label, ratio in RECOVERY_CASES:
                resid_total = 0.0
                for iso in DEFICIT_ISOS:
                    resid_total += deficit_after(cases, rec, iso, year, ratio, scenario)
                resid_totals.append(resid_total)
        baseline_low.append(min(base_totals))
        baseline_high.append(max(base_totals))
        residual_low.append(min(resid_totals))
        residual_high.append(max(resid_totals))

    top = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = baseline_mid[iso]
        ax.fill_between(
            years,
            top,
            top + vals,
            color=ISO_COLORS[iso],
            alpha=0.56,
            linewidth=0.35,
            edgecolor="white",
            zorder=3,
        )
        top += vals

    bottom = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = residual_mid[iso]
        ax.fill_between(
            years,
            -bottom,
            -(bottom + vals),
            color=ISO_COLORS[iso],
            alpha=0.30,
            linewidth=0.35,
            edgecolor="white",
            zorder=3,
        )
        bottom += vals

    baseline_low = np.asarray(baseline_low)
    baseline_high = np.asarray(baseline_high)
    residual_low = np.asarray(residual_low)
    residual_high = np.asarray(residual_high)
    ax.fill_between(years, baseline_low, baseline_high, color="#8a8179", alpha=0.10, zorder=1)
    ax.fill_between(years, -residual_high, -residual_low, color="#168875", alpha=0.10, zorder=1)
    ax.plot(years, baseline_low, color="#7d736b", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, baseline_high, color="#7d736b", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, -residual_low, color="#168875", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, -residual_high, color="#168875", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)

    ax.axhline(0, color="#5b544d", lw=0.72, zorder=5)
    ax.text(2025.18, baseline_high.max() * 0.88, "baseline deficit", ha="left", va="center", fontsize=4.8, color="#6a443f")
    ax.text(2025.18, -residual_high.max() * 0.82, "residual with nuclear", ha="left", va="center", fontsize=4.8, color="#168875")

    top2035 = 0.0
    bot2035 = 0.0
    for iso in DEFICIT_ISOS:
        top2035 += baseline_mid[iso][-1]
        bot2035 += residual_mid[iso][-1]
    ax.text(2035.08, top2035, f"{top2035:.1f}", ha="left", va="center", fontsize=4.6, color="#4c413c", fontweight="bold")
    ax.text(2035.08, -bot2035, f"{bot2035:.1f}", ha="left", va="center", fontsize=4.6, color="#168875", fontweight="bold")

    y_limit = max(float(baseline_high.max()), float(residual_high.max()), 1.0) * 1.14
    ax.set_xlim(2025, 2035.55)
    ax.set_ylim(-y_limit, y_limit)
    tick_max = 20 if y_limit > 15 else 10
    ax.set_yticks([-tick_max, -tick_max / 2, 0, tick_max / 2, tick_max])
    ax.set_yticklabels([f"{tick_max:g}", f"{tick_max/2:g}", "0", f"{tick_max/2:g}", f"{tick_max:g}"])
    ax.set_xticks([2026, 2028, 2030, 2032, 2034])
    ax.set_xlabel("year", labelpad=1.0)
    ax.set_ylabel("GW", labelpad=1.0)
    ax.grid(axis="x", color="#eee5dc", lw=0.65, zorder=0)
    ax.grid(axis="y", color="#f4eee7", lw=0.35, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#5b544d")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(axis="both", labelsize=5.2, length=2.1)
    handles = [
        Line2D([0], [0], color=ISO_COLORS["PJM"], lw=4.0, alpha=0.58, label="PJM"),
        Line2D([0], [0], color=ISO_COLORS["MISO"], lw=4.0, alpha=0.58, label="MISO"),
        Line2D([0], [0], color=ISO_COLORS["SPP"], lw=4.0, alpha=0.58, label="SPP"),
        Line2D([0], [0], color="#7d736b", lw=0.85, linestyle=(0, (2, 1)), label="uncertainty envelope"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.22),
        ncol=4,
        handlelength=1.0,
        columnspacing=0.50,
        handletextpad=0.25,
        borderaxespad=0.0,
        fontsize=4.3,
    )

    pd.DataFrame(rows).to_csv(OUT / "fig4d_deficit_offset_mirror_stream_revised_source.csv", index=False)
    return fig


def draw_panel_e(cases: pd.DataFrame, rec: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.15, 1.55), facecolor="white")
    rows = []
    case_labels = [("none", 0.0), *RECOVERY_CASES]
    x_positions = np.arange(len(case_labels))
    y_positions = np.arange(len(DEFICIT_ISOS))[::-1]
    max_residual = max(deficit_after(cases, rec, iso, 2035, 0.0, "mid") for iso in DEFICIT_ISOS)

    for yi, iso in zip(y_positions, DEFICIT_ISOS):
        color = ISO_COLORS[iso]
        residuals = []
        years = []
        avoided = []
        for label, ratio in case_labels:
            residual_2035 = deficit_after(cases, rec, iso, 2035, ratio, "mid")
            first_year = first_deficit_year(cases, rec, iso, ratio, "mid")
            base_gw_yr = sum(deficit_after(cases, rec, iso, year, 0.0, "mid") for year in YEARS)
            rec_gw_yr = sum(deficit_after(cases, rec, iso, year, ratio, "mid") for year in YEARS)
            avoided_gw_yr = base_gw_yr - rec_gw_yr
            residuals.append(residual_2035)
            years.append(first_year)
            avoided.append(avoided_gw_yr)
            rows.append(
                {
                    "ISO": iso,
                    "recovery_credit": label,
                    "first_deficit_year": first_year,
                    "residual_2035_GW": residual_2035,
                    "credited_capacity_2035_GW": ratio * recoverable_gw(rec, iso, 2035),
                    "avoided_deficit_GWyr": avoided_gw_yr,
                }
            )

        ax.plot(x_positions, np.repeat(yi, len(x_positions)), color=color, linewidth=0.75, alpha=0.45, zorder=1)
        for xi, (label, _ratio), residual, first_year in zip(x_positions, case_labels, residuals, years):
            size = 26 + 86 * np.sqrt(max(residual, 0) / max_residual)
            if label == "none":
                face = "white"
                edge = color
                lw = 1.05
                alpha = 1.0
            else:
                face = color
                edge = "white"
                lw = 0.55
                alpha = {"80%": 0.58, "90%": 0.80, "100%": 0.96}[label]
            ax.scatter(xi, yi, s=size, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha, zorder=5)
            if label in {"none", "90%", "100%"}:
                txt_color = color if label == "none" else "white"
                ax.text(xi, yi, f"{residual:.1f}", ha="center", va="center", fontsize=3.95, color=txt_color, fontweight="bold", zorder=6)
            if label == "90%":
                ax.scatter(xi, yi, s=size + 30, facecolor="none", edgecolor=color, linewidth=0.65, alpha=0.80, zorder=4)
                year_txt = f"{int(first_year)}" if first_year < 2036 else ">2035"
                ax.text(xi, yi - 0.34, year_txt, ha="center", va="top", fontsize=3.9, color="#5a5149")

        ax.barh(yi + 0.28, avoided[2] / 10.0, left=3.45, height=0.08, color=color, alpha=0.72, zorder=2)
        ax.text(3.47 + avoided[2] / 10.0, yi + 0.28, f"{avoided[2]:.1f}", ha="left", va="center", fontsize=3.9, color=color)

    ax.text(3.45, y_positions[0] + 0.50, "avoided\nGW-yr", fontsize=3.9, color="#5a5149", ha="left", va="bottom", linespacing=0.82)
    ax.set_xlim(-0.45, 4.2)
    ax.set_ylim(-0.70, len(DEFICIT_ISOS) - 0.35)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ratio in case_labels])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(DEFICIT_ISOS, fontweight="bold")
    ax.set_xlabel("recoverable nuclear credit")
    ax.grid(axis="x", color="#eee5dc", linewidth=0.55)
    ax.grid(axis="y", color="#f3eee8", linewidth=0.50)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#6a625a")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(axis="both", labelsize=5.2)
    ax.tick_params(axis="y", length=0)
    ax.text(1.95, -0.54, "small labels = first deficit year at 90% credit", ha="center", va="center", fontsize=3.85, color="#5a5149")

    pd.DataFrame(rows).to_csv(OUT / "fig4e_timing_severity_recovery_bubbles_source.csv", index=False)
    return fig


def draw_panel_e_macc() -> plt.Figure:
    summary = pd.read_csv(TABLE / "fig4_unit_marginal_value_summary_canonical.csv")
    summary = summary.copy()
    summary["credit_mid_GW"] = summary["recoverable_GW"] * 0.90
    summary = summary[summary["credit_mid_GW"].gt(1e-6)].copy()
    summary["value_density_mid"] = summary["value_mid"] / summary["credit_mid_GW"]
    summary["value_density_low"] = summary["value_low"] / summary["credit_mid_GW"]
    summary["value_density_high"] = summary["value_high"] / summary["credit_mid_GW"]
    summary["is_positive"] = summary["value_mid"].gt(1e-6)
    positive = summary[summary["is_positive"]].sort_values(["value_density_mid", "value_mid"], ascending=False).copy()
    zero = summary[~summary["is_positive"]].copy()
    plot_df = pd.concat([positive, zero], ignore_index=True)
    plot_df["x0"] = plot_df["credit_mid_GW"].cumsum() - plot_df["credit_mid_GW"]
    plot_df["xmid"] = plot_df["x0"] + plot_df["credit_mid_GW"] / 2

    fig, ax = plt.subplots(figsize=(3.15, 1.62), facecolor="white")
    rows = []
    for idx, r in plot_df.iterrows():
        color = ISO_COLORS.get(str(r["ISO"]), "#b8b0aa")
        height = float(r["value_density_mid"])
        display_height = height if r["is_positive"] else 0.18
        width = float(r["credit_mid_GW"])
        x0 = float(r["x0"])
        pathway = str(r["pathway"])
        face = color if r["is_positive"] else "#d2cdc7"
        alpha = 0.78 if r["is_positive"] else 0.40
        hatch = "////" if pathway == "restart" else None
        ax.bar(
            x0,
            display_height,
            width=width,
            align="edge",
            color=face,
            alpha=alpha,
            edgecolor="white" if pathway == "retention" else color,
            linewidth=0.36 if pathway == "retention" else 0.45,
            hatch=hatch,
            zorder=3,
        )
        low = float(r["value_density_low"])
        high = float(r["value_density_high"])
        if r["is_positive"]:
            ax.vlines(float(r["xmid"]), low, high, color="#4f4944", lw=0.45, alpha=0.70, zorder=4)
        rows.append(
            {
                "plant_unit": short_unit_name(r),
                "ISO": r["ISO"],
                "pathway": r["pathway"],
                "credit_mid_GW": width,
                "cumulative_credit_mid_GW": float(r["x0"] + width),
                "avoided_GWyr_mid": float(r["value_mid"]),
                "avoided_GWyr_per_GW_mid": height,
                "avoided_GWyr_per_GW_low": low,
                "avoided_GWyr_per_GW_high": high,
            }
        )

    label_df = plot_df[plot_df["is_positive"]].head(4).copy()
    for _, r in label_df.iterrows():
        label = short_unit_name(r)
        x = float(r["xmid"])
        y = float(r["value_density_high"]) + 0.18
        ax.text(x, y, label, ha="center", va="bottom", fontsize=3.55, color="#3f3935", rotation=34)
    restart_pair = plot_df[plot_df["pathway"].eq("restart")]
    if not restart_pair.empty:
        x0 = float(restart_pair["x0"].min())
        x1 = float((restart_pair["x0"] + restart_pair["credit_mid_GW"]).max())
        y = max(float(restart_pair["value_density_high"].max()) + 0.28, 3.6)
        ax.plot([x0, x1], [y, y], color=ISO_COLORS["PJM"], lw=0.65, alpha=0.80)
        ax.text((x0 + x1) / 2, y + 0.10, "restart candidates", ha="center", va="bottom", fontsize=3.55, color=ISO_COLORS["PJM"])

    total_positive = positive["value_mid"].sum()
    total_credit = summary["credit_mid_GW"].sum()
    zero_credit = zero["credit_mid_GW"].sum()
    ax.text(
        0.985,
        0.92,
        f"area = avoided GW-yr\n{total_positive:.1f} total",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.25,
        color="#4d4640",
        linespacing=0.90,
    )
    ax.text(
        0.035,
        0.88,
        "higher marginal value",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.1,
        color="#6a443f",
    )

    if zero_credit > 0:
        zero_start = float(plot_df.loc[plot_df["is_positive"], "credit_mid_GW"].sum())
        ax.text(zero_start + zero_credit / 2, 0.34, f"{zero_credit:.1f} GW\nlow/no value", ha="center", va="bottom", fontsize=3.75, color="#746c66", linespacing=0.85)

    ax.set_xlim(-0.05, max(total_credit, 0.1) + 0.12)
    ymax = max(positive["value_density_high"].max() * 1.18, 1.0)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("cumulative credited nuclear capacity (GW)")
    ax.set_ylabel("avoided deficit\nGW-yr per GW")
    ax.grid(axis="y", color="#eee5dc", lw=0.52)
    ax.grid(axis="x", color="#f5eee8", lw=0.35)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#5b544d")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(axis="both", labelsize=5.1, length=2.0)
    handles = [
        Line2D([0], [0], color=ISO_COLORS["PJM"], lw=4.0, alpha=0.78, label="PJM"),
        Line2D([0], [0], color=ISO_COLORS["MISO"], lw=4.0, alpha=0.78, label="MISO"),
        Line2D([0], [0], color=ISO_COLORS["SPP"], lw=4.0, alpha=0.78, label="SPP"),
        Rectangle((0, 0), 1, 1, facecolor="#d2cdc7", alpha=0.45, edgecolor="white", label="low/no value"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#7a6f68", hatch="////", label="restart"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.30),
        ncol=5,
        fontsize=4.1,
        handlelength=0.95,
        columnspacing=0.42,
        handletextpad=0.25,
        borderaxespad=0.0,
    )

    pd.DataFrame(rows).to_csv(OUT / "fig4e_marginal_abatement_curve_revised_source.csv", index=False)
    return fig


def save_panel(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.018)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)
    print(OUT / f"{stem}.pdf")


def main() -> None:
    setup_style()
    cases, rec = load_inputs()
    save_panel(draw_panel_d(cases, rec), "fig4d_deficit_offset_mirror_stream_revised")
    save_panel(draw_panel_e_macc(), "fig4e_marginal_abatement_curve_revised")


if __name__ == "__main__":
    main()
