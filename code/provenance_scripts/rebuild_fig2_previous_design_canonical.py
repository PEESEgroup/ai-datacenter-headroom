from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from rebuild_fig2_5_canonical_figures import load_geo


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig2_previous_design_canonical_20260514"
CURRENT = TABLE_ARCHIVE / "figures" / "Fig2_final_tight_layout copy.pdf"

YEARS = list(range(2025, 2036))
ISO_ORDER_RISK = ["MISO", "SPP", "PJM", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
ISO_ORDER_PIPE = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "ISO-NE", "NYISO"]
FOCUS_ISOS = ["MISO", "SPP", "PJM", "ERCOT"]
SCENARIOS = [("Low", "low"), ("Mid", "mid"), ("High", "high")]

COL = {
    "headroom": "#8FCF7E",
    "supply_pos": "#42B7A9",
    "supply_neg": "#B96A4A",
    "nonai": "#F2B65D",
    "ai_burden": "#B86B3F",
    "final_pos": "#1F9D78",
    "final_neg": "#CF493C",
    "active": "#C9DCCB",
    "ia": "#E6C45B",
    "gen": "#D97A3A",
    "eff": "#1F9D84",
    "solar": "#F2B65D",
    "wind": "#5EAAA4",
    "thermal": "#B96A4A",
    "storage": "#9BD6CF",
    "grid": "#ECE7DF",
    "row": "#F4F0EA",
    "axis": "#6B665F",
    "text": "#33302D",
    "muted": "#8B8580",
    "warn": "#D79A2B",
}

ISO_COLORS = {
    "PJM": "#E7298A",
    "SPP": "#B8860B",
    "ERCOT": "#D95F02",
    "MISO": "#7570B3",
    "CAISO": "#1B9E77",
    "NYISO": "#4DA3C7",
    "ISO-NE": "#6AA84F",
}


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLE / rel)


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.50,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def pbox(box: list[float], rel: list[float]) -> list[float]:
    return [box[0] + rel[0] * box[2], box[1] + rel[1] * box[3], rel[2] * box[2], rel[3] * box[3]]


def style_axis(ax, left: bool = True, bottom: bool = True) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_color(COL["axis"])
    ax.spines["bottom"].set_color(COL["axis"])
    ax.spines["left"].set_linewidth(0.50)
    ax.spines["bottom"].set_linewidth(0.50)
    ax.tick_params(colors=COL["text"], width=0.44, length=1.8, pad=1.0)


def panel_label(fig: plt.Figure, box: list[float], label: str) -> None:
    fig.text(box[0] - 0.004, box[1] + box[3] - 0.004, label, fontsize=9.4, weight="bold", ha="left", va="top")


def add_iso_labels_compact(ax, iso_polys) -> None:
    label_shift = {
        "CAISO": (-26000, 82000),
        "ERCOT": (-42000, -76000),
        "SPP": (-82000, 76000),
        "MISO": (-112000, 72000),
        "PJM": (62000, -52000),
        "NYISO": (15000, -62000),
        "ISO-NE": (76000, -38000),
    }
    font_size = {
        "CAISO": 3.25,
        "ERCOT": 3.25,
        "SPP": 3.22,
        "MISO": 3.18,
        "PJM": 3.22,
        "NYISO": 2.78,
        "ISO-NE": 2.78,
    }
    polys = iso_polys.copy()
    polys["label_pt"] = polys.geometry.representative_point()
    for _, row in polys.iterrows():
        name = str(row["ISO_ASSIGNED"])
        point = row["label_pt"]
        dx, dy = label_shift.get(name, (0, 0))
        txt = f"{name}\n{float(row['headroom_gw']):.1f} GW"
        t = ax.text(
            point.x + dx,
            point.y + dy,
            txt,
            ha="center",
            va="center",
            fontsize=font_size.get(name, 4.2),
            weight="bold",
            color="#213026",
            linespacing=0.82,
            zorder=30,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, boxstyle="round,pad=0.075"),
        )
        t.set_path_effects([pe.withStroke(linewidth=0.32, foreground="#FFFFFF", alpha=0.96)])


def first_negative_year(row: pd.Series) -> int | None:
    neg = row[row < 0]
    if neg.empty:
        return None
    return int(neg.index[0])


def interpolated_crossing(row: pd.Series, cross_year: int | None) -> float | None:
    if cross_year is None:
        return None
    if cross_year <= min(YEARS):
        return float(cross_year)
    prev = cross_year - 1
    y0 = float(row.loc[prev])
    y1 = float(row.loc[cross_year])
    if y0 == y1:
        return float(cross_year)
    return prev + (0.0 - y0) / (y1 - y0)


def build_mats() -> dict[str, pd.DataFrame]:
    heat = read_csv("fig2/fig2d_margin_clock_heatmap_canonical_source.csv")
    mats: dict[str, pd.DataFrame] = {}
    for label, scenario in SCENARIOS:
        rows = []
        for iso in ISO_ORDER_RISK:
            vals = []
            for year in YEARS:
                sub = heat[(heat["ISO"].eq(iso)) & (heat["scenario"].eq(scenario)) & (heat["year"].eq(year))]
                vals.append(float(sub["margin_GW"].iloc[0]))
            rows.append(vals)
        mats[label] = pd.DataFrame(rows, index=ISO_ORDER_RISK, columns=YEARS)
    return mats


def sign_sensitive_flags() -> pd.DataFrame:
    heat = read_csv("fig2/fig2d_margin_clock_heatmap_canonical_source.csv")
    return heat[["ISO", "scenario", "year", "sign_sensitive_to_headroom_extrapolation"]].copy()


def read_bridge_data() -> pd.DataFrame:
    raw = read_csv("fig2/fig2b_iso_margin_bridge_2035_canonical_source.csv")
    rows = []
    for _, r in raw.iterrows():
        nonai_change = float(r["NonAI_load_change_to_2035_MW"]) / 1000.0
        net_headroom_change = float(r["Net_headroom_change_2025_2035_MW"]) / 1000.0
        # Keep the previous visual grammar: 2025 headroom + supply-side change
        # - non-AI load change - AI load growth = 2035 margin. The canonical
        # table stores signed headroom change net of non-AI load; undo that here
        # only for the bridge decomposition so the plotted sum is unchanged.
        supply_change = net_headroom_change + nonai_change
        rows.append(
            {
                "ISO": r["ISO"],
                "headroom_gw": float(r["Carryover_2025_MW"]) / 1000.0,
                "supply_gw": supply_change,
                "nonai_gw": -nonai_change,
                "ai_gw": -float(r["AI_growth_to_2035_MW"]) / 1000.0,
                "margin_2035_gw": float(r["Margin_2035_MW"]) / 1000.0,
            }
        )
    order = ["PJM", "MISO", "SPP", "NYISO", "ISO-NE", "ERCOT", "CAISO"]
    return pd.DataFrame(rows).set_index("ISO").loc[order].reset_index()


def read_pipeline_data() -> pd.DataFrame:
    raw = read_csv("fig2/fig2c_generator_pipeline_canonical_source.csv")
    out = pd.DataFrame(
        {
            "ISO": raw["ISO_RTO"],
            "active_total_gw": raw["MW_sum_total_active_2026_2028"] / 1000.0,
            "ia_total_gw": raw["MW_sum_total_IAstage_2026_2028"] / 1000.0,
            "ia_genonly_gw": raw["MW_sum_genonly_IAstage_2026_2028"] / 1000.0,
            "add_eff_gw": raw["add_eff_MW"] / 1000.0,
            "thermal_eff_gw": raw["thermal_eff_MW"] / 1000.0,
            "wind_eff_gw": raw["wind_eff_MW"] / 1000.0,
            "solar_eff_gw": raw["solar_eff_MW"] / 1000.0,
            "storage_credit_plus_gw": raw["storage_20pct_credit_delta_MW"] / 1000.0,
            "top10_share": raw["top10_share_eff"].fillna(0.0),
            "effective_to_active_pct": 100.0 * raw["effective_over_active_queue"].fillna(0.0),
        }
    )
    return out.set_index("ISO").loc[ISO_ORDER_PIPE].reset_index()


def draw_a(fig: plt.Figure, box: list[float]) -> None:
    _, cty, states, g_iso, iso_polys, bubbles = load_geo()
    p = bubbles["county_potential_mw"].to_numpy(dtype=float)
    pmax = max(float(np.nanmax(p)), 1.0)
    sizes = 1.7 + (np.log10(p + 1) / np.log10(pmax + 1)) ** 1.35 * 11.2
    cmap = LinearSegmentedColormap.from_list("headroom_green", ["#F7FCF5", "#E2F3DC", "#B9E1B3", "#66BD75", "#1B8A4B"])
    norm = Normalize(vmin=0.0, vmax=float(np.ceil(iso_polys["headroom_gw"].max())))

    ax = fig.add_axes(pbox(box, [0.000, 0.030, 0.890, 0.930]))
    cax = fig.add_axes(pbox(box, [0.923, 0.255, 0.026, 0.500]))
    cty.plot(ax=ax, color="#FFFFFF", edgecolor="#F0F0F0", linewidth=0.065, zorder=0)
    states.boundary.plot(ax=ax, color="#D8D8D8", linewidth=0.28, alpha=0.55, zorder=1)
    iso_polys.plot(ax=ax, color=[cmap(norm(v)) for v in iso_polys["headroom_gw"].astype(float)], edgecolor="#FFFFFF", linewidth=0.30, alpha=0.84, zorder=5)
    g_iso.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.09, alpha=0.20, zorder=6)
    iso_polys.boundary.plot(ax=ax, color="#FFFFFF", linewidth=1.55, alpha=0.95, zorder=8)
    iso_polys.boundary.plot(ax=ax, color="#506258", linewidth=0.62, alpha=0.84, zorder=9)
    ax.scatter(bubbles.geometry.x.to_numpy(), bubbles.geometry.y.to_numpy(), s=sizes, facecolors="#238B45", edgecolors="#F7FFF7", linewidths=0.14, alpha=0.085, zorder=12)
    add_iso_labels_compact(ax, iso_polys)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("2025 headroom (GW)", fontsize=5.7, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.2, width=0.40, length=1.8, pad=1.0)
    cb.outline.set_linewidth(0.40)

    ref_vals = [100, 1000, 5000]
    ref_sizes = [2.0 + (np.log10(v + 1) / np.log10(pmax + 1)) ** 1.35 * 14.0 for v in ref_vals]
    handles = [ax.scatter([], [], s=s, facecolors="#238B45", edgecolors="#F7FFF7", linewidths=0.18, alpha=0.38) for s in ref_sizes]
    leg = ax.legend(handles, ["100 MW", "1 GW", "5 GW"], title="County potential", loc="lower left", bbox_to_anchor=(0.010, 0.032), frameon=True, fancybox=False, framealpha=0.88, borderpad=0.20, labelspacing=0.10, handlelength=0.80, handletextpad=0.28, fontsize=5.0, title_fontsize=5.0)
    leg.get_frame().set_edgecolor("#D8D8D8")
    leg.get_frame().set_linewidth(0.42)
    ax.set_xlim(*cty.total_bounds[[0, 2]])
    ax.set_ylim(*cty.total_bounds[[1, 3]])
    ax.axis("off")


def draw_b(fig: plt.Figure, box: list[float]) -> None:
    df = read_bridge_data()
    ax = fig.add_axes(pbox(box, [0.165, 0.215, 0.810, 0.700]))
    y = np.arange(len(df))
    ax.axvline(0, color="#777777", lw=0.58, zorder=0)
    ax.grid(axis="x", color=COL["grid"], lw=0.44, zorder=0)
    for row_y in np.arange(0.5, len(df), 1):
        ax.axhline(row_y, color=COL["row"], lw=0.36, zorder=0)
    offsets = [0.16, 0.052, -0.052, -0.16]
    for yi, row in df.iterrows():
        cum = 0.0
        last_y = None
        pieces = [
            (row["headroom_gw"], COL["headroom"]),
            (row["supply_gw"], COL["supply_pos"] if row["supply_gw"] >= 0 else COL["supply_neg"]),
            (row["nonai_gw"], COL["nonai"]),
            (row["ai_gw"], COL["ai_burden"]),
        ]
        for idx, (delta, color) in enumerate(pieces):
            if abs(delta) < 1e-7:
                continue
            nxt = cum + delta
            yseg = yi + offsets[idx]
            if last_y is not None:
                ax.plot([cum, cum], [last_y, yseg], color="#777777", lw=0.30, alpha=0.55, zorder=2)
            ax.plot([cum, nxt], [yseg, yseg], color=color, lw=2.55, solid_capstyle="butt", zorder=3)
            ax.plot([nxt, nxt], [yseg - 0.058, yseg + 0.058], color="#555555", lw=0.42, zorder=4)
            cum = nxt
            last_y = yseg
        final = row["margin_2035_gw"]
        fcol = COL["final_pos"] if final >= 0 else COL["final_neg"]
        ax.scatter(final, yi + offsets[-1], s=12, color=fcol, edgecolor="white", linewidth=0.38, zorder=6)
        ax.text(final + (0.55 if final >= 0 else -0.55), yi + offsets[-1], f"{final:+.1f}", ha="left" if final >= 0 else "right", va="center", fontsize=5.0, color=fcol, weight="bold", zorder=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, boxstyle="round,pad=0.05"))
    ax.set_yticks(y)
    ax.set_yticklabels(df["ISO"], fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlim(-22, 42)
    ax.set_xticks([-20, -10, 0, 10, 20, 30, 40])
    ax.set_xlabel("GW", fontsize=5.9, labelpad=1.0)
    ax.tick_params(axis="y", length=0, pad=1.0)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    handles = [
        Line2D([0], [0], color=COL["headroom"], lw=3.6, label="headroom"),
        Line2D([0], [0], color=COL["supply_pos"], lw=3.6, label="net supply"),
        Line2D([0], [0], color=COL["nonai"], lw=3.6, label="non-AI"),
        Line2D([0], [0], color=COL["ai_burden"], lw=3.6, label="AI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["final_neg"], markeredgecolor="white", markersize=4.5, label="2035 margin"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(box[0] + box[2] * 0.56, box[1] + box[3] * 0.035), ncol=5, frameon=False, fontsize=5.0, handlelength=1.00, columnspacing=0.42, handletextpad=0.24, borderaxespad=0.0)


def draw_c(fig: plt.Figure, box: list[float]) -> None:
    mats = build_mats()
    flags = sign_sensitive_flags()
    all_vals = np.concatenate([mat.to_numpy().ravel() for mat in mats.values()])
    norm = TwoSlopeNorm(vmin=float(np.floor(np.nanmin(all_vals) / 2) * 2), vcenter=0.0, vmax=float(np.ceil(np.nanmax(all_vals) / 2) * 2))
    cmap = LinearSegmentedColormap.from_list("margin_clock", ["#B2182B", "#E66F51", "#F7F1E7", "#8CCFC1", "#178F71"], N=256)
    left, bottom, width, height = box
    heat_left = left + width * 0.064
    cbar_w = width * 0.014
    cbar_left = left + width - cbar_w
    heat_w = cbar_left - heat_left - width * 0.030
    row_h = height * 0.255
    row_gap = height * 0.018
    top = bottom + height * 0.962
    im = None
    for ridx, (label, scenario) in enumerate(SCENARIOS):
        y0 = top - (ridx + 1) * row_h - ridx * row_gap
        ax = fig.add_axes([heat_left, y0, heat_w, row_h])
        mat = mats[label]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_yticks(np.arange(len(ISO_ORDER_RISK)))
        ax.set_yticklabels(ISO_ORDER_RISK, fontsize=5.05)
        ax.set_xticks(np.arange(len(YEARS)))
        ax.set_xticklabels([str(y) for y in YEARS], fontsize=4.95)
        if ridx < 2:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="both", length=0, pad=0.8)
        ax.set_xticks(np.arange(-0.5, len(YEARS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ISO_ORDER_RISK), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.42)
        ax.tick_params(which="minor", bottom=False, left=False)
        for yi, iso in enumerate(ISO_ORDER_RISK):
            row = mat.loc[iso]
            cross_year = first_negative_year(row)
            if cross_year is not None:
                xidx = YEARS.index(cross_year)
                ax.add_patch(Rectangle((xidx - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="#272727", linewidth=0.52, zorder=5))
                cross_exact = interpolated_crossing(row, cross_year)
                if cross_exact is not None:
                    ax.plot([cross_exact - 2025, cross_exact - 2025], [yi - 0.30, yi + 0.30], color="#272727", lw=0.66, solid_capstyle="round", zorder=6)
            for year in row[(row > 0) & (row <= 2.0)].index:
                ax.scatter(YEARS.index(int(year)), yi, s=8.2, facecolor="#FFF7D7", edgecolor=COL["warn"], linewidth=0.50, zorder=7)
            sens = flags[(flags["ISO"].eq(iso)) & (flags["scenario"].eq(scenario)) & (flags["sign_sensitive_to_headroom_extrapolation"].astype(bool))]
            for year in sens["year"].astype(int):
                xidx = YEARS.index(year)
                ax.plot([xidx - 0.24, xidx + 0.24], [yi + 0.24, yi - 0.24], color="#5C5C5C", lw=0.48, zorder=8)
        ax.text(0.005, 1.015, f"{label}-growth", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.60, weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([cbar_left, bottom + height * 0.245, cbar_w, height * 0.610])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("margin (GW)", fontsize=5.4, labelpad=1.5)
    cb.ax.tick_params(labelsize=4.8, length=1.6, width=0.38, pad=0.8)
    cb.outline.set_linewidth(0.38)
    handles = [
        Line2D([0], [0], color="#272727", lw=1.05, label="first deficit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#FFF7D7", markeredgecolor=COL["warn"], markersize=4.0, label="0-2 GW"),
        Line2D([0], [0], color="#5C5C5C", lw=0.92, label="sign-sensitive"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(left + width * 0.50, bottom + height * 0.004), frameon=False, ncol=3, fontsize=5.2, handlelength=1.12, columnspacing=0.85, handletextpad=0.32, borderaxespad=0.0)


def pct_label(v: float) -> str:
    if v <= 0:
        return "0%"
    if v < 0.1:
        return "<0.1%"
    return f"{v:.1f}%"


def draw_d(fig: plt.Figure, box: list[float]) -> None:
    df = read_pipeline_data()
    y = np.arange(len(df))
    left, bottom, width, height = box
    ax1 = fig.add_axes([left + width * 0.064, bottom + height * 0.220, width * 0.370, height * 0.650])
    ax2 = fig.add_axes([left + width * 0.474, bottom + height * 0.220, width * 0.320, height * 0.650], sharey=ax1)
    ax3 = fig.add_axes([left + width * 0.832, bottom + height * 0.220, width * 0.168, height * 0.650], sharey=ax1)
    for ax in [ax1, ax2, ax3]:
        for yy in np.arange(0.5, len(df), 1):
            ax.axhline(yy, color=COL["row"], lw=0.38, zorder=0)
    stages = [
        ("active_total_gw", COL["active"], "o", 20),
        ("ia_total_gw", COL["ia"], "D", 18),
        ("ia_genonly_gw", COL["gen"], "s", 17),
        ("add_eff_gw", COL["eff"], "o", 24),
    ]
    for yi, row in df.iterrows():
        xs = [max(float(row[col]), 0.070) for col, *_ in stages]
        ax1.plot(xs, [yi] * len(xs), color="#CFC8BE", lw=0.66, zorder=1)
        ax1.plot([xs[-2], xs[-1]], [yi, yi], color=COL["eff"], lw=0.82, alpha=0.86, zorder=2)
        for col, color, marker, size in stages:
            ax1.scatter(max(float(row[col]), 0.070), yi, s=size, marker=marker, facecolor=color, edgecolor="white", linewidth=0.34, zorder=4)
        ax1.text(390, yi, pct_label(float(row["effective_to_active_pct"])), ha="right", va="center", fontsize=4.45, color=COL["muted"])
    ax1.set_xscale("log")
    ax1.set_xlim(0.055, 420)
    ax1.set_xticks([0.1, 1, 10, 100, 300])
    ax1.set_xticklabels(["0.1", "1", "10", "100", "300"], fontsize=5.1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["ISO"], fontsize=5.9)
    ax1.invert_yaxis()
    ax1.grid(axis="x", which="major", color=COL["grid"], lw=0.42)
    ax1.set_xlabel("queue capacity (GW, log)", fontsize=5.6, labelpad=0.8)
    ax1.text(0.0, 1.035, "screened queue", transform=ax1.transAxes, fontsize=5.6, weight="bold", ha="left")
    ax1.text(0.985, 1.035, "effective/active", transform=ax1.transAxes, fontsize=4.35, color=COL["muted"], ha="right")
    style_axis(ax1)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0, pad=0.8)

    left_vals = np.zeros(len(df))
    for col, color in [("thermal_eff_gw", COL["thermal"]), ("wind_eff_gw", COL["wind"]), ("solar_eff_gw", COL["solar"])]:
        vals = df[col].to_numpy(dtype=float)
        ax2.barh(y, vals, left=left_vals, color=color, height=0.44, edgecolor="white", linewidth=0.24, zorder=3)
        left_vals += vals
    storage = df["storage_credit_plus_gw"].to_numpy(dtype=float)
    ax2.barh(y, storage, left=left_vals, height=0.44, facecolor="none", edgecolor=COL["storage"], linewidth=0.74, alpha=0.95, hatch="////", zorder=2)
    for yi, row in df.iterrows():
        eff = float(row["add_eff_gw"])
        ax2.text(eff + 0.16 if eff > 0.15 else 0.09, yi, f"{eff:.1f}" if eff > 0.15 else "0", va="center", ha="left", fontsize=4.8, color=COL["text"] if eff > 0.15 else COL["muted"])
    ax2.set_xlim(0, 15.5)
    ax2.set_xticks([0, 5, 10, 15])
    ax2.grid(axis="x", color=COL["grid"], lw=0.42)
    ax2.set_xlabel("effective additions (GW)", fontsize=5.6, labelpad=0.8)
    ax2.text(0.0, 1.035, "resource mix", transform=ax2.transAxes, fontsize=5.6, weight="bold", ha="left")
    ax2.set_yticks(y)
    ax2.tick_params(axis="y", left=False, labelleft=False, length=0)
    style_axis(ax2, left=False)

    storage_delta = df["storage_credit_plus_gw"].astype(float).to_numpy()
    max_storage = max(float(np.nanmax(storage_delta)), 1.0)
    ax3.set_xlim(0, 1)
    ax3.set_xticks([])
    for yi, row in df.iterrows():
        share = float(row["top10_share"]) * 100.0
        ax3.scatter(0.22, yi, s=9 + share * 0.85, facecolor="#D8ECE2", edgecolor="#7EA88E", linewidth=0.40, zorder=4)
        ax3.text(0.38, yi, f"{share:.0f}%" if share > 0 else "0%", ha="left", va="center", fontsize=4.45, color="#6B7F70")
        sd = float(row["storage_credit_plus_gw"])
        if sd > 0.05:
            x0 = 0.66
            x1 = x0 + 0.27 * sd / max_storage
            ax3.plot([x0, x1], [yi, yi], color=COL["storage"], lw=1.45, solid_capstyle="butt", zorder=3)
            ax3.text(min(x1 + 0.023, 0.985), yi, f"+{sd:.1f}", ha="left" if x1 < 0.90 else "right", va="center", fontsize=4.35, color="#5C9D94")
        else:
            ax3.text(0.66, yi, "0", ha="left", va="center", fontsize=4.35, color=COL["muted"])
    ax3.text(0.22, 1.018, "top-10\nshare", transform=ax3.transAxes, ha="center", va="bottom", fontsize=4.45, color=COL["muted"], linespacing=0.84)
    ax3.text(0.78, 1.018, "storage\ncredit", transform=ax3.transAxes, ha="center", va="bottom", fontsize=4.45, color=COL["muted"], linespacing=0.84)
    ax3.set_yticks(y)
    ax3.tick_params(axis="y", left=False, labelleft=False, length=0)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["active"], markeredgecolor="white", markersize=4.2, label="active"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COL["ia"], markeredgecolor="white", markersize=4.0, label="IA-stage"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COL["gen"], markeredgecolor="white", markersize=4.0, label="IA gen-only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["eff"], markeredgecolor="white", markersize=4.2, label="effective"),
        Patch(facecolor=COL["thermal"], label="thermal"),
        Patch(facecolor=COL["wind"], label="wind"),
        Patch(facecolor=COL["solar"], label="solar"),
        Patch(facecolor="none", edgecolor=COL["storage"], hatch="////", label="storage credit"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(left + width * 0.50, bottom + height * 0.012), ncol=8, frameon=False, fontsize=5.55, handlelength=0.96, handletextpad=0.24, columnspacing=0.50, borderaxespad=0.0)


def draw_e(fig: plt.Figure, box: list[float]) -> None:
    traj = read_csv("fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv")
    left, bottom, width, height = box
    axes = []
    inner_left = left + width * 0.055
    inner_w = width * 0.945
    gap_w = width * 0.030
    ax_w = (inner_w - 3 * gap_w) / 4
    ax_bottom = bottom + height * 0.255
    ax_h = height * 0.620
    rows = []
    for idx, iso in enumerate(FOCUS_ISOS):
        ax = fig.add_axes([inner_left + idx * (ax_w + gap_w), ax_bottom, ax_w, ax_h])
        axes.append(ax)
        color = ISO_COLORS[iso]
        d = traj[traj["ISO"].eq(iso)].sort_values("year")
        low = d["low_growth_margin_GW"].to_numpy(dtype=float)
        mid = d["mid_growth_margin_GW"].to_numpy(dtype=float)
        high = d["high_growth_margin_GW"].to_numpy(dtype=float)
        variant_low = d["headroom_extrapolation_low_GW"].to_numpy(dtype=float)
        variant_high = d["headroom_extrapolation_high_GW"].to_numpy(dtype=float)
        ymin = min(np.nanmin(high), np.nanmin(variant_low), -0.5)
        ymax = max(np.nanmax(low), np.nanmax(variant_high), 0.5)
        pad = max(0.55, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.axhspan(0, ymax + pad, color="#EFF7F1", zorder=0)
        ax.axhspan(ymin - pad, 0, color="#FDEDEA", zorder=0)
        ax.axhline(0, color="#5D5D5D", lw=0.60, zorder=2)
        ax.fill_between(YEARS, high, low, color=color, alpha=0.14, linewidth=0, zorder=1)
        ax.fill_between(YEARS, variant_low, variant_high, color="#D8A23A", alpha=0.18, linewidth=0, zorder=2)
        ax.plot(YEARS, low, color=color, lw=0.58, ls=(0, (2, 2)), alpha=0.55, zorder=3)
        ax.plot(YEARS, high, color=color, lw=0.58, ls=(0, (2, 2)), alpha=0.55, zorder=3)
        ax.plot(YEARS, mid, color=color, lw=1.20, marker="o", ms=2.0, zorder=4)
        row = pd.Series(mid, index=YEARS)
        cross = first_negative_year(row)
        cross_exact = interpolated_crossing(row, cross)
        label = "no deficit by 2035"
        if cross_exact is not None:
            ax.axvline(cross_exact, color="#444444", lw=0.56, ls="--", zorder=5)
            ax.scatter(cross_exact, np.interp(cross_exact, YEARS, mid), s=11, color=COL["final_neg"], zorder=6)
            label = f"cross: {cross}"
        end_mid = float(mid[-1])
        ax.text(0.965, 0.12 if end_mid >= 0 else 0.88, f"{end_mid:+.1f} GW", transform=ax.transAxes, ha="right", va="bottom" if end_mid >= 0 else "top", fontsize=5.45, color=COL["final_pos"] if end_mid >= 0 else COL["final_neg"], weight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.62, boxstyle="round,pad=0.08"))
        ax.text(0.035, 0.94, iso, transform=ax.transAxes, ha="left", va="top", fontsize=6.05, weight="bold")
        ax.text(0.035, 0.77, label, transform=ax.transAxes, ha="left", va="top", fontsize=4.75, color="#666666")
        ax.set_xlim(2025, 2035)
        ax.set_xticks([2025, 2030, 2035])
        ax.grid(axis="y", color="#E6E6E6", lw=0.36)
        ax.grid(axis="x", color="#EFEFEF", lw=0.36)
        ax.tick_params(axis="y", labelsize=5.0, length=1.5, width=0.40, pad=0.8)
        ax.tick_params(axis="x", labelsize=5.1, length=1.5, width=0.40, pad=0.8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.45)
        ax.spines["bottom"].set_linewidth(0.45)
        for year, lo, mi, hi, vlo, vhi in zip(YEARS, low, mid, high, variant_low, variant_high, strict=True):
            rows.append({"ISO": iso, "year": year, "low_growth_margin_GW": lo, "mid_growth_margin_GW": mi, "high_growth_margin_GW": hi, "headroom_sensitivity_low_GW": vlo, "headroom_sensitivity_high_GW": vhi})
    axes[0].set_ylabel("hosting margin (GW)", fontsize=5.8, labelpad=1.0)
    handles = [
        Patch(facecolor="#777777", alpha=0.14, edgecolor="none", label="AI low-high scenarios"),
        Line2D([0], [0], color="#444444", lw=1.25, marker="o", ms=3.1, label="central trajectory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.05, bbox_to_anchor=(left + width * 0.50, bottom + height * 0.020), handlelength=1.34, columnspacing=0.90, handletextpad=0.34, borderaxespad=0.0)
    pd.DataFrame(rows).to_csv(OUT / "fig2_previous_design_e_trajectory_source.csv", index=False)


def save_ab_panels() -> None:
    for stem, draw_fn in [
        ("fig2a_previous_design_iso_headroom_map", draw_a),
        ("fig2b_previous_design_iso_margin_bridge", draw_b),
    ]:
        fig = plt.figure(figsize=(3.44, 1.78), facecolor="white", constrained_layout=False)
        box = [0.035, 0.070, 0.930, 0.875]
        draw_fn(fig, box)
        panel_label(fig, box, stem[4])
        fig.savefig(OUT / f"{stem}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.012)
        fig.savefig(OUT / f"{stem}_1200dpi.png", dpi=1200, facecolor="white", bbox_inches="tight", pad_inches=0.012)
        plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    save_ab_panels()
    fig = plt.figure(figsize=(7.1, 8.1), facecolor="white", constrained_layout=False)
    top_left = 0.040
    top_right = 0.975
    top_gap = 0.045
    top_w = (top_right - top_left - top_gap) / 2.0
    boxes = {
        "a": [top_left, 0.760, top_w, 0.220],
        "b": [top_left + top_w + top_gap, 0.748, top_w, 0.220],
        "c": [0.040, 0.500, 0.935, 0.245],
        "d": [0.040, 0.287, 0.935, 0.190],
        "e": [0.040, 0.045, 0.935, 0.220],
    }
    draw_a(fig, boxes["a"])
    draw_b(fig, boxes["b"])
    draw_c(fig, boxes["c"])
    draw_d(fig, boxes["d"])
    draw_e(fig, boxes["e"])
    for label, box in boxes.items():
        panel_label(fig, box, label)
    pdf = OUT / "Fig2_previous_design_canonical.pdf"
    png = OUT / "Fig2_previous_design_canonical_1200dpi.png"
    svg = OUT / "Fig2_previous_design_canonical.svg"
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png, dpi=1200, facecolor="white", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(svg, facecolor="white", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    shutil.copyfile(pdf, CURRENT)
    print(pdf)
    print(png)
    print(f"updated {CURRENT}")


if __name__ == "__main__":
    main()
