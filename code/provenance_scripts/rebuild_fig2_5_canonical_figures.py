from __future__ import annotations

import importlib.util
import math
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


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig2_5_canonical_20260514"
FIG = TABLE_ARCHIVE / "figures"
GEO_SCRIPT = Path(__file__).resolve().parents[2] / "code" / "provenance_scripts" / "make_fig2a_iso_headroom_map.py"

ISO7 = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
ISO_CLOCK_ORDER = ["MISO", "SPP", "PJM", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
SCENARIOS = ["low", "mid", "high"]
SCENARIO_LABEL = {"low": "Low-growth", "mid": "Mid-growth", "high": "High-growth"}

COL = {
    "green": "#86C77A",
    "teal": "#43B5A6",
    "orange": "#F3B65C",
    "red": "#E75B4E",
    "red_dark": "#C43D33",
    "blue": "#6F99D6",
    "purple": "#8078B9",
    "gold": "#C99A20",
    "gray": "#8E8E8E",
    "lightgray": "#E8E8E8",
    "ink": "#333333",
}

ISO_COLORS = {
    "PJM": "#E7298A",
    "MISO": "#8078B9",
    "SPP": "#C99A20",
    "ERCOT": "#E66101",
    "CAISO": "#1B9E77",
    "NYISO": "#5B8DB8",
    "ISO-NE": "#7AA6C7",
}

ARCH_COLORS = {
    "deficit pressure": "#C65F4B",
    "low-headroom growth": "#D89A3D",
    "supply-buffered expansion": "#61A88F",
    "high-margin limited growth": "#5B8DB8",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLE / rel)


def clean_axis(ax, grid: str | None = None) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.55)
    if grid == "x":
        ax.grid(axis="x", color="#EAEAEA", linewidth=0.55)
    elif grid == "y":
        ax.grid(axis="y", color="#EAEAEA", linewidth=0.55)
    elif grid == "both":
        ax.grid(color="#EAEAEA", linewidth=0.45)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.09, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="black",
    )


def short_zone_name(value: object) -> str:
    text = str(value)
    mapping = {
        "PACIFICGASANDELECTRIC": "PG&E",
        "SANDIEGOGASELECTRIC": "SDG&E",
        "SOUTHERNCALIFORNIAEDISON": "SCE",
        "VALLEYELECTRICASSOCIATION": "Valley Electric",
        "COMMONWEALTH EDISON CO": "ComEd",
        "American Electric Power": "AEP",
        "DOMINION ENERGY": "Dominion",
    }
    for k, v in mapping.items():
        if text.upper() == k.upper():
            return v
    if text.upper().startswith("LRZ"):
        return text[:4].replace("ST", "")
    text = text.replace("ELECTRIC", " Elec.").replace("POWER", " Pwr.")
    text = text.replace("COMPANY", "Co.").replace("TRANSMISSION", "Trans.")
    text = text.title()
    return text[:24]


def save_fig(fig: plt.Figure, stem: str, current_pdf: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.015)
    shutil.copyfile(pdf, FIG / current_pdf)
    plt.close(fig)
    print(f"saved {pdf}")


def load_geo():
    geo = load_module("fig2a_geo_helpers", GEO_SCRIPT)
    cty = geo.load_counties()
    g_iso, iso_polys = geo.build_iso_polygons(cty)
    head = read_csv("fig2/fig2a_iso_headroom_map_canonical_source.csv")
    head["ISO"] = head["ISO"].astype(str)
    iso_polys = iso_polys.drop(columns=[c for c in ["headroom_gw"] if c in iso_polys.columns]).merge(
        head[["ISO", "headroom_2025_GW"]],
        left_on="ISO_ASSIGNED",
        right_on="ISO",
        how="left",
    )
    iso_polys["headroom_gw"] = iso_polys["headroom_2025_GW"]
    bubbles = geo.build_bubbles(cty, iso_polys)
    states = cty.dissolve(by="STATEFP", as_index=False)
    return geo, cty, states, g_iso, iso_polys, bubbles


def plot_iso_base_map(
    ax,
    cty,
    states,
    g_iso,
    iso_polys,
    value_col: str,
    cmap,
    norm,
    boundary_color="#5F6A62",
    non_iso_alpha=0.55,
):
    cty.plot(ax=ax, color="#FFFFFF", edgecolor="#F0F0F0", linewidth=0.06, zorder=0)
    states.boundary.plot(ax=ax, color="#D8D8D8", linewidth=0.28, alpha=non_iso_alpha, zorder=1)
    colors = [cmap(norm(v)) if pd.notna(v) else (0.95, 0.95, 0.95, 1) for v in iso_polys[value_col]]
    iso_polys.plot(ax=ax, color=colors, edgecolor="#FFFFFF", linewidth=0.25, alpha=0.86, zorder=5)
    g_iso.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.09, alpha=0.20, zorder=6)
    iso_polys.boundary.plot(ax=ax, color="#FFFFFF", linewidth=1.55, alpha=0.94, zorder=8)
    iso_polys.boundary.plot(ax=ax, color=boundary_color, linewidth=0.60, alpha=0.88, zorder=9)
    ax.set_axis_off()


def add_iso_map_labels(ax, iso_polys, value_col: str, suffix: str = "GW", fs: float = 5.7) -> None:
    shifts = {
        "CAISO": (0, -25000),
        "ERCOT": (0, -45000),
        "SPP": (-15000, 45000),
        "MISO": (-50000, 35000),
        "PJM": (30000, -65000),
        "NYISO": (25000, -50000),
        "ISO-NE": (90000, -25000),
    }
    polys = iso_polys.copy()
    polys["pt"] = polys.geometry.representative_point()
    for _, r in polys.iterrows():
        name = str(r["ISO_ASSIGNED"])
        dx, dy = shifts.get(name, (0, 0))
        txt = f"{name}\n{float(r[value_col]):.1f} {suffix}" if pd.notna(r[value_col]) else name
        t = ax.text(
            r["pt"].x + dx,
            r["pt"].y + dy,
            txt,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold",
            color="#26342A",
            linespacing=0.86,
            zorder=40,
        )
        t.set_path_effects([pe.withStroke(linewidth=1.45, foreground="white", alpha=0.96)])


def add_map_bubbles(ax, bubbles, color="#4BAE67", alpha=0.16, edge="#F7FFF7") -> None:
    p = bubbles["county_potential_mw"].to_numpy(dtype=float)
    pmax = max(float(np.nanmax(p)), 1.0)
    sizes = 2.2 + (np.log10(p + 1) / np.log10(pmax + 1)) ** 1.35 * 15.5
    ax.scatter(
        bubbles.geometry.x,
        bubbles.geometry.y,
        s=sizes,
        facecolors=color,
        edgecolors=edge,
        linewidths=0.20,
        alpha=alpha,
        zorder=12,
    )
    handles = []
    for val in [100, 1000, 5000]:
        s = 2.2 + (np.log10(val + 1) / np.log10(pmax + 1)) ** 1.35 * 15.5
        handles.append(
            ax.scatter([], [], s=s, facecolors=color, edgecolors=edge, linewidths=0.25, alpha=0.32)
        )
    leg = ax.legend(
        handles,
        ["100 MW", "1 GW", "5 GW"],
        title="Bubble area = county potential",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.04),
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.24,
        title_fontsize=5.7,
        fontsize=5.4,
    )
    leg.get_frame().set_edgecolor("#DDDDDD")
    leg.get_frame().set_linewidth(0.5)


def project_lonlat(cty, lon: pd.Series, lat: pd.Series):
    import geopandas as gpd

    g = gpd.GeoDataFrame(
        {"lon": lon.to_numpy(dtype=float), "lat": lat.to_numpy(dtype=float)},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(cty.crs)
    return g.geometry.x.to_numpy(), g.geometry.y.to_numpy()


def draw_fig2() -> None:
    geo, cty, states, g_iso, iso_polys, bubbles = load_geo()
    set_style()
    fig = plt.figure(figsize=(7.10, 8.10), facecolor="white")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[1.70, 1.55, 1.60, 1.30],
        width_ratios=[1.12, 1.00],
        hspace=0.26,
        wspace=0.15,
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.985,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])
    sub_e = gs[3, :].subgridspec(1, 4, wspace=0.20)
    e_axes = [fig.add_subplot(sub_e[0, i]) for i in range(4)]

    # a. ISO headroom map
    cmap_g = LinearSegmentedColormap.from_list("headroom_green", ["#F8FCF6", "#BFE3B5", "#38A35A", "#087538"])
    norm_g = Normalize(0, 20)
    plot_iso_base_map(ax_a, cty, states, g_iso, iso_polys, "headroom_gw", cmap_g, norm_g)
    add_map_bubbles(ax_a, bubbles)
    add_iso_map_labels(ax_a, iso_polys, "headroom_gw", fs=5.1)
    panel_label(ax_a, "a", x=-0.02, y=1.02)
    cax_a = ax_a.inset_axes([0.88, 0.28, 0.025, 0.48])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_g, cmap=cmap_g), cax=cax_a)
    cb.set_label("2025 headroom (GW)", fontsize=5.7, labelpad=2)
    cb.ax.tick_params(labelsize=5.2, width=0.45, length=2.0, pad=1)
    cb.outline.set_linewidth(0.45)

    # b. Corrected 2035 margin bridge.
    bridge = read_csv("fig2/fig2b_iso_margin_bridge_2035_canonical_source.csv")
    order = ["PJM", "MISO", "SPP", "NYISO", "ISO-NE", "ERCOT", "CAISO"]
    b = bridge.set_index("ISO").loc[order].reset_index()
    y = np.arange(len(b))
    ax_b.axvline(0, color="#777777", lw=0.7, zorder=1)
    for i, r in b.iterrows():
        start = 0.0
        carry = r["Carryover_2025_MW"] / 1000
        slack = r["Signed_net_slack_2035_MW"] / 1000
        ai = r["AI_growth_to_2035_MW"] / 1000
        margin = r["Margin_2035_MW"] / 1000
        ax_b.barh(i, carry, left=0, height=0.45, color=COL["green"], edgecolor="none", zorder=2)
        net = slack - carry
        ax_b.barh(i, net, left=carry, height=0.45, color=COL["teal"] if net >= 0 else "#B76546", edgecolor="none", zorder=2)
        ax_b.barh(i, -ai, left=slack, height=0.45, color=COL["red"], edgecolor="none", zorder=3)
        ax_b.plot([margin], [i], "o", ms=4.2, color=COL["red_dark"] if margin < 0 else "#1B9E77", zorder=5)
        ax_b.text(
            margin + (-0.55 if margin < 0 else 0.45),
            i,
            f"{margin:+.1f}",
            ha="right" if margin < 0 else "left",
            va="center",
            fontsize=6.3,
            weight="bold",
            color=COL["red_dark"] if margin < 0 else "#1B9E77",
        )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(order)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("GW")
    ax_b.set_xlim(-22, 44)
    clean_axis(ax_b, "x")
    panel_label(ax_b, "b", x=-0.08, y=1.02)
    ax_b.legend(
        [
            Patch(color=COL["green"]),
            Patch(color=COL["teal"]),
            Patch(color=COL["red"]),
            Line2D([0], [0], marker="o", lw=0, color="#555555"),
        ],
        ["2025 headroom", "net headroom change", "AI growth", "2035 margin"],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.99),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.7,
        handletextpad=0.35,
        fontsize=5.0,
    )
    ax_b.text(
        0.70,
        0.42,
        "bridge uses signed\n2035 pre-AI slack",
        transform=ax_b.transAxes,
        fontsize=5.2,
        color="#555555",
        ha="left",
        va="center",
    )

    # c. Margin clock heatmap.
    heat = read_csv("fig2/fig2d_margin_clock_heatmap_canonical_source.csv")
    years = list(range(2025, 2036))
    rows = []
    ylabels = []
    scenario_y0 = {}
    for s in SCENARIOS:
        scenario_y0[s] = len(rows)
        for iso in ISO_CLOCK_ORDER:
            vals = []
            for yr in years:
                v = heat[(heat["scenario"].eq(s)) & (heat["ISO"].eq(iso)) & (heat["year"].eq(yr))]["margin_GW"]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)
            rows.append(vals)
            ylabels.append(iso)
    mat = np.array(rows)
    cmap_m = LinearSegmentedColormap.from_list("margin", ["#C81D35", "#F08A63", "#F6F3EE", "#8DD0C2", "#009E73"])
    norm_m = TwoSlopeNorm(vmin=-16, vcenter=0, vmax=20)
    im = ax_c.imshow(mat, cmap=cmap_m, norm=norm_m, aspect="auto", interpolation="nearest")
    ax_c.set_xticks(range(len(years)))
    ax_c.set_xticklabels(years)
    ax_c.set_yticks(range(len(ylabels)))
    ax_c.set_yticklabels(ylabels)
    ax_c.tick_params(axis="y", length=0, pad=2)
    ax_c.tick_params(axis="x", length=0)
    for x in np.arange(-0.5, len(years), 1):
        ax_c.axvline(x + 0.5, color="white", lw=0.55)
    for yline in np.arange(-0.5, len(ylabels), 1):
        ax_c.axhline(yline + 0.5, color="white", lw=0.55)
    for s in SCENARIOS:
        y0 = scenario_y0[s]
        ax_c.text(-0.45, y0 - 0.58, SCENARIO_LABEL[s], fontsize=7.8, weight="bold", ha="left", va="bottom", color="#303030")
        if y0 > 0:
            ax_c.axhline(y0 - 0.5, color="white", lw=2.0)
    for s in SCENARIOS:
        for j, iso in enumerate(ISO_CLOCK_ORDER):
            yy = scenario_y0[s] + j
            row = heat[(heat["scenario"].eq(s)) & (heat["ISO"].eq(iso))].copy()
            neg = row[row["margin_GW"] < 0]
            if not neg.empty:
                first = int(neg["year"].min())
                xx = years.index(first)
                ax_c.add_patch(Rectangle((xx - 0.5, yy - 0.5), 1, 1, facecolor="none", edgecolor="#222222", lw=0.75, zorder=5))
            warn = row[row["warning_0_to_2GW"].astype(bool)]
            for _, wr in warn.iterrows():
                ax_c.plot(years.index(int(wr["year"])), yy, marker="o", ms=3.3, mfc="white", mec="#D99A1E", mew=0.85, zorder=7)
            sens = row[row["sign_sensitive_to_headroom_extrapolation"].astype(bool)]
            for _, sr in sens.iterrows():
                ax_c.add_patch(
                    Rectangle(
                        (years.index(int(sr["year"])) - 0.5, yy - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#4F4F4F",
                        lw=0.35,
                        hatch="////",
                        zorder=6,
                    )
                )
    ax_c.set_xlabel("Year")
    panel_label(ax_c, "c", x=-0.035, y=1.08)
    cax_c = ax_c.inset_axes([1.012, 0.02, 0.018, 0.96])
    cb = fig.colorbar(im, cax=cax_c)
    cb.set_label("margin (GW)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.0, length=2.0, width=0.5)
    ax_c.legend(
        [
            Patch(facecolor="none", edgecolor="#222222", linewidth=0.75),
            Line2D([0], [0], marker="o", mfc="white", mec="#D99A1E", mew=0.85, lw=0),
            Patch(facecolor="none", edgecolor="#4F4F4F", hatch="////", linewidth=0.35),
        ],
        ["first deficit cell", "0-2 GW warning", "sign-sensitive"],
        loc="upper center",
        bbox_to_anchor=(0.50, -0.17),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.0,
        fontsize=6.3,
    )
    for sp in ax_c.spines.values():
        sp.set_visible(False)

    # d. Generator pipeline credibility panel.
    ax_d.set_axis_off()
    pos = ax_d.get_position()
    ax_d1 = fig.add_axes([pos.x0 + 0.000, pos.y0 + 0.08 * pos.height, 0.39 * pos.width, 0.82 * pos.height])
    ax_d2 = fig.add_axes([pos.x0 + 0.45 * pos.width, pos.y0 + 0.08 * pos.height, 0.25 * pos.width, 0.82 * pos.height])
    ax_d3 = fig.add_axes([pos.x0 + 0.76 * pos.width, pos.y0 + 0.08 * pos.height, 0.18 * pos.width, 0.82 * pos.height])
    ax_d4 = fig.add_axes([pos.x0 + 0.955 * pos.width, pos.y0 + 0.08 * pos.height, 0.045 * pos.width, 0.82 * pos.height])
    pipe = read_csv("fig2/fig2c_generator_pipeline_canonical_source.csv").set_index("ISO_RTO").loc[order].reset_index()
    y = np.arange(len(pipe))
    ax_d1.hlines(y, pipe["add_eff_MW"] / 1000, pipe["MW_sum_total_active_2026_2028"] / 1000, color="#D6D1C7", lw=1.0, zorder=1)
    ax_d1.scatter(pipe["MW_sum_total_active_2026_2028"] / 1000, y, s=22, color="#C9DFC8", edgecolor="white", zorder=3, label="active")
    ax_d1.scatter(pipe["MW_sum_total_IAstage_2026_2028"] / 1000, y, s=23, marker="D", color="#E6BE42", edgecolor="white", zorder=4, label="IA-stage")
    ax_d1.scatter(pipe["MW_sum_genonly_IAstage_2026_2028"] / 1000, y, s=23, marker="s", color="#D87934", edgecolor="white", zorder=4, label="IA gen-only")
    ax_d1.scatter(pipe["add_eff_MW"] / 1000, y, s=34, color="#25A18E", edgecolor="white", zorder=5, label="effective")
    ax_d1.set_xscale("log")
    ax_d1.set_xlim(0.06, 360)
    ax_d1.set_yticks(y)
    ax_d1.set_yticklabels(order)
    ax_d1.invert_yaxis()
    ax_d1.set_xlabel("queue capacity (GW, log)")
    clean_axis(ax_d1, "x")
    panel_label(ax_d1, "d", x=-0.14, y=1.05)
    for i, r in pipe.iterrows():
        pct = 100 * float(r["effective_over_active_queue"])
        ax_d1.text(330, i, f"{pct:.1f}%" if pct >= 0.05 else "<0.1%", ha="right", va="center", fontsize=5.6, color="#777777")
    ax_d1.text(330, -0.55, "effective /\nactive queue", ha="right", va="top", fontsize=5.7, color="#777777", linespacing=0.9)

    stack_cols = [("thermal_eff_MW", "#B96A45", "thermal"), ("wind_eff_MW", "#62B0A6", "wind"), ("solar_eff_MW", "#F3B65C", "solar")]
    lefts = np.zeros(len(pipe))
    for col, color, lab in stack_cols:
        vals = pipe[col].to_numpy() / 1000
        ax_d2.barh(y, vals, left=lefts, height=0.45, color=color, edgecolor="white", linewidth=0.35, label=lab)
        lefts += vals
    stor = pipe["storage_20pct_credit_delta_MW"].to_numpy() / 1000
    ax_d2.barh(y, stor, left=lefts, height=0.45, facecolor="none", edgecolor="#8AD6C8", hatch="////", linewidth=0.8, label="storage credit")
    for i, total in enumerate(lefts):
        if total > 0.25:
            ax_d2.text(total + 0.12, i, f"{total:.1f}", va="center", ha="left", fontsize=5.8, color="#555555")
    ax_d2.set_xlim(0, 15.5)
    ax_d2.set_yticks([])
    ax_d2.set_ylim(len(pipe) - 0.5, -0.5)
    ax_d2.set_xlabel("effective additions (GW)")
    clean_axis(ax_d2, "x")

    m = pipe["Margin_2035_MW"].to_numpy() / 1000
    colors = np.where(m >= 0, "#2AA17E", COL["red_dark"])
    ax_d3.barh(y, m, height=0.42, color=colors, edgecolor="none")
    ax_d3.axvline(0, color="#777777", lw=0.65)
    for i, val in enumerate(m):
        ax_d3.text(val + (0.35 if val >= 0 else -0.35), i, f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=5.8, color=colors[i], weight="bold")
    ax_d3.set_yticks([])
    ax_d3.set_xlim(-20, 20)
    ax_d3.set_ylim(len(pipe) - 0.5, -0.5)
    ax_d3.set_xlabel("2035 margin (GW)")
    clean_axis(ax_d3, "x")

    share = pipe["top10_share_eff"].fillna(0).to_numpy()
    ax_d4.scatter(np.zeros(len(pipe)), y, s=26 + 260 * share, facecolor="#DCEFE1", edgecolor="#79B58D", lw=0.6)
    for i, s in enumerate(share):
        ax_d4.text(0.24, i, f"{100*s:.0f}%" if s > 0 else "0%", va="center", ha="left", fontsize=5.3, color="#777777")
    ax_d4.text(0, -0.55, "top-10\nshare", ha="center", va="top", fontsize=5.7, color="#4E755D", linespacing=0.9)
    ax_d4.set_xlim(-0.3, 1.0)
    ax_d4.set_ylim(len(pipe) - 0.5, -0.5)
    ax_d4.axis("off")
    ax_d1.legend(loc="upper left", bbox_to_anchor=(0.00, 1.17), ncol=4, frameon=False, handletextpad=0.35, columnspacing=0.75, fontsize=5.8)
    ax_d2.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.17), ncol=4, frameon=False, handletextpad=0.35, columnspacing=0.75, fontsize=5.8)

    # e. Trajectory ribbons.
    traj = read_csv("fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv")
    focus = ["MISO", "SPP", "PJM", "ERCOT"]
    for ax, iso in zip(e_axes, focus):
        d = traj[traj["ISO"].eq(iso)].sort_values("year")
        x = d["year"].to_numpy()
        ax.fill_between(x, d["low_growth_margin_GW"], d["high_growth_margin_GW"], color="#DDEBFA", alpha=0.55, edgecolor="none", label="AI low-high")
        ax.fill_between(x, d["headroom_extrapolation_low_GW"], d["headroom_extrapolation_high_GW"], color="#F3D7A8", alpha=0.45, edgecolor="none", label="headroom range")
        color = ISO_COLORS.get(iso, "#444444")
        ax.plot(x, d["mid_growth_margin_GW"], "-o", color=color, lw=1.5, ms=2.9, label="central trajectory")
        ax.axhline(0, color="#666666", lw=0.75)
        ymin = float(np.nanmin([d["low_growth_margin_GW"].min(), d["headroom_extrapolation_low_GW"].min(), d["mid_growth_margin_GW"].min()]))
        ymax = float(np.nanmax([d["high_growth_margin_GW"].max(), d["headroom_extrapolation_high_GW"].max(), d["mid_growth_margin_GW"].max()]))
        pad = max(1.0, 0.18 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
        yl = ax.get_ylim()
        ax.axhspan(0, yl[1], color="#EDF7EF", zorder=-2)
        ax.axhspan(yl[0], 0, color="#FBEDEA", zorder=-2)
        neg = d[d["mid_growth_margin_GW"] < 0]
        if not neg.empty:
            fy = int(neg["year"].min())
            ax.axvline(fy, color="#555555", lw=0.7, ls="--")
            ax.text(2025.3, 0.72, f"{iso}\ncross: {fy}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.4, weight="bold", color=COL["ink"])
        else:
            ax.text(2025.3, 0.72, f"{iso}\ncross: >2035", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.4, weight="bold", color=COL["ink"])
        end = float(d[d["year"].eq(2035)]["mid_growth_margin_GW"].iloc[0])
        ax.text(0.96, 0.82 if end < 0 else 0.18, f"{end:+.1f} GW", transform=ax.transAxes, ha="right", va="center", fontsize=6.7, weight="bold", color=COL["red_dark"] if end < 0 else "#1B9E77")
        ax.set_xlim(2025, 2035)
        ax.set_xticks([2025, 2030, 2035])
        clean_axis(ax, "y")
        if ax is e_axes[0]:
            ax.set_ylabel("hosting margin (GW)")
            panel_label(ax, "e", x=-0.24, y=1.06)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Year")
    e_axes[1].legend(
        [
            Patch(color="#DDEBFA", alpha=0.55),
            Patch(color="#F3D7A8", alpha=0.45),
            Line2D([0], [0], color="#555555", marker="o", lw=1.5, ms=3),
            Line2D([0], [0], color="#555555", ls="--", lw=0.8),
        ],
        ["AI low-high scenarios", "headroom extrapolation", "central trajectory", "mid first deficit"],
        loc="lower center",
        bbox_to_anchor=(1.08, -0.42),
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=1.4,
        columnspacing=0.9,
    )

    save_fig(fig, "Fig2_canonical_redrawn_20260514", "Fig2_final_tight_layout copy.pdf")


def draw_fig3() -> None:
    geo, cty, states, g_iso, iso_polys, bubbles = load_geo()
    set_style()
    fig = plt.figure(figsize=(7.10, 8.10), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.55, 1.35, 1.35],
        width_ratios=[1.02, 1.00],
        hspace=0.28,
        wspace=0.17,
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.985,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    arch = read_csv("fig3/fig3a_clustered_archetype_source_canonical.csv")
    polys = iso_polys.merge(arch[["ISO", "cluster", "margin_gw", "ai_gw", "headroom_gw"]], left_on="ISO_ASSIGNED", right_on="ISO", how="left")
    cty.plot(ax=ax_a, color="#FFFFFF", edgecolor="#F1F1F1", linewidth=0.05, zorder=0)
    states.boundary.plot(ax=ax_a, color="#DADADA", linewidth=0.25, alpha=0.55, zorder=1)
    for label, color in ARCH_COLORS.items():
        subset = polys[polys["cluster"].eq(label)]
        if not subset.empty:
            subset.plot(ax=ax_a, color=color, edgecolor="white", linewidth=0.28, alpha=0.88, zorder=4)
    g_iso.boundary.plot(ax=ax_a, color="white", linewidth=0.12, alpha=0.20, zorder=5)
    polys.boundary.plot(ax=ax_a, color="white", linewidth=1.5, alpha=0.94, zorder=6)
    polys.boundary.plot(ax=ax_a, color="#555555", linewidth=0.55, alpha=0.85, zorder=7)
    add_iso_map_labels(ax_a, polys.assign(display_val=polys["margin_gw"].fillna(0)), "display_val", suffix="GW", fs=5.1)
    ax_a.set_axis_off()
    panel_label(ax_a, "a", x=-0.03, y=1.02)
    ax_a.legend(
        [Patch(color=c) for c in ARCH_COLORS.values()],
        list(ARCH_COLORS.keys()),
        loc="lower left",
        bbox_to_anchor=(0.00, 0.00),
        frameon=True,
        framealpha=0.90,
        fontsize=5.5,
        title="mechanism archetype",
        title_fontsize=5.8,
        handlelength=0.9,
        labelspacing=0.25,
    )

    # Fingerprint matrix.
    norm = read_csv("fig3/fig3b_fingerprint_normalized_canonical.csv").set_index("ISO")
    actual = read_csv("fig3/fig3b_fingerprint_actual_values_canonical.csv").set_index("ISO")
    cols = [
        ("headroom_gw", "2025\nheadroom"),
        ("ai_gw", "AI\ngrowth"),
        ("new_supply_gw", "net\nheadroom"),
        ("margin_gw", "2035\nmargin"),
        ("ai_to_headroom", "AI /\nheadroom"),
        ("p90_beta_gw", "price\nstress"),
        ("firm_share", "firm\nshare"),
        ("vre_share", "VRE\nshare"),
    ]
    order = ISO_CLOCK_ORDER
    mat = norm.loc[order, [c[0] for c in cols]].to_numpy(dtype=float)
    im = ax_b.imshow(mat, cmap=LinearSegmentedColormap.from_list("fp", ["#F7F7F7", "#F7B267", "#C1272D"]), vmin=0, vmax=1, aspect="auto")
    ax_b.set_xticks(range(len(cols)))
    ax_b.set_xticklabels([c[1] for c in cols], rotation=0, ha="center", fontsize=5.6)
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(order)
    ax_b.tick_params(length=0)
    for i, iso in enumerate(order):
        for j, (col, _) in enumerate(cols):
            val = float(actual.loc[iso, col])
            text = f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
            ax_b.text(j, i, text, ha="center", va="center", fontsize=4.9, color="#333333")
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    ax_b.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.65)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.035, pad=0.015)
    cbar.set_label("relative exposure", fontsize=6.0)
    cbar.ax.tick_params(labelsize=5.6)
    panel_label(ax_b, "b", x=-0.10, y=1.02)

    # Archetype scatter with bubble size = deficit magnitude.
    for label, group in arch.groupby("cluster"):
        ax_c.scatter(
            group["headroom_gw"],
            group["ai_gw"],
            s=55 + 13 * np.abs(group["margin_gw"].to_numpy()),
            color=ARCH_COLORS.get(label, "#888888"),
            edgecolor="white",
            lw=0.6,
            alpha=0.90,
            label=label,
        )
        for _, r in group.iterrows():
            ax_c.text(r["headroom_gw"] + 0.25, r["ai_gw"], r["ISO"], fontsize=6.0, va="center", ha="left")
    ax_c.axline((0, 0), slope=1, color="#777777", lw=0.7, ls="--")
    ax_c.set_xlabel("2025 headroom (GW)")
    ax_c.set_ylabel("2035 AI growth (GW)")
    ax_c.set_xlim(0, max(arch["headroom_gw"]) + 3)
    ax_c.set_ylim(0, max(arch["ai_gw"]) + 4)
    clean_axis(ax_c, "both")
    panel_label(ax_c, "c", x=-0.11, y=1.05)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=4.8, handletextpad=0.25, labelspacing=0.08, markerscale=0.55)

    # Zone-level raincloud style distribution.
    z = read_csv("fig3/fig3_zone_lmp_stress_distribution_source_canonical.csv")
    rng = np.random.default_rng(15)
    data = [z[z["ISO"].eq(iso)]["beta_stress_gw"].clip(0, 180).to_numpy() for iso in order]
    parts = ax_d.violinplot(data, positions=np.arange(len(order)), vert=False, widths=0.78, showmeans=False, showextrema=False, showmedians=False)
    for body, iso in zip(parts["bodies"], order):
        body.set_facecolor(ISO_COLORS.get(iso, "#888888"))
        body.set_alpha(0.18)
        body.set_edgecolor("none")
    for i, iso in enumerate(order):
        vals = z[z["ISO"].eq(iso)]["beta_stress_gw"].clip(0, 180).to_numpy()
        jitter = rng.normal(0, 0.055, len(vals))
        ax_d.scatter(vals, i + jitter, s=9, color=ISO_COLORS.get(iso, "#888888"), alpha=0.65, edgecolor="white", lw=0.18)
        if len(vals):
            ax_d.plot(np.nanmedian(vals), i, marker="D", ms=3.6, color="#333333", mfc="white", mew=0.8)
    ax_d.set_yticks(np.arange(len(order)))
    ax_d.set_yticklabels(order)
    ax_d.set_xlabel("zone price-stress coefficient ($/MWh per GW)")
    ax_d.set_xlim(-3, 185)
    clean_axis(ax_d, "x")
    panel_label(ax_d, "d", x=-0.10, y=1.05)

    # National zone screening.
    screen = read_csv("fig3/fig3d_zone_pressure_screening_canonical.csv")
    if "screen_index" in screen.columns:
        top = screen.sort_values("screen_index", ascending=False).head(12).copy()
    else:
        top = screen.sort_values(screen.select_dtypes("number").columns[-1], ascending=False).head(12).copy()
    zone_col = "zone" if "zone" in top.columns else "label"
    iso_col = "ISO" if "ISO" in top.columns else "iso"
    val_col = "screen_index" if "screen_index" in top.columns else top.select_dtypes("number").columns[-1]
    top = top.iloc[::-1]
    y = np.arange(len(top))
    ax_e.hlines(y, 0.001, top[val_col], color=[ISO_COLORS.get(str(v), "#999999") for v in top[iso_col]], lw=1.8, alpha=0.80)
    ax_e.scatter(top[val_col], y, s=32, color=[ISO_COLORS.get(str(v), "#999999") for v in top[iso_col]], edgecolor="white", lw=0.45, zorder=4)
    labels = [f"{iso}: {short_zone_name(z)}" for iso, z in zip(top[iso_col], top[zone_col])]
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(labels, fontsize=5.3)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("combined zone stress screen")
    clean_axis(ax_e, "x")
    panel_label(ax_e, "e", x=-0.11, y=1.05)

    # PJM zones: AI additions with price-stress marker where available.
    pjm = read_csv("fig3/fig3f_pjm_zone_ai_exposure_canonical.csv")
    pjm = pjm[pjm["scenario"].eq("mid") & pjm["year"].eq(2035)].copy()
    pjm["zone_short"] = pjm["zone"].astype(str).str.title().str.replace(" Co", "", regex=False)
    pjm_top = pjm.sort_values("ai_growth_vs_mid2025_MW", ascending=False).head(12).iloc[::-1]
    yy = np.arange(len(pjm_top))
    ax_f.barh(yy, pjm_top["ai_growth_vs_mid2025_MW"] / 1000, color="#E7298A", alpha=0.70)
    ax_f.scatter(pjm_top["ai_demand_MW"] / 1000, yy, s=26, facecolor="white", edgecolor="#9C1B63", lw=0.85, zorder=4)
    ax_f.set_yticks(yy)
    ax_f.set_yticklabels([x[:27] for x in pjm_top["zone_short"]], fontsize=5.3)
    ax_f.set_xlabel("PJM zone AI demand, 2035 (GW)")
    clean_axis(ax_f, "x")
    panel_label(ax_f, "f", x=-0.10, y=1.05)
    ax_f.legend(
        [Patch(color="#E7298A", alpha=0.70), Line2D([0], [0], marker="o", lw=0, mfc="white", mec="#9C1B63")],
        ["growth since 2025", "2035 total"],
        loc="lower right",
        frameon=False,
        fontsize=5.8,
    )

    save_fig(fig, "Fig3_canonical_redrawn_20260514", "Fig3_final_dense_mechanism_composite_VECTOR copy.pdf")


def draw_fig4() -> None:
    geo, cty, states, g_iso, iso_polys, bubbles = load_geo()
    set_style()
    fig = plt.figure(figsize=(7.10, 8.10), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.08, 1.0], hspace=0.27, wspace=0.18, left=0.045, right=0.985, top=0.985, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.26)
    ax_d = fig.add_subplot(sub[0, 0])
    ax_e = fig.add_subplot(sub[1, 0])

    units = read_csv("fig4/fig4_unit_level_nuclear_candidates_canonical.csv")
    cty.plot(ax=ax_a, color="#FBFBFB", edgecolor="#EEEEEE", linewidth=0.05)
    states.boundary.plot(ax=ax_a, color="#D1D1D1", linewidth=0.28, alpha=0.65)
    iso_polys.boundary.plot(ax=ax_a, color="#777777", linewidth=0.45, alpha=0.60)
    x, y = project_lonlat(cty, units["longitude"], units["latitude"])
    path_col = {"retention": "#5ABF90", "restart": "#E5883A"}
    sizes = 35 + 230 * units["recoverable_GW"].clip(lower=0.05) / max(units["recoverable_GW"].max(), 0.1)
    for path, group in units.assign(_x=x, _y=y).groupby("pathway"):
        ax_a.scatter(group["_x"], group["_y"], s=sizes[group.index], color=path_col.get(path, "#999999"), edgecolor="white", lw=0.5, alpha=0.88, label=path, zorder=5)
    for _, r in units.sort_values("recoverable_GW", ascending=False).head(7).iterrows():
        t = ax_a.text(x[r.name] + 35000, y[r.name] + 30000, str(r["plant_name"]).split()[0], fontsize=5.0, color="#333333", zorder=8)
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white", alpha=0.96)])
    ax_a.set_axis_off()
    panel_label(ax_a, "a", x=-0.02, y=1.02)
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.90, fontsize=5.8, title="candidate pathway", title_fontsize=6.0)

    val = read_csv("fig4/fig4_unit_marginal_value_summary_canonical.csv")
    val["label"] = val["plant_name"].astype(str).str.replace(" Nuclear Generating Station", "", regex=False).str.replace(" Nuclear Power Plant", "", regex=False)
    val = val.sort_values("value_mid", ascending=False).head(14).iloc[::-1]
    yy = np.arange(len(val))
    ax_b.hlines(yy, val["value_low"], val["value_high"], color="#C9C9C9", lw=1.2)
    ax_b.scatter(val["value_mid"], yy, s=28 + 70 * val["recoverable_GW"], color=[path_col.get(p, "#999999") for p in val["pathway"]], edgecolor="white", lw=0.5, zorder=4)
    for i, r in val.iterrows():
        ax_b.text(r["value_high"] + 0.08, yy[list(val.index).index(i)], f"{int(r['available_year'])}", fontsize=5.2, va="center", color="#777777")
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels([str(x)[:21] for x in val["label"]], fontsize=5.8)
    ax_b.set_xlabel("avoided shortfall value (GW-year)")
    clean_axis(ax_b, "x")
    panel_label(ax_b, "b", x=-0.11, y=1.04)
    ax_b.legend(
        [Line2D([0], [0], marker="o", color="none", mfc=path_col["retention"], mec="white"), Line2D([0], [0], marker="o", color="none", mfc=path_col["restart"], mec="white")],
        ["retention", "restart"],
        loc="lower right",
        frameon=False,
        fontsize=5.8,
    )

    # c. Flow-style 2035 decomposition.
    off = read_csv("fig4/fig4_2035_offset_decomposition_canonical.csv")
    use = off[off["scenario"].eq("mid")].copy()
    use = use.set_index("ISO").loc[["MISO", "PJM", "SPP"]].reset_index()
    y = np.arange(len(use))
    ax_c.barh(y, use["baseline_shortfall_2035_GW"], color="#F4C6BF", edgecolor="none", height=0.54, label="baseline shortfall")
    ax_c.barh(y, -use["retention_used_mid_GW"], left=use["baseline_shortfall_2035_GW"], color="#5ABF90", edgecolor="white", linewidth=0.35, height=0.54, label="retention offset")
    ax_c.barh(y, -use["restart_used_mid_GW"], left=use["baseline_shortfall_2035_GW"] - use["retention_used_mid_GW"], color="#E5883A", edgecolor="white", linewidth=0.35, height=0.54, label="restart offset")
    ax_c.scatter(use["residual_shortfall_mid_GW"], y, s=35, color="#C43D33", edgecolor="white", zorder=5, label="residual")
    for i, r in use.iterrows():
        ax_c.text(r["baseline_shortfall_2035_GW"] + 0.35, i, f"{r['baseline_shortfall_2035_GW']:.1f}", fontsize=5.8, va="center", color="#9A4C42")
        ax_c.text(r["residual_shortfall_mid_GW"] - 0.25, i, f"{r['residual_shortfall_mid_GW']:.1f}", fontsize=5.8, va="center", ha="right", color="#C43D33", weight="bold")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(use["ISO"])
    ax_c.set_xlabel("2035 shortfall and nuclear offset (GW)")
    clean_axis(ax_c, "x")
    panel_label(ax_c, "c", x=-0.09, y=1.05)
    ax_c.legend(loc="lower right", frameon=False, ncol=2, fontsize=5.8, columnspacing=0.8)

    cap = read_csv("fig4/fig4_recoverable_nuclear_capacity_source.csv")
    piv = cap.pivot_table(index="year", columns="ISO", values="total_recoverable_GW", aggfunc="sum").fillna(0)
    piv = piv[[c for c in ["MISO", "PJM", "SPP", "NYISO", "CAISO", "ISO-NE", "ERCOT"] if c in piv.columns]]
    ax_d.stackplot(piv.index, [piv[c].to_numpy() for c in piv.columns], labels=piv.columns, colors=[ISO_COLORS.get(c, "#999999") for c in piv.columns], alpha=0.78)
    ax_d.set_ylabel("recoverable capacity (GW)")
    ax_d.set_xlim(2025, 2035)
    ax_d.set_xticks([2025, 2030, 2035])
    clean_axis(ax_d, "y")
    panel_label(ax_d, "d", x=-0.14, y=1.06)
    ax_d.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=4, frameon=False, fontsize=5.4, columnspacing=0.5)

    cases = read_csv("fig4/fig4_policy_margin_cases_canonical.csv")
    sel = cases[(cases["year"].eq(2035)) & (cases["recovery_case"].eq("mid_recovery")) & (cases["scenario"].isin(["low", "mid", "high"])) & (cases["policy_case"].isin(["baseline", "retention", "retention+restart"]))]
    sel = sel[sel["ISO"].isin(["MISO", "PJM", "SPP"])]
    labels = [f"{iso}-{sc}" for iso in ["MISO", "PJM", "SPP"] for sc in ["low", "mid", "high"]]
    mat = []
    for iso in ["MISO", "PJM", "SPP"]:
        for sc in ["low", "mid", "high"]:
            row = []
            for pc in ["baseline", "retention", "retention+restart"]:
                sub = sel[(sel["ISO"].eq(iso)) & (sel["scenario"].eq(sc)) & (sel["policy_case"].eq(pc))]
                row.append(float(sub["shortfall_GW"].iloc[0]) if len(sub) else np.nan)
            mat.append(row)
    im = ax_e.imshow(np.array(mat), cmap=LinearSegmentedColormap.from_list("short", ["#F5F5F5", "#F6A38F", "#C81D35"]), aspect="auto")
    ax_e.set_yticks(range(len(labels)))
    ax_e.set_yticklabels(labels, fontsize=5.2)
    ax_e.set_xticks([0, 1, 2])
    ax_e.set_xticklabels(["base", "retain", "retain+\nrestart"], fontsize=5.8)
    for i in range(len(labels)):
        for j in range(3):
            ax_e.text(j, i, f"{mat[i][j]:.1f}", ha="center", va="center", fontsize=4.9, color="#333333")
    for sp in ax_e.spines.values():
        sp.set_visible(False)
    panel_label(ax_e, "e", x=-0.14, y=1.06)
    cb = fig.colorbar(im, ax=ax_e, fraction=0.045, pad=0.02)
    cb.set_label("shortfall (GW)", fontsize=5.9)
    cb.ax.tick_params(labelsize=5.4)

    save_fig(fig, "Fig4_canonical_redrawn_20260514", "Fig4_intervention_audit_composite copy.pdf")


def draw_fig5() -> None:
    geo, cty, states, g_iso, iso_polys, bubbles = load_geo()
    set_style()
    fig = plt.figure(figsize=(7.10, 8.10), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.42, 1.35, 1.30], width_ratios=[1.14, 1.0], hspace=0.30, wspace=0.18, left=0.045, right=0.975, top=0.985, bottom=0.055)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    sub_c = gs[1, 1].subgridspec(2, 2, hspace=0.20, wspace=0.18)
    c_axes = [fig.add_subplot(sub_c[i, j]) for i in range(2) for j in range(2)]
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    sites = read_csv("fig5/fig5a_site_level_solar_resource_source.csv")
    sites = sites.dropna(subset=["lon", "lat", "tilted_latitude_annual_kwh_m2_day"])
    cty.plot(ax=ax_a, color="#FBF7EF", edgecolor="#EEE9DF", linewidth=0.05)
    states.boundary.plot(ax=ax_a, color="#D9D2C8", linewidth=0.28)
    iso_polys.boundary.plot(ax=ax_a, color="#86837B", linewidth=0.62, alpha=0.70)
    x, y = project_lonlat(cty, sites["lon"], sites["lat"])
    solar_cmap = LinearSegmentedColormap.from_list("solar", ["#F8E9BE", "#F1B05A", "#D4662B", "#8F3B25"])
    norm = Normalize(3.5, 6.4)
    weights = pd.to_numeric(sites["site_weight_mw"], errors="coerce").fillna(200)
    sizes = 7 + (np.sqrt(weights.clip(50, 10000)) / np.sqrt(10000)) * 130
    ax_a.scatter(x, y, s=sizes, c=sites["tilted_latitude_annual_kwh_m2_day"], cmap=solar_cmap, norm=norm, edgecolor="white", linewidth=0.35, alpha=0.88, zorder=5)
    ax_a.set_axis_off()
    panel_label(ax_a, "a", x=-0.015, y=1.01)
    cax = ax_a.inset_axes([0.905, 0.13, 0.018, 0.70])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=solar_cmap), cax=cax)
    cb.set_label("annual tilted solar\n(kWh m$^{-2}$ d$^{-1}$)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5)
    handles = []
    for val, lab in [(200, "0.2 GW"), (1000, "1 GW"), (5000, "5 GW"), (10000, "10 GW")]:
        handles.append(ax_a.scatter([], [], s=7 + (math.sqrt(val) / math.sqrt(10000)) * 130, color="#E48643", edgecolor="white", linewidth=0.35))
    ax_a.legend(handles, [h for _, h in [(200, "0.2 GW"), (1000, "1 GW"), (5000, "5 GW"), (10000, "10 GW")]], title="AI site size", loc="lower left", bbox_to_anchor=(0.03, 0.06), frameon=True, framealpha=0.90, title_fontsize=6.0, fontsize=5.7)

    # b. Fingerprint bubble table.
    fp = read_csv("fig5/fig5b_on_site_generation_fingerprint_canonical.csv").set_index("ISO")
    metrics = [
        ("solar_kwh_m2_day", "solar\nkWh/m2/d", "{:.1f}", "#D86C28"),
        ("winter_summer_solar_ratio_pct", "seasonal\nsolar ratio", "{:.0f}%", "#5B8DB8"),
        ("retail_minus_pv_lcoe_dollars_mwh", "retail-PV\n$/MWh", "{:.0f}", "#B58ACD"),
        ("p95_lmp_minus_gas_cost_dollars_mwh", "P95 LMP-\ngas $/MWh", "{:.0f}", "#E56B5D"),
        ("firm_gap_2035_GW", "firm gap\n2035", "{:.1f}", "#E0483E"),
        ("pv_100pct_4h_residual_pct", "PV+4h\nresidual", "{:.0f}%", "#C89B4E"),
        ("gas_co2_mt_mid_cf50", "gas CO2\nMt", "{:.1f}", "#8E5A3A"),
    ]
    order = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
    ax_b.set_xlim(-0.55, len(metrics) - 0.45)
    ax_b.set_ylim(len(order) - 0.5, -0.5)
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels(order)
    ax_b.set_xticks(range(len(metrics)))
    ax_b.set_xticklabels([m[1] for m in metrics], fontsize=5.6)
    ax_b.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax_b.tick_params(axis="y", length=0)
    for i, iso in enumerate(order):
        for j, (col, _, fmt, color) in enumerate(metrics):
            vals = fp[col].astype(float)
            v = float(fp.loc[iso, col])
            mn, mx = float(vals.min()), float(vals.max())
            frac = 0 if mx == mn else (v - mn) / (mx - mn)
            size = 80 + 290 * frac
            ax_b.scatter(j, i, s=size, color=color, alpha=0.70, edgecolor="#FFFFFF", lw=0.6)
            ax_b.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5, color="#2F2F2F")
    for xline in np.arange(-0.5, len(metrics), 1):
        ax_b.axvline(xline, color="#F0F0F0", lw=0.5, zorder=0)
    for yline in np.arange(-0.5, len(order), 1):
        ax_b.axhline(yline, color="#F0F0F0", lw=0.5, zorder=0)
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    panel_label(ax_b, "b", x=-0.08, y=1.14)

    # c. PV-storage residual surfaces.
    surf = read_csv("fig5/fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")
    surf = surf[(surf["scenario"].eq("mid")) & (surf["year"].eq(2035)) & (surf["storage_power_ratio_to_pv_capacity"].eq(0.5))].copy()
    focus = ["PJM", "MISO", "SPP", "ERCOT"]
    cmap_res = LinearSegmentedColormap.from_list("residual", ["#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C"])
    norm_res = Normalize(0, 100)
    im = None
    for ax, iso in zip(c_axes, focus):
        d = surf[surf["ISO_RTO"].eq(iso)]
        piv = d.pivot_table(index="storage_duration_h", columns="pv_nameplate_ratio_to_incremental_ai_peak", values="residual_firm_share_of_incremental_ai_peak")
        piv = piv.sort_index().sort_index(axis=1) * 100
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=cmap_res, norm=norm_res, shading="auto")
        ax.contour(piv.columns, piv.index, piv.values, levels=[1, 25, 50], colors=["#2F2F2F", "#777777", "#AAAAAA"], linewidths=[0.8, 0.55, 0.45])
        ax.text(0.04, 0.92, iso, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, weight="bold", color=ISO_COLORS.get(iso, "#333333"))
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 8)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticks([0, 2, 4, 6, 8])
        clean_axis(ax, None)
        if ax in [c_axes[0], c_axes[2]]:
            ax.set_ylabel("storage duration (h)", fontsize=5.8)
        else:
            ax.set_yticklabels([])
        if ax in [c_axes[2], c_axes[3]]:
            ax.set_xlabel("PV / AI peak", fontsize=5.8)
        else:
            ax.set_xticklabels([])
    panel_label(c_axes[0], "c", x=-0.24, y=1.14)
    cbar = fig.colorbar(im, ax=c_axes, fraction=0.042, pad=0.02)
    cbar.set_label("residual firm need\n(% of AI peak)", fontsize=5.8)
    cbar.ax.tick_params(labelsize=5.4)

    # d. Firm self-generation requirement.
    firm = read_csv("fig5/fig5_firm_selfgen_requirement_canonical.csv")
    d2035 = firm[firm["year"].eq(2035) & firm["ISO_RTO"].isin(["PJM", "MISO", "SPP", "ERCOT"])].copy()
    xloc = np.arange(4)
    width = 0.22
    scen_cols = {"low": "#8CC6B0", "mid": "#E6A443", "high": "#C95858"}
    for k, sc in enumerate(["low", "mid", "high"]):
        vals = d2035[d2035["scenario"].eq(sc)].set_index("ISO_RTO").loc[["PJM", "MISO", "SPP", "ERCOT"]]["firm_onsite_capacity_required_MW"] / 1000
        ax_d.bar(xloc + (k - 1) * width, vals, width=width, color=scen_cols[sc], label=sc, edgecolor="white", linewidth=0.35)
    ax_d.set_xticks(xloc)
    ax_d.set_xticklabels(["PJM", "MISO", "SPP", "ERCOT"])
    ax_d.set_ylabel("AI-attributable firm on-site need (GW)")
    clean_axis(ax_d, "y")
    panel_label(ax_d, "d", x=-0.08, y=1.05)
    ax_d.legend(loc="upper right", frameon=False, ncol=3, fontsize=5.8, handlelength=0.9, columnspacing=0.8)

    # e. Price and gas backstop screen.
    gas = read_csv("fig5/fig5_firm_selfgen_fuel_cost_emissions_canonical.csv")
    g = gas[(gas["scenario"].eq("mid")) & (gas["capacity_factor_sensitivity"].eq(0.50))]
    summary = (
        g.groupby("ISO_RTO", as_index=False)
        .agg(
            capacity_MW=("firm_onsite_capacity_required_MW", "max"),
            co2_Mt=("annual_co2_million_metric_tons", "mean"),
            cost=("variable_fuel_plus_vom_cost_dollars_mwh", "mean"),
        )
        .set_index("ISO_RTO")
    )
    summary["capacity_GW"] = summary["capacity_MW"] / 1000.0
    price = read_csv("fig5/fig5_price_cost_official_screen_with_isone.csv").set_index("ISO_RTO")
    use = summary.loc[[i for i in ["PJM", "MISO", "SPP", "ERCOT", "ISO-NE", "NYISO", "CAISO"] if i in summary.index]].copy()
    use["p95_minus_gas"] = price.loc[use.index, "p95_lmp_minus_gas_variable_cost_median"]
    x = use["p95_minus_gas"].to_numpy()
    y = use["co2_Mt"].to_numpy()
    sizes = 28 + 22 * use["capacity_GW"].to_numpy()
    ax_e.scatter(x, y, s=sizes, color=[ISO_COLORS.get(i, "#999999") for i in use.index], edgecolor="white", lw=0.6, alpha=0.88)
    for iso, r in use.iterrows():
        ax_e.text(r["p95_minus_gas"] + 1.2, r["co2_Mt"], iso, fontsize=5.8, va="center")
        ax_e.vlines(r["p95_minus_gas"], 0, r["co2_Mt"], color=ISO_COLORS.get(iso, "#999999"), alpha=0.18, lw=1.0)
    ax_e.set_xlabel("P95 LMP - gas variable cost ($/MWh)")
    ax_e.set_ylabel("gas backstop CO2 at 50% CF (Mt)")
    clean_axis(ax_e, "both")
    panel_label(ax_e, "e", x=-0.10, y=1.05)
    ax_e.legend(
        [Line2D([0], [0], marker="o", lw=0, mfc="#BBBBBB", mec="white", ms=s) for s in [4, 7, 10]],
        ["2 GW", "6 GW", "10 GW"],
        title="firm capacity",
        loc="upper left",
        frameon=False,
        fontsize=5.5,
        title_fontsize=5.8,
        handletextpad=0.5,
    )

    save_fig(fig, "Fig5_canonical_redrawn_20260514", "Fig5_onsite_generation_nature_v4_finegrid copy.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()


if __name__ == "__main__":
    main()
