import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT = Path(__file__).resolve().parents[2]
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig4_previous_design_canonical_20260514"
CURRENT = TABLE_ARCHIVE / "figures" / "Fig4_intervention_audit_composite copy.pdf"
YEARS = list(range(2025, 2036))
SCENARIOS = ["low", "mid", "high"]
RECOVERY_CASES = [("low", 0.80), ("mid", 0.90), ("high", 1.00)]
ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
DEFICIT_ISOS = ["PJM", "MISO", "SPP"]

ISO_COLORS = {
    "PJM": "#d83b8c",
    "MISO": "#7a73b7",
    "SPP": "#bc8c00",
    "ERCOT": "#df6500",
    "CAISO": "#2aa07f",
    "NYISO": "#4f80ad",
    "ISO-NE": "#7899b7",
}

PATHWAY_COLORS = {
    "retention": "#cf6a3a",
    "restart": "#7b5aa6",
    "excluded": "#c9c1b7",
    "derating": "#e8d8c6",
    "residual": "#d95749",
    "unused": "#b8d6ca",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.7,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.1,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.58,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 700,
        "pdf.compression": 9,
    })


def clean_spines(ax, left=True, bottom=True):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#58524d")
        ax.spines[side].set_linewidth(0.58)


def save(fig, stem, out_dir=OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    rasterize_non_text(fig)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)
    return png, pdf


def rasterize_non_text(fig):
    """Keep text editable while flattening dense data/map layers for Illustrator.

    Illustrator becomes very slow when county geometries, alluvial ribbons,
    stream polygons and dense markers are preserved as individual vector
    objects.  Rasterizing non-text artists keeps typography and labels
    editable, but collapses the heavy visual layers into high-resolution image
    tiles inside the PDF.
    """
    for ax in fig.axes:
        for artist in list(ax.collections) + list(ax.patches) + list(ax.lines) + list(ax.images):
            if hasattr(artist, "set_rasterized"):
                artist.set_rasterized(True)

        # Put all typography above rasterized layers and keep it vector.
        text_artists = list(ax.texts)
        text_artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        text_artists.extend(ax.get_xticklabels())
        text_artists.extend(ax.get_yticklabels())
        for text in text_artists:
            if hasattr(text, "set_rasterized"):
                text.set_rasterized(False)
            text.set_zorder(100)

        legend = ax.get_legend()
        if legend is not None:
            legend.set_zorder(100)
            for text in legend.get_texts():
                text.set_rasterized(False)
                text.set_zorder(101)


def load_all():
    gap = pd.read_csv(TABLE / "fig4" / "fig4_baseline_margin_canonical_input.csv")
    units = pd.read_csv(TABLE / "fig4" / "fig4_unit_level_nuclear_candidates_canonical.csv")
    rec = pd.read_csv(TABLE / "fig4" / "fig4_recoverable_nuclear_capacity_source.csv")
    cases = gap[["ISO", "year", "scenario", "baseline_margin_GW"]].copy()
    unit_long = pd.read_csv(TABLE / "fig4" / "fig4_unit_marginal_value_long_canonical.csv")
    unit_summary = pd.read_csv(TABLE / "fig4" / "fig4_unit_marginal_value_summary_canonical.csv")
    unit_summary["unit_label"] = (
        unit_summary["plant_name"]
        .astype(str)
        .str.replace(" Nuclear Generating Station", "", regex=False)
        .str.replace(" Nuclear Power Plant", "", regex=False)
        .str.replace(" Nuclear Station", "", regex=False)
        .str.replace(" Nuclear Facility", "", regex=False)
        .str.replace(" Clean Energy Center", "", regex=False)
        .str.replace(" Generating Station", "", regex=False)
        + " " + unit_summary["unit"].astype(str).replace({"1": "1", "2": "2"})
    )
    return gap, units, rec, cases, unit_long, unit_summary


def panel_a_opportunity_atlas(units, cases):
    counties = gpd.read_file(PROJECT / "data" / "us_counties.geojson")
    counties = counties[~counties["STATE"].isin(["02", "15", "72", "60", "66", "69", "78"])].to_crs(5070)
    regions = gpd.read_file(PROJECT / "out" / "iso7_regions_region_only_clean.gpkg").to_crs(5070)
    regions = regions.dissolve(by="ISO").reset_index()
    regions["ISO"] = regions["ISO"].replace({"ISONE": "ISO-NE", "ISO_NE": "ISO-NE"})

    m2035 = cases[(cases["scenario"].eq("mid")) & (cases["year"].eq(2035))][["ISO", "baseline_margin_GW"]]
    regions = regions.merge(m2035, on="ISO", how="left")
    cmap = LinearSegmentedColormap.from_list("margin_map", ["#c84c3e", "#f7eee5", "#49a37f"])
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=18)

    op_path = PROJECT / "Final Figs" / "fig4a_nuclear_fleet_pipeline_bubble_map_operating_source.csv"
    fleet = pd.read_csv(op_path)
    fleet["ISO"] = fleet["ISO"].replace({"ISONE": "ISO-NE", "ISO_NE": "ISO-NE"})
    fleet_gdf = gpd.GeoDataFrame(
        fleet,
        geometry=gpd.points_from_xy(fleet["Longitude"], fleet["Latitude"]),
        crs=4326,
    ).to_crs(5070)

    unit_gdf = gpd.GeoDataFrame(
        units.copy(),
        geometry=gpd.points_from_xy(units["longitude"], units["latitude"]),
        crs=4326,
    ).to_crs(5070)

    grouped = []
    for (plant, iso), g in unit_gdf.groupby(["plant_name", "ISO"]):
        geom = g.geometry.iloc[0]
        grouped.append({
            "plant_name": plant,
            "ISO": iso,
            "x": geom.x,
            "y": geom.y,
            "raw_GW": g["raw_capacity_GW"].sum(),
            "ret_GW": g.loc[g["pathway"].eq("retention"), "recoverable_GW"].sum(),
            "restart_GW": g.loc[g["pathway"].eq("restart"), "recoverable_GW"].sum(),
            "year": int(g["available_year"].min()),
        })
    opp = pd.DataFrame(grouped)

    planned_path = PROJECT / "Final Figs" / "fig4a_nuclear_fleet_pipeline_bubble_map_planned_source.csv"
    planned = pd.read_csv(planned_path) if planned_path.exists() else pd.DataFrame()
    planned_gdf = None
    if not planned.empty and {"Longitude", "Latitude"}.issubset(planned.columns):
        planned_gdf = gpd.GeoDataFrame(
            planned,
            geometry=gpd.points_from_xy(planned["Longitude"], planned["Latitude"]),
            crs=4326,
        ).to_crs(5070)

    fig, ax = plt.subplots(figsize=(4.62, 2.40))
    counties.plot(ax=ax, facecolor="#fbfaf7", edgecolor="#e6e0d8", linewidth=0.095, zorder=0)
    for _, r in regions.iterrows():
        color = cmap(norm(r["baseline_margin_GW"])) if pd.notna(r["baseline_margin_GW"]) else "#eeeeee"
        gpd.GeoDataFrame([r], geometry="geometry", crs=regions.crs).plot(
            ax=ax, facecolor=color, edgecolor="#71736d", linewidth=0.56, alpha=0.78, zorder=2
        )
    regions.boundary.plot(ax=ax, color="white", linewidth=0.22, alpha=0.9, zorder=3)

    fleet_sizes = 5 + 18 * np.sqrt(fleet_gdf["operating_capacity_GW"].fillna(0.2).clip(lower=0.05))
    ax.scatter(
        fleet_gdf.geometry.x, fleet_gdf.geometry.y,
        s=fleet_sizes,
        facecolor="#a7b0ad", edgecolor="white",
        linewidth=0.35, alpha=0.36, zorder=4,
    )

    for _, r in opp.iterrows():
        total = r["ret_GW"] + r["restart_GW"]
        if total <= 0 and r["raw_GW"] <= 0:
            continue
        radius = 15500 + 20000 * math.sqrt(max(r["raw_GW"], 0.05) / 1.5)
        ax.add_patch(Circle((r["x"], r["y"]), radius * 1.10, facecolor="none", edgecolor="white",
                            linewidth=1.15, alpha=0.96, zorder=5))
        ax.add_patch(Circle((r["x"], r["y"]), radius, facecolor="white", edgecolor="#2f2a26",
                            linewidth=0.70, alpha=0.98, zorder=6))
        if total > 0:
            theta = 90.0
            ret_angle = 360.0 * r["ret_GW"] / total if total else 0.0
            if r["ret_GW"] > 0:
                ax.add_patch(Wedge((r["x"], r["y"]), radius, theta, theta + ret_angle,
                                   width=radius * 0.56, facecolor=PATHWAY_COLORS["retention"],
                                   edgecolor="white", linewidth=0.42, zorder=7))
            if r["restart_GW"] > 0:
                ax.add_patch(Wedge((r["x"], r["y"]), radius, theta + ret_angle, theta + 360,
                                   width=radius * 0.56, facecolor=PATHWAY_COLORS["restart"],
                                   edgecolor="white", linewidth=0.42, zorder=7))
            ax.add_patch(Circle((r["x"], r["y"]), radius, facecolor="none", edgecolor="#2f2a26",
                                linewidth=0.48, alpha=0.96, zorder=7.5))
            ax.add_patch(Circle((r["x"], r["y"]), radius * 0.42, facecolor="white",
                                edgecolor="none", alpha=0.98, zorder=7.6))
        if r["raw_GW"] > 0:
            ax.text(r["x"], r["y"], f"{r['year']%100:02d}", ha="center", va="center",
                    fontsize=4.5, fontweight="bold", color="#333333", zorder=8)

    if planned_gdf is not None and not planned_gdf.empty:
        ax.scatter(planned_gdf.geometry.x, planned_gdf.geometry.y, marker="D", s=32,
                   facecolor="#2f9d80", edgecolor="white", linewidth=0.5, zorder=8)

    label_offsets = {
        "PJM": (40000, -35000),
        "MISO": (50000, 0),
        "SPP": (-105000, 0),
        "ERCOT": (20000, -70000),
        "CAISO": (-28000, -28000),
        "NYISO": (-52000, 8000),
        "ISO-NE": (95000, 85000),
    }
    region_index = regions.set_index("ISO")
    for iso, (dx, dy) in label_offsets.items():
        if iso not in region_index.index:
            continue
        r = region_index.loc[iso]
        pt = r.geometry.representative_point()
        txt = f"{iso}\n{r['baseline_margin_GW']:+.1f} GW"
        t = ax.text(pt.x + dx, pt.y + dy, txt, ha="center", va="center",
                    fontsize=5.55, fontweight="bold", color="#333333", zorder=10)
        t.set_path_effects([pe.withStroke(linewidth=1.65, foreground="white", alpha=0.96)])

    xmin, ymin, xmax, ymax = counties.total_bounds
    ax.set_xlim(xmin - (xmax - xmin) * 0.025, xmax + (xmax - xmin) * 0.025)
    ax.set_ylim(ymin - (ymax - ymin) * 0.035, ymax + (ymax - ymin) * 0.035)
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#a7b0ad",
               markeredgecolor="white", markersize=3.7, alpha=0.55, label="operating fleet"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["retention"],
               markeredgecolor="#2f2a26", markeredgewidth=0.95, markersize=5.8, label="retention"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["restart"],
               markeredgecolor="#2f2a26", markeredgewidth=0.95, markersize=5.8, label="restart"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#2f9d80",
               markeredgecolor="white", markersize=3.9, label="planned"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, facecolor="white",
              edgecolor="#ded8d1", framealpha=0.95, ncol=2, borderpad=0.35,
              handletextpad=0.30, columnspacing=0.60, fontsize=4.8)
    ax.text(0.002, 0.988, "a", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.8, fontweight="bold", color="black", zorder=20)
    cax = fig.add_axes([0.835, 0.315, 0.015, 0.31])
    cax.set_xlim(0, 1)
    cax.set_ylim(-10, 18)
    vals = np.linspace(-10, 18, 64)
    for lo, hi in zip(vals[:-1], vals[1:]):
        cax.add_patch(Rectangle((0, lo), 1, hi - lo, facecolor=cmap(norm((lo + hi) / 2)), edgecolor="none"))
    cax.set_xticks([])
    cax.set_yticks([-10, 0, 10, 18])
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", labelsize=4.9, length=1.8, pad=1)
    cax.set_title("margin\n(GW)", fontsize=4.8, pad=2.0)
    for spine in cax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#3f3934")
    return fig


def panel_b_swimmer(unit_summary, unit_long):
    df = unit_summary.sort_values("value_mid", ascending=False).head(14).copy()
    df = df.sort_values(["ISO", "available_year", "value_mid"], ascending=[True, True, False]).reset_index(drop=True)
    order = {iso: i for i, iso in enumerate(ISO_ORDER)}
    df["iso_order"] = df["ISO"].map(order).fillna(99)
    df = df.sort_values(["iso_order", "available_year", "value_mid"], ascending=[True, True, False]).reset_index(drop=True)
    y = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(2.62, 2.40))
    ax.set_xlim(2024.8, 2037.7)
    for i, (_, r) in enumerate(df.iterrows()):
        yy = y[i]
        iso_color = ISO_COLORS.get(r["ISO"], "#777777")
        start = int(r["available_year"])
        lw = 1.5 + 2.6 * math.sqrt(max(r["recoverable_GW"], 0) / 1.5)
        if r["pathway"] == "restart":
            ax.plot([2025, start], [yy, yy], color="#cfc8be", lw=0.9, linestyle=(0, (2, 1)), zorder=1)
            ax.plot([start, 2035], [yy, yy], color=PATHWAY_COLORS["restart"], lw=lw, solid_capstyle="round", zorder=3)
            marker = "^"
        else:
            ax.plot([2025, start], [yy, yy], color="#cfc8be", lw=0.9, zorder=1)
            ax.plot([start, 2035], [yy, yy], color=PATHWAY_COLORS["retention"], lw=lw, solid_capstyle="round", zorder=3)
            marker = "o"
        ax.scatter(start, yy, s=16 + 45 * math.sqrt(max(r["recoverable_GW"], 0) / 1.5),
                   marker=marker, facecolor=iso_color, edgecolor="white", linewidth=0.45, zorder=5)
        ax.plot([2035.85, 2035.85 + r["value_mid"] * 0.34], [yy, yy], color=iso_color,
                lw=1.8, alpha=0.72, solid_capstyle="round", zorder=4)
        ax.text(2035.85 + r["value_mid"] * 0.34 + 0.06, yy, f"{r['value_mid']:.1f}",
                va="center", ha="left", fontsize=4.9, color=iso_color)
        name = str(r["unit_label"])
        if len(name) > 16:
            name = name[:14] + "..."
        ax.text(2024.68, yy, f"{name}", ha="right", va="center", fontsize=5.0, color="#333333")
        ax.scatter(2024.84, yy, marker="s", s=10.5, color=iso_color, clip_on=False)

    ax.set_yticks([])
    ax.set_xticks([2025, 2028, 2031, 2035])
    ax.grid(axis="x", color="#eee5dc", lw=0.7)
    ax.axvspan(2035.45, 2037.65, color="#f8f3ed", zorder=0)
    ax.text(2035.92, len(df) - 0.15, "avoided\nGW-yr", ha="left", va="top", fontsize=4.7, color="#5a5149")
    ax.text(-0.10, 1.04, "b", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color="black")
    ax.set_xlabel("available year", fontsize=5.9, labelpad=1.5)
    clean_spines(ax, left=False)
    ax.tick_params(axis="x", labelsize=5.4)
    ax.tick_params(axis="y", length=0)
    return fig


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.82, lw=0.0):
    dx = (x1 - x0) * 0.48
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none",
                           alpha=alpha, lw=lw, zorder=2))


def panel_c_alluvial(units, rec, offset_df):
    rec2035 = rec[rec["year"].eq(2035)].copy()
    raw_ret = float(units.loc[units["pathway"].eq("retention"), "raw_capacity_GW"].sum())
    raw_restart = float(units.loc[units["pathway"].eq("restart"), "raw_capacity_GW"].sum())
    ret_rec = float(rec2035["retention_recoverable_GW"].sum())
    restart_rec = float(rec2035["restart_recoverable_GW"].sum())
    ret_eff = ret_rec * 0.90
    restart_eff = restart_rec * 0.90
    used_ret = float(offset_df["retention_used_mid_GW"].sum())
    used_restart = float(offset_df["restart_used_mid_GW"].sum())
    used_iso = offset_df.set_index("ISO")["offset_mid_GW"].to_dict()
    unused_ret = max(0.0, ret_eff - used_ret)
    unused_restart = max(0.0, restart_eff - used_restart)

    screened = max(0.0, raw_ret + raw_restart - ret_rec - restart_rec)
    derate = (ret_rec + restart_rec) * 0.10
    miso_offset = used_iso.get("MISO", 0.0)
    pjm_offset = used_iso.get("PJM", 0.0)
    spp_offset = used_iso.get("SPP", 0.0)
    not_binding = unused_ret + unused_restart
    total_raw = raw_ret + raw_restart
    colors = {
        "retention candidates": PATHWAY_COLORS["retention"],
        "restart candidates": PATHWAY_COLORS["restart"],
        "recoverable retention": PATHWAY_COLORS["retention"],
        "recoverable restart": PATHWAY_COLORS["restart"],
        "screened out": PATHWAY_COLORS["excluded"],
        "effective retention": "#d17d55",
        "effective restart": "#8e70b0",
        "recovery derate": PATHWAY_COLORS["derating"],
        "MISO offset": ISO_COLORS["MISO"],
        "PJM offset": ISO_COLORS["PJM"],
        "SPP offset": ISO_COLORS["SPP"],
        "not binding by 2035": PATHWAY_COLORS["unused"],
    }

    stages = [
        [("retention candidates", raw_ret), ("restart candidates", raw_restart)],
        [("recoverable retention", ret_rec), ("recoverable restart", restart_rec), ("screened out", screened)],
        [("effective retention", ret_eff), ("effective restart", restart_eff), ("recovery derate", derate), ("screened out", screened)],
        [("MISO offset", miso_offset), ("PJM offset", pjm_offset), ("SPP offset", spp_offset),
         ("not binding by 2035", not_binding), ("recovery derate", derate), ("screened out", screened)],
    ]
    flows = [
        (0, "retention candidates", 1, "recoverable retention", ret_rec, PATHWAY_COLORS["retention"], 0.70),
        (0, "retention candidates", 1, "screened out", max(0.0, raw_ret - ret_rec), PATHWAY_COLORS["excluded"], 0.42),
        (0, "restart candidates", 1, "recoverable restart", restart_rec, PATHWAY_COLORS["restart"], 0.70),
        (0, "restart candidates", 1, "screened out", max(0.0, raw_restart - restart_rec), PATHWAY_COLORS["excluded"], 0.38),
        (1, "recoverable retention", 2, "effective retention", ret_eff, "#d17d55", 0.72),
        (1, "recoverable retention", 2, "recovery derate", ret_rec * 0.10, PATHWAY_COLORS["derating"], 0.52),
        (1, "recoverable restart", 2, "effective restart", restart_eff, "#8e70b0", 0.72),
        (1, "recoverable restart", 2, "recovery derate", restart_rec * 0.10, PATHWAY_COLORS["derating"], 0.52),
        (1, "screened out", 2, "screened out", screened, PATHWAY_COLORS["excluded"], 0.38),
        (2, "effective retention", 3, "MISO offset", float(offset_df["retention_used_mid_GW"].sum() if offset_df.empty else offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("MISO", 0.0)), ISO_COLORS["MISO"], 0.74),
        (2, "effective retention", 3, "PJM offset", float(offset_df["retention_used_mid_GW"].sum() if offset_df.empty else offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("PJM", 0.0)), ISO_COLORS["PJM"], 0.74),
        (2, "effective retention", 3, "SPP offset", float(offset_df["retention_used_mid_GW"].sum() if offset_df.empty else offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("SPP", 0.0)), ISO_COLORS["SPP"], 0.74),
        (2, "effective restart", 3, "PJM offset", float(offset_df["restart_used_mid_GW"].sum() if offset_df.empty else offset_df.set_index("ISO").get("restart_used_mid_GW", pd.Series()).get("PJM", 0.0)), ISO_COLORS["PJM"], 0.64),
        (2, "effective retention", 3, "not binding by 2035", unused_ret, PATHWAY_COLORS["unused"], 0.52),
        (2, "effective restart", 3, "not binding by 2035", unused_restart, PATHWAY_COLORS["unused"], 0.52),
        (2, "recovery derate", 3, "recovery derate", derate, PATHWAY_COLORS["derating"], 0.50),
        (2, "screened out", 3, "screened out", screened, PATHWAY_COLORS["excluded"], 0.36),
    ]

    fig, ax = plt.subplots(figsize=(3.05, 2.08))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    xs = [0.075, 0.365, 0.650, 0.900]
    node_w = 0.030
    top = 0.86
    usable_h = 0.73
    gap = 0.012
    scale = (usable_h - gap * 4) / total_raw

    node_spans = []
    for stage in stages:
        y = top
        spans = {}
        for name, val in stage:
            if val <= 0.001:
                continue
            h = max(0.006, val * scale)
            spans[name] = [y - h, y]
            y -= h + gap
        node_spans.append(spans)

    out_cursor = [{k: v[0] for k, v in spans.items()} for spans in node_spans]
    in_cursor = [{k: v[0] for k, v in spans.items()} for spans in node_spans]

    def take(cursor, stage_i, name, val):
        if val <= 0.001 or name not in cursor[stage_i]:
            return None
        h = max(0.004, val * scale)
        lo = cursor[stage_i][name]
        cursor[stage_i][name] = lo + h
        return lo, lo + h

    for s0, n0, s1, n1, val, col, alpha in flows:
        if val <= 0.001:
            continue
        a = take(out_cursor, s0, n0, val)
        b = take(in_cursor, s1, n1, val)
        if a is None or b is None:
            continue
        ribbon(ax, xs[s0] + node_w / 2, xs[s1] - node_w / 2,
               a[0], a[1], b[0], b[1], col, alpha=alpha)

    for i, stage in enumerate(stages):
        ax.text(xs[i], 0.965, ["candidate", "recoverable", "effective", "2035 use"][i],
                ha="center", va="top", fontsize=5.7, fontweight="bold", color="#333333")
        for name, val in stage:
            if val <= 0.001 or name not in node_spans[i]:
                continue
            lo, hi = node_spans[i][name]
            ax.add_patch(Rectangle((xs[i] - node_w / 2, lo), node_w, hi - lo,
                                   facecolor=colors[name], edgecolor="white",
                                   lw=0.40, alpha=0.98, zorder=5))
            if hi - lo > 0.040:
                ax.text(xs[i], (lo + hi) / 2, f"{val:.1f}", ha="center", va="center",
                        fontsize=4.7, color="white", fontweight="bold", rotation=90, zorder=6)

    left_labels = [("retention candidates", "retention"), ("restart candidates", "restart")]
    for name, lab in left_labels:
        lo, hi = node_spans[0].get(name, (None, None))
        if lo is None:
            continue
        ax.text(xs[0] - 0.030, (lo + hi) / 2, lab, ha="right", va="center",
                fontsize=5.1, color="#4f4841")

    right_labels = [
        ("MISO offset", "MISO"),
        ("PJM offset", "PJM"),
        ("SPP offset", "SPP"),
        ("not binding by 2035", "not binding"),
        ("recovery derate", "derate"),
        ("screened out", "screened out"),
    ]
    for name, lab in right_labels:
        lo, hi = node_spans[-1].get(name, (None, None))
        if lo is None:
            continue
        ax.text(xs[-1] + 0.030, (lo + hi) / 2, lab, ha="left", va="center",
                fontsize=4.9, color="#4f4841")

    ax.text(0.020, 0.045, "same 19.4 GW candidate denominator; mid recovery = 0.90",
            ha="left", va="bottom", fontsize=4.7, color="#6a625a")
    ax.text(-0.045, 1.01, "c", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color="black")
    return fig


def compute_offset_df(cases, rec):
    rec2035 = rec[rec["year"].eq(2035)].set_index("ISO")
    rows = []
    for iso in DEFICIT_ISOS:
        ret_raw = float(rec2035.loc[iso, "retention_recoverable_GW"]) if iso in rec2035.index else 0.0
        restart_raw = float(rec2035.loc[iso, "restart_recoverable_GW"]) if iso in rec2035.index else 0.0
        mid_row = cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(2035)) & (cases["scenario"].eq("mid"))]
        baseline_mid = max(0.0, -float(mid_row["baseline_margin_GW"].iloc[0]))
        retention_mid = ret_raw * 0.90
        restart_mid = restart_raw * 0.90
        retention_used = min(baseline_mid, retention_mid)
        restart_used = min(max(0.0, baseline_mid - retention_used), restart_mid)
        residual_mid = max(0.0, baseline_mid - retention_used - restart_used)
        rows.append({
            "ISO": iso,
            "baseline_mid_GW": baseline_mid,
            "retention_used_mid_GW": retention_used,
            "restart_used_mid_GW": restart_used,
            "offset_mid_GW": retention_used + restart_used,
            "residual_mid_GW": residual_mid,
        })
    return pd.DataFrame(rows)


def panel_d_mirror_stream(cases, rec):
    rec_by = rec.set_index(["ISO", "year"])
    mid = cases[cases["scenario"].eq("mid")].copy()
    years = np.array(YEARS)
    baseline = {}
    residual = {}
    low_total = []
    high_total = []
    residual_low_total = []
    residual_high_total = []
    for iso in DEFICIT_ISOS:
        b = []
        rmid = []
        for year in YEARS:
            row = mid[(mid["ISO"].eq(iso)) & (mid["year"].eq(year))]
            base_margin = float(row["baseline_margin_GW"].iloc[0])
            ret = 0.0
            res = 0.0
            if (iso, year) in rec_by.index:
                recrow = rec_by.loc[(iso, year)]
                ret = float(recrow["retention_recoverable_GW"])
                res = float(recrow["restart_recoverable_GW"])
            b.append(max(0.0, -base_margin))
            rmid.append(max(0.0, -(base_margin + 0.90 * (ret + res))))
        baseline[iso] = np.array(b)
        residual[iso] = np.array(rmid)

    for year in YEARS:
        sub = cases[cases["year"].eq(year)]
        vals_base = []
        vals_resid = []
        for scenario in SCENARIOS:
            total_base = 0.0
            total_resid = []
            for ratio_name, ratio in RECOVERY_CASES:
                total_r = 0.0
                for iso in DEFICIT_ISOS:
                    row = sub[(sub["ISO"].eq(iso)) & (sub["scenario"].eq(scenario))]
                    if row.empty:
                        continue
                    margin = float(row["baseline_margin_GW"].iloc[0])
                    recrow = rec_by.loc[(iso, year)] if (iso, year) in rec_by.index else None
                    recoverable = 0.0 if recrow is None else float(recrow["retention_recoverable_GW"] + recrow["restart_recoverable_GW"])
                    total_r += max(0.0, -(margin + ratio * recoverable))
                    if ratio_name == "mid":
                        total_base += max(0.0, -margin)
                total_resid.append(total_r)
            vals_base.append(total_base)
            vals_resid.extend(total_resid)
        low_total.append(min(vals_base))
        high_total.append(max(vals_base))
        residual_low_total.append(min(vals_resid))
        residual_high_total.append(max(vals_resid))

    fig, ax = plt.subplots(figsize=(2.36, 2.12))
    y_top = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = baseline[iso]
        ax.fill_between(years, y_top, y_top + vals, color=ISO_COLORS[iso], alpha=0.56,
                        edgecolor="white", linewidth=0.35, label=f"{iso} baseline")
        y_top = y_top + vals
    y_bot = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = residual[iso]
        ax.fill_between(years, -y_bot, -(y_bot + vals), color=ISO_COLORS[iso], alpha=0.30,
                        edgecolor="white", linewidth=0.35)
        y_bot = y_bot + vals
    ax.plot(years, high_total, color="#7d736b", lw=0.75, linestyle=(0, (2, 1)), alpha=0.82)
    ax.plot(years, low_total, color="#7d736b", lw=0.75, linestyle=(0, (2, 1)), alpha=0.82)
    ax.plot(years, -np.array(residual_high_total), color="#2f8170", lw=0.75, linestyle=(0, (2, 1)), alpha=0.82)
    ax.plot(years, -np.array(residual_low_total), color="#2f8170", lw=0.75, linestyle=(0, (2, 1)), alpha=0.82)
    ax.axhline(0, color="#5b544d", lw=0.7)
    ax.set_xlim(2025, 2035)
    ymax = max(max(high_total), max(residual_high_total), 1)
    ax.set_ylim(-ymax * 1.15, ymax * 1.18)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(["20", "10", "0", "10", "20"])
    ax.set_xlabel("year", fontsize=5.9, labelpad=1.5)
    ax.set_ylabel("GW", fontsize=5.8)
    ax.grid(axis="x", color="#eee5dc", lw=0.7)
    clean_spines(ax)
    ax.tick_params(axis="both", labelsize=5.4)
    ax.text(0.01, 0.96, "baseline deficit", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.2, color="#6a443f")
    ax.text(0.01, 0.055, "residual with nuclear", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.2, color="#2f8170")
    ax.text(-0.19, 1.045, "d", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color="black")
    handles = [
        Line2D([0], [0], color=ISO_COLORS["PJM"], lw=4, alpha=0.56, label="PJM"),
        Line2D([0], [0], color=ISO_COLORS["MISO"], lw=4, alpha=0.56, label="MISO"),
        Line2D([0], [0], color=ISO_COLORS["SPP"], lw=4, alpha=0.56, label="SPP"),
        Line2D([0], [0], color="#7d736b", lw=0.85, linestyle=(0, (2, 1)), label="AI x recovery envelope"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.54, -0.28),
              ncol=2, handlelength=1.05, columnspacing=0.55, handletextpad=0.30,
              borderaxespad=0.0, fontsize=4.5)
    return fig


def frontier_curve(cases, rec, iso, scenario, ratios):
    row = cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(2035)) & (cases["scenario"].eq(scenario))]
    if row.empty:
        return pd.DataFrame()
    margin = float(row["baseline_margin_GW"].iloc[0])
    recrow = rec[(rec["ISO"].eq(iso)) & (rec["year"].eq(2035))]
    recoverable = 0.0
    if not recrow.empty:
        recoverable = float(recrow["retention_recoverable_GW"].iloc[0] + recrow["restart_recoverable_GW"].iloc[0])
    rows = [{"capacity_GW": 0.0, "residual_shortfall_GW": max(0.0, -margin), "ratio": 0.0}]
    for ratio in ratios:
        rows.append({
            "capacity_GW": recoverable * ratio,
            "residual_shortfall_GW": max(0.0, -(margin + recoverable * ratio)),
            "ratio": ratio,
        })
    return pd.DataFrame(rows)


def first_deficit(cases, rec, iso, scenario, ratio):
    sub = cases[(cases["ISO"].eq(iso)) & (cases["scenario"].eq(scenario))].copy().sort_values("year")
    rec_by = rec.set_index(["ISO", "year"])
    for _, r in sub.iterrows():
        recov = 0.0
        key = (iso, int(r["year"]))
        if key in rec_by.index:
            rr = rec_by.loc[key]
            recov = float(rr["retention_recoverable_GW"] + rr["restart_recoverable_GW"])
        if r["baseline_margin_GW"] + ratio * recov < 0:
            return int(r["year"])
    return 2036


def panel_e_policy_frontier(cases, rec):
    fig, ax = plt.subplots(figsize=(2.36, 2.12))
    ratios = [0.80, 0.90, 1.00]
    for iso in DEFICIT_ISOS:
        color = ISO_COLORS[iso]
        curves = {sc: frontier_curve(cases, rec, iso, sc, ratios) for sc in SCENARIOS}
        mid = curves["mid"]
        all_points = pd.concat(curves.values(), ignore_index=True)
        for cap in sorted(all_points["capacity_GW"].unique()):
            sub = all_points[np.isclose(all_points["capacity_GW"], cap)]
            if len(sub) > 1:
                ax.plot([cap, cap], [sub["residual_shortfall_GW"].min(), sub["residual_shortfall_GW"].max()],
                        color=color, lw=4.0, alpha=0.13, solid_capstyle="round", zorder=1)
        ax.plot(mid["capacity_GW"], mid["residual_shortfall_GW"], color=color, lw=1.8,
                marker="o", markersize=4.0, markeredgecolor="white", markeredgewidth=0.45,
                label=iso, zorder=4)
        ax.fill_between(mid["capacity_GW"], 0, mid["residual_shortfall_GW"],
                        color=color, alpha=0.045, zorder=0)
        label_rows = mid[mid["ratio"].eq(0.90)]
        for _, r in label_rows.iterrows():
            fd = first_deficit(cases, rec, iso, "mid", float(r["ratio"]))
            txt = ">35" if fd > 2035 else str(fd)
            ax.text(r["capacity_GW"], r["residual_shortfall_GW"] + 0.38, txt,
                    ha="center", va="bottom", fontsize=5.5, color=color)
        end = mid.iloc[-1]
        ax.text(end["capacity_GW"] + 0.12, end["residual_shortfall_GW"], iso,
                va="center", ha="left", color=color, fontweight="bold", fontsize=6.6)

    ax.set_xlim(-0.08, 3.65)
    ax.set_ylim(-0.25, 15.4)
    ax.set_xlabel("credited capacity (GW)", fontsize=5.9, labelpad=1.5)
    ax.set_ylabel("GW", fontsize=5.8)
    ax.grid(color="#eee5dc", lw=0.75)
    clean_spines(ax)
    ax.tick_params(axis="both", labelsize=5.4)
    ax.text(0.012, 0.96, "labels = first deficit year",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.1, color="#5a5149")
    ax.text(-0.18, 1.045, "e", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color="black")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.015, 0.03),
              ncol=1, handlelength=1.20, columnspacing=0.5, borderaxespad=0.0,
              fontsize=4.8)
    return fig


def compose_pdf(panel_paths: dict[str, Path]) -> Path:
    build = Path("/private/tmp/fig4_previous_design_hybrid")
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    for label, path in panel_paths.items():
        shutil.copy2(path, build / f"{label}.pdf")
    tex = build / "fig4_vector_composite.tex"
    tex.write_text(
        r"""\documentclass{article}
\usepackage[
  paperwidth=7.1in,
  paperheight=5.05in,
  left=0.035in,
  right=0.025in,
  top=0.025in,
  bottom=0.025in
]{geometry}
\usepackage{graphicx}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\fboxsep}{0pt}
\newcommand{\panelgraphic}[3]{\includegraphics[width=#1]{#3}}
\begin{document}
\panelgraphic{4.45in}{a}{a.pdf}\hspace{0.10in}%
\panelgraphic{2.36in}{b}{b.pdf}\\[-0.03in]
\panelgraphic{2.80in}{c}{c.pdf}\hspace{0.11in}%
\panelgraphic{2.03in}{d}{d.pdf}\hspace{0.08in}%
\panelgraphic{1.98in}{e}{e.pdf}
\end{document}
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex.name],
        cwd=build,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = OUT / "Fig4_intervention_audit_composite_HYBRID.pdf"
    shutil.copy2(build / "fig4_vector_composite.pdf", out)
    shutil.copy2(out, CURRENT)
    return out


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    gap, units, rec, cases, unit_long, unit_summary = load_all()
    offset_df = compute_offset_df(cases, rec)

    panel_pdfs = {}
    fig = panel_a_opportunity_atlas(units, cases)
    panel_pdfs["a"] = save(fig, "fig4a_opportunity_atlas_glyph_map")[1]
    fig = panel_b_swimmer(unit_summary, unit_long)
    panel_pdfs["b"] = save(fig, "fig4b_unit_retirement_restart_swimmer")[1]
    fig = panel_c_alluvial(units, rec, offset_df)
    panel_pdfs["c"] = save(fig, "fig4c_capacity_screen_alluvial")[1]
    fig = panel_d_mirror_stream(cases, rec)
    panel_pdfs["d"] = save(fig, "fig4d_deficit_offset_mirror_stream")[1]
    fig = panel_e_policy_frontier(cases, rec)
    panel_pdfs["e"] = save(fig, "fig4e_policy_value_frontier")[1]

    units.to_csv(OUT / "fig4_new_unit_opportunity_source.csv", index=False)
    rec.to_csv(OUT / "fig4_new_recoverable_capacity_source.csv", index=False)
    cases.to_csv(OUT / "fig4_new_policy_margin_cases_source.csv", index=False)
    unit_summary.to_csv(OUT / "fig4_new_unit_value_summary_source.csv", index=False)
    offset_df.to_csv(OUT / "fig4_new_2035_offset_accounting_source.csv", index=False)

    out = compose_pdf(panel_pdfs)
    print(out)
    print(CURRENT)


if __name__ == "__main__":
    main()
