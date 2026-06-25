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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath

import rebuild_fig4_previous_design_canonical as old


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT = Path(__file__).resolve().parents[2]
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig4_previous_design_canonical_20260514"
CURRENT = TABLE_ARCHIVE / "figures" / "Fig4_intervention_audit_composite copy.pdf"
GEO_SCRIPT = PROJECT / "scripts" / "make_fig2a_iso_headroom_map.py"

YEARS = list(range(2025, 2036))
SCENARIOS = ["low", "mid", "high"]
RECOVERY_CASES = [("low", 0.80), ("mid", 0.90), ("high", 1.00)]
ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
DEFICIT_ISOS = ["PJM", "MISO", "SPP"]

ISO_COLORS = old.ISO_COLORS
PATHWAY_COLORS = old.PATHWAY_COLORS


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.7,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 700,
            "pdf.compression": 9,
        }
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_spines(ax, left=True, bottom=True) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#5d5750")
        ax.spines[side].set_linewidth(0.55)


def panel_label(ax, label: str, x: float = -0.055, y: float = 1.025) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="black",
        zorder=200,
    )


def short_unit_label(plant: object, unit: object) -> str:
    name = str(plant)
    replacements = [
        " Nuclear Generating Station",
        " Nuclear Power Plant",
        " Nuclear Station",
        " Nuclear Facility",
        " Nuclear Plant",
        " Clean Energy Center",
        " Generating Station",
        " Power Station",
    ]
    for text in replacements:
        name = name.replace(text, "")
    name = name.replace("Donald C. Cook", "Donald C Cook")
    unit_s = str(unit)
    if unit_s and unit_s.lower() != "nan" and not name.endswith(unit_s):
        if name in {"Dresden", "Quad Cities", "Point Beach", "Perry", "Crane", "Clinton", "Monticello", "Prairie Island", "Arkansas"}:
            name = f"{name} {unit_s}"
    return name


def canonical_iso_map(cases: pd.DataFrame):
    geo = load_module("fig2_geo_for_fig4", GEO_SCRIPT)
    counties = geo.load_counties()
    g_iso, iso_polys = geo.build_iso_polygons(counties)
    states = counties.dissolve(by="STATEFP", as_index=False)
    iso_polys["ISO_ASSIGNED"] = iso_polys["ISO_ASSIGNED"].replace({"ISONE": "ISO-NE", "ISO_NE": "ISO-NE"})
    m2035 = cases[(cases["scenario"].eq("mid")) & (cases["year"].eq(2035))][["ISO", "baseline_margin_GW"]]
    iso_polys = iso_polys.drop(columns=[c for c in ["baseline_margin_GW"] if c in iso_polys.columns]).merge(
        m2035,
        left_on="ISO_ASSIGNED",
        right_on="ISO",
        how="left",
    )
    return counties, states, g_iso, iso_polys


def draw_panel_a(ax, units: pd.DataFrame, cases: pd.DataFrame, fig: plt.Figure) -> None:
    counties, states, g_iso, iso_polys = canonical_iso_map(cases)
    cmap = LinearSegmentedColormap.from_list("margin_map", ["#c84c3e", "#f8efe8", "#43a784"])
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=18)

    counties.plot(ax=ax, facecolor="#fbfaf7", edgecolor="#ece5dc", linewidth=0.06, zorder=0)
    states.boundary.plot(ax=ax, color="#ded6cb", linewidth=0.28, alpha=0.56, zorder=1)
    colors = [
        cmap(norm(v)) if pd.notna(v) else "#eeeeee"
        for v in iso_polys["baseline_margin_GW"]
    ]
    iso_polys.plot(ax=ax, facecolor=colors, edgecolor="#ffffff", linewidth=0.30, alpha=0.83, zorder=2)
    g_iso.boundary.plot(ax=ax, color="#ffffff", linewidth=0.07, alpha=0.22, zorder=3)
    iso_polys.boundary.plot(ax=ax, color="#ffffff", linewidth=1.15, alpha=0.94, zorder=4)
    iso_polys.boundary.plot(ax=ax, color="#666a63", linewidth=0.50, alpha=0.82, zorder=5)

    op_path = PROJECT / "Final Figs" / "fig4a_nuclear_fleet_pipeline_bubble_map_operating_source.csv"
    fleet = pd.read_csv(op_path)
    fleet_gdf = gpd.GeoDataFrame(
        fleet,
        geometry=gpd.points_from_xy(fleet["Longitude"], fleet["Latitude"]),
        crs=4326,
    ).to_crs(5070)
    fleet_sizes = 4.0 + 15.0 * np.sqrt(fleet_gdf["operating_capacity_GW"].fillna(0.2).clip(lower=0.05))
    ax.scatter(
        fleet_gdf.geometry.x,
        fleet_gdf.geometry.y,
        s=fleet_sizes,
        facecolor="#a9b0ad",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.30,
        zorder=6,
    )

    unit_gdf = gpd.GeoDataFrame(
        units.copy(),
        geometry=gpd.points_from_xy(units["longitude"], units["latitude"]),
        crs=4326,
    ).to_crs(5070)
    grouped = []
    for (plant, iso), g in unit_gdf.groupby(["plant_name", "ISO"]):
        geom = g.geometry.iloc[0]
        grouped.append(
            {
                "plant_name": plant,
                "ISO": iso,
                "x": geom.x,
                "y": geom.y,
                "raw_GW": g["raw_capacity_GW"].sum(),
                "ret_GW": g.loc[g["pathway"].eq("retention"), "recoverable_GW"].sum(),
                "restart_GW": g.loc[g["pathway"].eq("restart"), "recoverable_GW"].sum(),
                "year": int(g["available_year"].min()),
            }
        )
    opp = pd.DataFrame(grouped)

    for _, r in opp.iterrows():
        total = float(r["ret_GW"] + r["restart_GW"])
        if total <= 0 and r["raw_GW"] <= 0:
            continue
        radius = 11000 + 16500 * math.sqrt(max(float(r["raw_GW"]), 0.05) / 1.5)
        ax.add_patch(Circle((r["x"], r["y"]), radius * 1.10, facecolor="white", edgecolor="white", linewidth=1.0, alpha=0.85, zorder=8))
        ax.add_patch(Circle((r["x"], r["y"]), radius, facecolor="white", edgecolor="#2f2a26", linewidth=0.62, alpha=0.98, zorder=9))
        if total > 0:
            theta = 90.0
            ret_angle = 360.0 * float(r["ret_GW"]) / total
            if r["ret_GW"] > 0:
                ax.add_patch(Wedge((r["x"], r["y"]), radius, theta, theta + ret_angle, width=radius * 0.55, facecolor=PATHWAY_COLORS["retention"], edgecolor="white", linewidth=0.35, zorder=10))
            if r["restart_GW"] > 0:
                ax.add_patch(Wedge((r["x"], r["y"]), radius, theta + ret_angle, theta + 360, width=radius * 0.55, facecolor=PATHWAY_COLORS["restart"], edgecolor="white", linewidth=0.35, zorder=10))
            ax.add_patch(Circle((r["x"], r["y"]), radius * 0.42, facecolor="white", edgecolor="none", alpha=0.96, zorder=11))
        ax.text(r["x"], r["y"], f"{int(r['year'])%100:02d}", ha="center", va="center", fontsize=4.0, fontweight="bold", color="#333333", zorder=12)

    planned_path = PROJECT / "Final Figs" / "fig4a_nuclear_fleet_pipeline_bubble_map_planned_source.csv"
    if planned_path.exists():
        planned = pd.read_csv(planned_path)
        if not planned.empty and {"Longitude", "Latitude"}.issubset(planned.columns):
            planned_gdf = gpd.GeoDataFrame(
                planned,
                geometry=gpd.points_from_xy(planned["Longitude"], planned["Latitude"]),
                crs=4326,
            ).to_crs(5070)
            ax.scatter(planned_gdf.geometry.x, planned_gdf.geometry.y, marker="D", s=26, facecolor="#1d9b82", edgecolor="white", linewidth=0.45, zorder=12)

    label_offsets = {
        "PJM": (25000, -38000),
        "MISO": (42000, 22000),
        "SPP": (-70000, 23000),
        "ERCOT": (10000, -65000),
        "CAISO": (-20000, -23000),
        "NYISO": (-45000, 3000),
        "ISO-NE": (75000, 66000),
    }
    region_index = iso_polys.set_index("ISO_ASSIGNED")
    for iso, (dx, dy) in label_offsets.items():
        if iso not in region_index.index:
            continue
        r = region_index.loc[iso]
        pt = r.geometry.representative_point()
        txt = f"{iso}\n{float(r['baseline_margin_GW']):+.1f} GW"
        t = ax.text(pt.x + dx, pt.y + dy, txt, ha="center", va="center", fontsize=5.3, fontweight="bold", color="#333333", zorder=30, linespacing=0.83)
        t.set_path_effects([pe.withStroke(linewidth=1.55, foreground="white", alpha=0.96)])

    xmin, ymin, xmax, ymax = counties.total_bounds
    xspan = xmax - xmin
    yspan = ymax - ymin
    ax.set_xlim(xmin + xspan * 0.010, xmax + xspan * 0.006)
    ax.set_ylim(ymin - yspan * 0.012, ymax + yspan * 0.012)
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.02, y=1.00)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#a9b0ad", markeredgecolor="white", markersize=3.5, alpha=0.55, label="operating fleet"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["retention"], markeredgecolor="#2f2a26", markeredgewidth=0.9, markersize=5.2, label="retention"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PATHWAY_COLORS["restart"], markeredgecolor="#2f2a26", markeredgewidth=0.9, markersize=5.2, label="restart"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#1d9b82", markeredgecolor="white", markersize=3.5, label="planned"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.010, 0.015), frameon=True, facecolor="white", edgecolor="#ded8d1", framealpha=0.92, ncol=2, borderpad=0.25, handletextpad=0.28, columnspacing=0.54, fontsize=4.7)

    cax = ax.inset_axes([0.918, 0.290, 0.020, 0.360])
    vals = np.linspace(-10, 18, 80)
    cax.imshow(vals.reshape(-1, 1), cmap=cmap, norm=norm, origin="lower", aspect="auto", extent=[0, 1, -10, 18])
    cax.set_xticks([])
    cax.set_yticks([-10, 0, 10, 18])
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", labelsize=4.8, length=1.6, pad=1)
    cax.set_ylabel("margin (GW)", fontsize=4.35, rotation=90, labelpad=1.0)
    cax.yaxis.set_label_position("right")
    for spine in cax.spines.values():
        spine.set_linewidth(0.42)
        spine.set_color("#3f3934")


def draw_panel_b(ax, unit_summary: pd.DataFrame) -> None:
    all_units = unit_summary.copy()
    all_units["credit_mid_GW"] = all_units["recoverable_GW"] * 0.90

    df = all_units.copy()
    df["unit_label"] = [short_unit_label(p, u) for p, u in zip(df["plant_name"], df["unit"])]
    df = df[df["credit_mid_GW"].gt(0.02)].copy()
    df = df.sort_values("credit_mid_GW", ascending=False).head(14).copy()
    df["iso_order"] = df["ISO"].map({iso: i for i, iso in enumerate(ISO_ORDER)}).fillna(99)
    df = df.sort_values(["available_year", "pathway", "iso_order", "credit_mid_GW"], ascending=[True, False, True, False]).reset_index(drop=True)
    y = np.arange(len(df))[::-1]

    ax.set_xlim(2024.65, 2035.20)
    ax.set_ylim(-0.7, len(df) - 0.25)
    ax.axvspan(2028, 2032, color="#fbf7f2", zorder=0, alpha=0.65)
    for i, (_, r) in enumerate(df.iterrows()):
        yy = y[i]
        iso_color = ISO_COLORS.get(r["ISO"], "#777777")
        start = int(r["available_year"])
        lw = 2.35
        if r["pathway"] == "restart":
            ax.plot([2025, start], [yy, yy], color="#cfc7bd", lw=0.8, linestyle=(0, (2, 1)), zorder=1)
            ax.plot([start, 2035], [yy, yy], color=PATHWAY_COLORS["restart"], lw=lw, solid_capstyle="round", zorder=3)
            marker = "^"
        else:
            ax.plot([2025, start], [yy, yy], color="#cfc7bd", lw=0.8, zorder=1)
            ax.plot([start, 2035], [yy, yy], color=PATHWAY_COLORS["retention"], lw=lw, solid_capstyle="round", zorder=3)
            marker = "o"
        ax.scatter(start, yy, s=24, marker=marker, facecolor=iso_color, edgecolor="white", linewidth=0.4, zorder=5)
        ax.text(2024.55, yy, str(r["unit_label"]), ha="right", va="center", fontsize=4.9, color="#333333")
        ax.scatter(2024.72, yy, marker="s", s=9, color=iso_color, clip_on=False, zorder=5)

    checkpoints = [
        ("MISO\n2028", 2028),
        ("SPP\n2029", 2029),
        ("PJM\n2031", 2031),
        ("2035", 2035),
    ]
    inset_rows = []
    for label, year in checkpoints:
        sub = all_units[all_units["credit_mid_GW"].gt(0.02) & all_units["available_year"].le(year)]
        ret = float(sub.loc[sub["pathway"].eq("retention"), "credit_mid_GW"].sum())
        restart = float(sub.loc[sub["pathway"].eq("restart"), "credit_mid_GW"].sum())
        inset_rows.append({"label": label, "year": year, "retention_GW": ret, "restart_GW": restart, "total_GW": ret + restart})
    pd.DataFrame(inset_rows).to_csv(OUT / "fig4b_inset_cumulative_credited_capacity_source.csv", index=False)

    iax = ax.inset_axes([0.090, 0.075, 0.335, 0.255])
    iax.set_facecolor((1, 1, 1, 0.92))
    x = np.arange(len(inset_rows))
    ret_vals = np.asarray([r["retention_GW"] for r in inset_rows])
    restart_vals = np.asarray([r["restart_GW"] for r in inset_rows])
    iax.bar(x, ret_vals, width=0.58, color=PATHWAY_COLORS["retention"], alpha=0.86, edgecolor="white", linewidth=0.25)
    iax.bar(x, restart_vals, bottom=ret_vals, width=0.58, color=PATHWAY_COLORS["restart"], alpha=0.86, edgecolor="white", linewidth=0.25)
    for xi, total in zip(x, ret_vals + restart_vals):
        iax.text(xi, total + 0.20, f"{total:.1f}", ha="center", va="bottom", fontsize=3.25, color="#3d3834")
    iax.text(0.02, 0.96, "credited by clock", transform=iax.transAxes, ha="left", va="top", fontsize=3.65, color="#3d3834", fontweight="bold")
    iax.text(0.02, 0.82, "GW available", transform=iax.transAxes, ha="left", va="top", fontsize=3.25, color="#6d665f")
    iax.set_xticks(x)
    iax.set_xticklabels([r["label"] for r in inset_rows], fontsize=3.2, linespacing=0.85)
    iax.set_ylim(0, max(ret_vals + restart_vals) * 1.28)
    iax.set_yticks([0, 4, 8])
    iax.tick_params(axis="y", labelsize=3.2, length=1.2, pad=0.8, width=0.32)
    iax.tick_params(axis="x", length=0, pad=0.8)
    iax.grid(axis="y", color="#eee5dc", lw=0.30)
    for side in ["top", "right"]:
        iax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        iax.spines[side].set_linewidth(0.34)
        iax.spines[side].set_color("#8a8179")
    iax.text(0.98, 0.06, "retention", transform=iax.transAxes, ha="right", va="bottom", fontsize=3.05, color=PATHWAY_COLORS["retention"])
    iax.text(0.98, 0.20, "restart", transform=iax.transAxes, ha="right", va="bottom", fontsize=3.05, color=PATHWAY_COLORS["restart"])

    ax.set_yticks([])
    ax.set_xticks([2025, 2028, 2031, 2035])
    ax.grid(axis="x", color="#eee5dc", lw=0.62)
    ax.set_xlabel("available year", fontsize=5.8, labelpad=1.2)
    panel_label(ax, "b", x=-0.095, y=1.005)
    clean_spines(ax, left=False)
    ax.tick_params(axis="x", labelsize=5.35)
    ax.tick_params(axis="y", length=0)


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.82) -> None:
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
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2))


def draw_panel_c(ax, units: pd.DataFrame, rec: pd.DataFrame, offset_df: pd.DataFrame) -> None:
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
    total_raw = raw_ret + raw_restart

    stages = [
        [("retention", raw_ret), ("restart", raw_restart)],
        [("retention", ret_rec), ("restart", restart_rec), ("screened out", screened)],
        [("retention", ret_eff), ("restart", restart_eff), ("derate", derate), ("screened out", screened)],
        [
            ("MISO", used_iso.get("MISO", 0.0)),
            ("PJM", used_iso.get("PJM", 0.0)),
            ("SPP", used_iso.get("SPP", 0.0)),
            ("not binding", unused_ret + unused_restart),
            ("derate", derate),
            ("screened out", screened),
        ],
    ]
    colors = {
        "retention": PATHWAY_COLORS["retention"],
        "restart": PATHWAY_COLORS["restart"],
        "screened out": PATHWAY_COLORS["excluded"],
        "derate": PATHWAY_COLORS["derating"],
        "MISO": ISO_COLORS["MISO"],
        "PJM": ISO_COLORS["PJM"],
        "SPP": ISO_COLORS["SPP"],
        "not binding": PATHWAY_COLORS["unused"],
    }
    flows = [
        (0, "retention", 1, "retention", ret_rec, PATHWAY_COLORS["retention"], 0.68),
        (0, "retention", 1, "screened out", max(0.0, raw_ret - ret_rec), PATHWAY_COLORS["excluded"], 0.38),
        (0, "restart", 1, "restart", restart_rec, PATHWAY_COLORS["restart"], 0.68),
        (0, "restart", 1, "screened out", max(0.0, raw_restart - restart_rec), PATHWAY_COLORS["excluded"], 0.34),
        (1, "retention", 2, "retention", ret_eff, "#d17d55", 0.70),
        (1, "retention", 2, "derate", ret_rec * 0.10, PATHWAY_COLORS["derating"], 0.48),
        (1, "restart", 2, "restart", restart_eff, "#8e70b0", 0.70),
        (1, "restart", 2, "derate", restart_rec * 0.10, PATHWAY_COLORS["derating"], 0.48),
        (1, "screened out", 2, "screened out", screened, PATHWAY_COLORS["excluded"], 0.34),
        (2, "retention", 3, "MISO", float(offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("MISO", 0.0)), ISO_COLORS["MISO"], 0.72),
        (2, "retention", 3, "PJM", float(offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("PJM", 0.0)), ISO_COLORS["PJM"], 0.72),
        (2, "retention", 3, "SPP", float(offset_df.set_index("ISO").get("retention_used_mid_GW", pd.Series()).get("SPP", 0.0)), ISO_COLORS["SPP"], 0.72),
        (2, "restart", 3, "PJM", float(offset_df.set_index("ISO").get("restart_used_mid_GW", pd.Series()).get("PJM", 0.0)), ISO_COLORS["PJM"], 0.62),
        (2, "retention", 3, "not binding", unused_ret, PATHWAY_COLORS["unused"], 0.50),
        (2, "restart", 3, "not binding", unused_restart, PATHWAY_COLORS["unused"], 0.50),
        (2, "derate", 3, "derate", derate, PATHWAY_COLORS["derating"], 0.46),
        (2, "screened out", 3, "screened out", screened, PATHWAY_COLORS["excluded"], 0.32),
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    xs = [0.09, 0.37, 0.64, 0.89]
    node_w = 0.028
    top = 0.945
    usable_h = 0.900
    gap = 0.007
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
        if a is not None and b is not None:
            ribbon(ax, xs[s0] + node_w / 2, xs[s1] - node_w / 2, a[0], a[1], b[0], b[1], col, alpha=alpha)

    for i, stage in enumerate(stages):
        ax.text(xs[i], 0.990, ["candidate", "recoverable", "effective", "2035 use"][i], ha="center", va="top", fontsize=5.6, fontweight="bold", color="#333333")
        for name, val in stage:
            if val <= 0.001 or name not in node_spans[i]:
                continue
            lo, hi = node_spans[i][name]
            ax.add_patch(Rectangle((xs[i] - node_w / 2, lo), node_w, hi - lo, facecolor=colors[name], edgecolor="white", lw=0.34, alpha=0.98, zorder=5))
            if hi - lo > 0.040:
                ax.text(xs[i], (lo + hi) / 2, f"{val:.1f}", ha="center", va="center", fontsize=4.5, color="white", fontweight="bold", rotation=90, zorder=6)

    for name in ["retention", "restart"]:
        lo, hi = node_spans[0].get(name, (None, None))
        if lo is not None:
            ax.text(xs[0] - 0.028, (lo + hi) / 2, name, ha="right", va="center", fontsize=4.9, color="#4f4841")
    for name in ["MISO", "PJM", "SPP", "not binding", "derate", "screened out"]:
        lo, hi = node_spans[-1].get(name, (None, None))
        if lo is not None:
            label = name.replace("not binding", "not\nbinding").replace("screened out", "screened\nout")
            ax.text(xs[-1] + 0.026, (lo + hi) / 2, label, ha="left", va="center", fontsize=4.35, color="#4f4841", linespacing=0.88)
    ax.text(0.055, 0.010, "19.4 GW candidates; mid recovery = 0.90", ha="left", va="bottom", fontsize=4.35, color="#6a625a")
    panel_label(ax, "c", x=-0.035, y=1.00)


def residual_by_iso_year(cases: pd.DataFrame, rec: pd.DataFrame, ratio: float, scenario: str):
    rec_by = rec.set_index(["ISO", "year"])
    base = {}
    resid = {}
    for iso in DEFICIT_ISOS:
        b, r = [], []
        for year in YEARS:
            row = cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(year)) & (cases["scenario"].eq(scenario))]
            margin = float(row["baseline_margin_GW"].iloc[0])
            recov = 0.0
            if (iso, year) in rec_by.index:
                rr = rec_by.loc[(iso, year)]
                recov = float(rr["retention_recoverable_GW"] + rr["restart_recoverable_GW"])
            b.append(max(0.0, -margin))
            r.append(max(0.0, -(margin + ratio * recov)))
        base[iso] = np.asarray(b)
        resid[iso] = np.asarray(r)
    return base, resid


def draw_panel_d(ax, cases: pd.DataFrame, rec: pd.DataFrame) -> None:
    years = np.asarray(YEARS)
    base_mid, resid_mid = residual_by_iso_year(cases, rec, 0.90, "mid")

    base_low = []
    base_high = []
    resid_low = []
    resid_high = []
    for year in YEARS:
        base_totals = []
        resid_totals = []
        for scenario in SCENARIOS:
            base_total = 0.0
            for iso in DEFICIT_ISOS:
                base_total += max(0.0, -float(cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(year)) & (cases["scenario"].eq(scenario))]["baseline_margin_GW"].iloc[0]))
            base_totals.append(base_total)
            for _label, ratio in RECOVERY_CASES:
                resid_total = 0.0
                for iso in DEFICIT_ISOS:
                    resid_total += max(0.0, -(float(cases[(cases["ISO"].eq(iso)) & (cases["year"].eq(year)) & (cases["scenario"].eq(scenario))]["baseline_margin_GW"].iloc[0]) + ratio * float(rec[(rec["ISO"].eq(iso)) & (rec["year"].eq(year))]["total_recoverable_GW"].sum())))
                resid_totals.append(resid_total)
        base_low.append(min(base_totals))
        base_high.append(max(base_totals))
        resid_low.append(min(resid_totals))
        resid_high.append(max(resid_totals))

    base_total_mid = sum(base_mid.values())
    resid_total_mid = sum(resid_mid.values())
    y0 = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = base_mid[iso]
        ax.fill_between(years, y0, y0 + vals, color=ISO_COLORS[iso], alpha=0.56, edgecolor="white", linewidth=0.28, zorder=3)
        y0 += vals

    y1 = np.zeros_like(years, dtype=float)
    for iso in DEFICIT_ISOS:
        vals = resid_mid[iso]
        ax.fill_between(years, -y1, -(y1 + vals), color=ISO_COLORS[iso], alpha=0.30, edgecolor="white", linewidth=0.28, zorder=3)
        y1 += vals

    base_low = np.asarray(base_low)
    base_high = np.asarray(base_high)
    resid_low = np.asarray(resid_low)
    resid_high = np.asarray(resid_high)
    ax.fill_between(years, base_low, base_high, color="#8a8179", alpha=0.10, edgecolor="none", zorder=1)
    ax.fill_between(years, -resid_high, -resid_low, color="#168875", alpha=0.10, edgecolor="none", zorder=1)
    ax.plot(years, base_low, color="#7d736b", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, base_high, color="#7d736b", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, -resid_low, color="#168875", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)
    ax.plot(years, -resid_high, color="#168875", lw=0.65, linestyle=(0, (2, 1)), alpha=0.78, zorder=4)

    ax.axhline(0, color="#5b544d", lw=0.62)
    ax.text(2025.12, max(base_high) * 0.86, "baseline deficit", ha="left", va="center", fontsize=4.7, color="#6a443f")
    ax.text(2025.12, -max(resid_high) * 0.82, "residual with nuclear", ha="left", va="center", fontsize=4.7, color="#168875")
    ax.text(2034.70, float(base_total_mid[-1]) + 0.25, f"{float(base_total_mid[-1]):.1f}", ha="right", va="center", fontsize=4.6, color="#4c413c", fontweight="bold")
    ax.text(2034.70, -float(resid_total_mid[-1]) - 0.25, f"{float(resid_total_mid[-1]):.1f}", ha="right", va="center", fontsize=4.6, color="#168875", fontweight="bold")
    ax.set_xlim(2025, 2035)
    ymax = max(float(max(base_high)), float(max(resid_high)), 1.0) * 1.14
    ax.set_ylim(-ymax, ymax)
    tick_max = 20 if ymax > 15 else 10
    ax.set_yticks([-tick_max, -tick_max / 2, 0, tick_max / 2, tick_max])
    ax.set_yticklabels([f"{tick_max:g}", f"{tick_max/2:g}", "0", f"{tick_max/2:g}", f"{tick_max:g}"])
    ax.set_xlabel("year", labelpad=1.2)
    ax.set_ylabel("GW", labelpad=1.2)
    ax.grid(axis="x", color="#eee5dc", lw=0.66)
    ax.grid(axis="y", color="#f3eee8", lw=0.42)
    clean_spines(ax)
    ax.tick_params(axis="both", labelsize=5.35)
    handles = [
        Line2D([0], [0], color=ISO_COLORS["PJM"], lw=3.4, alpha=0.55, label="PJM"),
        Line2D([0], [0], color=ISO_COLORS["MISO"], lw=3.4, alpha=0.55, label="MISO"),
        Line2D([0], [0], color=ISO_COLORS["SPP"], lw=3.4, alpha=0.55, label="SPP"),
        Line2D([0], [0], color="#7d736b", lw=0.85, linestyle=(0, (2, 1)), label="envelope"),
    ]
    leg2 = ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.20), borderaxespad=0, ncol=4, handlelength=0.95, columnspacing=0.45, handletextpad=0.25, fontsize=4.10)
    ax.add_artist(leg2)
    panel_label(ax, "d", x=-0.115, y=1.005)


def draw_panel_e(ax, cases: pd.DataFrame, rec: pd.DataFrame) -> None:
    summary = pd.read_csv(TABLE / "fig4" / "fig4_unit_marginal_value_summary_canonical.csv")
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

    rows = []
    for _, r in plot_df.iterrows():
        color = ISO_COLORS.get(str(r["ISO"]), "#b8b0aa")
        height = float(r["value_density_mid"])
        display_height = height if r["is_positive"] else 0.18
        width = float(r["credit_mid_GW"])
        face = color if r["is_positive"] else "#d2cdc7"
        alpha = 0.78 if r["is_positive"] else 0.40
        hatch = "////" if str(r["pathway"]) == "restart" else None
        ax.bar(float(r["x0"]), display_height, width=width, align="edge", color=face, alpha=alpha, edgecolor="white" if str(r["pathway"]) == "retention" else color, linewidth=0.36, hatch=hatch, zorder=3)
        if r["is_positive"]:
            yerr_low = max(0.0, height - float(r["value_density_low"]))
            yerr_high = max(0.0, float(r["value_density_high"]) - height)
            ax.errorbar(
                float(r["xmid"]),
                height,
                yerr=np.array([[yerr_low], [yerr_high]]),
                fmt="none",
                ecolor="#4f4944",
                elinewidth=0.48,
                capsize=1.55,
                capthick=0.48,
                alpha=0.76,
                zorder=4,
            )
        rows.append(
            {
                "plant_unit": short_unit_label(r["plant_name"], r["unit"]),
                "ISO": r["ISO"],
                "pathway": r["pathway"],
                "credit_mid_GW": width,
                "cumulative_credit_mid_GW": float(r["x0"] + width),
                "avoided_GWyr_mid": float(r["value_mid"]),
                "avoided_GWyr_per_GW_mid": height,
            }
        )

    for _, r in plot_df[plot_df["is_positive"]].head(4).iterrows():
        ax.text(float(r["xmid"]), float(r["value_density_high"]) + 0.18, short_unit_label(r["plant_name"], r["unit"]), ha="center", va="bottom", fontsize=3.45, color="#3f3935", rotation=34)
    restart_pair = plot_df[plot_df["pathway"].eq("restart")]
    if not restart_pair.empty:
        x0 = float(restart_pair["x0"].min())
        x1 = float((restart_pair["x0"] + restart_pair["credit_mid_GW"]).max())
        y = max(float(restart_pair["value_density_high"].max()) + 0.28, 3.6)
        ax.plot([x0, x1], [y, y], color=ISO_COLORS["PJM"], lw=0.65, alpha=0.80)
        ax.text((x0 + x1) / 2, y + 0.10, "restart candidates", ha="center", va="bottom", fontsize=3.50, color=ISO_COLORS["PJM"])

    total_positive = float(positive["value_mid"].sum())
    total_credit = float(summary["credit_mid_GW"].sum())
    zero_credit = float(zero["credit_mid_GW"].sum())
    ax.text(0.975, 0.92, f"area = avoided GW-yr\n{total_positive:.1f} total", transform=ax.transAxes, ha="right", va="top", fontsize=4.10, color="#4d4640", linespacing=0.90)
    ax.text(0.035, 0.88, "higher marginal value", transform=ax.transAxes, ha="left", va="center", fontsize=4.05, color="#6a443f")
    if zero_credit > 0:
        zero_start = float(plot_df.loc[plot_df["is_positive"], "credit_mid_GW"].sum())
        ax.text(zero_start + zero_credit / 2, 0.34, f"{zero_credit:.1f} GW\nlow/no value", ha="center", va="bottom", fontsize=3.65, color="#746c66", linespacing=0.85)

    pd.DataFrame(rows).to_csv(OUT / "fig4e_marginal_abatement_curve_revised_source.csv", index=False)
    ax.set_xlim(-0.05, max(total_credit, 0.1) + 0.12)
    ymax = max(float(positive["value_density_high"].max()) * 1.18, 1.0)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("cumulative credited nuclear capacity (GW)", labelpad=1.2)
    ax.set_ylabel("GW-yr per GW", labelpad=0.7, fontsize=5.2)
    ax.grid(axis="y", color="#eee5dc", lw=0.52)
    ax.grid(axis="x", color="#f5eee8", lw=0.35)
    clean_spines(ax)
    ax.tick_params(axis="both", labelsize=5.35)
    handles = [
        Line2D([0], [0], color=ISO_COLORS["PJM"], lw=3.4, alpha=0.78, label="PJM"),
        Line2D([0], [0], color=ISO_COLORS["MISO"], lw=3.4, alpha=0.78, label="MISO"),
        Line2D([0], [0], color=ISO_COLORS["SPP"], lw=3.4, alpha=0.78, label="SPP"),
        Rectangle((0, 0), 1, 1, facecolor="#d2cdc7", alpha=0.45, edgecolor="white", label="low/no value"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#7a6f68", hatch="////", label="restart"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.28), ncol=5, fontsize=3.95, handlelength=0.92, columnspacing=0.36, handletextpad=0.22, borderaxespad=0.0)
    panel_label(ax, "e", x=-0.095, y=1.005)


def rasterize_non_text(fig: plt.Figure) -> None:
    for ax in fig.axes:
        for artist in list(ax.collections) + list(ax.patches) + list(ax.lines) + list(ax.images):
            if hasattr(artist, "set_rasterized"):
                artist.set_rasterized(True)
        text_artists = list(ax.texts)
        text_artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        text_artists.extend(ax.get_xticklabels())
        text_artists.extend(ax.get_yticklabels())
        for text in text_artists:
            if hasattr(text, "set_rasterized"):
                text.set_rasterized(False)
            text.set_zorder(200)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_zorder(200)
            for text in legend.get_texts():
                text.set_rasterized(False)
                text.set_zorder(201)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    gap, units, rec, cases, unit_long, unit_summary = old.load_all()
    offset_df = old.compute_offset_df(cases, rec)

    fig = plt.figure(figsize=(7.1, 4.28), facecolor="white")
    gs = fig.add_gridspec(
        2,
        14,
        left=0.020,
        right=0.990,
        bottom=0.080,
        top=0.975,
        wspace=0.92,
        hspace=0.235,
        height_ratios=[1.00, 1.00],
    )
    ax_a = fig.add_subplot(gs[0, 0:8])
    ax_b = fig.add_subplot(gs[0, 8:14])
    ax_c = fig.add_subplot(gs[1, 0:5])
    ax_d = fig.add_subplot(gs[1, 5:9])
    ax_e = fig.add_subplot(gs[1, 9:14])

    # Balance the lower row: the alluvial audit needs more vertical reading
    # room, while the two quantitative summaries read better as tighter panels.
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_c.y0 - 0.008, pos_c.width, pos_c.height + 0.016])
    for ax in (ax_d, ax_e):
        pos = ax.get_position()
        shrink_w = 0.955
        shrink_h = 0.900
        ax.set_position(
            [
                pos.x0 + pos.width * (1 - shrink_w) / 2,
                pos.y1 - pos.height * shrink_h,
                pos.width * shrink_w,
                pos.height * shrink_h,
            ]
        )

    draw_panel_a(ax_a, units, cases, fig)
    draw_panel_b(ax_b, unit_summary)
    draw_panel_c(ax_c, units, rec, offset_df)
    draw_panel_d(ax_d, cases, rec)
    draw_panel_e(ax_e, cases, rec)

    rasterize_non_text(fig)
    out_pdf = OUT / "Fig4_intervention_audit_composite_aligned.pdf"
    out_png = OUT / "Fig4_intervention_audit_composite_aligned.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=700)
    shutil.copy2(out_pdf, CURRENT)
    plt.close(fig)
    print(out_pdf)
    print(CURRENT)


if __name__ == "__main__":
    main()
