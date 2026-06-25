from __future__ import annotations

import os
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT = Path(__file__).resolve().parents[2]
TABLE = SOURCE_ROOT / "tables" / "fig2_5_canonical_20260514"
OUT = ROOT / "figures" / "replacement_panels_same_size_20260517_noniso_shaded_2023counties"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
MAP_ORDER = ["CAISO", "ERCOT", "MISO", "PJM", "NYISO", "ISO-NE", "SPP"]

CLUSTER_COLORS = {
    "deficit pressure": "#cf5d45",
    "low-headroom growth": "#d99b3a",
    "supply-buffered expansion": "#62a992",
    "high-margin limited growth": "#6f98ba",
}

RESOURCE_COLORS = {
    "Gas": "#b87a4a",
    "Coal": "#7f7f7a",
    "Nuclear": "#8d75bd",
    "Hydro": "#76a9c4",
    "VRE": "#91bd78",
}

TEXT = "#35302b"
NON_ISO_FILL = "#eee8df"
NON_ISO_EDGE = "#ddd4c8"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def canonical_core() -> pd.DataFrame:
    core = pd.read_csv(TABLE / "fig3" / "fig3a_clustered_archetype_source_canonical.csv")
    core["cluster"] = core["archetype_rule_label"].fillna(core["cluster"])
    core["cluster_color"] = core["cluster"].map(CLUSTER_COLORS).fillna("#999999")
    return core.set_index("ISO").loc[ISO_ORDER].reset_index()


def resource_mix(row: pd.Series) -> dict[str, float]:
    vals = {
        "Gas": float(row.get("gas_share", 0)),
        "Coal": float(row.get("coal_share", 0)),
        "Nuclear": float(row.get("nuclear_share", 0)),
        "Hydro": float(row.get("hydro_share", 0)),
        "VRE": float(row.get("vre_share", 0)),
    }
    total = sum(vals.values())
    return {k: (v / total if total else 0.0) for k, v in vals.items()}


def draw_resource_bar(ax: plt.Axes, x: float, y: float, w: float, h: float, row: pd.Series) -> None:
    xpos = x
    for name, frac in resource_mix(row).items():
        ax.add_patch(Rectangle((xpos, y), w * frac, h, fc=RESOURCE_COLORS[name], ec="white", lw=0.25))
        xpos += w * frac


def single_assignment_counties(core: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    counties = gpd.read_file(PROJECT / "out_std" / "GEO" / "cb2023_county_5m" / "cb_2023_us_county_5m.shp")
    counties = counties[~counties["STATEFP"].astype(str).isin(["02", "15", "72", "60", "66", "69", "78"])].copy()
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)

    assignment = pd.read_csv(PROJECT / "out" / "county_to_iso_area.csv")
    assignment["GEOID"] = assignment["GEOID"].astype(str).str.zfill(5)
    assignment["ISO"] = assignment["ISO_ASSIGNED"].where(assignment["ISO_ASSIGNED"].isin(ISO_ORDER))
    assignment = assignment[["GEOID", "ISO", "share"]]

    mapped = counties.merge(assignment, on="GEOID", how="left")
    mapped = mapped.merge(core[["ISO", "cluster", "cluster_color"]], on="ISO", how="left")
    mapped = mapped.to_crs(5070)

    assigned = mapped[mapped["ISO"].isin(ISO_ORDER)].copy()
    outlines = assigned.dissolve(by="ISO", as_index=False)
    return mapped, outlines


def draw_panel(with_label: bool = False) -> plt.Figure:
    setup_style()
    core = canonical_core()
    counties, outlines = single_assignment_counties(core)

    fig = plt.figure(figsize=(7.1, 3.25), facecolor="white")
    ax = fig.add_axes([0.010, 0.035, 0.595, 0.930])
    ax_cards = fig.add_axes([0.620, 0.090, 0.370, 0.840])

    # Draw the full county universe first so counties outside the seven analyzed
    # ISO/RTO footprints read as a deliberate background category, not holes.
    counties.plot(ax=ax, facecolor=NON_ISO_FILL, edgecolor=NON_ISO_EDGE, linewidth=0.105)
    assigned = counties[counties["ISO"].isin(ISO_ORDER)].copy()
    for cluster, g in assigned.groupby("cluster", dropna=True):
        color = CLUSTER_COLORS.get(str(cluster), "#999999")
        g.plot(ax=ax, facecolor=color, edgecolor="#f8f2ea", linewidth=0.050, alpha=0.84)
    outlines.plot(ax=ax, facecolor="none", edgecolor="#70746d", linewidth=0.62, alpha=0.95)

    offsets = {
        "CAISO": (-50000, -45000),
        "ERCOT": (-20000, -65000),
        "SPP": (-85000, 0),
        "MISO": (65000, 20000),
        "PJM": (25000, -20000),
        "NYISO": (-55000, 6000),
        "ISO-NE": (85000, 74000),
    }
    outline_idx = outlines.set_index("ISO")
    for iso in MAP_ORDER:
        pt = outline_idx.loc[iso].geometry.representative_point()
        dx, dy = offsets[iso]
        ax.text(
            pt.x + dx,
            pt.y + dy,
            iso,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="#353a36")],
        )

    xmin, ymin, xmax, ymax = counties.total_bounds
    ax.set_xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.025)
    ax.set_ylim(ymin - (ymax - ymin) * 0.03, ymax + (ymax - ymin) * 0.035)
    ax.set_axis_off()

    handles = [
        Rectangle((0, 0), 1, 1, fc=CLUSTER_COLORS[name], ec="none", label=name)
        for name in ["deficit pressure", "low-headroom growth", "supply-buffered expansion", "high-margin limited growth"]
        if name in set(core["cluster"])
    ]
    handles.append(
        Rectangle((0, 0), 1, 1, fc=NON_ISO_FILL, ec=NON_ISO_EDGE, lw=0.45, label="outside analyzed footprint")
    )
    leg = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#dddddd",
        title="Descriptive planning profile",
        title_fontsize=6.7,
        fontsize=6.1,
        borderpad=0.42,
        labelspacing=0.28,
        handlelength=0.9,
        handletextpad=0.42,
    )
    leg._legend_box.align = "left"

    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    core_idx = core.set_index("ISO")
    y_top = 0.985
    h = 0.117
    gap = 0.013
    for idx, iso in enumerate(MAP_ORDER):
        row = core_idx.loc[iso]
        y = y_top - idx * (h + gap) - h
        color = row["cluster_color"]
        ax_cards.add_patch(Rectangle((0.02, y), 0.955, h, fc="#fbfaf7", ec="#dedad2", lw=0.55))
        ax_cards.add_patch(Rectangle((0.02, y), 0.014, h, fc=color, ec="none"))
        ax_cards.text(0.055, y + h * 0.72, iso, fontsize=7.5, fontweight="bold", color="#2f2f2f")
        ax_cards.text(0.055, y + h * 0.43, f"H {row['headroom_ratio']:.2f}x", fontsize=5.9, color="#3f3f3f")
        ax_cards.text(0.200, y + h * 0.43, f"AI +{row['ai_gw']:.1f}", fontsize=5.9, color="#3f3f3f")
        ax_cards.text(0.370, y + h * 0.43, f"Supply {row['new_supply_gw']:+.1f}", fontsize=5.9, color="#3f3f3f")
        ax_cards.text(0.055, y + h * 0.15, row["cluster"], fontsize=5.55, color=color, ha="left", va="center")
        draw_resource_bar(ax_cards, 0.620, y + h * 0.66, 0.305, h * 0.15, row)
        margin_col = "#219a75" if row["margin_gw"] >= 0 else "#cc4b40"
        ax_cards.text(0.925, y + h * 0.43, f"{row['margin_gw']:+.1f}", fontsize=5.8, color=margin_col, ha="right")

    x = 0.055
    for name in RESOURCE_COLORS:
        ax_cards.add_patch(Rectangle((x, 0.002), 0.026, 0.020, color=RESOURCE_COLORS[name], ec="none"))
        ax_cards.text(x + 0.032, 0.012, name, va="center", fontsize=5.7, color="#333333")
        x += 0.150 if name != "Nuclear" else 0.172

    if with_label:
        fig.text(0.002, 0.992, "a", ha="left", va="top", fontsize=10.5, fontweight="bold", color="black")

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, with_label in [("", False), ("_with_label", True)]:
        fig = draw_panel(with_label=with_label)
        fig.savefig(OUT / f"fig3a_replacement_county_unique_noniso_shaded{suffix}.png", bbox_inches="tight", pad_inches=0.01, dpi=600)
        fig.savefig(OUT / f"fig3a_replacement_county_unique_noniso_shaded{suffix}_1200dpi.png", bbox_inches="tight", pad_inches=0.01, dpi=1200)
        plt.close(fig)

    core = canonical_core()
    audit_cols = ["ISO", "headroom_ratio", "ai_gw", "new_supply_gw", "margin_gw", "cluster"]
    core[audit_cols].to_csv(OUT / "fig3a_replacement_source_audit.csv", index=False)
    print(f"Wrote replacement Fig. 3a PNGs to {OUT}")
    print(core[audit_cols].to_string(index=False))


if __name__ == "__main__":
    main()
