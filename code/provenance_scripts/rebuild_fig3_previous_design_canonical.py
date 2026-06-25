from __future__ import annotations

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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT = Path(__file__).resolve().parents[2]
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig3_previous_design_canonical_20260514"
PANEL_DIR = OUT / "panels"
CURRENT = TABLE_ARCHIVE / "figures" / "Fig3_final_dense_mechanism_composite_VECTOR copy.pdf"

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

ISO_COLORS = {
    "PJM": "#E62E8A",
    "MISO": "#7A6FB0",
    "SPP": "#B98A13",
    "ERCOT": "#D86613",
    "CAISO": "#1B9E77",
    "NYISO": "#C9A227",
    "ISO-NE": "#6FA76F",
}

PJM_ZONE_MAP = {
    "ATLANTIC CITY ELECTRIC CO": "AECO",
    "Alleghany Power": "APS",
    "American Electric Power": "AEP",
    "American Transmission Systems, Inc": "ATSI",
    "BALTIMORE GAS & ELECTRIC CO": "BGE",
    "COMMONWEALTH EDISON CO": "COMED",
    "DAYTON POWER & LIGHT CO": "DAY",
    "DELMARVA POWER": "DPL",
    "Duke Energy Ohio/Kentucky": "DEOK",
    "Dominion Energy": "DOM",
    "DUQUESNE LIGHT CO": "DUQ",
    "East Kentucky Power Coop": "EKPC",
    "JERSEY CENTRAL POWER & LT CO": "JCPL",
    "MECKLENBURG ELECTRIC COOPERATIVE": "MECK",
    "METROPOLITAN EDISON CO": "METED",
    "PENNSYLVANIA ELECTRIC CO": "PENELEC",
    "PECO ENERGY CO": "PECO",
    "POTOMAC ELECTRIC POWER CO": "PEPCO",
    "PPL ELECTRIC UTILITIES CORP": "PPL",
    "PUBLIC SERVICE ELEC & GAS CO": "PSEG",
    "ROCKLAND ELECTRIC CO": "RECO",
}

TEXT = "#35302b"
GRID = "#eadfd4"
BASE = "#ead4bc"
RANGE = "#4a4038"


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
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def clean_spines(ax: plt.Axes, left: bool = True, bottom: bool = True) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_color("#57514c")
    ax.spines["bottom"].set_color("#57514c")
    ax.tick_params(colors=TEXT, width=0.55, length=2.2, pad=1.0)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLE / rel)


def pct_rank(values: pd.Series | np.ndarray, high_is_risky: bool = True) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    if s.nunique(dropna=True) <= 1:
        r = np.full(len(s), 0.5)
    else:
        r = s.rank(method="average", pct=True).to_numpy()
    return r if high_is_risky else 1 - r


def short_zone(zone: object) -> str:
    text = str(zone).upper()
    special = {
        "PACIFICGASANDELECTRIC": "PG&E",
        "SANDIEGOGASELECTRIC": "SDG&E",
        "SOUTHERNCALIFORNIAEDISON": "SCE",
        "VALLEYELECTRICASSOCIATION": "VEA",
        "MASSACHUSETTS": "MA",
        "VERMONT": "VT",
        "CONNECTICUT": "CT",
        "INDIANA": "INDN",
    }
    if text in special:
        return special[text]
    if text.startswith("LRZ"):
        return text.replace("LRZ", "LRZ ")[:7]
    if len(text) > 13:
        return text[:11] + "..."
    return text


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


def canonical_core() -> pd.DataFrame:
    core = read_csv("fig3/fig3a_clustered_archetype_source_canonical.csv")
    core["cluster"] = core["archetype_rule_label"].fillna(core["cluster"])
    core["cluster_color"] = core["cluster"].map(CLUSTER_COLORS).fillna("#999999")
    core["effective_to_active_pct"] = 100.0 * core["effective_over_active_queue"].fillna(0.0)
    return core.set_index("ISO").loc[ISO_ORDER].reset_index()


def lmp_summary() -> pd.DataFrame:
    return read_csv("fig3/fig3_zone_lmp_stress_summary_canonical.csv")


def draw_panel_a(core: pd.DataFrame) -> plt.Figure:
    counties = gpd.read_file(PROJECT / "data/us_counties.geojson")
    counties = counties[~counties["STATE"].isin(["02", "15", "72", "60", "66", "69", "78"])].to_crs(5070)
    regions = gpd.read_file(PROJECT / "out/iso7_regions_region_only_clean.gpkg").to_crs(5070)
    regions = regions.dissolve(by="ISO").reset_index()
    regions = regions.merge(core[["ISO", "cluster", "cluster_color"]], on="ISO", how="left")

    fig = plt.figure(figsize=(7.1, 3.25), facecolor="white")
    ax = fig.add_axes([0.010, 0.035, 0.595, 0.930])
    ax_cards = fig.add_axes([0.620, 0.090, 0.370, 0.840])

    counties.plot(ax=ax, facecolor="#f8f5ef", edgecolor="#e5ded4", linewidth=0.13)
    for _, r in regions.iterrows():
        gpd.GeoDataFrame([r], geometry="geometry", crs=regions.crs).plot(
            ax=ax,
            facecolor=r["cluster_color"],
            edgecolor="#70746d",
            linewidth=0.62,
            alpha=0.82,
        )

    offsets = {
        "CAISO": (-50000, -45000),
        "ERCOT": (-20000, -65000),
        "SPP": (-85000, 0),
        "MISO": (65000, 20000),
        "PJM": (25000, -20000),
        "NYISO": (-55000, 6000),
        "ISO-NE": (85000, 74000),
    }
    region_index = regions.set_index("ISO")
    for iso in MAP_ORDER:
        pt = region_index.loc[iso].geometry.representative_point()
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
    return fig


def first_deficit_year(iso: str) -> float | None:
    traj = read_csv("fig2/fig2e_margin_trajectory_ribbons_canonical_source.csv")
    g = traj[traj["ISO"].eq(iso)].sort_values("year")
    years = g["year"].to_numpy(dtype=float)
    vals = g["mid_growth_margin_GW"].to_numpy(dtype=float)
    if len(vals) == 0 or np.all(vals > 0):
        return None
    hit = np.where(vals <= 0)[0][0]
    if hit == 0:
        return float(years[hit])
    y0, y1 = years[hit - 1], years[hit]
    v0, v1 = vals[hit - 1], vals[hit]
    if v0 == v1:
        return float(y1)
    return float(y0 + (0 - v0) * (y1 - y0) / (v1 - v0))


def draw_panel_b(core: pd.DataFrame) -> plt.Figure:
    lmp = lmp_summary().set_index("ISO")
    df = core.copy()
    df["p90_beta_gw"] = df["ISO"].map(lmp["p90_beta_gw"])
    df["first_deficit_mid"] = [first_deficit_year(iso) for iso in df["ISO"]]
    df["first_deficit_basis"] = df["first_deficit_mid"].apply(
        lambda x: 2036 if pd.isna(x) else float(np.ceil(x - 1e-9))
    )

    specs: list[tuple[str, str, str, bool]] = [
        ("headroom_pct", "H25", "{:.0f}%", False),
        ("ai_to_headroom", "AI/H", "{:.1f}x", True),
        ("nonai_to_headroom", "NonAI/H", "{:.1f}x", True),
        ("new_supply_gw", "Supply", "{:+.1f}", False),
        ("effective_to_active_pct", "Queue", "{:.1f}%", False),
        ("vre_pct", "VRE", "{:.0f}%", True),
        ("p90_beta_gw", "LMP p90", "{:.0f}", True),
        ("first_deficit_basis", "First", "{:.0f}", False),
        ("margin_gw", "M35", "{:+.1f}", False),
    ]
    df["headroom_pct"] = df["headroom_ratio"] * 100.0
    df["vre_pct"] = df["vre_share"] * 100.0

    row_order = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
    df = df.set_index("ISO").loc[row_order].reset_index()
    risk = []
    for col, _, _, high in specs:
        risk.append(pct_rank(df[col].fillna(df[col].median()), high_is_risky=high))
    mat = np.array(risk).T
    cmap = LinearSegmentedColormap.from_list("risk_teal_warm", ["#4fa28e", "#f6edda", "#ef8a62", "#b2182b"])

    fig, ax = plt.subplots(figsize=(3.55, 2.35))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fc=cmap(float(mat[i, j])),
                    ec="white",
                    lw=0.95,
                    zorder=0,
                )
            )
    ax.set_xticks(np.arange(len(specs)))
    ax.set_xticklabels([label for _, label, _, _ in specs])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.tick_params(length=0)

    for i, iso in enumerate(row_order):
        color = df.loc[i, "cluster_color"]
        ax.add_patch(Rectangle((-0.62, i - 0.47), 0.11, 0.94, fc=color, ec="none", clip_on=False))
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight("bold")

    for i in range(mat.shape[0]):
        for j, (name, _, fmt, _) in enumerate(specs):
            if name == "first_deficit_basis":
                raw = df.loc[i, "first_deficit_basis"]
                label = ">35" if float(raw) >= 2036 else f"{int(float(raw))}"
            else:
                raw = df.loc[i, name]
                label = "n/a" if pd.isna(raw) else fmt.format(float(raw))
            txt_color = "white" if mat[i, j] > 0.78 or mat[i, j] < 0.14 else "#2f2f2f"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.2, color=txt_color)

    ax.set_xlim(-0.64, len(specs) - 0.5)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    ax.tick_params(axis="x", labelsize=5.7, pad=2)
    ax.tick_params(axis="y", labelsize=6.3, pad=2)
    clean_spines(ax, left=False, bottom=False)

    cax = ax.inset_axes([1.018, 0.145, 0.026, 0.700])
    for k, val in enumerate(np.linspace(0.0, 1.0, 36)):
        cax.add_patch(Rectangle((0, k / 36), 1, 1 / 36, fc=cmap(float(val)), ec="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([0, 0.5, 1])
    cax.set_yticklabels(["low", "mid", "high"])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.tick_params(labelsize=5.4, length=1.5, width=0.4, pad=1)
    cax.set_ylabel("risk", fontsize=5.8, labelpad=2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#3f3934")
    ax.text(0.0, -0.20, "Color ranks each mechanism; numbers show actual values.", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color="#5b5b5b")
    fig.tight_layout(pad=0.40)
    return fig


def draw_panel_c(core: pd.DataFrame) -> plt.Figure:
    tx = read_csv("fig3/fig3d_zone_pressure_screening_canonical.csv")
    tx["iso"] = tx["ISO"].fillna(tx["iso"])
    tx["zone_short"] = tx["zone"].map(short_zone)
    tx["zone_stress_mid"] = tx["screen_index"].clip(lower=0)
    plot = tx[tx["beta_gw"].gt(0)].copy()

    fig, ax = plt.subplots(figsize=(3.55, 2.35))
    y_min, y_max = 0.7, 760
    ax.set_yscale("log")
    ax.set_xlim(-0.62, len(ISO_ORDER) - 0.38)
    ax.set_ylim(y_min, y_max)
    ax.axhspan(50, y_max, color="#fff2dc", alpha=0.45, zorder=0)
    ax.axhline(50, color="#806b5e", lw=0.7, ls=(0, (3, 2)), zorder=1)

    max_stress = max(float(plot["zone_stress_mid"].quantile(0.98)), 1.0)
    grid_log = np.linspace(np.log10(y_min), np.log10(y_max), 220)
    y_grid = 10 ** grid_log
    rng = np.random.default_rng(5)
    for xpos, iso in enumerate(ISO_ORDER):
        g = plot[plot["iso"].eq(iso)].sort_values("beta_gw").copy()
        if g.empty:
            continue
        vals = g["beta_gw"].clip(lower=0.5).to_numpy()
        logs = np.log10(vals)
        bw = max(0.115, 0.42 * np.nanstd(logs)) if len(logs) > 1 else 0.13
        density = np.exp(-0.5 * ((grid_log[:, None] - logs[None, :]) / bw) ** 2).sum(axis=1)
        density = density / max(float(density.max()), 1e-9)
        cloud_w = 0.31 * density
        color = ISO_COLORS.get(iso, "#999999")
        ax.fill_betweenx(y_grid, xpos - cloud_w, xpos - 0.018, color=color, alpha=0.18, lw=0, zorder=1.5)
        ax.plot(xpos - cloud_w, y_grid, color=color, lw=0.55, alpha=0.45, zorder=2)
        point_density = np.exp(-0.5 * ((logs[:, None] - logs[None, :]) / bw) ** 2).sum(axis=1)
        point_density = point_density / max(float(point_density.max()), 1e-9)
        offsets = rng.uniform(0.15, 0.92, len(g))
        x_sina = xpos + 0.030 + (0.28 * point_density * offsets)
        sizes = 14 + 78 * np.sqrt(np.clip(g["zone_stress_mid"].to_numpy() / max_stress, 0, 1))
        ax.scatter(x_sina, vals, s=sizes, color=color, edgecolor="white", linewidth=0.45, alpha=0.86, zorder=3)
        q10, q25, q50, q75, q90 = np.nanpercentile(vals, [10, 25, 50, 75, 90])
        stat_x = xpos + 0.35
        ax.plot([stat_x, stat_x], [q10, q90], color="#5d514b", lw=0.64, alpha=0.52, zorder=2)
        ax.plot([stat_x, stat_x], [q25, q75], color="#5d514b", lw=1.25, alpha=0.64, zorder=2)
        ax.scatter([stat_x], [q50], s=17, marker="_", color="#2f2925", linewidth=1.05, zorder=4)

    label_df = plot.sort_values("zone_stress_mid", ascending=False).head(6)
    iso_x = {iso: i for i, iso in enumerate(ISO_ORDER)}
    label_slots = {
        1: (2.28, 585),
        2: (2.28, 380),
        3: (4.22, 515),
        4: (0.24, 255),
        5: (2.28, 170),
        6: (0.24, 108),
    }
    for rank, (_, r) in enumerate(label_df.iterrows(), start=1):
        sx = iso_x.get(r["iso"], 0)
        sy = max(float(r["beta_gw"]), 0.5)
        tx_label, ty_label = label_slots.get(
            rank,
            (min(sx + 0.22, len(ISO_ORDER) - 0.82), min(y_max / 1.35, sy * 1.20)),
        )
        ax.annotate("", xy=(sx, sy), xytext=(tx_label, ty_label), arrowprops=dict(arrowstyle="-", color="#9d8069", lw=0.38), zorder=4)
        ax.text(tx_label, ty_label, f"{rank} {r['iso']} {r['zone_short']}", fontsize=5.3, color="#3e342e", ha="left", va="center", zorder=4)

    ax.set_xlabel("ISO planning profile")
    ax.set_ylabel("zone load-price sensitivity ($/MWh per GW)")
    ax.set_xticks(np.arange(len(ISO_ORDER)))
    ax.set_xticklabels(ISO_ORDER, rotation=28, ha="right")
    ax.set_yticks([1, 5, 10, 50, 100, 500])
    ax.set_yticklabels(["1", "5", "10", "50", "100", "500"])
    ax.grid(color=GRID, lw=0.5, which="major", zorder=0)
    clean_spines(ax)
    ax.text(0.02, 0.965, "sina dot area = sensitivity x AI/headroom", transform=ax.transAxes, ha="left", va="top", fontsize=5.55, color="#6b5a4d")
    handles = [Line2D([0], [0], marker="o", lw=0, markerfacecolor=ISO_COLORS[iso], markeredgecolor="white", markersize=4.2, label=iso) for iso in ISO_ORDER]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, -0.015), columnspacing=0.50, handletextpad=0.20, markerscale=0.82)
    fig.subplots_adjust(left=0.155, right=0.995, top=0.980, bottom=0.205)
    return fig


def pjm_zone_exposure() -> pd.DataFrame:
    pjm = read_csv("fig3/fig3f_pjm_zone_ai_exposure_canonical.csv")
    pjm["zone_short"] = pjm["zone"].map(PJM_ZONE_MAP).fillna(pjm["zone"].astype(str).str[:8])
    wide = pjm.pivot_table(index="zone_short", columns=["scenario", "year"], values="ai_demand_MW", fill_value=0.0) / 1000.0
    out = pd.DataFrame(index=wide.index)
    for scenario in ["low", "mid", "high"]:
        for year in [2025, 2035]:
            out[f"{scenario}_{year}_gw"] = wide.get((scenario, year), pd.Series(0.0, index=wide.index))
    out["mid_growth_gw"] = out["mid_2035_gw"] - out["mid_2025_gw"]
    out = out.reset_index().rename(columns={"zone_short": "zone"})
    tx = read_csv("fig3/fig3d_zone_pressure_screening_canonical.csv")
    tx = tx[tx["ISO"].eq("PJM")].copy()
    out = out.merge(tx[["zone", "beta_gw", "hc_tx_mw_medium_dP10"]], on="zone", how="left")
    out["beta_gw"] = out["beta_gw"].fillna(0.0)
    out["stress_mid"] = out["mid_2035_gw"] * out["beta_gw"].clip(lower=0)
    out["stress_low"] = out["low_2035_gw"] * out["beta_gw"].clip(lower=0)
    out["stress_high"] = out["high_2035_gw"] * out["beta_gw"].clip(lower=0)
    return out.sort_values("stress_mid", ascending=False).reset_index(drop=True)


def draw_panel_d() -> plt.Figure:
    df = pjm_zone_exposure().sort_values("stress_mid", ascending=False).head(14).sort_values("stress_mid").copy()
    fig = plt.figure(figsize=(3.55, 2.35))
    ax = fig.add_axes([0.175, 0.165, 0.530, 0.720])
    axr = fig.add_axes([0.750, 0.165, 0.235, 0.720], sharey=ax)
    y = np.arange(len(df))
    max_beta = max(df["beta_gw"].max(), 1)
    cmap = LinearSegmentedColormap.from_list("beta_warm", ["#f6d8b8", "#e3833c", "#b2182b"])
    norm = Normalize(vmin=max(0.1, df["beta_gw"].quantile(0.05)), vmax=df["beta_gw"].quantile(0.95))

    for i, r in enumerate(df.itertuples()):
        color = cmap(norm(max(r.beta_gw, 0.1)))
        for axis in [ax, axr]:
            axis.barh(i, r.mid_2025_gw, color=BASE, edgecolor="none", height=0.56, zorder=2)
            axis.barh(i, max(r.mid_growth_gw, 0), left=r.mid_2025_gw, color=color, edgecolor="none", height=0.56, zorder=2)
            axis.errorbar(r.mid_2035_gw, i, xerr=np.array([[max(0, r.mid_2035_gw - r.low_2035_gw)], [max(0, r.high_2035_gw - r.mid_2035_gw)]]), fmt="none", ecolor=RANGE, elinewidth=0.78, capsize=2.0, zorder=4)
            axis.scatter(r.mid_2025_gw, i, s=16, facecolor="white", edgecolor="#2d2926", linewidth=0.72, zorder=5)
        ax.scatter(-0.76, i, s=20 + 38 * min(r.beta_gw / max_beta, 1), color=color, edgecolor="white", linewidth=0.35, clip_on=False, zorder=5)
        if r.mid_2035_gw > 15:
            axr.text(r.mid_2035_gw + 0.35, i, f"{r.mid_2035_gw:.1f}", fontsize=5.8, va="center", color=TEXT)
        elif r.mid_2035_gw > 0.15:
            ax.text(r.mid_2035_gw + 0.16, i, f"{r.mid_2035_gw:.1f}", fontsize=5.4, va="center", color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(df["zone"], fontsize=6.5)
    ax.set_xlim(-1.05, 6.4)
    axr.set_xlim(18.0, max(31.5, df["high_2035_gw"].max() * 1.04))
    for axis in [ax, axr]:
        axis.grid(axis="x", color=GRID, lw=0.55, zorder=0)
        clean_spines(axis)
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xticks([0, 2, 4, 6])
    axr.set_xticks([20, 25, 30])
    fig.text(0.56, 0.030, "PJM zone AI data-center demand (GW)", ha="center", va="bottom", fontsize=7.4, color="black")
    ax.text(-0.98, len(df) - 0.25, "price-stress\nscreen", fontsize=5.4, ha="left", va="top", color="#6d5141")
    d = 0.010
    kwargs = dict(transform=ax.transAxes, color="#57514c", clip_on=False, lw=0.65)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs = dict(transform=axr.transAxes, color="#57514c", clip_on=False, lw=0.65)
    axr.plot((-d, +d), (-d, +d), **kwargs)
    axr.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    inset = axr.inset_axes([0.02, 0.11, 0.94, 0.36])
    ranked = df.sort_values("stress_mid", ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(ranked) + 1)
    for col, label, color, ls in [("stress_low", "low", "#d79a52", (0, (2, 1.6))), ("stress_mid", "mid", "#bd5a2a", "solid"), ("stress_high", "high", "#8f2d24", (0, (1, 1.3)))]:
        vals = ranked[col].clip(lower=0).to_numpy()
        share = np.cumsum(vals) / vals.sum() * 100 if vals.sum() > 0 else np.zeros_like(vals)
        inset.plot(ranks, share, color=color, lw=0.95, ls=ls, marker="o", markersize=2.1, label=label)
    top3 = ranked.head(3)["stress_mid"].sum() / ranked["stress_mid"].sum() * 100 if ranked["stress_mid"].sum() > 0 else 0
    inset.text(0.05, 0.94, f"top 3 = {top3:.0f}%", transform=inset.transAxes, ha="left", va="top", fontsize=5.5, color="#6b3a2c")
    inset.set_xlim(1, len(ranked))
    inset.set_ylim(0, 104)
    inset.set_xticks([1, 3, 5, 10, len(ranked)])
    inset.set_yticks([25, 50, 75, 100])
    inset.tick_params(labelsize=5.1, length=1.4, width=0.4, pad=1)
    inset.grid(axis="y", color="#efe4db", lw=0.45)
    clean_spines(inset)
    inset.legend(frameon=False, loc="lower right", fontsize=5.2, handlelength=1.1, borderpad=0.1)

    handles = [Patch(fc=BASE, ec="none", label="2025 baseline"), Patch(fc="#d56732", ec="none", label="2035 addition"), Line2D([0], [0], color=RANGE, lw=0.8, label="low-high")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.58, 0.987), ncol=3, columnspacing=0.60, handlelength=0.9, handletextpad=0.28)
    return fig


def national_zone_screen() -> pd.DataFrame:
    ns = read_csv("fig3/fig3e_national_zone_screening_canonical.csv")
    ns = ns.copy()
    ns["zone_short"] = ns["zone"].map(short_zone)
    piv = ns.pivot_table(index=["iso", "zone", "zone_short"], columns="scenario", values="national_zone_screen_index", aggfunc="max").reset_index()
    growth = ns.pivot_table(index=["iso", "zone", "zone_short"], columns="scenario", values="ai_growth_vs_mid2025_GW", aggfunc="max").reset_index()
    out = piv.merge(growth, on=["iso", "zone", "zone_short"], suffixes=("", "_growth"))
    out = out.rename(columns={"low": "screen_low", "mid": "screen_mid", "high": "screen_high", "mid_growth": "ai_growth_mid_gw"})
    if "ai_growth_mid_gw" not in out.columns and "mid_growth" in out.columns:
        out["ai_growth_mid_gw"] = out["mid_growth"]
    out["screen_mid"] = out["screen_mid"].fillna(0)
    out["screen_index_norm"] = out["screen_mid"] / max(out["screen_mid"].max(), 1e-9) * 100.0
    return out.sort_values("screen_mid", ascending=False).reset_index(drop=True)


def draw_panel_e() -> plt.Figure:
    plot = national_zone_screen()
    plot["rank"] = np.arange(1, len(plot) + 1)
    fig = plt.figure(figsize=(3.55, 2.35))
    ax = fig.add_axes([0.135, 0.205, 0.825, 0.735])
    for iso, g in plot.groupby("iso"):
        size_metric = g["ai_growth_mid_gw"] if "ai_growth_mid_gw" in g else 1
        ax.scatter(g["rank"], g["screen_index_norm"], s=17 + 2.2 * np.sqrt(pd.Series(size_metric).clip(lower=0)), color=ISO_COLORS.get(iso, "#999999"), edgecolor="white", linewidth=0.35, alpha=0.88, label=iso, zorder=3)
    ax.plot(plot["rank"], plot["screen_index_norm"], color="#8d7565", lw=0.55, alpha=0.62, zorder=1)
    top = plot.head(6)
    label_pos = {
        1: (2.30, 110),
        2: (3.70, 78),
        3: (5.05, 55),
        4: (6.25, 39),
        5: (7.45, 28),
        6: (8.65, 20),
    }
    for _, r in top.iterrows():
        rank = int(r["rank"])
        tx_label, ty_label = label_pos.get(
            rank,
            (min(rank + 1.0, 8.2), min(118, r["screen_index_norm"] * 1.02)),
        )
        ax.annotate("", xy=(r["rank"], r["screen_index_norm"]), xytext=(tx_label - 0.18, ty_label), arrowprops=dict(arrowstyle="-", color="#9d8069", lw=0.38), zorder=4)
        ax.text(tx_label, ty_label, f"{rank} {r['iso']} {r['zone_short']}", fontsize=5.4, color="#433a35", va="center")
    ax.set_yscale("log")
    ax.set_xlim(0.4, len(plot) + 1.5)
    ax.set_ylim(max(0.08, plot["screen_index_norm"].min() * 0.75), 130)
    ax.set_xlabel("load zones ranked by stress-screening index")
    ax.set_ylabel("screening index (top zone = 100)")
    ax.grid(color=GRID, lw=0.5, which="major", zorder=0)
    clean_spines(ax)

    ax2 = ax.inset_axes([0.620, 0.580, 0.335, 0.335])
    ranks = np.arange(1, len(plot) + 1)
    for col, label, color, ls in [("screen_low", "low AI", "#d99b3a", (0, (2, 1.4))), ("screen_mid", "mid AI", "#bd5a2a", "solid"), ("screen_high", "high AI", "#8f2d24", (0, (1, 1.2)))]:
        vals = plot[col].fillna(0).clip(lower=0).to_numpy()
        share = np.cumsum(vals) / vals.sum() * 100 if vals.sum() > 0 else np.zeros_like(vals)
        ax2.plot(ranks, share, color=color, lw=1.0, ls=ls, marker="o", markersize=2.2, label=label)
    top10 = plot.head(10)["screen_mid"].sum() / plot["screen_mid"].sum() * 100 if plot["screen_mid"].sum() > 0 else 0
    top20 = plot.head(20)["screen_mid"].sum() / plot["screen_mid"].sum() * 100 if plot["screen_mid"].sum() > 0 else 0
    ax2.text(0.07, 0.90, f"top 10 = {top10:.0f}%\ntop 20 = {top20:.0f}%", transform=ax2.transAxes, ha="left", va="top", fontsize=5.2, color="#6b3a2c")
    ax2.set_xlim(1, len(plot))
    ax2.set_ylim(0, 103)
    ax2.tick_params(labelsize=4.8, length=1.5, width=0.4, pad=1)
    ax2.grid(axis="y", color=GRID, lw=0.5)
    clean_spines(ax2)
    ax2.legend(frameon=False, loc="lower right", fontsize=4.9, handlelength=1.2, borderpad=0.1)
    handles = [Line2D([0], [0], marker="o", lw=0, markerfacecolor=ISO_COLORS[iso], markeredgecolor="white", markersize=4.0, label=iso) for iso in ISO_ORDER]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.00, 0.010), ncol=4, fontsize=5.9, columnspacing=0.48, handletextpad=0.18, markerscale=0.78)
    return fig


def save_panel(fig: plt.Figure, name: str) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANEL_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(PANEL_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=600)
    fig.savefig(PANEL_DIR / f"{name}_1200dpi.png", bbox_inches="tight", pad_inches=0.01, dpi=1200)
    plt.close(fig)


def compose_pdf() -> Path:
    build = Path("/private/tmp/fig3_previous_design_vector")
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    for name in ["a", "b", "c", "d", "e"]:
        shutil.copy2(PANEL_DIR / f"{name}.pdf", build / f"{name}.pdf")
    tex = build / "fig3_vector_composite.tex"
    tex.write_text(
        r"""\documentclass{article}
\usepackage[
  paperwidth=7.1in,
  paperheight=7.85in,
  left=0.14in,
  right=0.05in,
  top=0.02in,
  bottom=0.02in
]{geometry}
\usepackage{graphicx}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\fboxsep}{0pt}
\newsavebox{\panelbox}
\newcommand{\paneltext}[1]{\fontsize{8.5}{8.5}\selectfont\bfseries #1}
\newcommand{\panelgraphic}[3]{%
  \sbox{\panelbox}{\includegraphics[width=#1]{#3}}%
  \makebox[0pt][r]{\raisebox{\dimexpr\ht\panelbox-0.060in\relax}{\paneltext{#2}\hspace{0.035in}}}%
  \usebox{\panelbox}%
}
\begin{document}
\panelgraphic{\textwidth}{a}{a.pdf}

\vspace{0.035in}
\begin{minipage}[t]{0.486\textwidth}
\panelgraphic{\linewidth}{b}{b.pdf}
\end{minipage}\hfill
\begin{minipage}[t]{0.486\textwidth}
\panelgraphic{\linewidth}{c}{c.pdf}
\end{minipage}

\vspace{0.045in}
\begin{minipage}[t]{0.486\textwidth}
\panelgraphic{\linewidth}{d}{d.pdf}
\end{minipage}\hfill
\begin{minipage}[t]{0.486\textwidth}
\panelgraphic{\linewidth}{e}{e.pdf}
\end{minipage}
\end{document}
""",
        encoding="utf-8",
    )
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    subprocess.run(cmd, cwd=build, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_pdf = OUT / "Fig3_previous_design_canonical_VECTOR.pdf"
    shutil.copy2(build / "fig3_vector_composite.pdf", out_pdf)
    shutil.copy2(out_pdf, CURRENT)
    return out_pdf


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    core = canonical_core()
    core.to_csv(OUT / "fig3_previous_design_core_canonical.csv", index=False)
    pjm_zone_exposure().to_csv(OUT / "fig3_previous_design_pjm_zone_exposure_source.csv", index=False)
    national_zone_screen().to_csv(OUT / "fig3_previous_design_national_zone_screen_source.csv", index=False)

    save_panel(draw_panel_a(core), "a")
    save_panel(draw_panel_b(core), "b")
    save_panel(draw_panel_c(core), "c")
    save_panel(draw_panel_d(), "d")
    save_panel(draw_panel_e(), "e")
    pdf = compose_pdf()
    print(pdf)
    print(CURRENT)


if __name__ == "__main__":
    main()
