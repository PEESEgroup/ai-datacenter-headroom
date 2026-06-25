from __future__ import annotations

import os
import shutil
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator


TABLE_ARCHIVE = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT = Path(__file__).resolve().parents[2]
TABLE = TABLE_ARCHIVE / "tables" / "fig2_5_canonical_20260514"
OUT = TABLE_ARCHIVE / "figures" / "Fig5_previous_design_canonical_20260514"
CURRENT = TABLE_ARCHIVE / "figures" / "Fig5_onsite_generation_nature_v4_finegrid copy.pdf"

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
FOCUS_ISOS = ["PJM", "MISO", "SPP", "ERCOT"]
ISO_COL = {
    "PJM": "#d83b8c",
    "MISO": "#7870b6",
    "SPP": "#bd8b00",
    "ERCOT": "#df6500",
    "CAISO": "#2aa07f",
    "NYISO": "#4f80ad",
    "ISO-NE": "#7899b7",
    "Non-ISO": "#9a9a9a",
}
COL = {
    "text": "#2f2f2f",
    "muted": "#777777",
    "axis": "#5a5650",
    "grid": "#ece6de",
    "track": "#eee8df",
    "solar": "#c96b2c",
    "season": "#6789b3",
    "value": "#9867a8",
    "stress": "#c85145",
    "deficit": "#d24b3e",
    "hybrid": "#b57b26",
    "co2": "#8f5a3b",
    "green": "#2a9d78",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 5.8,
            "axes.labelsize": 5.7,
            "axes.titlesize": 6.0,
            "xtick.labelsize": 5.1,
            "ytick.labelsize": 5.1,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 800,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(ax, left: bool = True, bottom: bool = True) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color(COL["axis"])
    ax.tick_params(colors=COL["text"], pad=1.2)


def panel_label(ax, label: str, x: float = -0.055, y: float = 1.035) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        fontweight="bold",
        color="black",
    )


def vector_colorbar(ax, cmap, vmin: float, vmax: float, ticks, label: str, n: int = 72, norm_obj=None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(vmin, vmax)
    vals = np.linspace(vmin, vmax, n + 1)
    norm = norm_obj or Normalize(vmin=vmin, vmax=vmax)
    for lo, hi in zip(vals[:-1], vals[1:]):
        ax.add_patch(
            Rectangle(
                (0, lo),
                1,
                hi - lo,
                facecolor=cmap(norm((lo + hi) / 2)),
                edgecolor="none",
                linewidth=0,
            )
        )
    ax.set_xticks([])
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=4.8, length=1.7, pad=1.0, width=0.45)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(label, fontsize=5.0, labelpad=1.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.35)
        s.set_color("#3f3a35")


def load_data() -> dict[str, pd.DataFrame]:
    fig5 = TABLE / "fig5"
    hybrid_fine = pd.read_csv(fig5 / "fig5_hybrid_pv_storage_residual_firm_backstop_canonical_fine.csv")
    for df in [hybrid_fine]:
        if "pv_nameplate_ratio_to_incremental_dc_peak" not in df.columns:
            df["pv_nameplate_ratio_to_incremental_dc_peak"] = df["pv_nameplate_ratio_to_incremental_ai_peak"]
        if "residual_firm_share_of_incremental_dc_peak" not in df.columns:
            df["residual_firm_share_of_incremental_dc_peak"] = df["residual_firm_share_of_incremental_ai_peak"]
        if "incremental_dc_peak_MW" not in df.columns:
            df["incremental_dc_peak_MW"] = df["incremental_ai_peak_MW"]
    out = {
        "sites": pd.read_csv(fig5 / "fig5a_site_level_solar_resource_source.csv"),
        "solar_iso": pd.read_csv(fig5 / "fig5a_iso_solar_resource_summary.csv"),
        "cost": pd.read_csv(fig5 / "fig5_price_cost_official_screen_with_isone.csv"),
        "hybrid": hybrid_fine.copy(),
        "hybrid_fine": hybrid_fine,
        "firm": pd.read_csv(fig5 / "fig5_firm_selfgen_fuel_cost_emissions_canonical.csv"),
        "req": pd.read_csv(fig5 / "fig5_firm_selfgen_requirement_canonical.csv"),
        "zone": pd.read_csv(fig5 / "fig5_zone_price_cost_value_screen_with_isone.csv"),
        "fingerprint": pd.read_csv(fig5 / "fig5b_on_site_generation_fingerprint_canonical.csv"),
    }
    return out


def load_geometries():
    counties = gpd.read_file(PROJECT / "data" / "us_counties.geojson")
    counties = counties[~counties["STATE"].isin(["02", "15", "60", "66", "69", "72", "78"])].copy()
    states = counties.dissolve(by="STATE", as_index=False)
    iso = gpd.read_file(PROJECT / "out" / "iso7_regions_region_only_clean.gpkg")
    iso["ISO"] = iso["ISO"].replace({"ISONE": "ISO-NE", "ISO_NE": "ISO-NE"})
    iso = iso[iso["ISO"].isin(ISO_ORDER)].copy()
    return states.to_crs(5070), iso.to_crs(5070)


def draw_panel_a(ax, cax, data: dict[str, pd.DataFrame]) -> None:
    states, iso = load_geometries()
    sites = data["sites"].copy()
    sites = sites[np.isfinite(sites["lat"]) & np.isfinite(sites["lon"])].copy()
    gsites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs="EPSG:4326",
    ).to_crs(5070)

    states.plot(ax=ax, facecolor="#fbfaf8", edgecolor="#e3dfd8", linewidth=0.22, zorder=0)
    iso.plot(ax=ax, facecolor="#f5f2ec", edgecolor="#ffffff", linewidth=0.45, alpha=0.75, zorder=1)
    iso.boundary.plot(ax=ax, color="#716d66", linewidth=0.55, alpha=0.85, zorder=2)

    solar = gsites["tilted_latitude_annual_kwh_m2_day"].astype(float)
    mw = gsites["site_weight_mw"].fillna(gsites["mw_peak_mw"]).fillna(100.0).astype(float)
    sizes = 5.0 + 28.0 * np.sqrt(np.clip(mw, 0, None) / max(float(mw.max()), 1.0))
    solar_cmap = LinearSegmentedColormap.from_list(
        "site_solar",
        ["#fff3d2", "#efb35a", "#d66a2c", "#8d2f20"],
    )
    norm = Normalize(vmin=3.7, vmax=6.7)
    ax.scatter(
        gsites.geometry.x,
        gsites.geometry.y,
        c=solar,
        s=sizes,
        cmap=solar_cmap,
        norm=norm,
        edgecolors="#fffaf0",
        linewidths=0.28,
        alpha=0.88,
        zorder=4,
    )
    top = gsites.nlargest(8, "site_weight_mw")
    ax.scatter(
        top.geometry.x,
        top.geometry.y,
        s=5.5 + 34.0 * np.sqrt(top["site_weight_mw"].astype(float) / max(float(mw.max()), 1.0)),
        facecolors="none",
        edgecolors="#3c3530",
        linewidths=0.55,
        alpha=0.85,
        zorder=5,
    )
    vector_colorbar(
        cax,
        solar_cmap,
        3.7,
        6.7,
        [4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        "Annual tilted solar resource\n(kWh m$^{-2}$ d$^{-1}$)",
    )

    ref = [200, 1000, 5000, 10000]
    handles = [
        ax.scatter(
            [],
            [],
            s=5.0 + 28.0 * np.sqrt(v / max(float(mw.max()), 1.0)),
            facecolor="#d66a2c",
            edgecolor="#fffaf0",
            linewidth=0.28,
            alpha=0.88,
        )
        for v in ref
    ]
    leg = ax.legend(
        handles,
        ["0.2 GW", "1 GW", "5 GW", "10 GW"],
        title="AI site size",
        loc="lower left",
        bbox_to_anchor=(0.006, 0.040),
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        borderpad=0.22,
        labelspacing=0.12,
        handletextpad=0.35,
        fontsize=4.7,
        title_fontsize=4.8,
    )
    leg.get_frame().set_edgecolor("#ded8cf")
    leg.get_frame().set_linewidth(0.42)

    ax.text(
        0.006,
        0.965,
        "site-level solar resource at AI data-center locations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        fontweight="bold",
        color=COL["text"],
    )
    ax.text(
        0.006,
        0.900,
        f"NASA POWER climatology; n={len(gsites)} sites",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.9,
        color=COL["muted"],
    )
    xmin, ymin, xmax, ymax = states.total_bounds
    ax.set_xlim(xmin - 0.03e5, xmax + 0.05e5)
    ax.set_ylim(ymin + 0.10e5, ymax - 0.05e5)
    ax.set_aspect("equal", adjustable="box", anchor="W")
    ax.axis("off")
    panel_label(ax, "a", x=-0.030, y=0.995)


def build_fingerprint(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "fingerprint" in data:
        df = data["fingerprint"].copy()
        df = df.rename(
            columns={
                "winter_summer_solar_ratio_pct": "winter_summer_pct",
                "retail_minus_pv_lcoe_dollars_mwh": "retail_minus_pv_lcoe",
                "p95_lmp_minus_gas_cost_dollars_mwh": "p95_lmp_minus_gas",
                "firm_gap_2035_GW": "firm_gap_gw_mid2035",
                "pv_100pct_4h_residual_pct": "residual_after_1pv_4h_pct",
            }
        )
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "fig5_v3_panel_b_fingerprint_source.csv", index=False)
        return df
    solar = data["solar_iso"].copy().set_index("ISO")
    cost = data["cost"].copy().set_index("ISO_RTO")
    req = data["req"].copy()
    firm_req = (
        req[(req["scenario"].eq("mid")) & (req["year"].eq(2035))]
        .set_index("ISO_RTO")["firm_onsite_capacity_required_MW"]
        .div(1000.0)
    )
    hybrid = data["hybrid"]
    res = (
        hybrid[
            hybrid["scenario"].eq("mid")
            & np.isclose(hybrid["pv_nameplate_ratio_to_incremental_dc_peak"], 1.0)
            & np.isclose(hybrid["storage_duration_h"], 4.0)
        ]
        .set_index("ISO_RTO")["residual_firm_share_of_incremental_dc_peak"]
        .mul(100.0)
    )
    firm = data["firm"]
    co2 = (
        firm[
            firm["scenario"].eq("mid")
            & np.isclose(firm["capacity_factor_sensitivity"], 0.5)
            & firm["gas_techdetail_atb"].eq("NG 1-on-1 Combined Cycle (H-Frame)")
        ]
        .set_index("ISO_RTO")["annual_co2_million_metric_tons"]
    )

    rows = []
    for iso in ISO_ORDER:
        rows.append(
            {
                "ISO": iso,
                "solar_kwh_m2_day": float(
                    solar.loc[iso, "tilted_latitude_annual_kwh_m2_day_mw_weighted_mean"]
                ),
                "winter_summer_pct": float(
                    solar.loc[iso, "winter_to_summer_tilted_lat_ratio_mw_weighted_mean"] * 100.0
                ),
                "retail_minus_pv_lcoe": float(
                    cost.loc[iso, "commercial_retail_minus_comm_pv_lcoe_median"]
                ),
                "p95_lmp_minus_gas": float(
                    cost.loc[iso, "p95_lmp_minus_gas_variable_cost_median"]
                    if iso in cost.index and pd.notna(cost.loc[iso, "p95_lmp_minus_gas_variable_cost_median"])
                    else np.nan
                ),
                "firm_gap_gw_mid2035": float(firm_req.get(iso, 0.0)),
                "residual_after_1pv_4h_pct": float(res.get(iso, 0.0)),
                "gas_co2_mt_mid_cf50": float(co2.get(iso, 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fig5_v3_panel_b_fingerprint_source.csv", index=False)
    return df


def draw_metric_cell(ax, x, y, value, vmin, vmax, color, fmt, missing: bool = False):
    w = 0.78
    h = 0.56
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="#fbfaf7", edgecolor="#ebe4da", linewidth=0.36))
    if missing:
        ax.text(x, y, "n.a.", ha="center", va="center", fontsize=4.7, color="#9a948c")
        return
    frac = 0.0 if vmax == vmin else (float(value) - vmin) / (vmax - vmin)
    frac = float(np.clip(frac, 0, 1))
    ax.add_patch(Rectangle((x - w / 2 + 0.07, y - 0.045), w - 0.14, 0.09, facecolor=COL["track"], edgecolor="none"))
    ax.add_patch(Rectangle((x - w / 2 + 0.07, y - 0.045), (w - 0.14) * frac, 0.09, facecolor=color, edgecolor="none"))
    ax.plot(x - w / 2 + 0.07 + (w - 0.14) * frac, y, marker="o", ms=2.3, color=color, mec="white", mew=0.25)
    ax.text(x, y + 0.155, fmt.format(value), ha="center", va="center", fontsize=4.85, color=COL["text"])


def draw_panel_b(ax, data: dict[str, pd.DataFrame]) -> None:
    df = build_fingerprint(data)
    metrics = [
        ("Solar\nkWh/m2/d", "solar_kwh_m2_day", COL["solar"], "{:.1f}"),
        ("Winter /\nsummer", "winter_summer_pct", COL["season"], "{:.0f}%"),
        ("Retail -\nPV LCOE", "retail_minus_pv_lcoe", COL["value"], "{:.0f}"),
        ("P95 LMP -\ngas cost", "p95_lmp_minus_gas", COL["stress"], "{:.0f}"),
        ("Firm gap\n2035", "firm_gap_gw_mid2035", COL["deficit"], "{:.1f}"),
        ("PV+4h\nresidual", "residual_after_1pv_4h_pct", COL["hybrid"], "{:.0f}%"),
        ("Gas CO$_2$\nH-frame\nCF=50%", "gas_co2_mt_mid_cf50", COL["co2"], "{:.1f}"),
    ]
    ax.set_xlim(-1.25, len(metrics) - 0.15)
    ax.set_ylim(len(ISO_ORDER) - 0.42, -1.12)
    ax.text(
        0.00,
        1.020,
        "regional on-site generation fingerprint",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=COL["text"],
    )
    for y in np.arange(-0.5, len(ISO_ORDER), 1):
        ax.axhline(y, color="#f0ebe4", lw=0.42, zorder=0)
    for j, (label, key, color, fmt) in enumerate(metrics):
        vals = df[key].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmin = float(np.nanmin(finite)) if len(finite) else 0.0
        vmax = float(np.nanmax(finite)) if len(finite) else 1.0
        if key in {"firm_gap_gw_mid2035", "residual_after_1pv_4h_pct", "gas_co2_mt_mid_cf50"}:
            vmin = 0.0
        for i, row in df.iterrows():
            val = row[key]
            missing = not np.isfinite(float(val)) if pd.notna(val) else True
            if missing:
                ax.text(j, i, "n.a.", ha="center", va="center", fontsize=4.6, color="#9a948c")
                continue
            frac = 0.0 if vmax == vmin else (float(val) - vmin) / (vmax - vmin)
            frac = float(np.clip(frac, 0.02, 1.0))
            size = 28 + 188 * np.sqrt(frac)
            ax.scatter(j, i, s=size, color=color, alpha=0.22 + 0.58 * frac,
                       edgecolor=color, linewidth=0.55, zorder=2)
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=4.7,
                    color=COL["text"], zorder=3)
        ax.text(j, -0.62, label, ha="center", va="bottom", fontsize=4.65, color=COL["text"], linespacing=0.90)

    for i, iso in enumerate(ISO_ORDER):
        ax.text(-1.10, i, iso, ha="left", va="center", fontsize=5.45, color=COL["text"])
        ax.add_patch(Rectangle((-0.61, i - 0.18), 0.07, 0.36, facecolor=ISO_COL[iso], edgecolor="none"))
    ax.text(
        0.00,
        -0.10,
        "Circle size and opacity are normalized within each metric; price columns: $/MWh",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.5,
        color=COL["muted"],
    )
    ax.axis("off")
    panel_label(ax, "b", x=-0.09, y=1.02)


def pivot_residual(hybrid: pd.DataFrame, iso: str, scenario: str) -> pd.DataFrame:
    sub = hybrid[(hybrid["ISO_RTO"].eq(iso)) & (hybrid["scenario"].eq(scenario))].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    sub["pv_pct"] = sub["pv_nameplate_ratio_to_incremental_dc_peak"] * 100.0
    sub["resid_pct"] = sub["residual_firm_share_of_incremental_dc_peak"] * 100.0
    return sub.pivot_table(index="storage_duration_h", columns="pv_pct", values="resid_pct", aggfunc="mean").sort_index().sort_index(axis=1)


def pivot_signed_gap(hybrid: pd.DataFrame, iso: str, scenario: str) -> pd.DataFrame:
    sub = hybrid[(hybrid["ISO_RTO"].eq(iso)) & (hybrid["scenario"].eq(scenario))].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    sub["pv_pct"] = sub["pv_nameplate_ratio_to_incremental_dc_peak"] * 100.0
    sub["signed_gap_pct"] = (
        -sub["available_margin_MW_after_pv_storage"].astype(float)
        / sub["incremental_dc_peak_MW"].replace(0, np.nan).astype(float)
        * 100.0
    )
    return sub.pivot_table(index="storage_duration_h", columns="pv_pct", values="signed_gap_pct", aggfunc="mean").sort_index().sort_index(axis=1)


def pivot_signed_ai_balance(hybrid: pd.DataFrame, iso: str, scenario: str) -> pd.DataFrame:
    sub = hybrid[(hybrid["ISO_RTO"].eq(iso)) & (hybrid["scenario"].eq(scenario))].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    delta = sub["incremental_dc_peak_MW"].replace(0, np.nan).astype(float)
    ai_need = sub["residual_firm_backstop_required_MW"].clip(lower=0).astype(float)
    onsite_surplus = sub["available_margin_MW_after_pv_storage"].clip(lower=0).astype(float)
    sub["pv_pct"] = sub["pv_nameplate_ratio_to_incremental_dc_peak"] * 100.0
    sub["signed_ai_balance_pct"] = (ai_need - onsite_surplus) / delta * 100.0
    return (
        sub.pivot_table(index="storage_duration_h", columns="pv_pct", values="signed_ai_balance_pct", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )


def interpolated_metric(hybrid: pd.DataFrame, iso: str, scenario: str, metric: str = "residual"):
    if metric == "residual":
        piv = pivot_residual(hybrid, iso, scenario)
    elif metric == "signed_ai":
        piv = pivot_signed_ai_balance(hybrid, iso, scenario)
    else:
        piv = pivot_signed_gap(hybrid, iso, scenario)
    x = piv.columns.values.astype(float)
    y = piv.index.values.astype(float)
    if len(x) >= 30 and len(y) >= 12:
        vals = piv.values.astype(float)
        return x, y, np.clip(vals, 0, 100) if metric == "residual" else vals
    xi = np.arange(x.min(), x.max() + 0.01, 5.0)
    yi = np.arange(y.min(), y.max() + 0.01, 0.5)
    interp = RegularGridInterpolator((y, x), piv.values.astype(float), bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(yi, xi, indexing="ij")
    zi = interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(len(yi), len(xi))
    return xi, yi, np.clip(zi, 0, 100) if metric == "residual" else zi


def interpolated_residual(hybrid: pd.DataFrame, iso: str, scenario: str):
    return interpolated_metric(hybrid, iso, scenario, "residual")


def interpolated_signed_gap(hybrid: pd.DataFrame, iso: str, scenario: str):
    return interpolated_metric(hybrid, iso, scenario, "signed")


def interpolated_signed_ai_balance(hybrid: pd.DataFrame, iso: str, scenario: str):
    return interpolated_metric(hybrid, iso, scenario, "signed_ai")


def residual_at_case(hybrid: pd.DataFrame, iso: str, scenario: str, pv_ratio: float, storage_h: float) -> float:
    sub = hybrid[
        hybrid["ISO_RTO"].eq(iso)
        & hybrid["scenario"].eq(scenario)
        & np.isclose(hybrid["pv_nameplate_ratio_to_incremental_dc_peak"], pv_ratio)
        & np.isclose(hybrid["storage_duration_h"], storage_h)
    ].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    if sub.empty:
        return np.nan
    return float(sub["residual_firm_share_of_incremental_dc_peak"].iloc[0] * 100.0)


def signed_ai_balance_at_case(hybrid: pd.DataFrame, iso: str, scenario: str, pv_ratio: float, storage_h: float) -> float:
    sub = hybrid[
        hybrid["ISO_RTO"].eq(iso)
        & hybrid["scenario"].eq(scenario)
        & np.isclose(hybrid["pv_nameplate_ratio_to_incremental_dc_peak"], pv_ratio)
        & np.isclose(hybrid["storage_duration_h"], storage_h)
    ].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    if sub.empty:
        return np.nan
    row = sub.iloc[0]
    delta = float(row["incremental_dc_peak_MW"])
    if delta <= 0:
        return np.nan
    ai_need = max(0.0, float(row["residual_firm_backstop_required_MW"]))
    onsite_surplus = max(0.0, float(row["available_margin_MW_after_pv_storage"]))
    return (ai_need - onsite_surplus) / delta * 100.0


def signed_gap_at_case(hybrid: pd.DataFrame, iso: str, scenario: str, pv_ratio: float, storage_h: float) -> float:
    sub = hybrid[
        hybrid["ISO_RTO"].eq(iso)
        & hybrid["scenario"].eq(scenario)
        & np.isclose(hybrid["pv_nameplate_ratio_to_incremental_dc_peak"], pv_ratio)
        & np.isclose(hybrid["storage_duration_h"], storage_h)
    ].copy()
    if "year" in sub.columns:
        sub = sub[sub["year"].eq(2035)].copy()
    if "storage_power_ratio_to_pv_capacity" in sub.columns:
        sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
    if sub.empty:
        return np.nan
    row = sub.iloc[0]
    delta = float(row["incremental_dc_peak_MW"])
    return -float(row["available_margin_MW_after_pv_storage"]) / delta * 100.0 if delta > 0 else np.nan


def pv_threshold_at_storage(hybrid: pd.DataFrame, iso: str, scenario: str, storage_h: float, target: float = 1.0) -> float:
    piv = pivot_residual(hybrid, iso, scenario)
    h = piv.index.values.astype(float)
    h_idx = int(np.argmin(np.abs(h - storage_h)))
    row = piv.iloc[h_idx].dropna()
    xs = row.index.values.astype(float)
    ys = row.values.astype(float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if len(xs) == 0:
        return np.nan
    if ys[0] <= target:
        return float(xs[0])
    for i in range(len(xs) - 1):
        if ys[i] >= target >= ys[i + 1] and ys[i] != ys[i + 1]:
            return float(xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
        if ys[i + 1] <= target:
            return float(xs[i + 1])
    return np.nan


def draw_panel_c(fig, spec, data: dict[str, pd.DataFrame]) -> None:
    subgs = spec.subgridspec(2, 3, hspace=0.30, wspace=0.18, width_ratios=[1, 1, 0.035])
    hybrid_c = data.get("hybrid_fine", data["hybrid"])
    hybrid_c.to_csv(OUT / "fig5_v3_panel_c_fine_pv_storage_surface_source.csv", index=False)
    cmap = LinearSegmentedColormap.from_list(
        "ai_signed_residual_surplus",
        ["#117264", "#65b7aa", "#f7f3e8", "#efa27b", "#b2182b"],
    )
    gap_vmin, gap_vmax = -250, 100
    norm = TwoSlopeNorm(vmin=gap_vmin, vcenter=0, vmax=gap_vmax)
    axes = []
    for k, iso in enumerate(FOCUS_ISOS):
        ax = fig.add_subplot(subgs[k // 2, k % 2])
        axes.append(ax)
        x, y, vals = interpolated_signed_ai_balance(hybrid_c, iso, "mid")
        vals_plot = np.clip(vals, gap_vmin, gap_vmax)
        xx, yy = np.meshgrid(x, y)
        ax.contourf(
            xx,
            yy,
            vals_plot,
            levels=np.linspace(gap_vmin, gap_vmax, 58),
            cmap=cmap,
            norm=norm,
            antialiased=True,
        )
        line_specs = [
            ("low", "#577083", (0, (4.0, 2.2)), 0.82),
            ("mid", "#252525", "solid", 1.05),
            ("high", "#9d3d32", (0, (1.2, 1.6)), 0.95),
        ]
        for scenario, color, ls, lw in line_specs:
            _, _, zz = interpolated_residual(hybrid_c, iso, scenario)
            try:
                ax.contour(xx, yy, zz, levels=[1], colors=color, linewidths=lw, linestyles=[ls], alpha=0.96)
            except Exception:
                pass
        _, _, signed_vals = interpolated_signed_ai_balance(hybrid_c, iso, "mid")
        try:
            cs_neg = ax.contour(
                xx,
                yy,
                signed_vals,
                levels=[-200, -150, -100, -50, -25],
                colors="#5f746d",
                linewidths=0.32,
                linestyles="dotted",
            )
            ax.clabel(cs_neg, inline=True, fmt=lambda v: f"{int(v):d}", fontsize=3.8)
        except Exception:
            pass
        try:
            cs_pos = ax.contour(xx, yy, vals, levels=[25, 50, 75], colors="#7a7167", linewidths=0.35, linestyles="dotted")
            ax.clabel(cs_pos, inline=True, fmt="%d", fontsize=3.8)
        except Exception:
            pass
        ax.axvline(100, color="#6d655d", lw=0.42, ls=(0, (2.0, 2.2)), alpha=0.62, zorder=2)
        ax.axhline(4, color="#6d655d", lw=0.42, ls=(0, (2.0, 2.2)), alpha=0.62, zorder=2)
        ax.scatter([100], [4], s=9, facecolor="white", edgecolor="#2f2f2f", linewidth=0.45, zorder=5)
        bench = residual_at_case(hybrid_c, iso, "mid", 1.0, 4.0)
        signed_bench = signed_ai_balance_at_case(hybrid_c, iso, "mid", 1.0, 4.0)
        thresh = pv_threshold_at_storage(hybrid_c, iso, "mid", 4.0, target=1.0)
        thresh_txt = ">200%" if not np.isfinite(thresh) else f"{thresh:.0f}%"
        balance_txt = "n.a." if not np.isfinite(signed_bench) else f"{signed_bench:+.0f}%"
        if iso != "ERCOT":
            ax.text(
                0.985,
                0.765,
                f"4h <=1%: {thresh_txt}\n100%+4h: {balance_txt}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=4.05,
                color=COL["text"],
                linespacing=0.90,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.6),
            )
        if iso == "ERCOT":
            ax.text(
                0.52,
                0.50,
                f"AI residual = 0%\n100%+4h: {signed_bench:+.0f}%",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=5.2,
                color="#3b6d5d",
                fontweight="bold",
                linespacing=0.9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.58, pad=0.8),
            )
        ax.set_title(iso, loc="left", fontsize=5.8, color=ISO_COL[iso], fontweight="bold", pad=1.0)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 12)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([0, 4, 8, 12])
        if k // 2 == 1:
            ax.set_xlabel("PV nameplate / AI peak (%)", fontsize=5.0, labelpad=1.0)
        else:
            ax.set_xticklabels([])
        if k % 2 == 0:
            ax.set_ylabel("storage duration (h)", fontsize=5.0, labelpad=1.0)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=4.7, pad=1.0)
        for s in ax.spines.values():
            s.set_linewidth(0.45)
            s.set_color(COL["axis"])
    axes[0].text(-0.075, 1.16, "c", transform=axes[0].transAxes, fontsize=8.1, fontweight="bold")
    axes[0].text(
        0.00,
        1.16,
        "PV+storage AI-residual surface",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=COL["text"],
    )
    axes[1].legend(
        [
            Line2D([0], [0], color="#577083", lw=0.82, ls=(0, (4.0, 2.2))),
            Line2D([0], [0], color="#252525", lw=1.05, ls="solid"),
            Line2D([0], [0], color="#9d3d32", lw=0.95, ls=(0, (1.2, 1.6))),
        ],
        ["low", "mid", "high"],
        title="1% residual contour",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.31),
        fontsize=4.5,
        title_fontsize=4.6,
        handlelength=1.6,
    )
    cax = fig.add_subplot(subgs[:, 2])
    cpos = cax.get_position()
    cax.set_position([cpos.x0 - 0.010, cpos.y0 + 0.012 * cpos.height, cpos.width * 0.82, cpos.height * 0.972])
    vector_colorbar(
        cax,
        cmap,
        gap_vmin,
        gap_vmax,
        [-250, -150, -50, 0, 50, 100],
        "AI residual (+) / screened buffer (-)\n(% AI peak)",
        norm_obj=norm,
    )


def build_selected_uncertainty(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cases = [
        ("none", 0.0, 0.0),
        ("50% PV + 4h", 0.5, 4.0),
        ("100% PV + 4h", 1.0, 4.0),
        ("150% PV + 8h", 1.5, 8.0),
    ]
    rows = []
    hy = data.get("hybrid_fine", data["hybrid"])
    for iso in FOCUS_ISOS:
        for label, pv, h in cases:
            sub = hy[
                hy["ISO_RTO"].eq(iso)
                & np.isclose(hy["pv_nameplate_ratio_to_incremental_dc_peak"], pv)
                & np.isclose(hy["storage_duration_h"], h)
            ].copy()
            if "year" in sub.columns:
                sub = sub[sub["year"].eq(2035)].copy()
            if "storage_power_ratio_to_pv_capacity" in sub.columns:
                sub = sub[np.isclose(sub["storage_power_ratio_to_pv_capacity"], 0.5)].copy()
            vals = {}
            for _, r in sub.iterrows():
                delta = float(r["incremental_dc_peak_MW"])
                vals[r["scenario"]] = (
                    -float(r["available_margin_MW_after_pv_storage"]) / delta * 100.0
                    if delta > 0
                    else np.nan
                )
            rows.append(
                {
                    "ISO": iso,
                    "case": label,
                    "pv_ratio": pv,
                    "storage_h": h,
                    "low": vals.get("low", np.nan),
                    "mid": vals.get("mid", np.nan),
                    "high": vals.get("high", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fig5_v3_panel_d_selected_uncertainty_source.csv", index=False)
    return out


def draw_panel_d(ax, data: dict[str, pd.DataFrame]) -> None:
    df = build_selected_uncertainty(data)
    case_order = ["none", "50% PV + 4h", "100% PV + 4h", "150% PV + 8h"]
    cmap = LinearSegmentedColormap.from_list("d_signed", ["#1d7f6e", "#8fcbbd", "#f7f3e8", "#efa27b", "#b2182b"])
    norm = TwoSlopeNorm(vmin=-220, vcenter=0, vmax=100)
    x = np.arange(len(case_order))
    y = np.arange(len(FOCUS_ISOS))
    for i, iso in enumerate(FOCUS_ISOS):
        sub = df[df["ISO"].eq(iso)].set_index("case").loc[case_order]
        for j, case in enumerate(case_order):
            mid = float(sub.loc[case, "mid"])
            hi = float(sub.loc[case, "high"])
            lo = float(sub.loc[case, "low"])
            color = cmap(norm(float(np.clip(mid, -220, 100))))
            spread = abs(hi - lo)
            s_hi = 52 + 230 * np.sqrt(min(max(abs(hi), abs(lo)), 220) / 220)
            s_mid = 42 + 330 * np.sqrt(min(abs(mid), 220) / 220)
            ax.scatter(j, i, s=s_hi, facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.55, zorder=1)
            ax.scatter(j, i, s=s_mid, facecolor=color, edgecolor="white", alpha=0.92, linewidth=0.45, zorder=2)
            if spread > 2:
                ax.plot([j - 0.22, j + 0.22], [i, i], color="#5b5550", lw=0.60, alpha=0.70, zorder=3)
            label = "0" if abs(mid) < 0.5 else f"{mid:+.0f}"
            text_color = "white" if abs(mid) >= 55 else COL["text"]
            ax.text(j, i, label, ha="center", va="center", fontsize=4.45, color=text_color, zorder=4)
    for xi in np.arange(-0.5, len(case_order), 1):
        ax.axvline(xi + 0.5, color="#f0ebe4", lw=0.42, zorder=0)
    for yi in np.arange(-0.5, len(FOCUS_ISOS), 1):
        ax.axhline(yi + 0.5, color="#f0ebe4", lw=0.42, zorder=0)
    ax.set_xlim(-0.85, len(case_order) - 0.45)
    ax.set_ylim(len(FOCUS_ISOS) - 0.45, -0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "50% PV\n+4h", "100% PV\n+4h", "150% PV\n+8h"], fontsize=4.6)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    for i, iso in enumerate(FOCUS_ISOS):
        ax.text(
            -0.78,
            i,
            iso,
            ha="left",
            va="center",
            fontsize=5.05,
            color=ISO_COL[iso],
            fontweight="bold",
        )
    clean_axis(ax, left=False, bottom=False)
    panel_label(ax, "d")
    ax.set_title("net backstop gap by build-out", loc="left", pad=2.0, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="#b2182b", lw=0, ms=5.5, alpha=0.75, label="+ backstop"),
        Line2D([0], [0], marker="o", color="#1d7f6e", lw=0, ms=5.5, alpha=0.75, label="- buffer"),
        Line2D([0], [0], color="#5b5550", lw=0.65, label="low-high"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        ncol=1,
        columnspacing=0.55,
        handlelength=1.2,
        handletextpad=0.35,
        fontsize=4.25,
    )


def build_gas_frontier(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    firm = data["firm"].copy()
    firm = firm[firm["ISO_RTO"].isin(ISO_ORDER)].copy()
    firm["capacity_gw"] = firm["firm_onsite_capacity_required_MW"] / 1000.0
    firm = firm[firm["gas_techdetail_atb"].eq("NG 1-on-1 Combined Cycle (H-Frame)")].copy()
    df = firm[
        firm["scenario"].isin(["low", "mid", "high"])
        & firm["capacity_factor_sensitivity"].isin([0.1, 0.5, 0.9])
    ].copy()
    df["annual_variable_cost_billion_dollars"] = (
        df["variable_fuel_plus_vom_cost_dollars_mwh"] * df["annual_generation_mwh"] / 1e9
    )
    df.to_csv(OUT / "fig5_v3_panel_e_gas_backstop_source.csv", index=False)
    return df


def draw_panel_e(ax, data: dict[str, pd.DataFrame]) -> None:
    df = build_gas_frontier(data)
    iso_order = ["PJM", "MISO", "SPP"]
    scen_order = ["low", "mid", "high"]
    x_lookup = {s: i for i, s in enumerate(scen_order)}
    cf50 = df[
        df["ISO_RTO"].isin(iso_order)
        & df["scenario"].isin(scen_order)
        & np.isclose(df["capacity_factor_sensitivity"], 0.5)
    ].copy()
    max_cost = max(float(cf50["annual_variable_cost_billion_dollars"].max()), 1e-9)
    for y in np.arange(-0.5, len(iso_order), 1):
        ax.axhline(y + 0.5, color="#f0ebe4", lw=0.42, zorder=0)
    for x in np.arange(-0.5, len(scen_order), 1):
        ax.axvline(x + 0.5, color="#f0ebe4", lw=0.42, zorder=0)
    for i, iso in enumerate(iso_order):
        color = ISO_COL[iso]
        sub = df[df["ISO_RTO"].eq(iso)]
        for scenario in scen_order:
            seg = sub[sub["scenario"].eq(scenario)].set_index("capacity_factor_sensitivity")
            if seg.empty:
                continue
            row50 = seg.loc[0.5]
            row90 = seg.loc[0.9]
            x = x_lookup[scenario]
            co2_50 = float(row50["annual_co2_million_metric_tons"])
            co2_90 = float(row90["annual_co2_million_metric_tons"])
            cost = float(row50["annual_variable_cost_billion_dollars"])
            cap = float(row50["capacity_gw"])
            halo = 40 + 7.5 * max(co2_90, 0)
            core = 24 + 8.8 * max(co2_50, 0)
            ax.scatter(x, i, s=halo, facecolor=color, edgecolor=color, linewidth=0.45, alpha=0.13, zorder=1)
            ax.scatter(x, i, s=core, facecolor=color, edgecolor="white", linewidth=0.55, alpha=0.88, zorder=2)
            ax.text(x, i - 0.035, f"{co2_50:.0f}", ha="center", va="center", fontsize=4.1, color="white", zorder=3)
            ybar = i + 0.295
            track_w = 0.43
            bar_w = 0.055 + 0.375 * cost / max_cost
            ax.add_patch(
                Rectangle(
                    (x - track_w / 2, ybar - 0.026),
                    track_w,
                    0.052,
                    facecolor="#f0ebe4",
                    edgecolor="none",
                    alpha=0.95,
                    zorder=1.5,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x - track_w / 2, ybar - 0.026),
                    bar_w,
                    0.052,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.52,
                    zorder=2.5,
                )
            )
            ax.text(
                x + track_w / 2 + 0.035,
                ybar,
                f"${cost:.2f}",
                ha="left",
                va="center",
                fontsize=3.35,
                color=COL["text"],
                zorder=3,
            )
            ax.text(
                x + 0.18,
                i - 0.235,
                f"{cap:.1f}GW",
                ha="left",
                va="center",
                fontsize=3.4,
                color=color,
                alpha=0.92,
                zorder=3,
            )
    ax.set_xlim(-0.55, len(scen_order) - 0.45)
    ax.set_ylim(len(iso_order) - 0.55, -0.55)
    ax.set_xticks(np.arange(len(scen_order)))
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_yticks(np.arange(len(iso_order)))
    ax.set_yticklabels(iso_order)
    for tick, iso in zip(ax.get_yticklabels(), iso_order):
        tick.set_color(ISO_COL[iso])
        tick.set_fontweight("bold")
    ax.set_xlabel("AI growth scenario")
    clean_axis(ax, left=False, bottom=True)
    panel_label(ax, "e")
    ax.set_title("gas backstop CO$_2$-cost envelope", loc="left", pad=2.0, fontweight="bold")


def kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(grid)
    if len(values) == 1:
        bw = max(abs(values[0]) * 0.08, 4.0)
    else:
        std = np.std(values, ddof=1)
        bw = max(1.06 * std * len(values) ** (-1 / 5), 3.0)
    z = (grid[:, None] - values[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))
    return dens


def draw_panel_f(ax, data: dict[str, pd.DataFrame]) -> None:
    zone = data["zone"].copy()
    zone = zone[zone["ISO_RTO"].isin(["CAISO", "ERCOT", "ISO-NE", "MISO", "NYISO", "PJM", "SPP"])].copy()
    zone["spread"] = zone["p95_lmp_minus_gas_variable_cost"].astype(float)
    order = [iso for iso in ["ISO-NE", "NYISO", "PJM", "ERCOT", "MISO", "SPP", "CAISO"] if iso in set(zone["ISO_RTO"])]
    xmin = max(0.0, float(zone["spread"].min()) - 8.0)
    xmax = max(150.0, float(zone["spread"].max()) + 10.0)
    grid = np.linspace(xmin, xmax, 300)
    rng = np.random.default_rng(7)
    for i, iso in enumerate(order):
        vals = zone[zone["ISO_RTO"].eq(iso)]["spread"].dropna().to_numpy()
        color = ISO_COL[iso]
        dens = kde_1d(vals, grid)
        if dens.max() > 0:
            width = dens / dens.max() * 0.36
            ax.fill_between(grid, i, i + width, color=color, alpha=0.24, lw=0)
            ax.plot(grid, i + width, color=color, lw=0.75, alpha=0.85)
        jitter = rng.normal(0, 0.035, len(vals))
        ax.scatter(vals, np.full(len(vals), i) - 0.11 + jitter, s=12, color=color, edgecolor="white", linewidth=0.25, alpha=0.82, zorder=3)
        if len(vals):
            med = float(np.nanmedian(vals))
            ax.plot([med, med], [i - 0.23, i + 0.30], color="#3d3833", lw=0.55, zorder=4)
            ax.text(xmax - 2, i + 0.17, f"n={len(vals)}", ha="right", va="center", fontsize=4.6, color=COL["muted"])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    for tick, iso in zip(ax.get_yticklabels(), order):
        tick.set_color(ISO_COL[iso])
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("zone P95 LMP - gas variable cost ($/MWh)")
    ax.grid(axis="x", color=COL["grid"], lw=0.45)
    clean_axis(ax)
    panel_label(ax, "f")
    ax.set_title("zone-level scarcity value", loc="left", pad=2.0, fontweight="bold")
    if "ISO-NE" not in set(zone["ISO_RTO"]):
        ax.text(
            0.98,
            0.04,
            "ISO-NE: no zonal LMP panel",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=4.7,
            color=COL["muted"],
        )
    zone.to_csv(OUT / "fig5_v3_panel_f_zone_price_raincloud_source.csv", index=False)


def draw_figure() -> tuple[Path, Path]:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()

    fig = plt.figure(figsize=(7.10, 6.78), facecolor="white", constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=6,
        left=0.038,
        right=0.976,
        top=0.970,
        bottom=0.066,
        height_ratios=[1.056, 1.08, 1.00],
        width_ratios=[1.34, 1.34, 1.34, 0.98, 0.98, 0.98],
        hspace=0.275,
        wspace=0.285,
    )

    ax_a = fig.add_subplot(gs[0, :3])
    pos_a = ax_a.get_position()
    cax_a = fig.add_axes([pos_a.x1 - 0.044, pos_a.y0 + 0.20 * pos_a.height, 0.009, 0.56 * pos_a.height])
    draw_panel_a(ax_a, cax_a, data)

    ax_b = fig.add_subplot(gs[0, 3:])
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0 - 0.004, pos_b.y0, pos_b.width + 0.004, pos_b.height])
    draw_panel_b(ax_b, data)

    draw_panel_c(fig, gs[1, :], data)

    ax_d = fig.add_subplot(gs[2, :2])
    draw_panel_d(ax_d, data)
    ax_e = fig.add_subplot(gs[2, 2:4])
    draw_panel_e(ax_e, data)
    ax_f = fig.add_subplot(gs[2, 4:])
    draw_panel_f(ax_f, data)

    pdf = OUT / "Fig5_onsite_generation_nature_v4_finegrid.pdf"
    png = OUT / "Fig5_onsite_generation_nature_v4_finegrid.png"
    svg = OUT / "Fig5_onsite_generation_nature_v4_finegrid.svg"
    fig.savefig(pdf)
    fig.savefig(png, dpi=800)
    fig.savefig(svg)
    shutil.copy2(pdf, CURRENT)
    (OUT / Path(__file__).name).write_text(Path(__file__).read_text(), encoding="utf-8")
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    pdf_path, png_path = draw_figure()
    print(pdf_path)
    print(png_path)
