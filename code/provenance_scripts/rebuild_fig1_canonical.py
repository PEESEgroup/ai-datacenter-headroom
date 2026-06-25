from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
BASE_FIG1_SCRIPT = Path(__file__).resolve().parents[2] / "code" / "provenance_scripts" / "make_fig1_optimized_nature_layout.py"
OUT_DIR = TABLE_ARCHIVE_ROOT / "figures" / "Fig1_canonical_20260514"
TABLE_DIR = TABLE_ARCHIVE_ROOT / "tables" / "fig1_canonical_20260514"

SCENARIOS = ("low", "mid", "high")
YEARS = (2025, 2035)
DEMAND_TAG = "20260407_152405"
ISO7 = ("CAISO", "ERCOT", "MISO", "PJM", "NYISO", "ISO-NE", "SPP")
ISO_ORDER = ["PJM", "SPP", "ERCOT", "MISO", "CAISO", "NYISO", "ISO-NE"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_iso(value: object) -> str:
    if pd.isna(value):
        return "Non-ISO"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "Non-ISO"
    if text.upper() in {"ISONE", "ISO NE", "ISO-NE", "ISO_NE"}:
        return "ISO-NE"
    if text.upper() in {"NON-ISO", "NON ISO", "NONISO"}:
        return "Non-ISO"
    return text if text in set(ISO7) else "Non-ISO"


def scenario_path(scenario: str) -> Path:
    path = (
        PROJECT_ROOT
        / "out_std"
        / "DC"
        / "demand_dynamic"
        / f"dc_demand_county_year_MWpeak_2025_2035_LF0.8_AeffBaseline_{DEMAND_TAG}_{scenario}.csv"
    )
    if path.exists():
        return path
    pattern = f"dc_demand_county_year_MWpeak_2025_2035_LF0.8_AeffBaseline_*_{scenario}.csv"
    files = sorted(path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No county demand file found for {scenario}")
    return files[0]


def national_average_mw() -> pd.DataFrame:
    curve = pd.read_csv(PROJECT_ROOT / "out_std" / "ANALYSIS" / "national_dc_twh_high_mid_low_2025_2035.csv")
    rows = []
    for _, row in curve.iterrows():
        year = int(row["year"])
        if year not in YEARS:
            continue
        for scenario in SCENARIOS:
            twh = float(row[f"TWh_{scenario}"])
            rows.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "national_TWh": twh,
                    "national_average_GW": twh / 8.76,
                    "national_average_MW": twh * 1_000_000.0 / 8760.0,
                }
            )
    return pd.DataFrame(rows)


def gamma_map() -> dict[str, float]:
    gamma = pd.read_csv(PROJECT_ROOT / "out_std" / "ANALYSIS" / "iso_peak_conversion_factors.csv")
    gamma = gamma[gamma["dc_shape_name"].astype(str).str.lower().eq("avg")].copy()
    out = {norm_iso(row["ISO"]): float(row["Gamma"]) for _, row in gamma.iterrows()}
    out["Non-ISO"] = 1.0
    return out


def load_county_meta(base) -> pd.DataFrame:
    meta = base.fig1county.load_county_metadata(PROJECT_ROOT)
    meta["GEOID"] = meta["GEOID"].astype(str).str.zfill(5)
    meta["state_abb"] = meta["GEOID"].str[:2].map(base.fig1county.STATE_ABBR).fillna(meta["state_abb"])
    meta["iso_meta"] = meta["iso"].map(norm_iso)
    return meta[["GEOID", "state_abb", "label", "iso_meta"]].drop_duplicates("GEOID")


def load_iso_assignment(geo: dict[str, object]) -> pd.DataFrame:
    iso = geo["iso_assign"][["GEOID", "ISO_ASSIGNED"]].copy()
    iso["GEOID"] = iso["GEOID"].astype(str).str.zfill(5)
    iso["iso"] = iso["ISO_ASSIGNED"].map(norm_iso)
    return iso[["GEOID", "iso"]].drop_duplicates("GEOID")


def canonical_county_data(base, geo: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nat = national_average_mw()
    gammas = gamma_map()
    iso_assign = load_iso_assignment(geo)
    meta = load_county_meta(base)

    pieces = []
    audit_rows = []
    for scenario in SCENARIOS:
        source_path = scenario_path(scenario)
        raw = pd.read_csv(source_path, dtype={"GEOID": str})
        raw["GEOID"] = raw["GEOID"].astype(str).str.zfill(5)
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype(int)
        raw["MW_peak_total"] = pd.to_numeric(raw["MW_peak_total"], errors="coerce").fillna(0.0)
        raw = raw[raw["year"].isin(YEARS)].copy()
        raw = raw.merge(iso_assign, on="GEOID", how="left").merge(meta, on="GEOID", how="left")
        raw["iso"] = raw["iso"].combine_first(raw["iso_meta"]).map(norm_iso)
        raw["state_abb"] = raw["state_abb"].fillna(raw["GEOID"].str[:2].map(base.fig1county.STATE_ABBR))
        raw["label"] = raw["label"].fillna(raw["GEOID"])
        raw["gamma"] = raw["iso"].map(gammas).fillna(1.0)

        for year in YEARS:
            subset = raw[raw["year"].eq(year)].copy()
            total_old = float(subset["MW_peak_total"].sum())
            if total_old <= 0:
                raise ValueError(f"{source_path.name} has zero national total in {year}")
            nat_row = nat[(nat["scenario"].eq(scenario)) & (nat["year"].eq(year))].iloc[0]
            subset["raw_spatial_share"] = subset["MW_peak_total"] / total_old
            subset["MW_canonical"] = (
                subset["raw_spatial_share"] * float(nat_row["national_average_MW"]) * subset["gamma"]
            )
            subset["GW_canonical"] = subset["MW_canonical"] / 1000.0
            subset["scenario"] = scenario
            pieces.append(
                subset[
                    [
                        "GEOID",
                        "year",
                        "scenario",
                        "iso",
                        "state_abb",
                        "label",
                        "raw_spatial_share",
                        "MW_peak_total",
                        "MW_canonical",
                        "GW_canonical",
                        "gamma",
                    ]
                ]
            )
            iso7_old = subset[subset["iso"].isin(ISO7)]["MW_peak_total"].sum()
            iso7_new = subset[subset["iso"].isin(ISO7)]["MW_canonical"].sum()
            audit_rows.append(
                {
                    "scenario": scenario,
                    "year": year,
                    "source_file": source_path.name,
                    "old_source_peak_GW_total": total_old / 1000.0,
                    "raw_ISO7_share": iso7_old / total_old,
                    "new_national_average_GW": float(nat_row["national_average_GW"]),
                    "new_ISO7_coincident_GW": iso7_new / 1000.0,
                    "new_nonISO_coincident_GW": subset[~subset["iso"].isin(ISO7)]["MW_canonical"].sum() / 1000.0,
                    "gamma_note": "ISO uses avg 3h coincidence Gamma; non-ISO uses 1.0 because no ISO peak window is defined.",
                }
            )

    county = pd.concat(pieces, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return county, audit


def rank_counties(county: pd.DataFrame) -> pd.DataFrame:
    use = county[county["scenario"].eq("mid") & county["year"].isin(YEARS)].copy()
    use["gw"] = use["GW_canonical"]
    ranked = []
    for year, group in use.groupby("year"):
        group = group[group["gw"] > 0].sort_values("gw", ascending=False).copy()
        group["rank"] = np.arange(1, len(group) + 1)
        ranked.append(group)
    return pd.concat(ranked, ignore_index=True)


def build_iso_tables(county: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    mid2025 = (
        county[county["scenario"].eq("mid") & county["year"].eq(2025) & county["iso"].isin(ISO7)]
        .groupby("iso", as_index=False)["GW_canonical"]
        .sum()
        .rename(columns={"iso": "ISO_ASSIGNED", "GW_canonical": "GW_mid"})
    )
    mid2025["GW_low"] = mid2025["GW_mid"]
    mid2025["GW_high"] = mid2025["GW_mid"]
    mid2025["MW_mid"] = mid2025["GW_mid"] * 1000.0
    mid2025["MW_low"] = mid2025["MW_mid"]
    mid2025["MW_high"] = mid2025["MW_mid"]
    mid2025["MW_2025"] = mid2025["MW_mid"]
    mid2025["GW_2025"] = mid2025["GW_mid"]
    mid2025["year"] = 2025
    mid2025 = mid2025.sort_values("GW_mid", ascending=False).reset_index(drop=True)

    rows2035 = []
    for iso in ISO7:
        row = {"ISO_ASSIGNED": iso, "year": 2035}
        for scenario in SCENARIOS:
            val = county[
                county["scenario"].eq(scenario) & county["year"].eq(2035) & county["iso"].eq(iso)
            ]["GW_canonical"].sum()
            row[f"GW_{scenario}"] = float(val)
        rows2035.append(row)
    iso2035 = pd.DataFrame(rows2035).sort_values("GW_mid", ascending=False).reset_index(drop=True)

    iso_data = iso2035.rename(
        columns={"GW_low": "gw_2035_low", "GW_mid": "gw_2035", "GW_high": "gw_2035_high"}
    ).merge(
        mid2025[["ISO_ASSIGNED", "GW_mid"]].rename(columns={"GW_mid": "gw_2025"}),
        on="ISO_ASSIGNED",
        how="left",
    )
    for col in ["gw_2025", "gw_2035_low", "gw_2035", "gw_2035_high"]:
        iso_data[col] = pd.to_numeric(iso_data[col], errors="coerce").fillna(0.0)
    iso_data["growth"] = (iso_data["gw_2035"] - iso_data["gw_2025"]).clip(lower=0.0)
    iso_data["share_2035"] = iso_data["gw_2035"] / iso_data["gw_2035"].sum()
    iso_data["order"] = iso_data["ISO_ASSIGNED"].map({iso: idx for idx, iso in enumerate(ISO_ORDER)})
    iso_data = iso_data.sort_values("order").reset_index(drop=True)

    conc_rows = []
    for year in YEARS:
        if year == 2025:
            vals = mid2025["GW_mid"].sort_values(ascending=False).to_numpy()
        else:
            vals = iso2035["GW_mid"].sort_values(ascending=False).to_numpy()
        total = float(vals.sum())
        for top_n in range(1, len(vals) + 1):
            conc_rows.append({"year": year, "top_n": top_n, "share": vals[:top_n].sum() / total})
    iso_conc = pd.DataFrame(conc_rows)
    return {"iso_2025": mid2025, "iso_2035": iso2035}, iso_data, iso_conc


def build_state_tables(base, county: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mid2025 = (
        county[county["scenario"].eq("mid") & county["year"].eq(2025)]
        .groupby("state_abb", as_index=False)["GW_canonical"]
        .sum()
        .rename(columns={"GW_canonical": "gw_2025"})
    )

    state_rows = []
    for state, group_mid in county[county["scenario"].eq("mid") & county["year"].eq(2035)].groupby("state_abb"):
        iso_sum = group_mid.groupby("iso")["GW_canonical"].sum().sort_values(ascending=False)
        total_mid = float(group_mid["GW_canonical"].sum())
        dominant = iso_sum.index[0] if len(iso_sum) else "Non-ISO"
        share = float(iso_sum.iloc[0] / total_mid) if total_mid > 0 and len(iso_sum) else 0.0
        if dominant not in ISO7 or share < 0.50:
            dominant = "Non-ISO"
        row = {"state_abb": state, "dominant_iso": dominant, "iso_share_2035": share}
        for scenario in SCENARIOS:
            row[f"gw_2035_{scenario}"] = float(
                county[
                    county["scenario"].eq(scenario)
                    & county["year"].eq(2035)
                    & county["state_abb"].eq(state)
                ]["GW_canonical"].sum()
            )
        state_rows.append(row)

    state_data = pd.DataFrame(state_rows).merge(mid2025, on="state_abb", how="left")
    state_data["gw_2025"] = pd.to_numeric(state_data["gw_2025"], errors="coerce").fillna(0.0)
    state_data = state_data.rename(columns={"gw_2035_mid": "gw_2035"})
    state_data["growth"] = (state_data["gw_2035"] - state_data["gw_2025"]).clip(lower=0.0)
    state_data = state_data.sort_values("gw_2035", ascending=False).head(10).reset_index(drop=True)

    conc_rows = []
    for year in YEARS:
        vals = (
            county[county["scenario"].eq("mid") & county["year"].eq(year)]
            .groupby("state_abb")["GW_canonical"]
            .sum()
            .sort_values(ascending=False)
            .to_numpy()
        )
        total = float(vals.sum())
        for top_n in range(1, 11):
            conc_rows.append({"year": year, "top_n": top_n, "share": vals[:top_n].sum() / total})
    state_conc = pd.DataFrame(conc_rows)

    # Keep old column spelling expected by the existing drawing functions.
    state_data = state_data[
        [
            "state_abb",
            "gw_2035_low",
            "gw_2035",
            "gw_2035_high",
            "gw_2025",
            "dominant_iso",
            "iso_share_2035",
            "growth",
        ]
    ]
    return state_data, state_conc


def scale_facility_points(base, facility: pd.DataFrame, iso_2025: pd.DataFrame) -> pd.DataFrame:
    out = facility.copy()
    target_mw = float(iso_2025["GW_mid"].sum() * 1000.0)
    current_mw = float(out["mw_alloc_2025"].sum())
    scale = target_mw / current_mw if current_mw > 0 else 1.0
    out["mw_alloc_2025"] = out["mw_alloc_2025"] * scale
    out["size"] = base.fig1map.size_from_mw(out["mw_alloc_2025"])
    return out


def write_tables(
    county: pd.DataFrame,
    audit: pd.DataFrame,
    county_rank: pd.DataFrame,
    state_data: pd.DataFrame,
    state_conc: pd.DataFrame,
    iso_data: pd.DataFrame,
    iso_conc: pd.DataFrame,
    map_tables: dict[str, pd.DataFrame],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    county.to_csv(TABLE_DIR / "fig1_canonical_county_all_scenarios.csv", index=False)
    audit.to_csv(TABLE_DIR / "fig1_canonical_audit.csv", index=False)
    county_rank.to_csv(TABLE_DIR / "fig1c_county_rank_canonical.csv", index=False)
    state_data.to_csv(TABLE_DIR / "fig1d_state_canonical.csv", index=False)
    state_conc.to_csv(TABLE_DIR / "fig1d_state_concentration_canonical.csv", index=False)
    iso_data.to_csv(TABLE_DIR / "fig1e_iso_canonical.csv", index=False)
    iso_conc.to_csv(TABLE_DIR / "fig1e_iso_concentration_canonical.csv", index=False)
    map_tables["iso_2025"].to_csv(TABLE_DIR / "fig1a_iso_2025_canonical.csv", index=False)
    map_tables["iso_2035"].to_csv(TABLE_DIR / "fig1b_iso_2035_canonical.csv", index=False)


def patch_county_insets(base) -> None:
    """Update Fig. 1c insets while leaving the main county rank-size panel unchanged."""

    fc = base.fig1county
    full_color = fc.TIME_COLOR_2035
    subset_color = fc.TIME_COLOR_2025
    subset_fill = fc.TIME_COLOR_2025_LIGHT

    def add_topk_inset(ax, d2035: pd.DataFrame) -> None:
        top_k = np.array([1, 3, 5, 10, 25, 50])
        full = d2035.sort_values("rank").copy()
        subset = d2035[d2035["iso"].isin(ISO7)].sort_values("gw", ascending=False).copy()
        full_total = float(full["gw"].sum())
        subset_total = float(subset["gw"].sum())
        full_shares = np.array([full.head(int(k))["gw"].sum() / full_total * 100.0 for k in top_k])
        subset_shares = np.array([subset.head(int(k))["gw"].sum() / subset_total * 100.0 for k in top_k])

        inset = ax.inset_axes([0.610, 0.712, 0.360, 0.258])
        inset.set_zorder(10)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.98)
        inset.set_xscale("log")
        inset.plot(
            top_k,
            full_shares,
            color=full_color,
            lw=1.15,
            marker="o",
            markersize=3.1,
            markerfacecolor=full_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
        inset.plot(
            top_k,
            subset_shares,
            color=subset_color,
            lw=1.05,
            ls=(0, (2.0, 1.45)),
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgecolor=subset_color,
            markeredgewidth=0.55,
        )
        inset.fill_between(top_k, full_shares, 0, color=full_color, alpha=0.09, linewidth=0)
        inset.fill_between(top_k, subset_shares, full_shares, color=subset_fill, alpha=0.24, linewidth=0)
        inset.set_title("Top-county share", fontsize=6.0, pad=1.0, color=fc.TEXT_COLOR)
        inset.set_xlim(1, 55)
        inset.set_ylim(0, 91)
        inset.set_xticks([1, 5, 10, 50])
        inset.set_xticklabels(["1", "5", "10", "50"])
        inset.set_yticks([25, 50, 75])
        inset.set_yticklabels(["25", "50", "75"])
        inset.grid(axis="y", color=fc.GRID_COLOR, lw=0.42)
        inset.tick_params(axis="both", labelsize=6.0, length=1.55, width=0.45, pad=1.0, colors=fc.TEXT_COLOR)
        full_top50 = full_shares[list(top_k).index(50)]
        subset_top50 = subset_shares[list(top_k).index(50)]
        inset.text(47, full_top50 - 5.0, f"full {full_top50:.0f}%", ha="right", va="top", fontsize=6.0, color=full_color)
        inset.text(47, min(subset_top50 - 3.5, 80.5), f"7-region {subset_top50:.0f}%", ha="right", va="top", fontsize=6.0, color=subset_color)
        for spine in ["top", "right"]:
            inset.spines[spine].set_visible(False)
        inset.spines["left"].set_linewidth(0.45)
        inset.spines["bottom"].set_linewidth(0.45)
        inset.spines["left"].set_color(fc.AXIS_COLOR)
        inset.spines["bottom"].set_color(fc.AXIS_COLOR)

    def add_threshold_count_inset(ax, d2025: pd.DataFrame, d2035: pd.DataFrame) -> None:
        thresholds = np.array([1.0, 0.5, 0.1])
        labels = ["≥1", "≥0.5", "≥0.1"]
        full = d2035.copy()
        subset = d2035[d2035["iso"].isin(ISO7)].copy()
        counts_full = np.array([(full["gw"] >= value).sum() for value in thresholds], dtype=float)
        counts_subset = np.array([(subset["gw"] >= value).sum() for value in thresholds], dtype=float)

        inset = ax.inset_axes([0.055, 0.070, 0.380, 0.245])
        inset.set_zorder(10)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.98)

        y = np.arange(len(thresholds))
        height = 0.30
        inset.barh(y + height / 2, counts_subset, height=height, color=subset_fill, edgecolor="none", label="7-region")
        inset.barh(y - height / 2, counts_full, height=height, color=full_color, edgecolor="none", alpha=0.95, label="full")

        for yi, value in zip(y - height / 2, counts_full):
            inset.text(value + 4, yi, f"{int(value)}", ha="left", va="center", fontsize=6.0, color=full_color)
        for yi, value in zip(y + height / 2, counts_subset):
            inset.text(value + 4, yi, f"{int(value)}", ha="left", va="center", fontsize=6.0, color=subset_color)

        inset.set_title("Counties above threshold", fontsize=6.0, pad=1.0, color=fc.TEXT_COLOR)
        inset.set_yticks(y)
        inset.set_yticklabels(labels)
        max_count = max(float(counts_full.max()), float(counts_subset.max()))
        inset.set_xlim(0, max(180, max_count * 1.22))
        inset.set_xticks([0, 50, 100, 150])
        inset.set_xlabel("count", fontsize=6.0, labelpad=0.8, color=fc.TEXT_COLOR)
        inset.tick_params(axis="both", labelsize=6.0, length=1.35, width=0.40, pad=0.9, colors=fc.TEXT_COLOR)
        inset.grid(axis="x", color=fc.GRID_COLOR, lw=0.38)
        inset.set_axisbelow(True)
        inset.legend(
            loc="lower right",
            frameon=False,
            fontsize=6.0,
            handlelength=0.8,
            handletextpad=0.25,
            borderpad=0.0,
            labelspacing=0.10,
        )
        for spine in ["top", "right"]:
            inset.spines[spine].set_visible(False)
        inset.spines["left"].set_linewidth(0.36)
        inset.spines["bottom"].set_linewidth(0.36)
        inset.spines["left"].set_color(fc.AXIS_COLOR)
        inset.spines["bottom"].set_color(fc.AXIS_COLOR)

    base.fig1county.add_topk_inset = add_topk_inset
    base.fig1county.add_threshold_count_inset = add_threshold_count_inset


def save_legacy_size_county_panel(base, county_rank: pd.DataFrame) -> tuple[Path, Path]:
    """Save the standalone Fig. 1c panel in the earlier square-ish panel geometry."""

    fc = base.fig1county
    fc.set_nature_rcparams()
    data = county_rank.copy()
    fig, ax = plt.subplots(figsize=(fc.PANEL_WIDTH_MM / fc.MM_PER_INCH, fc.PANEL_HEIGHT_MM / fc.MM_PER_INCH))
    d2025 = data[data["year"].eq(2025)].sort_values("rank").copy()
    d2035 = data[data["year"].eq(2035)].sort_values("rank").copy()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", color=fc.GRID_COLOR, lw=0.55)
    ax.grid(which="minor", axis="y", color="#F6EFE8", lw=0.35)

    ax.plot(d2025["rank"], d2025["gw"], color=fc.TIME_COLOR_2025, lw=1.22, ls=(0, (2.2, 1.8)), alpha=0.82, zorder=2)
    ax.plot(d2035["rank"], d2035["gw"], color=fc.TIME_COLOR_2035, lw=1.48, alpha=0.96, zorder=3)

    colors = d2035["iso"].map(fc.ISO_COLOR).fillna(fc.ISO_COLOR["Non-ISO"])
    point_sizes = np.select([d2035["rank"] <= 50, d2035["rank"] <= 150], [16.0, 10.5], default=8.0)
    ax.scatter(
        d2035["rank"],
        d2035["gw"],
        s=point_sizes,
        c=colors,
        edgecolor="#FFF9F2",
        linewidth=0.26,
        alpha=np.where(d2035["rank"] <= 50, 0.94, 0.66),
        zorder=4,
    )
    ax.scatter(
        d2025["rank"],
        d2025["gw"],
        s=8.0,
        facecolor="white",
        edgecolor=fc.TIME_COLOR_2025,
        linewidth=0.45,
        alpha=0.70,
        zorder=3,
    )

    ax.axhline(1.0, color=fc.TIME_COLOR_2025_LIGHT, lw=0.70, ls=(0, (1.2, 1.7)), zorder=1)
    ax.text(12, 1.08, "1 GW threshold", ha="left", va="bottom", fontsize=6.0, color=fc.TIME_COLOR_2025)

    label_names = ["Loudoun, VA", "Maricopa, AZ", "Prince William, VA", "Polk, IA", "Carson, TX"]
    text_positions = {
        "Loudoun, VA": (1.18, 16.4),
        "Maricopa, AZ": (2.15, 8.55),
        "Prince William, VA": (3.60, 6.15),
        "Carson, TX": (5.35, 4.55),
        "Polk, IA": (6.25, 3.35),
    }
    for _, row in d2035[d2035["label"].isin(label_names)].iterrows():
        ax.annotate(
            str(row["label"]),
            xy=(row["rank"], row["gw"]),
            xytext=text_positions.get(str(row["label"]), (row["rank"] * 1.12, row["gw"])),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=6.0,
            color=fc.TEXT_COLOR,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.35,
                "color": fc.AXIS_COLOR,
                "alpha": 0.52,
                "shrinkA": 0,
                "shrinkB": 2,
            },
        )

    ax.set_xlim(1, max(360, float(d2035["rank"].max()) * 1.03))
    top_line_label_y = {10: 0.940, 50: 0.665, 100: 0.665}
    for x_value, label in [(10, "Top 10"), (50, "Top 50"), (100, "Top 100")]:
        ax.axvline(x_value, color="#E9DCD0", lw=0.50, ls=(0, (1.1, 2.0)), zorder=1)
        ax.text(
            x_value,
            top_line_label_y[x_value],
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.0,
            color=fc.TIME_COLOR_2025,
        )

    fc.add_topk_inset(ax, d2035)
    fc.add_threshold_count_inset(ax, d2025, d2035)

    ax.set_ylim(0.015, 20)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("County rank", labelpad=2.4)
    ax.set_ylabel("County demand (GW)", labelpad=2.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(fc.AXIS_COLOR)
    ax.spines["bottom"].set_color(fc.AXIS_COLOR)
    ax.tick_params(axis="both", length=2.4, width=0.62, pad=1.9, colors=fc.TEXT_COLOR)

    line_handles = [
        Line2D([0], [0], color=fc.TIME_COLOR_2025, lw=1.22, ls=(0, (2.2, 1.8)), label="2025"),
        Line2D([0], [0], color=fc.TIME_COLOR_2035, lw=1.48, label="2035"),
    ]
    fig.legend(
        handles=line_handles,
        loc="lower left",
        bbox_to_anchor=(0.145, 0.034),
        frameon=False,
        ncol=2,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.0,
    )

    fig.text(
        0.160,
        0.976,
        f"{len(d2035)} positive counties in 2035; {int((d2035['gw'] >= 1.0).sum())} exceed 1 GW",
        ha="left",
        va="top",
        fontsize=6.15,
        color=fc.TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.160, right=0.975, top=0.920, bottom=0.155)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig1c_optimized_county_rank_two_denominators_1200dpi.pdf"
    png_path = OUT_DIR / "fig1c_optimized_county_rank_two_denominators_1200dpi.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=1200)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    base = load_module("fig1_base", BASE_FIG1_SCRIPT)
    base.OUT_DIR = OUT_DIR
    base.set_rc()
    base.patch_map_styling()
    patch_county_insets(base)

    base.fig1map.PIPELINE_POINT_COLOR = base.PIPELINE_RED
    geo = base.fig1map.load_geography(PROJECT_ROOT)
    county, audit = canonical_county_data(base, geo)
    county_rank = rank_counties(county)
    map_tables, iso_data, iso_conc = build_iso_tables(county)
    state_data, state_conc = build_state_tables(base, county)
    write_tables(county, audit, county_rank, state_data, state_conc, iso_data, iso_conc, map_tables)

    facility = base.fig1map.build_2025_facility_points(PROJECT_ROOT, geo)
    facility = facility.assign(**{"geometry.x": facility.geometry.x, "geometry.y": facility.geometry.y})
    facility = scale_facility_points(base, facility, map_tables["iso_2025"])
    pipeline = base.fig1map.build_2035_pipeline_points(PROJECT_ROOT, geo)

    max_gw = max(
        30.0,
        float(map_tables["iso_2025"]["GW_mid"].max()),
        float(map_tables["iso_2035"][["GW_low", "GW_mid", "GW_high"]].max().max()),
    )
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=max_gw * 1000.0)

    base.save_individual_panels(
        geo,
        map_tables,
        facility,
        pipeline,
        norm,
        county_rank,
        state_data,
        state_conc,
        iso_data,
        iso_conc,
    )
    pdf, png, svg = base.save_composite(
        geo,
        map_tables,
        facility,
        pipeline,
        norm,
        county_rank,
        state_data,
        state_conc,
        iso_data,
        iso_conc,
    )

    print("Canonical Fig. 1 written:")
    print(pdf)
    print(png)
    print(svg)
    print("Audit table:")
    print(TABLE_DIR / "fig1_canonical_audit.csv")
    print(audit[["scenario", "year", "old_source_peak_GW_total", "raw_ISO7_share", "new_national_average_GW", "new_ISO7_coincident_GW"]].to_string(index=False))


if __name__ == "__main__":
    main()
