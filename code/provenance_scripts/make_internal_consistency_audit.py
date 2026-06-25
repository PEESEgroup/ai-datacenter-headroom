from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "out_std" / "ANALYSIS"
FIG2 = ROOT / "Final Figs" / "Fig2_subfigs"
FIG3 = ROOT / "Final Figs" / "Fig3_subfigs"

OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ISO_ORDER = ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]


def read_csv(path):
    return pd.read_csv(path)


def crossing_year(group):
    """Linearly interpolate the first mid-growth zero crossing."""
    g = group.sort_values("year")
    prev_year = None
    prev_margin = None
    for _, row in g.iterrows():
        year = float(row["year"])
        margin = float(row["mid_growth_margin_GW"])
        if margin < 0:
            if prev_margin is None or prev_margin < 0:
                return year
            if np.isclose(prev_margin, margin):
                return year
            frac = prev_margin / (prev_margin - margin)
            return prev_year + frac * (year - prev_year)
        prev_year = year
        prev_margin = margin
    return np.nan


fig1_2025 = (
    read_csv(ANALYSIS / "map_iso_dc_2025_single_iso_gw.csv")
    .rename(columns={"ISO_ASSIGNED": "ISO", "GW_2025": "fig1_2025_demand_gw"})
    [["ISO", "fig1_2025_demand_gw"]]
)

fig1_2035 = (
    read_csv(ANALYSIS / "map_iso_dc_2035_mid_with_low_mid_high_bars_iso_gw.csv")
    .rename(
        columns={
            "ISO_ASSIGNED": "ISO",
            "GW_low": "fig1_2035_low_demand_gw",
            "GW_mid": "fig1_2035_mid_demand_gw",
            "GW_high": "fig1_2035_high_demand_gw",
        }
    )
    [
        [
            "ISO",
            "fig1_2035_low_demand_gw",
            "fig1_2035_mid_demand_gw",
            "fig1_2035_high_demand_gw",
        ]
    ]
)

bridge = (
    read_csv(FIG2 / "optimized_20260511" / "fig2b_optimized_bridge_source.csv")
    .rename(
        columns={
            "headroom_gw": "headroom_2025_gw",
            "supply_gw": "net_supply_expansion_gw",
            "nonai_gw": "nonai_load_growth_signed_gw",
            "ai_gw": "ai_growth_signed_gw",
            "margin_2035_gw": "margin_2035_mid_gw",
        }
    )
)
bridge["nonai_load_growth_gw"] = -bridge["nonai_load_growth_signed_gw"]
bridge["ai_growth_bridge_gw"] = -bridge["ai_growth_signed_gw"]
bridge["margin_recalculated_gw"] = (
    bridge["headroom_2025_gw"]
    + bridge["net_supply_expansion_gw"]
    - bridge["nonai_load_growth_gw"]
    - bridge["ai_growth_bridge_gw"]
)
bridge["margin_calc_residual_gw"] = (
    bridge["margin_recalculated_gw"] - bridge["margin_2035_mid_gw"]
)
bridge = bridge[
    [
        "ISO",
        "headroom_2025_gw",
        "net_supply_expansion_gw",
        "nonai_load_growth_gw",
        "ai_growth_bridge_gw",
        "margin_2035_mid_gw",
        "margin_calc_residual_gw",
    ]
]

traj = read_csv(FIG2 / "final_tight_layout_20260511" / "fig2_final_e_trajectory_source.csv")
cross = (
    traj.groupby("ISO")
    .apply(crossing_year, include_groups=False)
    .reset_index(name="first_deficit_interpolated_year")
)
clock = read_csv(FIG2 / "optimized_20260511" / "fig2c_optimized_margin_clock_source.csv")
clock_margin = (
    clock.pivot(index="ISO", columns="scenario", values="margin_2035_GW")
    .rename(
        columns={
            "Low": "margin_2035_low_growth_gw",
            "Mid": "margin_2035_mid_clock_gw",
            "High": "margin_2035_high_growth_gw",
        }
    )
    .reset_index()
)
clock_first = (
    clock.pivot(index="ISO", columns="scenario", values="first_deficit")
    .rename(
        columns={
            "Low": "first_deficit_low",
            "Mid": "first_deficit_mid",
            "High": "first_deficit_high",
        }
    )
    .reset_index()
)

fig3 = (
    read_csv(FIG3 / "optimized_mechanism_20260511" / "fig3_clustered_archetype_source.csv")
    .rename(
        columns={
            "AI_growth_to_2035_MW": "fig3_ai_growth_to_2035_mw",
            "dc_growth_mw": "fig3_dc_growth_metric_mw",
            "gen_growth_mw": "fig3_gen_growth_metric_mw",
            "headroom_ratio": "fig3_headroom_ratio",
            "cluster": "fig3_cluster",
        }
    )
    [
        [
            "ISO",
            "fig3_ai_growth_to_2035_mw",
            "fig3_dc_growth_metric_mw",
            "fig3_gen_growth_metric_mw",
            "fig3_headroom_ratio",
            "fig3_cluster",
        ]
    ]
)
for col in ["fig3_ai_growth_to_2035_mw", "fig3_dc_growth_metric_mw", "fig3_gen_growth_metric_mw"]:
    fig3[col.replace("_mw", "_gw")] = fig3[col] / 1000.0
fig3 = fig3.drop(
    columns=["fig3_ai_growth_to_2035_mw", "fig3_dc_growth_metric_mw", "fig3_gen_growth_metric_mw"]
)

audit = (
    fig1_2025.merge(fig1_2035, on="ISO", how="outer")
    .merge(bridge, on="ISO", how="outer")
    .merge(cross, on="ISO", how="outer")
    .merge(clock_margin, on="ISO", how="outer")
    .merge(clock_first, on="ISO", how="outer")
    .merge(fig3, on="ISO", how="outer")
)

audit["fig1_mid_growth_2025_2035_gw"] = (
    audit["fig1_2035_mid_demand_gw"] - audit["fig1_2025_demand_gw"]
)
audit["bridge_implied_2035_demand_gw"] = (
    audit["fig1_2025_demand_gw"] + audit["ai_growth_bridge_gw"]
)
audit["bridge_minus_fig1_2035_demand_gw"] = (
    audit["bridge_implied_2035_demand_gw"] - audit["fig1_2035_mid_demand_gw"]
)
audit["fig2_minus_fig3_ai_growth_gw"] = (
    audit["ai_growth_bridge_gw"] - audit["fig3_ai_growth_to_2035_gw"]
)
audit["legacy_dc_metric_minus_bridge_ai_growth_gw"] = (
    audit["fig3_dc_growth_metric_gw"] - audit["ai_growth_bridge_gw"]
)

def status(row):
    flags = []
    if abs(row["margin_calc_residual_gw"]) > 0.05:
        flags.append("margin equation check")
    if abs(row["bridge_minus_fig1_2035_demand_gw"]) > 0.5:
        flags.append("Fig1 demand vs bridge AI-growth basis")
    if abs(row["legacy_dc_metric_minus_bridge_ai_growth_gw"]) > 0.5:
        flags.append("legacy Fig3 dc_growth label")
    return "; ".join(flags) if flags else "OK"


audit["audit_flag"] = audit.apply(status, axis=1)

audit["sort"] = audit["ISO"].map({iso: i for i, iso in enumerate(ISO_ORDER)})
audit = audit.sort_values("sort").drop(columns=["sort"])

ordered_cols = [
    "ISO",
    "fig1_2025_demand_gw",
    "fig1_2035_mid_demand_gw",
    "fig1_mid_growth_2025_2035_gw",
    "ai_growth_bridge_gw",
    "bridge_implied_2035_demand_gw",
    "bridge_minus_fig1_2035_demand_gw",
    "headroom_2025_gw",
    "net_supply_expansion_gw",
    "nonai_load_growth_gw",
    "margin_2035_mid_gw",
    "margin_2035_mid_clock_gw",
    "margin_calc_residual_gw",
    "margin_2035_low_growth_gw",
    "margin_2035_high_growth_gw",
    "first_deficit_interpolated_year",
    "first_deficit_mid",
    "first_deficit_low",
    "first_deficit_high",
    "fig3_ai_growth_to_2035_gw",
    "fig2_minus_fig3_ai_growth_gw",
    "fig3_dc_growth_metric_gw",
    "legacy_dc_metric_minus_bridge_ai_growth_gw",
    "fig3_headroom_ratio",
    "fig3_cluster",
    "audit_flag",
]
audit = audit[ordered_cols]

csv_path = OUT / "internal_consistency_audit_iso_table.csv"
audit.to_csv(csv_path, index=False, float_format="%.4f")

rounded = audit.copy()
num_cols = rounded.select_dtypes(include=[np.number]).columns
rounded[num_cols] = rounded[num_cols].round(2)
md_path = OUT / "internal_consistency_audit_iso_table.md"
with md_path.open("w") as f:
    f.write("# Internal consistency audit table\n\n")
    f.write(
        "Canonical demand values use the Fig. 1 county-to-ISO allocation. "
        "Bridge variables use the Fig. 2 supply-headroom accounting convention: "
        "2035 margin = 2025 headroom + net supply expansion - non-AI load growth - AI load growth. "
        "The audit intentionally reports both Fig. 1 displayed 2035 demand and the bridge-implied 2035 demand, "
        "because those are currently the main cross-figure consistency items.\n\n"
    )
    f.write(rounded.to_markdown(index=False))
    f.write("\n")

tex_cols = [
    "ISO",
    "fig1_2025_demand_gw",
    "fig1_2035_mid_demand_gw",
    "ai_growth_bridge_gw",
    "bridge_minus_fig1_2035_demand_gw",
    "headroom_2025_gw",
    "net_supply_expansion_gw",
    "nonai_load_growth_gw",
    "margin_2035_mid_gw",
    "first_deficit_mid",
    "audit_flag",
]
tex = audit[tex_cols].copy()
tex = tex.rename(
    columns={
        "fig1_2025_demand_gw": "2025 AI demand",
        "fig1_2035_mid_demand_gw": "2035 AI demand",
        "ai_growth_bridge_gw": "AI growth in margin bridge",
        "bridge_minus_fig1_2035_demand_gw": "Bridge--Fig.1 demand difference",
        "headroom_2025_gw": "2025 headroom",
        "net_supply_expansion_gw": "Net supply expansion",
        "nonai_load_growth_gw": "Non-AI load growth",
        "margin_2035_mid_gw": "2035 margin",
        "first_deficit_mid": "First deficit year",
        "audit_flag": "Audit note",
    }
)
tex_path = OUT / "internal_consistency_audit_iso_table.tex"
with tex_path.open("w") as f:
    f.write(
        tex.to_latex(
            index=False,
            escape=True,
            float_format="%.2f",
            caption=(
                "Internal consistency audit for ISO-level demand, supply-headroom "
                "and margin-accounting variables. Values are GW unless otherwise noted."
            ),
            label="tab:internal-consistency-audit",
        )
    )

summary = {
    "sum_fig1_2025_demand_gw": audit["fig1_2025_demand_gw"].sum(),
    "sum_fig1_2035_mid_demand_gw": audit["fig1_2035_mid_demand_gw"].sum(),
    "sum_fig1_mid_growth_gw": audit["fig1_mid_growth_2025_2035_gw"].sum(),
    "sum_bridge_ai_growth_gw": audit["ai_growth_bridge_gw"].sum(),
    "sum_bridge_implied_2035_demand_gw": audit["bridge_implied_2035_demand_gw"].sum(),
    "max_margin_calc_abs_residual_gw": audit["margin_calc_residual_gw"].abs().max(),
    "max_fig2_minus_fig3_ai_growth_abs_gw": audit["fig2_minus_fig3_ai_growth_gw"].abs().max(),
    "max_legacy_dc_metric_minus_bridge_ai_growth_abs_gw": audit[
        "legacy_dc_metric_minus_bridge_ai_growth_gw"
    ].abs().max(),
}
summary_path = OUT / "internal_consistency_audit_summary.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False, float_format="%.4f")

print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
print(f"Wrote {tex_path}")
print(f"Wrote {summary_path}")
print(pd.DataFrame([summary]).round(4).to_string(index=False))
