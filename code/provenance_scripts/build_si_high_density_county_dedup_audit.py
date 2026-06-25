from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "source_data" / "derived_tables"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IM3_DIR = PROJECT_ROOT / "out_std" / "DC" / "im3_processed"
OUT_DIR = TABLE_ARCHIVE_ROOT / "tables" / "si_dedup_audit_20260522"

RAW_RECORDS = IM3_DIR / "im3_records_dedup_flags_from_rawxlsx_r500m_20260407_150708.csv"
FINAL_RECORDS = IM3_DIR / "im3_records_with_dedup_flags_20260407_150720.csv"
COUNTY_COMPARE = IM3_DIR / "im3_county_area_weights_compare_raw_vs_final_r500m_20260407_150720.csv"
NN_BY_COUNTY = IM3_DIR / "im3_cluster_need_nn_by_county_20260407_150720.csv"
FIG1_COUNTY = TABLE_ARCHIVE_ROOT / "tables" / "fig1_canonical_20260514" / "fig1c_county_rank_canonical.csv"

STATE_FIPS = {
    "AL": "01",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "FL": "12",
    "GA": "13",
    "IA": "19",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "MA": "25",
    "MD": "24",
    "ME": "23",
    "MI": "26",
    "MN": "27",
    "MO": "29",
    "MS": "28",
    "MT": "30",
    "NC": "37",
    "ND": "38",
    "NE": "31",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NV": "32",
    "NY": "36",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VA": "51",
    "VT": "50",
    "WA": "53",
    "WI": "55",
    "WV": "54",
    "WY": "56",
}


def geoid(state_abb: object, county_id: object) -> str:
    state = str(state_abb).strip()
    return f"{STATE_FIPS.get(state, '00')}{int(float(county_id)):03d}"


FIPS_STATE = {v: k for k, v in STATE_FIPS.items()}


def state_county_from_geoid(value: object) -> tuple[str, int]:
    text = str(value).zfill(5)
    return FIPS_STATE[text[:2]], int(text[2:])


def latex_escape(value: object) -> str:
    if pd.isna(value):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "--"
    return f"{100 * x:.{digits}f}\\%"


def fmt_pp(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "--"
    sign = "+" if x > 0 else ""
    return f"{sign}{100 * x:.{digits}f} pp"


def fmt_num(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def token_set(*values: object) -> set[str]:
    tokens: set[str] = set()
    stop = {
        "data",
        "center",
        "centre",
        "datacenter",
        "campus",
        "building",
        "llc",
        "inc",
        "the",
        "project",
        "technology",
        "park",
    }
    for value in values:
        if pd.isna(value):
            continue
        for token in str(value).lower().replace("/", " ").replace("-", " ").split():
            clean = "".join(ch for ch in token if ch.isalnum())
            if len(clean) >= 3 and clean not in stop:
                tokens.add(clean)
    return tokens


def review_flag(campus: pd.Series, building: pd.Series, distance_m: float) -> tuple[str, str]:
    campus_tokens = token_set(campus.get("operator"), campus.get("name"), campus.get("ref"))
    building_tokens = token_set(building.get("operator"), building.get("name"), building.get("ref"))
    overlap = campus_tokens & building_tokens
    same = bool(overlap)
    campus_named = bool(campus_tokens)
    if same and distance_m <= 500:
        return "low", f"near retained building with shared token(s): {', '.join(sorted(overlap)[:3])}"
    if distance_m <= 150 and not campus_named:
        return "moderate", "near retained building, but campus operator/name missing"
    if distance_m <= 500:
        return "moderate", "within dedup radius; operator/name evidence incomplete or not shared"
    return "review", "outside intended radius"


def nearest_building_pairs(raw: pd.DataFrame, selected_keys: set[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for state_abb, county_id in sorted(selected_keys):
        county = raw[(raw["state_abb"].eq(state_abb)) & (raw["county_id"].astype(int).eq(county_id))].copy()
        buildings = county[county["type"].eq("building") & county["keep_in_final"].astype(bool)].copy()
        dropped = county[county["type"].eq("campus") & ~county["keep_in_final"].astype(bool)].copy()
        if buildings.empty or dropped.empty:
            continue
        for _, campus in dropped.iterrows():
            best_dist = np.inf
            best = None
            for _, building in buildings.iterrows():
                dist = haversine_m(float(campus["lat"]), float(campus["lon"]), float(building["lat"]), float(building["lon"]))
                if dist < best_dist:
                    best_dist = dist
                    best = building
            if best is None:
                continue
            flag, reason = review_flag(campus, best, best_dist)
            rows.append(
                {
                    "GEOID": geoid(state_abb, county_id),
                    "state_abb": state_abb,
                    "county_id": int(county_id),
                    "county": campus["county"],
                    "campus_id": campus["id"],
                    "campus_operator": campus.get("operator"),
                    "campus_name": campus.get("name"),
                    "campus_ref": campus.get("ref"),
                    "campus_sqft": float(campus.get("sqft", 0.0)),
                    "nearest_building_id": best["id"],
                    "nearest_building_operator": best.get("operator"),
                    "nearest_building_name": best.get("name"),
                    "nearest_building_ref": best.get("ref"),
                    "nearest_building_sqft": float(best.get("sqft", 0.0)),
                    "nearest_distance_m": best_dist,
                    "review_flag": flag,
                    "review_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def concentration_for_counterfactual(
    weights: pd.DataFrame,
    canonical: pd.DataFrame,
    national_2025_average_gw: float,
    label: str,
    note: str,
) -> list[dict[str, object]]:
    canonical_mid = canonical[canonical["scenario"].eq("mid")].copy()
    final_2025 = canonical_mid[canonical_mid["year"].eq(2025)][["GEOID", "GW_canonical", "gamma"]].rename(
        columns={"GW_canonical": "final_2025_gw"}
    )
    final_2035 = canonical_mid[canonical_mid["year"].eq(2035)][["GEOID", "GW_canonical"]].rename(
        columns={"GW_canonical": "final_2035_gw"}
    )
    joined = weights.merge(final_2025, on="GEOID", how="outer").merge(final_2035, on="GEOID", how="outer")
    joined[["weight", "final_2025_gw", "final_2035_gw", "gamma"]] = joined[
        ["weight", "final_2025_gw", "final_2035_gw", "gamma"]
    ].fillna(0.0)
    joined["alt_2025_gw"] = joined["weight"] * joined["gamma"] * national_2025_average_gw
    joined["final_increment_2025_2035_gw"] = (joined["final_2035_gw"] - joined["final_2025_gw"]).clip(lower=0.0)
    joined["alt_2035_gw"] = joined["alt_2025_gw"] + joined["final_increment_2025_2035_gw"]
    scale = joined["final_2035_gw"].sum() / joined["alt_2035_gw"].sum()
    vals = (joined["alt_2035_gw"] * scale).fillna(0.0).sort_values(ascending=False)
    total = float(vals.sum())
    return [
        {
            "case": label,
            "top_n": n,
            "share": float(vals.head(n).sum() / total) if total > 0 else np.nan,
            "total_gw": total,
            "note": note,
        }
        for n in (10, 20, 50, 100)
    ]


def build_concentration(raw_final: pd.DataFrame, canonical: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    county = raw_final.copy()
    county["GEOID"] = [geoid(s, c) for s, c in zip(county["state_abb"], county["county_id"])]
    canonical_mid = canonical[canonical["scenario"].eq("mid")].copy()
    final_2025 = canonical_mid[canonical_mid["year"].eq(2025)][["GEOID", "GW_canonical", "gamma"]].rename(
        columns={"GW_canonical": "final_2025_gw"}
    )
    final_2035 = canonical_mid[canonical_mid["year"].eq(2035)][["GEOID", "GW_canonical"]].rename(
        columns={"GW_canonical": "final_2035_gw"}
    )
    joined = county.merge(final_2025, on="GEOID", how="outer").merge(final_2035, on="GEOID", how="outer")
    joined[["w_us_final_r500m", "final_2025_gw", "final_2035_gw", "gamma"]] = joined[
        ["w_us_final_r500m", "final_2025_gw", "final_2035_gw", "gamma"]
    ].fillna(0.0)
    national_2025_average_gw = joined["final_2025_gw"].sum() / (
        (joined["w_us_final_r500m"] * joined["gamma"]).sum()
    )

    rows = []
    canonical_vals = canonical_mid[canonical_mid["year"].eq(2035)]["GW_canonical"].fillna(0.0).sort_values(
        ascending=False
    )
    canonical_total = float(canonical_vals.sum())
    for n in (10, 20, 50, 100):
        rows.append(
            {
                "case": "Canonical Fig. 1 allocation",
                "top_n": n,
                "share": float(canonical_vals.head(n).sum() / canonical_total),
                "total_gw": canonical_total,
                "note": "Full calibrated county matrix used in Fig. 1.",
            }
        )

    for radius in (300, 500, 800):
        keep = records["type"].eq("building") | (
            records["type"].eq("campus")
            & (records["dist_to_nearest_building_m"].isna() | (records["dist_to_nearest_building_m"] > radius))
        )
        radius_records = records.loc[keep].copy()
        radius_records["GEOID"] = [
            geoid(s, c) for s, c in zip(radius_records["state_abb"], radius_records["county_id"])
        ]
        weights = radius_records.groupby("GEOID", as_index=False).agg(area=("sqft", "sum"))
        weights["weight"] = weights["area"] / weights["area"].sum()
        rows.extend(
            concentration_for_counterfactual(
                weights[["GEOID", "weight"]],
                canonical,
                national_2025_average_gw,
                f"{radius} m radius sensitivity",
                "Holds canonical 2025--2035 incremental county additions fixed and changes only the existing-baseline dedup radius.",
            )
        )

    review_weights = county[["GEOID", "w_review_inclusive"]].rename(columns={"w_review_inclusive": "weight"})
    rows.extend(
        concentration_for_counterfactual(
            review_weights,
            canonical,
            national_2025_average_gw,
            "Review-inclusive add-back",
            "Reintroduces dropped campus records lacking shared operator/name/reference evidence with the nearest retained building.",
        )
    )
    return pd.DataFrame(rows)


def audit_interpretation(row: pd.Series) -> str:
    campus = int(row["n_campus"])
    dropped = int(row["dropped_campus_count"])
    if campus == 0:
        return "Building-only baseline; no campus-overlap removal."
    if dropped == 0:
        return "Campus record retained; no nearby building-overlap removal."
    if dropped == campus:
        return "Campus records resolved by the building-preferred overlap screen."
    return "Mixed county: only overlapping campus records are removed."


def write_record_resolution_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Selected top-10 existing-county record-type audit for the Fig. 1 baseline allocation. Counties are the ten largest counties in the canonical 2025 existing-facility allocation. The table reports record-type resolution rather than gross-area changes because source floor-area fields mix campus polygons and building footprints. Future-growth hotspots are excluded because they enter through the announced-project pipeline allocation rather than the existing-facility campus--building deduplication screen.}",
        r"\label{tab:top10-existing-record-resolution-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}L{0.15\linewidth}L{0.055\linewidth}L{0.075\linewidth}L{0.075\linewidth}L{0.07\linewidth}L{0.075\linewidth}L{0.12\linewidth}L{0.22\linewidth}@{}}",
        r"\toprule",
        r"County & 2025 rank & Source records & Building records & Campus records & Campus excluded & Pair audit & Interpretation \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values("rank_2025").iterrows():
        dropped = int(row["dropped_campus_count"])
        if dropped > 0:
            pair_audit = f"{int(row['low_review_pairs'])} low-risk; {int(row['moderate_review_pairs'])} review"
        else:
            pair_audit = "No dropped campus pairs"
        lines.append(
            " & ".join(
                [
                    latex_escape(row["county_label"]),
                    str(int(row["rank_2025"])),
                    str(int(row["n_rows"])),
                    str(int(row["n_building"])),
                    str(int(row["n_campus"])),
                    str(dropped),
                    latex_escape(pair_audit),
                    latex_escape(audit_interpretation(row)),
                ]
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_concentration_table(conc: pd.DataFrame, path: Path) -> None:
    pivot = conc.pivot(index="case", columns="top_n", values="share").reset_index()
    order = [
        "Canonical Fig. 1 allocation",
        "300 m radius sensitivity",
        "500 m radius sensitivity",
        "800 m radius sensitivity",
        "Review-inclusive add-back",
    ]
    pivot["case"] = pd.Categorical(pivot["case"], categories=order, ordered=True)
    pivot = pivot.dropna(subset=["case"]).sort_values("case")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Robustness of the 2035 top-county concentration result to existing-baseline deduplication choices. Radius sensitivities change only the campus--building deduplication radius for the existing-facility baseline while holding the calibrated 2025--2035 incremental county additions fixed. The review-inclusive add-back reintroduces dropped campus records whose nearest retained building lacks shared operator/name/reference evidence.}",
        r"\label{tab:dedup-concentration-audit}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Case & Top 10 & Top 20 & Top 50 & Top 100 \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            latex_escape(row["case"])
            + " & "
            + " & ".join(fmt_pct(float(row[n]), 1) for n in (10, 20, 50, 100))
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_RECORDS)
    final_records = pd.read_csv(FINAL_RECORDS)
    audit_records = final_records.merge(raw[["id", "ref"]], on="id", how="left")
    compare = pd.read_csv(COUNTY_COMPARE)
    nn = pd.read_csv(NN_BY_COUNTY)
    canonical = pd.read_csv(FIG1_COUNTY, dtype={"GEOID": str})
    canonical["GEOID"] = canonical["GEOID"].str.zfill(5)

    compare["GEOID"] = [geoid(s, c) for s, c in zip(compare["state_abb"], compare["county_id"])]
    compare["county_label"] = compare["county"].str.replace(" County", "", regex=False) + ", " + compare["state_abb"]

    canonical_top10 = (
        canonical[(canonical["scenario"].eq("mid")) & (canonical["year"].eq(2025))]
        .sort_values("GW_canonical", ascending=False)
        .head(10)[["GEOID", "rank", "GW_canonical"]]
        .copy()
    )
    canonical_top10["GEOID"] = canonical_top10["GEOID"].str.zfill(5)
    top_keys = {state_county_from_geoid(x) for x in canonical_top10["GEOID"]}

    all_keys = set(
        tuple(x)
        for x in audit_records[["state_abb", "county_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    all_pairs = nearest_building_pairs(audit_records, {(s, int(c)) for s, c in all_keys})
    review_readd = (
        all_pairs[all_pairs["review_flag"].ne("low")]
        .groupby("GEOID", as_index=False)
        .agg(review_readd_sqft=("campus_sqft", "sum"))
    )
    compare = compare.merge(review_readd, on="GEOID", how="left")
    compare["review_readd_sqft"] = compare["review_readd_sqft"].fillna(0.0)
    compare["area_review_inclusive"] = compare["area_sqft_final_r500m"] + compare["review_readd_sqft"]
    compare["w_review_inclusive"] = compare["area_review_inclusive"] / compare["area_review_inclusive"].sum()

    pairs = all_pairs[all_pairs["GEOID"].isin([geoid(s, c) for s, c in top_keys])].copy()
    if pairs.empty:
        pair_counts = pd.DataFrame(columns=["GEOID", "review_flag", "n"])
    else:
        pair_counts = pairs.groupby(["GEOID", "review_flag"]).size().reset_index(name="n")

    dropped = (
        audit_records[audit_records["type"].eq("campus") & ~audit_records["keep_in_final"].astype(bool)]
        .assign(GEOID=lambda x: [geoid(s, c) for s, c in zip(x["state_abb"], x["county_id"])])
        .groupby("GEOID", as_index=False)
        .agg(dropped_campus_count=("id", "count"), dropped_campus_sqft=("sqft", "sum"))
    )
    if not pairs.empty:
        pair_summary = (
            pairs.groupby("GEOID", as_index=False)
            .agg(
                max_dropped_campus_nearest_building_m=("nearest_distance_m", "max"),
                median_dropped_campus_nearest_building_m=("nearest_distance_m", "median"),
            )
            .merge(
                pair_counts[pair_counts["review_flag"].eq("low")][["GEOID", "n"]].rename(
                    columns={"n": "low_review_pairs"}
                ),
                on="GEOID",
                how="left",
            )
            .merge(
                pair_counts[pair_counts["review_flag"].ne("low")]
                .groupby("GEOID", as_index=False)["n"]
                .sum()
                .rename(columns={"n": "moderate_review_pairs"}),
                on="GEOID",
                how="left",
            )
        )
    else:
        pair_summary = pd.DataFrame(columns=["GEOID"])

    summary = (
        compare[compare["GEOID"].isin([geoid(s, c) for s, c in top_keys])]
        .merge(dropped, on="GEOID", how="left")
        .merge(nn[["state_abb", "county_id", "q50", "q95", "share_nn_lt_500m"]], on=["state_abb", "county_id"], how="left")
        .merge(pair_summary, on="GEOID", how="left")
        .merge(
            canonical_top10.rename(columns={"rank": "rank_2025", "GW_canonical": "gw_2025"})[
                ["GEOID", "rank_2025", "gw_2025"]
            ],
            on="GEOID",
            how="left",
        )
        .fillna(
            {
                "dropped_campus_count": 0,
                "dropped_campus_sqft": 0.0,
                "review_readd_sqft": 0.0,
                "low_review_pairs": 0,
                "moderate_review_pairs": 0,
                "max_dropped_campus_nearest_building_m": 0,
                "median_dropped_campus_nearest_building_m": 0,
            }
        )
        .sort_values("rank_2025")
        .reset_index(drop=True)
    )

    conc = build_concentration(compare, canonical, audit_records)

    summary.to_csv(OUT_DIR / "high_density_county_dedup_summary.csv", index=False)
    pairs.sort_values(["GEOID", "campus_sqft"], ascending=[True, False]).to_csv(
        OUT_DIR / "high_density_county_dedup_nearest_building_pairs.csv", index=False
    )
    all_pairs.sort_values(["GEOID", "campus_sqft"], ascending=[True, False]).to_csv(
        OUT_DIR / "all_dropped_campus_nearest_building_pairs.csv", index=False
    )
    conc.to_csv(OUT_DIR / "high_density_county_dedup_concentration.csv", index=False)

    write_record_resolution_table(summary, OUT_DIR / "high_density_county_record_resolution_table.tex")
    write_concentration_table(conc, OUT_DIR / "high_density_county_dedup_concentration_table.tex")

    manifest = pd.DataFrame(
        [
            {
                "output": "high_density_county_dedup_summary.csv",
                "rows": len(summary),
                "source_inputs": "; ".join(str(p) for p in [RAW_RECORDS, FINAL_RECORDS, COUNTY_COMPARE, NN_BY_COUNTY]),
                "description": "Machine-readable record-type diagnostics for the top-10 counties in the canonical 2025 existing-facility baseline.",
            },
            {
                "output": "high_density_county_record_resolution_table.tex",
                "rows": len(summary),
                "source_inputs": "; ".join(str(p) for p in [RAW_RECORDS, FINAL_RECORDS, FIG1_COUNTY]),
                "description": "Printed SI table showing source-record, building-record, campus-record and campus-exclusion counts for the top-10 existing-facility counties.",
            },
            {
                "output": "high_density_county_dedup_nearest_building_pairs.csv",
                "rows": len(pairs),
                "source_inputs": "; ".join(str(p) for p in [RAW_RECORDS, FINAL_RECORDS]),
                "description": "Dropped campus records in high-density counties matched to nearest retained building records with operator/name/ref fields.",
            },
            {
                "output": "all_dropped_campus_nearest_building_pairs.csv",
                "rows": len(all_pairs),
                "source_inputs": "; ".join(str(p) for p in [RAW_RECORDS, FINAL_RECORDS]),
                "description": "All dropped campus records matched to nearest retained building records; used for review-inclusive sensitivity.",
            },
            {
                "output": "high_density_county_dedup_concentration.csv",
                "rows": len(conc),
                "source_inputs": "; ".join(str(p) for p in [COUNTY_COMPARE, FIG1_COUNTY]),
                "description": "Top-N concentration under canonical, dedup-radius and review-inclusive 2035 counterfactual weights.",
            },
        ]
    )
    manifest.to_csv(OUT_DIR / "high_density_county_dedup_manifest.csv", index=False)

    print(f"Wrote {OUT_DIR}")
    print(
        summary[
            [
                "county_label",
                "rank_2025",
                "n_rows",
                "n_building",
                "n_campus",
                "dropped_campus_count",
                "low_review_pairs",
                "moderate_review_pairs",
            ]
        ].to_string(index=False)
    )
    print(conc[conc["top_n"].eq(50)].to_string(index=False))


if __name__ == "__main__":
    main()
