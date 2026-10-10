"""Frozen allocation functions; CSV adapter replaces only Excel ingestion."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
REGIONS = ["PJM", "MISO", "SPP", "ERCOT", "WECC-California", "NYISO", "ISO-NE"]
CORE_ACTIVE = {"Proposed", "Approved/Permitted/Under construction"}
BROAD_ACTIVE = CORE_ACTIVE | {"Pre-proposal", "Expanding"}
HIGH_EVIDENCE = {"Approved/Permitted/Under construction"}
COUNTY_REF = ROOT / "census_county_reference_2023.csv"
COUNTY_ISO = ROOT / "county_to_iso_area_snapshot.csv"
LATEST_RECORD_GEOID = ROOT / "fractracker_live_record_geoid_mapping_20260917.csv"
COUNTY_SHP_EXTERNAL = ROOT / "not_included_county_polygon.shp"

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_county(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(
        r"\b(city and borough|census area|municipality|borough|parish|county|city)\b",
        " ",
        text,
    )
    text = re.sub(r"\bsaint\b", "st", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_mw(value: object) -> tuple[float, float, float, str]:
    """Parse an explicitly reported point, range, or one-sided MW value."""
    if pd.isna(value):
        return np.nan, np.nan, np.nan, "missing"
    text = str(value).strip().replace(",", "").replace("–", "-").replace("—", "-")
    if not text or text.lower() in {"nan", "unknown", "tbd", "none"}:
        return np.nan, np.nan, np.nan, "missing"
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    numbers = [item for item in numbers if item > 0]
    if not numbers:
        return np.nan, np.nan, np.nan, "unparsed"
    if len(numbers) == 1:
        value_num = numbers[0]
        parse_class = "reported_lower_bound" if re.search(r">|at least|more than", text.lower()) else "reported_point"
        return value_num, value_num, value_num, parse_class
    low, high = min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
    return low, (low + high) / 2.0, high, "reported_range_midpoint"

def parse_sqft(value: object) -> tuple[float, str]:
    """Parse a reported facility-area point or range in square feet."""
    if pd.isna(value):
        return np.nan, "missing"
    text = (
        str(value)
        .strip()
        .lower()
        .replace(",", "")
        .replace("–", "-")
        .replace("—", "-")
    )
    if not text or text in {"nan", "unknown", "tbd", "none"}:
        return np.nan, "missing"
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    numbers = [item for item in numbers if item > 0]
    if not numbers:
        return np.nan, "unparsed"
    multiplier = 1.0
    if "million" in text or re.search(r"m\s*(?:sq\.?\s*ft\.?)?$", text):
        multiplier = 1_000_000.0
    elif "thousand" in text or re.search(r"k\s*(?:sq\.?\s*ft\.?)?$", text):
        multiplier = 1_000.0
    if len(numbers) >= 2 and ("-" in text or " to " in text):
        return multiplier * float(np.mean(numbers[:2])), "reported_range_midpoint"
    parse_class = (
        "reported_lower_bound"
        if re.search(r">|at least|more than", text)
        else "reported_point"
    )
    return multiplier * numbers[0], parse_class

def size_rank_proxy(value: object) -> float:
    """Return a transparent size-class proxy used only in sensitivity tests."""
    text = normalize_text(value)
    if "small" in text:
        return 5.0
    if "medium" in text:
        return 30.5
    if "large" in text and "mega" not in text:
        return 75.0
    if "hyperscale" in text:
        return 550.0
    if "mega" in text:
        return 1000.0
    return np.nan

def parse_online_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    years = [int(item) for item in re.findall(r"\b20\d{2}\b", str(value))]
    years = [year for year in years if 2024 <= year <= 2040]
    return float(min(years)) if years else np.nan

def harmonize_inventory(path: Path, vintage: str) -> pd.DataFrame:
    if vintage == "latest_20260917":
        raw = pd.read_csv(path)
        rename = {
            "facility_name": "facility_name",
            "operator_name": "operator",
        }
    else:
        raw = pd.read_csv(path)
        rename = {"name": "facility_name", "operator": "operator"}
    raw = raw.rename(columns=rename).copy()
    required = [
        "facility_name",
        "state",
        "county",
        "lat",
        "long",
        "status",
        "purpose",
        "mw",
        "facility_size_sqft",
        "sizerank",
        "expected_date_online",
        "date_created",
        "date_updated",
        "address",
        "city",
        "operator",
        "tenant",
        "other_info",
        "information_source",
        "info_source_1",
        "info_source_2",
        "info_source_3",
        "info_source_4",
        "info_source_5",
        "info_source_6",
        "info_source_7",
        "info_source_8",
    ]
    for column in required:
        if column not in raw:
            raw[column] = pd.NA

    rows = raw[required].copy()
    rows.insert(0, "vintage", vintage)
    rows.insert(1, "source_row", np.arange(1, len(rows) + 1))
    rows["state"] = rows["state"].astype("string").str.upper().str.strip()
    rows["county_reported"] = rows["county"].astype("string").str.strip()

    # One current record omits county but states it in the public address field.
    override = (
        rows["facility_name"].astype(str).eq("Nixxy-Tachyon 9 Data Center")
        & rows["state"].eq("ND")
        & rows["county"].isna()
    )
    rows.loc[override, "county"] = "Williams"
    rows.loc[override, "county_reported"] = "Williams (from source address)"

    parsed = rows["mw"].apply(parse_mw).apply(pd.Series)
    parsed.columns = ["reported_mw_low", "reported_mw_mid", "reported_mw_high", "mw_parse_class"]
    rows = pd.concat([rows, parsed], axis=1)
    area_parsed = rows["facility_size_sqft"].apply(parse_sqft).apply(pd.Series)
    area_parsed.columns = ["reported_sqft", "sqft_parse_class"]
    rows = pd.concat([rows, area_parsed], axis=1)
    rows["size_rank_proxy_mw"] = rows["sizerank"].apply(size_rank_proxy)
    rows["online_year_parsed"] = rows["expected_date_online"].apply(parse_online_year)
    rows["status"] = rows["status"].astype("string").str.strip()
    rows["purpose"] = rows["purpose"].astype("string").str.strip()
    rows["purpose_norm"] = rows["purpose"].apply(normalize_text)
    rows["county_norm"] = rows["county"].apply(normalize_county)
    rows["facility_norm"] = rows["facility_name"].apply(normalize_text)
    rows["explicit_non_data_center"] = rows["facility_norm"].str.contains(
        r"\bnot (?:a )?data center\b", regex=True
    )
    rows["ai_tagged_purpose"] = rows["purpose"].fillna("").str.contains(
        r"(?i)(?:\bAI\b|artificial intelligence)", regex=True
    )
    rows["lat"] = pd.to_numeric(rows["lat"], errors="coerce")
    rows["long"] = pd.to_numeric(rows["long"], errors="coerce")
    rows["record_key"] = (
        rows["facility_norm"]
        + "|"
        + rows["state"].fillna("").astype(str)
        + "|"
        + rows["lat"].round(4).astype(str)
        + "|"
        + rows["long"].round(4).astype(str)
    )
    return rows

def map_to_regions(rows: pd.DataFrame, apply_california_alignment: bool = True) -> pd.DataFrame:
    counties = pd.read_csv(COUNTY_REF, dtype={"GEOID": str})
    counties["GEOID"] = counties["GEOID"].str.zfill(5)
    counties["state"] = counties["STUSPS"].astype(str).str.upper()
    counties["county_norm"] = counties["NAME"].apply(normalize_county)
    lookup = counties[["GEOID", "state", "county_norm", "NAME"]].drop_duplicates(
        ["state", "county_norm"]
    )
    mapped = rows.merge(
        lookup.rename(columns={"state": "state_lookup", "NAME": "county_census"}),
        left_on=["state", "county_norm"],
        right_on=["state_lookup", "county_norm"],
        how="left",
    )
    mapped["county_mapping_method"] = np.where(
        mapped["GEOID"].notna(), "reported_state_county_name", "unmatched"
    )

    if LATEST_RECORD_GEOID.exists() and rows["vintage"].iloc[0] == "latest_20260917":
        stored = pd.read_csv(LATEST_RECORD_GEOID, dtype={"GEOID_stored": str})
        stored["GEOID_stored"] = stored["GEOID_stored"].str.zfill(5)
        mapped = mapped.merge(
            stored[["record_key", "GEOID_stored"]], on="record_key", how="left"
        )
        stored_fill = mapped["GEOID"].isna() & mapped["GEOID_stored"].notna()
        mapped.loc[stored_fill, "GEOID"] = mapped.loc[stored_fill, "GEOID_stored"]
        mapped.loc[stored_fill, "county_mapping_method"] = "frozen_record_geoid_fallback"
        mapped = mapped.drop(columns=["GEOID_stored"])

    # Coordinate fallback is deterministic and only used when the name join fails.
    missing = mapped["GEOID"].isna() & mapped["lat"].notna() & mapped["long"].notna()
    if missing.any() and COUNTY_SHP_EXTERNAL.exists():
        import geopandas as gpd

        points = gpd.GeoDataFrame(
            mapped.loc[missing, ["source_row", "lat", "long"]].copy(),
            geometry=gpd.points_from_xy(
                mapped.loc[missing, "long"], mapped.loc[missing, "lat"]
            ),
            crs="EPSG:4326",
        )
        polygons = gpd.read_file(COUNTY_SHP_EXTERNAL)[["GEOID", "geometry"]].to_crs(
            points.crs
        )
        spatial = gpd.sjoin(points, polygons, how="left", predicate="within")
        spatial_map = spatial.set_index("source_row")["GEOID"].to_dict()
        fill_values = mapped.loc[missing, "source_row"].map(spatial_map)
        mapped.loc[missing, "GEOID"] = fill_values.values
        used = missing & mapped["GEOID"].notna()
        mapped.loc[used, "county_mapping_method"] = "coordinate_fallback"

    mapped["GEOID"] = mapped["GEOID"].astype("string").str.zfill(5)
    crosswalk = pd.read_csv(COUNTY_ISO, dtype={"GEOID": str})
    crosswalk["GEOID"] = crosswalk["GEOID"].str.zfill(5)
    mapped = mapped.merge(crosswalk, on="GEOID", how="left")
    outside_conus = mapped["ISO_ASSIGNED"].isna() & mapped["state"].isin(["AK", "HI"])
    mapped.loc[outside_conus, "ISO_ASSIGNED"] = "Outside-CONUS"
    mapped["ISO_ASSIGNED"] = mapped["ISO_ASSIGNED"].fillna("Unmatched")
    aligned_path = ROOT.parent.parent / "alignment/project_footprint_crosswalk.csv"
    if apply_california_alignment and aligned_path.exists():
        aligned = pd.read_csv(aligned_path)
        if aligned.record_key.isna().any() or aligned.record_key.duplicated().any():
            raise ValueError("Invalid California footprint crosswalk key")
        mapped["previous_spatial_region"] = mapped["ISO_ASSIGNED"]
        lookup = aligned.set_index("record_key").aligned_spatial_region
        replacement = mapped.record_key.map(lookup)
        affected = mapped.state.isin(["CA", "NV"]) | mapped.ISO_ASSIGNED.eq("CAISO")
        if replacement[affected].isna().any():
            raise ValueError("California/Nevada record missing from aligned crosswalk")
        mapped.loc[affected, "ISO_ASSIGNED"] = replacement[affected]
    mapped["in_seven_region_footprint"] = mapped["ISO_ASSIGNED"].isin(REGIONS)
    mapped["mega_project_scale_flag"] = np.where(
        mapped["reported_mw_mid"].ge(5000.0),
        "reported_scale_at_least_5_GW; locational_weight_not_realization_forecast",
        "",
    )
    return mapped

def allocation_cases(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    eligible = ~rows["explicit_non_data_center"]
    core = eligible & rows["status"].isin(CORE_ACTIVE)
    broad = eligible & rows["status"].isin(BROAD_ACTIVE)
    high = eligible & rows["status"].isin(HIGH_EVIDENCE)
    reported = rows["reported_mw_mid"].notna()
    mapped = rows["in_seven_region_footprint"]
    proxy_filled = rows["reported_mw_mid"].fillna(rows["size_rank_proxy_mw"])
    definitions = {
        "central_reported_mid": (core & reported & mapped, rows["reported_mw_mid"]),
        "central_reported_low": (core & reported & mapped, rows["reported_mw_low"]),
        "central_reported_high": (core & reported & mapped, rows["reported_mw_high"]),
        "high_evidence_reported_mid": (high & reported & mapped, rows["reported_mw_mid"]),
        "broad_reported_mid": (broad & reported & mapped, rows["reported_mw_mid"]),
        "central_site_count": (core & mapped, pd.Series(1.0, index=rows.index)),
        "broad_site_count": (broad & mapped, pd.Series(1.0, index=rows.index)),
        "central_reported_plus_size_proxy_sensitivity": (
            core & proxy_filled.notna() & mapped,
            proxy_filled,
        ),
    }
    output = []
    subsets: dict[str, pd.DataFrame] = {}
    for case, (selector, weights) in definitions.items():
        subset = rows.loc[selector].copy()
        subset["allocation_weight"] = weights.loc[selector].astype(float)
        subset = subset[subset["allocation_weight"] > 0].copy()
        subsets[case] = subset
        total = subset["allocation_weight"].sum()
        grouped = subset.groupby("ISO_ASSIGNED")["allocation_weight"].sum()
        for region in REGIONS:
            weight = float(grouped.get(region, 0.0))
            output.append(
                {
                    "allocation_case": case,
                    "model_region": region,
                    "records_used": int((subset["ISO_ASSIGNED"] == region).sum()),
                    "allocation_weight": weight,
                    "allocation_share": weight / total if total > 0 else np.nan,
                    "allocated_per_100_GW": 100.0 * weight / total if total > 0 else np.nan,
                    "fixed_total_rule": "renormalized_to_same_scenario_year_total",
                    "weight_interpretation": "locational evidence, not realized load",
                }
            )
    return pd.DataFrame(output), subsets
