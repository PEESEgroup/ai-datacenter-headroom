"""EIA reader helpers preserving literal identifiers and blank numeric cells."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

BA_TO_REGION = {
    "MISO": "MISO",
    "PJM": "PJM",
    "SWPP": "SPP",
    "ERCO": "ERCOT",
    "NYIS": "NYISO",
    "ISNE": "ISO-NE",
    "CISO": "CAISO",
}

def normalize_key(plant_id: object, generator_id: object) -> str:
    if pd.isna(plant_id) or pd.isna(generator_id):
        return ""
    try:
        plant = str(int(float(plant_id)))
    except (TypeError, ValueError):
        plant = str(plant_id).strip().upper()
    generator = re.sub(r"\s+", "", str(generator_id).strip().upper())
    return f"{plant}::{generator}" if plant and generator else ""

def find_header(path: Path, sheet: str) -> int:
    preview = pd.read_excel(path, sheet_name=sheet, header=None, nrows=8)
    for index, row in preview.iterrows():
        values = {str(value).strip() for value in row if pd.notna(value)}
        if {"Plant ID", "Generator ID"}.issubset(values):
            return int(index)
    raise ValueError(f"Could not locate header in {path.name}:{sheet}")

def read_860m(path: Path, sheet: str) -> pd.DataFrame:
    try:
        header = find_header(path, sheet)
    except ValueError:
        return pd.DataFrame(columns=["unit_key"])
    # "NA" is a real Generator ID; only blank cells are missing at import.
    frame = pd.read_excel(path, sheet_name=sheet, header=header,
                          keep_default_na=False, na_values=[""])
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Plant ID" not in frame or "Generator ID" not in frame:
        return pd.DataFrame(columns=["unit_key"])
    frame["unit_key"] = [
        normalize_key(plant, generator)
        for plant, generator in zip(frame["Plant ID"], frame["Generator ID"])
    ]
    return frame[frame["unit_key"].ne("")].copy()

def status_code(value: object) -> str:
    match = re.match(r"\(([^)]+)\)", str(value).strip())
    return match.group(1).upper() if match else "UNKNOWN"

def stage_group(code: str) -> str:
    if code in {"P", "L", "T"}:
        return "preconstruction"
    if code == "U":
        return "early_construction"
    if code in {"V", "TS"}:
        return "advanced_or_complete"
    return "unknown_stage"
