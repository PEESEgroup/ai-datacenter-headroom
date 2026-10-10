"""Refresh reader-facing derived exports, preserving superseded copies."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "alignment/baseline/reader_exports_pre_final_audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    record = []
    for destination in sorted((ROOT / "source_data").glob("*.csv")):
        source = ROOT / "numerical/results/source_quality" / destination.name
        if not source.exists():
            continue
        old_hash = sha(destination)
        if old_hash != sha(source):
            archived = ARCHIVE / destination.name
            if not archived.exists():
                shutil.copy2(destination, archived)
            shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
        record.append({"export": str(destination.relative_to(ROOT)),
                       "authoritative_output": str(source.relative_to(ROOT)), "sha256": sha(source)})
    source = ROOT / "numerical/results/124_fig1_spatial_allocation_plotting.csv"
    weights = pd.read_csv(source).pivot(index="allocation_case", columns="planning_region", values="allocation_share")
    weights = weights[["ERCOT", "ISO-NE", "MISO", "NYISO", "PJM", "SPP", "WECC-California"]]
    destination = ROOT / "source_data/fixed_total_allocation_weights.csv"
    existing = pd.read_csv(destination)
    if "CAISO" in existing.columns and not (ARCHIVE / destination.name).exists():
        shutil.copy2(destination, ARCHIVE / destination.name)
    weights.to_csv(destination)
    record.append({"export": str(destination.relative_to(ROOT)), "authoritative_output": str(source.relative_to(ROOT)),
                   "transform": "pivot to region columns; numerical ID WECC-California", "sha256": sha(destination)})
    result = {"passed": True, "exports": record, "model_or_scenario_rules_changed": False,
              "superseded_exports_preserved": str(ARCHIVE.relative_to(ROOT))}
    (ROOT / "alignment/reader_export_sync_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "reader_exports_synchronized": len(record), "model_rules_changed": False}))


if __name__ == "__main__":
    main()
