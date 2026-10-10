"""Reparse project MW and reconstruct fixed-total weights on frozen geography."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from run_review_package import compare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "spatial"


def main():
    spec = importlib.util.spec_from_file_location("allocation_rules", DATA / "allocation_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    raw = rules.harmonize_inventory(DATA / "project_records.csv", "latest_20260917")
    mapped = rules.map_to_regions(raw)
    assert len(mapped) == len(raw) and not mapped.source_row.duplicated().any()
    weights, subsets = rules.allocation_cases(mapped)
    out = ROOT / "results/spatial"
    out.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out / "project_records_reparsed.csv", index=False)
    weights.to_csv(out / "fixed_total_weights.csv", index=False)
    compare(ROOT / "sources/lbnl_2025_update/spatial_fixed_total_allocation_sensitivity.csv", out / "fixed_total_weights.csv",
            ["allocation_case", "model_region"], ["records_used", "allocation_weight", "allocation_share", "allocated_per_100_GW"])
    assert np.allclose(weights.groupby("allocation_case").allocation_share.sum(), 1)
    collisions = raw[raw.record_key.duplicated(keep=False)].sort_values("record_key")
    collisions.to_csv(out / "possible_duplicate_keys_for_review.csv", index=False)
    core = subsets["central_reported_mid"]
    full_share = core.groupby("ISO_ASSIGNED").allocation_weight.sum() / core.allocation_weight.sum()
    leave = []
    for row in core.itertuples():
        retained = core[core.source_row.ne(row.source_row)]
        shares = retained.groupby("ISO_ASSIGNED").allocation_weight.sum() / retained.allocation_weight.sum()
        for region in rules.REGIONS:
            leave.append({"removed_source_row": row.source_row, "removed_name": row.facility_name,
                          "removed_reported_MW": row.reported_mw_mid, "model_region": region,
                          "original_share": full_share.get(region, 0), "remaining_share": shares.get(region, 0),
                          "change_pp": 100 * (shares.get(region, 0) - full_share.get(region, 0))})
    pd.DataFrame(leave).to_csv(out / "leave_one_project_out.csv", index=False)
    record = {"passed": True, "source_rows": len(raw), "mapped_rows": len(mapped), "allocation_rows": len(weights),
              "eight_weight_rules_reproduced": True, "raw_MW_parsing_recomputed": True,
              "possible_duplicate_records": len(collisions), "source_rows_silently_removed": False,
              "county_fallback_and_area_crosswalk_frozen": True, "actual_grid_connection_certified": False}
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
