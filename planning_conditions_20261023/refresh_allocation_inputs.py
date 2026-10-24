"""Refresh allocation-dependent extension inputs from the adjacent baseline.

Run `--stage candidates` after the baseline, then run_pool_consistency.py;
run `--stage conditions` before the remaining extension entries.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reproduction_20261009"
KEY = ["planning_region", "year", "data_center_scenario_id", "post2030_extension_id",
       "footprint_capture_case", "peak_mapping_case", "allocation_case"]


def refresh_manifest(name):
    p = HERE / name
    obj = json.loads(p.read_text())
    obj["files"] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in obj["files"]}
    p.write_text(json.dumps(obj, indent=2) + "\n")


def candidates():
    for source, target, columns, rename in [
        ("contribution/01_annual_paired_outcomes.csv.gz", "candidate_paths.csv.gz",
         KEY + ["candidate_gross_path_GW"], {"candidate_gross_path_GW": "candidate_GW"}),
        ("source_quality/03_lower_endpoint_annual_pairs.csv.gz", "lower_candidates.csv.gz",
         KEY + ["quality_case", "candidate_GW"], {}),
        ("source_quality/10_scope_annual_pairs.csv.gz", "scope_candidates.csv.gz",
         KEY + ["quality_case", "candidate_GW"], {})]:
        data = pd.read_csv(BASE / "numerical/results" / source)[columns].drop_duplicates().rename(columns=rename)
        data.to_csv(HERE / "pool_inputs" / target, index=False,
                    compression={"method": "gzip", "mtime": 0})
    geo = pd.read_csv(BASE / "default_weight_update/geographic_assignment_results.csv")
    geo[["region", "resources", "response", "assignment", "candidate_peak_increment_GW"]].to_csv(
        HERE / "pool_inputs/geographic_candidates.csv", index=False)
    refresh_manifest("POOL_INPUT_MANIFEST.json")
    refresh_manifest("PAIRED_REALIZATION_INPUT_MANIFEST.json")


def conditions():
    import run_pool_consistency as pool
    paths = pool.display(pd.read_csv(HERE / "pool_results/timing_annual.csv.gz"))
    paths = paths[paths.planning_region.eq("MISO") & paths.resource_condition.eq("anticipated")
                  & paths.response_condition.eq("published_all")].copy()
    paths["background_balance_GW"] = paths.background_GW
    paths["candidate_balance_GW"] = paths.balance_GW
    paths.to_csv(HERE / "inputs/miso_display_timing_paths.csv", index=False)
    records_path = BASE / "default_weight_update/geographic_records_used.csv"
    records = pd.read_csv(records_path)
    weights = records.groupby("ISO_ASSIGNED").reviewed_load_weight_MW.sum()
    entries = []
    for matrix in [0, 1]:
        for frontier in [0, 1]:
            reduced = weights.copy()
            for sid, keep in [(1120, matrix), (935, frontier)]:
                row = records[records.source_row.eq(sid)].iloc[0]
                reduced[row.ISO_ASSIGNED] -= (1-keep) * row.reviewed_load_weight_MW
            for region in ["MISO", "ERCOT"]:
                entries.append(dict(keep_matrix=matrix, keep_frontier=frontier, planning_region=region,
                    reference_region_weight_MW=weights[region], retained_region_weight_MW=reduced[region],
                    reference_seven_region_weight_MW=weights.sum(), retained_seven_region_weight_MW=reduced.sum()))
    pd.DataFrame(entries).to_csv(HERE / "pool_inputs/campus_retention_weights.csv", index=False)
    p = HERE / "CAMPUS_RETENTION_INPUT_MANIFEST.json"
    obj = json.loads(p.read_text())
    obj.update(records=len(records), weight_source_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest())
    p.write_text(json.dumps(obj, indent=2) + "\n")
    for name in ["INPUT_MANIFEST.json", "CAMPUS_RETENTION_INPUT_MANIFEST.json",
                 "PAIRED_REALIZATION_INPUT_MANIFEST.json"]:
        refresh_manifest(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["candidates", "conditions"], required=True)
    globals()[parser.parse_args().stage]()
