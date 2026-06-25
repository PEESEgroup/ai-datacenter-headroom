from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fnum(row, key):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return float("nan")
    return float(value)

def check_manifest():
    manifest = read_csv(ROOT / "source_data" / "file_manifest.csv")
    checked = 0
    failed = []
    for row in manifest:
        rel = row["relative_path"]
        if rel == "source_data/file_manifest.csv" or rel.startswith("outputs/"):
            continue
        path = ROOT / rel
        checked += 1
        if not path.exists():
            failed.append((rel, "missing"))
            continue
        if sha256(path) != row["sha256"]:
            failed.append((rel, "sha256 mismatch"))
    return checked, failed

def headline_metrics():
    rows = read_csv(ROOT / "source_data" / "derived_tables" / "tables" / "fig1_canonical_20260514" / "fig1c_county_rank_canonical.csv")
    mid2035 = [r for r in rows if r["year"] == "2035" and r["scenario"] == "mid"]
    total = sum(fnum(r, "gw") for r in mid2035)
    top50 = sum(fnum(r, "gw") for r in mid2035 if int(float(r["rank"])) <= 50)
    top50_share = 100 * top50 / total

    iso_rows = read_csv(ROOT / "source_data" / "derived_tables" / "tables" / "fig2_5_canonical_20260514" / "shared" / "canonical_iso_ai_demand_yearly.csv")
    seven2035 = [r for r in iso_rows if r["year"] == "2035" and r["scenario"] == "mid"]
    deficit_possible = {"MISO", "SPP", "PJM"}
    seven_total = sum(fnum(r, "ai_demand_GW") for r in seven2035)
    exposed = sum(fnum(r, "ai_demand_GW") for r in seven2035 if r["ISO"] in deficit_possible)
    exposed_share = 100 * exposed / seven_total

    bridge = read_csv(ROOT / "source_data" / "derived_tables" / "tables" / "fig2_5_canonical_20260514" / "fig2" / "fig2b_iso_margin_bridge_2035_canonical_source.csv")
    margins = {r["ISO"]: fnum(r, "Margin_2035_MW") / 1000 for r in bridge}

    return [
        {"metric": "Fig.1 central-2035 top-50 county share, full county allocation (%)", "value": f"{top50_share:.1f}", "expected_text_value": "74.3", "source": "fig1c_county_rank_canonical.csv"},
        {"metric": "Central-2035 AI load share in potential deficit-facing grid regions (%)", "value": f"{exposed_share:.1f}", "expected_text_value": "77.8", "source": "canonical_iso_ai_demand_yearly.csv"},
        {"metric": "PJM central-2035 modeled margin (GW)", "value": f"{margins.get('PJM', float('nan')):.1f}", "expected_text_value": "-6.0", "source": "fig2b_iso_margin_bridge_2035_canonical_source.csv"},
        {"metric": "MISO central-2035 modeled margin (GW)", "value": f"{margins.get('MISO', float('nan')):.1f}", "expected_text_value": "-6.0", "source": "fig2b_iso_margin_bridge_2035_canonical_source.csv"},
        {"metric": "SPP central-2035 modeled margin (GW)", "value": f"{margins.get('SPP', float('nan')):.1f}", "expected_text_value": "-2.3 to -4.8 depending current figure/caption convention", "source": "fig2b_iso_margin_bridge_2035_canonical_source.csv"},
    ]

def main():
    checked, failed = check_manifest()
    metrics = headline_metrics()
    with (OUT / "headline_metric_check.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "expected_text_value", "source"])
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT / "manifest_check.txt").open("w", encoding="utf-8") as f:
        f.write(f"Checked {checked} files against source_data/file_manifest.csv\n")
        if failed:
            f.write("Failures:\n")
            for rel, reason in failed:
                f.write(f"- {rel}: {reason}\n")
        else:
            f.write("All checked file hashes matched.\n")
    print(f"Checked {checked} files; failures: {len(failed)}")
    print(f"Wrote {OUT / 'headline_metric_check.csv'}")
    print(f"Wrote {OUT / 'manifest_check.txt'}")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
