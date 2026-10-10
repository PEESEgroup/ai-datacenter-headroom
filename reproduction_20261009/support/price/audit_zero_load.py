"""Flag unexplained zeros; exclusion is a diagnostic, not source correction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import build_price_fixed_effects as price

HERE = Path(__file__).resolve().parent


def main():
    counts, common, zones = [], [], []
    for iso in ["PJM", "SPP"]:
        path, zone_col = price.PANEL_SPECS[iso]
        panel, _ = price.prepare_panel(iso, path, zone_col)
        zero = panel[panel.load_mw.eq(0)]
        counts.append({
            "ISO": iso, "zero_zone_hours": len(zero),
            "zero_timestamps": zero.timestamp_utc.nunique(),
            "zero_years": ";".join(map(str, sorted(zero.year.unique()))),
            "negative_zone_hours": int(panel.load_mw.lt(0).sum()),
            "source_sha256": price.sha256(path),
            "zero_as_missing_certified": False,
        })
        restandardized = panel[panel.load_mw.ne(0)].copy()
        grouped = restandardized.groupby(["zone", "year"]).load_mw
        restandardized["load_z"] = ((restandardized.load_mw - grouped.transform("mean"))
                                    / grouped.transform("std").replace(0, np.nan))
        restandardized = restandardized.dropna(subset=["load_z"])
        for name, data in [
            ("retained_zero_baseline", panel),
            ("exclude_zero_fixed_original_scale", panel[panel.load_mw.ne(0)].copy()),
            ("exclude_zero_restandardized", restandardized),
        ]:
            common.append(price.fit_common_slope(data, name))
            result, _ = price.fit_zone_slopes(data)
            result["sample"] = name
            zones.append(result)
        zero[["ISO", "zone", "timestamp_utc", "load_mw", "lmp_raw"]].to_csv(
            price.OUT / f"zero_load_records_{iso}.csv", index=False)
    table = pd.DataFrame(common)
    table.to_csv(price.OUT / "04_zero_load_common_sensitivity.csv", index=False)
    pd.concat(zones, ignore_index=True).to_csv(
        price.OUT / "04_zero_load_zone_sensitivity.csv", index=False)
    pd.DataFrame(counts).to_csv(price.OUT / "04_zero_load_quality.csv", index=False)
    baseline = table[table["sample"].eq("retained_zero_baseline")].set_index("ISO")
    diagnostic = table[table["sample"].eq("exclude_zero_restandardized")].set_index("ISO")
    column = "coefficient_dollars_per_mwh_per_within_zone_year_load_sd"
    change = 100 * (diagnostic[column] / baseline[column] - 1)
    receipt = {
        "passed": bool(np.isfinite(table[column]).all()),
        "zero_as_missing_certified": False,
        "main_price_outputs_unchanged": True,
        "diagnostic_scaling": "both fixed-original-scale and restandardized exclusion are archived; reported percent change uses restandardized load",
        "common_slope_change_percent": change.to_dict(),
        "quality_counts": counts,
    }
    (price.OUT / "zero_load_diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
