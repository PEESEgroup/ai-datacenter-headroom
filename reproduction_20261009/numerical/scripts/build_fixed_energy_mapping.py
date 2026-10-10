"""Normalize transferred CEC shape scenarios without changing annual energy."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources/lbnl_2025_update"
SHAPES = {"low": "IEPR2025_Mar_low", "avg": "IEPR2025_avg",
          "high": "IEPR2025_Oct_high"}
DAY_WEIGHTS = {"weekday": 5 / 7, "weekend": 2 / 7}
TIME_ZONES = {"WECC-California": "America/Los_Angeles", "ERCOT": "America/Chicago",
              "MISO": "America/Chicago", "SPP": "America/Chicago",
              "PJM": "America/New_York", "NYISO": "America/New_York",
              "ISO-NE": "America/New_York"}


def build():
    profile = pd.read_csv(SOURCE / "cec_iepr_2025_dc_hourly_load_factor_profiles_combined.csv")
    windows = pd.read_csv(SOURCE / "iso_coincidence_peak_window_3h.csv")
    assert len(profile) == 48 and not profile.duplicated(["day_type", "hour"]).any()
    assert set(profile.hour) == set(range(1, 25))
    assert set(profile.day_type) == set(DAY_WEIGHTS)
    assert all(profile[c].between(0, 100).all() for c in SHAPES.values())
    rows, normalized, checks = [], [], []
    for shape, column in SHAPES.items():
        hourly = sum(profile.loc[profile.day_type.eq(day)].set_index("hour")[column]
                     * weight / 100 for day, weight in DAY_WEIGHTS.items())
        mean = hourly.mean()
        unit = hourly / mean
        checks.append({"check": f"{shape}_fixed_energy_mean", "passed": bool(np.isclose(unit.mean(), 1)),
                       "residual": float(abs(unit.mean() - 1))})
        normalized.extend({"source_shape": shape, "hour_ending": h,
                           "day_weighted_load_factor": v,
                           "unit_mean_shape": unit.loc[h]} for h, v in hourly.items())
        for region in sorted(TIME_ZONES):
            selected = windows.loc[windows.ISO.eq(region), "iso_peak_hour"].unique()
            assert len(selected) == 1
            peak = int(selected[0])
            # Preserve the frozen JJA diurnal diagnostic window, not a new peak forecast.
            hours = list(range(max(1, peak - 1), min(24, peak + 1) + 1))
            coincidence = hourly.loc[hours].mean()
            rows.append({"model_region": region, "source_shape": shape,
                         "shape_mean_load_factor": mean,
                         "peak_window_coincidence": coincidence, "gamma": coincidence / mean,
                         "local_time_zone": TIME_ZONES[region], "hour_convention": "hour ending 1-24",
                         "peak_window_hours": "/".join(map(str, hours)),
                         "window_basis": ("Transferred CAISO diagnostic window used as an explicit California load-shape scenario; not observed WECC noncoincident peak"
                                          if region == "WECC-California" else "2025 June-July-August mean regional diurnal demand; three hours around maximum"),
                         "weekday_weight": 5 / 7, "weekend_weight": 2 / 7,
                         "shape_role": "transferred complete-shape scenario; not an uncertainty quantile",
                         "provenance": "CEC November 2025 presentation pp. 17-18; existing one-decimal digitization"})
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE / "peak_mapping_factors_fixed_energy_20261004.csv", index=False)
    pd.DataFrame(normalized).to_csv(SOURCE / "fixed_energy_normalized_hourly_shapes.csv", index=False)
    pd.DataFrame(checks).to_csv(ROOT / "results/140_fixed_energy_mapping_qa.csv", index=False)
    manifest = []
    for path in sorted(SOURCE.glob("cec*")):
        manifest.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(SOURCE / "cec_provenance_hashes.csv", index=False)
    protocol = {"annual_energy_invariant": "weighted mean of each normalized complete shape equals one",
                "day_weights": DAY_WEIGHTS, "annual_template_assumption": "repeat the selected day-type template; months are shape scenarios, not an annual seasonal reconstruction",
                "shape_case_labels": {"low": "March-template", "avg": "average-template", "high": "October-template"},
                "presentation_url": "https://www.energy.ca.gov/sites/default/files/2025-11/2025_IEPR_Preliminary_Data_Center_Forecast_ada.pdf",
                "methodology_url": "https://www.energy.ca.gov/sites/default/files/2026-04/Data_Center_Methodology_Memo_ada.pdf",
                "digitization_precision_percentage_points": 0.1,
                "peak_window_policy": "unchanged 2025 JJA mean-diurnal diagnostic; not observed annual maximum or PV top-100 hours",
                "choice_independent_of_balance_results": True}
    (SOURCE / "fixed_energy_mapping_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(out[["model_region", "source_shape", "shape_mean_load_factor", "gamma"]].to_string(index=False))


if __name__ == "__main__":
    build()
