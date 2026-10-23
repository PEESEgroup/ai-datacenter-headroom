"""Make physical-hour results authoritative in reader-facing PV exports."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "physical_intervals"


def main():
    subprocess.run([sys.executable, str(HERE / "run_interval_check.py"), "--numeric-root", str(ROOT/"numerical")],check=True)
    out = HERE / "interval_results"
    grid = pd.read_csv(out / "pv_storage_fine_grid.csv")
    sites = pd.read_csv(out / "pv_storage_site_cases.csv")
    annual = pd.read_csv(out / "pv_annual_energy.csv")
    reference = pd.read_csv(HERE / "reference_results/pv_storage_fine_grid.csv")
    pd.testing.assert_frame_equal(grid,reference,check_exact=False,atol=1e-12,rtol=1e-12)
    for path in [ROOT/"numerical/results/intervention/pv_storage_fine_grid.csv",ROOT/"plotting_tables/pv_recomputed_full_grid.csv"]:
        grid.to_csv(path,index=False)
    sites.to_csv(ROOT/"plotting_tables/pv_2025_site_peak_variability.csv",index=False)
    for relative in ["plotting_tables/pv_recomputed_site_years.csv","numerical/results/intervention/pv_storage_site_cases.csv"]:
        path=ROOT/relative
        if path.exists():
            old=pd.read_csv(path)
            keys=["ISO","representative_id","year","pv_ratio","storage_hours"]
            revised=old.drop(columns=[c for c in ["summer_peak_mean_residual","summer_peak_p95_residual"] if c in old]).merge(sites,on=keys,validate="one_to_one")
            assert len(revised)==len(old)
            revised.to_csv(path,index=False)
    annual.to_csv(ROOT/"plotting_tables/pv_annual_weather_years.csv",index=False)
    fixed = grid.merge(annual.groupby(["ISO","pv_ratio","storage_hours"]).annual_displacement_share.agg(yearly_displacement_min="min",yearly_displacement_max="max").reset_index(),on=["ISO","pv_ratio","storage_hours"],validate="one_to_one")
    fixed.to_csv(ROOT/"plotting_tables/pv_display_with_variability.csv",index=False)
    fixed.to_csv(ROOT/"source_data/displayed_pv_storage_results.csv",index=False)
    subprocess.run([sys.executable,str(ROOT/"export_2025_pv_comparison.py")],check=True)
    comparison=pd.read_csv(ROOT/"plotting_tables/pv_2025_configuration_comparison.csv")
    comparison["physical_time_join"]="EIA hour ending t -> POWER hour starting t-1h"
    comparison.to_csv(ROOT/"plotting_tables/pv_2025_configuration_comparison.csv",index=False)
    (HERE/"export_verification.json").write_text(json.dumps(dict(passed=True,grid_points=len(grid),source_corrected_join=True,legacy_checks_separate=True),indent=2)+"\n")


if __name__ == "__main__":
    main()
