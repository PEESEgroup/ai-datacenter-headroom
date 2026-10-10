"""Export existing annual and summer results for the same 2025 year."""
from pathlib import Path
import csv
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'plotting_tables'
KEYS = ['ISO', 'pv_ratio', 'storage_hours']


def main():
    years = pd.read_csv(TABLES / 'pv_annual_weather_years.csv')
    annual = years.loc[years.year.eq(2025), KEYS + ['annual_displacement_share']].rename(
        columns={'annual_displacement_share': 'annual_displacement_2025_share'})
    display = pd.read_csv(TABLES / 'pv_display_with_variability.csv').rename(
        columns={'annual_displacement_share': 'annual_displacement_2016_2025_share',
                 'summer_peak_p95_residual': 'summer_peak_p95_residual_2025_share'})
    grid = pd.read_csv(TABLES / 'pv_recomputed_full_grid.csv')
    grid['pv_ratio'] = grid['pv_ratio'].astype(float)
    mean = grid[KEYS + ['summer_peak_mean_residual']].rename(
        columns={'summer_peak_mean_residual': 'summer_peak_mean_residual_2025_share'})
    merged = display.merge(annual, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    assert len(merged) == 12 and merged['_merge'].eq('both').all()
    merged = merged.drop(columns='_merge').merge(mean, on=KEYS, how='left', validate='one_to_one')
    assert merged['summer_peak_mean_residual_2025_share'].notna().all()
    merged['year'] = 2025
    merged['dispatch'] = 'PV-only charging; causal self-consumption'
    merged['site_aggregation'] = 'equal-weight hourly mean across five independently dispatched sites'
    merged.to_csv(TABLES / 'pv_2025_configuration_comparison.csv', index=False)
    fields = ['Region/configuration', 'Ten-year energy displaced',
              '2025 energy displaced', 'Summer mean residual',
              'Summer P95 residual', 'Full-year maximum']
    maximum = grid.set_index(KEYS)['full_year_max_residual']
    with (ROOT / 'display_tables/SI_revision_20261003__pv_results_cells.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for region in ['PJM', 'MISO', 'SPP', 'ERCOT']:
            for ratio, hours in [(1, 0), (2, 4), (3, 8)]:
                row = merged.loc[merged.ISO.eq(region) & merged.pv_ratio.eq(ratio)
                                 & merged.storage_hours.eq(hours)].iloc[0]
                writer.writerow([f'{region}: {ratio}x PV + {hours}h'] + [f'{100 * value:.1f}' for value in [
                    row.annual_displacement_2016_2025_share,
                    row.annual_displacement_2025_share,
                    row.summer_peak_mean_residual_2025_share,
                    row.summer_peak_p95_residual_2025_share,
                    maximum.loc[(region, ratio, hours)]]])
    print('Exported 12 same-year configuration comparisons; no model assumptions changed.')


if __name__ == '__main__':
    main()
