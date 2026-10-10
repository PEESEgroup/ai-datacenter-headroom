from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PANEL_ROOT = ROOT.parent / "inputs/price"
PANEL_SPECS = {
    region: (PANEL_ROOT / (region + ".csv.gz"), "zone_key_norm" if region == "ISO-NE" else "zone")
    for region in ["PJM", "MISO", "SPP", "ERCOT", "CAISO", "NYISO", "ISO-NE"]
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_demean(matrix: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.zeros((n_groups, matrix.shape[1]), dtype=float)
    np.add.at(sums, codes, matrix)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    means = sums / np.maximum(counts[:, None], 1.0)
    return matrix - means[codes]


def absorb_two_way(
    matrix: np.ndarray,
    zone_codes: np.ndarray,
    time_codes: np.ndarray,
    max_iter: int = 25,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int, float]:
    result = matrix.astype(float, copy=True)
    n_zone = int(zone_codes.max()) + 1
    n_time = int(time_codes.max()) + 1
    last_delta = np.inf
    for iteration in range(1, max_iter + 1):
        previous = result.copy()
        result = group_demean(result, zone_codes, n_zone)
        result = group_demean(result, time_codes, n_time)
        last_delta = float(np.max(np.abs(result - previous)))
        if last_delta < tolerance:
            break
    return result, iteration, last_delta


def cluster_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    cluster_codes: np.ndarray,
) -> np.ndarray:
    xtx_inv = np.linalg.pinv(x.T @ x)
    n_cluster = int(cluster_codes.max()) + 1
    scores = np.zeros((n_cluster, x.shape[1]), dtype=float)
    np.add.at(scores, cluster_codes, x * residual[:, None])
    meat = scores.T @ scores
    n = x.shape[0]
    k = x.shape[1]
    correction = (n_cluster / max(n_cluster - 1, 1)) * ((n - 1) / max(n - k, 1))
    return correction * xtx_inv @ meat @ xtx_inv


def prepare_panel(iso: str, path: Path, zone_col: str) -> tuple[pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(path, float_precision="round_trip")
    data = data.rename(columns={zone_col: "zone"})
    data["timestamp_utc"] = pd.to_datetime(data["ts_utc"], utc=True, errors="coerce")
    data["load_mw"] = pd.to_numeric(data["load_mw"], errors="coerce")
    data["lmp_raw"] = pd.to_numeric(data["lmp"], errors="coerce")
    data["zone"] = data["zone"].astype(str).str.strip()
    data = data[
        data["timestamp_utc"].notna()
        & data["load_mw"].notna()
        & data["lmp_raw"].notna()
        & ~data["zone"].str.upper().str.contains("TOTAL", na=False)
    ].copy()
    data["year"] = data["timestamp_utc"].dt.year
    data = data[data["year"].between(2023, 2025)].copy()

    counts = data.groupby("zone").size()
    keep_zones = counts[counts >= 1000].index
    data = data[data["zone"].isin(keep_zones)].copy()
    timestamp_counts = data.groupby("timestamp_utc")["zone"].transform("nunique")
    data = data[timestamp_counts >= 2].copy()

    quantiles = data.groupby("year")["lmp_raw"].quantile([0.005, 0.995]).unstack()
    lower = data["year"].map(quantiles[0.005])
    upper = data["year"].map(quantiles[0.995])
    data["lmp"] = data["lmp_raw"].clip(lower=lower, upper=upper)

    grouped = data.groupby(["zone", "year"])["load_mw"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    data["load_z"] = (data["load_mw"] - mean) / std.replace(0, np.nan)
    data = data.dropna(subset=["load_z"]).copy()
    data["date_utc"] = data["timestamp_utc"].dt.floor("D")
    data["ISO"] = iso
    data = data.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
    coverage = {
        "ISO": iso,
        "source_path": str(path.relative_to(ROOT.parent)),
        "source_sha256": sha256(path),
        "rows": len(data),
        "zones": data["zone"].nunique(),
        "first_timestamp": data["timestamp_utc"].min(),
        "last_timestamp": data["timestamp_utc"].max(),
        "years": ";".join(map(str, sorted(data["year"].unique()))),
        "price_missing_after_cleaning": int(data["lmp"].isna().sum()),
        "load_missing_after_cleaning": int(data["load_z"].isna().sum()),
    }
    return data, coverage


def fit_zone_slopes(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    zones = sorted(data["zone"].unique())
    zone_index = {zone: i for i, zone in enumerate(zones)}
    zone_codes = data["zone"].map(zone_index).to_numpy(int)
    _, time_codes = np.unique(data["timestamp_utc"].astype("int64").to_numpy(), return_inverse=True)
    _, cluster_codes = np.unique(data["date_utc"].astype("int64").to_numpy(), return_inverse=True)
    x = np.zeros((len(data), len(zones)), dtype=float)
    x[np.arange(len(data)), zone_codes] = data["load_z"].to_numpy(float)
    matrix = np.column_stack([data["lmp"].to_numpy(float), x])
    residualized, iterations, delta = absorb_two_way(matrix, zone_codes, time_codes)
    y_r = residualized[:, 0]
    x_r = residualized[:, 1:]
    beta, _, _, _ = np.linalg.lstsq(x_r, y_r, rcond=None)
    residual = y_r - x_r @ beta
    covariance = cluster_covariance(x_r, residual, cluster_codes)
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    dof = max(int(cluster_codes.max()), 1)
    t_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df=dof)
    rows = []
    for i, zone in enumerate(zones):
        zone_data = data[data["zone"] == zone]
        rows.append(
            {
                "ISO": data["ISO"].iloc[0],
                "zone": zone,
                "coefficient_dollars_per_mwh_per_within_zone_year_load_sd": beta[i],
                "clustered_standard_error": se[i],
                "t_value": t_value[i],
                "p_value": p_value[i],
                "observations": len(zone_data),
                "load_sd_mw_median_across_years": zone_data.groupby("year")["load_mw"].std().median(),
                "mean_lmp_raw": zone_data["lmp_raw"].mean(),
                "p95_lmp_raw": zone_data["lmp_raw"].quantile(0.95),
                "model": "zone-specific load slopes + zone FE + timestamp FE; SE clustered by UTC date",
                "interpretation": "descriptive conditional association; not causal",
            }
        )
    diagnostics = {
        "ISO": data["ISO"].iloc[0],
        "model": "zone-specific slopes",
        "observations": len(data),
        "zones": len(zones),
        "timestamps": int(data["timestamp_utc"].nunique()),
        "date_clusters": int(data["date_utc"].nunique()),
        "absorption_iterations": iterations,
        "absorption_last_max_delta": delta,
        "residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "positive_zone_share": float(np.mean(beta > 0)),
        "positive_p_lt_0_05_zone_share": float(np.mean((beta > 0) & (p_value < 0.05))),
    }
    return pd.DataFrame(rows), diagnostics


def fit_common_slope(data: pd.DataFrame, sample: str) -> dict[str, object] | None:
    if len(data) < 2000 or data["zone"].nunique() < 2 or data["timestamp_utc"].nunique() < 500:
        return None
    zone_codes, zone_labels = pd.factorize(data["zone"], sort=True)
    time_codes, _ = pd.factorize(data["timestamp_utc"], sort=True)
    cluster_codes, _ = pd.factorize(data["date_utc"], sort=True)
    matrix = np.column_stack([data["lmp"].to_numpy(float), data["load_z"].to_numpy(float)])
    residualized, iterations, delta = absorb_two_way(matrix, zone_codes, time_codes)
    y_r = residualized[:, 0]
    x_r = residualized[:, 1:2]
    denom = float((x_r.T @ x_r).item())
    if denom <= 0:
        return None
    beta = float((x_r.T @ y_r).item() / denom)
    residual = y_r - x_r[:, 0] * beta
    covariance = cluster_covariance(x_r, residual, cluster_codes)
    se = float(np.sqrt(max(covariance[0, 0], 0.0)))
    dof = max(int(cluster_codes.max()), 1)
    t_value = beta / se if se > 0 else np.nan
    p_value = float(2.0 * stats.t.sf(abs(t_value), df=dof)) if np.isfinite(t_value) else np.nan
    return {
        "ISO": data["ISO"].iloc[0],
        "sample": sample,
        "coefficient_dollars_per_mwh_per_within_zone_year_load_sd": beta,
        "clustered_standard_error": se,
        "t_value": t_value,
        "p_value": p_value,
        "observations": len(data),
        "zones": len(zone_labels),
        "timestamps": data["timestamp_utc"].nunique(),
        "date_clusters": data["date_utc"].nunique(),
        "absorption_iterations": iterations,
        "absorption_last_max_delta": delta,
        "model": "common load slope + zone FE + timestamp FE; SE clustered by UTC date",
        "interpretation": "descriptive conditional association; not causal",
    }


def sensitivity_samples(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    samples: list[tuple[str, pd.DataFrame]] = [("full_winsorized", data)]
    year_p99 = data.groupby("year")["lmp_raw"].transform(lambda s: s.quantile(0.99))
    samples.append(("exclude_top_1pct_raw_price", data[data["lmp_raw"] <= year_p99]))
    years = sorted(data["year"].unique())
    for year in years:
        samples.append((f"year_{year}", data[data["year"] == year]))
    if len(years) >= 2:
        for year in years:
            samples.append((f"leave_out_{year}", data[data["year"] != year]))
    return samples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    slope_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for iso, (path, zone_col) in PANEL_SPECS.items():
        panel, coverage = prepare_panel(iso, path, zone_col)
        coverage_rows.append(coverage)
        slopes, diagnostics = fit_zone_slopes(panel)
        slope_frames.append(slopes)
        diagnostic_rows.append(diagnostics)
        for sample_name, sample in sensitivity_samples(panel):
            result = fit_common_slope(sample.copy(), sample_name)
            if result is not None:
                sensitivity_rows.append(result)
        print(f"[{iso}] {len(panel):,} rows, {panel['zone'].nunique()} zones")

    slopes = pd.concat(slope_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    sensitivities = pd.DataFrame(sensitivity_rows)
    coverage = pd.DataFrame(coverage_rows)
    slopes.to_csv(OUT / "04_price_zone_timestamp_fe_slopes.csv", index=False)
    diagnostics.to_csv(OUT / "04_price_zone_timestamp_fe_diagnostics.csv", index=False)
    sensitivities.to_csv(OUT / "04_price_fe_robustness.csv", index=False)
    coverage.to_csv(OUT / "04_price_panel_coverage.csv", index=False)

    region_summary = (
        slopes.groupby("ISO", as_index=False)
        .agg(
            zones=("zone", "size"),
            median_conditional_slope=("coefficient_dollars_per_mwh_per_within_zone_year_load_sd", "median"),
            mean_conditional_slope=("coefficient_dollars_per_mwh_per_within_zone_year_load_sd", "mean"),
            positive_zone_share=("coefficient_dollars_per_mwh_per_within_zone_year_load_sd", lambda s: float((s > 0).mean())),
            positive_significant_zone_share=("p_value", lambda s: np.nan),
        )
    )
    sig_share = slopes.assign(sig=(slopes["coefficient_dollars_per_mwh_per_within_zone_year_load_sd"] > 0) & (slopes["p_value"] < 0.05)).groupby("ISO")["sig"].mean()
    region_summary["positive_significant_zone_share"] = region_summary["ISO"].map(sig_share)
    region_summary.to_csv(OUT / "04_price_fe_region_summary.csv", index=False)

    manifest = coverage[["ISO", "source_path", "source_sha256", "first_timestamp", "last_timestamp", "rows", "zones"]].copy()
    manifest["role"] = "harmonized zone-hour load and day-ahead LMP panel"
    manifest["estimand_limit"] = "descriptive conditional association only"
    manifest.to_csv(OUT / "04_price_source_manifest.csv", index=False)
    print("\n", region_summary.to_string(index=False))


if __name__ == "__main__":
    main()
