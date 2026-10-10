"""Unchanged pure numerical functions from the frozen WP6 builder."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def solar_geometry_and_pv(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    ghi_w_m2: np.ndarray,
    air_temperature_c: np.ndarray,
    wind_speed_m_s: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    settings = config["pv_conversion"]
    day = timestamps.dayofyear.to_numpy(float)
    hour_midpoint = (
        timestamps.hour.to_numpy(float)
        + timestamps.minute.to_numpy(float) / 60.0
        + 0.5
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour_midpoint - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    true_solar_minutes = np.mod(hour_midpoint * 60.0 + equation_of_time + 4.0 * longitude, 1440.0)
    hour_angle = np.radians(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = math.radians(latitude)
    tilt_deg = min(max(abs(latitude), 10.0), 40.0)
    tilt_rad = math.radians(tilt_deg)
    cos_zenith = (
        math.sin(latitude_rad) * np.sin(declination)
        + math.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)
    cos_incidence = (
        math.sin(latitude_rad - tilt_rad) * np.sin(declination)
        + math.cos(latitude_rad - tilt_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_incidence = np.clip(cos_incidence, 0.0, 1.0)

    extraterrestrial_normal = 1367.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day / 365.0))
    extraterrestrial_horizontal = extraterrestrial_normal * cos_zenith
    kt = np.divide(
        ghi_w_m2,
        extraterrestrial_horizontal,
        out=np.zeros_like(ghi_w_m2, dtype=float),
        where=extraterrestrial_horizontal > 1.0,
    )
    kt = np.clip(kt, 0.0, 1.2)
    diffuse_fraction = np.empty_like(kt)
    low = kt <= 0.22
    middle = (kt > 0.22) & (kt <= 0.80)
    high = kt > 0.80
    diffuse_fraction[low] = 1.0 - 0.09 * kt[low]
    diffuse_fraction[middle] = (
        0.9511
        - 0.1604 * kt[middle]
        + 4.388 * kt[middle] ** 2
        - 16.638 * kt[middle] ** 3
        + 12.336 * kt[middle] ** 4
    )
    diffuse_fraction[high] = 0.165
    diffuse_fraction = np.clip(diffuse_fraction, 0.0, 1.0)
    dhi = diffuse_fraction * ghi_w_m2
    dni = np.divide(
        ghi_w_m2 - dhi,
        cos_zenith,
        out=np.zeros_like(ghi_w_m2, dtype=float),
        where=cos_zenith > 0.065,
    )
    dni = np.clip(dni, 0.0, 1400.0)
    poa_beam = dni * cos_incidence
    poa_sky = dhi * (1.0 + math.cos(tilt_rad)) / 2.0
    poa_ground = (
        float(settings["ground_albedo"])
        * ghi_w_m2
        * (1.0 - math.cos(tilt_rad))
        / 2.0
    )
    poa = np.clip(poa_beam + poa_sky + poa_ground, 0.0, 1600.0)
    denominator = (
        float(settings["faiman_u0_w_m2k"])
        + float(settings["faiman_u1_w_s_m3k"]) * np.clip(wind_speed_m_s, 0.0, None)
    )
    cell_temperature_c = air_temperature_c + np.divide(
        poa, denominator, out=np.zeros_like(poa), where=denominator > 0
    )
    temperature_factor = np.clip(
        1.0
        + float(settings["temperature_coefficient_per_c"])
        * (cell_temperature_c - 25.0),
        0.0,
        None,
    )
    dc_pu = poa / 1000.0 * temperature_factor
    ac_pu = np.clip(
        dc_pu * (1.0 - float(settings["system_losses_fraction"])),
        0.0,
        float(settings["ac_output_limit_per_kwdc"]),
    )
    return pd.DataFrame(
        {
            "ghi_w_m2": ghi_w_m2,
            "air_temperature_c": air_temperature_c,
            "wind_speed_m_s": wind_speed_m_s,
            "cos_zenith": cos_zenith,
            "dhi_w_m2": dhi,
            "dni_w_m2": dni,
            "poa_w_m2": poa,
            "cell_temperature_c": cell_temperature_c,
            "pv_ac_kw_per_kwdc": ac_pu,
            "surface_tilt_deg": tilt_deg,
        }
    )

def simulate_storage(
    pv_pu: np.ndarray,
    pv_ratio: float,
    storage_duration_h: float,
    rte: float,
    config: dict,
    warmup_days: int | None = None,
) -> dict[str, np.ndarray | float]:
    settings = config["dispatch"]
    load = float(settings["load_mw"])
    charge_power = float(settings["battery_charge_power_per_mw_load"])
    discharge_power = float(settings["battery_discharge_power_per_mw_load"])
    energy_capacity = float(storage_duration_h) * discharge_power
    eta_charge = math.sqrt(float(rte))
    eta_discharge = math.sqrt(float(rte))
    days = int(settings["warmup_days"] if warmup_days is None else warmup_days)
    warmup_hours = days * 24
    extended = np.concatenate([pv_pu[-warmup_hours:], pv_pu])
    residual = np.empty(len(extended), dtype=float)
    curtailed = np.empty(len(extended), dtype=float)
    direct_to_load = np.empty(len(extended), dtype=float)
    charged_from_pv_path = np.empty(len(extended), dtype=float)
    discharged_to_load_path = np.empty(len(extended), dtype=float)
    soc_path = np.empty(len(extended), dtype=float)
    soc = 0.0
    for hour, resource in enumerate(extended):
        pv = max(0.0, float(resource)) * float(pv_ratio)
        direct = min(load, pv)
        deficit = load - direct
        surplus = max(0.0, pv - load)
        if energy_capacity > 0:
            charged_from_pv = min(
                surplus,
                charge_power,
                max(0.0, energy_capacity - soc) / eta_charge,
            )
            soc += charged_from_pv * eta_charge
            discharged_to_load = min(deficit, discharge_power, soc * eta_discharge)
            soc -= discharged_to_load / eta_discharge
        else:
            charged_from_pv = 0.0
            discharged_to_load = 0.0
        residual[hour] = max(0.0, deficit - discharged_to_load)
        curtailed[hour] = max(0.0, surplus - charged_from_pv)
        direct_to_load[hour] = direct
        charged_from_pv_path[hour] = charged_from_pv
        discharged_to_load_path[hour] = discharged_to_load
        soc_path[hour] = soc
    return {
        "residual": residual[warmup_hours:],
        "curtailed": curtailed[warmup_hours:],
        "direct_to_load": direct_to_load[warmup_hours:],
        "charged_from_pv": charged_from_pv_path[warmup_hours:],
        "discharged_to_load": discharged_to_load_path[warmup_hours:],
        "soc": soc_path[warmup_hours:],
        "analysis_start_soc": float(soc_path[warmup_hours - 1]),
        "analysis_end_soc": float(soc_path[-1]),
    }
