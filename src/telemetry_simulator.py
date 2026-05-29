"""Simulated satellite telemetry / sensor data for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "battery_voltage_v",
    "solar_current_a",
    "cpu_temp_c",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "comm_signal_dbm",
    "storage_used_pct",
]


def generate_telemetry(
    n_samples: int = 5000,
    anomaly_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
  Generate multivariate telemetry with injected anomalies.

  Normal operation follows smooth diurnal + orbital patterns; anomalies
  are point spikes, drift, or correlated multi-sensor faults.
  """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float)

    # Orbital period ~ 90 min sampled every 10 s -> 540 samples/orbit
    orbit_phase = 2 * np.pi * t / 540
    eclipse = (np.sin(orbit_phase - np.pi / 2) < -0.2).astype(float)

    battery = 28.0 - 0.8 * eclipse + 0.05 * np.sin(orbit_phase / 3)
    solar = 2.5 * (1 - eclipse) + 0.1 * rng.normal(size=n_samples)
    cpu_temp = 35 + 8 * eclipse + 0.3 * np.sin(orbit_phase / 7)
    gyro_x = 0.02 * np.sin(orbit_phase) + 0.005 * rng.normal(size=n_samples)
    gyro_y = 0.015 * np.cos(orbit_phase * 1.1) + 0.005 * rng.normal(size=n_samples)
    gyro_z = 0.01 * np.sin(orbit_phase * 0.7) + 0.005 * rng.normal(size=n_samples)
    comm = -72 + 2 * rng.normal(size=n_samples)
    storage = np.clip(40 + 0.002 * t + 0.5 * rng.normal(size=n_samples), 0, 100)

    df = pd.DataFrame(
        {
            "timestamp_idx": t.astype(int),
            "battery_voltage_v": battery,
            "solar_current_a": solar,
            "cpu_temp_c": cpu_temp,
            "gyro_x_dps": gyro_x,
            "gyro_y_dps": gyro_y,
            "gyro_z_dps": gyro_z,
            "comm_signal_dbm": comm,
            "storage_used_pct": storage,
        }
    )

    labels = np.zeros(n_samples, dtype=int)
    target_anomaly_samples = int(n_samples * anomaly_rate)
    used = set()
    attempts = 0
    max_attempts = target_anomaly_samples * 20

    while labels.sum() < target_anomaly_samples and attempts < max_attempts:
        attempts += 1
        idx = int(rng.integers(0, n_samples - 20))
        if idx in used:
            continue
        used.add(idx)
        fault = int(rng.integers(0, 5))

        if fault == 0:
            df.loc[idx, "battery_voltage_v"] -= rng.uniform(4, 7)
            labels[idx] = 1
        elif fault == 1:
            df.loc[idx, ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]] += rng.uniform(0.5, 1.2)
            labels[idx] = 1
        elif fault == 2:
            df.loc[idx, "cpu_temp_c"] += rng.uniform(15, 25)
            labels[idx] = 1
        elif fault == 3:
            df.loc[idx, "comm_signal_dbm"] -= rng.uniform(12, 20)
            labels[idx] = 1
        else:
            span = min(int(rng.integers(8, 18)), n_samples - idx)
            drift = np.linspace(0, rng.uniform(15, 25), span)
            end = idx + span
            if labels[idx:end].sum() >= target_anomaly_samples - labels.sum():
                continue
            df.loc[idx : end - 1, "storage_used_pct"] += drift
            labels[idx:end] = 1

    df["is_anomaly"] = labels
    return df


