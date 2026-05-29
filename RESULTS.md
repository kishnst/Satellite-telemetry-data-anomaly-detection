# Project Results — Satellite Telemetry Anomaly Detection

Summary of outputs from `notebooks/satellite_anomaly_detection.ipynb` (same pipeline as `scripts/evaluate.py`).

**Last generated:** 2026-05-29 (UTC) · **Random seed:** 42

---

## Project overview

Machine learning pipeline that detects anomalies in **simulated** satellite telemetry. Uses Python, Pandas, and Scikit-learn for preprocessing and classification; Matplotlib/Seaborn for plots in the notebook.

**Notebook:** `notebooks/satellite_anomaly_detection.ipynb`  
**Raw metrics (JSON):** `results/metrics.json`

---

## Dataset

| Item | Value |
|------|------:|
| Total samples | 6,000 |
| Labeled anomalies | 279 |
| Anomaly rate | 4.65% |
| Sensor channels | 8 (power, solar, CPU temp, 3× gyro, comms, storage) |
| Engineered features | 43 |
| Train / test split | 4,498 / 1,500 (75% / 25%, stratified) |

**Fault types injected:** battery sag, gyro spike, thermal spike, comm dropout, storage drift.

---

## Feature engineering

Per-sensor rolling mean & std (window = 15), first differences, z-scores vs rolling baseline, plus:

- `gyro_magnitude` — combined attitude rate
- `power_ratio` — solar current / battery voltage
- `thermal_per_volt` — CPU temp / battery voltage

---

## Model tuning (5-fold stratified CV, scoring = F1)

### Random Forest

| Hyperparameter | Best value |
|----------------|------------|
| `n_estimators` | 100 |
| `max_depth` | 8 |
| `min_samples_leaf` | 5 |
| **CV F1** | **0.947** |

### Gradient Boosting

| Hyperparameter | Best value |
|----------------|------------|
| `n_estimators` | 120 |
| `learning_rate` | 0.1 |
| `max_depth` | 5 |
| **CV F1** | **0.902** |

### Isolation Forest (baseline)

Unsupervised; `contamination` set to training anomaly rate. No grid search.

---

## Hold-out test metrics

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|------------:|-------:|---:|--------:|
| Isolation Forest | 0.21 | 0.24 | 0.23 | 0.80 |
| Gradient Boosting (tuned) | 0.95 | 0.86 | 0.90 | 0.98 |
| **Random Forest (tuned)** | **1.00** | **0.87** | **0.93** | **0.99** |

**Best model (by F1):** Random Forest (tuned)

### Confusion matrix — Random Forest (tuned)

|  | Predicted normal | Predicted anomaly |
|--|----------------:|------------------:|
| **Actual normal** | 1,430 | 0 |
| **Actual anomaly** | 9 | 61 |

- **Accuracy:** 99.4%
- **Normal class F1:** 0.997
- **Anomaly class F1:** 0.931

### Confusion matrix — Gradient Boosting (tuned)

|  | Predicted normal | Predicted anomaly |
|--|----------------:|------------------:|
| **Actual normal** | 1,427 | 3 |
| **Actual anomaly** | 10 | 60 |

---

## Top features (Random Forest importances)

| Rank | Feature | Importance |
|-----:|---------|----------:|
| 1 | `storage_used_pct_roll_std` | 0.173 |
| 2 | `storage_used_pct_z` | 0.135 |
| 3 | `storage_used_pct_delta` | 0.099 |
| 4 | `storage_used_pct` | 0.095 |
| 5 | `thermal_per_volt` | 0.041 |
| 6 | `cpu_temp_c` | 0.029 |
| 7 | `comm_signal_dbm` | 0.024 |
| 8 | `comm_signal_dbm_delta` | 0.023 |
| 9 | `cpu_temp_c_z` | 0.023 |
| 10 | `battery_voltage_v` | 0.022 |

Storage-related rolling and z-score features dominate, which matches the injected storage-drift fault type.

---

## Notebook visualizations

The notebook produces:

1. Raw telemetry time series with anomaly markers (8 sensors)
2. Model comparison bar chart (F1 vs ROC-AUC)
3. Confusion matrix for the best model
4. ROC curves (all models)
5. Precision–recall curve (best model)
6. Timeline overlay (true vs predicted anomalies on a sensor channel)
7. Feature importance bar chart (tuned Random Forest)

---

## Regenerate metrics

```bash
source .venv/bin/activate
python scripts/evaluate.py
```

Updates `results/metrics.json`. Edit this file or re-run the script after notebook changes to refresh numbers.
