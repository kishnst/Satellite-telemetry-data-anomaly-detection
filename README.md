# Satellite Telemetry Anomaly Detection

Machine learning pipeline to detect anomalies in **simulated** satellite telemetry / sensor streams. Built with Python, Pandas, and Scikit-learn; exploration and reporting live in a Jupyter notebook with Matplotlib visualizations.

## Features

- Multivariate simulated telemetry (power, thermal, attitude, comms, storage)
- Pandas preprocessing and rolling-window **feature engineering**
- Scikit-learn classifiers with **hyperparameter tuning** (`GridSearchCV`)
- Performance metrics (precision, recall, F1, ROC-AUC) and anomaly plots

## Project layout

```
Space_track/
├── README.md
├── requirements.txt
├── src/
│   └── telemetry_simulator.py   # synthetic dataset generator
├── data/                        # optional exported CSV
└── notebooks/
    └── satellite_anomaly_detection.ipynb
```

## Setup

```bash
cd Space_track
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/satellite_anomaly_detection.ipynb
```

## Quick run (no notebook)

```bash
python -m src.telemetry_simulator
```

## Method overview

1. **Data** — `generate_telemetry()` produces labeled normal/anomaly samples (spikes, thermal faults, comm dropouts, storage drift).
2. **Features** — per-sensor rolling mean/std, deltas, gyro magnitude, power balance ratio.
3. **Models** — `IsolationForest` (unsupervised baseline) and tuned `RandomForestClassifier` / `GradientBoostingClassifier` on engineered features.
4. **Evaluation** — stratified train/test split, confusion matrix, classification report, ROC curve, time-series anomaly overlay.

## License

MIT (educational / portfolio use).
