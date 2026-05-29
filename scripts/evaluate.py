"""Run the notebook pipeline and export metrics to results/."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry_simulator import SENSOR_COLUMNS, generate_telemetry  # noqa: E402

RANDOM_STATE = 42
N_SAMPLES = 6000
ANOMALY_RATE = 0.045
TEST_SIZE = 0.25
FEATURE_WINDOW = 15


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp_idx").reset_index(drop=True).copy()
    out = out.drop_duplicates(subset=["timestamp_idx"])
    for col in SENSOR_COLUMNS:
        out[col] = out[col].interpolate(method="linear").bfill().ffill()
    out["storage_used_pct"] = out["storage_used_pct"].clip(0, 100)
    return out


def engineer_features(df: pd.DataFrame, window: int = FEATURE_WINDOW) -> pd.DataFrame:
    feat = df[["timestamp_idx"] + SENSOR_COLUMNS].copy()
    for col in SENSOR_COLUMNS:
        roll = feat[col].rolling(window, min_periods=3)
        feat[f"{col}_roll_mean"] = roll.mean()
        feat[f"{col}_roll_std"] = roll.std()
        feat[f"{col}_delta"] = feat[col].diff()
        feat[f"{col}_z"] = (feat[col] - feat[f"{col}_roll_mean"]) / (feat[f"{col}_roll_std"] + 1e-6)
    feat["gyro_magnitude"] = np.sqrt(
        feat["gyro_x_dps"] ** 2 + feat["gyro_y_dps"] ** 2 + feat["gyro_z_dps"] ** 2
    )
    feat["power_ratio"] = feat["solar_current_a"] / (feat["battery_voltage_v"] + 1e-6)
    feat["thermal_per_volt"] = feat["cpu_temp_c"] / (feat["battery_voltage_v"] + 1e-6)
    return feat.dropna()


def model_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> dict:
    raw = generate_telemetry(n_samples=N_SAMPLES, anomaly_rate=ANOMALY_RATE, seed=RANDOM_STATE)
    df = preprocess(raw)

    features = engineer_features(df)
    labels = df.loc[features.index, "is_anomaly"].values
    features = features.reset_index(drop=True)
    feature_cols = [c for c in features.columns if c != "timestamp_idx"]

    X_train, X_test, y_train, y_test = train_test_split(
        features[feature_cols],
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    iso = IsolationForest(contamination=float(labels.mean()), random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_s)
    iso_pred = (iso.predict(X_test_s) == -1).astype(int)
    iso_scores = -iso.score_samples(X_test_s)
    iso_proba = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

    rf_search = GridSearchCV(
        Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))]),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(X_train_s, y_train)

    gb_search = GridSearchCV(
        Pipeline([("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
        {
            "clf__n_estimators": [80, 120],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gb_search.fit(X_train_s, y_train)

    results = {}
    for name, pred, proba in [
        ("Isolation Forest", iso_pred, iso_proba),
        (
            "Random Forest (tuned)",
            rf_search.predict(X_test_s),
            rf_search.predict_proba(X_test_s)[:, 1],
        ),
        (
            "Gradient Boosting (tuned)",
            gb_search.predict(X_test_s),
            gb_search.predict_proba(X_test_s)[:, 1],
        ),
    ]:
        results[name] = model_metrics(y_test, pred, proba)

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pred = {
        "Isolation Forest": iso_pred,
        "Random Forest (tuned)": rf_search.predict(X_test_s),
        "Gradient Boosting (tuned)": gb_search.predict(X_test_s),
    }[best_name]

    report = classification_report(
        y_test, best_pred, target_names=["normal", "anomaly"], output_dict=True
    )

    rf_clf = rf_search.best_estimator_.named_steps["clf"]
    top_features = (
        pd.Series(rf_clf.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset": {
            "n_samples": int(len(raw)),
            "n_anomalies": int(raw["is_anomaly"].sum()),
            "anomaly_rate_pct": round(float(raw["is_anomaly"].mean()) * 100, 2),
            "n_features_after_engineering": len(feature_cols),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "sensor_channels": SENSOR_COLUMNS,
        },
        "cross_validation": {
            "random_forest_best_params": rf_search.best_params_,
            "random_forest_cv_f1": round(float(rf_search.best_score_), 4),
            "gradient_boosting_best_params": gb_search.best_params_,
            "gradient_boosting_cv_f1": round(float(gb_search.best_score_), 4),
        },
        "holdout_metrics": results,
        "best_model": {
            "name": best_name,
            "metrics": results[best_name],
            "classification_report": report,
        },
        "top_feature_importances_random_forest": {
            k: round(float(v), 4) for k, v in top_features.items()
        },
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    return payload


if __name__ == "__main__":
    main()
