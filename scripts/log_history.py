"""One-time script to backfill historical experiment results into MLflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import log_to_mlflow

RUNS = [
    # LSTM: zone-based split (original, no time features)
    {
        "run_name": "demand_lstm",
        "params": {
            "split": "zone-based",
            "window_size": 24,
            "epochs": 3,
            "hidden_size": 64,
            "num_layers": 2,
        },
        "metrics": {"mae": 10.07},
        "features": ["trip_count"],
    },
    # LSTM: switched to time-based split
    {
        "run_name": "demand_lstm",
        "params": {
            "split": "time-based",
            "window_size": 24,
            "epochs": 3,
            "hidden_size": 64,
            "num_layers": 2,
        },
        "metrics": {"mae": 7.03},
        "features": ["trip_count"],
    },
    # LSTM: added pickup_hour and pickup_dow
    {
        "run_name": "demand_lstm",
        "params": {
            "split": "time-based",
            "window_size": 24,
            "epochs": 3,
            "hidden_size": 64,
            "num_layers": 2,
        },
        "metrics": {"mae": 6.62},
        "features": ["trip_count", "pickup_hour", "pickup_dow"],
    },
    # LSTM: added weather features (current model B)
    {
        "run_name": "demand_lstm",
        "params": {
            "split": "time-based",
            "window_size": 24,
            "epochs": 3,
            "hidden_size": 64,
            "num_layers": 2,
        },
        "metrics": {"mae": 6.64},
        "features": [
            "trip_count",
            "pickup_hour",
            "pickup_dow",
            "temperature_2m",
            "precipitation",
            "snowfall",
        ],
    },
    # Fare linear: before trip_duration
    {
        "run_name": "fare_linear",
        "params": {},
        "metrics": {"mae": 9.60, "mape": 1.14},
        "features": ["trip_distance", "Airport_fee", "extra"],
    },
    # Fare linear: after trip_duration (current model A)
    {
        "run_name": "fare_linear",
        "params": {},
        "metrics": {"mae": 4.43, "mape": 0.86},
        "features": ["trip_distance", "Airport_fee", "extra", "trip_duration"],
    },
    # Fare XGBoost: before trip_duration
    {
        "run_name": "fare_xgboost",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "sample_frac": 0.1,
        },
        "metrics": {"mae": 3.82, "mape": 0.86},
        "features": [
            "trip_distance",
            "Airport_fee",
            "extra",
            "PULocationID",
            "DOLocationID",
        ],
    },
    # Fare XGBoost: after trip_duration (current model B)
    {
        "run_name": "fare_xgboost",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "sample_frac": 0.1,
        },
        "metrics": {"mae": 2.36, "mape": 0.76},
        "features": [
            "trip_distance",
            "Airport_fee",
            "extra",
            "trip_duration",
            "PULocationID",
            "DOLocationID",
        ],
    },
]

if __name__ == "__main__":
    for run in RUNS:
        log_to_mlflow(
            model_name=run["run_name"],
            mae=run["metrics"]["mae"],
            features=run["features"],
            params=run["params"],
            mape=run["metrics"].get("mape"),
            tags={"source": "historical"},
        )

    print(f"Logged {len(RUNS)} historical runs.")
