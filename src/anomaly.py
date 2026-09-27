import logging
import os

import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import DEMAND_DATA, MLFLOW_TRACKING_URI, ROOT
from .logger import setup_logging
from .utils import load_data

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

ANOMALY_PATH = ROOT / "data" / "processed" / "anomalies.parquet"
FEATURES = ["PULocationID", "pickup_hour", "pickup_dow", "trip_count",
            "temperature_2m", "precipitation", "snowfall", "is_holiday"]
CONTAMINATION = 0.01  # expected fraction of anomalies


def run_anomaly_detection(contamination: float = CONTAMINATION) -> None:
    demand = load_data(DEMAND_DATA)
    demand = demand.dropna(subset=FEATURES)

    cutoff = int(len(demand) * 0.8)
    train_df = demand.iloc[:cutoff]
    full_df = demand.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURES])
    X_full = scaler.transform(full_df[FEATURES])

    logger.info(f"Fitting IsolationForest (contamination={contamination})...")
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(X_train)

    full_df["anomaly_score"] = iso.score_samples(X_full)   # lower = more anomalous
    full_df["is_anomaly"] = iso.predict(X_full) == -1

    n_anomalies = full_df["is_anomaly"].sum()
    rate = n_anomalies / len(full_df)
    logger.info(f"Flagged {n_anomalies:,} anomalies ({rate:.2%} of records)")

    top = (
        full_df[full_df["is_anomaly"]]
        .nsmallest(10, "anomaly_score")[
            ["PULocationID", "pickup_hour_ts", "trip_count", "anomaly_score"]
        ]
    )
    logger.info(f"Top 10 most anomalous records:\n{top.to_string(index=False)}")

    ANOMALY_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df[full_df["is_anomaly"]].to_parquet(ANOMALY_PATH, index=False)
    logger.info(f"Saved anomalies → {ANOMALY_PATH}")

    mlflow.set_experiment("anomaly_detection")
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        mlflow.log_metric("n_anomalies", int(n_anomalies))
        mlflow.log_metric("anomaly_rate", round(rate, 4))
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("n_estimators", 100)


if __name__ == "__main__":
    setup_logging()
    run_anomaly_detection()
