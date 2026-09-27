import logging
import os

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEMAND_DATA, MLFLOW_TRACKING_URI, ROOT
from .logger import setup_logging
from .utils import load_data, time_split_df

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEATURES = [
    "PULocationID", "pickup_hour", "pickup_dow", "pickup_week",
    "temperature_2m", "precipitation", "snowfall", "is_holiday",
]
MODEL_PATH = ROOT / "models" / "demand_classifier.joblib"
HIGH_DEMAND_QUANTILE = 0.75


def make_target(demand: pd.DataFrame) -> pd.Series:
    threshold = demand["trip_count"].quantile(HIGH_DEMAND_QUANTILE)
    logger.info(f"High-demand threshold (p75): {threshold:.1f} trips/hour")
    return (demand["trip_count"] >= threshold).astype(int)


def run_training() -> None:
    demand = load_data(DEMAND_DATA)
    demand = demand.dropna(subset=FEATURES)

    y = make_target(demand)
    train_df, test_df = time_split_df(demand, "pickup_hour_ts")
    y_train = y.loc[train_df.index]
    y_test = y.loc[test_df.index]

    logger.info(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    logger.info(f"High-demand rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=1.0,
        )),
    ])

    logger.info("Training logistic regression...")
    pipeline.fit(train_df[FEATURES], y_train)

    y_pred = pipeline.predict(test_df[FEATURES])
    y_proba = pipeline.predict_proba(test_df[FEATURES])[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["normal", "high_demand"])

    logger.info(f"\n{report}")
    logger.info(f"ROC-AUC:           {auc:.4f}")
    logger.info(f"Average Precision: {ap:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Saved → {MODEL_PATH}")

    mlflow.set_experiment("classification")
    with mlflow.start_run(run_name="logistic_regression_high_demand"):
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("average_precision", ap)
        mlflow.log_param("features", ", ".join(FEATURES))
        mlflow.log_param("C", 1.0)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("high_demand_quantile", HIGH_DEMAND_QUANTILE)


if __name__ == "__main__":
    setup_logging()
    run_training()
