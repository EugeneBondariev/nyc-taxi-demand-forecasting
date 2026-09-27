import calendar
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from xgboost import XGBRegressor

from .config import FEATURE_STATS_PATH, MLFLOW_TRACKING_URI
from .database import ModelVersion

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = logging.getLogger(__name__)


def compute_sample_weights(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Combine two weights into one sample_weight array for XGBoost.

    Peak weight: Q80+ rows get 2× — reduces Q5 under-prediction.
    Zone weight: inverse of zone mean demand — balances low-demand zone MAPE.
    """
    q80 = float(y_train.quantile(0.80))
    peak_weight = np.where(y_train.values >= q80, 2.0, 1.0)

    zone_mean = (
        pd.DataFrame({"zone": X_train["PULocationID"].values, "y": y_train.values})
        .groupby("zone")["y"]
        .mean()
    )
    zone_demand = X_train["PULocationID"].map(zone_mean).fillna(zone_mean.mean())
    inv_weight = 1.0 / (zone_demand / zone_demand.mean() + 0.1)
    inv_weight = (inv_weight / inv_weight.mean()).values

    combined = peak_weight * inv_weight
    return (combined / combined.mean()).astype(np.float32)


def save_feature_baseline(df: pd.DataFrame, features: list[str]) -> None:
    """Persist per-feature decile breakpoints for PSI drift detection."""
    baseline: dict = {}
    for feat in features:
        col = df[feat].dropna().values.astype(float)
        baseline[feat] = {
            "breakpoints": np.percentile(col, np.linspace(0, 100, 11)).tolist(),
            "n": int(len(col)),
        }
    FEATURE_STATS_PATH.parent.mkdir(exist_ok=True)
    with open(FEATURE_STATS_PATH, "w") as f:
        json.dump(baseline, f)
    logger.info(f"Feature baseline saved → {FEATURE_STATS_PATH}")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def predict_and_evaluate(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float]:
    predictions = model.predict(X_test)
    mae = round(mean_absolute_error(y_pred=predictions, y_true=y_test), 2)
    mape = round(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test), 2)

    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MAPE: {mape:.1%}")

    return mae, mape


def time_split_df(
    df: pd.DataFrame, timestamp_col: str, ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_ts = df[timestamp_col].min()
    max_ts = df[timestamp_col].max()
    cutoff = min_ts + (max_ts - min_ts) * ratio
    return df[df[timestamp_col] < cutoff].copy(), df[df[timestamp_col] >= cutoff].copy()


def is_valid_file(file: Path, year: int, month: int) -> bool:
    try:
        df = pd.read_parquet(file, columns=["tpep_pickup_datetime"])
        last_day = calendar.monthrange(year, month)[1]
        expected_last = pd.Timestamp(year=year, month=month, day=last_day)
        return df["tpep_pickup_datetime"].max().normalize() >= expected_last
    except Exception:  # noqa: BLE001 — any read/parse failure means the file is unusable
        return False


def get_features_and_target(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features]
    y = df[target]

    return X, y


def train_linear(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[LinearRegression, dict]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear regression training complete")
    return model, {}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 9,
    sample_weight: np.ndarray | None = None,
) -> tuple[XGBRegressor, dict]:
    parameters = {"n_estimators": 300, "learning_rate": 0.05}
    model = XGBRegressor(
        n_estimators=parameters["n_estimators"],
        random_state=42,
        max_depth=max_depth,
        learning_rate=parameters["learning_rate"],
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("XGBoost training complete")
    return model, parameters


def split_data(
    df: pd.DataFrame, features: list[str], target: str, timestamp_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    min_ts = df[timestamp_col].min()
    max_ts = df[timestamp_col].max()
    cutoff = min_ts + (max_ts - min_ts) * 0.8
    train = df[df[timestamp_col] < cutoff]
    test = df[df[timestamp_col] >= cutoff]
    X_train, y_train = get_features_and_target(train, features, target)
    X_test, y_test = get_features_and_target(test, features, target)
    logger.info("Data split complete")
    return X_train, X_test, y_train, y_test


def save_model(model: RegressorMixin, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        path.replace(path.with_suffix(".prev.joblib"))
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def rollback_model(path: Path) -> bool:
    prev = path.with_suffix(".prev.joblib")
    if prev.exists():
        prev.replace(path)
        logger.info(f"Rolled back model at {path}")
        return True
    logger.warning(f"No previous model found for {path}")
    return False


def log_to_mlflow(
    model_name: str,
    mae: float,
    features: list[str],
    params: dict,
    mape: float | None = None,
    tags: dict | None = None,
    shap_importances: dict[str, float] | None = None,
    experiment: str = "default",
) -> None:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        mlflow.log_metric("mae", mae)
        if mape is not None:
            mlflow.log_metric("mape", mape)
        if params:
            mlflow.log_params(params)
        mlflow.log_param("features", ", ".join(features))
        if tags:
            mlflow.set_tags(tags)
        if shap_importances:
            mlflow.log_metrics({f"shap_{k}": v for k, v in shap_importances.items()})


def save_to_database(
    mae: float, mape: float, parameters: dict, model_name: str, engine: Engine
) -> None:
    with Session(engine) as session:
        session.add(
            ModelVersion(
                name=model_name,
                mae=mae,
                mape=mape,
                n_estimators=parameters.get("n_estimators"),
                learning_rate=parameters.get("learning_rate"),
            )
        )
        session.commit()
