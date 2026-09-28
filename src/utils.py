import calendar
import json
import logging
import os
import joblib
import mlflow
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
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


def _col_array(df: pd.DataFrame, feat: str) -> np.ndarray:
    return df[feat].dropna().values.astype(float)


def save_feature_baseline(
    df: pd.DataFrame,
    features: list[str],
    categorical_features: set[str] | None = None,
) -> None:
    """Persist per-feature decile breakpoints for PSI drift detection."""
    skip = categorical_features or set()
    baseline: dict = {}
    for feat in features:
        if feat in skip:
            logger.debug(f"Skipping {feat} from baseline: nominal categorical")
            continue
        col = _col_array(df, feat)
        breakpoints = np.percentile(col, np.linspace(0, 100, 11))
        if len(np.unique(breakpoints)) < len(breakpoints):
            logger.debug(
                f"Skipping {feat} from baseline: bins collapse (low cardinality)"
            )
            continue
        baseline[feat] = {"breakpoints": breakpoints.tolist(), "n": len(col)}
    ensure_parent(FEATURE_STATS_PATH)
    with open(FEATURE_STATS_PATH, "w") as f:
        json.dump(baseline, f)
    logger.info(f"Feature baseline saved → {FEATURE_STATS_PATH}")


def _compute_psi(breakpoints: list[float], actual: np.ndarray) -> float:
    bins = np.array(breakpoints, dtype=float)
    bins[0], bins[-1] = -np.inf, np.inf
    n_bins = len(bins) - 1
    expected_pct = np.full(n_bins, 1.0 / n_bins)
    actual_counts = np.histogram(actual, bins=bins)[0]
    actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
    actual_pct = actual_pct / actual_pct.sum()
    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )


def detect_feature_drift(df: pd.DataFrame, threshold: float = 0.2) -> list[str]:
    if not FEATURE_STATS_PATH.exists():
        logger.info("No feature baseline — skipping drift check")
        return []
    with open(FEATURE_STATS_PATH) as f:
        baseline = json.load(f)
    drifted = []
    for feat, stats in baseline.items():
        if feat not in df.columns:
            continue
        psi = _compute_psi(stats["breakpoints"], _col_array(df, feat))
        if psi > threshold:
            logger.warning(f"Feature drift: {feat}  PSI={psi:.3f} > {threshold}")
            drifted.append(feat)
    return drifted


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
    if timestamp_col not in df.columns:
        raise ValueError(
            f"timestamp_col '{timestamp_col}' not found in DataFrame columns: {list(df.columns)}"
        )
    min_ts = df[timestamp_col].min()
    max_ts = df[timestamp_col].max()
    cutoff = min_ts + (max_ts - min_ts) * ratio
    train = df[df[timestamp_col] < cutoff].copy()
    test = df[df[timestamp_col] >= cutoff].copy()
    return train, test


def is_valid_file(file: Path, year: int, month: int) -> bool:
    try:
        df = pd.read_parquet(file, columns=["tpep_pickup_datetime"])
        last_day = calendar.monthrange(year, month)[1]
        expected_last = pd.Timestamp(year=year, month=month, day=last_day)
        return df["tpep_pickup_datetime"].max().normalize() >= expected_last
    except (
        Exception
    ):  # noqa: BLE001 — any read/parse failure means the file is unusable
        return False


def get_features_and_target(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features]
    y = df[target]

    return X, y


def train_linear(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LinearRegression, dict]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear regression training complete")
    return model, {}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 9,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    sample_weight: np.ndarray | None = None,
) -> tuple[XGBRegressor, dict]:
    parameters = {"n_estimators": n_estimators, "learning_rate": learning_rate}
    model = XGBRegressor(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("XGBoost training complete")
    return model, parameters


def split_data(
    df: pd.DataFrame, features: list[str], target: str, timestamp_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train, test = time_split_df(df, timestamp_col)
    X_train, y_train = get_features_and_target(train, features, target)
    X_test, y_test = get_features_and_target(test, features, target)
    logger.info("Data split complete")
    return X_train, X_test, y_train, y_test


def save_model(model: BaseEstimator, path: Path) -> None:
    ensure_parent(path)
    if path.exists():
        path.replace(path.with_suffix(".prev.joblib"))
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_to_mlflow(
    model_name: str,
    features: list[str],
    params: dict,
    mae: float | None,
    mape: float | None = None,
    tags: dict | None = None,
    shap_importances: dict[str, float] | None = None,
    metrics: dict[str, float] | None = None,
    experiment: str = "default",
    model: BaseEstimator | None = None,
) -> None:
    import mlflow.sklearn

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        if mae is not None:
            mlflow.log_metric("mae", mae)
        if mape is not None:
            mlflow.log_metric("mape", mape)
        if params:
            mlflow.log_params(params)
        if features:
            mlflow.log_param("features", ", ".join(features))
        if tags:
            mlflow.set_tags(tags)
        if shap_importances:
            mlflow.log_metrics({f"shap_{k}": v for k, v in shap_importances.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        if model is not None:
            mlflow.sklearn.log_model(model, "model")

    if model is not None:
        try:
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", mv.version)
            logger.info(
                f"Registered {model_name} v{mv.version} as champion in MLflow registry"
            )
        except Exception as e:
            logger.warning(f"MLflow registry unavailable — model not registered: {e}")


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
