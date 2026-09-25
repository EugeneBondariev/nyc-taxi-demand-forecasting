import joblib
import pandas as pd
import calendar
import logging
from pathlib import Path
from collections.abc import Callable
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from xgboost import XGBRegressor
from .database import ModelVersion

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def predict_and_evaluate(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float]:
    predictions = model.predict(X_test)
    mae = round(mean_absolute_error(y_pred=predictions, y_true=y_test), 2)
    mape = round(mean_absolute_percentage_error(y_pred=predictions, y_true=y_test), 2)

    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MAPE: {mape:.1%}")

    return mae, mape


def is_valid_file(file: Path, year: int, month: int) -> bool:
    try:
        df = pd.read_parquet(file, columns=["tpep_pickup_datetime"])
        last_day = calendar.monthrange(year, month)[1]
        expected_last = pd.Timestamp(year=year, month=month, day=last_day)
        return df["tpep_pickup_datetime"].max().normalize() >= expected_last
    except Exception:
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


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 9) -> tuple[XGBRegressor, dict]:
    parameters = {"n_estimators": 300, "learning_rate": 0.05}
    model = XGBRegressor(
        n_estimators=parameters["n_estimators"],
        random_state=42,
        max_depth=max_depth,
        learning_rate=parameters["learning_rate"],
    )
    model.fit(X_train, y_train)
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
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


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
