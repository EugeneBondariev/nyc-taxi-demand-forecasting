import logging
import os
import mlflow
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sqlalchemy.orm import Session
from .utils import predict_and_evaluate, get_features_and_target
from .config import (
    ROOT,
    DEMAND_DATA,
    MODEL_PATH_A,
    ModelName,
    DEMAND_FEATURES_A,
    DEMAND_TARGET,
    DB_URL,
)
from .train_lstm import run_training_pipeline as run_lstm_pipeline
from .database import init_db, ModelVersion
from .logger import setup_logging

os.environ["MLFLOW_ARTIFACT_ROOT"] = str(ROOT / "mlflow_artifacts")

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DEMAND_DATA)


def split_data(
    demand: pd.DataFrame, demand_features: list[str], demand_target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    min_ts = demand["pickup_hour_ts"].min()
    max_ts = demand["pickup_hour_ts"].max()
    cutoff = min_ts + (max_ts - min_ts) * 0.8

    train = demand[demand["pickup_hour_ts"] < cutoff]
    test = demand[demand["pickup_hour_ts"] >= cutoff]

    X_train, y_train = get_features_and_target(train, demand_features, demand_target)
    X_test, y_test = get_features_and_target(test, demand_features, demand_target)

    logger.info("Data split complete")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBRegressor, dict]:
    parameters = {"n_estimators": 300, "learning_rate": 0.05}
    model = XGBRegressor(
        n_estimators=parameters["n_estimators"],
        random_state=42,
        max_depth=9,
        learning_rate=parameters["learning_rate"],
    )
    model.fit(X_train, y_train)
    logger.info("The model training complete")

    return model, parameters


def tune_model(X_train, y_train) -> None:
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [6, 9],
        "learning_rate": [0.05],
        "min_child_weight": [1],
    }
    model = XGBRegressor(random_state=42)
    search = GridSearchCV(
        model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    search.fit(X_train, y_train)
    logger.info("Best params:", search.best_params_)
    logger.info("Best CV MAE:", -search.best_score_)
    return search.best_estimator_


def save_model(model: XGBRegressor, path: Path) -> None:
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def save_to_database(
    mae: float, mape: float, parameters: dict, model_name: str
) -> None:
    with Session(engine) as session:
        session.add(
            ModelVersion(
                name=model_name,
                mae=mae,
                mape=mape,
                n_estimators=parameters["n_estimators"],
                learning_rate=parameters["learning_rate"],
            )
        )
        session.commit()


def run_mlflow() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs()
    logger.info(runs[["metrics.mae", "params.n_estimators", "params.learning_rate"]])


def process_ab_test_version(
    demand: pd.DataFrame,
    demand_features: list[str],
    demand_target: str,
    model_path: Path,
    model_name: str,
    min_trips: int,
):
    demand = demand[demand["trip_count"] > min_trips]
    X_train, X_test, y_train, y_test = split_data(
        demand, demand_features, demand_target
    )
    model, parameters = train_model(X_train, y_train)
    mae, mape = predict_and_evaluate(model, X_test, y_test)

    model_path.parent.mkdir(exist_ok=True)
    save_model(model, model_path)
    save_to_database(mae, mape, parameters, model_name)


def run_training_pipeline() -> None:
    demand = load_data()
    process_ab_test_version(
        demand,
        DEMAND_FEATURES_A,
        DEMAND_TARGET,
        MODEL_PATH_A,
        ModelName.A.value,
        min_trips=0,
    )
    run_lstm_pipeline()


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
    run_mlflow()
