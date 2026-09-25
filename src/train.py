import logging
import os
import mlflow
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from .utils import (
    predict_and_evaluate,
    load_data,
    split_data,
    save_model,
    save_to_database,
    train_xgboost,
)
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
from .database import init_db
from .logger import setup_logging

os.environ["MLFLOW_ARTIFACT_ROOT"] = str(ROOT / "mlflow_artifacts")

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)


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
        demand, demand_features, demand_target, "pickup_hour_ts"
    )
    model, parameters = train_xgboost(X_train, y_train)
    mae, mape = predict_and_evaluate(model, X_test, y_test)

    model_path.parent.mkdir(exist_ok=True)
    save_model(model, model_path)
    save_to_database(mae, mape, parameters, model_name, engine)


def run_training_pipeline() -> None:
    demand = load_data(DEMAND_DATA)
    process_ab_test_version(
        demand,
        DEMAND_FEATURES_A,
        DEMAND_TARGET,
        MODEL_PATH_A,
        ModelName.DEMAND_XGB.value,
        min_trips=0,
    )
    run_lstm_pipeline()


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
    run_mlflow()
