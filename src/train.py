import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
import shap

from .config import (
    DB_URL,
    DEMAND_DATA,
    DEMAND_FEATURES_A,
    DEMAND_TARGET,
    MODEL_PATH_A,
    ROOT,
    ModelName,
)
from .database import init_db
from .logger import setup_logging
from .train_lstm import run_training_pipeline as run_lstm_pipeline
from .utils import (
    load_data,
    log_to_mlflow,
    predict_and_evaluate,
    save_model,
    save_to_database,
    split_data,
    train_xgboost,
)

os.environ["MLFLOW_ARTIFACT_ROOT"] = str(ROOT / "mlflow_artifacts")

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)


def run_mlflow() -> None:
    runs = mlflow.search_runs()
    logger.info(runs[["metrics.mae", "params.n_estimators", "params.learning_rate"]])


def compute_shap_importances(model, X_train: pd.DataFrame) -> dict[str, float]:
    explainer = shap.TreeExplainer(model)
    sample = X_train.sample(min(500, len(X_train)), random_state=42)
    values = explainer.shap_values(sample)
    return {col: float(abs(values[:, i]).mean()) for i, col in enumerate(X_train.columns)}


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
    shap_importances = compute_shap_importances(model, X_train)
    log_to_mlflow(
        model_name, mae, demand_features, parameters, mape,
        shap_importances=shap_importances, experiment="demand",
    )
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
