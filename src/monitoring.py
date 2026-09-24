import logging
import joblib
import pandas as pd
from pathlib import Path
from .features import (
    download_taxi_data,
    load_and_clean_taxi_data,
    build_demand_table,
)
from .utils import predict_and_evaluate, get_features_and_target
from .config import (
    MODEL_PATH_A,
    MODEL_PATH_B,
    TAXI_DATA_FOLDER,
    DEMAND_FEATURES_A,
    DEMAND_FEATURES_B,
    DEMAND_TARGET,
    MAE_THRESHOLD,
    MAPE_THRESHOLD,
)
from .train import run_training_pipeline
from .logger import setup_logging

logger = logging.getLogger(__name__)


def compare_predictions(mae: float, mape: float) -> None:
    if mae > MAE_THRESHOLD or mape > MAPE_THRESHOLD:
        logger.warning("Trigger retraining")
        run_training_pipeline()


def monitor_ab_test_version(
    demand: pd.DataFrame,
    demand_features: list[str],
    demand_target: str,
    model_path: Path,
):
    X, y = get_features_and_target(demand, demand_features, demand_target)
    model = joblib.load(model_path)
    mae, mape = predict_and_evaluate(model, X, y)
    compare_predictions(mae, mape)


if __name__ == "__main__":
    setup_logging()
    download_taxi_data(2025)
    df = load_and_clean_taxi_data(TAXI_DATA_FOLDER, 2025, 1)
    demand = build_demand_table(df)

    monitor_ab_test_version(demand, DEMAND_FEATURES_A, DEMAND_TARGET, MODEL_PATH_A)
    monitor_ab_test_version(demand, DEMAND_FEATURES_B, DEMAND_TARGET, MODEL_PATH_B)
