import logging
from collections.abc import Callable
from pathlib import Path
import pandas as pd
from sqlalchemy.engine import Engine
from .config import (
    TAXI_DATA_FOLDER,
    FARE_AMOUNT_FEATURES_A,
    FARE_AMOUNT_FEATURES_B,
    FARE_TARGET,
    MODEL_PATH_FARE_A,
    MODEL_PATH_FARE_B,
    ModelName,
    DB_URL,
)
from .database import init_db
from .features import load_and_clean_taxi_data
from .logger import setup_logging
from .utils import (
    predict_and_evaluate,
    split_data,
    save_model,
    save_to_database,
    train_xgboost,
    train_linear,
)

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)


def train_fare_version(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_fn: Callable,
    model_path: Path,
    model_name: str,
    **model_kwargs,
) -> None:
    X_train, X_test, y_train, y_test = split_data(
        df, features, target, "tpep_pickup_datetime"
    )
    model, params = model_fn(X_train, y_train, **model_kwargs)
    mae, mape = predict_and_evaluate(model, X_test, y_test)
    model_path.parent.mkdir(exist_ok=True)
    save_model(model, model_path)
    save_to_database(mae, mape, params, model_name, engine)


def run_training_pipeline() -> None:
    df = load_and_clean_taxi_data(TAXI_DATA_FOLDER)

    # train_fare_version(
    #     df,
    #     FARE_AMOUNT_FEATURES_A,
    #     FARE_TARGET,
    #     train_linear,
    #     MODEL_PATH_FARE_A,
    #     ModelName.FARE_LINEAR.value,
    # )
    train_fare_version(
        df.sample(frac=0.1, random_state=42),
        FARE_AMOUNT_FEATURES_B,
        FARE_TARGET,
        train_xgboost,
        MODEL_PATH_FARE_B,
        ModelName.FARE_XGB.value,
        max_depth=6,
    )


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
