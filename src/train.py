import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from .config import (
    CONFORMAL_MARGIN_PATH,
    DB_URL,
    DEMAND_DATA,
    DEMAND_FEATURES_V2,
    DEMAND_TARGET,
    MODEL_PATH_A,
    ROOT,
    ModelName,
)
from .database import init_db
from .features import add_lag_features
from .logger import setup_logging
from .train_lstm import run_training_pipeline as run_lstm_pipeline
from .utils import (
    compute_sample_weights,
    ensure_parent,
    load_data,
    log_to_mlflow,
    predict_and_evaluate,
    save_feature_baseline,
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


def compute_shap_importances(
    model, X_train: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame, "shap.Explanation"]:
    explainer = shap.TreeExplainer(model)
    sample = X_train.sample(min(500, len(X_train)), random_state=42)
    shap_values = explainer(sample)
    importances = {
        col: float(abs(shap_values[:, i].values).mean())
        for i, col in enumerate(X_train.columns)
    }
    return importances, sample, shap_values


def log_shap_plot(shap_values: "shap.Explanation", sample: pd.DataFrame) -> None:
    shap.summary_plot(shap_values, sample, show=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "shap_summary.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=100)
        plt.close()
        mlflow.log_artifact(str(plot_path), artifact_path="shap")


def process_ab_test_version(
    demand: pd.DataFrame,
    demand_features: list[str],
    demand_target: str,
    model_path: Path,
    model_name: str,
    min_trips: int,
    conformal_margin_path: Path | None = None,
):
    demand = demand[demand["trip_count"] > min_trips]
    X_train, X_test, y_train, y_test = split_data(
        df=demand,
        features=demand_features,
        target=demand_target,
        timestamp_col="pickup_hour_ts",
    )
    sample_weight = compute_sample_weights(X_train=X_train, y_train=y_train)
    model, parameters = train_xgboost(
        X_train=X_train, y_train=y_train, sample_weight=sample_weight
    )
    save_feature_baseline(
        df=X_train, features=demand_features, categorical_features={"PULocationID"}
    )
    mae, mape = predict_and_evaluate(model=model, X_test=X_test, y_test=y_test)
    shap_importances, shap_sample, shap_values = compute_shap_importances(
        model, X_train
    )
    log_to_mlflow(
        model_name=model_name,
        mae=mae,
        features=demand_features,
        params=parameters,
        mape=mape,
        shap_importances=shap_importances,
        experiment="demand",
        model=model,
    )
    log_shap_plot(shap_values=shap_values, sample=shap_sample)
    save_model(model=model, path=model_path)
    save_to_database(
        mae=mae, mape=mape, parameters=parameters, model_name=model_name, engine=engine
    )

    if conformal_margin_path is not None:
        residuals = np.abs(y_test.values - model.predict(X_test))
        margin = float(np.quantile(residuals, 0.80))
        ensure_parent(conformal_margin_path)
        np.save(conformal_margin_path, np.array([margin]))
        logger.info(
            f"Conformal margin (80%): {margin:.2f} — saved to {conformal_margin_path}"
        )


def run_training_pipeline() -> None:
    demand_raw = load_data(DEMAND_DATA)
    demand = add_lag_features(demand_raw)
    process_ab_test_version(
        demand=demand,
        demand_features=DEMAND_FEATURES_V2,
        demand_target=DEMAND_TARGET,
        model_path=MODEL_PATH_A,
        model_name=ModelName.DEMAND_XGB.value,
        min_trips=0,
        conformal_margin_path=CONFORMAL_MARGIN_PATH,
    )
    run_lstm_pipeline()


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
    run_mlflow()
