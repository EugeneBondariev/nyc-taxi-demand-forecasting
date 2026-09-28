import logging
import os
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import (
    DB_URL,
    DEMAND_FEATURES_V2,
    DEMAND_TARGET,
    MAE_THRESHOLD,
    MAPE_THRESHOLD,
    MODEL_PATH_A,
    TAXI_DATA_FOLDER,
    WEATHER_DATA_FOLDER,
    ModelName,
)
from ..core.database import ModelVersion, init_db
from ..data.features import (
    build_demand_table,
    download_events_data,
    download_taxi_data,
    download_weather_data,
    load_and_clean_taxi_data,
    load_and_clean_weather_data,
    load_events_data,
)
from ..core.logger import setup_logging
from ..training.train import run_training_pipeline
from ..core.utils import detect_feature_drift, get_features_and_target, predict_and_evaluate

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)


def send_alert(message: str) -> None:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        httpx.post(webhook_url, json={"text": message}, timeout=5)
        logger.info("Slack alert sent")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to send Slack alert — check SLACK_WEBHOOK_URL")


def detect_drift(current_mae: float, model_name: str, lookback: int = 5) -> bool:
    with Session(engine) as session:
        historical = session.execute(
            select(ModelVersion.mae)
            .where(ModelVersion.name == model_name)
            .order_by(ModelVersion.trained_at.desc())
            .limit(lookback + 1)
        ).scalars().all()

    if len(historical) < 3:
        return False

    baseline_mae = float(np.mean(list(historical)[1:]))
    drift_pct = (current_mae - baseline_mae) / baseline_mae
    if drift_pct > 0.10:
        logger.warning(
            f"Drift detected: MAE {current_mae:.2f} is {drift_pct:.1%} above "
            f"baseline {baseline_mae:.2f} (last {lookback} runs)"
        )
        return True
    return False


def compare_predictions(
    mae: float, mape: float, feature_drift: list[str] | None = None
) -> None:
    output_drifted = detect_drift(mae, ModelName.DEMAND_XGB.value)
    input_drifted = bool(feature_drift)
    if mae > MAE_THRESHOLD or mape > MAPE_THRESHOLD or output_drifted or input_drifted:
        reason = []
        if mae > MAE_THRESHOLD:
            reason.append(f"MAE {mae:.2f} > threshold {MAE_THRESHOLD}")
        if mape > MAPE_THRESHOLD:
            reason.append(f"MAPE {mape:.2%} > threshold {MAPE_THRESHOLD:.2%}")
        if output_drifted:
            reason.append("output drift (>10% above rolling baseline)")
        if input_drifted:
            reason.append(f"input feature drift: {', '.join(feature_drift)}")  # type: ignore[arg-type]
        msg = "Retraining triggered: " + "; ".join(reason)
        logger.warning(msg)
        send_alert(f":warning: *uber-api* — {msg}")
        run_training_pipeline()


def monitor_ab_test_version(
    demand: pd.DataFrame,
    demand_features: list[str],
    demand_target: str,
    model_path: Path,
):
    X, y = get_features_and_target(df=demand, features=demand_features, target=demand_target)
    model = joblib.load(model_path)
    mae, mape = predict_and_evaluate(model=model, X_test=X, y_test=y)
    drifted_features = detect_feature_drift(demand)
    compare_predictions(mae=mae, mape=mape, feature_drift=drifted_features)


if __name__ == "__main__":
    from datetime import datetime

    setup_logging()
    now = datetime.now(tz=None)  # noqa: DTZ005 — local time intentional for pipeline scheduling
    year = now.year
    month = now.month

    download_taxi_data(year)
    download_weather_data(year)
    download_events_data(year)

    taxi_df = load_and_clean_taxi_data(TAXI_DATA_FOLDER, year, month)
    weather_df = load_and_clean_weather_data(WEATHER_DATA_FOLDER)
    holiday_dates = load_events_data()

    demand = build_demand_table(taxi_df=taxi_df, weather_df=weather_df, holiday_dates=holiday_dates)
    monitor_ab_test_version(demand=demand, demand_features=DEMAND_FEATURES_V2, demand_target=DEMAND_TARGET, model_path=MODEL_PATH_A)
