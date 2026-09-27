from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api import app
from src.config import DEMAND_FEATURES_A, DEMAND_TARGET, ModelName
from src.database import Prediction
from src.train import engine, process_ab_test_version


def _make_demand(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "pickup_hour_ts": ts,
        "PULocationID": rng.integers(1, 10, n),
        "pickup_hour": ts.hour,
        "pickup_dow": ts.dayofweek,
        "pickup_week": ts.isocalendar().week.astype(int),
        "is_holiday": np.zeros(n, dtype=int),
        "trip_count": rng.integers(1, 100, n),
    })


def test_process_ab_test_version_saves_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    with (
        patch("src.train.log_to_mlflow"),
        patch("src.train.save_to_database"),
        patch("src.train.log_shap_plot"),
    ):
        process_ab_test_version(
            _make_demand(), DEMAND_FEATURES_A, DEMAND_TARGET,
            model_path, ModelName.DEMAND_XGB.value, min_trips=0,
        )
    assert model_path.exists()


def test_process_ab_test_version_mae_is_positive(tmp_path):
    model_path = tmp_path / "model.joblib"
    captured = {}

    def fake_log(name, mae, *args, **kwargs):
        captured["mae"] = mae

    with (
        patch("src.train.log_to_mlflow", side_effect=fake_log),
        patch("src.train.save_to_database"),
        patch("src.train.log_shap_plot"),
    ):
        process_ab_test_version(
            _make_demand(), DEMAND_FEATURES_A, DEMAND_TARGET,
            model_path, ModelName.DEMAND_XGB.value, min_trips=0,
        )
    assert captured["mae"] > 0


# ── Integration tests ────────────────────────────────────────────────────────


def test_predict_v1_returns_200():
    with TestClient(app) as client, patch("src.api.random.random", return_value=0.9):
        response = client.post(
            "/v1/predict",
            json={"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"},
        )
    assert response.status_code == 200
    assert response.json()["predicted_trips"] >= 0


def test_predict_v1_has_correlation_id_header():
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            json={"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"},
        )
    assert "x-correlation-id" in response.headers


def test_predict_writes_to_database():
    with Session(engine) as session:
        before = session.execute(select(func.count()).select_from(Prediction)).scalar()

    with TestClient(app) as client, patch("src.api.random.random", return_value=0.9):
        client.post(
            "/v1/predict",
            json={"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"},
        )

    with Session(engine) as session:
        after = session.execute(select(func.count()).select_from(Prediction)).scalar()

    assert after > before


def test_predict_legacy_route_still_works():
    with TestClient(app) as client, patch("src.api.random.random", return_value=0.9):
        response = client.post(
            "/predict",
            json={"zone_id": 161, "hour": 18, "day_of_week": 2, "week": 10},
            )
    assert response.status_code == 200
