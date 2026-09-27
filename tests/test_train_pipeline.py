from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api import app
from src.config import (
    AIRPORT_ZONES,
    DEMAND_FEATURES_A,
    DEMAND_FEATURES_V2,
    DEMAND_TARGET,
    ModelName,
)
from src.database import Prediction
from src.train import engine, process_ab_test_version


def _make_demand(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    zones = rng.integers(1, 10, n)
    return pd.DataFrame({
        "pickup_hour_ts": ts,
        "PULocationID": zones,
        "pickup_hour": ts.hour,
        "pickup_dow": ts.dayofweek,
        "pickup_week": ts.isocalendar().week.astype(int),
        "is_holiday": np.zeros(n, dtype=int),
        "snowfall": rng.uniform(0, 2, n),
        "lag_24h": rng.integers(1, 100, n).astype(float),
        "lag_168h": rng.integers(1, 100, n).astype(float),
        "is_airport": pd.Series(zones).isin(AIRPORT_ZONES).astype(int).to_numpy(),
        "trip_count": rng.integers(1, 100, n),
    })


def test_process_ab_test_version_saves_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    with (
        patch("src.train.log_to_mlflow"),
        patch("src.train.save_to_database"),
        patch("src.train.log_shap_plot"),
        patch("src.train.save_feature_baseline"),
    ):
        process_ab_test_version(
            _make_demand(), DEMAND_FEATURES_A, DEMAND_TARGET,
            model_path, ModelName.DEMAND_XGB.value, min_trips=0,
        )
    assert model_path.exists()


def test_process_ab_test_version_mae_is_positive(tmp_path):
    model_path = tmp_path / "model.joblib"
    captured = {}

    def fake_log(_name, mae, *_args, **_kwargs):
        captured["mae"] = mae

    with (
        patch("src.train.log_to_mlflow", side_effect=fake_log),
        patch("src.train.save_to_database"),
        patch("src.train.log_shap_plot"),
        patch("src.train.save_feature_baseline"),
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


def test_explain_v1_returns_feature_contributions():
    with TestClient(app) as client:
        response = client.post(
            "/v1/explain",
            json={"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"},
        )
    assert response.status_code == 200
    body = response.json()
    assert "feature_contributions" in body
    assert "PULocationID" in body["feature_contributions"]
    assert len(body["feature_contributions"]) == len(DEMAND_FEATURES_V2)
