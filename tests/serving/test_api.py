from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


class TestPredict:
    def test_returns_valid_response(self):
        with TestClient(app) as client:
            with patch("src.serving.api.random.random", return_value=0.9):  # always route to model A
                response = client.post(
                    "/predict",
                    json={
                        "zone_id": 161,
                        "hour": 18,
                        "day_of_week": 2,
                        "week": 10,
                    },
                )
            assert response.status_code == 200
            data = response.json()
            assert "predicted_trips" in data
            assert data["predicted_trips"] >= 0

    @pytest.mark.parametrize(
        "zone,hour,dow,week",
        [
            (-1, 18, 2, 10),  # negative zone
            (166, 25, 2, 10),  # too big hour
            (166, 18, 8, 10),  # too big dow
            (166, 0, 0, 0),  # too small week
            (None, 18, 2, 10),  # missing value
        ],
    )
    def test_invalid_param(self, zone, hour, dow, week):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "zone_id": zone,
                    "hour": hour,
                    "day_of_week": dow,
                    "week": week,
                },
            )
            assert response.status_code == 422
