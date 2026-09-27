import concurrent.futures
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app

_V1_PAYLOAD = {"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"}


def test_predict_latency_under_500ms():
    with TestClient(app) as client, patch("src.api.random.random", return_value=0.9):
        start = time.perf_counter()
        response = client.post("/v1/predict", json=_V1_PAYLOAD)
        elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 0.5


def test_predict_concurrent_no_errors():
    n_requests = 20

    with TestClient(app) as client, patch("src.api.random.random", return_value=0.9):
        def _call(_):
            return client.post("/v1/predict", json=_V1_PAYLOAD).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            statuses = list(executor.map(_call, range(n_requests)))

    assert all(s == 200 for s in statuses)
