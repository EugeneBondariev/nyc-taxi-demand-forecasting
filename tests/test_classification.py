import numpy as np
import pandas as pd

from src.train_classification import FEATURES, HIGH_DEMAND_QUANTILE, make_target


def _make_demand(n=300):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "PULocationID": rng.integers(1, 10, n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(0, 7, n),
        "pickup_week": rng.integers(1, 53, n),
        "temperature_2m": rng.uniform(-5, 35, n),
        "precipitation": rng.uniform(0, 10, n),
        "snowfall": rng.uniform(0, 5, n),
        "is_holiday": rng.integers(0, 2, n),
        "trip_count": rng.integers(1, 200, n),
        "pickup_hour_ts": pd.date_range("2024-01-01", periods=n, freq="h"),
    })


def test_make_target_binary():
    y = make_target(_make_demand())
    assert set(y.unique()).issubset({0, 1})


def test_make_target_correct_high_demand_fraction():
    demand = _make_demand(n=10_000)
    y = make_target(demand)
    expected = 1 - HIGH_DEMAND_QUANTILE
    assert abs(y.mean() - expected) < 0.02


def test_make_target_length_matches():
    demand = _make_demand()
    assert len(make_target(demand)) == len(demand)


def test_features_present_in_synthetic_data():
    assert all(f in _make_demand().columns for f in FEATURES)
