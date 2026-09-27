import numpy as np
import pandas as pd
import pytest

from src.baseline import HistoricalMeanBaseline


def _make_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "PULocationID": rng.integers(1, 10, n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(0, 7, n),
        "trip_count": rng.integers(1, 100, n).astype(float),
    })


def test_predict_shape():
    df = _make_df()
    preds = HistoricalMeanBaseline().fit(df).predict(df)
    assert preds.shape == (len(df),)


def test_predict_all_positive():
    df = _make_df()
    preds = HistoricalMeanBaseline().fit(df).predict(df)
    assert (preds > 0).all()


def test_unknown_key_falls_back_to_global_mean():
    df = _make_df()
    baseline = HistoricalMeanBaseline().fit(df)
    unknown = pd.DataFrame({
        "PULocationID": [9999],
        "pickup_hour": [0],
        "pickup_dow": [0],
        "trip_count": [50.0],
    })
    preds = baseline.predict(unknown)
    assert preds[0] == pytest.approx(baseline._global_mean)


def test_beats_trivial_constant():
    rng = np.random.default_rng(1)
    n = 2000
    # Zone 1 always has ~10 trips, zone 2 always has ~80 trips
    zones = np.repeat([1, 2], n // 2)
    counts = np.where(zones == 1, rng.normal(10, 2, n), rng.normal(80, 5, n))
    df = pd.DataFrame({
        "PULocationID": zones,
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(0, 7, n),
        "trip_count": np.clip(counts, 0, None),
    })
    train, test = df.iloc[:1600], df.iloc[1600:]
    baseline = HistoricalMeanBaseline().fit(train)
    preds = baseline.predict(test)
    from sklearn.metrics import mean_absolute_error
    mae_baseline = mean_absolute_error(test["trip_count"], preds)
    mae_constant = mean_absolute_error(test["trip_count"], np.full(len(test), train["trip_count"].mean()))
    assert mae_baseline < mae_constant
