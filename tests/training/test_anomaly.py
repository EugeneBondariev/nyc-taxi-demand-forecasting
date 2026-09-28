
import numpy as np
import pandas as pd

from src.training.anomaly import CONTAMINATION, FEATURES


def _make_demand(n=500):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "PULocationID": rng.integers(1, 10, n),
        "pickup_hour": rng.integers(0, 24, n),
        "pickup_dow": rng.integers(0, 7, n),
        "trip_count": rng.integers(1, 100, n),
        "temperature_2m": rng.uniform(-5, 35, n),
        "precipitation": rng.uniform(0, 10, n),
        "snowfall": rng.uniform(0, 5, n),
        "is_holiday": rng.integers(0, 2, n),
        "pickup_hour_ts": pd.date_range("2024-01-01", periods=n, freq="h"),
        "pickup_week": rng.integers(1, 53, n),
        "pickup_is_weekend": rng.integers(0, 2, n).astype(bool),
    })
    return df


def test_features_present_in_synthetic_data():
    df = _make_demand()
    assert all(f in df.columns for f in FEATURES)


def test_contamination_default():
    assert 0 < CONTAMINATION < 1


def test_anomaly_detection_flags_expected_rate():
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df = _make_demand(n=1000)
    df = df.dropna(subset=FEATURES)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    iso = IsolationForest(contamination=CONTAMINATION, random_state=42)
    df["is_anomaly"] = iso.fit_predict(X) == -1

    rate = df["is_anomaly"].mean()
    assert abs(rate - CONTAMINATION) < 0.02


def test_anomaly_detection_no_nan_features():
    df = _make_demand()
    assert df[FEATURES].isna().sum().sum() == 0
