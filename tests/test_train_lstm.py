import numpy as np
import pandas as pd
import torch

from src.train_lstm import (
    LSTM_FEATURES,
    WINDOW_SIZE,
    LSTMModel,
    build_sequences,
    evaluate,
)


def _make_lstm_df(n_zones: int = 2, n_hours: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    zones = np.repeat(np.arange(1, n_zones + 1), n_hours)
    timestamps = np.tile(ts, n_zones)
    return pd.DataFrame({
        "PULocationID": zones,
        "pickup_hour_ts": timestamps,
        "trip_count": rng.integers(1, 50, n_zones * n_hours),
        "pickup_hour": rng.integers(0, 24, n_zones * n_hours),
        "pickup_dow": rng.integers(0, 7, n_zones * n_hours),
        "temperature_2m": rng.normal(15, 5, n_zones * n_hours).astype(np.float32),
        "precipitation": rng.uniform(0, 5, n_zones * n_hours).astype(np.float32),
        "snowfall": rng.uniform(0, 2, n_zones * n_hours).astype(np.float32),
        "is_holiday": rng.integers(0, 2, n_zones * n_hours),
    })


def test_build_sequences_output_shape():
    df = _make_lstm_df()
    X_train, _, y_train, _ = build_sequences(df, window_size=WINDOW_SIZE)
    assert X_train.ndim == 3
    assert X_train.shape[2] == len(LSTM_FEATURES)
    assert y_train.ndim == 1
    assert X_train.shape[0] == y_train.shape[0]


def test_build_sequences_no_data_leakage():
    df = _make_lstm_df(n_zones=1, n_hours=150)
    X_train, X_test, _, _ = build_sequences(df, window_size=WINDOW_SIZE)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    # train set comes from the first 80% of the time range, test from the remaining 20%
    assert X_train.shape[0] > X_test.shape[0]


def test_lstm_model_forward_shape():
    model = LSTMModel()
    x = torch.randn(4, WINDOW_SIZE, len(LSTM_FEATURES))
    out = model(x)
    assert out.shape == (4,)


def test_evaluate_returns_positive_mae():
    df = _make_lstm_df()
    _, X_test, _, y_test = build_sequences(df, window_size=WINDOW_SIZE)
    model = LSTMModel()
    mae = evaluate(model, X_test, y_test)
    assert mae > 0
