import pandas as pd
import numpy as np
from src.utils import split_data
from src.config import DEMAND_FEATURES_A, DEMAND_TARGET

N_ROWS = 366  # days in the year of 2024


def make_demand_df(n_rows=N_ROWS):
    timestamps = pd.date_range("2024-01-01", "2024-12-31", periods=n_rows)
    return pd.DataFrame(
        {
            "pickup_hour_ts": timestamps,
            "PULocationID": np.random.randint(1, 265, n_rows),
            "pickup_hour": timestamps.hour,
            "pickup_dow": timestamps.dayofweek,
            "pickup_week": timestamps.isocalendar().week.astype(int),
            "trip_count": np.random.randint(1, 100, n_rows),
        }
    )


def test_split_data_correct_cutoff():
    X_train, X_test, y_train, y_test = split_data(
        make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
    )
    assert 0.79 <= len(X_train) / N_ROWS <= 0.81
    assert 0.79 <= len(y_train) / N_ROWS <= 0.81
    assert 0.19 <= len(X_test) / N_ROWS <= 0.21
    assert 0.19 <= len(y_test) / N_ROWS <= 0.21


def test_split_data_correct_columns():
    X_train, X_test, y_train, y_test = split_data(
        make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
    )
    assert len(X_train.columns) == len(X_test.columns) == len(DEMAND_FEATURES_A)
    assert y_train.name == y_test.name == "trip_count"


def test_split_data_no_leakage():
    X_train, X_test, y_train, y_test = split_data(
        make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
    )
    assert X_train.index.max() < X_test.index.min()
