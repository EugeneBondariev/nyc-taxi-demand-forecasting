import numpy as np
import pandas as pd

from src.core.config import DEMAND_FEATURES_A, DEMAND_TARGET
from src.core.utils import split_data

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
            "is_holiday": np.zeros(n_rows, dtype=int),
            "trip_count": np.random.randint(1, 100, n_rows),
        }
    )


class TestSplitData:
    def test_correct_cutoff(self):
        X_train, X_test, y_train, y_test = split_data(
            make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
        )
        assert 0.79 <= len(X_train) / N_ROWS <= 0.81
        assert 0.79 <= len(y_train) / N_ROWS <= 0.81
        assert 0.19 <= len(X_test) / N_ROWS <= 0.21
        assert 0.19 <= len(y_test) / N_ROWS <= 0.21

    def test_correct_columns(self):
        X_train, X_test, y_train, y_test = split_data(
            make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
        )
        assert len(X_train.columns) == len(X_test.columns) == len(DEMAND_FEATURES_A)
        assert y_train.name == y_test.name == "trip_count"

    def test_no_leakage(self):
        X_train, X_test, _y_train, _y_test = split_data(
            make_demand_df(), DEMAND_FEATURES_A, DEMAND_TARGET, "pickup_hour_ts"
        )
        assert X_train.index.max() < X_test.index.min()
