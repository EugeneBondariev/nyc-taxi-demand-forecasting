import numpy as np
import pandas as pd
import pytest

from src.training.cv import (
    conformal_interval,
    cross_validate,
    dispatch_simulation,
    time_series_splits,
)


def _make_demand(n_hours: int = 300, n_zones: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    zones = np.repeat(np.arange(1, n_zones + 1), n_hours)
    timestamps = pd.DatetimeIndex(np.tile(ts, n_zones))
    return pd.DataFrame({
        "PULocationID": zones,
        "pickup_hour_ts": timestamps,
        "trip_count": rng.integers(1, 80, n_zones * n_hours).astype(float),
        "pickup_hour": timestamps.hour,
        "pickup_dow": timestamps.dayofweek,
        "pickup_week": timestamps.isocalendar().week.values,
        "is_holiday": np.zeros(n_zones * n_hours, dtype=int),
    })


class TestTimeSeriesSplits:
    def test_count(self):
        df = _make_demand()
        splits = list(time_series_splits(df, n_splits=3))
        assert len(splits) == 3

    def test_no_leakage(self):
        df = _make_demand()
        for train, test in time_series_splits(df, n_splits=3):
            assert train["pickup_hour_ts"].max() < test["pickup_hour_ts"].max()
            assert len(train) > 0
            assert len(test) > 0


class TestCrossValidate:
    def test_returns_expected_keys(self):
        df = _make_demand()
        result = cross_validate(df, ["pickup_hour", "pickup_dow", "PULocationID"], "trip_count", n_splits=2)
        assert set(result.keys()) == {"mean", "std", "min", "max", "folds"}
        assert len(result["folds"]) == 2
        assert result["mean"] > 0

    def test_conformal_interval_coverage(self):
        rng = np.random.default_rng(42)
        n = 500
        X = pd.DataFrame({"x": rng.normal(size=n)})
        y = pd.Series(rng.normal(size=n) * 5 + 20)

        from src.core.utils import train_xgboost
        model, _ = train_xgboost(X, y)

        cal_X, cal_y = X.iloc[:250], y.iloc[:250]
        test_X, test_y = X.iloc[250:], y.iloc[250:]

        _y_pred, lo, hi = conformal_interval(model, cal_X, cal_y, test_X, coverage=0.80)
        empirical_coverage = float(((test_y.values >= lo) & (test_y.values <= hi)).mean())
        assert empirical_coverage >= 0.75  # allow small margin around 80%


class TestDispatchSimulation:
    def test_keys(self):
        y_true = np.array([10.0, 20.0, 5.0, 50.0])
        y_pred = np.array([12.0, 18.0, 6.0, 40.0])
        result = dispatch_simulation(y_true, y_pred)
        assert set(result.keys()) == {"utilization_rate", "unfulfilled_trip_rate",
                                       "total_unfulfilled", "wasted_driver_slots"}
        assert 0 <= result["utilization_rate"] <= 1

    def test_perfect_forecast_high_utilization(self):
        y = np.full(100, 20.0)
        result = dispatch_simulation(y, y, dispatch_factor=1.0)
        assert result["utilization_rate"] == pytest.approx(1.0)
        assert result["total_unfulfilled"] == 0
