import pandas as pd
import pytest

from src.core.config import (
    CONFORMAL_MARGIN_PATH,
    DEMAND_DATA,
    DEMAND_FEATURES_A,
    DEMAND_TARGET,
    FEATURE_STATS_PATH,
    MODEL_PATH_A,
    MODEL_PATH_FARE_A,
    MODEL_PATH_FARE_B,
    MODEL_PATH_LSTM,
    ROOT,
)
from src.training.train_lstm import LSTM_FEATURES


class TestRoot:
    def test_points_to_project_root(self):
        assert (ROOT / "src").is_dir()
        assert (ROOT / "tests").is_dir()
        assert (ROOT / "requirements.txt").exists()


requires_data = pytest.mark.skipif(not DEMAND_DATA.exists(), reason="demand.parquet not available (CI)")


class TestDataPaths:
    @requires_data
    def test_demand_data_exists(self):
        assert DEMAND_DATA.exists(), f"demand.parquet not found at {DEMAND_DATA}"

    @requires_data
    def test_demand_data_has_xgb_columns(self):
        # lag_24h / lag_168h are computed by add_lag_features() at train time, not stored
        base_cols = set(DEMAND_FEATURES_A + [DEMAND_TARGET, "pickup_hour_ts"])
        df = pd.read_parquet(DEMAND_DATA)
        missing = base_cols - set(df.columns)
        assert not missing, f"demand.parquet missing columns: {missing}"

    @requires_data
    def test_demand_data_has_lstm_columns(self):
        required = set(LSTM_FEATURES + ["PULocationID", "pickup_hour_ts"])
        df = pd.read_parquet(DEMAND_DATA)
        missing = required - set(df.columns)
        assert not missing, f"demand.parquet missing LSTM columns: {missing}"

    @requires_data
    def test_demand_data_not_empty(self):
        df = pd.read_parquet(DEMAND_DATA)
        assert len(df) > 0


class TestModelPaths:
    def test_model_paths_under_root(self):
        for path in (MODEL_PATH_A, MODEL_PATH_LSTM, MODEL_PATH_FARE_A, MODEL_PATH_FARE_B,
                     CONFORMAL_MARGIN_PATH, FEATURE_STATS_PATH):
            assert path.is_relative_to(ROOT), f"{path} is not under project root"

    def test_model_dirs_are_demand_or_fare(self):
        for path in (MODEL_PATH_A, MODEL_PATH_LSTM, CONFORMAL_MARGIN_PATH, FEATURE_STATS_PATH):
            assert "demand" in path.parts, f"{path} should be under models/demand/"
        for path in (MODEL_PATH_FARE_A, MODEL_PATH_FARE_B):
            assert "fare" in path.parts, f"{path} should be under models/fare/"
