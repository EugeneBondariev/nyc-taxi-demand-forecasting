import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.core.utils import (
    _compute_psi,
    compute_sample_weights,
    detect_feature_drift,
    save_feature_baseline,
    time_split_df,
)


def _make_weight_inputs(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    zone_ids = rng.choice([1, 2, 3], size=n)
    counts = rng.integers(1, 300, size=n)
    X = pd.DataFrame({"PULocationID": zone_ids})
    y = pd.Series(counts, dtype=float)
    return X, y


def _make_time_df(n: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"ts": ts, "value": range(n)})


def _make_baseline_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "continuous": rng.uniform(0, 100, n),
        "binary": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
        "categorical": rng.choice([1, 2, 3, 4, 5, 6, 7], size=n),
    })


class TestComputePsi:
    def test_identical_distribution_is_near_zero(self):
        breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        actual = np.linspace(0, 10, 1000)
        assert _compute_psi(breakpoints, actual) < 0.05

    def test_shifted_distribution_exceeds_threshold(self):
        breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        actual = np.linspace(7, 10, 1000)  # concentrated in top bins
        assert _compute_psi(breakpoints, actual) > 0.2

    def test_empty_array_does_not_crash(self):
        breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = _compute_psi(breakpoints, np.array([]))
        assert isinstance(result, float)


class TestComputeSampleWeights:
    def test_length_matches_input(self):
        X, y = _make_weight_inputs()
        weights = compute_sample_weights(X, y)
        assert len(weights) == len(y)

    def test_mean_is_one(self):
        X, y = _make_weight_inputs()
        weights = compute_sample_weights(X, y)
        assert abs(weights.mean() - 1.0) < 0.01

    def test_peak_rows_are_heavier(self):
        X, y = _make_weight_inputs()
        weights = compute_sample_weights(X, y)
        q80 = float(y.quantile(0.80))
        peak_mean = weights[y.values >= q80].mean()
        rest_mean = weights[y.values < q80].mean()
        assert peak_mean > rest_mean


class TestTimeSplitDf:
    def test_ratio(self):
        df = _make_time_df(100)
        train, test = time_split_df(df, "ts", ratio=0.8)
        assert len(train) + len(test) == 100
        assert abs(len(train) / 100 - 0.8) < 0.05

    def test_no_overlap(self):
        df = _make_time_df(100)
        train, test = time_split_df(df, "ts")
        assert train["ts"].max() < test["ts"].min()

    def test_missing_column_raises(self):
        df = _make_time_df(100)
        with pytest.raises(ValueError, match="timestamp_col"):
            time_split_df(df, "nonexistent")


class TestFeatureBaseline:
    def test_save_excludes_low_cardinality(self, tmp_path):
        df = _make_baseline_df()
        stats_path = tmp_path / "feature_stats.json"
        with patch("src.core.utils.FEATURE_STATS_PATH", stats_path):
            save_feature_baseline(df=df, features=["continuous", "binary", "categorical"])
        baseline = json.loads(stats_path.read_text())
        assert "continuous" in baseline
        assert "binary" not in baseline      # 95% zeros → bins collapse
        assert "categorical" not in baseline  # 7 values → bins collapse

    def test_save_excludes_categorical_param(self, tmp_path):
        df = _make_baseline_df()
        stats_path = tmp_path / "feature_stats.json"
        with patch("src.core.utils.FEATURE_STATS_PATH", stats_path):
            save_feature_baseline(df, ["continuous"], categorical_features={"continuous"})
        baseline = json.loads(stats_path.read_text())
        assert "continuous" not in baseline

    def test_detect_no_drift(self, tmp_path):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"continuous": rng.uniform(0, 100, 500)})
        stats_path = tmp_path / "feature_stats.json"
        with patch("src.core.utils.FEATURE_STATS_PATH", stats_path):
            save_feature_baseline(df=df, features=["continuous"])
            drifted = detect_feature_drift(df)
        assert drifted == []

    def test_detect_detects_shift(self, tmp_path):
        rng = np.random.default_rng(2)
        baseline_df = pd.DataFrame({"continuous": rng.uniform(0, 100, 500)})
        shifted_df = pd.DataFrame({"continuous": rng.uniform(80, 100, 500)})
        stats_path = tmp_path / "feature_stats.json"
        with patch("src.core.utils.FEATURE_STATS_PATH", stats_path):
            save_feature_baseline(df=baseline_df, features=["continuous"])
            drifted = detect_feature_drift(shifted_df)
        assert "continuous" in drifted

    def test_detect_no_baseline_returns_empty(self, tmp_path):
        df = pd.DataFrame({"continuous": [1.0, 2.0, 3.0]})
        stats_path = tmp_path / "feature_stats.json"
        with patch("src.core.utils.FEATURE_STATS_PATH", stats_path):
            drifted = detect_feature_drift(df)
        assert drifted == []
