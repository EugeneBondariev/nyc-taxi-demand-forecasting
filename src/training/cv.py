import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ..core.utils import get_features_and_target, train_xgboost

logger = logging.getLogger(__name__)


def time_series_splits(
    demand: pd.DataFrame,
    n_splits: int = 5,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Expanding-window time-series CV — each fold trains on more history."""
    min_ts = demand["pickup_hour_ts"].min()
    max_ts = demand["pickup_hour_ts"].max()
    span = (max_ts - min_ts) / (n_splits + 1)

    for i in range(n_splits):
        split_ts = min_ts + span * (i + 1)
        end_ts = min_ts + span * (i + 2)
        train = demand[demand["pickup_hour_ts"] <= split_ts]
        test = demand[
            (demand["pickup_hour_ts"] > split_ts)
            & (demand["pickup_hour_ts"] <= end_ts)
        ]
        if len(train) and len(test):
            yield train, test


def cross_validate(
    demand: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Time-series CV returning MAE mean ± std across folds."""
    maes: list[float] = []
    for fold, (train, test) in enumerate(time_series_splits(demand, n_splits)):
        X_train, y_train = get_features_and_target(train, feature_cols, target_col)
        X_test, y_test = get_features_and_target(test, feature_cols, target_col)
        model, _ = train_xgboost(X_train, y_train)
        mae = float(mean_absolute_error(y_test, model.predict(X_test)))
        maes.append(mae)
        logger.info(f"Fold {fold + 1}/{n_splits}  MAE={mae:.2f}")

    return {
        "mean": round(float(np.mean(maes)), 3),
        "std": round(float(np.std(maes)), 3),
        "min": round(float(np.min(maes)), 3),
        "max": round(float(np.max(maes)), 3),
        "folds": [round(m, 3) for m in maes],
    }


def conformal_interval(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    coverage: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marginal split conformal prediction interval with `coverage` guarantee.

    Calibrate on a held-out set, then apply the same quantile to the test set.
    Expected coverage: at least `coverage` fraction of test actuals fall inside.
    """
    residuals = np.abs(y_cal.values - model.predict(X_cal))
    margin = float(np.quantile(residuals, coverage))
    y_pred = model.predict(X_test)
    return y_pred, y_pred - margin, y_pred + margin


def dispatch_simulation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dispatch_factor: float = 1.1,
) -> dict[str, float]:
    """Simulate dispatching ceil(pred × factor) drivers and measure outcome."""
    dispatched = np.maximum(np.ceil(y_pred * dispatch_factor).astype(int), 1)
    served = np.minimum(dispatched, y_true.astype(int))
    unfulfilled = np.maximum(y_true.astype(int) - dispatched, 0)
    waste = dispatched - served
    return {
        "utilization_rate": round(float((served / dispatched).mean()), 3),
        "unfulfilled_trip_rate": round(float((unfulfilled > 0).mean()), 3),
        "total_unfulfilled": round(float(unfulfilled.sum()), 0),
        "wasted_driver_slots": round(float(waste.sum()), 0),
    }
