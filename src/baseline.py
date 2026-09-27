import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from .config import DEMAND_DATA, DEMAND_TARGET
from .logger import setup_logging
from .utils import load_data, log_to_mlflow, time_split_df

logger = logging.getLogger(__name__)


class HistoricalMeanBaseline:
    """Predict mean trip count per (zone, hour, day_of_week) from training data.

    This is the minimum bar the model must clear — if XGBoost doesn't beat it,
    the added complexity delivers no value.
    """

    def __init__(self) -> None:
        self._lookup: dict[tuple[int, int, int], float] = {}
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> "HistoricalMeanBaseline":
        self._global_mean = float(df[DEMAND_TARGET].mean())
        self._lookup = (
            df.groupby(["PULocationID", "pickup_hour", "pickup_dow"])[DEMAND_TARGET]
            .mean()
            .to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        keys = zip(df["PULocationID"], df["pickup_hour"], df["pickup_dow"])
        return np.array([self._lookup.get(k, self._global_mean) for k in keys])


def evaluate_baseline(demand: pd.DataFrame) -> tuple[float, float]:
    train, test = time_split_df(demand, "pickup_hour_ts")
    baseline = HistoricalMeanBaseline().fit(train)
    preds = baseline.predict(test)
    y_test = test[DEMAND_TARGET].values

    mae = round(float(mean_absolute_error(y_test, preds)), 2)
    mape = round(float(mean_absolute_percentage_error(y_test, preds)), 2)

    logger.info(f"Baseline  MAE={mae:.2f}  MAPE={mape:.1%}")
    log_to_mlflow(
        "demand_baseline",
        mae,
        ["PULocationID", "pickup_hour", "pickup_dow"],
        {},
        mape=mape,
        experiment="demand",
    )
    return mae, mape


if __name__ == "__main__":
    setup_logging()
    evaluate_baseline(load_data(DEMAND_DATA))
