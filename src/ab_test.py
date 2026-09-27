import logging
import numpy as np
import pandas as pd
import torch
import joblib
import mlflow
import os
from scipy import stats

from .config import MODEL_PATH_A, MODEL_PATH_LSTM, DEMAND_DATA, MLFLOW_TRACKING_URI, DEMAND_FEATURES_A
from .train_lstm import LSTMModel, LSTM_FEATURES, WINDOW_SIZE
from .utils import load_data
from .logger import setup_logging

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_test_samples(
    demand: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Mirrors build_sequences test split so both models predict identical points.
    Returns XGBoost feature rows, LSTM sequences, and ground-truth trip counts.
    """
    xgb_rows, lstm_seqs, y_vals = [], [], []

    for _, group in demand.groupby("PULocationID"):
        group = group.sort_values("pickup_hour_ts").reset_index(drop=True)
        features = group[LSTM_FEATURES].values.astype(np.float32)
        cutoff = int(len(group) * 0.8)

        for i in range(cutoff, len(group) - WINDOW_SIZE):
            target = group.iloc[i + WINDOW_SIZE]
            lstm_seqs.append(features[i : i + WINDOW_SIZE])
            xgb_rows.append({col: target[col] for col in DEMAND_FEATURES_A})
            y_vals.append(target["trip_count"])

    return (
        pd.DataFrame(xgb_rows),
        np.array(lstm_seqs, dtype=np.float32),
        np.array(y_vals, dtype=np.float32),
    )


def run_ab_test() -> None:
    logger.info("Loading demand data...")
    demand = load_data(DEMAND_DATA)

    logger.info("Building test samples (same split as training)...")
    X_xgb, X_lstm, y_true = get_test_samples(demand)
    logger.info(f"Test samples: {len(y_true):,}")

    model_a = joblib.load(MODEL_PATH_A)
    model_b = LSTMModel()
    model_b.load_state_dict(torch.load(MODEL_PATH_LSTM, weights_only=True))
    model_b.eval()

    logger.info("XGBoost predictions...")
    y_pred_a = np.clip(model_a.predict(X_xgb), 0, None)

    logger.info("LSTM predictions...")
    with torch.no_grad():
        y_pred_b = np.clip(model_b(torch.tensor(X_lstm)).numpy(), 0, None)

    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)
    mae_a = float(errors_a.mean())
    mae_b = float(errors_b.mean())

    # Paired t-test — H0: mean absolute errors are equal
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b)

    # 95% CI on the MAE difference (XGB - LSTM)
    diff = errors_a - errors_b
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)
    )

    winner = None
    if p_value < 0.05:
        winner = "LSTM" if mae_b < mae_a else "XGBoost"

    logger.info("=" * 52)
    logger.info(f"  XGBoost MAE:            {mae_a:.3f} trips")
    logger.info(f"  LSTM MAE:               {mae_b:.3f} trips")
    logger.info(f"  Difference (XGB-LSTM):  {mae_a - mae_b:+.3f} trips")
    logger.info(f"  t-statistic:            {t_stat:.4f}")
    logger.info(f"  p-value:                {p_value:.6f}")
    logger.info(f"  95% CI on difference:   ({ci_low:.3f}, {ci_high:.3f})")
    if winner:
        logger.info(f"  Result: {winner} is significantly better (p < 0.05)")
    else:
        logger.info("  Result: no significant difference (p >= 0.05)")
    logger.info("=" * 52)

    mlflow.set_experiment("demand")
    with mlflow.start_run(run_name="ab_test"):
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        mlflow.log_metric("mae_xgboost", mae_a)
        mlflow.log_metric("mae_lstm", mae_b)
        mlflow.log_metric("mae_diff", mae_a - mae_b)
        mlflow.log_metric("t_statistic", t_stat)
        mlflow.log_metric("p_value", p_value)
        mlflow.log_metric("ci_low", ci_low)
        mlflow.log_metric("ci_high", ci_high)
        mlflow.log_param("n_samples", len(y_true))
        mlflow.log_param("alpha", 0.05)
        mlflow.log_param("test", "paired t-test")
        if winner:
            mlflow.set_tag("winner", winner)


if __name__ == "__main__":
    setup_logging()
    run_ab_test()
