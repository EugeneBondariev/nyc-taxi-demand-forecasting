import pandas as pd
import calendar
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor


def predict_and_evaluate(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float]:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_pred=predictions, y_true=y_test)
    mape = mean_absolute_percentage_error(y_pred=predictions, y_true=y_test)

    print(f"MAE: {mae:.1f} trips")
    print(f"MAPE: {mape:.1f} trips")

    return mae, mape


def is_valid_file(file: Path, year: int, month: int) -> bool:
    try:
        df = pd.read_parquet(file, columns=["tpep_pickup_datetime"])
        last_day = calendar.monthrange(year, month)[1]
        expected_last = pd.Timestamp(year=year, month=month, day=last_day)
        return df["tpep_pickup_datetime"].max().normalize() >= expected_last
    except Exception:
        return False


def get_features_and_target(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features]
    y = df[target]

    return X, y
