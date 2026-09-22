import os
import mlflow
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from .utils import predict_and_evaluate, get_features_and_target
from .config import ROOT, DEMAND_DATA, MODEL_PATH, DEMAND_FEATURES, DEMAND_TARGET

os.environ["MLFLOW_ARTIFACT_ROOT"] = str(ROOT / "mlflow_artifacts")


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DEMAND_DATA)


def split_data(
    demand: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    min_ts = demand["pickup_hour_ts"].min()
    max_ts = demand["pickup_hour_ts"].max()
    cutoff = min_ts + (max_ts - min_ts) * 0.8

    train = demand[demand["pickup_hour_ts"] < cutoff]
    test = demand[demand["pickup_hour_ts"] >= cutoff]

    X_train, y_train = get_features_and_target(train, DEMAND_FEATURES, DEMAND_TARGET)
    X_test, y_test = get_features_and_target(test, DEMAND_FEATURES, DEMAND_TARGET)

    print("Data split complete")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300, random_state=42, max_depth=9, learning_rate=0.05
    )
    model.fit(X_train, y_train)
    print("The model training complete")

    return model


def tune_model(X_train, y_train):
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [6, 9],
        "learning_rate": [0.05],
        "min_child_weight": [1],
    }
    model = XGBRegressor(random_state=42)
    search = GridSearchCV(
        model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)
    return search.best_estimator_


def save_model(model: XGBRegressor, path: Path) -> None:
    joblib.dump(model, path)
    print(f"Model saved to {MODEL_PATH}")


def run_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs()
    print(runs[["metrics.mae", "params.n_estimators", "params.learning_rate"]])


def run_training_pipeline() -> None:
    demand = load_data()
    X_train, X_test, y_train, y_test = split_data(demand)
    model = train_model(X_train, y_train)
    mae, mape = predict_and_evaluate(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    run_training_pipeline()
    run_mlflow()
