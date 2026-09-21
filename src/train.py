import os
import mlflow
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
DEMAND_DATA = ROOT / "data" / "demand.parquet"
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

    features = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
    X_train = train[features]
    y_train = train["trip_count"]
    X_test = test[features]
    y_test = test["trip_count"]

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


if __name__ == "__main__":
    demand = load_data()
    X_train, X_test, y_train, y_test = split_data(demand)
    print(X_train.shape, X_test.shape)

    model = train_model(X_train, y_train)
    # print(model.get_params())
    predictions = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, predictions)
    print(f"MAPE: {mape:.1f} trips")

    model_path = ROOT / "models" / "xgb_demand.joblib"
    model_path.parent.mkdir(exist_ok=True)
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    demand = pd.read_parquet("data/demand.parquet")
    print(demand["trip_count"].describe())

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs()
    print(runs[["metrics.mae", "params.n_estimators", "params.learning_rate"]])
