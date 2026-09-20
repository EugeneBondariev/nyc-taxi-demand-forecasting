import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).parent.parent
DEMAND_DATA = ROOT / "data" / "demand.parquet"


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DEMAND_DATA)


def split_data(demand: pd.DataFrame) -> list:
    cutoff = pd.Timestamp("2024-01-25")
    train = demand[demand["pickup_hour_ts"] < cutoff]
    test = demand[demand["pickup_hour_ts"] >= cutoff]

    features = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
    X_train = train[features]
    y_train = train["trip_count"]
    X_test = test[features]
    y_test = test["trip_count"]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    demand = load_data()
    X_train, X_test, y_train, y_test = split_data(demand)
    print(X_train.shape, X_test.shape)

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae:.1f} trips")
