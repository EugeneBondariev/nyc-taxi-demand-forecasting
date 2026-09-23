import os

# Must be set before any src imports so load_dotenv() doesn't override it
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

import joblib
import pandas as pd
import pytest
from xgboost import XGBRegressor
from src.config import MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def test_model():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model = XGBRegressor(n_estimators=1, random_state=42)
    X = pd.DataFrame(
        {
            "PULocationID": [1, 2, 3],
            "pickup_hour": [0, 12, 18],
            "pickup_dow": [0, 3, 5],
            "pickup_week": [1, 26, 52],
        }
    )
    y = pd.Series([10.0, 20.0, 30.0])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    yield
    MODEL_PATH.unlink(missing_ok=True)
