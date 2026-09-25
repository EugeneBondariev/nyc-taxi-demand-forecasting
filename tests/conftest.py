import os
from pathlib import Path

# Must be set before any src imports so load_dotenv() doesn't override it
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["MODEL_PATH_A"] = str(
    Path(__file__).parent / "models" / "xgb_demand_a.joblib"
)
os.environ["MODEL_PATH_B"] = str(
    Path(__file__).parent / "models" / "xgb_demand_b.joblib"
)

import joblib
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.config import MODEL_PATH_A, MODEL_PATH_B, ModelName
from src.database import Base
from src.train import engine, save_to_database


@pytest.fixture(scope="session", autouse=True)
def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@pytest.fixture(scope="session", autouse=True)
def test_models():
    for path, extra in [(MODEL_PATH_A, False), (MODEL_PATH_B, True)]:
        path.parent.mkdir(exist_ok=True)
        model = XGBRegressor(n_estimators=1, random_state=42)
        X = pd.DataFrame(
            {
                "PULocationID": [1, 2, 3],
                "pickup_hour": [0, 12, 18],
                "pickup_dow": [0, 3, 5],
                "pickup_week": [1, 26, 52],
            }
        )
        if extra:
            X["pickup_is_weekend"] = [False, False, True]
        y = pd.Series([10.0, 20.0, 30.0])
        model.fit(X, y)
        joblib.dump(model, path)
        save_to_database(
            mae=5,
            mape=1.1,
            parameters={"n_estimators": 300, "learning_rate": 0.05},
            model_name=ModelName.DEMAND_LSTM.value if extra else ModelName.DEMAND_XGB.value,
        )
    yield
    MODEL_PATH_A.unlink(missing_ok=True)
    MODEL_PATH_B.unlink(missing_ok=True)
