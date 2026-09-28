import os
from pathlib import Path

# Must be set before any src imports so load_dotenv() doesn't override them
os.environ["DATABASE_URL"] = "sqlite:///./dbs/test.db"
os.environ["MODEL_PATH_A"] = str(Path(__file__).parent / "models" / "demand" / "xgb_demand_a.joblib")
os.environ["MODEL_PATH_LSTM"] = str(Path(__file__).parent / "models" / "demand" / "lstm_demand.pt")

import joblib
import pandas as pd
import pytest
import torch
from xgboost import XGBRegressor

from src.core.config import MODEL_PATH_A, MODEL_PATH_LSTM, ModelName
from src.core.database import Base
from src.core.utils import save_to_database
from src.training.train import engine
from src.training.train_lstm import LSTMModel


@pytest.fixture(scope="session", autouse=True)
def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@pytest.fixture(scope="session", autouse=True)
def test_models():
    MODEL_PATH_A.parent.mkdir(exist_ok=True)

    model_a = XGBRegressor(n_estimators=1, random_state=42)
    X = pd.DataFrame({
        "PULocationID": [1, 2, 132],
        "pickup_hour": [0, 12, 18],
        "pickup_dow": [0, 3, 5],
        "pickup_week": [1, 26, 52],
        "is_holiday": [0, 0, 1],
        "snowfall": [0.0, 0.0, 0.5],
        "lag_24h": [10.0, 20.0, 15.0],
        "lag_168h": [12.0, 18.0, 14.0],
        "is_airport": [1, 0, 1],
        "is_nyc_event": [0, 0, 0],
    })
    y = pd.Series([10.0, 20.0, 30.0])
    model_a.fit(X, y)
    joblib.dump(model_a, MODEL_PATH_A)
    save_to_database(mae=5, mape=0.1, parameters={"n_estimators": 1, "learning_rate": 0.3}, model_name=ModelName.DEMAND_XGB.value, engine=engine)

    MODEL_PATH_LSTM.parent.mkdir(exist_ok=True)
    torch.save(LSTMModel().state_dict(), MODEL_PATH_LSTM)
    save_to_database(mae=6, mape=0.1, parameters={}, model_name=ModelName.DEMAND_LSTM.value, engine=engine)

    yield
    MODEL_PATH_A.unlink(missing_ok=True)
    MODEL_PATH_LSTM.unlink(missing_ok=True)
