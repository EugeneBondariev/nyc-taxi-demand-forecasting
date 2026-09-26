import os
from pathlib import Path

# Must be set before any src imports so load_dotenv() doesn't override them
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["MODEL_PATH_A"] = str(Path(__file__).parent / "models" / "xgb_demand_a.joblib")
os.environ["MODEL_PATH_LSTM"] = str(Path(__file__).parent / "models" / "lstm_demand.pt")

import joblib
import pandas as pd
import pytest
import torch
from xgboost import XGBRegressor

from src.config import MODEL_PATH_A, MODEL_PATH_LSTM, ModelName
from src.database import Base
from src.train import engine
from src.train_lstm import LSTMModel
from src.utils import save_to_database


@pytest.fixture(scope="session", autouse=True)
def reset_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@pytest.fixture(scope="session", autouse=True)
def test_models():
    MODEL_PATH_A.parent.mkdir(exist_ok=True)

    model_a = XGBRegressor(n_estimators=1, random_state=42)
    X = pd.DataFrame({
        "PULocationID": [1, 2, 3],
        "pickup_hour": [0, 12, 18],
        "pickup_dow": [0, 3, 5],
        "pickup_week": [1, 26, 52],
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
