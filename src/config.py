import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).parent.parent
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nyc_taxi"
)

MODEL_FOLDER_A = str(ROOT / "models" / "xgb_demand_a.joblib")
MODEL_PATH_A = Path(os.getenv("MODEL_PATH_A", MODEL_FOLDER_A))
MODEL_PATH_LSTM = ROOT / "models" / "lstm_demand.pt"

RAW_DATA_FOLDER = ROOT / "data" / "raw"
TAXI_DATA_FOLDER = RAW_DATA_FOLDER / "taxi"
WEATHER_DATA_FOLDER = RAW_DATA_FOLDER / "weather"
DEMAND_DATA = ROOT / "data" / "processed" / "demand.parquet"

DEMAND_FEATURES_A = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
DEMAND_TARGET = "trip_count"

MAE_THRESHOLD = 7
MAPE_THRESHOLD = 0.15


class ModelName(Enum):
    A = "xgboost"
    B = "lstm"
