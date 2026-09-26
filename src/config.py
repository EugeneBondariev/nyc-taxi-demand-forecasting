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
MODEL_PATH_LSTM = Path(os.getenv("MODEL_PATH_LSTM", str(ROOT / "models" / "lstm_demand.pt")))
MODEL_PATH_FARE_A = ROOT / "models" / "fare_linear.joblib"
MODEL_PATH_FARE_B = ROOT / "models" / "fare_xgb.joblib"

RAW_DATA_FOLDER = ROOT / "data" / "raw"
TAXI_DATA_FOLDER = RAW_DATA_FOLDER / "taxi"
WEATHER_DATA_FOLDER = RAW_DATA_FOLDER / "weather"
DEMAND_DATA = ROOT / "data" / "processed" / "demand.parquet"

DEMAND_FEATURES_A = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
DEMAND_TARGET = "trip_count"
FARE_AMOUNT_FEATURES_A = ["trip_distance", "Airport_fee", "extra", "trip_duration"]
FARE_AMOUNT_FEATURES_B = [*FARE_AMOUNT_FEATURES_A, "PULocationID", "DOLocationID"]
FARE_TARGET = "fare_amount"

MAE_THRESHOLD = 7
MAPE_THRESHOLD = 0.15

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class ModelName(Enum):
    DEMAND_XGB = "demand_xgboost"
    DEMAND_LSTM = "demand_lstm"
    FARE_LINEAR = "fare_linear"
    FARE_XGB = "fare_xgboost"
