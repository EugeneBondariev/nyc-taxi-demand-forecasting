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
MODEL_FOLDER_B = str(ROOT / "models" / "xgb_demand_b.joblib")
MODEL_PATH_A = Path(os.getenv("MODEL_PATH_A", MODEL_FOLDER_A))
MODEL_PATH_B = Path(os.getenv("MODEL_PATH_B", MODEL_FOLDER_B))

RAW_DATA_FOLDER = ROOT / "data" / "raw"
DEMAND_DATA = ROOT / "data" / "processed" / "demand.parquet"

DEMAND_FEATURES_A = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
DEMAND_FEATURES_B = [*DEMAND_FEATURES_A, "pickup_is_weekend"]
DEMAND_TARGET = "trip_count"

MAE_THRESHOLD = 12.0
MAPE_THRESHOLD = 0.15


class ModelName(Enum):
    A = "DEMAND_FEATURES_A"
    B = "DEMAND_FEATURES_B"
