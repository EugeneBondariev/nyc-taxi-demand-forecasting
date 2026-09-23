import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).parent.parent
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nyc_taxi"
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "xgb_demand.joblib")))
RAW_DATA_FOLDER = ROOT / "data" / "raw"
DEMAND_DATA = ROOT / "data" / "processed" / "demand.parquet"

DEMAND_FEATURES = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
DEMAND_TARGET = "trip_count"

MAE_THRESHOLD = 12.0
MAPE_THRESHOLD = 0.15
