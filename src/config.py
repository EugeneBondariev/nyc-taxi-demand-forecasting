from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "xgb_demand.joblib"
RAW_DATA_FOLDER = ROOT / "data" / "raw"
DEMAND_DATA = ROOT / "data" / "processed" / "demand.parquet"
DEMAND_FEATURES = ["PULocationID", "pickup_hour", "pickup_dow", "pickup_week"]
DEMAND_TARGET = "trip_count"
MAE_THRESHOLD = 12.0
MAPE_THRESHOLD = 0.15
