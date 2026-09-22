import joblib
from .features import (
    download_data,
    load_and_clean,
    build_demand_table,
)
from .utils import predict_and_evaluate, get_features_and_target
from .config import MODEL_PATH, RAW_DATA_FOLDER, DEMAND_FEATURES, DEMAND_TARGET
from .train import run_training_pipeline


def compare_predictions(mae: float, mape: float) -> None:
    mae_threshold = 12
    mape_threshold = 0.15

    if mae > mae_threshold or mape > mape_threshold:
        print("Trigger retraining")
        run_training_pipeline()


if __name__ == "__main__":
    download_data(2025)
    df = load_and_clean(RAW_DATA_FOLDER, 2025, 1)
    demand = build_demand_table(df)
    X, y = get_features_and_target(demand, DEMAND_FEATURES, DEMAND_TARGET)
    model = joblib.load(MODEL_PATH)
    mae, mape = predict_and_evaluate(model, X, y)
    compare_predictions(mae, mape)
