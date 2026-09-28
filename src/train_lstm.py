import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DB_URL, DEMAND_DATA, MODEL_PATH_LSTM, ModelName
from .database import init_db
from .logger import setup_logging
from .utils import ensure_parent, load_data, log_to_mlflow, save_to_database

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)

WINDOW_SIZE = 24
LSTM_FEATURES = [
    "trip_count",
    "pickup_hour",
    "pickup_dow",
    "temperature_2m",
    "precipitation",
    "snowfall",
]


def build_sequences(
    demand: pd.DataFrame, window_size: int = WINDOW_SIZE, ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []

    for _, group in demand.groupby("PULocationID"):
        group = group.sort_values("pickup_hour_ts")
        features = group[LSTM_FEATURES].values
        counts = group["trip_count"].values
        cutoff = int(len(counts) * ratio)

        for i in range(cutoff - window_size):
            x_train_list.append(features[i : i + window_size])
            y_train_list.append(counts[i + window_size])

        for i in range(cutoff, len(counts) - window_size):
            x_test_list.append(features[i : i + window_size])
            y_test_list.append(counts[i + window_size])

    X_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = np.array(x_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    return X_train, X_test, y_train, y_test


class LSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(LSTM_FEATURES),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 0.001,
    patience: int = 3,
) -> LSTMModel:
    val_size = max(1, int(len(X_train) * 0.1))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_X_t = torch.tensor(X_val)
    val_y_t = torch.tensor(y_val)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X_t), val_y_t).item()

        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch + 1}/{epochs}  train={train_loss / len(loader):.4f}"
            f"  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (patience={patience})"
                )
                model.load_state_dict(best_state)
                break

    return model


def evaluate(
    model: LSTMModel, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()
    mae = float(np.mean(np.abs(preds - y_test)))
    mape = float(np.mean(np.abs((preds - y_test) / np.where(y_test == 0, 1, y_test))))
    logger.info(f"MAE: {mae:.2f} trips  MAPE: {mape:.1%}")
    return mae, mape


def save_model(model: LSTMModel, path: Path) -> None:
    if path.exists():
        path.replace(path.with_suffix(".prev.pt"))
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def run_training_pipeline() -> None:
    demand = load_data(DEMAND_DATA)
    latest_year = demand["pickup_hour_ts"].dt.year.max()
    demand = demand[demand["pickup_hour_ts"].dt.year == latest_year]
    X_train, X_test, y_train, y_test = build_sequences(demand=demand)
    model = train_model(X_train=X_train, y_train=y_train, epochs=10)
    mae, mape = evaluate(model=model, X_test=X_test, y_test=y_test)
    ensure_parent(MODEL_PATH_LSTM)
    save_model(model, MODEL_PATH_LSTM)
    lstm_params = {
        "max_epochs": 20,
        "batch_size": 512,
        "lr": 0.001,
        "patience": 3,
        "window_size": WINDOW_SIZE,
        "hidden_size": 64,
        "num_layers": 2,
        "early_stopping": True,
    }
    log_to_mlflow(
        model_name=ModelName.DEMAND_LSTM.value,
        mae=mae,
        features=LSTM_FEATURES,
        params=lstm_params,
        mape=mape,
        experiment="demand",
    )
    save_to_database(
        mae=mae,
        mape=mape,
        parameters=lstm_params,
        model_name=ModelName.DEMAND_LSTM.value,
        engine=engine,
    )


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
