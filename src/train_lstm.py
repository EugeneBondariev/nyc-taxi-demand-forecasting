import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sqlalchemy.orm import Session
from .config import DEMAND_DATA, MODEL_PATH_LSTM, ModelName, DB_URL
from .database import init_db, ModelVersion
from .logger import setup_logging

logger = logging.getLogger(__name__)
engine = init_db(DB_URL)

WINDOW_SIZE = 24


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DEMAND_DATA)


def build_sequences(
    demand: pd.DataFrame, window_size: int = WINDOW_SIZE, ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = [], [], [], []

    for _, group in demand.groupby("PULocationID"):
        group = group.sort_values("pickup_hour_ts")
        features = group[["trip_count", "pickup_hour", "pickup_dow", "temperature_2m", "precipitation", "snowfall"]].values
        counts = group["trip_count"].values
        cutoff = int(len(counts) * ratio)

        for i in range(cutoff - window_size):
            X_train.append(features[i : i + window_size])
            y_train.append(counts[i + window_size])

        for i in range(cutoff, len(counts) - window_size):
            X_test.append(features[i : i + window_size])
            y_test.append(counts[i + window_size])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return X_train, X_test, y_train, y_test


class LSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=6,
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
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 0.001,
) -> LSTMModel:
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{len(loader)} loss: {loss.item():.4f}")
        logger.info(f"Epoch {epoch + 1}/{epochs} loss: {total_loss / len(loader):.4f}")

    return model


def evaluate(model: LSTMModel, X_test: np.ndarray, y_test: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()
    mae = float(np.mean(np.abs(preds - y_test)))
    logger.info(f"MAE: {mae:.2f} trips")
    return mae


def save_model(model: LSTMModel, path: Path) -> None:
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def save_to_database(mae: float) -> None:
    with Session(engine) as session:
        session.add(ModelVersion(name=ModelName.B.value, mae=mae))
        session.commit()


def run_training_pipeline() -> None:
    demand = load_data()
    X_train, X_test, y_train, y_test = build_sequences(demand)
    model = train_model(X_train, y_train)
    mae = evaluate(model, X_test, y_test)
    MODEL_PATH_LSTM.parent.mkdir(exist_ok=True)
    save_model(model, MODEL_PATH_LSTM)
    save_to_database(mae)


if __name__ == "__main__":
    setup_logging()
    run_training_pipeline()
