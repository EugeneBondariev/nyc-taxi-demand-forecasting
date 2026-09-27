import logging
import os
import random
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from .config import DB_URL, MODEL_PATH_A, MODEL_PATH_LSTM, ModelName
from .database import DemandHistory, ModelVersion, Prediction, init_db
from .logger import setup_logging
from .train_lstm import LSTMModel

logger = logging.getLogger(__name__)
model_a = None
model_b = None
model_a_version_id = None
model_b_version_id = None
engine = None

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
limiter = Limiter(key_func=get_remote_address)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> None:
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    global model_a, model_b, model_a_version_id, model_b_version_id, engine
    model_a = joblib.load(MODEL_PATH_A)
    model_b = LSTMModel()
    model_b.load_state_dict(torch.load(MODEL_PATH_LSTM, weights_only=True))
    model_b.eval()
    logger.info("Models loaded successfully")
    engine = init_db(DB_URL)
    with Session(engine) as session:
        row_a = session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == ModelName.DEMAND_XGB.value)
            .order_by(ModelVersion.trained_at.desc())
        ).scalars().first()
        model_a_version_id = row_a.id if row_a else None

        row_b = session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == ModelName.DEMAND_LSTM.value)
            .order_by(ModelVersion.trained_at.desc())
        ).scalars().first()
        model_b_version_id = row_b.id if row_b else None
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CorrelationIdMiddleware)


class PredictionRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    week: int = Field(ge=1, le=53)
    is_holiday: int = Field(default=0, ge=0, le=1)


class PredictionResponse(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int
    week: int
    predicted_trips: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", dependencies=[Security(verify_api_key)])
@limiter.limit("60/minute")
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    zone_id = body.zone_id
    hour = body.hour
    day_of_week = body.day_of_week
    week = body.week
    is_holiday = body.is_holiday

    use_model_b = random.random() < 0.5
    version_id = model_b_version_id if use_model_b else model_a_version_id

    if use_model_b:
        with Session(engine) as session:
            rows = session.execute(
                select(DemandHistory)
                .where(DemandHistory.zone_id == zone_id)
                .order_by(DemandHistory.pickup_hour_ts.desc())
                .limit(24)
            ).scalars().all()
        if len(rows) < 24:
            raise HTTPException(status_code=422, detail=f"Not enough history for zone {zone_id}")
        rows = sorted(rows, key=lambda r: r.pickup_hour_ts)
        X_lstm = np.array(
            [[r.trip_count, r.pickup_hour, r.pickup_dow, r.temperature_2m or 0.0,
              r.precipitation or 0.0, r.snowfall or 0.0, r.is_holiday or 0]
             for r in rows],
            dtype=np.float32,
        )
        X_tensor = torch.tensor(X_lstm).unsqueeze(0)
        with torch.no_grad():
            result = round(max(0.0, model_b(X_tensor).item()), 2)
    else:
        X = pd.DataFrame(
            {
                "PULocationID": [zone_id],
                "pickup_hour": [hour],
                "pickup_dow": [day_of_week],
                "pickup_week": [week],
                "is_holiday": [is_holiday],
            }
        )
        prediction = model_a.predict(X)
        result = round(max(0.0, float(prediction[0])), 2)

    record = Prediction(
        model_version_id=version_id,
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
        week=week,
        predicted_trips=result,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()

    return PredictionResponse(
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
        week=week,
        predicted_trips=result,
    )
