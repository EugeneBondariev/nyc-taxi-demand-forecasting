import logging
import os
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from .config import DB_URL, MODEL_PATH_A, MODEL_PATH_LSTM, ModelName
from .context import correlation_id_var
from .database import DemandHistory, ModelVersion, Prediction, init_db
from .logger import setup_logging
from .train_lstm import LSTMModel

logger = logging.getLogger(__name__)
model_a = None
model_b = None
model_a_version_id = None
model_b_version_id = None
engine = None

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


app = FastAPI(
    title="NYC Taxi Demand API",
    description="Real-time trip-count demand forecasting for NYC taxi zones. "
                "Uses XGBoost (v1) and LSTM (A/B) models with drift-based retraining.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(CorrelationIdMiddleware)

v1 = APIRouter(prefix="/v1")


class PredictionRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265, description="NYC taxi zone ID (1-265)")
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(ge=0, le=6, description="Day of week (0=Monday)")
    week: int = Field(ge=1, le=53, description="ISO week number (1-53)")
    is_holiday: int = Field(default=0, ge=0, le=1, description="1 if a US public holiday")


class PredictionRequestV1(BaseModel):
    zone_id: int = Field(ge=1, le=265, description="NYC taxi zone ID (1-265)")
    prediction_time: datetime = Field(description="Timestamp for which to forecast demand")
    is_holiday: int = Field(default=0, ge=0, le=1, description="1 if a US public holiday")

    model_config = {
        "json_schema_extra": {
            "example": {"zone_id": 161, "prediction_time": "2024-03-05T18:00:00", "is_holiday": 0}
        }
    }


class PredictionResponse(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int
    week: int
    predicted_trips: float = Field(description="Forecast trip count (≥ 0)")


class ExplainRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265, description="NYC taxi zone ID (1-265)")
    prediction_time: datetime = Field(description="Timestamp to explain")
    is_holiday: int = Field(default=0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {"zone_id": 161, "prediction_time": "2024-03-05T18:00:00"}
        }
    }


class ExplainResponse(BaseModel):
    zone_id: int
    feature_contributions: dict[str, float] = Field(
        description="SHAP contribution of each feature to the prediction"
    )


@app.get("/health", summary="Health check")
@v1.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


def _predict_logic(body: PredictionRequest) -> PredictionResponse:
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

    with Session(engine) as session:
        session.add(Prediction(
            model_version_id=version_id,
            zone_id=zone_id,
            hour=hour,
            day_of_week=day_of_week,
            week=week,
            predicted_trips=result,
        ))
        session.commit()

    logger.info(f"zone={zone_id} model={'lstm' if use_model_b else 'xgb'} prediction={result}")

    return PredictionResponse(
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
        week=week,
        predicted_trips=result,
    )


@v1.post(
    "/predict",
    summary="Forecast taxi demand",
    description="Returns the predicted trip count for a zone at the given timestamp. "
                "Traffic is split 50/50 between XGBoost and LSTM models (A/B test).",
    response_description="Predicted trip count and the echoed request fields",
    dependencies=[Security(verify_api_key)],
)
@limiter.limit("60/minute")
def predict_v1(request: Request, body: PredictionRequestV1) -> PredictionResponse:
    dt = body.prediction_time
    internal = PredictionRequest(
        zone_id=body.zone_id,
        hour=dt.hour,
        day_of_week=dt.weekday(),
        week=dt.isocalendar()[1],
        is_holiday=body.is_holiday,
    )
    return _predict_logic(internal)


@v1.post(
    "/explain",
    summary="Explain a demand prediction (XGBoost SHAP)",
    description="Returns per-feature SHAP contributions for the XGBoost model at the given timestamp.",
    response_description="SHAP feature contributions summing to the model output",
    dependencies=[Security(verify_api_key)],
)
@limiter.limit("60/minute")
def explain_v1(request: Request, body: ExplainRequest) -> ExplainResponse:
    dt = body.prediction_time
    X = pd.DataFrame({
        "PULocationID": [body.zone_id],
        "pickup_hour": [dt.hour],
        "pickup_dow": [dt.weekday()],
        "pickup_week": [dt.isocalendar()[1]],
        "is_holiday": [body.is_holiday],
    })
    explainer = shap.TreeExplainer(model_a)
    shap_values = explainer(X)
    contributions = {
        col: round(float(shap_values[0, i].values), 4)
        for i, col in enumerate(X.columns)
    }
    return ExplainResponse(zone_id=body.zone_id, feature_contributions=contributions)


# Legacy route — kept for backwards compatibility
@app.post("/predict", deprecated=True, include_in_schema=False, dependencies=[Security(verify_api_key)])
@limiter.limit("60/minute")
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    return _predict_logic(body)


app.include_router(v1)
