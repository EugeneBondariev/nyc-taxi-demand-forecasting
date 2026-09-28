import logging
import os
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import httpx
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
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from .config import (
    AIRPORT_ZONES,
    CONFORMAL_MARGIN_PATH,
    DB_URL,
    DEMAND_FEATURES_V2,
    FEATURE_STATS_PATH,
    MODEL_PATH_A,
    MODEL_PATH_LSTM,
    ModelName,
)
from .context import correlation_id_var
from .database import DemandHistory, ModelVersion, Prediction, init_db
from .events import build_event_lookup
from .features import load_events_data
from .logger import setup_logging
from .train_lstm import LSTMModel
from .utils import detect_feature_drift

logger = logging.getLogger(__name__)
model_a = None
model_b = None
model_a_version_id = None
model_b_version_id = None
engine = None
conformal_margin = None
nyc_event_lookup: set[tuple[str, int]] = set()
holiday_dates: set[str] = set()

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
    global model_a, model_b, model_a_version_id, model_b_version_id, engine, conformal_margin, nyc_event_lookup, holiday_dates
    nyc_event_lookup = build_event_lookup(list(range(2020, 2036)))
    holiday_dates = load_events_data()
    model_a = joblib.load(MODEL_PATH_A)
    try:
        model_b = LSTMModel()
        model_b.load_state_dict(torch.load(MODEL_PATH_LSTM, weights_only=True))
        model_b.eval()
        logger.info("Models loaded successfully")
    except Exception as e:
        model_b = None
        logger.warning(f"LSTM model failed to load — A/B test will use XGBoost only: {e}")
    if CONFORMAL_MARGIN_PATH.exists():
        conformal_margin = float(np.load(CONFORMAL_MARGIN_PATH)[0])
        logger.info(f"Conformal margin loaded: {conformal_margin:.2f}")
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
    snowfall: float = Field(default=0.0, ge=0.0, description="Snowfall in cm")
    lag_24h: float = Field(default=0.0, ge=0.0, description="Trip count 24 h ago for this zone")
    lag_168h: float = Field(default=0.0, ge=0.0, description="Trip count 168 h ago for this zone")
    is_nyc_event: int = Field(default=0, ge=0, le=1, description="1 if a major NYC event is near this zone today")


class PredictionRequestV1(BaseModel):
    zone_id: int = Field(ge=1, le=265, description="NYC taxi zone ID (1-265)")
    prediction_time: datetime = Field(description="Timestamp for which to forecast demand")

    model_config = {
        "json_schema_extra": {
            "example": {
                "zone_id": 161,
                "prediction_time": "2024-03-05T18:00:00",
            }
        }
    }


class PredictionResponse(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int
    week: int
    predicted_trips: float = Field(description="Forecast trip count (≥ 0)")
    lower_bound: float | None = Field(default=None, description="80% conformal lower bound")
    upper_bound: float | None = Field(default=None, description="80% conformal upper bound")


class ExplainRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265, description="NYC taxi zone ID (1-265)")
    prediction_time: datetime = Field(description="Timestamp to explain")

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


class ModelVersionResponse(BaseModel):
    id: int
    name: str
    trained_at: datetime | None
    mae: float | None
    mape: float | None
    n_estimators: int | None
    learning_rate: float | None


class ABResultEntry(BaseModel):
    id: int
    name: str
    trained_at: datetime | None
    mae: float | None
    mape: float | None
    prediction_count: int


class DriftResponse(BaseModel):
    baseline_exists: bool
    features_checked: list[str]
    drifted_features: list[str]
    rows_analyzed: int


def _fetch_snowfall(dt: datetime) -> float:
    date_str = dt.strftime("%Y-%m-%d")
    try:
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.7128, "longitude": -74.006,
                "hourly": "snowfall", "timezone": "America/New_York",
                "start_date": date_str, "end_date": date_str,
            },
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()
        target = dt.strftime("%Y-%m-%dT%H:00")
        hours = data["hourly"]["time"]
        snowfall_vals = data["hourly"]["snowfall"]
        if target in hours:
            return float(snowfall_vals[hours.index(target)])
    except Exception:
        pass
    return 0.0


def _build_xgb_features(
    zone_id: int, hour: int, day_of_week: int, week: int,
    is_holiday: int, snowfall: float, lag_24h: float, lag_168h: float,
    is_nyc_event: int,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "PULocationID": zone_id,
        "pickup_hour": hour,
        "pickup_dow": day_of_week,
        "pickup_week": week,
        "is_holiday": is_holiday,
        "snowfall": snowfall,
        "lag_24h": lag_24h,
        "lag_168h": lag_168h,
        "is_airport": int(zone_id in AIRPORT_ZONES),
        "is_nyc_event": is_nyc_event,
    }])[DEMAND_FEATURES_V2]


def _fetch_lag_features(
    zone_id: int, prediction_time: datetime, session: Session
) -> tuple[float, float]:
    """Return (lag_24h, lag_168h) trip counts from DemandHistory; 0.0 if not found."""
    def fetch(ts: datetime) -> float:
        row = session.execute(
            select(DemandHistory.trip_count)
            .where(DemandHistory.zone_id == zone_id)
            .where(DemandHistory.pickup_hour_ts == ts)
        ).scalar_one_or_none()
        return float(row) if row is not None else 0.0

    return (
        fetch(prediction_time - timedelta(hours=24)),
        fetch(prediction_time - timedelta(hours=168)),
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

    use_model_b = model_b is not None and random.random() < 0.5
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
        lo = hi = None
    else:
        X = _build_xgb_features(
            zone_id, hour, day_of_week, week,
            body.is_holiday, body.snowfall, body.lag_24h, body.lag_168h,
            body.is_nyc_event,
        )
        prediction = model_a.predict(X)
        result = round(max(0.0, float(prediction[0])), 2)
        lo = hi = None
        if conformal_margin is not None:
            lo = round(max(0.0, result - conformal_margin), 2)
            hi = round(result + conformal_margin, 2)

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
        lower_bound=lo,
        upper_bound=hi,
    )


@v1.post(
    "/predict",
    summary="Forecast taxi demand",
    description="Returns the predicted trip count for a zone at the given timestamp. "
                "Lag features are fetched from demand history; pass snowfall for weather-aware predictions. "
                "Traffic is split 50/50 between XGBoost and LSTM models (A/B test). "
                "XGBoost responses include an 80% conformal prediction interval.",
    response_description="Predicted trip count with optional confidence interval",
    dependencies=[Security(verify_api_key)],
)
@limiter.limit("60/minute")
def predict_v1(request: Request, body: PredictionRequestV1) -> PredictionResponse:
    dt = body.prediction_time
    with Session(engine) as session:
        lag_24h, lag_168h = _fetch_lag_features(body.zone_id, dt, session)
    internal = PredictionRequest(
        zone_id=body.zone_id,
        hour=dt.hour,
        day_of_week=dt.weekday(),
        week=dt.isocalendar()[1],
        is_holiday=int(dt.strftime("%Y-%m-%d") in holiday_dates),
        snowfall=_fetch_snowfall(dt),
        lag_24h=lag_24h,
        lag_168h=lag_168h,
        is_nyc_event=int((dt.strftime("%Y-%m-%d"), body.zone_id) in nyc_event_lookup),
    )
    return _predict_logic(internal)


@v1.post(
    "/explain",
    summary="Explain a demand prediction (XGBoost SHAP)",
    description="Returns per-feature SHAP contributions for the XGBoost model at the given timestamp. "
                "Lag features are fetched from demand history automatically.",
    response_description="SHAP feature contributions summing to the model output",
    dependencies=[Security(verify_api_key)],
)
@limiter.limit("60/minute")
def explain_v1(request: Request, body: ExplainRequest) -> ExplainResponse:
    dt = body.prediction_time
    with Session(engine) as session:
        lag_24h, lag_168h = _fetch_lag_features(body.zone_id, dt, session)
    X = _build_xgb_features(
        body.zone_id, dt.hour, dt.weekday(), dt.isocalendar()[1],
        int(dt.strftime("%Y-%m-%d") in holiday_dates), _fetch_snowfall(dt),
        lag_24h, lag_168h,
        int((dt.strftime("%Y-%m-%d"), body.zone_id) in nyc_event_lookup),
    )
    explainer = shap.TreeExplainer(model_a)
    shap_values = explainer(X)
    contributions = {
        col: round(float(shap_values[0, i].values), 4)
        for i, col in enumerate(X.columns)
    }
    return ExplainResponse(zone_id=body.zone_id, feature_contributions=contributions)


@v1.get("/models", summary="List trained model versions")
def models_v1() -> list[ModelVersionResponse]:
    with Session(engine) as session:
        rows = session.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc()).limit(20)
        ).scalars().all()
    return [
        ModelVersionResponse(
            id=r.id, name=r.name, trained_at=r.trained_at,
            mae=r.mae, mape=r.mape,
            n_estimators=r.n_estimators, learning_rate=r.learning_rate,
        )
        for r in rows
    ]


@v1.get("/drift", summary="Check input feature drift against training baseline")
def drift_v1() -> DriftResponse:
    with Session(engine) as session:
        rows = session.execute(
            select(DemandHistory).order_by(DemandHistory.pickup_hour_ts.desc()).limit(2000)
        ).scalars().all()
    baseline_exists = FEATURE_STATS_PATH.exists()
    if not rows:
        return DriftResponse(baseline_exists=baseline_exists, features_checked=[], drifted_features=[], rows_analyzed=0)
    df = pd.DataFrame([{
        "PULocationID": r.zone_id,
        "pickup_hour": r.pickup_hour,
        "pickup_dow": r.pickup_dow,
        "is_holiday": r.is_holiday or 0,
        "snowfall": r.snowfall or 0.0,
        "is_airport": int(r.zone_id in AIRPORT_ZONES),
    } for r in rows])
    drifted = detect_feature_drift(df)
    return DriftResponse(
        baseline_exists=baseline_exists,
        features_checked=list(df.columns),
        drifted_features=drifted,
        rows_analyzed=len(df),
    )


@v1.get("/ab-results", summary="Prediction counts and metrics per model version")
def ab_results_v1() -> list[ABResultEntry]:
    with Session(engine) as session:
        versions = session.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc()).limit(10)
        ).scalars().all()
        results = []
        for v in versions:
            count = session.execute(
                select(func.count(Prediction.id)).where(Prediction.model_version_id == v.id)
            ).scalar_one()
            results.append(ABResultEntry(
                id=v.id, name=v.name, trained_at=v.trained_at,
                mae=v.mae, mape=v.mape, prediction_count=count,
            ))
    return results


# Legacy route — kept for backwards compatibility
@app.post("/predict", deprecated=True, include_in_schema=False, dependencies=[Security(verify_api_key)])
@limiter.limit("60/minute")
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    return _predict_logic(body)


app.include_router(v1)
