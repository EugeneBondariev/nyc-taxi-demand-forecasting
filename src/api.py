import logging
import joblib
import random
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from .train import ModelName
from .logger import setup_logging
from .database import init_db, Prediction, ModelVersion
from .config import MODEL_PATH_A, MODEL_PATH_B, DB_URL

logger = logging.getLogger(__name__)
model_a = None
model_b = None
model_a_version_id = None
model_b_version_id = None
engine = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    global model_a, model_b, model_a_version_id, model_b_version_id, engine
    model_a = joblib.load(MODEL_PATH_A)
    model_b = joblib.load(MODEL_PATH_B)
    logger.info("Models loaded successfully")
    engine = init_db(DB_URL)
    with Session(engine) as session:
        model_a_version_id = session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == ModelName.A.value)
            .order_by(ModelVersion.trained_at.desc())
        ).scalars().first().id
        model_b_version_id = session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == ModelName.B.value)
            .order_by(ModelVersion.trained_at.desc())
        ).scalars().first().id
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    zone_id: int = Field(ge=1, le=265)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    week: int = Field(ge=1, le=53)


class PredictionResponse(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int
    week: int
    predicted_trips: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    zone_id = request.zone_id
    hour = request.hour
    day_of_week = request.day_of_week
    week = request.week

    use_model_b = random.random() < 0.5
    model = model_b if use_model_b else model_a
    version_id = model_b_version_id if use_model_b else model_a_version_id

    X = pd.DataFrame(
        {
            "PULocationID": [zone_id],
            "pickup_hour": [hour],
            "pickup_dow": [day_of_week],
            "pickup_week": [week],
        }
    )
    if use_model_b:
        X["pickup_is_weekend"] = [day_of_week >= 5]

    prediction = model.predict(X)
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
