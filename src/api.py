import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from .logger import setup_logging
from .database import init_db, Prediction
from .config import MODEL_PATH, DB_URL

logger = logging.getLogger(__name__)
model = None
engine = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    global model, engine
    model = joblib.load(MODEL_PATH)
    engine = init_db(DB_URL)
    logger.info("Model loaded successfully")
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

    X = pd.DataFrame(
        {
            "PULocationID": [zone_id],
            "pickup_hour": [hour],
            "pickup_dow": [day_of_week],
            "pickup_week": [week],
        }
    )

    prediction = model.predict(X)
    result = round(max(0.0, float(prediction[0])), 2)

    record = Prediction(
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
