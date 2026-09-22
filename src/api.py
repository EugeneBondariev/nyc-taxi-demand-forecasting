import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from .config import MODEL_PATH
from .logger import setup_logging

logger = logging.getLogger(__name__)
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    global model
    model = joblib.load(MODEL_PATH)
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
    X = pd.DataFrame(
        {
            "PULocationID": [request.zone_id],
            "pickup_hour": [request.hour],
            "pickup_dow": [request.day_of_week],
            "pickup_week": [request.week],
        }
    )

    prediction = model.predict(X)

    return {
        "zone_id": request.zone_id,
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "week": request.week,
        "predicted_trips": round(max(0.0, float(prediction[0])), 2),
    }
