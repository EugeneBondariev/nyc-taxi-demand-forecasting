# NYC Taxi Demand Forecasting

Predicts hourly taxi trip demand by pickup zone in New York City using XGBoost and LSTM models, served via a REST API with A/B testing.

**Live API:** https://uber-api.onrender.com

## Architecture

```
NYC TLC data + Open-Meteo weather
        ↓
   src/features.py        ← download & process raw data
        ↓
   PostgreSQL             ← demand_history table
        ↓
   src/train.py           ← XGBoost demand model
   src/train_lstm.py      ← LSTM demand model
   src/train_fare.py      ← fare prediction models
        ↓
   MLflow                 ← experiment tracking
        ↓
   src/api.py             ← FastAPI, A/B test between XGBoost and LSTM
```

Airflow runs the full pipeline on the 1st of every month.

## Stack

| Layer | Tool |
|---|---|
| Models | XGBoost, LSTM (PyTorch) |
| API | FastAPI |
| Orchestration | Apache Airflow (DockerOperator) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Database | PostgreSQL |
| Containerization | Docker Compose |

## Quick Start

```bash
# Copy env file and set your Postgres password
cp .env.example .env

# Start all services
docker compose up postgres postgres-airflow mlflow airflow-scheduler -d

# Run the pipeline manually
make features
make train
make train-lstm
make train-fare

# Start the API
make api
```

## API

```
POST /predict
{
  "zone_id": 161,
  "hour": 9,
  "day_of_week": 1,
  "week": 14
}

→ { "predicted_trips": 42.3, ... }

GET /health → { "status": "ok" }
```

## Retraining

After each monthly run, snapshot the data and models:

```bash
make dvc-push
git add data.dvc models.dvc
git commit -m "retrain: $(date +%B %Y)"
```

## MLflow UI

```bash
make mlflow
# open http://localhost:5000
```
