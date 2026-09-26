.PHONY: features train train-lstm train-fare monitoring api frontend test mlflow airflow

include .env
export

DATABASE_URL=postgresql://postgres:$(POSTGRES_PASSWORD)@localhost:5433/nyc_taxi
PYTHONUTF8=1
PYTHON=.venv/Scripts/python.exe

features:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.features

train:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train

train-lstm:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train_lstm

train-fare:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train_fare

monitoring:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.monitoring

api:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py

test:
	$(PYTHON) -m pytest tests/ -v

mlflow:
	docker compose up mlflow -d

airflow:
	docker compose up airflow-init airflow-webserver airflow-scheduler -d
