.PHONY: features train train-lstm train-fare monitoring api frontend test mlflow

include .env
export

DATABASE_URL=postgresql://postgres:$(POSTGRES_PASSWORD)@localhost:5433/nyc_taxi
PYTHONUTF8=1

features:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m src.features

train:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m src.train

train-lstm:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m src.train_lstm

train-fare:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m src.train_fare

monitoring:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m src.monitoring

api:
	DATABASE_URL=$(DATABASE_URL) C:/Python313/python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py

test:
	C:/Python313/python.exe -m pytest tests/ -v

mlflow:
	docker compose up mlflow -d
