.PHONY: features features-spark train train-lstm train-fare train-classification monitoring ab-test cluster anomaly api frontend test mlflow airflow dvc-push dvc-pull migrate migrate-new rollback

include .env
export

DATABASE_URL=postgresql://postgres:$(POSTGRES_PASSWORD)@localhost:5433/nyc_taxi
PYTHONUTF8=1
PYTHON=.venv/Scripts/python.exe

features:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.features

features-spark:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.features_spark

train:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train

train-lstm:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train_lstm

train-fare:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.train_fare

monitoring:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.monitoring

ab-test:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.ab_test

cluster:
	$(PYTHON) -m src.clustering

anomaly:
	$(PYTHON) -m src.anomaly

train-classification:
	$(PYTHON) -m src.train_classification

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

dvc-push:
	$(PYTHON) -m dvc add data models && $(PYTHON) -m dvc push

dvc-pull:
	$(PYTHON) -m dvc pull

migrate:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m alembic upgrade head

migrate-new:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m alembic revision --autogenerate -m "$(msg)"

rollback:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -c "from src.utils import rollback_model; from src.config import MODEL_PATH_A; rollback_model(MODEL_PATH_A)"

k8s-deploy:
	kubectl create secret generic uber-secrets \
		--from-literal=database-url=$(DATABASE_URL) \
		--from-literal=api-key=$(API_KEY) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap uber-config \
		--from-literal=mlflow-uri=$(MLFLOW_TRACKING_URI) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/
