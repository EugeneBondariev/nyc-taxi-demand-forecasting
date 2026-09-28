.PHONY: features features-spark train train-lstm train-fare train-classification monitoring ab-test cluster anomaly api frontend test mlflow airflow dvc-push dvc-pull migrate migrate-new rollback baseline

include .env
export

DATABASE_URL=postgresql://postgres:$(POSTGRES_PASSWORD)@localhost:5433/nyc_taxi
PYTHONUTF8=1
PYTHON=.venv/Scripts/python.exe

features:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.data.features

features-spark:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.data.features_spark

train:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.training.train

train-lstm:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.training.train_lstm

train-fare:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.training.train_fare

monitoring:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.monitoring.monitoring

ab-test:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.training.ab_test

cluster:
	$(PYTHON) -m src.training.clustering

anomaly:
	$(PYTHON) -m src.training.anomaly

train-classification:
	$(PYTHON) -m src.training.train_classification

api:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

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

baseline:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m src.training.baseline

migrate:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m alembic upgrade head

migrate-new:
	DATABASE_URL=$(DATABASE_URL) $(PYTHON) -m alembic revision --autogenerate -m "$(msg)"

rollback:
	@echo "To rollback, set the 'champion' alias to a previous version in MLflow:"
	@echo "  mlflow models set-alias -m demand_xgboost --alias champion --version <N>"
	@echo "Or via the MLflow UI at $(MLFLOW_TRACKING_URI)/#/models/demand_xgboost"

k8s-deploy:
	kubectl create secret generic uber-secrets \
		--from-literal=database-url=$(DATABASE_URL) \
		--from-literal=api-key=$(API_KEY) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap uber-config \
		--from-literal=mlflow-uri=$(MLFLOW_TRACKING_URI) \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/
