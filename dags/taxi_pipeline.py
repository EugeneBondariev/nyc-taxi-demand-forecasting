from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

ENV = {
    "DATABASE_URL": "postgresql://postgres:{{ var.value.postgres_password }}@postgres:5432/nyc_taxi",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_USER": "Eugene",
    "PYTHONUTF8": "1",
}


def task(task_id: str, module: str) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image="uber-worker",
        command=f"python -m {module}",
        network_mode="uber_default",
        environment=ENV,
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[
            Mount(source="uber_models", target="/app/models", type="volume"),
        ],
    )


with DAG(
    dag_id="taxi_pipeline",
    description="Download data, retrain models, monitor performance",
    schedule="0 0 1 * *",  # 1st of every month
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    download = task("download", "src.features")
    train_demand = task("train_demand", "src.train")
    train_fare = task("train_fare", "src.train_fare")
    monitor = task("monitor", "src.monitoring")

    download >> [train_demand, train_fare] >> monitor
