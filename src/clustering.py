import logging
import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .config import DEMAND_DATA, ROOT, MLFLOW_TRACKING_URI
from .utils import load_data
from .logger import setup_logging

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

N_CLUSTERS = 6
CLUSTER_PATH = ROOT / "data" / "processed" / "zone_clusters.parquet"


def build_zone_profiles(demand: pd.DataFrame) -> pd.DataFrame:
    """Mean trip count per hour for each zone → (n_zones, 24) feature matrix."""
    return (
        demand.groupby(["PULocationID", "pickup_hour"])["trip_count"]
        .mean()
        .unstack(fill_value=0)
    )


def run_clustering(n_clusters: int = N_CLUSTERS) -> None:
    demand = load_data(DEMAND_DATA)

    logger.info("Building hourly demand profiles per zone...")
    profile = build_zone_profiles(demand)

    scaler = StandardScaler()
    X = scaler.fit_transform(profile.values)

    logger.info(f"Running KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    logger.info(f"Silhouette score: {sil:.4f}  (closer to 1 = better separation)")

    result = pd.DataFrame({"zone_id": profile.index, "cluster": labels})
    CLUSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CLUSTER_PATH, index=False)
    logger.info(f"Saved → {CLUSTER_PATH}")

    for c in range(n_clusters):
        zones = result[result["cluster"] == c]["zone_id"].tolist()
        logger.info(f"  Cluster {c}: {len(zones)} zones")

    mlflow.set_experiment("clustering")
    with mlflow.start_run(run_name="kmeans_zones"):
        mlflow.set_tag("mlflow.user", os.getenv("MLFLOW_USER", ""))
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_zones", len(profile))
        mlflow.log_param("features", "hourly_mean_trip_count_0-23")


if __name__ == "__main__":
    setup_logging()
    run_clustering()
