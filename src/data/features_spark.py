"""
PySpark version of features.py.
Requires Java 11+ installed and JAVA_HOME set.
Produces the same demand.parquet and populates demand_history table.
"""
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..core.config import DEMAND_DATA, TAXI_DATA_FOLDER, WEATHER_DATA_FOLDER
from .features import download_taxi_data, download_weather_data, populate_demand_history
from ..core.logger import setup_logging

logger = logging.getLogger(__name__)

TESTED_YEAR = 2024


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("nyc_taxi_demand")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def clean_taxi_spark(df, year: int, month: int):
    return (
        df.filter(
            (F.year("tpep_pickup_datetime") == year) &
            (F.month("tpep_pickup_datetime") == month)
        )
        .filter((F.col("trip_distance") > 0) & (F.col("fare_amount") > 0))
        .filter((F.col("fare_amount") <= 150) & (F.col("trip_distance") <= 40))
        .fillna({"Airport_fee": 0.0, "extra": 0.0})
        .withColumn(
            "trip_duration",
            (F.unix_timestamp("tpep_dropoff_datetime") -
             F.unix_timestamp("tpep_pickup_datetime")) / 60,
        )
    )


def build_demand_spark(taxi_df, weather_df):
    taxi_df = taxi_df.withColumn(
        "pickup_hour_ts",
        F.date_trunc("hour", F.col("tpep_pickup_datetime")),
    )

    demand = (
        taxi_df.groupBy("PULocationID", "pickup_hour_ts")
        .count()
        .withColumnRenamed("count", "trip_count")
    )

    demand = (
        demand
        .withColumn("pickup_hour", F.hour("pickup_hour_ts"))
        # Spark dayofweek: 1=Sun..7=Sat → shift to match pandas (0=Mon..6=Sun)
        .withColumn("pickup_dow", (F.dayofweek("pickup_hour_ts") + 5) % 7)
        .withColumn("pickup_week", F.weekofyear("pickup_hour_ts"))
        .withColumn("pickup_is_weekend", F.col("pickup_dow") >= 5)
    )

    weather_df = weather_df.withColumn(
        "pickup_hour_ts",
        F.to_timestamp("time").cast("timestamp"),
    ).drop("time")

    demand = demand.join(weather_df, on="pickup_hour_ts", how="left")
    return demand


def run_features_spark() -> None:
    logger.info("Downloading data (same as pandas pipeline)...")
    download_taxi_data(TESTED_YEAR)
    download_weather_data(2025)

    spark = build_spark_session()
    logger.info(f"Spark version: {spark.version}")

    logger.info("Reading taxi parquets...")
    taxi_raw = spark.read.parquet(str(TAXI_DATA_FOLDER / "**" / "*.parquet"))

    logger.info("Cleaning taxi data...")
    year = TESTED_YEAR
    cleaned_months = []
    for month in range(1, 13):
        try:
            cleaned_months.append(clean_taxi_spark(taxi_raw, year, month))
        except Exception:  # noqa: BLE001
            logger.debug(f"Skipping month {month} — data not available or invalid")
    taxi_df = cleaned_months[0]
    for df in cleaned_months[1:]:
        taxi_df = taxi_df.union(df)

    logger.info("Reading weather parquets...")
    weather_df = spark.read.parquet(str(WEATHER_DATA_FOLDER / "**" / "*.parquet"))

    logger.info("Building demand table...")
    demand = build_demand_spark(taxi_df, weather_df)

    logger.info(f"Demand rows: {demand.count():,}")

    DEMAND_DATA.parent.mkdir(parents=True, exist_ok=True)
    demand.toPandas().to_parquet(DEMAND_DATA, index=False)
    logger.info(f"Saved → {DEMAND_DATA}")

    logger.info("Populating demand_history table...")
    demand_pd = demand.toPandas()
    populate_demand_history(demand_pd)

    spark.stop()


if __name__ == "__main__":
    setup_logging()
    run_features_spark()
