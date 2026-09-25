import logging
import httpx
import pandas as pd
from urllib.request import urlretrieve
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
from sqlalchemy import insert
from sqlalchemy.orm import Session
from .utils import is_valid_file
from .config import (
    TAXI_DATA_FOLDER,
    WEATHER_DATA_FOLDER,
    DEMAND_DATA,
    DEMAND_TARGET,
    DB_URL,
)
from .database import init_db, DemandHistory
from .logger import setup_logging

FIRST_YEAR_AVAILABLE = 2009
NEXT_YEAR = datetime.now().year + 1
TESTED_YEAR = 2024

logger = logging.getLogger(__name__)


def download_taxi_data(year: int) -> None:
    if year < FIRST_YEAR_AVAILABLE or year >= NEXT_YEAR:
        raise ValueError(
            f"Unexpected year: {year}. Should be between {FIRST_YEAR_AVAILABLE} and {NEXT_YEAR}"
        )

    for i in range(1, 13):
        file = TAXI_DATA_FOLDER / str(year) / f"yellow_tripdata_{year}-{i:02d}.parquet"
        file.parent.mkdir(parents=True, exist_ok=True)

        if not file.exists() or not is_valid_file(file, year, i):
            logger.info(f"Downloading the file for {year}-{i:02d}")
            try:
                urlretrieve(
                    f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{i:02d}.parquet",
                    file,
                )
            except HTTPError as e:
                if e.code == 404:
                    logger.error(f"Data for {year}-{i:02d} not yet available, skipping")
                else:
                    raise
        else:
            logger.info(f"The {year}-{i:02d} file already exists - skipping")


def download_weather_data(year: int):
    file = WEATHER_DATA_FOLDER / str(year) / f"open-meteo-{year}.parquet"
    file.parent.mkdir(parents=True, exist_ok=True)

    if not file.exists():
        logger.info(f"Downloading the {year} year file")

        r = httpx.get(
            f"https://archive-api.open-meteo.com/v1/archive?latitude=40.7128&longitude=-74.006&start_date={year}-01-01&end_date={year}-12-31&hourly=temperature_2m,precipitation,snowfall&timezone=America%2FNew_York",
        )
        df = pd.DataFrame(r.json()["hourly"])
        df.to_parquet(file)
    else:
        logger.info(f"The {year} file already exists - skipping")


def load_and_clean_taxi_data(
    path: Path, year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for file in path.rglob("*.parquet"):
        file_year, file_month = map(int, file.stem.split("_")[2].split("-"))

        if year is not None and file_year != year:
            continue
        if month is not None and file_month != month:
            continue

        df = pd.read_parquet(file)
        df = clean_taxi_data(df, file_year, file_month)

        dfs.append(df)
        logger.info(f"{file} was successfully processed")

    if not dfs:
        raise ValueError(f"No parquet files found for year={year}, month={month}")

    return pd.concat(dfs, ignore_index=True).sort_values("tpep_pickup_datetime")


def clean_taxi_data(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df = df[
        (df["tpep_pickup_datetime"].dt.year == year)
        & (df["tpep_pickup_datetime"].dt.month == month)
    ]
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]
    df[["Airport_fee", "extra"]] = df[["Airport_fee", "extra"]].fillna(0)
    return df


def load_and_clean_weather_data(path: Path):
    dfs: list[pd.DataFrame] = []

    for file in path.rglob("*.parquet"):
        df = pd.read_parquet(file)
        df = clean_weather_data(df)

        dfs.append(df)
        logger.info(f"{file} was successfully processed")

    if not dfs:
        raise ValueError(f"No parquet files found")

    return pd.concat(dfs, ignore_index=True)


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour_ts"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.drop(columns=["time"])
    return df


def build_demand_table(taxi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    demand = add_taxi_data(taxi_df)
    demand = add_weather_data(demand, weather_df)
    print(demand.columns)

    return demand


def add_taxi_data(taxi_df: pd.DataFrame):
    taxi_df["pickup_hour_ts"] = taxi_df["tpep_pickup_datetime"].dt.floor("h")
    demand = (
        taxi_df.groupby(["PULocationID", "pickup_hour_ts"])
        .size()
        .reset_index(name="trip_count")
    )
    demand["pickup_hour"] = demand["pickup_hour_ts"].dt.hour  # 11
    demand["pickup_dow"] = demand["pickup_hour_ts"].dt.dayofweek  # 2 (Wednesday)
    demand["pickup_week"] = demand["pickup_hour_ts"].dt.isocalendar().week.astype(int)
    demand["pickup_is_weekend"] = demand["pickup_dow"] >= 5

    return demand


def add_weather_data(taxi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    demand = pd.merge(left=taxi_df, right=weather_df, how="left", on="pickup_hour_ts")
    return demand


def populate_demand_history(demand: pd.DataFrame) -> None:
    from .database import Base
    engine = init_db(DB_URL)
    DemandHistory.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(engine)
    records = (
        demand[
            [
                "PULocationID",
                "pickup_hour_ts",
                "trip_count",
                "pickup_hour",
                "pickup_dow",
                "temperature_2m",
                "precipitation",
                "snowfall",
            ]
        ]
        .rename(columns={"PULocationID": "zone_id"})
        .to_dict("records")
    )
    with Session(engine) as session:
        session.execute(insert(DemandHistory), records)
        session.commit()
    logger.info(f"Populated demand_history with {len(records)} rows")


if __name__ == "__main__":
    setup_logging()
    download_taxi_data(TESTED_YEAR)
    download_weather_data(2025)

    taxi_data = load_and_clean_taxi_data(TAXI_DATA_FOLDER)
    weather_data = load_and_clean_weather_data(WEATHER_DATA_FOLDER)

    demand = build_demand_table(taxi_data, weather_data)
    demand.to_parquet(DEMAND_DATA, index=False)
    populate_demand_history(demand)

    logger.info(demand[DEMAND_TARGET].describe())
    logger.info(f"Saved {len(demand)} rows to {DEMAND_DATA}")
