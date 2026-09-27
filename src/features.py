import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import httpx
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .config import (
    AIRPORT_ZONES,
    DB_URL,
    DEMAND_DATA,
    DEMAND_TARGET,
    EVENTS_DATA_FOLDER,
    TAXI_DATA_FOLDER,
    WEATHER_DATA_FOLDER,
)
from .database import DemandHistory, init_db
from .events import add_nyc_event_feature, build_event_lookup
from .logger import setup_logging
from .utils import ensure_parent, is_valid_file
from .validation import validate_demand, validate_taxi

FIRST_YEAR_AVAILABLE = 2009
NEXT_YEAR = (
    datetime.now(timezone.utc).year + 1
)
TESTED_YEAR = 2024

logger = logging.getLogger(__name__)


def download_taxi_data(year: int) -> None:
    if year < FIRST_YEAR_AVAILABLE or year >= NEXT_YEAR:
        raise ValueError(
            f"Unexpected year: {year}. Should be between {FIRST_YEAR_AVAILABLE} and {NEXT_YEAR}"
        )

    for i in range(1, 13):
        file = TAXI_DATA_FOLDER / str(year) / f"yellow_tripdata_{year}-{i:02d}.parquet"
        ensure_parent(file)

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


def download_events_data(year: int) -> None:
    file = EVENTS_DATA_FOLDER / str(year) / f"holidays-{year}.parquet"
    ensure_parent(file)
    if not file.exists():
        logger.info(f"Downloading {year} US public holidays")
        r = httpx.get(f"https://date.nager.at/api/v3/PublicHolidays/{year}/US")
        r.raise_for_status()
        df = pd.DataFrame(r.json())[["date", "name"]]
        df.to_parquet(file)
    else:
        logger.info(f"Holiday data for {year} already exists - skipping")


def load_events_data() -> set:
    dates: set = set()
    for file in EVENTS_DATA_FOLDER.rglob("*.parquet"):
        df = pd.read_parquet(file)
        dates.update(df["date"].astype(str).tolist())
    return dates


def download_weather_data(year: int):
    file = WEATHER_DATA_FOLDER / str(year) / f"open-meteo-{year}.parquet"
    ensure_parent(file)

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

    return pd.concat(dfs, ignore_index=True)


def clean_taxi_data(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df = df[
        (df["tpep_pickup_datetime"].dt.year == year)
        & (df["tpep_pickup_datetime"].dt.month == month)
    ]

    df = df[(df["trip_distance"] > 0.15) & (df["fare_amount"] > 3.5)]
    df = df[(df["fare_amount"] <= 150) & (df["trip_distance"] <= 40)]
    df[["Airport_fee", "extra"]] = df[["Airport_fee", "extra"]].fillna(0)

    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return df


def load_and_clean_weather_data(path: Path):
    dfs: list[pd.DataFrame] = []

    for file in path.rglob("*.parquet"):
        df = pd.read_parquet(file)
        df = clean_weather_data(df)

        dfs.append(df)
        logger.info(f"{file} was successfully processed")

    if not dfs:
        raise ValueError("No parquet files found")

    return pd.concat(dfs, ignore_index=True)


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour_ts"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.drop(columns=["time"])
    return df


def add_event_features(demand: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    demand["is_holiday"] = (
        demand["pickup_hour_ts"].dt.strftime("%Y-%m-%d").isin(holiday_dates).astype(int)
    )
    return demand


def add_lag_features(
    demand: pd.DataFrame, lags: list[int] | None = None
) -> pd.DataFrame:
    """Compute per-zone trip_count lags; drop rows without full history."""
    if lags is None:
        lags = [24, 168]
    df = demand.sort_values(["PULocationID", "pickup_hour_ts"]).copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby("PULocationID")["trip_count"].shift(lag)
    return df.dropna(subset=[f"lag_{lag}h" for lag in lags]).reset_index(drop=True)


def build_demand_table(
    taxi_df: pd.DataFrame, weather_df: pd.DataFrame, holiday_dates: set
) -> pd.DataFrame:
    validate_taxi(taxi_df)
    demand = add_taxi_data(taxi_df)
    demand = add_weather_data(demand, weather_df)
    demand = add_event_features(demand, holiday_dates)
    years = demand["pickup_hour_ts"].dt.year.unique().tolist()
    demand = add_nyc_event_feature(demand, build_event_lookup(years))
    validate_demand(demand)
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
    demand["is_airport"] = demand["PULocationID"].isin(AIRPORT_ZONES).astype(int)

    return demand


def add_weather_data(taxi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    demand = pd.merge(left=taxi_df, right=weather_df, how="left", on="pickup_hour_ts")
    return demand


def populate_demand_history(demand: pd.DataFrame) -> None:
    engine = init_db(DB_URL)
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
                "is_holiday",
            ]
        ]
        .rename(columns={"PULocationID": "zone_id"})
        .to_dict("records")
    )
    upsert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    with Session(engine) as session:
        session.execute(upsert(DemandHistory).on_conflict_do_nothing(), records)
        session.commit()
    logger.info(f"Upserted demand_history with {len(records)} rows")


if __name__ == "__main__":
    setup_logging()
    download_taxi_data(TESTED_YEAR)
    download_weather_data(2025)
    download_events_data(TESTED_YEAR)
    download_events_data(2025)

    taxi_data = load_and_clean_taxi_data(TAXI_DATA_FOLDER)
    weather_data = load_and_clean_weather_data(WEATHER_DATA_FOLDER)
    holiday_dates = load_events_data()

    demand = build_demand_table(taxi_data, weather_data, holiday_dates)
    demand.to_parquet(DEMAND_DATA, index=False)
    populate_demand_history(demand)

    logger.info(demand[DEMAND_TARGET].describe())
    logger.info(f"Saved {len(demand)} rows to {DEMAND_DATA}")
