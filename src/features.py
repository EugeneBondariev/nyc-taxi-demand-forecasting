import logging
import pandas as pd
from urllib.request import urlretrieve
from urllib.error import HTTPError
from pathlib import Path
from datetime import datetime
from .utils import is_valid_file
from .config import RAW_DATA_FOLDER, DEMAND_DATA, DEMAND_TARGET
from .logger import setup_logging

FIRST_YEAR_AVAILABLE = 2009
NEXT_YEAR = datetime.now().year + 1

logger = logging.getLogger(__name__)


def download_data(year: int) -> None:
    if year < FIRST_YEAR_AVAILABLE or year >= NEXT_YEAR:
        raise ValueError(
            f"Unexpected year: {year}. Should be between {FIRST_YEAR_AVAILABLE} and {NEXT_YEAR}"
        )

    for i in range(1, 13):
        file = RAW_DATA_FOLDER / str(year) / f"yellow_tripdata_{year}-{i:02d}.parquet"
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
                    logger.info(f"Data for {year}-{i:02d} not yet available, skipping")
                else:
                    raise
        else:
            logger.info(f"The {year}-{i:02d} file already exists - skipping")


def load_and_clean(
    path: Path, year: int | None = None, month: int | None = None
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for file in path.rglob("yellow_tripdata_*.parquet"):
        file_year, file_month = map(int, file.stem.split("_")[2].split("-"))

        if year is not None and file_year != year:
            continue
        if month is not None and file_month != month:
            continue

        df = pd.read_parquet(file)
        df = clean(df, file_year, file_month)

        dfs.append(df)
        logger.info(f"{file} was successfully processed")

    if not dfs:
        raise ValueError(f"No parquet files found for year={year}, month={month}")

    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df = df[
        (df["tpep_pickup_datetime"].dt.year == year)
        & (df["tpep_pickup_datetime"].dt.month == month)
    ]
    return df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]


def build_demand_table(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour_ts"] = df["tpep_pickup_datetime"].dt.floor("h")
    demand = (
        df.groupby(["PULocationID", "pickup_hour_ts"])
        .size()
        .reset_index(name="trip_count")
    )
    demand["pickup_hour"] = demand["pickup_hour_ts"].dt.hour  # 11
    demand["pickup_dow"] = demand["pickup_hour_ts"].dt.dayofweek  # 2 (Wednesday)
    demand["pickup_week"] = demand["pickup_hour_ts"].dt.isocalendar().week.astype(int)

    return demand


if __name__ == "__main__":
    setup_logging()
    for year in range(FIRST_YEAR_AVAILABLE, NEXT_YEAR):
        download_data(year)
    df = load_and_clean(RAW_DATA_FOLDER)
    demand = build_demand_table(df)

    demand.to_parquet(DEMAND_DATA, index=False)
    logger.info(demand[DEMAND_TARGET].describe())
    logger.info(f"Saved {len(demand)} rows to {DEMAND_DATA}")
