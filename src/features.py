import pandas as pd
import calendar
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve

ROOT = Path(__file__).parent.parent
RAW_DATA_FOLDER = ROOT / "data"
FIRST_YEAR_AVAILABLE = 2009
NEXT_YEAR = datetime.now().year + 1


def download_data(year: int) -> None:
    if year < 2009 or year > 2030:
        raise ValueError(f"Unexpected year: {year}. Should be between 2009 and 2030")

    for i in range(1, 13):
        file = RAW_DATA_FOLDER / f"yellow_tripdata_{year}-{i:02d}.parquet"

        if not file.exists() or not is_valid_file(file, year, i):
            print(f"Downloading the file for {year}-{i:02d}")
            urlretrieve(
                f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{i:02d}.parquet",
                file,
            )
        else:
            print(f"The {year}-{i:02d} file already exists - skipping")


import calendar


def is_valid_file(file: Path, year: int, month: int) -> bool:
    try:
        df = pd.read_parquet(file, columns=["tpep_pickup_datetime"])
        last_day = calendar.monthrange(year, month)[1]
        expected_last = pd.Timestamp(year=year, month=month, day=last_day)
        return df["tpep_pickup_datetime"].max().normalize() >= expected_last
    except Exception:
        return False


def load_and_clean(path: Path) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for file in list(path.glob("yellow_tripdata_*.parquet")):
        parts = file.stem.split("_")
        year, month = parts[2].split("-")

        df = pd.read_parquet(file)
        df = df[
            (df["tpep_pickup_datetime"].dt.year == int(year))
            & (df["tpep_pickup_datetime"].dt.month == int(month))
        ]
        df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]

        dfs.append(df)
        print(f"{file} was successfully processed")

    return pd.concat(dfs, ignore_index=True)


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
    for year in range(FIRST_YEAR_AVAILABLE, NEXT_YEAR):
        download_data(year)
    df = load_and_clean(RAW_DATA_FOLDER)
    demand = build_demand_table(df)
    print(demand.head())

    output = ROOT / "data" / "demand.parquet"
    demand.to_parquet(output, index=False)
    print(demand["trip_count"].describe())
    print(f"Saved {len(demand)} rows to {output}")
