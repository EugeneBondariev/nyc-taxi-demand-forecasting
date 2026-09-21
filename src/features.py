import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DATA = ROOT / "data" / "yellow_tripdata_2024-01.parquet"


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[
        (df["tpep_pickup_datetime"] >= "2024-01-01")
        & (df["tpep_pickup_datetime"] < "2024-02-01")
    ]
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]

    return df


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
    df = load_and_clean(RAW_DATA)
    demand = build_demand_table(df)
    print(demand.head())

    output = ROOT / "data" / "demand.parquet"
    demand.to_parquet(output, index=False)
    print(demand["trip_count"].describe())
    print(f"Saved {len(demand)} rows to {output}")
