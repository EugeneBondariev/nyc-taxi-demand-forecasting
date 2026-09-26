import pandas as pd
from src.features import clean_taxi_data

SAMPLE_TIMESTAMPS = [
    pd.Timestamp("2024-01-01 17:28:05"),
    pd.Timestamp("2024-01-01 18:28:05"),
    pd.Timestamp("2024-01-01 19:28:05"),
]
SAMPLE_DROPOFFS = [
    pd.Timestamp("2024-01-01 17:38:05"),
    pd.Timestamp("2024-01-01 18:38:05"),
    pd.Timestamp("2024-01-01 19:38:05"),
]


def make_df(trip_distance, fare_amount, pickups=SAMPLE_TIMESTAMPS, dropoffs=SAMPLE_DROPOFFS):
    n = len(pickups)
    return pd.DataFrame({
        "tpep_pickup_datetime": pickups,
        "tpep_dropoff_datetime": dropoffs,
        "trip_distance": trip_distance,
        "fare_amount": fare_amount,
        "Airport_fee": [0.0] * n,
        "extra": [0.0] * n,
    })


def test_clean_filters_negative_fares():
    df = clean_taxi_data(make_df([1, 2, 3], [-7, 2, 15]), 2024, 1)
    assert len(df) == 2
    assert -7 not in df["fare_amount"].values


def test_clean_filters_zero_distance():
    df = clean_taxi_data(make_df([1, 0, 3], [5, 2, 15]), 2024, 1)
    assert len(df) == 2
    assert 0 not in df["trip_distance"].values


def test_clean_filters_wrong_dates():
    pickups = [
        pd.Timestamp("2025-01-01 17:28:05"),
        pd.Timestamp("2024-01-01 18:28:05"),
        pd.Timestamp("2024-02-02 19:28:05"),
    ]
    dropoffs = [
        pd.Timestamp("2025-01-01 17:38:05"),
        pd.Timestamp("2024-01-01 18:38:05"),
        pd.Timestamp("2024-02-02 19:38:05"),
    ]
    df = clean_taxi_data(make_df([1, 2, 3], [5, 10, 15], pickups, dropoffs), 2024, 1)
    assert df.iloc[0]["tpep_pickup_datetime"] == pd.Timestamp("2024-01-01 18:28:05")


def test_clean_returns_empty_when_all_rows_invalid():
    pickups = [pd.Timestamp("2025-03-01 17:28:05")]
    dropoffs = [pd.Timestamp("2025-03-01 17:38:05")]
    df = clean_taxi_data(make_df([-1], [-5], pickups, dropoffs), 2024, 1)
    assert len(df) == 0
