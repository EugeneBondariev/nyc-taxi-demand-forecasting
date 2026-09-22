import pandas as pd
from src.features import clean

SAMPLE_TIMESTAMPS = [
    pd.Timestamp("2024-01-01 17:28:05"),
    pd.Timestamp("2024-01-01 18:28:05"),
    pd.Timestamp("2024-01-01 19:28:05"),
]


def test_clean_filters_negative_fares():
    uncleaned = pd.DataFrame(
        {
            "tpep_pickup_datetime": SAMPLE_TIMESTAMPS,
            "trip_distance": [1, -0.27, 3],
            "fare_amount": [-7, 2, 15],
        }
    )

    df = clean(uncleaned, 2024, 1)
    assert df.iloc[0]["fare_amount"] == 15


def test_clean_filters_zero_distance():
    uncleaned = pd.DataFrame(
        {
            "tpep_pickup_datetime": SAMPLE_TIMESTAMPS,
            "trip_distance": [1, 0, 3],
            "fare_amount": [0, 2, 15],
        }
    )

    df = clean(uncleaned, 2024, 1)
    assert df.iloc[0]["trip_distance"] == 3


def test_clean_filters_wrong_dates():
    uncleaned = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2025-01-01 17:28:05"),
                pd.Timestamp("2024-01-01 18:28:05"),
                pd.Timestamp("2024-02-02 19:28:05"),
            ],
            "trip_distance": [1, 2, 3],
            "fare_amount": [5, 10, 15],
        }
    )

    df = clean(uncleaned, 2024, 1)
    assert df.iloc[0]["tpep_pickup_datetime"] == pd.Timestamp("2024-01-01 18:28:05")


def test_clean_returns_empty_when_all_rows_invalid():
    uncleaned = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2025-03-01 17:28:05")],
            "trip_distance": [-1],
            "fare_amount": [-5],
        }
    )
    df = clean(uncleaned, 2024, 1)
    assert len(df) == 0
