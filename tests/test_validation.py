import pandas as pd
import pandera.pandas as pa  # type: ignore[import-untyped]
import pytest

from src.validation import validate_demand, validate_taxi


def _valid_demand() -> pd.DataFrame:
    return pd.DataFrame({
        "PULocationID": [161],
        "pickup_hour_ts": pd.to_datetime(["2024-01-01 18:00:00"]),
        "trip_count": [50],
        "pickup_hour": [18],
        "pickup_dow": [2],
        "pickup_week": [1],
        "temperature_2m": [10.0],
        "precipitation": [0.0],
        "snowfall": [0.0],
        "is_holiday": [0],
    })


def _valid_taxi() -> pd.DataFrame:
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2024-01-01 18:00:00"]),
        "trip_distance": [2.5],
        "fare_amount": [12.0],
        "PULocationID": [161],
    })


def test_valid_demand_passes():
    result = validate_demand(_valid_demand())
    assert len(result) == 1


def test_invalid_zone_id_fails():
    df = _valid_demand()
    df["PULocationID"] = 999
    with pytest.raises(pa.errors.SchemaErrors):
        validate_demand(df)


def test_negative_trip_count_fails():
    df = _valid_demand()
    df["trip_count"] = -1
    with pytest.raises(pa.errors.SchemaErrors):
        validate_demand(df)


def test_invalid_is_holiday_fails():
    df = _valid_demand()
    df["is_holiday"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate_demand(df)


def test_valid_taxi_passes():
    result = validate_taxi(_valid_taxi())
    assert len(result) == 1


def test_zero_trip_distance_fails():
    df = _valid_taxi()
    df["trip_distance"] = 0.0
    with pytest.raises(pa.errors.SchemaErrors):
        validate_taxi(df)
