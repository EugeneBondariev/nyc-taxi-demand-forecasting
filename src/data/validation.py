import logging

import pandas as pd
import pandera.pandas as pa  # type: ignore[import-untyped]
from pandera.pandas import (  # type: ignore[import-untyped]
    Check,
    Column,
    DataFrameSchema,
)

logger = logging.getLogger(__name__)

demand_schema = DataFrameSchema(
    {
        "PULocationID": Column(int, Check.in_range(1, 265), coerce=True),
        "pickup_hour_ts": Column(pa.DateTime),
        "trip_count": Column(int, Check.ge(0), coerce=True),
        "pickup_hour": Column(int, Check.in_range(0, 23), coerce=True),
        "pickup_dow": Column(int, Check.in_range(0, 6), coerce=True),
        "pickup_week": Column(int, Check.in_range(1, 53), coerce=True),
        "temperature_2m": Column(float, nullable=True, coerce=True),
        "precipitation": Column(float, Check.ge(0), nullable=True, coerce=True),
        "snowfall": Column(float, Check.ge(0), nullable=True, coerce=True),
        "is_holiday": Column(int, Check.isin([0, 1]), coerce=True),
    },
)

taxi_schema = DataFrameSchema(
    {
        "tpep_pickup_datetime": Column(pa.DateTime),
        "trip_distance": Column(float, Check.gt(0), coerce=True),
        "fare_amount": Column(float, Check.gt(0), coerce=True),
        "PULocationID": Column(int, Check.in_range(1, 265), coerce=True),
    },
)


def validate_demand(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return demand_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        logger.error(f"Demand schema validation failed:\n{e.failure_cases}")
        raise


def validate_taxi(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return taxi_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        logger.error(f"Taxi schema validation failed:\n{e.failure_cases}")
        raise
