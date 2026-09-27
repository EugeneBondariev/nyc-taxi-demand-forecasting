import pandas as pd

from src.clustering import build_zone_profiles


def _make_demand():
    return pd.DataFrame([
        {"PULocationID": z, "pickup_hour": h, "trip_count": float(z * 10 + h)}
        for z in [1, 2, 3]
        for h in range(24)
    ])


def test_build_zone_profiles_shape():
    profiles = build_zone_profiles(_make_demand())
    assert profiles.shape == (3, 24)


def test_build_zone_profiles_no_missing():
    profiles = build_zone_profiles(_make_demand())
    assert profiles.isna().sum().sum() == 0


def test_build_zone_profiles_values():
    profiles = build_zone_profiles(_make_demand())
    assert profiles.loc[1, 5] == 15.0   # zone 1, hour 5 → 1*10 + 5
    assert profiles.loc[2, 0] == 20.0   # zone 2, hour 0 → 2*10 + 0


def test_build_zone_profiles_fill_missing_hour():
    df = _make_demand()
    df = df[~((df["PULocationID"] == 1) & (df["pickup_hour"] == 3))]
    profiles = build_zone_profiles(df)
    assert profiles.loc[1, 3] == 0.0
