import pandas as pd

from src.data.events import _annual_events, add_nyc_event_feature, build_event_lookup


class TestBuildEventLookup:
    def test_annual_events_count(self):
        assert len(_annual_events(2024)) == 7

    def test_returns_set(self):
        lookup = build_event_lookup([2024])
        assert isinstance(lookup, set)
        assert len(lookup) > 0

    def test_nye_zones_in_lookup(self):
        lookup = build_event_lookup([2024])
        assert ("2024-12-31", 161) in lookup
        assert ("2024-12-31", 230) in lookup

    def test_non_event_date_not_in_lookup(self):
        lookup = build_event_lookup([2024])
        assert ("2024-06-15", 161) not in lookup

    def test_multi_year_lookup_size(self):
        one_year = build_event_lookup([2024])
        two_years = build_event_lookup([2024, 2025])
        assert len(two_years) > len(one_year)


class TestAddNycEventFeature:
    def test_shape(self):
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"pickup_hour_ts": ts, "PULocationID": [161] * 5, "trip_count": [10] * 5})
        lookup = build_event_lookup([2024])
        result = add_nyc_event_feature(df, lookup)
        assert "is_nyc_event" in result.columns
        assert result.shape[0] == 5

    def test_nye_flagged(self):
        ts = [pd.Timestamp("2024-12-31 20:00:00")]
        df = pd.DataFrame({"pickup_hour_ts": ts, "PULocationID": [161], "trip_count": [50]})
        lookup = build_event_lookup([2024])
        result = add_nyc_event_feature(df, lookup)
        assert result["is_nyc_event"].iloc[0] == 1

    def test_non_event_zero(self):
        ts = [pd.Timestamp("2024-06-15 12:00:00")]
        df = pd.DataFrame({"pickup_hour_ts": ts, "PULocationID": [161], "trip_count": [30]})
        lookup = build_event_lookup([2024])
        result = add_nyc_event_feature(df, lookup)
        assert result["is_nyc_event"].iloc[0] == 0
