import calendar
import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Zone IDs near major NYC event venues (TLC PULocationID taxonomy)
_TIMES_SQUARE = frozenset({161, 162, 163, 164, 230, 186})
_CENTRAL_PARK = frozenset({140, 141, 142, 161, 239})
_FIFTH_AVE    = frozenset({161, 162, 163, 230, 236, 237})
_WEST_VILLAGE = frozenset({68, 79, 113, 249})
_HUDSON       = frozenset({140, 141, 142, 236, 237})


def _first_weekday(year: int, month: int, weekday: int) -> date:
    """First occurrence of weekday (0=Mon … 6=Sun) in the given month."""
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7)


def _nth_thursday(year: int, month: int, n: int) -> date:
    return _first_weekday(year, month, 3) + timedelta(weeks=n - 1)


def _last_sunday(year: int, month: int) -> date:
    last = calendar.monthrange(year, month)[1]
    d = date(year, month, last)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def _annual_events(year: int) -> list[tuple[date, frozenset[int]]]:
    return [
        (date(year, 12, 31),               _TIMES_SQUARE),                    # NYE Times Square
        (date(year, 7, 4),                 _HUDSON | _TIMES_SQUARE),          # July 4th fireworks
        (date(year, 3, 17),                _FIFTH_AVE),                       # St. Patrick's Day Parade
        (date(year, 10, 31),               _WEST_VILLAGE),                    # Halloween Village Parade
        (_first_weekday(year, 11, 6),      _CENTRAL_PARK | _FIFTH_AVE),       # NYC Marathon (1st Sun Nov)
        (_nth_thursday(year, 11, 4),       _CENTRAL_PARK | _TIMES_SQUARE),    # Thanksgiving Parade
        (_last_sunday(year, 6),            _FIFTH_AVE | _WEST_VILLAGE),       # NYC Pride March
    ]


def build_event_lookup(years: list[int]) -> set[tuple[str, int]]:
    """Return (date_str, zone_id) pairs for all major NYC events across requested years."""
    lookup: set[tuple[str, int]] = set()
    for year in years:
        for event_date, zones in _annual_events(year):
            ds = event_date.strftime("%Y-%m-%d")
            for zone in zones:
                lookup.add((ds, zone))
    logger.info(f"NYC event lookup built: {len(lookup)} (date, zone) pairs for {len(years)} year(s)")
    return lookup


def add_nyc_event_feature(demand: pd.DataFrame, event_lookup: set[tuple[str, int]]) -> pd.DataFrame:
    """Add binary is_nyc_event: 1 if a major NYC event falls on this date near this zone."""
    date_str = demand["pickup_hour_ts"].dt.strftime("%Y-%m-%d")
    keys = pd.MultiIndex.from_arrays([date_str.values, demand["PULocationID"].values])
    lookup_idx = pd.MultiIndex.from_tuples(event_lookup)
    demand = demand.copy()
    demand["is_nyc_event"] = keys.isin(lookup_idx).astype(int)
    return demand
