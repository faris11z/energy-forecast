"""Tests for the drift monitor.

These exist because the monitor had a real bug: it compared the last 30 days
against the whole training set, so every September looked like drift. The guard is
that the reference window is the same season, a year earlier.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitor import (  # noqa: E402
    REFERENCE_LAG_DAYS,
    per_column_drift,
    reference_window,
)


def _synthetic_frame(days: int = 800) -> pd.DataFrame:
    """Hourly frame with a strong seasonal cycle and a weekly cycle."""
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    day_of_year = index.dayofyear.to_numpy()
    load = 50_000 + 10_000 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    return pd.DataFrame(
        {
            "load_mw": load,
            "temperature_2m": load / 2_000,
            "wind_speed_10m": np.full(len(index), 10.0),
            "shortwave_radiation": np.full(len(index), 100.0),
        },
        index=index,
    )


def test_reference_window_is_the_same_season_a_year_earlier():
    frame = _synthetic_frame(days=800)
    current = frame.iloc[-30 * 24 :]

    reference, matched = reference_window(frame, current)

    assert matched is True, "800 days of history must allow a 365-day comparison"
    assert len(reference) >= len(current) * 0.9
    # The core assertion: same time of year, one year apart.
    gap = current.index[0] - reference.index[0]
    assert gap.days == REFERENCE_LAG_DAYS
    assert reference.index[0].month == current.index[0].month


def test_reference_window_falls_back_when_history_is_too_short():
    """Barely more than a year of data cannot support a one-year lag, and must say
    so rather than quietly returning a seasonally confounded window.

    380 days: the recent window starts at day 350, and 350 - 365 falls before the
    start of the series, so there is no matching season to compare against.
    """
    frame = _synthetic_frame(days=380)
    current = frame.iloc[-30 * 24 :]
    assert (current.index[0] - pd.Timedelta(days=REFERENCE_LAG_DAYS)) < frame.index[0]

    reference, matched = reference_window(frame, current)

    assert matched is False
    assert len(reference) > 0  # still returns something usable


def test_seasonal_drift_is_not_flagged_but_a_real_shift_is():
    """The regression this file exists for: comparing September against September
    must not drift, while a genuine level shift must."""
    frame = _synthetic_frame(days=800)
    current = frame.iloc[-30 * 24 :]
    reference, _ = reference_window(frame, current)

    same_season = per_column_drift(reference, current)
    assert not same_season["load_mw"]["drifted"], (
        "same-season comparison must not flag drift"
    )

    # Add a 15% level shift to the recent window: now it is real drift.
    shifted = current.copy()
    shifted["load_mw"] = shifted["load_mw"] * 1.15
    real_shift = per_column_drift(reference, shifted)
    assert real_shift["load_mw"]["drifted"], "a 15% level shift must be detected"
