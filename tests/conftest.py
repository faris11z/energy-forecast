"""Shared test setup: make the API tests run without touching the network.

The real /predict path calls Open-Meteo for the next 24 hours of weather. That is
correct in production but wrong in a test and unacceptable in CI, so the weather
call is replaced here by a deterministic frame with the same shape.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))


def _synthetic_weather(start: datetime, hours: int) -> pd.DataFrame:
    """Plausible hourly weather for `hours` steps from `start`, no network."""
    index = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    daylight = (hour - 6) / 12
    return pd.DataFrame(
        {
            "temperature_2m": 10.0 + 8.0 * np.clip(daylight, 0, 2),
            "wind_speed_10m": 12.0,
            "shortwave_radiation": 400.0 * np.clip(daylight, 0, 1),
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def offline_weather(monkeypatch):
    """Swap the live weather call for the deterministic version in every test."""
    import api

    monkeypatch.setattr(api, "fetch_forecast_weather", _synthetic_weather)
