"""Build data/raw/load_weather.csv: hourly German electricity load + weather.

Sources (both free, no API key, no credentials needed):
  * Load  : api.energy-charts.info/total_power -> "Load (incl. self-consumption)" in MW.
            Bundesnetzagentur / SMARD, CC BY 4.0.
  * Weather: archive-api.open-meteo.com -> hourly temperature_2m, wind_speed_10m,
            shortwave_radiation, averaged over 3 German cities.

Everything is indexed by UTC. Re-run this script to refresh the dataset.
"""

from pathlib import Path

import pandas as pd
import requests

# 365 days of hourly data gives 8760 rows, the ~8700 the project expects.
DAYS = 365

# One representative city per region; Germany has no single "national" weather point.
CITIES = {
    "berlin": (52.52, 13.41),
    "frankfurt": (50.11, 8.68),
    "munich": (48.14, 11.58),
}

LOAD_URL = "https://api.energy-charts.info/total_power"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
LOAD_SERIES = "Load (incl. self-consumption)"

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "load_weather.csv"


def fetch_load(start: str, end: str) -> pd.Series:
    """Hourly gross load in MW, indexed by UTC."""
    params = {"bzn": "DE-LU", "start": start, "end": end}
    response = requests.get(LOAD_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    wanted = None
    for entry in payload["production_types"]:
        if entry["name"] == LOAD_SERIES:
            wanted = entry["data"]
            break
    if wanted is None:
        raise ValueError(f"{LOAD_SERIES!r} not found in API response")

    # unix_seconds are hourly and start at :00, so this is a safe timestamp.
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    return pd.Series(wanted, index=index, name="load_mw")


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Hourly weather averaged over CITIES, indexed by UTC."""
    latitudes = [lat for lat, _ in CITIES.values()]
    longitudes = [lon for _, lon in CITIES.values()]
    params = {
        "latitude": ",".join(map(str, latitudes)),
        "longitude": ",".join(map(str, longitudes)),
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    # Multiple coordinates come back as a list of per-city responses.
    frames = []
    for city in payload:
        hourly = city["hourly"]
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly["time"], utc=True),
                "temperature_2m": hourly["temperature_2m"],
                "wind_speed_10m": hourly["wind_speed_10m"],
                "shortwave_radiation": hourly["shortwave_radiation"],
            }
        )
        frames.append(frame.set_index("time"))

    # Average cities so one weather column still covers the whole country.
    # axis=0 stacks the cities, then groupby averages them per hour. Concat on
    # axis=1 would only place the cities side by side and average nothing.
    return pd.concat(frames).groupby(level=0).mean()


def main() -> None:
    # End yesterday so every weather hour has a completed observation.
    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=DAYS - 1)
    print(f"Fetching {start.date()} .. {end.date()} (UTC)")

    load = fetch_load(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    weather = fetch_weather(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    frame = load.to_frame().join(weather, how="inner")
    # Renumber to a clean UTC index after the inner join dropped boundary hours.
    frame.index.name = "time"

    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW)
    print(f"Wrote {len(frame)} rows to {RAW}")
    print(frame.describe().loc[["min", "mean", "max"]])


if __name__ == "__main__":
    main()