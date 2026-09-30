"""Flask API for day-ahead German load forecasts.

  GET /health   -> liveness plus whether a model is loaded
  GET /predict  -> the next 24 hourly load predictions, UTC

The 24 future hours are predicted one at a time and each prediction is appended to
the history before the next step. That recursive step is needed because lag_24h and
lag_168h of hour t+2 must reflect what we predicted for hour t+1; a single batch call
would leave those lags empty.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import requests
from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import MODEL_PATH, load_raw  # noqa: E402
from features import TARGET, build_features  # noqa: E402

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CITIES = {
    "berlin": (52.52, 13.41),
    "frankfurt": (50.11, 8.68),
    "munich": (48.14, 11.58),
}
HOURS = 24

app = Flask(__name__)

# Loaded once at import so /predict does not re-read the model on every call.
MODEL = lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None


def fetch_forecast_weather(start: datetime, hours: int) -> pd.DataFrame:
    """Weather for the future hours, averaged over the same 3 cities.

    These are NWP forecast values, which is what a real day-ahead forecast has
    available. Ingested history is reanalysis, so the two sources differ slightly.
    """
    params = {
        "latitude": ",".join(str(lat) for lat, _ in CITIES.values()),
        "longitude": ",".join(str(lon) for _, lon in CITIES.values()),
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": (start + timedelta(hours=hours)).strftime("%Y-%m-%d"),
    }
    # Open-Meteo answers 503 fairly often under repeated calls, so retry a couple
    # of times before giving up on a forecast.
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(FORECAST_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"weather API failed after 3 attempts: {last_error}")

    frames = []
    for city in payload:
        hourly = city["hourly"]
        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"], utc=True),
                    "temperature_2m": hourly["temperature_2m"],
                    "wind_speed_10m": hourly["wind_speed_10m"],
                    "shortwave_radiation": hourly["shortwave_radiation"],
                }
            ).set_index("time")
        )
    # axis=0 stacks the cities, then groupby averages them per hour. Concat on
    # axis=1 would only place the cities side by side and average nothing.
    return pd.concat(frames).groupby(level=0).mean()


def next_24_features(hours: int = HOURS, now: datetime | None = None) -> pd.DataFrame:
    """Feature rows for the next `hours` hourly steps, ready for the model.

    Exposed separately so it can be unit tested for NaNs without going over HTTP.
    """
    now = now or datetime.now(timezone.utc)
    # The next full hour boundary: we forecast complete hours, not partial ones.
    start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    history = load_raw()
    future = fetch_forecast_weather(start, hours)
    future = future.reindex(
        pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    )
    # Placeholder target so the frame has the right columns; filled in per step.
    # A missing hour here would silently become a NaN feature, so fail loudly.
    weather_columns = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    if future[weather_columns].isna().any().any():
        raise ValueError("weather API returned an incomplete forecast window")
    future[TARGET] = float("nan")
    frame = pd.concat([history, future[weather_columns + [TARGET]]])
    return frame, start


def forecast(hours: int = HOURS, now: datetime | None = None):
    """Walk the horizon one hour at a time.

    Returns (predictions, features_used). Both are needed together: the feature row
    for step k+1 can only be built once step k's prediction has been written back
    into the history, because rolling_mean_7d ends at t-1. So the whole 24-hour
    feature block cannot be built in one go, and any test on the features has to
    follow this same path.
    """
    frame, start = next_24_features(hours=hours, now=now)

    predictions, rows = [], []
    for step in range(hours):
        step_time = start + timedelta(hours=step)
        features = build_features(frame)
        row = features.loc[[step_time]]
        rows.append(row)
        value = float(MODEL.predict(row[features.columns])[0])
        # Feed the prediction back so the next step's lags are defined.
        frame.loc[step_time, TARGET] = value
        predictions.append(
            {
                "time_utc": step_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "load_mw": round(value, 1),
            }
        )
    return predictions, pd.concat(rows)


def predict_next(hours: int = HOURS, now: datetime | None = None) -> list[dict]:
    """The next `hours` load values only."""
    return forecast(hours=hours, now=now)[0]


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "model_path": str(MODEL_PATH),
            "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    ), 200


@app.get("/predict")
def predict():
    if MODEL is None:
        return jsonify({"error": "no model found, run src/train.py first"}), 503
    try:
        predictions = predict_next()
    except (requests.RequestException, ValueError, KeyError) as error:
        return jsonify({"error": f"forecast failed: {error}"}), 502
    return jsonify(
        {
            "horizon_hours": HOURS,
            "timezone": "UTC",
            "predictions": predictions,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)