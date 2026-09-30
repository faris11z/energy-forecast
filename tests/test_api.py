"""API tests: health, prediction shape, and that no feature leaks a NaN into the model.

Run from the project root after src/train.py has written models/lgbm_load.txt:
    pytest -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402

from api import HOURS, MODEL, app, next_24_features  # noqa: E402
from features import build_features  # noqa: E402

client = app.test_client()


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_returns_24_values():
    assert MODEL is not None, "run src/train.py before the tests: no model on disk"
    response = client.get("/predict")
    assert response.status_code == 200

    payload = response.get_json()
    predictions = payload["predictions"]
    assert len(predictions) == HOURS
    assert all(isinstance(p["load_mw"], float) for p in predictions)
    # Timestamps must be UTC and strictly hourly.
    times = [p["time_utc"] for p in predictions]
    assert all(t.endswith("Z") for t in times)
    assert times == sorted(times)


def test_features_have_no_nans():
    """Every feature row handed to the model must be complete."""
    frame, start = next_24_features()
    features = build_features(frame)

    upcoming = features.loc[start : start + (HOURS - 1) * pd.Timedelta(hours=1)]
    assert len(upcoming) == HOURS
    assert not upcoming.isna().any().any(), upcoming.isna().sum()