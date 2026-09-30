"""API tests: health, prediction shape, and that no feature leaks a NaN into the model.

Run from the project root after src/train.py has written models/lgbm_load.txt:
    pytest -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from api import HOURS, MODEL, app, forecast  # noqa: E402

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
    """Every feature row handed to the model must be complete.

    Walks the horizon the same way /predict does, because a future row's
    rolling_mean_7d only exists after the previous hour has been predicted.
    """
    assert MODEL is not None, "run src/train.py before the tests: no model on disk"
    _, features = forecast()

    assert len(features) == HOURS
    nan_counts = features.isna().sum()
    assert not features.isna().any().any(), nan_counts[nan_counts > 0].to_dict()