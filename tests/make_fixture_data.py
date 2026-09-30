"""Build a small offline dataset and a model from it, for CI and for local smoke
tests. Nothing here touches the network.

The real pipeline downloads from energy-charts and Open-Meteo. CI must not do that
(it would be slow, flaky, and would hammer two free public APIs), so this writes a
synthetic series with the same schema and trains on it. The numbers it produces are
meaningless; only the shape and the plumbing matter.

Usage:  python tests/make_fixture_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HOURS = 24 * 120  # 120 days is enough for the 30/30 day split plus lag warm-up


def build_frame() -> pd.DataFrame:
    """A daily + weekly + weather shaped load series with a fixed seed."""
    rng = np.random.default_rng(seed=42)
    index = pd.date_range("2026-01-01", periods=HOURS, freq="h", tz="UTC")

    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    temperature = 8.0 + 9.0 * np.cos(2 * np.pi * (index.dayofyear.to_numpy() - 20) / 365)
    radiation = 300.0 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)

    # Weekly shape, a morning/evening ramp, a cold-weather bump and noise.
    load = (
        55_000
        + 6_000 * np.sin(2 * np.pi * (hour - 7) / 24)
        - 8_000 * (weekday >= 5)
        - 900 * (temperature - 8)
        + rng.normal(0, 700, HOURS)
    )
    return pd.DataFrame(
        {
            "load_mw": np.clip(load, 25_000, 90_000),
            "temperature_2m": temperature + rng.normal(0, 1.0, HOURS),
            "wind_speed_10m": 12.0 + rng.normal(0, 3.0, HOURS),
            "shortwave_radiation": radiation,
        },
        index=index,
    )


def main() -> None:
    from common import MODEL_PATH, RAW_CSV, load_raw, split_time
    from features import build_xy
    import lightgbm as lgb

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    build_frame().to_csv(RAW_CSV)
    print(f"Wrote {HOURS} synthetic rows to {RAW_CSV}")

    frame = load_raw()
    _, valid_raw, _ = split_time(frame)
    features, target = build_xy(frame)
    train_mask = features.index < valid_raw.index[0]
    X_train, y_train = features.loc[train_mask], target.loc[train_mask]

    model = lgb.LGBMRegressor(objective="regression", n_estimators=30, verbose=-1)
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    print(f"Trained a fixture model on {len(X_train)} rows -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
