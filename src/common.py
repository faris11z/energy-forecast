"""Shared helpers: params, the time-based split and the metrics.

Kept in one place so ingest, training, promotion, monitoring and the API all read
the same configuration and agree on what "test set" means.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "load_weather.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "lgbm_load.txt"
REPORTS = ROOT / "reports"
PARAMS_PATH = ROOT / "params.yaml"

# 30 days of hourly data each, taken from the end backwards, never at random.
# These are only fallbacks; params.yaml is the real source.
TEST_DAYS = 30
VALID_DAYS = 30


def load_params() -> dict:
    """Read params.yaml. Falls back to the hard-coded defaults above if the file
    is missing, so a single script still runs on its own."""
    if not PARAMS_PATH.exists():
        return {"data": {"test_window_days": TEST_DAYS, "valid_window_days": VALID_DAYS}}
    with open(PARAMS_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tracking_uri() -> str:
    """Where MLflow reads and writes.

    MLFLOW_TRACKING_URI switches this between the local ./mlruns store and DagsHub,
    which is why the value comes from the environment and not from the code.
    """
    # MLflow 3.x puts the file store in maintenance mode and raises unless this is
    # set. The brief asks for a local ./mlruns store, so opt in. Migrating to
    # sqlite:///mlflow.db is the documented long-term fix.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return os.environ.get("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())


def load_raw() -> pd.DataFrame:
    """The ingested CSV, indexed by UTC with a guaranteed clean hourly index."""
    frame = pd.read_csv(RAW_CSV, index_col=0, parse_dates=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.sort_index()


def load_processed() -> pd.DataFrame:
    """The feature table written by the DVC `features` stage, UTC index."""
    frame = pd.read_csv(PROCESSED_CSV, index_col=0, parse_dates=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.sort_index()


def split_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train, then the 30 days before that as validation,
    and the most recent 30 days as test. Returns (train, valid, test)."""
    params = load_params()
    test_days = params["data"]["test_window_days"]
    valid_days = params["data"]["valid_window_days"]
    test = frame.iloc[-test_days * 24 :]
    valid = frame.iloc[-(test_days + valid_days) * 24 : -test_days * 24]
    train = frame.iloc[: -(test_days + valid_days) * 24]
    return train, valid, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and MAPE in percent. MAPE is reported but read with care:
    load never approaches zero, so it stays numerically safe here."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / y_true)) * 100),
    }