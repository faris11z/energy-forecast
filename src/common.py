"""Shared helpers: the time-based split and the metrics, used by every script.

Kept in one place so baseline, training and the API all report the same numbers
and agree on what "test set" means.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "load_weather.csv"
MODEL_PATH = ROOT / "models" / "lgbm_load.txt"
REPORTS = ROOT / "reports"

# 30 days of hourly data each, taken from the end backwards, never at random.
TEST_DAYS = 30
VALID_DAYS = 30


def load_raw() -> pd.DataFrame:
    """The ingested CSV, indexed by UTC with a guaranteed clean hourly index."""
    frame = pd.read_csv(RAW_CSV, index_col=0, parse_dates=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.sort_index()


def split_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: train, then the 30 days before that as validation,
    and the most recent 30 days as test. Returns (train, valid, test)."""
    test = frame.iloc[-TEST_DAYS * 24 :]
    valid = frame.iloc[-(TEST_DAYS + VALID_DAYS) * 24 : -TEST_DAYS * 24]
    train = frame.iloc[: -(TEST_DAYS + VALID_DAYS) * 24]
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