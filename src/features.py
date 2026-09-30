"""Turn raw load + weather into model features.

Leakage rule used throughout: a feature for hour t may only use information that
exists at forecast time. Lags are therefore at least 24h and every rolling window
is shifted by one hour, so nothing derived from the target leaks backwards.
"""

import holidays
import pandas as pd

# Every column the model sees. Listed in one place so train and the API cannot drift.
FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "month",
    "is_holiday",
    "lag_24h",
    "lag_168h",
    "rolling_mean_7d",
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

TARGET = "load_mw"


def _german_holidays(index: pd.DatetimeIndex) -> pd.Series:
    """1 on a nationwide German public holiday, else 0. Holidays shift load
    noticeably and are calendar-known, so this is safe to use."""
    calendar = holidays.country_holidays("DE", years=range(2015, 2031))
    return pd.Series(index.isin(calendar), index=index).astype(int)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. All are known in advance for the forecast horizon."""
    out = pd.DataFrame(index=frame.index)
    out["hour"] = frame.index.hour
    out["weekday"] = frame.index.dayofweek
    out["month"] = frame.index.month
    out["is_holiday"] = _german_holidays(frame.index)
    return out


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Target-derived features, all shifted far enough back to be leakage-free."""
    load = frame[TARGET]
    out = pd.DataFrame(index=frame.index)
    out["lag_24h"] = load.shift(24)
    out["lag_168h"] = load.shift(168)
    # shift(1) first so the window ends at t-1 and never includes the target hour.
    out["rolling_mean_7d"] = load.shift(1).rolling(168, min_periods=168).mean()
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix (all FEATURE_COLUMNS) aligned to the frame's UTC index."""
    weather = frame[
        ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    ]
    features = pd.concat(
        [add_calendar_features(frame), add_lag_features(frame), weather], axis=1
    )
    return features[FEATURE_COLUMNS]


def build_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target with incomplete rows dropped.

    Rows go missing only at the very start of the series (lag warm-up), because
    lags always look backwards.
    """
    features = build_features(frame)
    target = frame[TARGET]
    keep = features.notna().all(axis=1) & target.notna()
    return features.loc[keep], target.loc[keep]