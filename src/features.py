"""Turn raw load + weather into model features.

Leakage rule used throughout: a feature for hour t may only use information that
exists at forecast time. Lags are therefore at least 24h and every rolling window
is shifted by one hour, so nothing derived from the target leaks backwards.
"""

import holidays
import pandas as pd

from common import load_params

_PARAMS = load_params()["features"]
LAG_HOURS = _PARAMS["lag_hours"]
ROLLING_WINDOW = _PARAMS["rolling_window_hours"]
WEATHER_COLUMNS = _PARAMS["weather_columns"]

# Every column the model sees. Listed in one place so train and the API cannot drift.
# The lag names are generated from params, so params stays the only place to change.
FEATURE_COLUMNS = (
    ["hour", "weekday", "month", "is_holiday"]
    + [f"lag_{hours}h" for hours in LAG_HOURS]
    + [f"rolling_mean_{ROLLING_WINDOW // 24}d"]
    + list(WEATHER_COLUMNS)
)

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
    for hours in LAG_HOURS:
        out[f"lag_{hours}h"] = load.shift(hours)
    # shift(1) first so the window ends at t-1 and never includes the target hour.
    out[f"rolling_mean_{ROLLING_WINDOW // 24}d"] = (
        load.shift(1).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    )
    return out


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix (all FEATURE_COLUMNS) aligned to the frame's UTC index."""
    weather = frame[WEATHER_COLUMNS]
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


def main() -> None:
    """Entry point for the DVC `features` stage: materialise the feature table so
    downstream stages depend on a real file instead of recomputing it."""
    from common import PROCESSED_CSV, load_raw

    features, target = build_xy(load_raw())
    table = features.copy()
    table[TARGET] = target

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED_CSV)
    print(f"Wrote {len(table)} rows x {len(table.columns)} cols to {PROCESSED_CSV}")


if __name__ == "__main__":
    main()