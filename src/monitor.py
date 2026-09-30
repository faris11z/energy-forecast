"""Drift monitoring: has the world moved away from what the model was trained on?

Two things are checked:
  * Data drift  - the feature and load distributions, last 30 days vs the training
    window, per column.
  * Prediction error - is the model actually still as accurate as when it shipped?

Evidently renders the HTML report. The pass/fail decision uses a per-column
Kolmogorov-Smirnov test, which is the same test Evidently's DataDriftPreset applies
to numeric columns by default. Computing it here rather than digging the numbers
out of the HTML keeps reports/drift.json stable enough for the CI gate to depend on.

The threshold lives in params.yaml (drift.drifted_column_share).
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import REPORTS, load_params, load_raw, split_time

_PARAMS = load_params()["drift"]
DRIFTED_SHARE_THRESHOLD = _PARAMS["drifted_column_share"]
P_VALUE = _PARAMS["p_value"]
RECENT_DAYS = _PARAMS["recent_window_days"]

TARGET = "load_mw"
# What the brief asks to watch, plus the three weather drivers of load.
MONITORED_COLUMNS = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    TARGET,
]

MODEL = REPORTS.parent / "models" / "lgbm_load.txt"


def per_column_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, dict]:
    """KS test per column. `p_value` below the threshold means the two samples
    look like they came from different distributions."""
    report = {}
    for column in MONITORED_COLUMNS:
        stat, p_value = ks_2samp(
            reference[column].dropna(), current[column].dropna()
        )
        report[column] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            # A p-value under the threshold is the signal we alert on.
            "drifted": bool(p_value < P_VALUE),
        }
    return report


def prediction_error(frame: pd.DataFrame) -> dict[str, float]:
    """MAE of the served model on the most recent window, next to the MAE it
    achieved on the test window when it was trained.

    Needs the model; skipped when it is not on disk so the report still works.
    """
    if not MODEL.exists():
        return {"available": False, "mae_mw": None}

    import lightgbm as lgb

    from common import load_params as _load_params
    from features import TARGET as _TARGET, build_features

    feature_columns = [c for c in build_features(frame).columns]
    booster = lgb.Booster(model_file=str(MODEL))
    recent = frame.iloc[-RECENT_DAYS * 24 :]
    features = build_features(recent)
    mask = features.notna().all(axis=1)
    mae = float(
        np.mean(np.abs(booster.predict(features[mask]) - recent[_TARGET][mask]))
    )
    del _load_params
    return {"available": True, "mae_mw": round(mae, 2), "window_days": RECENT_DAYS}


def write_html(reference: pd.DataFrame, current: pd.DataFrame) -> bool:
    """Evidently's visual report. Returns False if the HTML could not be built, so
    the JSON signal is still produced either way."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as error:
        print(f"evidently not available ({error}), skipping reports/drift.html")
        return False

    # Evidently 0.7 takes the frames directly; the explicit Dataset objects are
    # only needed when a DataDefinition has to be attached per column.
    current_data = Dataset.from_pandas(
        current.reset_index(), data_definition=DataDefinition(numerical_columns=MONITORED_COLUMNS)
    )
    reference_data = Dataset.from_pandas(
        reference.reset_index(), data_definition=DataDefinition(numerical_columns=MONITORED_COLUMNS)
    )
    snapshot = Report([DataDriftPreset()]).run(
        current_data=current_data, reference_data=reference_data
    )
    snapshot.save_html(str(REPORTS / "drift.html"))
    print(f"Wrote {REPORTS / 'drift.html'}")
    return True


def main() -> None:
    frame = load_raw()
    train, _, _ = split_time(frame)
    # The most recent window overlaps the test set by design: that is exactly the
    # period the model is judged on, so it is the right thing to watch.
    current = frame.iloc[-RECENT_DAYS * 24 :]

    columns = per_column_drift(train, current)
    drifted = [name for name, result in columns.items() if result["drifted"]]
    share = len(drifted) / len(columns)
    detected = share > DRIFTED_SHARE_THRESHOLD

    result = {
        "drift_detected": bool(detected),
        "drifted_column_share": round(float(share), 4),
        "threshold": DRIFTED_SHARE_THRESHOLD,
        "drifted_columns": drifted,
        "columns": columns,
        "reference_rows": int(len(train)),
        "current_rows": int(len(current)),
        "prediction_error": prediction_error(frame),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "drift.json").write_text(json.dumps(result, indent=2))
    write_html(train, current)

    print(f"drift_detected={detected}  drifted {len(drifted)}/{len(columns)} columns")
    for name, stats in columns.items():
        flag = "DRIFT" if stats["drifted"] else "ok   "
        print(f"  {flag} {name:22s} p={stats['p_value']:.4f}")
    print(f"Wrote {REPORTS / 'drift.json'}")


if __name__ == "__main__":
    main()
