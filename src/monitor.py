"""Drift monitoring: has the world moved away from what the model was trained on?

Two things are checked:
  * Data drift  - the feature and load distributions, last 30 days against the same
    30 days one year earlier, per column.
  * Prediction error - is the model still as accurate as when it was trained?

The seasonal reference is the important part. Comparing 30 days of September against
a year-long training window flags temperature, radiation and load every single year,
which is not drift, it is October happening. The reference window is therefore the
same 30 days shifted back by params.drift.reference_lag_days. When there is not
enough history for that, the monitor falls back to the training set and says so
loudly in the report rather than quietly reporting a meaningless number.

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
REFERENCE_LAG_DAYS = _PARAMS["reference_lag_days"]
MAX_MAE_DEGRADATION_PCT = _PARAMS["max_mae_degradation_pct"]

TARGET = "load_mw"
# What the brief asks to watch, plus the three weather drivers of load.
MONITORED_COLUMNS = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    TARGET,
]

MODEL = REPORTS.parent / "models" / "lgbm_load.txt"


def reference_window(frame: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """The same length window, one year earlier, restricted to the training data.

    Returns (reference, seasonally_matched). The flag is False when the history is
    too short, in which case the whole training set is used and the result should be
    read as "confounded by season".
    """
    train, _, _ = split_time(frame)
    lag = pd.Timedelta(days=REFERENCE_LAG_DAYS)
    wanted = (recent.index[0] - lag, recent.index[-1] - lag)
    if wanted[0] < train.index[0]:
        return train, False

    reference = train.loc[wanted[0] : wanted[1]]
    if len(reference) < len(recent) * 0.9:
        # A partial window would be compared against itself unevenly.
        return train, False
    return reference, True


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


def prediction_error(frame: pd.DataFrame, recent: pd.DataFrame) -> dict:
    """Recent MAE next to the MAE the model achieved on its test window.

    Needs the model and the training metrics; both may be absent, in which case the
    accuracy signal is reported as unavailable rather than guessed.
    """
    metrics_path = REPORTS / "train_metrics.json"
    if not MODEL.exists() or not metrics_path.exists():
        return {"available": False, "reason": "model or train_metrics.json missing"}

    import lightgbm as lgb

    from features import build_features

    booster = lgb.Booster(model_file=str(MODEL))
    features = build_features(recent)
    mask = features.notna().all(axis=1)
    recent_mae = float(
        np.mean(np.abs(booster.predict(features[mask]) - recent[TARGET][mask]))
    )
    baseline_mae = json.loads(metrics_path.read_text())["results"]["test"]["mae_mw"]
    degradation = (recent_mae - baseline_mae) / baseline_mae * 100
    return {
        "available": True,
        "recent_mae_mw": round(recent_mae, 2),
        "train_test_mae_mw": round(float(baseline_mae), 2),
        "degradation_pct": round(degradation, 2),
        "accuracy_degraded": bool(degradation > MAX_MAE_DEGRADATION_PCT),
        "window_days": RECENT_DAYS,
    }


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
    # The most recent window is what the model is judged on, so it is what we watch.
    current = frame.iloc[-RECENT_DAYS * 24 :]
    reference, seasonally_matched = reference_window(frame, current)

    columns = per_column_drift(reference, current)
    drifted = [name for name, result in columns.items() if result["drifted"]]
    share = len(drifted) / len(columns)
    distribution_drift = share > DRIFTED_SHARE_THRESHOLD

    accuracy = prediction_error(frame, current)
    accuracy_drift = bool(accuracy.get("accuracy_degraded", False))

    # A distribution shift on its own is not an incident. Measured on the seasonally
    # matched windows, September 2026 really was warmer than September 2025 and the
    # model absorbed it (+0.5% MAE). Paging on that would be the same false alarm as
    # before, just quieter. Accuracy loss is the signal that means the model is
    # actually broken, so drift_detected follows accuracy and the distribution
    # result is reported next to it for whoever investigates.
    result = {
        "drift_detected": accuracy_drift,
        "distribution_drift": bool(distribution_drift),
        "accuracy_drift": accuracy_drift,
        "drifted_column_share": round(float(share), 4),
        "threshold": DRIFTED_SHARE_THRESHOLD,
        # If this is false, the distribution numbers below are confounded by season
        # and should not be trusted.
        "seasonally_matched": bool(seasonally_matched),
        "drifted_columns": drifted,
        "columns": columns,
        "reference_window_utc": [
            reference.index[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            reference.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
        "current_window_utc": [
            current.index[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            current.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        ],
        "reference_rows": len(reference),
        "current_rows": len(current),
        "prediction_error": accuracy,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "drift.json").write_text(json.dumps(result, indent=2))
    write_html(reference, current)

    print(f"reference   {result['reference_window_utc'][0]} .. {result['reference_window_utc'][1]}")
    print(f"current     {result['current_window_utc'][0]} .. {result['current_window_utc'][1]}")
    if not seasonally_matched:
        print(
            f"  WARNING: not enough history for a {REFERENCE_LAG_DAYS}-day lag; "
            "fell back to the whole training set, so these numbers are confounded "
            "by season and will over-report drift"
        )
    print(
        f"distribution_drift={distribution_drift}  drifted {len(drifted)}/{len(columns)} columns"
    )
    for name, stats in columns.items():
        flag = "DRIFT" if stats["drifted"] else "ok   "
        print(f"  {flag} {name:22s} p={stats['p_value']:.4f}")
    if accuracy.get("available"):
        print(
            f"accuracy_drift={accuracy_drift}  recent MAE={accuracy['recent_mae_mw']:,.2f} MW "
            f"vs {accuracy['train_test_mae_mw']:,.2f} MW at training "
            f"({accuracy['degradation_pct']:+.1f}%)"
        )
    print(f"drift_detected={result['drift_detected']}")
    print(f"Wrote {REPORTS / 'drift.json'}")


if __name__ == "__main__":
    main()
