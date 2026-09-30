"""Baseline: predict load with the value observed at the same hour one week ago.

Reasoning behind the choice: a one-week seasonal naive forecast is the standard
yardstick in load forecasting. If a model cannot beat it, the model is not worth
deploying. Lags are shifted before evaluation so no future value is ever used.
"""

import json

import pandas as pd

from common import REPORTS, load_params, load_raw, regression_metrics, split_time

# 7 days * 24 hours. 168h is also far enough back to stay leakage-free.
WEEK_LAG = load_params()["baseline"]["lag_hours"]

# Seasonal naive needs the previous week, so the earliest 168 rows have no forecast.
RESULTS = {}


def main() -> None:
    frame = load_raw()
    train, valid, test = split_time(frame)

    naive = frame["load_mw"].shift(WEEK_LAG)

    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        # The first WEEK_LAG rows have no seasonal-naive forecast, so pair up
        # only the rows that actually have a prediction to score.
        paired = pd.concat([part["load_mw"], naive.loc[part.index]], axis=1).dropna()
        RESULTS[name] = regression_metrics(paired.iloc[:, 0], paired.iloc[:, 1])

    for name, scores in RESULTS.items():
        pretty = "  ".join(f"{k}={v:,.2f}" for k, v in scores.items())
        print(f"baseline/{name:5s}  {pretty}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "baseline_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, indent=2)
    print("\nWrote reports/baseline_metrics.json")


if __name__ == "__main__":
    main()