"""Train a LightGBM day-ahead load model and compare it to the baseline.

Model selection uses the validation window only; the test window is touched once,
at the end, to produce the final numbers. That ordering is what keeps the reported
test score honest.
"""

import json

import lightgbm as lgb
import pandas as pd

from common import MODEL_PATH, REPORTS, load_raw, regression_metrics, split_time
from features import FEATURE_COLUMNS, TARGET, build_xy

# Small, regularised trees: this dataset is ~8000 rows, so capacity is not the
# bottleneck and overfitting is the real risk.
PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}

RESULTS: dict[str, dict[str, float]] = {}


def _subset(index, other):
    """Rows present in both frames; build_xy drops the lag warm-up rows."""
    common = index.intersection(other.index)
    return other.loc[common]


def evaluate(model: lgb.LGBMRegressor, features, target, name: str) -> dict[str, float]:
    scores = regression_metrics(target.values, model.predict(features))
    RESULTS[name] = scores
    return scores


def main() -> None:
    frame = load_raw()
    _, valid_raw, test_raw = split_time(frame)

    # Features are computed once on the full series. Every feature looks strictly
    # backwards, so slicing afterwards cannot leak future information.
    features, target = build_xy(frame)
    train_raw = frame.iloc[: frame.index.get_loc(valid_raw.index[0])]

    X_train, y_train = _subset(train_raw.index, features), _subset(
        train_raw.index, target
    )
    X_valid, y_valid = _subset(valid_raw.index, features), _subset(
        valid_raw.index, target
    )
    X_test, y_test = _subset(test_raw.index, features), _subset(
        test_raw.index, target
    )
    print(
        f"train={len(X_train)}  valid={len(X_valid)}  test={len(X_test)}  "
        f"features={len(FEATURE_COLUMNS)}"
    )

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"best_iteration={model.best_iteration_}")

    for name, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        scores = evaluate(model, X, y, name)
        print(f"lightgbm/{name:5s}  " + "  ".join(f"{k}={v:,.2f}" for k, v in scores.items()))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    print(f"\nSaved model to {MODEL_PATH}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"params": PARAMS, "results": RESULTS, "features": FEATURE_COLUMNS},
            handle,
            indent=2,
        )
    print("Wrote reports/train_metrics.json")

    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\nFeature importance (split count):")
    print(importance.to_string())


if __name__ == "__main__":
    main()