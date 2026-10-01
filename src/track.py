"""Log two MLflow runs: one for the baseline, one for the LightGBM model.

Where they go is decided by common.tracking_uri(), which reads MLFLOW_TRACKING_URI.
With it unset that is the local ./mlruns store; pointed at DagsHub it is the remote
server. Nothing else in this file decides, so the same command works in both places.

Run after src/baseline.py and src/train.py have written reports/*.json.
"""

import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from common import MODEL_PATH, ROOT, load_raw, split_time, tracking_uri

REPORTS = ROOT / "reports"


def log_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Log per-split metrics as test_mae_mw, test_rmse_mw, ... so the two runs
    can be compared column by column in the MLflow UI."""
    for split, scores in metrics.items():
        mlflow.log_metrics({f"{split}_{key}": value for key, value in scores.items()})


def main() -> None:
    uri = tracking_uri()
    # MLflow 3.x puts the ./mlruns file store in maintenance mode and refuses to
    # write unless this flag is set. It is harmless against a remote server, but
    # it is only *needed* for the local store, so keep it off DagsHub.
    if uri.startswith("file:"):
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    print(f"tracking URI: {uri}")
    experiment = mlflow.set_experiment("energy-forecast-day-ahead")

    baseline = json.loads((REPORTS / "baseline_metrics.json").read_text())
    train = json.loads((REPORTS / "train_metrics.json").read_text())

    # Run 1: the yardstick the model has to beat.
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({"method": "seasonal_naive", "lag_hours": 168})
        mlflow.set_tag("model", "baseline")
        log_metrics(baseline)

    # Run 2: LightGBM with its hyperparameters, its feature list and the booster.
    frame = load_raw()
    train_raw, valid_raw, test_raw = split_time(frame)
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(
            {**train["params"], "features": ",".join(train["features"])}
        )
        mlflow.log_params(
            {
                "n_train_rows": len(train_raw),
                "n_valid_rows": len(valid_raw),
                "n_test_rows": len(test_raw),
            }
        )
        mlflow.set_tag("model", "lightgbm")
        log_metrics(train["results"])

        booster = lgb.Booster(model_file=str(MODEL_PATH))
        mlflow.lightgbm.log_model(booster, name="model")

        importance = pd.Series(
            booster.feature_importance(importance_type="gain"),
            index=booster.feature_name(),
        ).sort_values(ascending=False)
        importance.to_csv(REPORTS / "feature_importance.csv")

    print(f"mlflow runs written to {uri}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    for _, row in runs.iterrows():
        print(f"  {row['tags.mlflow.runName']:10s} test_mae_mw={row['metrics.test_mae_mw']:,.1f}")


if __name__ == "__main__":
    main()