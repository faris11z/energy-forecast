"""Promotion gate: give a freshly trained model the "production" alias only if it
beats the model that currently holds it, by the margin set in params.yaml.

Why a gate at all: retraining runs on a schedule, so a bad retrain must not be able
to take the API down with it. A model that is not clearly better is rejected, and the
reason is written to reports/promotion.json for the record.

The tracking URI comes from MLFLOW_TRACKING_URI (see common.tracking_uri), so the
same code runs against the local ./mlruns store and against DagsHub later.
"""

import json
from datetime import datetime, timezone

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.exceptions import MlflowException

from common import MODEL_PATH, REPORTS, load_params, tracking_uri

_PARAMS = load_params()["promote"]
REGISTERED_NAME = _PARAMS["registered_model_name"]
MIN_IMPROVEMENT_PCT = _PARAMS["min_improvement_pct"]
ALIAS = "production"


def log_candidate(metrics: dict) -> str:
    """Log the new booster as a new version of the registered model, with the same
    run carrying its metrics. Returns the new version number.

    The model is always logged, even when the gate rejects it: a rejected version is
    still a useful artefact to compare against later.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    mlflow.lightgbm.log_model(
        booster, name="model", registered_model_name=REGISTERED_NAME
    )
    return _latest_version()


def _latest_version() -> str:
    """Highest registered version.

    Sorting with order_by is not supported by every backend (the local file store
    rejects it), so pick the maximum in Python instead.
    """
    versions = mlflow.tracking.MlflowClient().search_model_versions(
        f"name='{REGISTERED_NAME}'"
    )
    return max(versions, key=lambda v: int(v.version)).version


def production_mae() -> tuple[float | None, str | None]:
    """Test MAE of the version behind the production alias.

    Returns (None, None) when nothing has been promoted yet, which is the normal
    state of a fresh project rather than an error.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        version = client.get_model_version_by_alias(REGISTERED_NAME, ALIAS).version
        run_id = client.get_model_version(REGISTERED_NAME, version).run_id
        return float(client.get_run(run_id).data.metrics["test_mae_mw"]), version
    except (MlflowException, KeyError, ValueError, IndexError):
        return None, None


def decide(new_mae: float, old_mae: float | None) -> tuple[bool, str]:
    """The gate itself, kept pure so it can be read (and tested) on its own."""
    if old_mae is None:
        return True, "no production model existed, promoting the first one"
    improvement = (old_mae - new_mae) / old_mae * 100
    if improvement >= MIN_IMPROVEMENT_PCT:
        return True, f"MAE {improvement:.2f}% better, above the {MIN_IMPROVEMENT_PCT}% gate"
    direction = "better" if improvement >= 0 else "worse"
    return (
        False,
        f"MAE is {abs(improvement):.2f}% {direction}, below the "
        f"{MIN_IMPROVEMENT_PCT}% gate, keeping the current production model",
    )


def main() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    metrics = json.loads((REPORTS / "train_metrics.json").read_text())
    new_mae = float(metrics["results"]["test"]["mae_mw"])

    old_mae, old_version = production_mae()
    promoted, reason = decide(new_mae, old_mae)

    result = {
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "new_model_mae_mw": new_mae,
        "production_mae_mw": old_mae,
        "production_version": old_version,
        "min_improvement_pct": MIN_IMPROVEMENT_PCT,
        "promoted": promoted,
        "reason": reason,
    }
    if old_mae is not None:
        result["improvement_pct"] = (old_mae - new_mae) / old_mae * 100

    with mlflow.start_run(run_name="promotion-candidate"):
        mlflow.log_params(
            {
                "registered_model_name": REGISTERED_NAME,
                "min_improvement_pct": MIN_IMPROVEMENT_PCT,
                "candidate_mae_mw": new_mae,
                "production_mae_mw": old_mae,
                "promoted": promoted,
            }
        )
        mlflow.log_metrics({"test_mae_mw": new_mae})
        result["candidate_version"] = log_candidate(metrics)

    if promoted:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            REGISTERED_NAME, ALIAS, result["candidate_version"]
        )
        result["promoted_version"] = result["candidate_version"]
    else:
        # The alias is deliberately left pointing at the old version.
        result["promoted_version"] = old_version

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "promotion.json").write_text(json.dumps(result, indent=2))

    verdict = "PROMOTED" if promoted else "REJECTED"
    served = (
        f"{old_mae:,.2f} MW (v{old_version})" if old_mae is not None else "none"
    )
    print(f"{verdict} v{result['candidate_version']}: {reason}")
    print(f"  candidate MAE={new_mae:,.2f} MW  production MAE={served}")
    print("Wrote reports/promotion.json")


if __name__ == "__main__":
    main()
