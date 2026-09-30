"""Evaluate stage: put the baseline and the model side by side in metrics.json.

DVC tracks this file, so `dvc diff` shows exactly how a retrain moved the numbers
without anyone having to remember where the reports went.
"""

import json

from common import REPORTS, ROOT, load_params


def _improvement(new: dict, old: dict) -> dict[str, float]:
    """How much better `new` is than `old`, in percent. Positive means better."""
    return {
        key: (old[key] - new[key]) / old[key] * 100 for key in ("mae_mw", "rmse_mw", "mape_pct")
    }


def main() -> None:
    baseline = json.loads((REPORTS / "baseline_metrics.json").read_text())
    model = json.loads((REPORTS / "train_metrics.json").read_text())

    metrics = {
        "split": {
            "test": {
                "baseline": baseline["test"],
                "lightgbm": model["results"]["test"],
                "improvement_pct": _improvement(
                    model["results"]["test"], baseline["test"]
                ),
            },
            "valid": {
                "baseline": baseline["valid"],
                "lightgbm": model["results"]["valid"],
                "improvement_pct": _improvement(
                    model["results"]["valid"], baseline["valid"]
                ),
            },
        },
        "params": load_params(),
    }

    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    test = metrics["split"]["test"]
    print("test window, model vs baseline")
    print(
        f"  baseline  MAE={test['baseline']['mae_mw']:>10,.2f}  "
        f"RMSE={test['baseline']['rmse_mw']:>10,.2f}  MAPE={test['baseline']['mape_pct']:.2f}%"
    )
    print(
        f"  lightgbm  MAE={test['lightgbm']['mae_mw']:>10,.2f}  "
        f"RMSE={test['lightgbm']['rmse_mw']:>10,.2f}  MAPE={test['lightgbm']['mape_pct']:.2f}%"
    )
    print(f"  improvement  MAE {test['improvement_pct']['mae_mw']:+.1f}%")
    print("\nWrote metrics.json")


if __name__ == "__main__":
    main()
