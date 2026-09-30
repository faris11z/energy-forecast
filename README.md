# energy-forecast

Day-ahead German electricity load forecasting: a LightGBM model served behind a
small Flask API, tracked with MLflow, and measured against a seasonal-naive baseline.

All timestamps are UTC. The model predicts hourly gross load in MW.

## Results

Test window = the most recent 30 days of the ingested data (720 hours). Split is by
time only, never random.

| model | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|---------:|----------:|---------:|
| baseline: load from the same hour one week earlier | 1,860.29 | 2,322.99 | 3.45 |
| **LightGBM (10 features)** | **1,529.96** | **2,006.93** | **2.81** |
| improvement | **-17.8%** | **-13.6%** | **-18.6%** |

The baseline is the yardstick: a model that cannot beat "the same hour last week" is
not worth deploying.

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src\ingest.py     # download load + weather into data\raw\load_weather.csv
python src\baseline.py   # seasonal-naive metrics -> reports\baseline_metrics.json
python src\train.py      # train LightGBM     -> models\lgbm_load.txt + reports\
python src\track.py      # 2 MLflow runs      -> mlruns\
python app\api.py        # serve on http://localhost:8000
pytest -q                # 3 API tests
```

Order matters: `train.py` needs `ingest.py` to have run, and `api.py` / the tests
need `models\lgbm_load.txt` to exist.

## The pieces

| file | what it does |
|------|--------------|
| `src/ingest.py` | Downloads hourly load and weather into `data/raw/load_weather.csv` |
| `src/common.py` | The time-based split, the metrics, and the shared paths |
| `src/features.py` | Turns raw columns into the 10 model features |
| `src/baseline.py` | Seasonal-naive benchmark, writes its metrics to `reports/` |
| `src/train.py` | Trains LightGBM, saves the booster, prints the comparison |
| `src/track.py` | Logs one MLflow run for the baseline and one for LightGBM |
| `app/api.py` | Flask API: `/health` and `/predict` |
| `tests/test_api.py` | 3 pytest tests |

## Data sources

Both are free and need no API key, so there is nothing to keep out of git:

- **Load**: [energy-charts.info](https://api.energy-charts.info) `total_power`,
  field "Load (incl. self-consumption)", MW. Bundesnetzagentur / SMARD, CC BY 4.0.
- **Weather**: [Open-Meteo](https://open-meteo.com) archive and forecast APIs,
  averaged over Berlin, Frankfurt and Munich. German public holidays come from the
  `holidays` package.

`data/`, `mlruns/`, `models/` and `.venv/` are gitignored, so run `src/ingest.py`
and `src/train.py` first after a fresh clone.

## Design notes worth knowing

- **No leakage.** Lags are 24h and 168h; the 7-day rolling mean is `.shift(1)` first
  so its window ends at t-1. Calendar and holiday features are all known in advance.
- **Weather at forecast time** comes from the Open-Meteo *forecast* API, not the
  archive, so the API is using NWP data like a real day-ahead run would. Training
  history is reanalysis, so the two differ slightly.
- **The 24-hour forecast is recursive.** Each prediction is appended to the history
  before the next step, because hour t+1's `rolling_mean_7d` needs hour t's value.
  The whole block cannot be built in one shot.
- **MLflow 3.x** refuses to write to `./mlruns` unless `MLFLOW_ALLOW_FILE_STORE=true`,
  which `src/track.py` sets for you. Migrating to `sqlite:///mlflow.db` is the
  documented long-term fix.

## Docker

```bash
docker build -t energy-forecast-api .
docker run -p 8000:8000 energy-forecast-api
```

The build context must contain `data/raw/load_weather.csv` and
`models/lgbm_load.txt`, because both are gitignored. **The image was written but
never built** — Docker was not available in the session where this was written.
