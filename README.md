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
| **LightGBM (10 features)** | **1,438.54** | **1,888.55** | **2.64** |
| improvement | **-22.7%** | **-18.7%** | **-23.5%** |

Trained on 2 years (17,518 rows). More history was not cosmetic: the drift monitor
compares the last 30 days against the same 30 days a year earlier, which is
impossible with only one year of data.

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

## Architecture

```mermaid
flowchart LR
  A[energy-charts<br/>Open-Meteo] -->|src/ingest.py| B[(data/raw<br/>DVC tracked)]
  B -->|src/features.py| C[(data/processed)]
  C -->|src/train.py| D[LightGBM]
  B --> E[src/baseline.py]
  D --> F[src/evaluate.py<br/>metrics.json]
  E --> F
  D --> G[src/track.py<br/>MLflow run]
  D --> H[src/promote.py]
  H -->|gate: 1% better MAE| I[(MLflow registry<br/>energy-load-forecaster<br/>alias production)]
  I -->|models:/...@production| J[app/api.py<br/>/health /predict]
  B --> K[src/monitor.py]
  J --> K
  K -->|drift.json| L{{weekly retrain<br/>.github/workflows}}
  L --> D
```

Data is versioned by DVC, every run is tracked by MLflow, and the API never reads a
model file directly: it reads the `production` alias, so a promotion changes what is
served without a redeploy. Drift is the feedback loop that triggers the next retrain.

## One-time manual setup

Nothing below is automated yet, and none of it can be done from CI without you.

1. **Create the DagsHub repo** (dags hub.com, free for personal use). Copy the repo
   URL. `dvc remote add origin <url>` and `dvc push` as below.
2. **Point DVC at DagsHub** (run in the project root, on your machine):
   ```powershell
   dvc remote add origin https://dagshub.com/<user>/<repo>.git
   dvc remote list          # confirm origin exists
   dvc push                 # first upload of data/
   ```
   Retrieve an access token from DagsHub under Settings -> Access Tokens.
3. **Add GitHub repository secrets** (Settings -> Secrets and variables -> Actions):
   - `DAGSHUB_TOKEN` - the DagsHub token from step 2
   - `DAGSHUB_REMOTE_URL` - e.g. `https://dagshub.com/<user>/<repo>.dvc`. The token
     is deliberately *not* part of this URL, because `.dvc/config` is committed to
     git. `retrain.yml` reads this to run `dvc remote add`, which is what lets a
     fresh CI checkout fetch versioned data instead of silently re-downloading from
     the public APIs. Without it that workflow still runs, on fresh data.
   - `MLFLOW_TRACKING_URI` - e.g. `https://dagshub.com/<user>/<repo>.mlflow`
   - `HF_TOKEN` - a Hugging Face write token, needed only for deploys
4. **Point MLflow at DagsHub locally** to test the same code path:
   ```powershell
   $env:MLFLOW_TRACKING_URI = "https://dagshub.com/<user>/<repo>.mlflow"
   python src\promote.py     # no code change needed
   Remove-Item Env:MLFLOW_TRACKING_URI
   ```
5. **Create the Hugging Face Space**: spaces.huggingface.co, Docker SDK, port 7860,
   then run the "Deploy to Hugging Face Space" workflow with `space` set to
   `<your-username>/<repo-name>`.
6. **First promotion**: `python src\promote.py`. Until this runs, the registry has no
   `production` alias and the API falls back to `models/lgbm_load.txt`.

## Running it all locally

```powershell
dvc repro                 # ingest -> features -> train -> evaluate
python src\promote.py     # registry + 1% gate
python src\monitor.py     # reports/drift.html + reports/drift.json
python app\api.py         # http://localhost:8000
pytest -q                 # 6 tests, offline, no external calls
dvc dag                   # the graph above, from the real pipeline
```

## Known limitations

- `params.yaml` is a whole-file DVC dep, so any param change reruns every stage
  including the network download. `dvc repro -s <stage>` is the workaround.
- `drift_detected` follows **accuracy**, not distribution. September 2026 really was
  warmer than September 2025 and the model absorbed it (+0.5% MAE), so paging on the
  distribution shift would be noise. `distribution_drift` is reported alongside it.
- The seasonal reference needs `data.ingest_days >= 730`. With a single year of
  history the monitor falls back to the training set and sets `seasonally_matched:
  false` rather than reporting a meaningless number.
- `dvc push`, the retrain workflow and the Space deploy were never executed against
  a real remote; they are syntax-checked only.
- **The tested Python is not the deployed Python.** Everything here was verified on
  3.14, but CI and the Space image are 3.11. `requirements.txt` bounds the majors so
  a new breaking version cannot appear, but no 3.11 run has actually happened yet.
  The first `ci.yml` run is the real test of this.
- The Space image installs the full `requirements.txt`, including `mlflow` and
  `pytest`, so it is larger than it needs to be. `mlflow` is required (the API
  prefers the registry alias), but `pytest` is not and could be dropped.

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
