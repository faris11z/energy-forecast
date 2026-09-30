# Devlog — agent-run-1

Agent work session log. All timestamps UTC.

## Session start

- **Start time (UTC): 2026-09-30T14:40:26Z**
- Timebox: 60 min total. No new phases started after T+50 (2026-09-30T15:30Z).
- Plan: phases 1-7 as specified in the task brief.

## Discovery (T+0)

The project directory `/mnt/d/Projects/energy-forecast` was **completely empty** at start:

- No `src/ingest.py` (the brief said it already exists)
- No `data/`, `app/`, `tests/`
- Not a git repository

Decision (no user available): treat this as a greenfield build, create the missing
scaffolding myself, and keep every step minimal and explainable. The phases themselves
run exactly as specified. Any dataset I have to create is logged explicitly below.
## Phase 1 - DATA CHECK (T+5 to T+16, verified by running)

`src/ingest.py` did **not** exist, so I wrote it. The project folder was empty.

Decisions:
- Load source: `api.energy-charts.info/total_power`, field **"Load (incl. self-consumption)"**
  in MW. Free, no key, CC BY 4.0 (Bundesnetzagentur / SMARD). The sibling `/price`
  endpoint returns EUR/MWh, not MW, so it is the wrong endpoint for this.
- Weather: `archive-api.open-meteo.com` for 3 cities (Berlin, Frankfurt, Munich),
  averaged. Germany has no single national weather point.
- Window: 365 days ending yesterday, so 8760 hours.

Result: **8758 rows**, `2025-09-30 00:00Z` .. `2026-09-29 21:00Z`.
Load min 34,455 / mean 57,084 / max 83,149 MW. Temp -12.9 .. 39.8 C.
0 NaNs, 0 duplicates, 0 missing hours inside the range.

**Deviation from the brief, logged not "fixed":** the brief expected load of
roughly 30,000-70,000 MW. The bulk of the distribution is inside that
(p25 49,075, median 57,086, p75 63,736) but the winter daytime peaks reach 83 GW.
Reason: the SMARD field counts on-site solar self-consumption, which adds up to
~20 GW at midday. That is a definitional difference, not broken data, so I left it
alone rather than switching series. If you want the classic 30-70 GW number, switch
the field to "Residual load" in `LOAD_SERIES`.

## Phase 2 - BASELINE (verified by running)

`src/baseline.py`: prediction = load at the same hour 168h (one week) earlier.
Time split from `src/common.py`: test = most recent 30 days, validation = the 30 days
before that, train = everything older. Never random.

| split | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|----------|-----------|----------|
| train | 3,288.38 | 4,702.61 | 5.75 |
| valid | 2,230.84 | 2,792.84 | 4.50 |
| **test** | **1,860.29** | **2,322.99** | **3.45** |

First run gave NaN on the train split because the first 168 rows have no lag.
Fixed by scoring only rows that have a prediction.

## Phase 3 - FEATURES + MODEL (verified by running)

`src/features.py` (10 features) and `src/train.py` (LightGBM 4.7.0).
Leakage rules: lags are 24h and 168h, the 7-day rolling mean is `.shift(1)` first,
and calendar/holiday features are all known in advance.
`is_holiday` uses nationwide German holidays from the `holidays` package.

train=7150 valid=720 test=720 features=10, best_iteration=100.

| model | test MAE (MW) | test RMSE (MW) | test MAPE (%) |
|-------|---------------|----------------|---------------|
| baseline (seasonal naive) | 1,860.29 | 2,322.99 | 3.45 |
| **LightGBM** | **1,559.92** | **2,004.86** | **2.90** |

LightGBM wins on all three metrics: -16.2% MAE, -13.7% RMSE, -15.9% MAPE.
Top features by split count: lag_24h, weekday, rolling_mean_7d, temperature_2m.
`is_holiday` has 0 splits, which is expected: 10 holidays in 7150 rows is too rare
for the tree to split on it, and weekday already carries most of that signal.

## Correction found during phase 5 (important)

While writing the API tests I found a bug in the weather merge in **both**
`src/ingest.py` and `app/api.py`:

```python
pd.concat(frames, axis=1).groupby(level=0).mean()   # wrong
```

With one frame per city, `axis=1` puts the three cities *side by side* as 9 columns.
`groupby(level=0).mean()` then averages nothing, because there are no duplicate index
values. The CSV had 10 columns instead of 4, and since `features.py` selects columns
by name, the model was silently using **Berlin only** (the first of each triple).

Fixed to `pd.concat(frames).groupby(level=0).mean()` (stack, then average).
Re-ran ingest -> baseline -> train -> track on the corrected data.
LightGBM test MAE improved from 1,559.92 to **1,529.96** MW, MAPE 2.90 -> **2.81**.

I also deleted the `mlruns/` directory I had created 5 minutes earlier and re-ran
`src/track.py`, so the store holds exactly the two runs the brief asks for with the
corrected numbers. That is a generated, gitignored artifact, not a source file.

## Phase 4 - TRACKING (verified by running)

`src/track.py`, local `./mlruns` (file:// URI), experiment
`energy-forecast-day-ahead`, two runs:

| run | test MAE (MW) | test RMSE (MW) | test MAPE (%) |
|-----|---------------|----------------|---------------|
| baseline | 1,860.30 | 2,322.99 | 3.45 |
| lightgbm | 1,529.96 | 2,006.93 | 2.81 |

Params logged, the LightGBM booster logged as a model artifact, feature importance
written to `reports/feature_importance.csv`.

**Blocker hit:** MLflow 3.16 refuses to write to a file store and raises
MlflowException. Fixed by setting `MLFLOW_ALLOW_FILE_STORE=true` inside `track.py`
with a comment pointing at the sqlite migration.

## Phase 5 - API (verified by running, with one caveat)

`app/api.py`, Flask, `GET /health` and `GET /predict` (next 24 hourly values).
Model is loaded once at import. Each prediction is fed back into the history before
the next step, because `rolling_mean_7d` for hour t+1 ends at t and needs the
value of t. Because of that, the 24-row feature block *cannot* be built in one pass:
only the first row is complete before any prediction exists. My first version of the
NaN test built all 24 rows at once and correctly failed with 23 NaNs in
`rolling_mean_7d`. The fix was not a fudge: the test now walks the same step-by-step
path the API uses (`forecast()` returns both predictions and the features used).

`pytest -q` -> **3 passed**.

**Two real problems found and fixed:**
1. Open-Meteo intermittently returns 503, which produced an empty weather block and
   NaN features. Added 3 retries with backoff and an explicit error if the window
   comes back incomplete, so a missing hour can never become a silent NaN.
2. The 3-city averaging bug described above.

**Caveat, not verified:** I could not get a real HTTP bind to work in this WSL
sandbox - `python app/api.py` starts, produces no output and never listens on 8000.
So /health and /predict are verified through the Werkzeug **test client** inside
pytest (200 on /health, exactly 24 values from /predict, no NaN features), not
through a real socket. Please run `python app\api.py` on Windows to confirm.

## Phase 6 - DOCKER (written, NOT built)

`Dockerfile` + `.dockerignore` + `requirements.txt` written. **Never built or run**:
`docker` is not installed in this WSL distro, so the image is untested. It copies the
gitignored `data/raw/load_weather.csv` and `models/lgbm_load.txt` from the build
context, so those two files must exist locally before `docker build`.

## Environment notes (WSL, not Windows)

- `python3 -m venv` produced a venv without pip (no `python3-venv` package). Worked
  around with `python3 -m venv --without-pip` + `get-pip.py`.
- `libgomp.so.1` is missing in this WSL image, so `import lightgbm` failed. Fetched
  the Ubuntu `libgomp1` .deb, extracted it into `.venv/lib/` and ran with
  `LD_LIBRARY_PATH=$PWD/.venv/lib`. **On your Windows venv you will not need this**;
  it is a WSL-only workaround. `.venv/lib/libgomp.so.1*` is inside the gitignored
  venv, so nothing was committed.
