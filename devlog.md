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

## FINAL REPORT

**Session start:** 2026-09-30T14:40:26Z. **Wrap-up:** ~2026-09-30T15:58Z.
Overran the 60-minute box by roughly 15 minutes, spent on the weather-merge bug that
would otherwise have left a wrong results table in the README.

### Phases completed (each one actually run)

| phase | status | evidence |
|-------|--------|----------|
| 1 Data check | **done, verified** | 8,758 rows, 0 NaN/dupes/gaps; range logged |
| 2 Baseline | **done, verified** | reports/baseline_metrics.json |
| 3 Features + model | **done, verified** | models/lgbm_load.txt, -17.8% MAE vs baseline |
| 4 Tracking | **done, verified** | 2 runs in ./mlruns |
| 5 API | **done, verified via test client only** | 3/3 pytest pass; real HTTP bind NOT verified |
| 6 Docker | **written, NOT built** | Docker absent in WSL |
| 7 Wrap-up | **done** | README.md + this file |

### Metrics (test set = most recent 30 days, 720 h)

| model | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|---------:|----------:|---------:|
| baseline (same hour 1 week earlier) | 1,860.29 | 2,322.99 | 3.45 |
| **LightGBM** | **1,529.96** | **2,006.93** | **2.81** |
| delta | -17.8% | -13.6% | -18.6% |

Train 3,288.38 / Valid 2,230.84 MAE for the baseline; LightGBM train 1,232.11,
valid 1,579.45 (see reports/*.json). Model beats the baseline, so the "stop and
explain why it failed" condition never triggered.

### Files created (all new; nothing pre-existing was edited)

`src/ingest.py`, `src/common.py`, `src/features.py`, `src/baseline.py`,
`src/train.py`, `src/track.py`, `app/api.py`, `tests/test_api.py`,
`requirements.txt`, `Dockerfile`, `.dockerignore`, `README.md`, `devlog.md`,
`.gitignore`. Branch `agent-run-1`, 2 commits, working tree clean. `main` untouched.

### Every assumption and decision I made

1. **The folder was empty.** No `src/ingest.py` existed, so I wrote it. Nothing
   pre-existing was overwritten at any point.
2. **Load source.** energy-charts `total_power`, "Load (incl. self-consumption)".
   The `/price` endpoint returns EUR/MWh, not MW. No key, no credentials.
3. **Load is 34.5-83.1 GW, not the 30-70 GW the brief expected.** I left it and
   logged it: the field includes on-site solar self-consumption, so midday winter
   peaks exceed 70 GW. Switching `LOAD_SERIES` to "Residual load" would match the
   brief's range instead.
4. **Weather** averaged over 3 cities, not a single point.
5. **Split** = test = last 30 days, valid = 30 days before, train = the rest, by
   index position only. No shuffling anywhere.
6. **`src/common.py` is an extra file** beyond the brief's list, so baseline, train
   and the API share one definition of the split and the metrics.
7. **Leakage rule** = lags >= 24h, rolling mean `.shift(1)`, weather features are
   forecast values for future hours.
8. **Recursive 24h prediction.** Not a choice so much as a consequence of the lags.
9. **`is_holiday` gets 0 splits** in LightGBM. Left as is: 10 holidays in 7,150 rows
   is too rare to earn a split and weekday already carries the signal.
10. **MLflow** needed `MLFLOW_ALLOW_FILE_STORE=true` to keep the brief's ./mlruns.
11. **Deleted `mlruns/`** once, to drop the pre-bug-fix runs and regenerate.
12. **git identity** set repo-local to `agent-run-1 <agent@localhost>`; your global
    config was untouched.
13. **`models/` is gitignored** (276 KB artifact, regenerable by `train.py`).
14. `LD_LIBRARY_PATH` for libgomp is a WSL-only workaround, not needed on Windows.

### Blockers and your exact next steps

1. **Docker is untested** - it does not exist in WSL. Run
   `docker build -t energy-forecast-api .` on Windows. Expect to create
   `data/raw/load_weather.csv` and `models/lgbm_load.txt` first (both gitignored).
2. **The API was never bound to a real socket** - this sandbox silently refuses.
   Run `python app\api.py` and `curl http://localhost:8000/predict`. This is the one
   claim in the README I could not verify myself.
3. **Nothing is proven on your own venv.** Every number above comes from Python
   3.14 in WSL. Recreate `.venv` on Windows and re-run the five commands in the
   README to confirm the metrics reproduce.
4. **Open-Meteo returns 503 now and then.** The API retries 3x and returns HTTP 502
   rather than NaN predictions. For anything more than a demo, cache the weather
   fetch and consider a proper commercial weather feed.
5. **Decide the load series** (see assumption 3) before you quote these numbers to
   anyone: gross-with-self-consumption or residual load.
6. **Model is a single direct 24h-ahead regressor.** It does not predict each hour
   directly from a 24h horizon the way a real day-ahead run would, it walks the
   horizon recursively. The test MAE therefore describes recursive 24-step accuracy,
   not 24 independent 24h-ahead forecasts. Worth fixing before production.

---

# Session 2 - MLOps layer (agent-run-2)

## Session start

- **Start time (UTC): 2026-09-30T15:31:41Z**
- Timebox: 60 min. No new phase started after 16:21:41Z. Last 10 min = wrap-up only.
- Starting point: branch `agent-run-1` @ b19d188, ML layer already working
  (3/3 pytest green, LightGBM test MAE 1,529.96 MW vs baseline 1,860.29 MW).
- Goal this session: DVC pipeline, model registry + promotion gate, API serving
  from the registry, drift monitoring, CI, scheduled retraining, HF Space config.

## Environment carry-over from session 1

WSL sandbox, not Windows. `python3 -m venv --without-pip` + `get-pip.py` was needed,
and `libgomp.so.1` is missing from the system image, so every command runs with
`LD_LIBRARY_PATH=$PWD/.venv/lib`. Also: a real HTTP bind does not work in this
sandbox (session 1 finding), so the API is exercised through the Flask test client.

## Phase 1 - DVC PIPELINE (verified by running)

dvc 3.67.1 installed, `dvc init` done, `dvc.yaml` + `params.yaml` written.
All four stages reproduce: `dvc repro` runs ingest -> features -> train -> evaluate
and writes `dvc.lock`, then a second run reports "Data and pipelines are up to date".
Pipeline output is unchanged from session 1: test MAE 1,529.96 MW.

`params.yaml` is now the single source of truth. src/common.py gained `load_params()`
and `tracking_uri()`; train.py, baseline.py, ingest.py and features.py all read it.
Verified: metrics are byte-identical before and after the refactor, and 3/3 pytest
still pass, so the params change did not alter results.

**Selective reruns: partially working, logged as a limitation.** Declaring
`params.yaml` as a whole-file dep means any param change reruns all four stages,
including the network download. I tried the per-key form
(`- params.yaml: [model.num_leaves]`) and the nested-list YAML did not validate
against DVC's schema; I reverted to the form that verifiably works rather than
leave a broken pipeline. Confirmed behaviour: `num_leaves: 31 -> 48` reran
ingest, features, train and evaluate. `dvc repro -s <stage>` is the workaround.

**No remote configured, as instructed.** data/ and the artifacts are tracked by DVC;
git only holds dvc.yaml, dvc.lock and .dvc/config.

## Phase 2 - REGISTRY + PROMOTION GATE (verified by running, both paths)

`src/promote.py`, registered name `energy-load-forecaster`, alias `production`,
gate = 1% better test MAE (params.promote.min_improvement_pct). Tracking URI comes
from `common.tracking_uri()`: `MLFLOW_TRACKING_URI` if set, else local ./mlruns, so
switching to DagsHub needs no code change.

| case | candidate MAE | production MAE | outcome |
|------|--------------|----------------|---------|
| no production model existed | 1,529.96 | none | **PROMOTED v2** |
| deliberately bad model (1 tree, 300 rows) | 6,561.45 | 1,529.96 (v2) | **REJECTED v3**, alias stayed on v2 |

The bad model was a real retrain of a deliberately crippled configuration, not a
faked number, and the good model was restored afterwards. reports/promotion.json
records both decisions with the reason string.

Three real problems fixed while building it: MLflow's file-store gate (now set in
common.tracking_uri so all modules share it), `mlflow.lightgbm.Booster` does not
exist (it is `lightgbm.Booster`), and the file store rejects
`order_by=["version desc"]` so the latest version is picked in Python instead.

## Phase 3 - API SERVES FROM THE REGISTRY (verified by running)

`app/api.py` loads `models:/energy-load-forecaster@production` and falls back to the
local model file when the registry has no alias, so a fresh clone still works.
/health now returns `model_source` describing exactly what is being served
(`registry:...@production=v2` or `file:lgbm_load.txt`). The server binds to `$PORT`
(7860 on a HF Space, 8000 locally). 3/3 pytest pass.

## Phase 4 - DRIFT MONITORING (verified by running)

`src/monitor.py`, evidently 0.7.23. Last 30 days vs the training window, per column,
for temperature, wind, radiation and load, plus the recent-window prediction error.
Wrote reports/drift.html (3.9 MB) and reports/drift.json. Threshold lives in
params.yaml (drift.drifted_column_share = 0.25).

Result: `drift_detected=true`, 3/4 columns drifted, recent MAE 1,490.58 MW.

**This alert is a false positive and you should treat it as one.** The reference
window is the whole training set, which spans all four seasons, while the current
window is 30 days in September. Temperature and radiation are *supposed* to differ.
The fix is a seasonal reference (same 30 days last year) rather than the full
training set; I did not implement it because the timebox was gone. Until then the
alert fires every September, so it will not wake you up for a real problem.

I used Evidently for the HTML and scipy's KS test for the drift.json decision, on
purpose: KS is the same test DataDriftPreset applies to numeric columns by default,
but computing it directly keeps the JSON stable enough for the CI gate to depend on.
Three API mismatches along the way (numeric_ vs numerical_, `definition` vs
`data_definition`, and Dataset construction) cost about 10 minutes.

## Phases 5, 6, 7 - WORKFLOWS AND DEPLOY (written, syntax-checked only)

- `.github/workflows/ci.yml` - lint, fixture build, pytest, dvc.yaml validation,
  docker build without push.
- `.github/workflows/retrain.yml` - weekly cron + workflow_dispatch, dvc pull
  (continue-on-error), dvc repro, promote, monitor, then a hard failure if
  `drift_detected and not promoted`, plus report artifacts.
- `.github/workflows/deploy.yml`, `space/Dockerfile`, `space/README.md` - HF Space
  on port 7860, manual dispatch only, HF_TOKEN from secrets.

All four YAML files parse with `yaml.safe_load` and the Space frontmatter was
extracted and parsed. Nothing was pushed, logged into, or deployed, as instructed.

I did not reproduce the CI job locally as a whole. What I did verify is the part
that usually breaks: `tests/make_fixture_data.py` plus `pytest` on a throwaway copy
of the repo with no data/ and no model/ -> 3 passed with no network access.

## FINAL REPORT (session 2)

**Start 15:31:41Z, wrap-up ~17:03Z. Overran the 60-minute box by about 30 minutes.**
Cause: `pip install dvc evidently` took ~25 minutes on the Windows-mounted drive and
ran in the background for most of the session, and one command hung for 53 minutes
before returning. I kept starting work rather than idling, but the overrun is real.

| phase | status |
|-------|--------|
| 1 DVC pipeline | **verified by running** (dvc repro, 4 stages, up-to-date on rerun) |
| 2 Registry + promotion gate | **verified by running**, both paths (promote + reject) |
| 3 API from registry | **verified by running** (3/3 pytest) |
| 4 Drift monitoring | **verified by running** (drift.html + drift.json written) |
| 5 CI workflow | **written, syntax-checked only** |
| 6 Retrain workflow | **written, syntax-checked only** |
| 7 HF Space deploy | **written, syntax-checked only** |
| 8 Wrap-up | done |

**`dvc repro`: passes.** 4 stages, `dvc.lock` written, second run reports up to date,
metrics unchanged from session 1 (test MAE 1,529.96 MW).
**`pytest -q`: 3 passed**, both against the real data and against a clean offline copy.

**Promotion gate, both cases observed:**
- no production model -> PROMOTED v2 (1,529.96 MW)
- bad model 6,561.45 MW vs production 1,529.96 MW -> REJECTED v3, alias stayed on v2

**Files created:** params.yaml, dvc.yaml, dvc.lock, .dvc/, src/promote.py,
src/evaluate.py, src/monitor.py, tests/conftest.py, tests/make_fixture_data.py,
.github/workflows/{ci,retrain,deploy}.yml, space/{Dockerfile,README.md}
**Files changed:** src/common.py (load_params, tracking_uri, params-driven split),
src/features.py (params-driven lags, __main__), src/train.py (params + processed
artifact), src/baseline.py, src/ingest.py, app/api.py (registry load, $PORT),
.gitgitignore
**Untracked from git (files kept on disk):** reports/train_metrics.json,
reports/baseline_metrics.json - DVC now owns them. 3 commits on `agent-run-2`.

**Decisions and assumptions**
1. `params.yaml` is the only place to change splits, hyperparams, the gate and the
   drift threshold; every script reads it through common.load_params().
2. DVC tracks data/ and the model artifacts; git keeps only dvc.yaml/dvc.lock/config.
   No remote configured, as instructed.
3. A rejected model is still logged as a version, so it can be compared later. The
   alias simply does not move.
4. The API falls back to the local model file so a fresh clone works before anything
   is promoted. /health says which source is live.
5. Evidently renders the report; the drift.json decision uses scipy KS, the same
   test Evidently applies by default, so CI depends on JSON rather than HTML.
6. The drift reference window is the whole training set. That is what makes the
   current alert a seasonal false positive; logged above, not fixed.
7. CI never touches the network: conftest replaces the weather call and
   make_fixture_data.py builds synthetic data. Verified offline in a throwaway copy.
8. secrets are referenced only as `${{ secrets.* }}`; no login, push or deploy was
   attempted.
9. WSL carry-over: every command needs `LD_LIBRARY_PATH=$PWD/.venv/lib` for libgomp.

**My manual checklist, in order**
1. Create the DagsHub repo; `dvc remote add origin <url>`; `dvc push`.
2. Add GitHub secrets: `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`, `HF_TOKEN`.
3. `python src/promote.py` once locally, so a `production` alias exists.
4. Create the HF Space (Docker, port 7860), then run the deploy workflow.
5. Run the retrain workflow once via workflow_dispatch and watch the artifacts.
6. Only then enable the schedule and trust the alerts, after fixing the seasonal
   drift reference in src/monitor.py.
