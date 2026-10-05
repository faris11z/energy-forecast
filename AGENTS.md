# AGENTS.md

Guidance for agents working in this repository. Written for an agent with no prior
context here.

## What this is

Day-ahead electricity load forecasting for the German/Luxembourg grid zone (DE-LU).
Hourly load in MW from `energy-charts.info`, hourly weather from the Open-Meteo
archive, a LightGBM regressor, and a Flask API that serves the model behind the MLflow
`production` alias.

Around 17,500 hourly rows from 730 days of history. The pipeline is DVC-tracked and
runs identically locally and in CI.

## Environment

**The virtualenv is Linux/WSL-only.** `.venv/` has `bin/`, no `Scripts/`, and
`pyvenv.cfg` points at `/usr/bin/python3.14`. PowerShell cannot activate it.
Do not use the Windows Python installation for this repository. Use WSL bash.

**Before running Python, pytest, or Python-based project commands, establish the
project environment with:**

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib" PATH="$PWD/.venv/bin:$PATH"
```

LightGBM does not bundle `libgomp.so.1`. Its shared object has a hard link to it:

```
$ ldd .venv/lib/python3.14/site-packages/lightgbm/lib/lib_lightgbm.so | grep gomp
	libgomp.so.1 => not found
```

`.venv/lib/libgomp.so.1` is what makes that resolve. Omit `LD_LIBRARY_PATH` and
`import lightgbm` raises `OSError: libgomp.so.1: cannot open shared object file`.
This is not optional.

## Commands that work

```bash
dvc repro                    # ingest -> features -> train -> evaluate
pytest -q                     # 15 tests, fully offline
ruff check src app tests      # what CI runs
python src/track.py           # log baseline + lightgbm runs
python src/promote.py         # 1% MAE promotion gate
python src/monitor.py         # writes reports/drift.json
```

`src/*.py` import each other by bare name, so run them from inside `src/`
(`cd src && ../.venv/bin/python track.py`).

## Architecture

Four DVC stages in `dvc.yaml`: `ingest` -> `features` -> `train` -> `evaluate`.
`params.yaml` is the single source of truth; every script and stage reads it.

- Registry name `energy-load-forecaster`, alias `production` (`params.yaml` `promote`)
- `common.tracking_uri()` is the contract: it reads `MLFLOW_TRACKING_URI` and falls
  back to the local `./mlruns` file store. **Every module that touches MLflow must
  route through it.** See the `track.py` bug below.
- Time-based splits only. Test set is the most recent 30 days, validation the 30 days
  before that. Nothing is shuffled.
- `app/api.py` prefers `models:/energy-load-forecaster@production` and falls back to
  `models/lgbm_load.txt`, reporting which it used on `/health`.

## Constraints that look like mistakes

Each of these was a deliberate decision. Changing them reintroduces a bug that was
already found and fixed.

**`drift_detected` follows accuracy, not distribution.** September 2026 really was
warmer than September 2025; `temperature_2m` and `load_mw` genuinely drifted
(p < 0.005), and the model's accuracy changed by +0.5%. Paging on that is noise.
`distribution_drift` is still reported alongside it for whoever investigates. An agent
that sees `drift_detected: false` while `distribution_drift: true` has not found a bug.

**`data.ingest_days: 730` is load-bearing.** The monitor compares the last 30 days
against the same 30 days one year earlier. With 365 days of history that comparison is
arithmetically impossible - there is no September-2025 to compare against. When history
is too short the monitor sets `seasonally_matched: false` rather than reporting a
confounded number. Do not reduce this without checking that condition.

**`params.yaml` is a whole-file DVC dependency**, so any param change reruns every
stage including the network download. This is known and accepted; per-key dependency
syntax failed DVC schema validation. Use `dvc repro -s <stage>` to run one stage.

**`.native/` must stay gitignored.** It holds an extracted `libgomp1` .deb from setup.
Unused at runtime - `LD_LIBRARY_PATH` points at `.venv/lib`, not here. It was tracked
by accident in `9e04dc9` and untracked in `eb52cc4`.

**`reports/train_metrics.json` and `reports/baseline_metrics.json` are gitignored**
because DVC owns them. `reports/drift.json`, `promotion.json` and
`feature_importance.csv` are tracked. `reports/drift.html` is a 3.7 MB generated
Evidently report, untracked as of `5768611`.

**Dates run to October 2026** in `devlog.md`, `dvc.lock` and `reports/`. The dataset
ends 2026-09-29. These are not typos.

## Retries

`src/ingest.py` retries on 429, 5xx, and connection/timeout failures with
exponential backoff (up to 4 attempts). Client errors (4xx other than 429) fail
immediately without retry so bad requests do not spin pointlessly.
Guarded by `tests/test_ingest.py`.

## DagsHub

Configured and verified working. `.dvc/config` (committed) holds the public S3
endpoint; `.dvc/config.local` (gitignored) holds credentials.

- One GitHub secret: `DAGSHUB_TOKEN`. The URLs are public constants in `retrain.yml`.
- The DVC S3 remote authenticates with the token as **both** `access_key_id` and
  `secret_access_key`. Verified working.
- MLflow needs `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
  `MLFLOW_TRACKING_PASSWORD`. All three, or the server returns 401.
- **Never paste a token into a chat, a commit message, or any file.** It belongs in
  the GitHub Actions secret and in `.dvc/config.local` only.

## Open items

Ordered by how likely they are to bite.

**1. `reports/drift.html` in git history.** ~11.6 MB across 12 blobs, untracked going
forward but not reclaimed. Deliberately not rewritten with `filter-repo`: force-pushing
a branch that is actively being run in CI is not worth 11.6 MB.

**2. Docker is unavailable from this WSL distro.** The user runs Docker from Windows
CMD. The CLI resolves on PATH but no daemon is reachable, so image builds cannot be
verified locally here. `ci.yml` builds and verifies the image (`RUN python -c "import lightgbm"`).

## Recently resolved items

- **Retry logic in `src/ingest.py`**: Added `get_with_retry` with exponential backoff
  for 429/5xx and network failures; 4xx client errors fail immediately. Unit tests in
  `tests/test_ingest.py`.
- **`libgomp1` in Dockerfiles**: Added `apt-get install -y --no-install-recommends libgomp1`
  and build-time verification `RUN python -c "import lightgbm"` to both `Dockerfile`
  and `space/Dockerfile`.

## Agent Execution & Permissions

Routine development commands are safe to execute without asking for confirmation when they operate inside this repository or its WSL virtual environment.

### Automatically allowed

The agent may run:

- `python ...`
- `python -m ...`
- `pytest`
- `pytest -q`
- `python -m pytest`
- `ruff check ...`
- `ruff format ...`
- `dvc status`
- `dvc repro`
- `dvc repro -s <stage>`
- `git status`
- `git diff`
- `git log`
- `git show`
- `git branch`
- `pip list`
- `pip show ...`
- `pip install ...` into the project `.venv`
- normal local scripts under `src/`, `tests/`, and project tooling
- file creation and modification inside the repository

Use the project's WSL `.venv` for Python commands. Do not use the Windows Python installation for this repository.

### Commands requiring confirmation

Ask before:

- `git commit`
- `git push`
- deleting important project files or directories
- deleting or overwriting source datasets
- modifying files outside the repository
- installing system-wide packages
- modifying system configuration
- changing Git remotes
- changing DVC remotes
- changing CI/CD configuration in a way that affects deployment or credentials
- deploying or publishing anything
- accessing, displaying, or modifying secrets or credentials
- force-pushing or rewriting Git history

Never paste secrets, tokens, passwords, or credentials into command output, source files, commits, or chat.

### Command execution principles

- Do not ask for confirmation for routine, non-destructive development commands covered above.
- Before running a command, check `AGENTS.md`, `pyproject.toml`, `dvc.yaml`, and relevant project configuration when necessary.
- Prefer existing project scripts and tooling over inventing new commands.
- Run relevant tests after code changes.
- Never claim a command succeeded unless it was actually executed.
- If a command fails, diagnose the failure rather than repeatedly retrying blindly.
- Keep changes focused and reversible.

## Testing

`pytest -q` is fully offline - `tests/conftest.py` replaces both public APIs with
synthetic series, so CI never depends on energy-charts or Open-Meteo being up.
`tests/make_fixture_data.py` builds a fixture model for that path.

`tests/test_monitor.py` and `tests/test_tracking.py` exist as regression guards for
bugs that were found and fixed. Read them before changing drift or tracking logic.

CI runs on Python 3.11. Local development is 3.14. No 3.11 run had happened until the
first successful CI pass at `6bab216`.
