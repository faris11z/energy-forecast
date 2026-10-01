"""Guards the tracking-URI wiring.

Regression: src/track.py once called mlflow.set_tracking_uri() with a hardcoded
local ./mlruns path instead of common.tracking_uri(). Promote and the API already
read the environment, so track.py was the only module ignoring MLFLOW_TRACKING_URI.
The effect was invisible locally - everything landed in ./mlruns anyway - but in CI
the runs went to the runner's disk and were discarded, and then promote.py looked
for runs on DagsHub and found none.

So this test asserts the *wiring*, cheaply, without contacting a server.
"""


import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402


def test_tracking_uri_defaults_to_local_store(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert common.tracking_uri().startswith("file:")


def test_tracking_uri_follows_the_environment(monkeypatch):
    remote = "https://dagshub.com/some-user/some-repo.mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", remote)
    assert common.tracking_uri() == remote


def test_track_py_does_not_hardcode_a_store():
    """track.py must go through common.tracking_uri().

    Read as source rather than by running it: actually logging takes minutes and
    needs a network server, and the mistake here is structural anyway.
    """
    source = (ROOT / "src" / "track.py").read_text()
    assert "tracking_uri()" in source, "track.py should use common.tracking_uri()"
    assert 'TRACKING_DIR.as_uri()' not in source, (
        "track.py must not set a hardcoded local store"
    )