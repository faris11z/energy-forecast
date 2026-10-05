from __future__ import annotations

"""Tests for retry behavior in src/ingest.py.

Guards the retry contract specified in AGENTS.md:
- Transient 5xx and 429 errors must retry with backoff.
- Connection/network errors must retry with backoff.
- Client 4xx errors (e.g. 400, 404) must fail immediately without retry.
- Exhausting retries must raise RuntimeError rather than failing silently.
"""

import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import get_with_retry


def _make_mock_response(status_code: int, payload: dict | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp.url = "https://api.example.com/data"
    if payload is not None:
        resp._content = str(payload).encode("utf-8")
    return resp


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_succeeds_first_try(mock_get, mock_sleep):
    mock_resp = _make_mock_response(200)
    mock_get.return_value = mock_resp

    result = get_with_retry("https://api.example.com/data", params={"a": 1})
    assert result == mock_resp
    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_retries_on_503_and_succeeds(mock_get, mock_sleep):
    resp_503 = _make_mock_response(503)
    resp_200 = _make_mock_response(200)
    mock_get.side_effect = [resp_503, resp_200]

    result = get_with_retry(
        "https://api.example.com/data",
        params={},
        max_retries=3,
        backoff_factor=1.0,
    )
    assert result == resp_200
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_retries_on_429_and_succeeds(mock_get, mock_sleep):
    resp_429 = _make_mock_response(429)
    resp_200 = _make_mock_response(200)
    mock_get.side_effect = [resp_429, resp_200]

    result = get_with_retry(
        "https://api.example.com/data",
        params={},
        max_retries=3,
        backoff_factor=1.0,
    )
    assert result == resp_200
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_fails_immediately_on_400(mock_get, mock_sleep):
    resp_400 = _make_mock_response(400)
    mock_get.return_value = resp_400

    with pytest.raises(requests.HTTPError):
        get_with_retry("https://api.example.com/data", params={})

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_fails_immediately_on_404(mock_get, mock_sleep):
    resp_404 = _make_mock_response(404)
    mock_get.return_value = resp_404

    with pytest.raises(requests.HTTPError):
        get_with_retry("https://api.example.com/data", params={})

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("ingest.time.sleep")
@patch("ingest.requests.get")
def test_get_with_retry_raises_runtime_error_when_exhausted(mock_get, mock_sleep):
    resp_503 = _make_mock_response(503)
    mock_get.return_value = resp_503

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        get_with_retry(
            "https://api.example.com/data",
            params={},
            max_retries=3,
            backoff_factor=0.5,
        )

    assert mock_get.call_count == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0), call(2.0)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
