from __future__ import annotations

from pathlib import Path

import responses

from fpl_automate.data.cache import FileCache
from fpl_automate.data.fpl_client import FplApiError, FplClient


def _client(tmp_path: Path, cache_ttl=300.0) -> FplClient:
    return FplClient(
        base_url="https://fantasy.premierleague.com/api",
        timeout_seconds=5,
        min_request_interval_seconds=0.0,
        max_retries=3,
        cache=FileCache(tmp_path / "cache"),
        cache_ttl_seconds=cache_ttl,
    )


@responses.activate
def test_get_bootstrap_static_success(tmp_path):
    responses.add(
        responses.GET,
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        json={"elements": [], "teams": []},
        status=200,
    )
    client = _client(tmp_path)
    data = client.get_bootstrap_static()
    assert data == {"elements": [], "teams": []}


@responses.activate
def test_retries_on_500_then_succeeds(tmp_path):
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, json={"ok": True}, status=200)
    client = _client(tmp_path)
    data = client.get_bootstrap_static()
    assert data == {"ok": True}


@responses.activate
def test_404_raises_fpl_api_error(tmp_path):
    responses.add(
        responses.GET,
        "https://fantasy.premierleague.com/api/entry/999999999/",
        status=404,
    )
    client = _client(tmp_path)
    try:
        client.get_entry(999999999)
        assert False, "expected FplApiError"
    except FplApiError:
        pass


@responses.activate
def test_cache_avoids_second_http_call(tmp_path):
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    responses.add(responses.GET, url, json={"call": 1}, status=200)
    client = _client(tmp_path, cache_ttl=300.0)

    first = client.get_bootstrap_static()
    second = client.get_bootstrap_static()

    assert first == second == {"call": 1}
    assert len(responses.calls) == 1


@responses.activate
def test_persistent_500_raises_after_max_retries(tmp_path):
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    for _ in range(5):
        responses.add(responses.GET, url, status=500)
    client = _client(tmp_path)
    try:
        client.get_bootstrap_static()
        assert False, "expected FplApiError"
    except FplApiError:
        pass
