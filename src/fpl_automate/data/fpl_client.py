"""Read-only client for the official FPL public JSON endpoints.

These endpoints (fantasy.premierleague.com/api/...) are the same ones the
FPL website itself uses and are publicly reachable with no login for all
data this project needs: player stats, fixtures, and any entry's (team's)
past picks/history/transfers. This project never logs in, never submits
data, and never bypasses any access control -- it only reads what your
browser could already load anonymously.

Politeness / ToS-friendliness built in:
  * a minimum interval is enforced between requests (rate limiting),
  * responses are cached to disk for a short TTL to avoid refetching,
  * requests time out and retry with backoff instead of hammering on error,
  * a descriptive User-Agent identifies this as a personal automation tool.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fpl_automate.data.cache import FileCache

logger = logging.getLogger(__name__)

USER_AGENT = "fpl-automate/0.1 (personal FPL decision-support tool; read-only)"


class FplApiError(RuntimeError):
    """Raised when the FPL API cannot be reached or returns something unusable."""


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._last_request_at: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class FplClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 15.0,
        min_request_interval_seconds: float = 1.0,
        max_retries: int = 3,
        cache: FileCache | None = None,
        cache_ttl_seconds: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._limiter = RateLimiter(min_request_interval_seconds)
        self._max_retries = max(1, max_retries)
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def _get(self, path: str, *, use_cache: bool = True) -> Any:
        url = f"{self._base_url}{path}"

        if use_cache and self._cache is not None:
            cached = self._cache.get(url, ttl_seconds=self._cache_ttl)
            if cached is not None:
                logger.debug("cache hit: %s", path)
                return cached

        data = self._get_with_retry(url)

        if use_cache and self._cache is not None:
            self._cache.set(url, data)
        return data

    def _get_with_retry(self, url: str) -> Any:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, _RetryableStatus)),
        )
        def _do_request() -> Any:
            self._limiter.wait()
            logger.debug("GET %s", url)
            try:
                resp = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                raise FplApiError(f"Network error contacting FPL API: {exc}") from exc

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "5"))
                logger.warning("FPL API rate-limited us; waiting %ss", retry_after)
                time.sleep(retry_after)
                raise _RetryableStatus(429)
            if resp.status_code >= 500:
                raise _RetryableStatus(resp.status_code)
            if resp.status_code == 404:
                raise FplApiError(f"Not found (404): {url}")
            if resp.status_code >= 400:
                raise FplApiError(f"FPL API returned HTTP {resp.status_code} for {url}")

            try:
                return resp.json()
            except ValueError as exc:
                raise FplApiError(f"FPL API returned non-JSON response for {url}") from exc

        try:
            return _do_request()
        except _RetryableStatus as exc:
            raise FplApiError(
                f"FPL API kept returning HTTP {exc.status_code} for {url} after retries"
            ) from exc

    # -- Public endpoints -------------------------------------------------

    def get_bootstrap_static(self) -> dict[str, Any]:
        """Players, teams, gameweeks (events), positions -- the core dataset."""
        return self._get("/bootstrap-static/")

    def get_fixtures(self, event: int | None = None) -> list[dict[str, Any]]:
        path = "/fixtures/" if event is None else f"/fixtures/?event={event}"
        return self._get(path)

    def get_element_summary(self, player_id: int) -> dict[str, Any]:
        """Per-fixture history + upcoming fixtures for a single player."""
        return self._get(f"/element-summary/{player_id}/")

    def get_entry(self, team_id: int) -> dict[str, Any]:
        return self._get(f"/entry/{team_id}/", use_cache=False)

    def get_entry_history(self, team_id: int) -> dict[str, Any]:
        """Season-by-gameweek history, bank/value, and chip usage for a team."""
        return self._get(f"/entry/{team_id}/history/", use_cache=False)

    def get_entry_picks(self, team_id: int, event: int) -> dict[str, Any]:
        return self._get(f"/entry/{team_id}/event/{event}/picks/", use_cache=False)

    def get_entry_transfers(self, team_id: int) -> list[dict[str, Any]]:
        return self._get(f"/entry/{team_id}/transfers/", use_cache=False)

    def get_league_standings(self, league_id: int, page: int = 1) -> dict[str, Any]:
        return self._get(
            f"/leagues-classic/{league_id}/standings/?page_standings={page}", use_cache=False
        )


class _RetryableStatus(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"retryable status {status_code}")
        self.status_code = status_code
