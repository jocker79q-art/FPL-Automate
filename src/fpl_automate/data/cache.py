"""Minimal on-disk JSON response cache with a TTL.

Keeps the FPL client polite (fewer redundant calls to a third party's
public API) without pulling in an extra dependency. Not a general-purpose
cache -- just enough for "don't refetch bootstrap-static every few seconds
during a single CLI invocation, and allow a short-TTL reuse across nearby
invocations."
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileCache:
    directory: Path
    default_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str, ttl_seconds: float | None = None) -> Any | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                envelope = json.load(fh)
            return envelope["data"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, key: str, data: Any) -> None:
        path = self._path_for(key)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump({"cached_at": time.time(), "key": key, "data": data}, fh)
        tmp_path.replace(path)
