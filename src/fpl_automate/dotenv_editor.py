"""Read and update a .env file without destroying comments or unrelated keys.

This exists so the GUI's Settings tab can write configuration for you --
per the whole point of that tab, you should never have to open .env in a
text editor. A save only ever touches the specific keys the form manages;
every comment line, blank line, and any other key is left exactly as-is.
"""
from __future__ import annotations

from pathlib import Path


def read_env_values(path: Path) -> dict[str, str]:
    """Returns {KEY: value} for every KEY=value line in the file (comments and
    blank lines ignored). Returns {} if the file doesn't exist yet."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Writes `updates` into the .env at `path`, creating it if necessary.

    Existing KEY=value lines for keys in `updates` are replaced in place
    (keeping their position); everything else -- comments, blank lines,
    unrelated keys -- is preserved verbatim. Keys in `updates` that aren't
    already present are appended at the end.
    """
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)

    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}")
                continue
        new_lines.append(line)

    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
