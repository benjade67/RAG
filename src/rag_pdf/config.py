from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_env_path() -> Path:
    return get_project_root() / ".env"


def load_environment() -> None:
    """Load environment variables from the project .env file if present.

    Uses python-dotenv when available and falls back to a tiny parser otherwise.
    """
    dotenv_path = get_env_path()
    if not dotenv_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_manually(dotenv_path)
        return

    load_dotenv(dotenv_path=dotenv_path, override=False)


def _load_env_file_manually(dotenv_path: Path) -> None:
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def update_environment(updates: dict[str, str]) -> None:
    dotenv_path = get_env_path()
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: list[str] = []
    if dotenv_path.exists():
        existing_lines = dotenv_path.read_text(encoding="utf-8").splitlines()

    pending_updates = dict(updates)
    rendered_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered_lines.append(line)
            continue

        key, _ = line.split("=", 1)
        key = key.strip()
        if key in pending_updates:
            value = pending_updates.pop(key)
            rendered_lines.append(f"{key}={value}")
        else:
            rendered_lines.append(line)

    if pending_updates:
        if rendered_lines and rendered_lines[-1].strip():
            rendered_lines.append("")
        for key, value in pending_updates.items():
            rendered_lines.append(f"{key}={value}")

    dotenv_path.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")

    for key, value in updates.items():
        os.environ[key] = value
