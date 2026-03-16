from __future__ import annotations

import os
from pathlib import Path

def load_environment() -> None:
    """Load environment variables from the project .env file if present.

    Uses python-dotenv when available and falls back to a tiny parser otherwise.
    """
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / ".env"
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
