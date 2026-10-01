"""Central settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load OPENAI_API_KEY etc. from ./.env
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    # Agent loop
    max_tool_calls: int = 8  # stopping policy: configured tool-call limit per turn
    history_messages: int = 12  # prior chat messages sent back to the model
    question_max_chars: int = 2000
    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120
    # Tool limits
    max_top_k: int = 10
    snippet_chars: int = 240
    navigate_snippet_chars: int = 400
    read_default_chars: int = 3000
    read_max_chars: int = 6000
    grep_max_results: int = 20
    grep_pattern_max_len: int = 100


def load_settings() -> Settings:
    data_dir = Path(os.getenv("HR_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
    db_path = Path(os.getenv("HR_DB_PATH", data_dir / "indexes" / "hr.sqlite")).expanduser()
    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        max_tool_calls=int(os.getenv("HR_MAX_TOOL_CALLS", "8")),
    )
