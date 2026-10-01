from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hr_chatbot.config import Settings  # noqa: E402
from hr_chatbot.embeddings import HashingEmbedder  # noqa: E402
from hr_chatbot.ingest import Ingestor  # noqa: E402
from hr_chatbot.sample_data import generate  # noqa: E402
from hr_chatbot.store import HRStore, Scope  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    data = tmp_path / "data"
    return Settings(data_dir=data, db_path=data / "indexes" / "hr.sqlite", max_tool_calls=4)


@pytest.fixture()
def env(settings):
    generate(settings.data_dir)
    store = HRStore(settings.db_path, HashingEmbedder(), settings)
    ingestor = Ingestor(store, settings.data_dir, settings)
    report = ingestor.ingest_all()
    assert not report.warnings, report.warnings
    return store, ingestor


@pytest.fixture()
def store(env):
    return env[0]


@pytest.fixture()
def scope():
    return Scope("candidate-001", "job-001")
