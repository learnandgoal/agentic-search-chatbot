"""OPT-IN: do the real model's tool trajectories match the prompt's query-to-tool examples?

Skipped unless HR_LIVE_TESTS=1 and credentials are configured (see .env.example). It calls the real chat model once
per scenario and costs a few cents. The deterministic tests in test_trajectories.py cannot answer this question,
because a scripted model does whatever the script says.

    HR_LIVE_TESTS=1 HR_EMBEDDER=hashing python -m pytest tests/test_trajectories_live.py -v
"""

from __future__ import annotations

import os

import pytest

from hr_chatbot.service import ChatSession, create_services
from trajectories import CANDIDATE, JOB, SCENARIOS, check_trajectory

pytestmark = pytest.mark.skipif(
    os.getenv("HR_LIVE_TESTS") != "1", reason="live-model test: set HR_LIVE_TESTS=1 and provide credentials"
)


@pytest.fixture()
def live_session(env, settings):
    services = create_services(settings)  # real chat model from hr_chatbot/llm.py; embedder from HR_EMBEDDER
    services.ingestor.ingest_scope(CANDIDATE, JOB)
    return ChatSession(services, CANDIDATE, JOB)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_real_model_follows_the_query_to_tool_examples(live_session, scenario):
    turn = live_session.ask(scenario.question)
    names = [t["name"] for t in turn.trace]
    problems = check_trajectory(scenario, names, turn.trace)
    assert not problems, f"{scenario.name}: trajectory {names} -> {problems}"
    if scenario.required:
        assert turn.output["source_evidence"], f"{scenario.name}: answer had no verified evidence ({names})"
    else:
        assert turn.intent == "unsupported"
