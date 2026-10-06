"""Headless smoke test of the Streamlit app with a scripted model (no API key, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hr_chatbot.service as service
from fakes import ai, call, last_tool_json, make_llm
from hr_chatbot.sample_data import generate


@pytest.fixture()
def app(tmp_path, monkeypatch):
    data = tmp_path / "data"
    generate(data)
    monkeypatch.setenv("HR_DATA_DIR", str(data))
    monkeypatch.delenv("HR_DB_PATH", raising=False)
    monkeypatch.setenv("HR_EMBEDDER", "hashing")

    def script(messages, tools, tool_choice, n):
        if not any(m.type == "tool" for m in messages):
            return ai(call("search", f"s{n}", query="python"))
        hit = last_tool_json(messages)["hits"][0]
        return ai(call("submit_answer", f"f{n}", intent="qa", response=f"Scripted answer {n}",
                       evidence=[{"chunk_id": hit["chunk_id"], "quote": hit["snippet"][:30]}]))

    monkeypatch.setattr(service, "get_chat_model", lambda: make_llm(script))
    from streamlit import cache_resource
    cache_resource.clear()
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dropdowns_start_chat_and_follow_up(app):
    at = app
    assert [o for o in at.sidebar.selectbox[0].options] == ["candidate-001", "candidate-002", "candidate-003", "candidate-004"]
    assert [o for o in at.sidebar.selectbox[1].options] == ["job-001", "job-002", "job-003"]
    assert not at.chat_message  # nothing until Start Chat

    at.sidebar.selectbox[0].select("candidate-002")
    at.sidebar.selectbox[1].select("job-002")
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert at.subheader[0].value.startswith("candidate-002")
    assert "Scripted answer" in at.chat_message[0].markdown[0].value  # automatic initial summary

    at.chat_input[0].set_value("Does the candidate know SQL?").run()
    assert not at.exception, at.exception
    assert [m.name for m in at.chat_message] == ["assistant", "user", "assistant"]
    assert "Scripted answer" in at.chat_message[2].markdown[0].value
