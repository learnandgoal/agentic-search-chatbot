from __future__ import annotations

from langchain_core.messages import ToolMessage

from fakes import ai, call, last_tool_json, make_llm
from hr_chatbot.graph import build_graph
from hr_chatbot.prompts import INITIAL_SUMMARY_PROMPT
from hr_chatbot.service import ChatSession, Services

CONTRACT_KEYS = {"session_id", "candidate_id", "job_id", "response", "source_evidence", "retrieved_context"}
QUOTE = "Designed a Kafka and Spark Structured Streaming pipeline"


def make_session(env, settings, script) -> tuple[ChatSession, object]:
    store, ingestor = env
    llm = make_llm(script)
    services = Services(settings, store, ingestor, build_graph(store, llm, settings))
    return ChatSession(services, "candidate-001", "job-001"), llm


def happy_script(quote=QUOTE, intent="qa"):
    def script(messages, tools, tool_choice, n):
        if n == 1:
            return ai(call("search", "c1", query="kafka streaming pipelines", top_k=4))
        if n == 2:
            hits = last_tool_json(messages)["hits"]
            resume_hit = next(h for h in hits if h["source_file"] == "resume.pdf")
            return ai(call("grep", "c2", document_id=resume_hit["document_id"], pattern="kafka"))
        matches = last_tool_json(messages)["matches"]
        return ai(call("submit_answer", "c3", intent=intent, response="She has streaming experience.",
                       evidence=[{"chunk_id": matches[0]["chunk_id"], "quote": quote}]))
    return script


def test_initial_summary_follows_the_output_contract(env, settings):
    session, llm = make_session(env, settings, happy_script())
    turn = session.start()
    out = turn.output

    assert set(out) == CONTRACT_KEYS
    assert (out["candidate_id"], out["job_id"], out["session_id"]) == ("candidate-001", "job-001", session.session_id)
    assert out["response"] == "She has streaming experience."
    assert out["source_evidence"] == [{"source_file": "resume.pdf", "page_or_section": "Page 1 - Experience", "content": QUOTE}]
    assert all(set(c) == {"source_file", "page_or_section", "content", "score"} for c in out["retrieved_context"])
    assert any(c["score"] is not None for c in out["retrieved_context"])  # from search
    assert any(c["score"] is None for c in out["retrieved_context"])  # from grep
    assert [t["name"] for t in turn.trace] == ["search", "grep"]
    assert turn.ingest is not None and not turn.ingest.warnings
    # every model call could use all six tools while budget remained
    assert set(llm.log[0]["tools"]) == {"search", "open", "navigate", "read", "grep", "submit_answer"}


def test_initial_summary_prompt_is_sent_and_not_classified(env, settings):
    seen = {}

    def script(messages, tools, tool_choice, n):
        seen["system"] = messages[0].content
        seen["human"] = messages[1].content
        return ai(call("submit_answer", "c1", intent="qa", response="ok"))

    session, _ = make_session(env, settings, script)
    session.start()
    assert seen["human"] == INITIAL_SUMMARY_PROMPT
    assert "automatic initial briefing" in seen["system"] and "candidate-001" in seen["system"]


def test_fabricated_and_foreign_citations_are_dropped(env, settings):
    store, _ = env
    foreign = store.document_chunks("candidates/candidate-002/resume.docx")[0]["chunk_id"]
    unseen = store.document_chunks("candidates/candidate-001/resume.pdf")[-1]["chunk_id"]  # never retrieved

    def script(messages, tools, tool_choice, n):
        if n == 1:
            return ai(call("search", "c1", query="kafka"))
        hit = last_tool_json(messages)["hits"][0]
        return ai(call("submit_answer", "c2", intent="qa", response="Answer.", evidence=[
            {"chunk_id": hit["chunk_id"], "quote": "this sentence does not exist in the document"},
            {"chunk_id": foreign, "quote": "Analyst with 3 years of experience"},
            {"chunk_id": unseen, "quote": "AWS Certified Data Analytics"},
        ]))

    out = make_session(env, settings, script)[0].start().output
    assert out["source_evidence"] == []
    assert "3 citation(s) could not be verified" in out["response"]


def test_quote_with_ellipsis_and_whitespace_differences_is_accepted(env, settings):
    quote = "Designed a Kafka and Spark   Structured Streaming ... reducing end-to-end latency"
    out = make_session(env, settings, happy_script(quote))[0].start().output
    assert len(out["source_evidence"]) == 1


def test_tool_budget_is_enforced_then_answer_is_forced(env, settings):
    # settings.max_tool_calls == 4; this model never stops searching on its own
    def script(messages, tools, tool_choice, n):
        if tools == ["submit_answer"]:
            return ai(call("submit_answer", "final", intent="qa", response="Partial answer; not everything verified."))
        return ai(call("search", f"s{n}", query=f"python {n}"))

    session, llm = make_session(env, settings, script)
    turn = session.start()
    assert len(turn.trace) == settings.max_tool_calls
    assert llm.log[-1]["tools"] == ["submit_answer"] and llm.log[-1]["tool_choice"] == "submit_answer"
    assert turn.output["response"].startswith("Partial answer")


def test_parallel_calls_over_budget_get_error_replies(env, settings):
    captured = []

    def script(messages, tools, tool_choice, n):
        if n == 1:
            return ai(*[call("search", f"p{i}", query=f"skill {i}") for i in range(6)])
        captured.extend(m for m in messages if isinstance(m, ToolMessage))
        return ai(call("submit_answer", "f", intent="qa", response="done"))

    turn = make_session(env, settings, script)[0].start()
    assert len(turn.trace) == settings.max_tool_calls
    assert len(captured) == 6  # every tool call id got a reply (providers require this)
    assert sum("budget exhausted" in m.content for m in captured) == 2


def test_submit_mixed_with_tool_calls_is_rejected_then_retried(env, settings):
    def script(messages, tools, tool_choice, n):
        if n == 1:
            return ai(call("search", "a", query="kafka"), call("submit_answer", "b", intent="qa", response="too early"))
        if n == 2:
            assert any("on its own" in m.content for m in messages if isinstance(m, ToolMessage))
            return ai(call("submit_answer", "c", intent="qa", response="after reviewing"))

    out = make_session(env, settings, script)[0].start().output
    assert out["response"] == "after reviewing" and out["retrieved_context"]


def test_unsupported_intent_returns_no_evidence(env, settings):
    def script(messages, tools, tool_choice, n):
        return ai(call("submit_answer", "c1", intent="unsupported",
                       response="I can't recommend a hiring decision, but I can compare skills to requirements.",
                       evidence=[{"chunk_id": 1, "quote": "should be dropped anyway"}]))

    session, _ = make_session(env, settings, script)
    turn = session.ask("Should we hire her?")
    assert turn.intent == "unsupported" and turn.trace == []
    assert turn.output["source_evidence"] == [] and "can't recommend" in turn.output["response"]


def test_plain_text_answer_is_accepted_without_evidence(env, settings):
    turn = make_session(env, settings, lambda *a: ai(text="Plain answer."))[0].ask("Hi?")
    assert turn.output["response"] == "Plain answer." and turn.output["source_evidence"] == []


def test_invalid_submit_payload_falls_back_safely(env, settings):
    turn = make_session(env, settings, lambda *a: ai(call("submit_answer", "x", intent="maybe")))[0].ask("?")
    assert "could not produce a reliable answer" in turn.output["response"]


def test_follow_up_keeps_history_but_resets_retrieval(env, settings):
    contexts: list[int] = []
    prompts: list[list[str]] = []

    def script(messages, tools, tool_choice, n):
        prompts.append([type(m).__name__ for m in messages])
        if tools != ["submit_answer"] and not any(isinstance(m, ToolMessage) for m in messages):
            return ai(call("search", f"s{n}", query="kafka"))
        hit = last_tool_json(messages)["hits"][0]
        contexts.append(hit["chunk_id"])
        return ai(call("submit_answer", f"f{n}", intent="qa", response=f"answer {len(contexts)}"))

    session, _ = make_session(env, settings, script)
    session.start()
    follow = session.ask("What about Kafka?")
    assert follow.output["response"] == "answer 2"
    assert contexts[0] == contexts[1]  # seen-chunk exclusion is per turn, so the same top hit is available again
    assert [type(m).__name__ for m in session.history] == ["HumanMessage", "AIMessage"] * 2
    assert prompts[-1][:3] == ["SystemMessage", "HumanMessage", "AIMessage"]  # prior turn is in context


def test_question_validation(env, settings):
    session, _ = make_session(env, settings, lambda *a: ai(text="x"))
    for bad in ("", "   ", "x" * (settings.question_max_chars + 1)):
        try:
            session.ask(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
