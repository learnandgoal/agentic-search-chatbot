"""Tool-trajectory scenarios shared by the deterministic tests and the optional live-model test.

A *trajectory* is the ordered list of tools the agent called for one question. The prompt's query-to-tool
examples define what a good trajectory looks like; ``check_trajectory`` turns that into pass/fail so the same
rules can be applied to a scripted model (always run) and to a real model (opt-in, see test_trajectories_live.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from fakes import ai, call, last_tool_json, make_llm
from hr_chatbot.graph import build_graph
from hr_chatbot.service import ChatSession, Services
from hr_chatbot.tools import CITABLE_TOOLS

CANDIDATE, JOB = "candidate-001", "job-001"
RESUME_DOC = "candidates/candidate-001/resume.pdf"
JOB_DOC = "jobs/job-001/job-description.docx"


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    # Essential steps in order; each step is a set of acceptable tools (a later call may satisfy it).
    required: tuple[frozenset, ...]
    # The exact trajectory the scripted "model that follows the prompt" takes.
    golden: tuple[str, ...]
    # Wastage budget: more tool calls than this means the model wandered (the golden path plus at most two spare calls).
    max_calls: int = 0


def _steps(*steps: str) -> tuple[frozenset, ...]:
    return tuple(frozenset(s.split("|")) for s in steps)


SCENARIOS = [
    Scenario(
        "exact_phrase",
        "Does the candidate's resume mention Apache Airflow?",
        _steps("search", "grep"),
        ("search", "grep"),
        max_calls=4,
    ),
    Scenario(
        "adjacent_information",
        "What does the resume list right after the Staff Data Engineer role?",
        _steps("search", "navigate"),
        ("search", "navigate"),
        max_calls=4,
    ),
    Scenario(
        "section_summary",
        "Summarize the Required Skills section of the job description.",
        _steps("search", "read"),
        ("search", "open", "read"),
        max_calls=4,
    ),
    Scenario(
        "simple_fact",
        "How many years of experience does the candidate state in the summary?",
        _steps("search", "read|grep"),
        ("search", "read"),
        max_calls=3,
    ),
    Scenario("unsupported", "Should we hire this candidate?", (), ()),
]
BY_NAME = {s.name: s for s in SCENARIOS}


def check_trajectory(scenario: Scenario, tools_called: list[str], trace: list[dict] | None = None) -> list[str]:
    """Return a list of problems (empty = the trajectory is acceptable).

    ``trace`` (the turn's tool-call records: name, args, ok) enables the wastage checks: repeated identical
    calls and calls that returned an error. The call-count limit applies to the names alone."""
    problems: list[str] = []
    if len(tools_called) > scenario.max_calls:
        problems.append(f"wasteful: {len(tools_called)} tool calls, at most {scenario.max_calls} expected")
    if trace:
        seen: set[str] = set()
        for record in trace:
            key = f"{record['name']}|{sorted((record.get('args') or {}).items())}"
            if key in seen:
                problems.append(f"repeated identical call: {record['name']} {record.get('args')}")
            seen.add(key)
            if record.get("ok") is False:
                problems.append(f"{record['name']} call failed: {record.get('args')}")
    if not scenario.required:  # unsupported request: no retrieval at all
        if tools_called:
            problems.append(f"unsupported request must not retrieve, but called {tools_called}")
        return problems

    position = 0
    for step in scenario.required:
        while position < len(tools_called) and tools_called[position] not in step:
            position += 1
        if position >= len(tools_called):
            problems.append(f"expected a call to {'/'.join(sorted(step))} (in order) in {tools_called}")
            break
        position += 1

    if not CITABLE_TOOLS & set(tools_called):
        problems.append("answered from search/open only: no read, grep or navigate call produced citable text")
    for i, name in enumerate(tools_called):
        if name == "open" and not {"read", "grep"} & set(tools_called[i + 1:]):
            problems.append("open was not followed by read or grep")
    return problems


# ---------------------------------------------------------------------------- scripted model

def make_session(env, settings, script) -> ChatSession:
    store, ingestor = env
    services = Services(settings, store, ingestor, build_graph(store, make_llm(script), settings))
    return ChatSession(services, CANDIDATE, JOB)


def _cite(payload: dict) -> tuple[int, str]:
    """(chunk_id, quote) taken from text a citable tool really returned."""
    if "matches" in payload:
        m = payload["matches"][0]
        return m["chunk_id"], m["context"]  # the match plus its surrounding text, as grep returned it
    if "results" in payload:
        r = payload["results"][0]
        return r["chunk_id"], r["snippet"][:60]
    c = payload["chunks"][0]
    return c["chunk_id"], c["text"][:60]


def golden_script(scenario: Scenario, store):
    """A model that follows the prompt's plan for this question type, then cites what it read."""
    experience = next(c for c in store.document_chunks(RESUME_DOC) if c["page_or_section"] == "Page 1 - Experience")
    summary = next(c for c in store.document_chunks(RESUME_DOC) if c["page_or_section"] == "Page 1 - Summary")
    plan = {
        "search": {"query": scenario.question},
        "grep": {"document_id": RESUME_DOC, "pattern": "Apache Airflow"},
        "navigate": {"chunk_id": experience["chunk_id"], "direction": "next"},
        "open": {"document_id": JOB_DOC},
        "read": (
            {"document_id": JOB_DOC, "page_or_section": "Required Skills"}
            if scenario.name == "section_summary"
            else {"chunk_id": summary["chunk_id"]}
        ),
    }

    def script(messages, tools, tool_choice, n):
        if n <= len(scenario.golden):
            name = scenario.golden[n - 1]
            return ai(call(name, f"c{n}", **plan[name]))
        if not scenario.golden:
            return ai(call("submit_answer", "u1", intent="unsupported",
                           response="I can't make or recommend a hiring decision, but I can compare the "
                                    "candidate's documented skills with the job requirements."))
        chunk_id, quote = _cite(last_tool_json(messages))
        return ai(call("submit_answer", f"s{n}", intent="qa", response="Answer grounded in the documents.",
                       evidence=[{"chunk_id": chunk_id, "quote": quote}]))

    return script
