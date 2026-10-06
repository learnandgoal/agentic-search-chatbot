"""Tool-trajectory tests: which tools the agent uses for which kind of question, and what may be cited.

The scripted model here follows the plan the system prompt describes. That proves the *application* accepts and
correctly handles each trajectory (and rejects bad ones). Whether a real model actually *chooses* these trajectories
is measured by test_trajectories_live.py, which needs an API key and is skipped by default.
"""

from __future__ import annotations

import pytest

from fakes import ai, call, last_tool_json
from hr_chatbot.prompts import build_system_prompt
from hr_chatbot.tools import DOC_TOOL_SPECS
from trajectories import BY_NAME, SCENARIOS, check_trajectory, golden_script, make_session


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_golden_trajectory_is_accepted_and_yields_verified_evidence(env, settings, scenario):
    turn = make_session(env, settings, golden_script(scenario, env[0])).ask(scenario.question)
    names = [t["name"] for t in turn.trace]

    assert names == list(scenario.golden)
    assert check_trajectory(scenario, names) == []
    assert all(t["ok"] for t in turn.trace)

    out = turn.output
    if not scenario.golden:  # unsupported -> no retrieval
        assert turn.intent == "unsupported"
        assert out["retrieved_context"] == [] and out["source_evidence"] == []
    else:
        assert turn.intent == "qa"
        assert len(out["source_evidence"]) == 1, out["response"]  # the citation survived verification
        assert "could not be verified" not in out["response"]


def test_exact_phrase_cites_the_grep_match(env, settings):
    scenario = BY_NAME["exact_phrase"]
    out = make_session(env, settings, golden_script(scenario, env[0])).ask(scenario.question).output
    assert "airflow" in out["source_evidence"][0]["content"].lower()
    assert out["source_evidence"][0]["source_file"] == "resume.pdf"


def test_adjacent_information_cites_the_neighbouring_section(env, settings):
    scenario = BY_NAME["adjacent_information"]
    out = make_session(env, settings, golden_script(scenario, env[0])).ask(scenario.question).output
    assert out["source_evidence"][0]["page_or_section"] == "Page 2 - Experience"
    assert "Contoso" in out["source_evidence"][0]["content"]


def test_section_summary_cites_the_section_text_that_was_read(env, settings):
    scenario = BY_NAME["section_summary"]
    out = make_session(env, settings, golden_script(scenario, env[0])).ask(scenario.question).output
    assert out["source_evidence"][0]["page_or_section"] == "Required Skills"
    assert out["source_evidence"][0]["source_file"] == "job-description.docx"


# ------------------------------------------------------------------ bad trajectories are caught

def test_checker_flags_answering_from_search_only():
    problems = check_trajectory(BY_NAME["simple_fact"], ["search"])
    assert any("search/open only" in p for p in problems) or any("expected a call" in p for p in problems)


def test_checker_flags_open_without_follow_up():
    problems = check_trajectory(BY_NAME["section_summary"], ["search", "open", "read"][:2])
    assert any("open was not followed by read or grep" in p for p in problems)


def test_checker_flags_retrieval_for_unsupported_requests():
    assert check_trajectory(BY_NAME["unsupported"], ["search"])
    assert check_trajectory(BY_NAME["unsupported"], []) == []


def test_checker_flags_missing_navigate_for_adjacent_questions():
    assert check_trajectory(BY_NAME["adjacent_information"], ["search", "read"])
    assert check_trajectory(BY_NAME["exact_phrase"], ["search", "read"])  # grep expected
    assert check_trajectory(BY_NAME["exact_phrase"], ["grep", "search"])  # order matters


def test_search_only_answer_loses_its_citations_in_the_graph(env, settings):
    """The same bad trajectory, end to end: the answer is delivered but nothing is cited."""
    def script(messages, tools, tool_choice, n):
        if n == 1:
            return ai(call("search", "c1", query="how many years of experience"))
        hit = last_tool_json(messages)["hits"][0]
        return ai(call("submit_answer", "c2", intent="qa", response="About 8 years.",
                       evidence=[{"chunk_id": hit["chunk_id"], "quote": hit["snippet"][:40]}]))

    turn = make_session(env, settings, script).ask("How many years of experience?")
    assert [t["name"] for t in turn.trace] == ["search"]
    assert check_trajectory(BY_NAME["simple_fact"], ["search"])  # flagged
    assert turn.output["source_evidence"] == []


# ------------------------------------------------------------------ the prompt carries the guidance

def prompt_lines() -> list[str]:
    return build_system_prompt("candidate-001", "job-001", 5, is_initial=False).splitlines()


def line_starting(prefix: str) -> str:
    return next(line for line in prompt_lines() if line.startswith(prefix))


def test_prompt_has_query_to_tool_examples_for_every_question_type():
    assert "Query-to-tool examples" in "\n".join(prompt_lines())
    exact = line_starting("- Exact phrase")
    assert "Apache Airflow" in exact and "grep" in exact and "search" in exact
    adjacent = line_starting("- Adjacent information")
    assert "navigate" in adjacent and "search" in adjacent
    summary = line_starting("- Section summary")
    assert "open" in summary and "read" in summary and summary.index("open") < summary.index("read")
    fact = line_starting("- Simple fact")
    assert "read" in fact and "snippet alone" in fact
    unsupported = line_starting("- Unsupported")
    assert "NO retrieval tool" in unsupported and '"unsupported"' in unsupported


def test_prompt_says_open_is_structure_only_and_search_is_not_citable():
    open_line = line_starting("- open:")
    assert "STRUCTURE only" in open_line and "no text" in open_line and "follow open with read" in open_line
    search_line = line_starting("- search:")
    assert "NOT citable" in search_line and "never evidence" in search_line
    assert "Only text returned by read, grep or navigate can be cited" in "\n".join(prompt_lines())


def test_tool_descriptions_match_the_prompt():
    desc = {t.name: t.description for t in DOC_TOOL_SPECS}
    assert "NOT citable" in desc["search"]
    assert "STRUCTURE" in desc["open"] and "NOT citable" in desc["open"] and "follow open with read" in desc["open"]
    for name in ("navigate", "read", "grep"):
        assert desc[name].rstrip().endswith("Citable."), name
