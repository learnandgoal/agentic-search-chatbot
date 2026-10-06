"""LangGraph flow: one ``agent`` node, one ``tools`` node, a bounded loop.

START -> agent --(document tool calls)--> tools --> agent ... --(submit_answer)--> END

* ``agent`` asks the model to either call a document tool or ``submit_answer`` (the structured final answer).
* ``tools`` executes the document tools, appends results to ``retrieved_context`` and routes back.
* The tool-call budget is enforced in code: once it is spent the model can only call ``submit_answer``.
* Evidence is verified before it is returned, so citations can't be invented. A citation counts only if its quote
  appears in text that ``read``, ``grep`` or ``navigate`` returned this turn (``search`` hits and ``open`` structure
  are never evidence). This is enforced here in code, not only in the prompt.
"""

from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from .config import Settings
from .prompts import INITIAL_SUMMARY_PROMPT, build_system_prompt
from .schemas import SUBMIT_NAME, EvidenceRef, SubmitAnswer
from .store import HRStore, Scope
from .tools import CITABLE_TOOLS, DOC_TOOL_SPECS, SUBMIT_TOOL, DocumentTools

log = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "I could not produce a reliable answer from the documents. Please try rephrasing the question."
)


class AgentState(TypedDict, total=False):
    session_id: str
    candidate_id: str
    job_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    retrieved_context: Annotated[list[dict], operator.add]
    source_evidence: list[dict]
    tool_calls: Annotated[list[dict], operator.add]
    final_response: dict | None
    intent: str


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def citable_text(retrieved: list[dict]) -> dict[int, list[str]]:
    """chunk_id -> the text read/grep/navigate actually returned for it this turn. search and open add nothing."""
    shown: dict[int, list[str]] = {}
    for item in retrieved:
        if item.get("tool") in CITABLE_TOOLS:
            shown.setdefault(item["chunk_id"], []).append(item["content"])
    return shown


def verify_evidence(
    store: HRStore, scope: Scope, shown: dict[int, list[str]], refs: list[EvidenceRef]
) -> tuple[list[dict], int]:
    """Keep only citations that (1) point at a chunk whose text came back from read, grep or navigate this turn,
    (2) are in scope, and (3) quote text that was actually returned (``shown``), not merely text somewhere in the chunk.
    Provenance (file, page/section) comes from the index, never from the model."""
    verified: list[dict] = []
    dropped = 0
    used: set[tuple[int, str]] = set()
    for ref in refs:
        chunk = store.get_chunk(ref.chunk_id) if ref.chunk_id in shown else None
        if chunk is None or not store.in_scope(chunk, scope):
            dropped += 1
            continue
        haystacks = [_norm(text) for text in shown[ref.chunk_id]]
        parts = [p.strip() for p in re.split(r"\.{3}|…", _norm(ref.quote)) if len(p.strip()) >= 8]
        if not parts or not any(all(p in h for p in parts) for h in haystacks):
            dropped += 1
            continue
        key = (chunk["chunk_id"], _norm(ref.quote))
        if key in used:
            continue
        used.add(key)
        verified.append(
            {
                "source_file": chunk["source_file"],
                "page_or_section": chunk["page_or_section"],
                "content": ref.quote.strip(),
            }
        )
    return verified, dropped


def _contract_context(items: list[dict]) -> list[dict]:
    return [
        {"source_file": i["source_file"], "page_or_section": i["page_or_section"], "content": i["content"], "score": i["score"]}
        for i in items
    ]


def build_graph(store: HRStore, llm: BaseChatModel, settings: Settings):
    tools = DocumentTools(store, settings)
    agent_llm = llm.bind_tools([*DOC_TOOL_SPECS, SUBMIT_TOOL], tool_choice="any")
    final_llm = llm.bind_tools([SUBMIT_TOOL], tool_choice=SUBMIT_NAME)  # budget exhausted: answer only

    # ------------------------------------------------------------------ agent node
    def agent_node(state: AgentState) -> dict[str, Any]:
        scope = Scope(state["candidate_id"], state["job_id"])
        budget_left = settings.max_tool_calls - len(state.get("tool_calls", []))
        system = SystemMessage(
            content=build_system_prompt(
                scope.candidate_id, scope.job_id, budget_left, state.get("current_query") == INITIAL_SUMMARY_PROMPT
            )
        )
        model = agent_llm if budget_left > 0 else final_llm
        ai: AIMessage = model.invoke([system, *state["messages"]])

        calls = list(ai.tool_calls or [])
        doc_calls = [c for c in calls if c["name"] != SUBMIT_NAME]
        submit = next((c for c in calls if c["name"] == SUBMIT_NAME), None)

        if doc_calls and budget_left > 0:
            return {"messages": [ai]}  # route to tools; a stray submit in the same message is rejected there
        return _finalize(state, scope, ai, submit)

    def _finalize(state: AgentState, scope: Scope, ai: AIMessage, submit: dict | None) -> dict[str, Any]:
        retrieved = state.get("retrieved_context", [])
        shown = citable_text(retrieved)  # search hits and open results are deliberately not in here
        intent, response, evidence, dropped = "qa", "", [], 0

        if submit is not None:
            try:
                answer = SubmitAnswer.model_validate(submit["args"])
            except ValidationError as exc:
                log.warning("submit_answer failed validation: %s", exc)
                answer = None
            if answer is not None:
                intent, response = answer.intent, answer.response.strip()
                if intent == "qa":
                    evidence, dropped = verify_evidence(store, scope, shown, answer.evidence)
        else:  # the model answered in plain text instead of calling submit_answer
            content = ai.content if isinstance(ai.content, str) else ""
            response = content.strip()

        if not response:
            response = FALLBACK_RESPONSE
        if dropped:
            response += (
                f"\n\n_Note: {dropped} citation(s) could not be verified (evidence must quote text returned by "
                "read, grep or navigate) and were omitted._"
            )

        contract = {
            "session_id": state["session_id"],
            "candidate_id": scope.candidate_id,
            "job_id": scope.job_id,
            "response": response,
            "source_evidence": evidence,
            "retrieved_context": _contract_context(retrieved),
        }
        return {
            "messages": [AIMessage(content=response)],  # keep history clean: no dangling tool call
            "source_evidence": evidence,
            "final_response": contract,
            "intent": intent,
        }

    # ------------------------------------------------------------------ tools node
    def tools_node(state: AgentState) -> dict[str, Any]:
        scope = Scope(state["candidate_id"], state["job_id"])
        last = state["messages"][-1]
        executed = len(state.get("tool_calls", []))
        existing = state.get("retrieved_context", [])
        seen = {c["chunk_id"] for c in existing}
        known = {(c["chunk_id"], c["content"], c.get("tool")) for c in existing}  # tool matters: a read of a short chunk equals its search snippet but is citable

        done = {json.dumps([c["name"], c["args"]], sort_keys=True, default=str) for c in state.get("tool_calls", [])}
        replies: list[ToolMessage] = []
        new_context: list[dict] = []
        new_calls: list[dict] = []

        def reply(call: dict, payload: dict) -> None:
            replies.append(
                ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=call["id"], name=call["name"])
            )

        for call in last.tool_calls:
            if call["name"] == SUBMIT_NAME:
                reply(call, {"error": "Not executed: call submit_answer on its own, after reviewing the results of the other tools."})
                continue
            if executed + len(new_calls) >= settings.max_tool_calls:
                reply(call, {"error": "Tool-call budget exhausted. Call submit_answer with what you have."})
                continue

            args = call.get("args") or {}
            signature = json.dumps([call["name"], args], sort_keys=True, default=str)
            if signature in done:  # same tool, same arguments: the earlier result is still in the conversation
                new_calls.append({"name": call["name"], "args": args, "ok": False, "results": 0, "duplicate": True})
                reply(call, {"error": "Duplicate call: identical arguments were already used this turn and its result is above. "
                                      "Use that result, or call a different tool, or call submit_answer."})
                continue
            done.add(signature)
            result = tools.call(call["name"], args, scope, seen)
            new_calls.append({"name": call["name"], "args": call.get("args") or {}, "ok": result.ok, "results": len(result.context)})
            for item in result.context:
                key = (item["chunk_id"], item["content"], item.get("tool"))
                if key not in known:
                    known.add(key)
                    new_context.append(item)
            reply(call, result.payload)

        return {"messages": replies, "retrieved_context": new_context, "tool_calls": new_calls}

    def route(state: AgentState) -> str:
        return END if state.get("final_response") else "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def recursion_limit(settings: Settings) -> int:
    return 2 * settings.max_tool_calls + 10
