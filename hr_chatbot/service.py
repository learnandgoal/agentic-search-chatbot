"""Wiring and chat-session handling (what the Streamlit app calls)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from .config import Settings, load_settings
from .embeddings import Embedder, get_embedder
from .graph import build_graph, recursion_limit
from .ingest import IngestReport, Ingestor
from .llm import get_chat_model
from .prompts import INITIAL_SUMMARY_PROMPT
from .store import HRStore


@dataclass
class Services:
    settings: Settings
    store: HRStore
    ingestor: Ingestor
    graph: object


def create_services(
    settings: Settings | None = None, llm: BaseChatModel | None = None, embedder: Embedder | None = None
) -> Services:
    settings = settings or load_settings()
    store = HRStore(settings.db_path, embedder or get_embedder(), settings)
    graph = build_graph(store, llm or get_chat_model(), settings)
    return Services(settings, store, Ingestor(store, settings.data_dir, settings), graph)


@dataclass
class TurnResult:
    output: dict  # the output contract: session_id, candidate_id, job_id, response, source_evidence, retrieved_context
    intent: str = "qa"
    trace: list[dict] = field(default_factory=list)  # tool calls made this turn
    ingest: IngestReport | None = None


class ChatSession:
    """One candidate x job conversation. History lives here (Streamlit session state holds the object)."""

    def __init__(self, services: Services, candidate_id: str, job_id: str) -> None:
        self.services = services
        self.candidate_id = candidate_id
        self.job_id = job_id
        self.session_id = uuid.uuid4().hex
        self.history: list[BaseMessage] = []

    def start(self) -> TurnResult:
        """Start Chat: refresh changed files for this pair, then run the predefined initial summary."""
        report = self.services.ingestor.ingest_scope(self.candidate_id, self.job_id)
        result = self._run(INITIAL_SUMMARY_PROMPT)
        result.ingest = report
        return result

    def ask(self, question: str) -> TurnResult:
        question = question.strip()
        if not question:
            raise ValueError("Empty question.")
        limit = self.services.settings.question_max_chars
        if len(question) > limit:
            raise ValueError(f"Question is too long (max {limit} characters).")
        return self._run(question)

    def _run(self, text: str) -> TurnResult:
        s = self.services.settings
        state = {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "messages": [*self.history[-s.history_messages:], HumanMessage(content=text)],
            "current_query": text,
            "retrieved_context": [],
            "tool_calls": [],
            "source_evidence": [],
            "final_response": None,
        }
        try:
            final = self.services.graph.invoke(state, config={"recursion_limit": recursion_limit(s)})
        except GraphRecursionError:
            output = {
                "session_id": self.session_id,
                "candidate_id": self.candidate_id,
                "job_id": self.job_id,
                "response": "The agent stopped because it exceeded its step limit. Please try a narrower question.",
                "source_evidence": [],
                "retrieved_context": [],
            }
            return TurnResult(output)

        output = final["final_response"]
        self.history += [HumanMessage(content=text), AIMessage(content=output["response"])]
        return TurnResult(output, final.get("intent", "qa"), final.get("tool_calls", []))
