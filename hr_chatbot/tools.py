"""The five Mistral-style document tools: search, open, navigate, read, grep.

Division of labour (as in the design): ``search`` *locates* (compact hits), the other four *explore*.
Tools never raise to the model: problems come back as ``{"error": ...}`` so the agent can recover.
The scope (candidate_id, job_id) is injected by the application, never chosen by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from .config import Settings
from .parsing import normalize_ws
from .schemas import (
    SUBMIT_NAME, GrepArgs, NavigateArgs, OpenArgs, ReadArgs, SearchArgs, SubmitAnswer,
)
from .store import HRStore, Scope, snippet

NOT_FOUND = "Unknown document_id/chunk_id for the selected candidate and job. Use ids returned by search."


@dataclass
class ToolResult:
    payload: dict[str, Any]  # JSON handed back to the model
    context: list[dict] = field(default_factory=list)  # items appended to retrieved_context

    @property
    def ok(self) -> bool:
        return "error" not in self.payload


def _ctx(row: dict, content: str, tool: str, score: float | None = None) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "source_file": row["source_file"],
        "page_or_section": row["page_or_section"],
        "content": content,
        "score": score,
        "tool": tool,
    }


class DocumentTools:
    def __init__(self, store: HRStore, settings: Settings) -> None:
        self.store = store
        self.s = settings
        self._impl: dict[str, tuple[type, Callable[..., ToolResult]]] = {
            "search": (SearchArgs, self.search),
            "open": (OpenArgs, self.open),
            "navigate": (NavigateArgs, self.navigate),
            "read": (ReadArgs, self.read),
            "grep": (GrepArgs, self.grep),
        }

    # ---------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict, scope: Scope, seen: set[int]) -> ToolResult:
        if name not in self._impl:
            return ToolResult({"error": f"Unknown tool {name!r}. Available: {', '.join(self._impl)}."})
        model, fn = self._impl[name]
        try:
            parsed = model(**(args or {}))
        except (ValidationError, TypeError) as exc:
            return ToolResult({"error": f"Invalid arguments for {name}: {exc}"})
        return fn(parsed, scope, seen)

    # ---------------------------------------------------------------- helpers
    def _document(self, document_id: str, scope: Scope) -> dict | None:
        doc = self.store.get_document(document_id)
        return doc if doc and self.store.in_scope(doc, scope) else None

    def _chunk(self, chunk_id: int, scope: Scope) -> dict | None:
        row = self.store.get_chunk(chunk_id)
        return row if row and self.store.in_scope(row, scope) else None

    # ---------------------------------------------------------------- search
    def search(self, a: SearchArgs, scope: Scope, seen: set[int]) -> ToolResult:
        hits = self.store.search(scope, a.query, a.top_k, set(a.exclude_ids) | seen)
        payload: dict[str, Any] = {
            "hits": [{k: v for k, v in h.items() if k != "text"} for h in hits],
            "excluded_previously_seen": len(seen),
        }
        if not hits:
            payload["note"] = (
                "No new matching chunks. Try different terms, or stop and state that the documents do not contain it."
            )
        return ToolResult(payload, [_ctx(h, h["snippet"], "search", h["score"]) for h in hits])

    # ---------------------------------------------------------------- open
    def open(self, a: OpenArgs, scope: Scope, seen: set[int]) -> ToolResult:
        doc = self._document(a.document_id, scope)
        if not doc:
            return ToolResult({"error": NOT_FOUND})
        sections: list[dict] = []
        for chunk in self.store.document_chunks(a.document_id):
            if sections and sections[-1]["page_or_section"] == chunk["page_or_section"]:
                sections[-1]["last_chunk_id"] = chunk["chunk_id"]
                sections[-1]["num_chunks"] += 1
            else:
                sections.append(
                    {
                        "page_or_section": chunk["page_or_section"],
                        "first_chunk_id": chunk["chunk_id"],
                        "last_chunk_id": chunk["chunk_id"],
                        "num_chunks": 1,
                        "preview": snippet(chunk["text"], 120),
                    }
                )
        return ToolResult(
            {
                "document_id": doc["document_id"],
                "source_file": doc["source_file"],
                "source_type": doc["source_type"],
                "num_chunks": doc["num_chunks"],
                "sections": sections,
            }
        )  # structure only: nothing here counts as retrieved context

    # ---------------------------------------------------------------- navigate
    def navigate(self, a: NavigateArgs, scope: Scope, seen: set[int]) -> ToolResult:
        origin = self._chunk(a.chunk_id, scope)
        if not origin:
            return ToolResult({"error": NOT_FOUND})
        chunks = self.store.document_chunks(origin["document_id"])
        pos = next(i for i, c in enumerate(chunks) if c["chunk_id"] == origin["chunk_id"])

        if a.direction in ("next", "previous"):
            steps = min(a.steps, 3)
            span = range(pos + 1, pos + 1 + steps) if a.direction == "next" else range(pos - 1, pos - 1 - steps, -1)
            targets = [chunks[i] for i in span if 0 <= i < len(chunks)]
        else:
            labels = list(dict.fromkeys(c["page_or_section"] for c in chunks))
            idx = labels.index(origin["page_or_section"]) + (1 if a.direction == "next_section" else -1)
            targets = []
            if 0 <= idx < len(labels):
                target_label = labels[idx]
                first = next(c for c in chunks if c["page_or_section"] == target_label)
                targets = [first]

        limit = self.s.navigate_snippet_chars
        results = [
            {
                "chunk_id": t["chunk_id"],
                "document_id": t["document_id"],
                "source_file": t["source_file"],
                "page_or_section": t["page_or_section"],
                "snippet": snippet(t["text"], limit),
            }
            for t in targets
        ]
        payload: dict[str, Any] = {"from_chunk_id": origin["chunk_id"], "direction": a.direction, "results": results}
        if not results:
            payload["boundary"] = "Already at the edge of this document; nothing further in that direction."
        return ToolResult(payload, [_ctx(t, snippet(t["text"], limit), "navigate") for t in targets])

    # ---------------------------------------------------------------- read
    def read(self, a: ReadArgs, scope: Scope, seen: set[int]) -> ToolResult:
        if a.chunk_id is not None:
            origin = self._chunk(a.chunk_id, scope)
            if not origin:
                return ToolResult({"error": NOT_FOUND})
            chunks = self.store.document_chunks(origin["document_id"])
            pos = next(i for i, c in enumerate(chunks) if c["chunk_id"] == origin["chunk_id"])
            window = chunks[max(0, pos - min(a.before, 5)): pos + min(a.after, 5) + 1]
        else:
            doc = self._document(a.document_id or "", scope)
            if not doc:
                return ToolResult({"error": NOT_FOUND})
            chunks = self.store.document_chunks(doc["document_id"])
            wanted = (a.page_or_section or "").strip().lower()
            window = [c for c in chunks if c["page_or_section"].lower() == wanted]
            if not window:
                labels = list(dict.fromkeys(c["page_or_section"] for c in chunks))
                return ToolResult({"error": f"No section {a.page_or_section!r}. Available: {labels}"})

        budget = max(200, min(a.max_chars, self.s.read_max_chars))
        out_chunks: list[dict] = []
        used = 0
        truncated = False
        for c in window:
            remaining = budget - used
            if remaining <= 0:
                truncated = True
                break
            text = c["text"]
            if len(text) > remaining:
                text, truncated = text[:remaining].rstrip() + "…", True
            out_chunks.append({"chunk_id": c["chunk_id"], "page_or_section": c["page_or_section"], "text": text})
            used += len(text)

        last_included = out_chunks[-1]["chunk_id"]
        all_ids = [c["chunk_id"] for c in chunks]
        after_idx = all_ids.index(last_included) + 1
        payload = {
            "document_id": window[0]["document_id"],
            "source_file": window[0]["source_file"],
            "chunks": out_chunks,
            "truncated": truncated,
            "next_chunk_id": all_ids[after_idx] if after_idx < len(all_ids) else None,
        }
        by_id = {c["chunk_id"]: c for c in window}
        return ToolResult(payload, [_ctx(by_id[o["chunk_id"]], o["text"], "read") for o in out_chunks])

    # ---------------------------------------------------------------- grep
    def grep(self, a: GrepArgs, scope: Scope, seen: set[int]) -> ToolResult:
        doc = self._document(a.document_id, scope)
        if not doc:
            return ToolResult({"error": NOT_FOUND})
        pattern = normalize_ws(a.pattern)
        if not pattern:
            return ToolResult({"error": "Empty pattern."})
        if len(pattern) > self.s.grep_pattern_max_len:
            return ToolResult({"error": f"Pattern too long (max {self.s.grep_pattern_max_len} characters)."})

        limit = min(a.max_results, self.s.grep_max_results)
        ctx = min(a.context_chars, 200)
        # Literal matching only (re.escape): model-supplied patterns can't trigger catastrophic regexes.
        regex = re.compile((r"\b%s\b" if a.whole_word else "%s") % re.escape(pattern), re.IGNORECASE)

        matches: list[dict] = []
        context: list[dict] = []
        truncated = False
        for chunk in self.store.document_chunks(doc["document_id"]):
            flat = normalize_ws(chunk["text"])
            for m in regex.finditer(flat):
                if len(matches) >= limit:
                    truncated = True
                    break
                around = flat[max(0, m.start() - ctx): m.end() + ctx]
                matches.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "page_or_section": chunk["page_or_section"],
                        "match": m.group(0),
                        "context": around,
                    }
                )
                context.append(_ctx(chunk, around, "grep"))
            if truncated:
                break
        payload: dict[str, Any] = {"document_id": doc["document_id"], "pattern": pattern, "matches": matches, "truncated": truncated}
        if not matches:
            payload["note"] = "No occurrences in this document."
        return ToolResult(payload, context)


# --------------------------------------------------------------------------- schemas for the model

def _stub(**_: Any) -> str:  # tools are executed by the graph's tools node, not by LangChain
    raise RuntimeError("Executed by the LangGraph tools node.")


def _spec(name: str, description: str, schema: type) -> StructuredTool:
    return StructuredTool.from_function(func=_stub, name=name, description=description, args_schema=schema)


DOC_TOOL_SPECS = [
    _spec(
        "search",
        "Find where relevant information lives across the selected candidate's and job's documents. Hybrid keyword + "
        "semantic retrieval. Returns compact hits (chunk_id, document_id, source_file, page_or_section, score, short "
        "snippet), not full text. Follow up with open/navigate/read/grep to verify details.",
        SearchArgs,
    ),
    _spec(
        "open",
        "Inspect one document found via search: its sections/pages with chunk id ranges and a short preview of each.",
        OpenArgs,
    ),
    _spec(
        "navigate",
        "Move to neighbouring chunks (next/previous) or sections (next_section/previous_section) around a known "
        "chunk_id without running a new search.",
        NavigateArgs,
    ),
    _spec(
        "read",
        "Read a larger range of full text: a chunk with neighbours, or a whole page/section of one document. "
        "Output is size-limited and keeps chunk_id/section metadata for citation.",
        ReadArgs,
    ),
    _spec(
        "grep",
        "Find exact words or phrases (skills, employers, dates, certifications) inside ONE document, with context.",
        GrepArgs,
    ),
]

SUBMIT_TOOL = _spec(
    SUBMIT_NAME,
    "Deliver your final answer to the recruiter. Call this exactly once, on its own, when you have enough evidence "
    "(or when no further search is likely to help, or the tool budget is used up).",
    SubmitAnswer,
)
