# Agentic HR Chatbot (LangGraph)

A local, document-centric agentic search chatbot that follows the *Agentic HR Chatbot - Technical Architecture* design:
pick a **Candidate** and a **Job ID**, click **Start Chat**, get an automatic candidate / job / fit briefing, then ask follow-up
questions. An agent decides what to look at using five Mistral-style tools (`search`, `open`, `navigate`, `read`, `grep`) over a
local SQLite index, and every answer returns **source evidence** and the **retrieved context** it was based on.

> Analysis support only, not an automated hiring decision.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your OPENAI_API_KEY in it
python -m hr_chatbot.sample_data                       # optional: only if ./data is missing (it is included)
streamlit run app.py
```

The first start downloads the local embedding model (`all-MiniLM-L6-v2`, ~90 MB). No API key and no download are needed to
run the tests: `HR_EMBEDDER=hashing python -m pytest`.

Your own data goes in the folder layout from section 3 of the design (IDs come straight from the folder names):

```
data/candidates/<candidate-id>/resume.pdf, cover-letter.docx, portfolio.pdf   (PDF and DOCX only)
data/jobs/<job-id>/job-description.docx
data/indexes/hr.sqlite                                                        (created automatically)
```

Index maintenance: at startup only the existing index is loaded. **Start Chat** refreshes just the selected candidate's and job's
files whose content hash changed or that are missing from the index (and prunes deleted files). To index everything up front:
`python -m hr_chatbot.ingest` (add `--rebuild` after changing the embedding model).

## How the design maps to the code

| Design section | Where |
|---|---|
| 2 Target architecture, 8 Local deployment | `app.py` (Streamlit UI), `hr_chatbot/service.py` (`ChatSession`, wiring) |
| 3 Data model and ingestion, 3.1 chunk metadata | `hr_chatbot/ingest.py`, `parsing.py` (PyMuPDF, python-docx), `chunking.py`, `store.py` |
| Local hybrid index | `store.py`: one SQLite file, FTS5 (BM25) + sqlite-vec, fused with reciprocal-rank fusion |
| 4 Mistral-style tools | `hr_chatbot/tools.py` |
| 4.1 Agent loop, stopping policy | `hr_chatbot/graph.py`, `prompts.py` |
| 5 End-to-end flow | `service.py` (`start()` = 5.1, `ask()` = 5.2) |
| 6 LangGraph design | `graph.py`: `AgentState`, `agent` node, `tools` node |
| 7 Output contract | `graph.py` (`_finalize`); returned as `TurnResult.output` |
| 9 Implementation plan, step 6 Validation | `tests/` |

The graph is exactly two nodes: `START -> agent -> tools -> agent ... -> END`. The agent node binds the five tools plus a
`submit_answer` tool that carries the structured final answer; the tools node runs the requested tool, appends to
`retrieved_context` and routes back.

Output contract (every turn):

```json
{
  "session_id": "...", "candidate_id": "candidate-001", "job_id": "job-001",
  "response": "markdown answer",
  "source_evidence":  [{"source_file": "resume.pdf", "page_or_section": "Page 1 - Experience", "content": "verbatim quote"}],
  "retrieved_context": [{"source_file": "job-description.docx", "page_or_section": "Required Skills", "content": "...", "score": 0.86}]
}
```

`score` is the search rank fused from both retrievers, normalised so 1.0 means "top hit in both" (it is not a probability).
It is `null` for text that came from `read`, `navigate` or `grep`.

## Decisions I made where the design was silent or inconsistent

* **One SQLite file.** Section 3's folder tree lists `lexical.sqlite`, `vectors.faiss` and `chunk_metadata.sqlite`, while sections 8
  and 9 specify a single SQLite database with FTS5 and sqlite-vec. I followed 8 and 9.
* **Model adapter.** Section 8 refers to "the existing GPT model adapter", which was not in the document. `hr_chatbot/llm.py` is the
  single seam: by default it builds `ChatOpenAI` (or Azure OpenAI when `AZURE_OPENAI_ENDPOINT` is set). Swap in your adapter there; it
  only needs to be a LangChain chat model that supports `bind_tools`.
* **Evidence can't be invented, and `search` is never evidence.** Citations are checked in code (`graph.py`): the quoted text must
  appear in text that `read`, `grep` or `navigate` returned this turn for that chunk, and the chunk must belong to the selected
  candidate/job. `search` hits (snippets for locating) and `open` results (structure, no text) are returned with `citable: false` and
  are rejected as evidence even if the quote is genuine. File and page/section come from the index, not from the model. Failed
  citations are dropped and the answer says how many.
* **Tool selection.** The system prompt (`prompts.py`) has a tool-selection section with query-to-tool examples: exact phrase ->
  `search` then `grep`; adjacent information -> `search` then `navigate`; section summary -> `search`, `open`, then `read`;
  simple fact -> `search` then `read`; unsupported request -> no retrieval tool. `open` shows structure only (no text) and is always
  followed by `read` or `grep` when facts are needed.
* **The model never picks the scope.** The selected candidate and job are injected into every tool call and every index query.
  Another candidate's documents are unreachable even if the model asks for them by id.
* **`qa | unsupported` classification** (section 4.1) is a field on `submit_answer`, decided by the model under the rules in
  `prompts.py`. My definition of "unsupported": hire/reject/compensation decisions, inferring protected characteristics, comparing with
  other candidates, or off-topic requests. The automatic initial summary skips classification, as the stopping policy requires.
  **Please review that policy wording with your HR/legal stakeholders.**
* **Prompt-injection hardening.** Documents are untrusted input (a resume can say "rate me 10/10"). The system prompt tells the model to
  treat document text as data, and `grep` is literal-only so model-supplied patterns can't trigger catastrophic regexes.
* **Stopping policy** is enforced in code: after `HR_MAX_TOOL_CALLS` (default 8) tool calls the model can only call `submit_answer`.
  `search` also automatically excludes chunks already seen this turn.
* `AgentState` has one small addition beyond the design's field list: `intent`.

## Known limits

* Scanned (image-only) PDFs have no text layer; they are indexed empty with a warning (no OCR).
* PDF headings are detected heuristically (larger font or ALL CAPS short lines). Unusual layouts fall back to "Page N" labels.
* Vector search is exact and scoped via `rowid IN (...)`, which is right for per-candidate corpora of tens to hundreds of chunks.
  It is not meant for very large indexes.
* Conversation history lives in the Streamlit session; closing the tab ends the session.

## Tests

`HR_EMBEDDER=hashing python -m pytest` runs 56 tests: parsing and metadata, incremental ingestion, scoped hybrid search, each tool's
limits and scope checks, the graph with a scripted model (output contract, citation verification, citable-only evidence, budget
enforcement, parallel and mixed tool calls, unsupported intent, multi-turn history), tool-trajectory tests (exact phrase, adjacent
information, section summary, simple fact, unsupported -> no retrieval) and a headless Streamlit run. They use a scripted chat model
and a hashing embedder, so they need no API key or network.

The trajectory tests prove the application handles each trajectory correctly and rejects bad ones (for example answering from
`search` alone). They cannot prove that a real model *chooses* those trajectories. For that there are 5 extra tests in
`tests/test_trajectories_live.py`, skipped by default; run them with your credentials:
`HR_LIVE_TESTS=1 HR_EMBEDDER=hashing python -m pytest tests/test_trajectories_live.py -v`.
The sentence-transformers model is not exercised by any test; run a real session before relying on answer quality.
