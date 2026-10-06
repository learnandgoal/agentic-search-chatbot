from __future__ import annotations

from hr_chatbot.tools import DocumentTools

RESUME = "candidates/candidate-001/resume.pdf"
JD = "jobs/job-001/job-description.docx"


def tools(store, settings):
    return DocumentTools(store, settings)


def test_search_returns_compact_hits_and_context(store, settings, scope):
    r = tools(store, settings).call("search", {"query": "airflow orchestration"}, scope, set())
    assert r.ok and r.payload["hits"]
    hit = r.payload["hits"][0]
    assert set(hit) == {
        "chunk_id", "document_id", "source_file", "source_type", "page_or_section", "score", "snippet", "citable",
    }
    assert len(hit["snippet"]) <= settings.snippet_chars
    assert r.context and r.context[0]["tool"] == "search" and r.context[0]["score"] is not None


def test_search_hits_are_not_citable(store, settings, scope):
    r = tools(store, settings).call("search", {"query": "airflow orchestration"}, scope, set())
    assert r.payload["hits"] and all(h["citable"] is False for h in r.payload["hits"])
    assert "not citable" in r.payload["note"]
    assert all(c["citable"] is False for c in r.context)


def test_only_text_returning_tools_are_citable(store, settings, scope):
    from hr_chatbot.tools import CITABLE_TOOLS

    assert CITABLE_TOOLS == {"navigate", "read", "grep"}
    t = tools(store, settings)
    chunk = next(c for c in store.document_chunks(RESUME) if c["page_or_section"] == "Page 1 - Summary")
    calls = {
        "navigate": {"chunk_id": chunk["chunk_id"], "direction": "next"},
        "read": {"chunk_id": chunk["chunk_id"]},
        "grep": {"document_id": RESUME, "pattern": "Python"},
    }
    for name, args in calls.items():
        r = t.call(name, args, scope, set())
        assert r.payload["citable"] is True and r.context and all(c["citable"] for c in r.context), name


def test_search_excludes_previously_seen(store, settings, scope):
    t = tools(store, settings)
    first = t.call("search", {"query": "python", "top_k": 3}, scope, set())
    seen = {h["chunk_id"] for h in first.payload["hits"]}
    second = t.call("search", {"query": "python", "top_k": 3}, scope, seen)
    assert not seen & {h["chunk_id"] for h in second.payload["hits"]}
    assert second.payload["excluded_previously_seen"] == len(seen)


def test_open_lists_sections_without_adding_context(store, settings, scope):
    r = tools(store, settings).call("open", {"document_id": RESUME}, scope, set())
    labels = [s["page_or_section"] for s in r.payload["sections"]]
    assert labels[:3] == ["Page 1 - Jordan Ellis", "Page 1 - Summary", "Page 1 - Experience"]
    assert r.context == []


def test_open_is_structure_only(store, settings, scope):
    r = tools(store, settings).call("open", {"document_id": RESUME}, scope, set())
    assert r.payload["citable"] is False and "read" in r.payload["note"]
    for section in r.payload["sections"]:
        assert set(section) == {"page_or_section", "first_chunk_id", "last_chunk_id", "num_chunks"}  # no text/preview
    document_text = " ".join(c["text"] for c in store.document_chunks(RESUME))
    assert not any(line.strip()[:30] in str(r.payload) for line in document_text.splitlines() if len(line.strip()) > 30)


def test_navigate_moves_within_bounds(store, settings, scope):
    t = tools(store, settings)
    chunks = store.document_chunks(RESUME)
    middle = next(c for c in chunks if c["page_or_section"] == "Page 1 - Summary")
    nxt = t.call("navigate", {"chunk_id": middle["chunk_id"], "direction": "next"}, scope, set())
    assert nxt.payload["results"][0]["page_or_section"] == "Page 1 - Experience"
    sec = t.call("navigate", {"chunk_id": middle["chunk_id"], "direction": "next_section"}, scope, set())
    assert sec.payload["results"][0]["page_or_section"] == "Page 1 - Experience"
    first = chunks[0]
    edge = t.call("navigate", {"chunk_id": first["chunk_id"], "direction": "previous", "steps": 3}, scope, set())
    assert edge.payload["results"] == [] and "boundary" in edge.payload
    many = t.call("navigate", {"chunk_id": first["chunk_id"], "direction": "next", "steps": 99}, scope, set())
    assert len(many.payload["results"]) <= 3


def test_read_by_section_and_by_chunk_with_limits(store, settings, scope):
    t = tools(store, settings)
    section = t.call("read", {"document_id": RESUME, "page_or_section": "page 2 - experience"}, scope, set())
    assert "Contoso" in section.payload["chunks"][0]["text"] and not section.payload["truncated"]
    chunk = next(c for c in store.document_chunks(RESUME) if c["page_or_section"] == "Page 1 - Experience")
    tiny = t.call("read", {"chunk_id": chunk["chunk_id"], "max_chars": 200}, scope, set())
    assert tiny.payload["truncated"] and len(tiny.payload["chunks"][0]["text"]) <= 201
    wide = t.call("read", {"chunk_id": chunk["chunk_id"], "before": 1, "after": 1}, scope, set())
    assert len(wide.payload["chunks"]) == 3 and wide.payload["next_chunk_id"]
    assert [c["tool"] for c in wide.context] == ["read"] * 3
    bad = t.call("read", {"document_id": RESUME, "page_or_section": "Nope"}, scope, set())
    assert "Available" in bad.payload["error"]
    assert not t.call("read", {}, scope, set()).ok  # needs chunk_id or document+section


def test_grep_is_literal_scoped_and_bounded(store, settings, scope):
    t = tools(store, settings)
    r = t.call("grep", {"document_id": RESUME, "pattern": "kafka"}, scope, set())
    assert {m["match"].lower() for m in r.payload["matches"]} == {"kafka"} and len(r.payload["matches"]) >= 2
    assert t.call("grep", {"document_id": RESUME, "pattern": "Zebra"}, scope, set()).payload["matches"] == []
    word = t.call("grep", {"document_id": RESUME, "pattern": "go", "whole_word": True}, scope, set())
    assert all(m["match"].lower() == "go" for m in word.payload["matches"])
    assert t.call("grep", {"document_id": RESUME, "pattern": "x" * 101}, scope, set()).payload["error"].startswith("Pattern too long")
    # regex metacharacters are treated literally (no ReDoS surface)
    assert t.call("grep", {"document_id": RESUME, "pattern": "(a+)+$"}, scope, set()).ok
    capped = t.call("grep", {"document_id": RESUME, "pattern": "e", "max_results": 500}, scope, set())
    assert len(capped.payload["matches"]) <= settings.grep_max_results


def test_other_candidates_documents_are_unreachable(store, settings, scope):
    t = tools(store, settings)
    other_doc = "candidates/candidate-002/resume.docx"
    other_chunk = store.document_chunks(other_doc)[0]["chunk_id"]
    other_job_chunk = store.document_chunks("jobs/job-002/job-description.docx")[0]["chunk_id"]
    for name, args in [
        ("open", {"document_id": other_doc}),
        ("grep", {"document_id": other_doc, "pattern": "SQL"}),
        ("read", {"document_id": other_doc, "page_or_section": "Skills"}),
        ("read", {"chunk_id": other_chunk}),
        ("read", {"chunk_id": other_job_chunk}),
        ("navigate", {"chunk_id": other_chunk, "direction": "next"}),
    ]:
        assert "error" in t.call(name, args, scope, set()).payload, name


def test_bad_calls_return_errors_instead_of_raising(store, settings, scope):
    t = tools(store, settings)
    assert not t.call("delete_everything", {}, scope, set()).ok
    assert not t.call("search", {"query": ""}, scope, set()).ok
    assert not t.call("open", {}, scope, set()).ok
