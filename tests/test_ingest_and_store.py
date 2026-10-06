from __future__ import annotations

from hr_chatbot.embeddings import HashingEmbedder
from hr_chatbot.ingest import Ingestor, list_candidates, list_jobs
from hr_chatbot.sample_data import write_docx
from hr_chatbot.store import HRStore, IndexMismatchError, Scope, build_fts_query

import pytest


def test_folders_drive_dropdown_ids(settings, env):
    assert list_candidates(settings.data_dir) == ["candidate-001", "candidate-002", "candidate-003", "candidate-004"]
    assert list_jobs(settings.data_dir) == ["job-001", "job-002", "job-003"]


def test_parsing_labels_and_metadata(store):
    chunks = store.document_chunks("candidates/candidate-001/resume.pdf")
    labels = [c["page_or_section"] for c in chunks]
    assert "Page 1 - Experience" in labels and "Page 2 - Skills" in labels  # heading carried across the page break
    assert "Page 2 - Certifications" in labels and "Page 2 - Education" in labels
    first = chunks[0]
    for key in ("chunk_id", "document_id", "source_type", "source_file", "candidate_id", "job_id",
                "page_or_section", "chunk_index", "text", "content_hash"):
        assert key in first
    assert first["source_type"] == "resume" and first["job_id"] is None
    docx = store.document_chunks("jobs/job-001/job-description.docx")
    assert {"Required Skills", "Preferred Skills"} <= {c["page_or_section"] for c in docx}
    assert docx[0]["candidate_id"] is None and docx[0]["job_id"] == "job-001"


def test_ingestion_is_incremental(settings, env):
    store, ingestor = env
    again = ingestor.ingest_scope("candidate-001", "job-001")
    assert not again.indexed and len(again.skipped) == 4

    # change one file -> only that file is re-indexed
    path = settings.data_dir / "jobs/job-001/job-description.docx"
    write_docx(path, [("title", "Senior Python Engineer"), ("h", "Required Skills"), ("p", "Rust and Go expert.")])
    changed = ingestor.ingest_scope("candidate-001", "job-001")
    assert changed.indexed == ["jobs/job-001/job-description.docx"] and len(changed.skipped) == 3
    texts = " ".join(c["text"] for c in store.document_chunks("jobs/job-001/job-description.docx"))
    assert "Rust and Go expert" in texts and "Airflow" not in texts

    # delete a file -> pruned from the index
    (settings.data_dir / "candidates/candidate-001/portfolio.pdf").unlink()
    pruned = ingestor.ingest_scope("candidate-001", "job-001")
    assert pruned.removed == ["candidates/candidate-001/portfolio.pdf"]
    assert store.get_document("candidates/candidate-001/portfolio.pdf") is None


def test_corrupt_and_unsupported_files_warn_not_crash(settings, env):
    store, ingestor = env
    (settings.data_dir / "candidates/candidate-001/broken.pdf").write_bytes(b"not a pdf")
    (settings.data_dir / "candidates/candidate-001/notes.txt").write_text("hello")
    report = ingestor.ingest_scope("candidate-001", "job-001")
    assert any("broken.pdf" in w for w in report.warnings)
    assert any("notes.txt" in w and "unsupported" in w for w in report.warnings)


def test_unknown_scope_is_rejected(env):
    _, ingestor = env
    with pytest.raises(ValueError):
        ingestor.ingest_scope("../etc", "job-001")


def test_hybrid_search_finds_relevant_chunks(store, scope):
    hits = store.search(scope, "kafka streaming", 5)
    assert hits and all(0 <= h["score"] <= 1 for h in hits)
    assert any("Kafka" in h["text"] and h["source_file"] == "resume.pdf" for h in hits)
    # lexical-only vocabulary still works through stemming ("developing"/"developed")
    assert any("PostgreSQL" in h["text"] for h in store.search(scope, "postgresql schema", 3))


def test_search_is_scoped_to_selected_candidate_and_job(store):
    hits = store.search(Scope("candidate-002", "job-002"), "kafka streaming airflow", 10)
    assert hits
    assert {h["document_id"].split("/")[1] for h in hits} <= {"candidate-002", "job-002"}


def test_exclude_ids_avoid_repeats(store, scope):
    first = store.search(scope, "python experience", 3)
    second = store.search(scope, "python experience", 3, {h["chunk_id"] for h in first})
    assert second and not ({h["chunk_id"] for h in first} & {h["chunk_id"] for h in second})


def test_top_k_is_capped(store, scope):
    assert len(store.search(scope, "experience", 500)) <= store.settings.max_top_k


def test_fts_query_is_injection_safe():
    assert build_fts_query('kafka" OR secret:* NEAR(') == '"kafka" OR "secret" OR "NEAR"'
    assert build_fts_query("the of and") == ""


def test_weird_queries_do_not_crash(store, scope):
    assert isinstance(store.search(scope, '")(*&^%$ NOT', 3), list)
    assert store.search(scope, "zzzzqqq", 3) is not None


def test_embedder_mismatch_is_detected(settings, env):
    class Other(HashingEmbedder):
        name = "other"

    with pytest.raises(IndexMismatchError):
        HRStore(settings.db_path, Other(), settings)


def test_search_matches_section_headings_but_returns_plain_chunk_text(store):
    hits = store.search(Scope("candidate-001", "job-001"), "Summary", 3)
    assert any(h["page_or_section"] == "Page 1 - Summary" for h in hits)
    chunk = next(c for c in store.document_chunks("candidates/candidate-001/resume.pdf") if c["page_or_section"] == "Page 1 - Summary")
    assert chunk["text"].startswith("Data platform engineer")  # the heading is indexed, never stored in the text


def test_an_index_from_an_older_format_asks_for_a_rebuild(settings, env):
    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    conn.execute("DELETE FROM meta WHERE key = 'index_format'")
    conn.commit(); conn.close()
    from hr_chatbot.embeddings import HashingEmbedder
    from hr_chatbot.store import HRStore, IndexMismatchError
    with pytest.raises(IndexMismatchError, match="older version"):
        HRStore(settings.db_path, HashingEmbedder(), settings)
