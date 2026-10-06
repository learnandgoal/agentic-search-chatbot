"""Offline ingestion: candidate/job folders -> parsed -> chunked -> embedded -> SQLite.

Incremental by design: a file is (re)indexed only when its content hash changed or its index
entry is missing, and index entries whose file disappeared are pruned.

    python -m hr_chatbot.ingest            # refresh everything
    python -m hr_chatbot.ingest --rebuild  # drop the database first (needed if the embedder changes)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_sections
from .config import Settings, load_settings
from .embeddings import Embedder, get_embedder
from .parsing import ParseError, parse_file
from .store import HRStore, index_text

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: "IngestReport") -> None:
        self.indexed += other.indexed
        self.skipped += other.skipped
        self.removed += other.removed
        self.warnings += other.warnings


def _list_dirs(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def list_candidates(data_dir: Path) -> list[str]:
    return _list_dirs(Path(data_dir) / "candidates")


def list_jobs(data_dir: Path) -> list[str]:
    return _list_dirs(Path(data_dir) / "jobs")


def classify_source(stem: str, kind: str) -> str:
    s = stem.lower()
    if kind == "jobs":
        return "job_description"
    if "resume" in s or s in {"cv", "curriculum-vitae"}:
        return "resume"
    if "cover" in s:
        return "cover_letter"
    if "portfolio" in s:
        return "portfolio"
    return "other"


class Ingestor:
    def __init__(self, store: HRStore, data_dir: Path, settings: Settings) -> None:
        self.store = store
        self.data_dir = Path(data_dir)
        self.settings = settings
        self.embedder: Embedder = store.embedder

    # -- public API ------------------------------------------------------------
    def ingest_scope(self, candidate_id: str, job_id: str) -> IngestReport:
        """Refresh only the selected candidate's and job's files (what Start Chat does)."""
        if candidate_id not in list_candidates(self.data_dir):
            raise ValueError(f"Unknown candidate: {candidate_id!r}")
        if job_id not in list_jobs(self.data_dir):
            raise ValueError(f"Unknown job: {job_id!r}")
        report = IngestReport()
        report.merge(self._ingest_folder("candidates", candidate_id))
        report.merge(self._ingest_folder("jobs", job_id))
        return report

    def ingest_all(self) -> IngestReport:
        report = IngestReport()
        for candidate_id in list_candidates(self.data_dir):
            report.merge(self._ingest_folder("candidates", candidate_id))
        for job_id in list_jobs(self.data_dir):
            report.merge(self._ingest_folder("jobs", job_id))
        return report

    # -- internals -------------------------------------------------------------
    def _ingest_folder(self, kind: str, owner_id: str) -> IngestReport:
        report = IngestReport()
        folder = self.data_dir / kind / owner_id
        is_candidate = kind == "candidates"
        present: set[str] = set()

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith(("~$", ".")):
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                report.warnings.append(f"{kind}/{owner_id}/{path.name}: unsupported file type, skipped")
                continue

            document_id = f"{kind}/{owner_id}/{path.name}"
            present.add(document_id)
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.store.document_hash(document_id) == content_hash:
                report.skipped.append(document_id)
                continue

            try:
                sections = parse_file(path)
            except ParseError as exc:
                self.store.delete_document(document_id)  # never keep a stale version of a broken file
                report.warnings.append(f"{document_id}: {exc}")
                continue

            drafts = chunk_sections(sections, self.settings.chunk_size, self.settings.chunk_overlap)
            if not drafts:
                report.warnings.append(
                    f"{document_id}: no extractable text (scanned PDF? OCR is not supported), indexed empty"
                )
            source_type = classify_source(path.stem, kind)
            # Embed each chunk together with its source and heading, so a query like "summary" or "required skills" can find it.
            vectors = self.embedder.embed([index_text(source_type, d.page_or_section, d.text) for d in drafts]) if drafts else []
            self.store.replace_document(
                {
                    "document_id": document_id,
                    "candidate_id": owner_id if is_candidate else None,
                    "job_id": None if is_candidate else owner_id,
                    "source_type": source_type,
                    "source_file": path.name,
                    "content_hash": content_hash,
                },
                drafts,
                vectors,
            )
            report.indexed.append(document_id)

        known = self.store.document_ids_for(**({"candidate_id": owner_id} if is_candidate else {"job_id": owner_id}))
        for document_id in known:
            if document_id not in present:
                self.store.delete_document(document_id)
                report.removed.append(document_id)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index candidate and job documents.")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing index first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = load_settings()
    if args.rebuild and settings.db_path.exists():
        settings.db_path.unlink()
    store = HRStore(settings.db_path, get_embedder(), settings)
    report = Ingestor(store, settings.data_dir, settings).ingest_all()
    print(f"indexed={len(report.indexed)} skipped={len(report.skipped)} removed={len(report.removed)}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    print(store.stats())


if __name__ == "__main__":
    main()
