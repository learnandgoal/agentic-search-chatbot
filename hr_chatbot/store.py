"""One SQLite database holding everything the agent searches:

* ``documents`` / ``chunks``  - normal tables for metadata and text
* ``chunks_fts``              - FTS5 index for lexical search (BM25)
* ``chunks_vec``              - sqlite-vec ``vec0`` index for embedding search

``chunk_id`` is the rowid in all three chunk-level structures, so results join cleanly.
Every query is scoped to the selected candidate and job; the model never chooses the scope.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import sqlite_vec

from .chunking import ChunkDraft
from .config import Settings
from .embeddings import Embedder
from .parsing import normalize_ws

RRF_K = 60  # reciprocal-rank-fusion constant


class IndexMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scope:
    candidate_id: str
    job_id: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    candidate_id TEXT,
    job_id       TEXT,
    source_type  TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    num_chunks   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    candidate_id    TEXT,
    job_id          TEXT,
    source_type     TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    page_or_section TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    content_hash    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_candidate ON chunks(candidate_id);
CREATE INDEX IF NOT EXISTS idx_chunks_job ON chunks(job_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#._-]*")
_STOP = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "does", "do", "did", "what", "which", "who", "how", "has", "have", "this", "that", "it", "as", "at", "by",
}


def build_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 expression: quoted tokens joined by OR."""
    tokens: list[str] = []
    for raw in _WORD.findall(query):
        token = raw.strip("._-")
        if token and token.lower() not in _STOP:
            tokens.append('"' + token.replace('"', '""') + '"')
    return " OR ".join(dict.fromkeys(tokens))


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def snippet(text: str, limit: int) -> str:
    flat = normalize_ws(text)
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


class HRStore:
    def __init__(self, db_path: Path, embedder: Embedder, settings: Settings) -> None:
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.settings = settings
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ connection
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation keeps Streamlit's thread model simple.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        signature = f"{self.embedder.name}|{self.embedder.dim}"
        with self._conn() as c:
            c.executescript(SCHEMA)
            row = c.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
            if row is None:
                c.execute("INSERT INTO meta(key, value) VALUES ('embedder', ?)", (signature,))
                c.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
                    f"USING vec0(embedding float[{int(self.embedder.dim)}] distance_metric=cosine)"
                )
            elif row["value"] != signature:
                raise IndexMismatchError(
                    f"Index was built with embedder '{row['value']}' but '{signature}' is configured. "
                    "Rebuild it with `python -m hr_chatbot.ingest --rebuild`."
                )

    # ------------------------------------------------------------------ writes
    def document_hash(self, document_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT content_hash FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return row["content_hash"] if row else None

    def replace_document(
        self,
        meta: dict,
        chunks: Sequence[ChunkDraft],
        vectors: np.ndarray,
    ) -> None:
        """Atomically (re)index one document. ``meta`` keys: document_id, candidate_id, job_id,
        source_type, source_file, content_hash."""
        with self._conn() as c:
            self._delete_document(c, meta["document_id"])
            c.execute(
                "INSERT INTO documents(document_id, candidate_id, job_id, source_type, source_file, "
                "content_hash, num_chunks) VALUES (?,?,?,?,?,?,?)",
                (
                    meta["document_id"], meta["candidate_id"], meta["job_id"], meta["source_type"],
                    meta["source_file"], meta["content_hash"], len(chunks),
                ),
            )
            for draft, vector in zip(chunks, vectors):
                cur = c.execute(
                    "INSERT INTO chunks(document_id, candidate_id, job_id, source_type, source_file, "
                    "page_or_section, chunk_index, text, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        meta["document_id"], meta["candidate_id"], meta["job_id"], meta["source_type"],
                        meta["source_file"], draft.page_or_section, draft.chunk_index, draft.text,
                        meta["content_hash"],
                    ),
                )
                chunk_id = cur.lastrowid
                c.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, draft.text))
                c.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (chunk_id, _blob(vector)))

    def delete_document(self, document_id: str) -> None:
        with self._conn() as c:
            self._delete_document(c, document_id)

    @staticmethod
    def _delete_document(c: sqlite3.Connection, document_id: str) -> None:
        ids = [r[0] for r in c.execute("SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,))]
        for chunk_id in ids:
            c.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
            c.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
        c.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        c.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def document_ids_for(self, *, candidate_id: str | None = None, job_id: str | None = None) -> list[str]:
        with self._conn() as c:
            if candidate_id is not None:
                rows = c.execute("SELECT document_id FROM documents WHERE candidate_id = ?", (candidate_id,))
            else:
                rows = c.execute("SELECT document_id FROM documents WHERE job_id = ?", (job_id,))
            return [r[0] for r in rows]

    # ------------------------------------------------------------------ reads
    @staticmethod
    def in_scope(row: sqlite3.Row | dict, scope: Scope) -> bool:
        return row["candidate_id"] == scope.candidate_id or row["job_id"] == scope.job_id

    def get_document(self, document_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def get_chunk(self, chunk_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM chunks WHERE chunk_id = ?", (int(chunk_id),)).fetchone()
        return dict(row) if row else None

    def document_chunks(self, document_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            return {
                "documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "embedder": self.embedder.name,
            }

    # ------------------------------------------------------------------ hybrid search
    def search(self, scope: Scope, query: str, top_k: int, exclude_ids: Iterable[int] = ()) -> list[dict]:
        top_k = max(1, min(int(top_k), self.settings.max_top_k))
        exclude = {int(i) for i in exclude_ids}
        pool = max(top_k * 4, 20)

        with self._conn() as c:
            rows = c.execute(
                "SELECT chunk_id FROM chunks WHERE candidate_id = ? OR job_id = ?",
                (scope.candidate_id, scope.job_id),
            ).fetchall()
            allowed = [r[0] for r in rows if r[0] not in exclude]
            if not allowed:
                return []

            lexical = self._lexical_ids(c, scope, query, pool + len(exclude), exclude)
            semantic = self._semantic_ids(c, query, allowed, pool)

            fused: dict[int, float] = defaultdict(float)
            for ranking in (lexical, semantic):
                for rank, chunk_id in enumerate(ranking):
                    fused[chunk_id] += 1.0 / (RRF_K + rank + 1)
            best = 2.0 / (RRF_K + 1)  # rank 1 in both retrievers
            ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            if not ordered:
                return []

            marks = ",".join("?" * len(ordered))
            meta = {
                r["chunk_id"]: r
                for r in c.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", [i for i, _ in ordered])
            }

        hits = []
        for chunk_id, score in ordered:
            row = meta[chunk_id]
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": row["document_id"],
                    "source_file": row["source_file"],
                    "source_type": row["source_type"],
                    "page_or_section": row["page_or_section"],
                    "score": round(score / best, 3),  # 1.0 = top rank in both retrievers; not a calibrated probability
                    "snippet": snippet(row["text"], self.settings.snippet_chars),
                    "text": row["text"],
                }
            )
        return hits

    @staticmethod
    def _lexical_ids(
        c: sqlite3.Connection, scope: Scope, query: str, limit: int, exclude: set[int]
    ) -> list[int]:
        expression = build_fts_query(query)
        if not expression:
            return []
        rows = c.execute(
            "SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts) AS s "
            "FROM chunks_fts JOIN chunks ch ON ch.chunk_id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND (ch.candidate_id = ? OR ch.job_id = ?) "
            "ORDER BY s LIMIT ?",
            (expression, scope.candidate_id, scope.job_id, limit),
        ).fetchall()
        return [r["chunk_id"] for r in rows if r["chunk_id"] not in exclude]

    def _semantic_ids(self, c: sqlite3.Connection, query: str, allowed: list[int], limit: int) -> list[int]:
        vector = self.embedder.embed([query])[0]
        marks = ",".join("?" * len(allowed))
        rows = c.execute(
            f"SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? AND rowid IN ({marks}) "
            f"ORDER BY distance",
            [_blob(vector), min(limit, len(allowed)), *allowed],
        ).fetchall()
        return [r["rowid"] for r in rows]
