"""PDF (PyMuPDF) and DOCX (python-docx) parsing into labelled sections.

A section label is what the design calls ``page_or_section``:
  * PDF  -> "Page 2 - Experience" (page number plus the nearest heading above)
  * DOCX -> "Required Skills"     (the heading text; "Document" when there is none)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass
class Section:
    label: str
    text: str


class ParseError(Exception):
    pass


# ----------------------------------------------------------------------------- PDF

_MAX_HEADING_LEN = 60


def _is_pdf_heading(text: str, size: float, body_size: float) -> bool:
    if not text or len(text) > _MAX_HEADING_LEN or text.endswith((".", ",", ";")):
        return False
    if size >= body_size * 1.15:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def parse_pdf(path: Path) -> list[Section]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # corrupt / encrypted / not a PDF
        raise ParseError(f"cannot open PDF: {exc}") from exc

    with doc:
        pages: list[list[tuple[str, float]]] = []
        weight: Counter[float] = Counter()
        for page in doc:
            lines: list[tuple[str, float]] = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    size = round(max(s["size"] for s in spans), 1)
                    lines.append((text, size))
                    weight[size] += len(text)
            pages.append(lines)

    if not weight:
        return []  # no extractable text (e.g. scanned PDF, no OCR in this version)
    body_size = weight.most_common(1)[0][0]

    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    buf_page = 1

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            label = f"Page {buf_page} - {heading}" if heading else f"Page {buf_page}"
            sections.append(Section(label, text))
        buf.clear()

    for page_no, lines in enumerate(pages, start=1):
        flush()
        buf_page = page_no
        for text, size in lines:
            if _is_pdf_heading(text, size, body_size):
                flush()
                heading = text
                buf_page = page_no
            else:
                buf.append(text)
    flush()
    return sections


# ---------------------------------------------------------------------------- DOCX


def _docx_heading(paragraph: Paragraph, text: str) -> bool:
    style = ((paragraph.style.name if paragraph.style is not None else "") or "").lower()
    if style.startswith("heading") or style == "title":
        return True
    runs = [r for r in paragraph.runs if r.text.strip()]
    return bool(runs) and len(text) <= _MAX_HEADING_LEN and all(r.bold for r in runs) and not text.endswith(".")


def parse_docx(path: Path) -> list[Section]:
    try:
        document = Document(str(path))
    except Exception as exc:
        raise ParseError(f"cannot open DOCX: {exc}") from exc

    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(heading or "Document", text))
        buf.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            if _docx_heading(paragraph, text):
                flush()
                heading = text.rstrip(":").strip()
            else:
                buf.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells: list[str] = []
                for cell in row.cells:
                    value = cell.text.strip()
                    if value and value not in cells:  # merged cells repeat
                        cells.append(value)
                if cells:
                    buf.append(" | ".join(cells))
    flush()
    return sections


def parse_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    raise ParseError(f"unsupported file type: {suffix}")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
