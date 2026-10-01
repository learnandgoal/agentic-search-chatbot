"""Split parsed sections into overlapping chunks that never cross a section boundary."""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import Section


@dataclass
class ChunkDraft:
    page_or_section: str
    chunk_index: int  # position within the whole document, used for navigation
    text: str


def split_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
            if cut < size * 0.5:
                cut = window.rfind(" ")
            if cut >= size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_sections(sections: list[Section], size: int = 900, overlap: int = 120) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for section in sections:
        for piece in split_text(section.text, size, overlap):
            drafts.append(ChunkDraft(section.label, len(drafts), piece))
    return drafts
