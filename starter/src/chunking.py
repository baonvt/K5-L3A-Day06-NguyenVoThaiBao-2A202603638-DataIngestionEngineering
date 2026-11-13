"""Reproducible chunking strategies A (fixed) and B/C (paragraph aware)."""
from __future__ import annotations

import hashlib

from .models import Chunk, DocumentPage


def _stable_id(strategy: str, page: DocumentPage, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{strategy}-{page.document_id}-{page.page}-{ordinal}-{digest}"


def _make_chunks(page: DocumentPage, strategy: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=_stable_id(strategy, page, index, text),
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=text,
            metadata={**page.metadata, "chunk_index": index, "chunk_chars": len(text)},
        )
        for index, text in enumerate(texts)
        if text.strip()
    ]


def _window(text: str, size: int, overlap: int) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    output: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        fragment = text[start:end].strip()
        if fragment:
            output.append(fragment)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return output


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into overlapping fixed character windows."""
    return _make_chunks(page, "A", _window(page.text, size, overlap))


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group intact paragraphs up to a limit, splitting only oversized paragraphs."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    paragraphs = [part.strip() for part in page.text.split("\n\n") if part.strip()]
    groups: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                groups.append(current)
                current = ""
            groups.extend(_window(paragraph, max_size, min(80, max_size // 8)))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_size:
            current += "\n\n" + paragraph
        else:
            groups.append(current)
            current = paragraph
    if current:
        groups.append(current)
    return _make_chunks(page, "B", groups)
