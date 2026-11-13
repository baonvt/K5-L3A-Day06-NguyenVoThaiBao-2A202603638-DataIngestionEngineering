"""PDF ingestion with page-level provenance and section metadata."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


CONTENT_START_PAGE = 14  # first page of numbered handbook content
CONTENT_END_PAGE = 324  # stop before the references and bibliography
MIN_USEFUL_PAGE_CHARS = 400


def normalize_text(text: str) -> str:
    """Normalize PDF text while keeping paragraph boundaries and useful symbols."""
    text = unicodedata.normalize("NFKC", text).replace("\ufeff", "")
    text = text.replace("\r", "\n").replace("\x00", " ").replace("\x1f", " ")
    paragraphs: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        if lines:
            paragraph = " ".join(part.strip() for part in lines if part.strip())
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                paragraphs.append(paragraph)
            lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if re.match(r"^NASA Systems Engineering Handbook\s+Rev\s+2\b", line, re.I):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        # Join PDF line wraps while preserving real compound-word hyphens.
        if lines and lines[-1].endswith("-") and line[:1].islower():
            lines[-1] = lines[-1] + line
        else:
            lines.append(line)
    flush()
    return "\n\n".join(paragraphs)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract useful numbered content pages and attach source/page/section metadata.

    Page numbers in metadata are physical 1-based PDF page numbers. The PDF's printed
    page label is also retained separately because the front matter uses roman numerals.
    Sparse and reserved pages are omitted from the retrieval corpus and reported by the
    caller from the manifest's page audit.
    """
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    chapter_id = "1.0"
    chapter_title = "Introduction"
    section_id = "1.0"
    section_title = chapter_title
    appendix: str | None = None

    for physical_page, pdf_page in enumerate(reader.pages, start=1):
        raw = pdf_page.extract_text() or ""
        if not CONTENT_START_PAGE <= physical_page <= CONTENT_END_PAGE:
            continue

        for line in raw.splitlines():
            heading = line.strip().replace("\x1f", " ")
            appendix_match = re.match(r"^Appendix\s+([A-Z])\s*:\s*(.+?)\s*$", heading, re.I)
            if appendix_match:
                appendix = f"Appendix {appendix_match.group(1).upper()}"
                chapter_id = appendix
                chapter_title = appendix_match.group(2).strip()
                section_id = appendix
                section_title = chapter_title
                continue

            major_match = re.match(r"^([1-7])\.0\s+(.+?)\s*$", heading)
            if major_match and appendix is None:
                chapter_id = f"{major_match.group(1)}.0"
                chapter_title = major_match.group(2).strip()
                section_id = chapter_id
                section_title = chapter_title
                continue

            section_match = re.match(r"^(\d+(?:\.\d+)+)\s+(.+?)\s*$", heading)
            if section_match:
                section_id = section_match.group(1)
                section_title = section_match.group(2).strip()

        text = normalize_text(raw)
        if len(text) < MIN_USEFUL_PAGE_CHARS or re.search(r"\bReserved\b", text[:180], re.I):
            continue

        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=physical_page,
            text=text,
            metadata={
                "page_label": physical_page - 13,
                "chapter": chapter_id,
                "chapter_title": chapter_title,
                "section": section_id,
                "section_title": section_title,
                "language": "en",
                "document_type": "handbook",
                "char_count": len(text),
            },
        ))
    return pages


def page_audit(path: Path) -> list[dict[str, int | str | bool]]:
    """Return counts and explicit exclusions for each numbered handbook page."""
    reader = PdfReader(str(path))
    audit: list[dict[str, int | str | bool]] = []
    for physical_page, pdf_page in enumerate(reader.pages, start=1):
        if not CONTENT_START_PAGE <= physical_page <= CONTENT_END_PAGE:
            continue
        text = normalize_text(pdf_page.extract_text() or "")
        reserved = bool(re.search(r"\bReserved\b", text[:180], re.I))
        included = len(text) >= MIN_USEFUL_PAGE_CHARS and not reserved
        audit.append({
            "physical_pdf_page": physical_page,
            "printed_page": physical_page - 13,
            "characters": len(text),
            "included": included,
            "reason": "included" if included else ("reserved" if reserved else "sparse_or_figure_only"),
        })
    return audit
