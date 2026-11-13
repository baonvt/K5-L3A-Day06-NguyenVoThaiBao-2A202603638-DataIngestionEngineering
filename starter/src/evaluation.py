"""Evidence retrieval metrics. Manual labels are deliberately kept separate."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


def _comparison_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold_pages = set(question.get("gold_pages", [question["gold_page"]]))
    gold_source = question["gold_source_id"]
    evidence_options = question.get("evidence_quotes", [question["evidence_quote"]])
    normalized_evidence = [_comparison_text(quote) for quote in evidence_options]
    enriched: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {})
        on_gold_page = metadata.get("document_id") == gold_source and metadata.get("page") in gold_pages
        normalized_hit = _comparison_text(hit.get("text", ""))
        evidence_match = on_gold_page and any(quote in normalized_hit for quote in normalized_evidence)
        item = {**hit, "rank": rank, "gold_page_match": on_gold_page, "evidence_match": evidence_match}
        enriched.append(item)
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": gold_source,
        "gold_pages": sorted(gold_pages),
        "evidence_quotes": evidence_options,
        "filter": question.get("filter"),
        "top1_hit": bool(enriched and enriched[0]["evidence_match"]),
        "top3_hit": any(item["evidence_match"] for item in enriched[:3]),
        "hits": enriched,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[(trace["strategy"], trace["split"])].append(trace)
    summary: dict[str, Any] = {}
    for (strategy, split), rows in sorted(grouped.items()):
        n = len(rows)
        h1 = sum(bool(row["top1_hit"]) for row in rows)
        h3 = sum(bool(row["top3_hit"]) for row in rows)
        summary.setdefault(strategy, {})[split] = {
            "questions": n,
            "hit_at_1": {"hits": h1, "total": n, "rate": h1 / n if n else 0.0},
            "evidence_hit_at_3": {"hits": h3, "total": n, "rate": h3 / n if n else 0.0},
        }
    return summary
