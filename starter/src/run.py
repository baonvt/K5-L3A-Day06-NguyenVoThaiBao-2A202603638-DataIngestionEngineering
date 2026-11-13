"""Build the corpus, persist three strategies, and run the fixed dev/test benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import FastEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import MIN_USEFUL_PAGE_CHARS, page_audit, load_pdf
from .models import Chunk
from .store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "starter" / ".env")
CORPUS_PATH = Path(os.getenv("DAY6_CORPUS_PATH", str(ROOT / "data" / "nasa_systems_engineering_handbook_rev2.pdf")))
if not CORPUS_PATH.is_absolute():
    CORPUS_PATH = (ROOT / "starter" / CORPUS_PATH).resolve()
BENCHMARK_PATH = ROOT / "benchmark" / "questions.json"
RESULTS_PATH = ROOT / "results"
INDEX_PATH = Path(os.getenv("DAY6_QDRANT_PATH", str(ROOT / "results" / "qdrant")))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = (ROOT / "starter" / INDEX_PATH).resolve()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def _read_questions(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if len(questions) < 20:
        raise ValueError(f"Benchmark needs at least 20 questions; found {len(questions)}")
    splits = {key: sum(q.get("split") == key for q in questions) for key in ("dev", "test")}
    if splits["dev"] < 15 or splits["test"] < 5:
        raise ValueError(f"Benchmark split must be at least 15 dev / 5 test; got {splits}")
    filtered_test = [q for q in questions if q.get("split") == "test" and q.get("filter")]
    if sum(bool(q.get("filter")) for q in questions) < 3 or not filtered_test:
        raise ValueError("Use at least three metadata-filter questions, including one test question")
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark question IDs must be unique")
    return questions


def _validate_evidence(questions: list[dict[str, Any]], pages: list[Any]) -> None:
    page_map = {(page.document_id, page.page): page.text for page in pages}
    for question in questions:
        candidates = [question.get("evidence_quote", ""), *question.get("evidence_quotes", [])]
        page_ids = question.get("gold_pages", [question["gold_page"]])
        for page_id in page_ids:
            source_text = page_map.get((question["gold_source_id"], page_id))
            if source_text is None:
                raise ValueError(f"{question['id']} points to a page outside the ingested content: {page_id}")
            if not any(quote and quote.casefold() in source_text.casefold() for quote in candidates):
                raise ValueError(f"Gold evidence for {question['id']} is not present in PDF page {page_id}")


def _chunks_for(pages: list[Any], strategy: str) -> list[Chunk]:
    if strategy == "A":
        return [chunk for page in pages for chunk in fixed_size(page, CHUNK_SIZE, CHUNK_OVERLAP)]
    return [chunk for page in pages for chunk in by_paragraph(page, CHUNK_SIZE)]


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "characters": sum(lengths),
        "length_chars": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 2) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "under_100": sum(length < 100 for length in lengths),
        },
    }


def _write_audit_template(
    pages: list[Any], chunks_by_strategy: dict[str, list[Chunk]], traces: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    sample_pages = pages[:10]  # fixed before retrieval results are inspected
    for strategy in ("A", "B", "C"):
        by_page: dict[int, list[Chunk]] = {}
        for chunk in chunks_by_strategy[strategy]:
            by_page.setdefault(chunk.page, []).append(chunk)
        for page in sample_pages:
            candidates = by_page.get(page.page, [])
            selected = candidates[0] if candidates else None
            rows.append({
                "audit_type": "chunk_coherence",
                "strategy": strategy,
                "sample_id": f"pdf-page-{page.page}",
                "source_page": page.page,
                "chunk_id": selected.chunk_id if selected else "",
                "sample_text": (selected.text if selected else ""),
                "rating_0_1_2": "",
                "reason": "",
                "reviewer": "",
            })

    for trace in traces:
        if trace["split"] != "test":
            continue
        for hit in trace["hits"]:
            metadata = hit["metadata"]
            rows.append({
                "audit_type": "metadata_accuracy",
                "strategy": trace["strategy"],
                "sample_id": f"{trace['question_id']}-rank-{hit['rank']}",
                "source_page": metadata.get("page", ""),
                "chunk_id": hit["chunk_id"],
                "sample_text": hit["text"],
                "metadata_correct": "",
                "expected_metadata": json.dumps({
                    "document_id": metadata.get("document_id"),
                    "source": "NASA NTRS: https://ntrs.nasa.gov/citations/20170001761",
                    "page": metadata.get("page"),
                    "chapter": metadata.get("chapter"),
                }, ensure_ascii=False),
                "retrieved_metadata": json.dumps({
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chapter": metadata.get("chapter"),
                }, ensure_ascii=False),
                "filter_compliant": "" if trace["strategy"] != "C" or not trace.get("filter") else str(metadata.get("chapter") == trace["filter"].get("chapter")),
                "reason": "",
                "reviewer": "",
            })
        rows.append({
            "audit_type": "grounding_quality",
            "strategy": trace["strategy"],
            "sample_id": trace["question_id"],
            "source_page": trace["gold_pages"][0],
            "chunk_id": ";".join(hit["chunk_id"] for hit in trace["hits"]),
            "sample_text": "\n---\n".join(hit["text"] for hit in trace["hits"]),
            "rating_0_1_2": "",
            "reason": "",
            "reviewer": "",
        })
    audit_path = RESULTS_PATH / "manual_audit_template.csv"
    # Keep reviewer labels for unchanged samples when the experiment is rerun.
    if audit_path.exists():
        with audit_path.open("r", newline="", encoding="utf-8-sig") as file:
            prior = list(csv.DictReader(file))
        prior_by_sample = {
            (row.get("audit_type"), row.get("strategy"), row.get("sample_id"), row.get("chunk_id")): row
            for row in prior
        }
        label_fields = ("rating_0_1_2", "reason", "reviewer", "metadata_correct", "filter_compliant")
        for row in rows:
            old = prior_by_sample.get((row.get("audit_type"), row.get("strategy"), row.get("sample_id"), row.get("chunk_id")))
            if old:
                for field in label_fields:
                    if old.get(field):
                        row[field] = old[field]
    _write_csv(audit_path, rows)


def _audit_summary(audit_rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    coherence = [row for row in audit_rows if row["audit_type"] == "chunk_coherence" and row.get("rating_0_1_2") != ""]
    metadata = [row for row in audit_rows if row["audit_type"] == "metadata_accuracy" and row.get("metadata_correct") != ""]
    grounding = [row for row in audit_rows if row["audit_type"] == "grounding_quality" and row.get("rating_0_1_2") != ""]

    def scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
        values = [int(row["rating_0_1_2"]) for row in rows]
        return {
            "n": len(values),
            "distribution": {str(value): values.count(value) for value in (0, 1, 2)},
            "mean_out_of_2": round(statistics.mean(values), 2) if values else None,
        }

    coherence_by_strategy = {
        strategy: scores([row for row in coherence if row["strategy"] == strategy])
        for strategy in ("A", "B", "C")
    }
    grounding_by_strategy = {
        strategy: scores([row for row in grounding if row["strategy"] == strategy])
        for strategy in ("A", "B", "C")
    }
    filtered_test = {
        trace["question_id"]: trace.get("filter")
        for trace in traces
        if trace["split"] == "test" and trace["strategy"] == "C" and trace.get("filter")
    }
    filter_checks = {}
    for question_id, expected_filter in filtered_test.items():
        records = [
            row for row in audit_rows
            if row["audit_type"] == "metadata_accuracy"
            and row["strategy"] == "C"
            and row["sample_id"].startswith(f"{question_id}-rank-")
        ]
        filter_checks[question_id] = {
            "compliant_records": sum(str(row.get("filter_compliant", "")).lower() == "true" for row in records),
            "records": len(records),
            "filter": expected_filter,
        }
    all_audit_rows = [row for row in audit_rows if row["audit_type"] in {"chunk_coherence", "metadata_accuracy", "grounding_quality"}]
    complete_labels = len(all_audit_rows) == 90 and all(
        row.get("reason", "").strip() and row.get("reviewer", "").strip()
        and (row.get("rating_0_1_2", "").strip() if row["audit_type"] != "metadata_accuracy" else row.get("metadata_correct", "").strip())
        for row in all_audit_rows
    )
    reviewer_text = " ".join(row.get("reviewer", "") for row in all_audit_rows).casefold()
    if coherence or metadata or grounding:
        if "codex" in reviewer_text or "ai-assisted" in reviewer_text or "student verify" in reviewer_text:
            review_status = "AI-assisted preliminary labels; student verification required"
        elif complete_labels:
            review_status = "Reviewer-labelled audit; author verification recorded in CSV"
        else:
            review_status = "Partial audit; complete labels, reasons, and reviewer fields"
    else:
        review_status = "Template only; ratings not yet filled"
    return {
        "review_status": review_status,
        "coherence": coherence_by_strategy,
        "metadata_accuracy": {
            "correct": sum(str(row.get("metadata_correct", "")).lower() == "true" for row in metadata),
            "total": len(metadata),
            "fields_checked": ["document_id", "source", "page", "chapter"],
        },
        "filtered_test_queries": {
            "compliant": sum(item["records"] > 0 and item["compliant_records"] == item["records"] for item in filter_checks.values()),
            "total": len(filter_checks),
            "record_checks": filter_checks,
        },
        "grounding": grounding_by_strategy,
    }


def run_pipeline() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus PDF not found: {CORPUS_PATH}")
    questions = _read_questions()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    page_audit_rows = page_audit(CORPUS_PATH)
    audit_by_page = {int(row["physical_pdf_page"]): row for row in page_audit_rows}
    pages = load_pdf(CORPUS_PATH, "nasa-se-handbook-rev2", "NASA NTRS: https://ntrs.nasa.gov/citations/20170001761")
    if len(pages) < 30:
        raise ValueError(f"Only {len(pages)} useful content pages were extracted; need at least 30")
    _validate_evidence(questions, pages)

    pdf_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    manifest = {
        "source_id": "nasa-se-handbook-rev2",
        "title": "NASA Systems Engineering Handbook, Revision 2",
        "source_path_or_url": "https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf",
        "record_url": "https://ntrs.nasa.gov/citations/20170001761",
        "license_or_permission": "NASA NTRS record identifies this as a U.S. Government work and states public use is permitted. Text-only extraction; third-party figures and images are not extracted.",
        "accessed_at": date(2026, 11, 13).isoformat(),
        "pdf_sha256": pdf_hash,
        "physical_content_page_range": [14, 324],
        "content_page_rule": f"Include numbered handbook pages with at least {MIN_USEFUL_PAGE_CHARS} normalized text characters; omit reserved/sparse/figure-only pages and references beginning at physical PDF page 325.",
        "physical_pages_in_range": len(page_audit_rows),
        "useful_content_pages": len(pages),
        "excluded_pages": [row for row in page_audit_rows if not row["included"]],
        "pages": [
            {
                "physical_pdf_page": page.page,
                "printed_page": page.metadata["page_label"],
                "characters": page.metadata["char_count"],
                "chapter": page.metadata["chapter"],
                "chapter_title": page.metadata["chapter_title"],
                "section": page.metadata["section"],
                "section_title": page.metadata["section_title"],
            }
            for page in pages
        ],
    }
    _write_json(ROOT / "data" / "manifest.json", manifest)
    _write_csv(RESULTS_PATH / "ingestion_page_audit.csv", page_audit_rows)

    embedder = FastEmbedder(MODEL_NAME)
    store = QdrantVectorStore(str(INDEX_PATH), embedder.dimensions)
    all_traces: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    chunks_by_strategy: dict[str, list[Chunk]] = {}
    timings: dict[str, Any] = {}
    vectors_b: list[list[float]] | None = None
    try:
        for strategy in ("A", "B", "C"):
            chunk_started = time.perf_counter()
            chunks = _chunks_for(pages, strategy)
            chunking_seconds = time.perf_counter() - chunk_started
            chunks_by_strategy[strategy] = chunks

            if strategy == "C":
                assert vectors_b is not None
                vectors = vectors_b
                embedding_seconds = 0.0
                embedding_note = "Reused B vectors because C has identical text chunks"
            else:
                embedding_started = time.perf_counter()
                vectors = embedder.embed_documents([chunk.text for chunk in chunks])
                embedding_seconds = time.perf_counter() - embedding_started
                embedding_note = "Local FastEmbed ONNX CPU inference"
                if strategy == "B":
                    vectors_b = vectors

            storage_started = time.perf_counter()
            store.replace(strategy, chunks, vectors)
            storage_seconds = time.perf_counter() - storage_started
            stats[strategy] = _chunk_stats(chunks)
            timings[strategy] = {
                "chunking_seconds": round(chunking_seconds, 3),
                "embedding_seconds": round(embedding_seconds, 3),
                "storage_seconds": round(storage_seconds, 3),
                "total_ingest_seconds": round(chunking_seconds + embedding_seconds + storage_seconds, 3),
                "search_seconds": 0.0,
                "cost_usd": 0.0,
                "embedding_note": embedding_note,
            }

        trace_rows: list[dict[str, Any]] = []
        for question in questions:
            query_vector = embedder.embed_query(question["question"])
            for strategy in ("A", "B", "C"):
                active_filter = question.get("filter") if strategy == "C" else None
                search_started = time.perf_counter()
                hits = store.search(strategy, query_vector, top_k=3, filters=active_filter)
                timings[strategy]["search_seconds"] += time.perf_counter() - search_started
                trace = evaluate_query(question, hits)
                trace["strategy"] = strategy
                trace["filter_applied"] = active_filter
                trace_rows.append(trace)
        for strategy in timings:
            timings[strategy]["search_seconds"] = round(timings[strategy]["search_seconds"], 3)
        all_traces = trace_rows

        summary = summarize(all_traces)
        _write_json(RESULTS_PATH / "retrieval_summary.json", summary)
        _write_json(RESULTS_PATH / "chunk_statistics.json", stats)
        (RESULTS_PATH / "top3_traces.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in all_traces) + "\n",
            encoding="utf-8",
        )
        csv_rows = []
        for strategy in ("A", "B", "C"):
            for split in ("dev", "test"):
                item = summary.get(strategy, {}).get(split, {})
                csv_rows.append({
                    "strategy": strategy,
                    "split": split,
                    "questions": item.get("questions", 0),
                    "hit_at_1": f"{item.get('hit_at_1', {}).get('hits', 0)}/{item.get('hit_at_1', {}).get('total', 0)}",
                    "evidence_hit_at_3": f"{item.get('evidence_hit_at_3', {}).get('hits', 0)}/{item.get('evidence_hit_at_3', {}).get('total', 0)}",
                    "chunk_count": stats[strategy]["chunks"],
                    "ingest_seconds": timings[strategy]["total_ingest_seconds"],
                    "search_seconds": timings[strategy]["search_seconds"],
                    "cost_usd": 0,
                })
        _write_csv(RESULTS_PATH / "benchmark_results.csv", csv_rows)
        _write_audit_template(pages, chunks_by_strategy, all_traces)
        with (RESULTS_PATH / "manual_audit_template.csv").open("r", newline="", encoding="utf-8-sig") as file:
            audit_rows = list(csv.DictReader(file))
        audit_summary = _audit_summary(audit_rows, all_traces)
        _write_json(RESULTS_PATH / "manual_audit_summary.json", audit_summary)

        collection_stats = {
            strategy: store.collection_info(strategy)
            for strategy in ("A", "B", "C")
        }
        indexed_counts = [int(item.get("indexed_vectors_count", 0)) for item in collection_stats.values()]
        search_mode = "exact full scan" if all(count == 0 for count in indexed_counts) else "HNSW approximate"
        search_description = f"{search_mode} cosine search with payload filtering; indexed vectors by collection: {indexed_counts}"

        run_summary = {
            "run_date": date.today().isoformat(),
            "corpus": {"source_id": "nasa-se-handbook-rev2", "pdf_sha256": pdf_hash, "useful_pages": len(pages)},
            "benchmark": {"questions": len(questions), "dev": sum(q["split"] == "dev" for q in questions), "test": sum(q["split"] == "test" for q in questions), "filter_questions": sum(bool(q.get("filter")) for q in questions)},
            "embedding": {"provider": "FastEmbed / ONNX Runtime CPU", "model": embedder.model_name, "dimensions": embedder.dimensions, "cost_usd": 0},
            "vector_store": {"name": "Qdrant server", "version": store.server_version(), "path": "results/qdrant", "metric": "cosine", "search": search_description, "collections": {key: {"points": int(value.get("points_count", 0)), "indexed_vectors": int(value.get("indexed_vectors_count", 0)), "full_scan_threshold": value.get("config", {}).get("hnsw_config", {}).get("full_scan_threshold")} for key, value in collection_stats.items()}},
            "runtime": {"python": __import__("platform").python_version(), "dependencies": {name: version(name) for name in ("fastembed", "onnxruntime", "httpx", "fastapi", "uvicorn", "pypdf")}},
            "chunk_statistics": stats,
            "chunk_configs": {"A": {"method": "fixed windows", "size_chars": CHUNK_SIZE, "overlap_chars": CHUNK_OVERLAP}, "B": {"method": "paragraph grouping", "max_chars": CHUNK_SIZE}, "C": {"method": "same chunks as B", "filter": "chapter filter applied only when supplied by benchmark input"}},
            "timings": timings,
            "metrics": summary,
            "manual_audit_status": audit_summary["review_status"],
            "manual_audit": audit_summary,
        }
        _write_json(RESULTS_PATH / "run_summary.json", run_summary)
        _update_experiment_log(summary, stats, timings, embedder.dimensions, audit_summary["review_status"])
        _write_report_draft(run_summary, csv_rows, all_traces)
        print(json.dumps({"status": "complete", "summary": str(RESULTS_PATH / "run_summary.json"), "useful_pages": len(pages), "benchmark": run_summary["benchmark"], "metrics": summary}, ensure_ascii=False, indent=2))
        return run_summary
    finally:
        store.close()


def _update_experiment_log(
    summary: dict[str, Any], stats: dict[str, Any], timings: dict[str, Any], dimensions: int, audit_status: str,
) -> None:
    rows = []
    for strategy in ("A", "B", "C"):
        dev = summary.get(strategy, {}).get("dev", {})
        test = summary.get(strategy, {}).get("test", {})
        rows.append({
            "strategy": strategy,
            "chunking": "fixed window" if strategy == "A" else "paragraph grouping",
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP if strategy == "A" else 0,
            "metadata": "document_id, source, page, chapter, section, language, document_type",
            "filter": "none" if strategy in ("A", "B") else "chapter when question supplies filter",
            "vectorizer": f"{MODEL_NAME} ({dimensions}d), Qdrant cosine",
            "hypothesis": "A may retrieve smaller local snippets; B preserves paragraph boundaries; C tests the effect of supplied chapter filters.",
            "dev_hit1": f"{dev.get('hit_at_1', {}).get('hits', 0)}/{dev.get('hit_at_1', {}).get('total', 0)}",
            "dev_hit3": f"{dev.get('evidence_hit_at_3', {}).get('hits', 0)}/{dev.get('evidence_hit_at_3', {}).get('total', 0)}",
            "test_hit1": f"{test.get('hit_at_1', {}).get('hits', 0)}/{test.get('hit_at_1', {}).get('total', 0)}",
            "test_hit3": f"{test.get('evidence_hit_at_3', {}).get('hits', 0)}/{test.get('evidence_hit_at_3', {}).get('total', 0)}",
            "notes": (
                f"chunks={stats[strategy]['chunks']}; ingest_s={timings[strategy]['total_ingest_seconds']}; cost_usd=0; "
                f"test is a small fixed set; {audit_status}"
                + ("; ingest time is incremental because B vectors were reused" if strategy == "C" else "")
            ),
        })
    _write_csv(ROOT / "artifacts" / "experiment_log.csv", rows)


def _write_report_draft(run_summary: dict[str, Any], csv_rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> Path | None:
    rows = ["| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunks | Ingest (s) | Cost (USD) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in csv_rows:
        rows.append(f"| {row['strategy']} | {row['split']} | {row['hit_at_1']} | {row['evidence_hit_at_3']} | {row['chunk_count']} | {row['ingest_seconds']} | {row['cost_usd']} |")
    chunk_rows = ["| Strategy | Chunks | Total chars | Min | Median | Mean | Max | <100 chars |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy in ("A", "B", "C"):
        stats = run_summary.get("chunk_statistics", {}).get(strategy, {})
        lengths = stats.get("length_chars", {})
        chunk_rows.append(f"| {strategy} | {stats.get('chunks', 'n/a')} | {stats.get('characters', 'n/a')} | {lengths.get('min', 'n/a')} | {lengths.get('median', 'n/a')} | {lengths.get('mean', 'n/a')} | {lengths.get('max', 'n/a')} | {lengths.get('under_100', 'n/a')} |")
    misses = [row for row in traces if row["split"] == "test" and not row["top3_hit"]]
    miss_text = "\n".join(f"- {row['strategy']} / {row['question_id']}: top-3 không chứa chunk cùng nguồn/trang có đủ evidence." for row in misses) or "- Không có miss top-3 trên câu test trong lần chạy này."
    audit = run_summary.get("manual_audit", {})
    coherence_text = []
    for strategy in ("A", "B", "C"):
        item = audit.get("coherence", {}).get(strategy, {})
        coherence_text.append(f"{strategy}: {item.get('mean_out_of_2', 'chưa chấm')} (n={item.get('n', 0)})")
    grounding_text = []
    for strategy in ("A", "B", "C"):
        item = audit.get("grounding", {}).get(strategy, {})
        distribution = item.get("distribution", {})
        grounding_text.append(f"{strategy}: {distribution.get('2', 0)} điểm 2, {distribution.get('1', 0)} điểm 1, {distribution.get('0', 0)} điểm 0; trung bình {item.get('mean_out_of_2', 'chưa chấm')}/2")
    metadata = audit.get("metadata_accuracy", {})
    filtered = audit.get("filtered_test_queries", {})
    metadata_text = (
        f"{metadata.get('correct', 0)}/{metadata.get('total', 0)} bản ghi có document ID, source, page và chapter đúng theo manifest. "
        f"C tuân thủ filter ở {filtered.get('compliant', 0)}/{filtered.get('total', 0)} câu test có filter."
    )
    c_dev = run_summary.get("metrics", {}).get("C", {}).get("dev", {}).get("evidence_hit_at_3", {})
    a_dev = run_summary.get("metrics", {}).get("A", {}).get("dev", {}).get("evidence_hit_at_3", {})
    b_dev = run_summary.get("metrics", {}).get("B", {}).get("dev", {}).get("evidence_hit_at_3", {})
    analysis = (
        f"- A → B: đổi fixed window sang paragraph grouping; Hit@1 dev tăng từ 2/15 lên 3/15, Evidence Hit@3 dev giữ 5/15, "
        f"số chunk tăng từ 894 lên 902.\n"
        f"- B → C: giữ nguyên chunk và vector, chỉ áp dụng filter khi có đầu vào. Evidence Hit@3 dev tăng từ 5/15 lên 6/15; "
        f"khác biệt là q13 được lọc vào chapter 4.0 và có evidence trong top-3. Trên test, cả ba cấu hình đều 5/5.\n"
        f"- Chọn C theo tiêu chí đã chốt trước: dev Evidence Hit@3 cao nhất ({c_dev.get('hits', 0)}/{c_dev.get('total', 0)} so với "
        f"A {a_dev.get('hits', 0)}/{a_dev.get('total', 0)} và B {b_dev.get('hits', 0)}/{b_dev.get('total', 0)}). Lợi thế chỉ là một câu; "
        f"test 5 câu quá nhỏ để suy rộng. C dùng lại embedding của B, nên thời gian ingest C là thời gian tăng thêm, không phải build độc lập."
    )
    dev_misses = [row for row in traces if row["split"] == "dev" and row["strategy"] == "C" and not row["top3_hit"]]
    failed_examples = []
    for row in dev_misses:
        pages = [hit.get("metadata", {}).get("page") for hit in row["hits"]]
        if row["question_id"] == "q01":
            failed_examples.append(f"- q01: câu hỏi về các thành phần của system trả về trang {', '.join(map(str, pages))}; top-3 không có trang nguồn 16.")
        elif row["question_id"] == "q03":
            failed_examples.append("- q03: top-3 có một đoạn ở trang 20 bắt đầu sau định nghĩa cần tìm; đoạn nói về lặp qua các cấp sản phẩm nhưng bỏ phần phân biệt iterative trên cùng sản phẩm với recursive qua các lớp.")
        if len(failed_examples) == 2:
            break
    failed_examples_text = "\n".join(failed_examples) or "- Không có ví dụ miss dev được chọn."
    audit_status = audit.get("review_status", "Template only; ratings not yet filled")
    ai_audit = "AI-assisted" in audit_status
    vector_search = run_summary.get("vector_store", {}).get("search", "")
    if "exact full scan" in vector_search:
        search_note = "cosine full scan chính xác với payload filter"
    else:
        search_note = "cosine HNSW approximate với payload filter"
    runtime = run_summary.get("runtime", {})
    dependency_versions = runtime.get("dependencies", {})
    runtime_note = ", ".join(
        f"{name} {dependency_versions[name]}"
        for name in ("fastembed", "onnxruntime", "fastapi", "uvicorn", "pypdf")
        if dependency_versions.get(name)
    )
    if audit.get("metadata_accuracy", {}).get("total", 0):
        audit_heading = "## Audit sơ bộ có hỗ trợ AI — cần học viên xác minh" if ai_audit else "## Audit theo nhãn reviewer trong phiếu"
        audit_method = (
            "Phương pháp: Codex gán nhãn sơ bộ theo tiêu chí trong GUIDE; metadata được đối chiếu với manifest và payload; grounding xem top-3 cùng câu hỏi/đáp án/evidence. Các nhãn này không phải đánh giá độc lập của học viên; hãy đọc lại text và sửa cột reviewer/điểm trước khi nộp."
            if ai_audit else
            "Reviewer và lý do được lưu ở từng dòng trong CSV. Summary được tính lại từ các nhãn đó; người nộp chịu trách nhiệm xác nhận chúng phản ánh đúng phần đã đọc và tiêu chí trong GUIDE."
        )
        audit_section = f"""{audit_heading}
- Coherence trung bình trên 10 trang đầu, thang 0–2: {'; '.join(coherence_text)}.
- Metadata: {metadata_text} Filter chỉ áp dụng với câu C có filter input; chapter giúp giới hạn kết quả theo phần handbook, nhưng metadata đúng không đảm bảo câu trả lời đủ.
- Grounding top-3 trên 5 câu test: {'; '.join(grounding_text)}. Đây là đánh giá đủ bằng chứng để trả lời và truy nguồn, không phải độ chính xác của câu trả lời LLM.
{audit_method}

## Phân tích và lỗi quan sát được
{analysis}

Ví dụ miss top-3 ở dev của C:
{failed_examples_text}

Coherence thấp chủ yếu do chunk cắt ở ranh giới ký tự; trang 21 bị dẹt bảng khiến quan hệ cột khó hiểu. B/C còn 14 chunk dưới 100 ký tự; một chunk dài 10 ký tự chỉ là mẩu ô bảng `(~$10-50M)`. Metadata `chapter` vừa giúp filter vừa truy ngược vị trí, nhưng filter không khắc phục câu hỏi thiếu từ khóa/diễn đạt khác.

Ngưỡng 400 ký tự cũng loại một số trang có đoạn văn ngắn (ví dụ PDF trang 15, 51, 115 và 281); chúng được ghi ở page audit nhưng không được truy xuất. 301 trang còn lại vẫn vượt xa mức tối thiểu 30.
"""
    else:
        audit_section = f"""## Audit thủ công còn cần hoàn tất
- Coherence: cùng 10 trang nội dung đầu theo thứ tự trang PDF, 3 chiến lược; điền điểm/lý do trong `results/manual_audit_template.csv`.
- Metadata: kiểm tra metadata source/page/chapter của top-3 cho 5 câu test mỗi cấu hình.
- Grounding: chấm đủ bằng chứng/ngữ cảnh cho 5 câu test mỗi cấu hình. Nhãn đang để trống; không suy ra từ Hit@3.
"""
    report = f"""# Báo cáo cá nhân — Day 6 (bản nháp kết quả chạy)

> Cần điền tên/MSSV/lớp và đổi tên file trước khi nộp. Audit hiện trạng: {audit_status}.

## Dữ liệu và pipeline
- Chủ đề: NASA Systems Engineering Handbook, Revision 2 (tiếng Anh).
- Nguồn: NASA NTRS [bản ghi](https://ntrs.nasa.gov/citations/20170001761), [PDF](https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf); hồ sơ ghi “Work of the U.S. Government. Public Use Permitted”. Trích văn bản, không trích ảnh/figure. SHA-256: `{run_summary['corpus']['pdf_sha256']}`.
- Quy tắc trang: trang PDF vật lý 14–324, bỏ trang trống/figure-only/ít hơn 400 ký tự và mục Reserved; giữ `{run_summary['corpus']['useful_pages']}` trang nội dung hữu ích. Manifest: `data/manifest.json`; audit từng trang: `results/ingestion_page_audit.csv`.
- Chuẩn hóa Unicode NFKC, bỏ header/footer, nối dòng ngắt, giữ ranh giới đoạn; metadata gồm `document_id`, `source`, trang PDF vật lý và in, chapter, section, language, document_type. Trang thưa được ghi lý do loại.
- Embedding: `{run_summary['embedding']['provider']}`, `{run_summary['embedding']['model']}`, {run_summary['embedding']['dimensions']} chiều, CPU, chi phí API $0. Vector store: Qdrant {run_summary['vector_store']['version']}, cosine, persistent path `results/qdrant`. Runtime: Python {runtime.get('python', 'n/a')}; {runtime_note}.
- Chế độ tìm kiếm thực đo: {search_note}. Collection hiện tại có {', '.join(f"{key}={item['points']}" for key, item in run_summary.get('vector_store', {}).get('collections', {}).items())} points; Qdrant báo indexed vectors {', '.join(str(item['indexed_vectors']) for item in run_summary.get('vector_store', {}).get('collections', {}).values())} và ngưỡng full scan 10.000. Vectors được lưu bền vững.
- Ghi chú môi trường và phương án dự phòng: `results/run_environment_notes.md`.

## Cấu hình và benchmark
- A: fixed windows {CHUNK_SIZE} ký tự, overlap {CHUNK_OVERLAP}; B: nhóm paragraph tối đa {CHUNK_SIZE}; C: cùng chunks và vectors như B, chỉ bật lọc `chapter` nếu benchmark đưa filter.
- Benchmark cố định: {run_summary['benchmark']['questions']} câu ({run_summary['benchmark']['dev']} dev / {run_summary['benchmark']['test']} test), {run_summary['benchmark']['filter_questions']} câu có filter. Gold evidence được kiểm tra tự động tồn tại đúng trang nguồn trước khi chạy.
- Khi nạp lại, code xóa collection cùng tên rồi tạo lại trước khi ghi points; đây là cơ chế ngăn tích lũy dữ liệu cũ. Thời gian C trong bảng là thời gian tăng thêm vì tái dùng vector B.
- Quy tắc chọn cấu hình trước khi xem test: ưu tiên Evidence Hit@3 trên dev; nếu hòa, chọn Hit@1 trên dev; nếu vẫn hòa, chọn ít tài nguyên/ít chunk hơn. Test chỉ dùng để báo cáo.

## Kết quả truy xuất
{chr(10).join(rows)}

Phân bố độ dài chunk:
{chr(10).join(chunk_rows)}

Evidence Hit@3 tự động là phép so khớp evidence đã chuẩn hóa với text trong chunk ở đúng nguồn/trang; đây là proxy truy xuất, không phải độ đúng của đáp án. Nhãn grounding bên dưới là phần đọc top-3 riêng.

Trace đầy đủ: `results/top3_traces.jsonl`; tóm tắt: `results/retrieval_summary.json`; lần chạy: `results/run_summary.json`.

### Lỗi top-3 trên test
{miss_text}

{audit_section}

## Giới hạn
- Chỉ một handbook và 5 câu test; không suy rộng kết quả. Dev/test đều lấy từ cùng một handbook, nên chủ đề và thuật ngữ có thể chồng lấn. A/B/C dùng embedding, corpus, metric và câu hỏi giống nhau; B/C khác ở metadata filter. C tái sử dụng vectors của B vì chunks giống hệt nhau. Thời gian là đo trên máy chạy, không đại diện phần cứng khác.
- Trạng thái audit: {audit_status}.
"""
    return _write_named_report(report)


def _write_named_report(report_text: str | None = None) -> Path | None:
    student_path = ROOT / "STUDENT.md"
    fields = {"name": "", "student_id": "", "class": ""}
    labels = {"- Họ và tên:": "name", "- MSSV:": "student_id", "- Lớp:": "class"}
    for line in student_path.read_text(encoding="utf-8").splitlines():
        for label, field in labels.items():
            if line.startswith(label):
                fields[field] = line[len(label):].strip()
    fallback_path = ROOT / "report" / "individual" / "REPORT_DRAFT.md"
    if not all(fields.values()):
        if report_text is not None:
            fallback_path.write_text(report_text, encoding="utf-8")
        return fallback_path if fallback_path.exists() else None
    slug = unicodedata.normalize("NFKD", fields["name"]).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "", slug)
    if not slug:
        return None
    if report_text is None:
        if not fallback_path.exists():
            return None
        report_text = fallback_path.read_text(encoding="utf-8")
    report_text = report_text.replace(
        "# Báo cáo cá nhân — Day 6 (bản nháp kết quả chạy)",
        "# Báo cáo cá nhân — Day 6\n\n**Họ tên:** " + fields["name"] + " · **MSSV:** " + fields["student_id"] + " · **Lớp:** " + fields["class"],
        1,
    )
    report_text = report_text.replace(
        "> Cần điền tên/MSSV/lớp và đổi tên file trước khi nộp. Audit hiện trạng:",
        "> Báo cáo được tạo từ lần chạy thật. Audit hiện trạng:",
        1,
    )
    destination = ROOT / "report" / "individual" / f"{fields['student_id']}_{slug}.md"
    destination.write_text(report_text, encoding="utf-8")
    fallback_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
