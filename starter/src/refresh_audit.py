"""Refresh audit summary and report without re-running ingestion or embeddings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import run


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    expected_counts = {"chunk_coherence": 30, "metadata_accuracy": 45, "grounding_quality": 15}
    actual_counts = {kind: sum(row.get("audit_type") == kind for row in rows) for kind in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"Expected audit rows {expected_counts}; found {actual_counts}")

    problems: list[str] = []
    for row in rows:
        label = row.get("metadata_correct", "") if row["audit_type"] == "metadata_accuracy" else row.get("rating_0_1_2", "")
        if not label.strip():
            problems.append(f"{row['audit_type']} / {row['sample_id']} / {row['strategy']}: missing label")
        elif row["audit_type"] != "metadata_accuracy" and label not in {"0", "1", "2"}:
            problems.append(f"{row['audit_type']} / {row['sample_id']} / {row['strategy']}: rating must be 0, 1, or 2")
        elif row["audit_type"] == "metadata_accuracy" and label.casefold() not in {"true", "false"}:
            problems.append(f"{row['audit_type']} / {row['sample_id']} / {row['strategy']}: metadata label must be True or False")
        for field in ("reason", "reviewer"):
            if not row.get(field, "").strip():
                problems.append(f"{row['audit_type']} / {row['sample_id']} / {row['strategy']}: missing {field}")
            if field == "reviewer" and any(tag in row[field].casefold() for tag in ("codex", "ai-assisted", "student verify")):
                problems.append(f"{row['audit_type']} / {row['sample_id']} / {row['strategy']}: replace preliminary reviewer tag after your review")

    if problems:
        preview = "\n".join(f"- {item}" for item in problems[:20])
        remainder = f"\n... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(f"Audit is not ready to finalize:\n{preview}{remainder}")
    return rows


def refresh() -> dict:
    root = run.ROOT
    audit_rows = _validated_audit_rows(root / "results" / "manual_audit_template.csv")
    with (root / "results" / "top3_traces.jsonl").open("r", encoding="utf-8") as file:
        traces = [json.loads(line) for line in file if line.strip()]

    audit_summary = run._audit_summary(audit_rows, traces)
    if audit_summary["review_status"] != "Reviewer-labelled audit; author verification recorded in CSV":
        raise ValueError(f"Audit reviewer status is not final: {audit_summary['review_status']}")
    if audit_summary["filtered_test_queries"]["compliant"] != audit_summary["filtered_test_queries"]["total"]:
        raise ValueError("At least one filtered test query has a top-3 result outside its requested chapter")

    run_summary_path = root / "results" / "run_summary.json"
    run_summary = _load_json(run_summary_path)
    run_summary["manual_audit"] = audit_summary
    run_summary["manual_audit_status"] = audit_summary["review_status"]
    run._write_json(run_summary_path, run_summary)
    run._write_json(root / "results" / "manual_audit_summary.json", audit_summary)
    run._update_experiment_log(
        run_summary["metrics"], run_summary["chunk_statistics"], run_summary["timings"],
        run_summary["embedding"]["dimensions"], audit_summary["review_status"],
    )

    with (root / "results" / "benchmark_results.csv").open("r", newline="", encoding="utf-8-sig") as file:
        result_rows = list(csv.DictReader(file))
    report_path = run._write_report_draft(run_summary, result_rows, traces)
    return {"audit": audit_summary, "report": str(report_path) if report_path else None}


if __name__ == "__main__":
    try:
        print(json.dumps(refresh(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
