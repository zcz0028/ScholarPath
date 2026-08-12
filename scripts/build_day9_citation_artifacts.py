from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.services.artifact_service import ArtifactService
from apps.api.settings import ApiSettings
from apps.api.services.search_service import (
    _apply_day8_stack,
    _paper_openalex_id,
)

from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.retrieval.citation_path_builder import (
    build_citation_path,
)

def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            payload = json.loads(value)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("qid") or "").strip(),
        str(row.get("result_openalex_id") or "").strip(),
    )


def _merge_rows(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        key = _row_key(row)
        if all(key):
            merged[key] = row
    for row in incoming:
        key = _row_key(row)
        if all(key):
            merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("qid") or ""),
            int(row.get("result_rank") or 10**9),
            str(row.get("result_openalex_id") or ""),
        ),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_day9_artifacts(
    *,
    output_dir: Path,
    top_k: int = 20,
    only_qid: str | None = None,
) -> dict[str, Any]:
    settings = ApiSettings()
    artifacts = ArtifactService(settings)
    queries = artifacts.benchmark_queries()
    day4_predictions = artifacts.day4_predictions(100)
    decomposer = ConstraintDecomposer()

    citation_path_file = output_dir / "citation_paths.jsonl"
    existing_rows = _load_existing_rows(citation_path_file)
    existing_keys = {_row_key(row) for row in existing_rows if all(_row_key(row))}

    citation_rows: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []
    total_results = 0
    total_paths = 0
    skipped_no_openalex = 0
    failed_paths = 0

    for qid, query_row in queries.items():
        if only_qid and qid != only_qid:
            continue

        prediction = day4_predictions.get(qid)
        if not prediction:
            continue

        question = str(query_row.get("question") or "").strip()
        if not question:
            continue

        decomposition = decomposer.decompose(question)
        day8_record = _apply_day8_stack(dict(prediction), decomposition)

        papers = (
            day8_record.get("papers")
            if isinstance(day8_record.get("papers"), list)
            else []
        )
        ranked_papers = [
            paper
            for paper in papers[:top_k]
            if isinstance(paper, dict)
        ]

        q_result_count = 0
        q_path_count = 0

        for rank, paper in enumerate(ranked_papers, start=1):
            q_result_count += 1
            total_results += 1

            openalex_id = _paper_openalex_id(paper)
            if not openalex_id:
                skipped_no_openalex += 1
                continue

            if (qid, openalex_id) in existing_keys:
                q_path_count += 1
                total_paths += 1
                continue

            citation_path = build_citation_path(
                paper=paper,
                qid=qid,
                seed_rank=rank,
            )
            if not citation_path:
                failed_paths += 1
                continue

            citation_rows.append(
                {
                    "qid": qid,
                    "result_rank": rank,
                    "result_openalex_id": openalex_id,
                    "result_title": str(paper.get("title") or ""),
                    **citation_path,
                }
            )
            q_path_count += 1
            total_paths += 1

        query_summaries.append(
            {
                "qid": qid,
                "result_count": q_result_count,
                "path_count": q_path_count,
                "coverage": (
                    q_path_count / q_result_count
                    if q_result_count
                    else 0.0
                ),
            }
        )
        print("[Day9-2B]", qid, f"coverage={q_path_count}/{q_result_count}")

    summary = {
        "top_k": top_k,
        "query_count": len(query_summaries),
        "total_results": total_results,
        "total_paths": total_paths,
        "coverage": (
            total_paths / total_results
            if total_results
            else 0.0
        ),
        "skipped_no_openalex": skipped_no_openalex,
        "failed_paths": failed_paths,
        "queries": query_summaries,
    }

    manifest = {
        "stage": "day9_2b_final_ranked_citation_artifacts",
        "source_benchmark": str(settings.resolved_benchmark_path),
        "source_day4_dir": str(settings.resolved_day4_dir),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "offline_generation": True,
        "benchmark_runtime_api_calls": 0,
    }

    merged_rows = _merge_rows(existing_rows, citation_rows)
    _write_jsonl(citation_path_file, merged_rows)
    _write_json(output_dir / "coverage_summary.json", summary)
    _write_json(output_dir / "execution_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final-ranked Day9 citation artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/week2_day9_citation",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--qid", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_day9_artifacts(
        output_dir=Path(args.output_dir),
        top_k=max(1, args.top_k),
        only_qid=args.qid,
    )
    print()
    print("[Day9-2B] completed")
    print(
        f"coverage={summary['total_paths']}/"
        f"{summary['total_results']} "
        f"({summary['coverage']:.4f})"
    )


if __name__ == "__main__":
    main()
