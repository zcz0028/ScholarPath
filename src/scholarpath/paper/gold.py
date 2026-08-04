from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import PaperRecord


def build_gold_records(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source_path = Path(input_path)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "queries": 0,
        "gold_papers": 0,
        "length_mismatches": 0,
        "empty_titles": 0,
        "empty_arxiv_ids": 0,
        "duplicate_arxiv_ids_within_query": 0,
        "duplicate_titles_within_query": 0,
        "output_path": str(target_path),
    }

    with source_path.open("r", encoding="utf-8-sig") as source, target_path.open("w", encoding="utf-8") as target:
        for line_no, raw in enumerate(source, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            titles = data.get("answer")
            arxiv_ids = data.get("answer_arxiv_id")
            if not isinstance(titles, list):
                raise ValueError(f"Line {line_no}: 'answer' must be a list")
            if not isinstance(arxiv_ids, list):
                raise ValueError(f"Line {line_no}: 'answer_arxiv_id' must be a list")
            if len(titles) != len(arxiv_ids):
                report["length_mismatches"] += 1
                raise ValueError(
                    f"Line {line_no}: answer={len(titles)}, answer_arxiv_id={len(arxiv_ids)}"
                )

            papers: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            seen_titles: set[str] = set()
            for title, arxiv_id in zip(titles, arxiv_ids, strict=True):
                paper = PaperRecord(
                    title=str(title or ""),
                    arxiv_id=arxiv_id,
                    source="realscholarquery_gold",
                )
                if not paper.title:
                    report["empty_titles"] += 1
                if not paper.arxiv_id:
                    report["empty_arxiv_ids"] += 1
                if paper.arxiv_id and paper.arxiv_id in seen_ids:
                    report["duplicate_arxiv_ids_within_query"] += 1
                if paper.normalized_title and paper.normalized_title in seen_titles:
                    report["duplicate_titles_within_query"] += 1
                if paper.arxiv_id:
                    seen_ids.add(paper.arxiv_id)
                if paper.normalized_title:
                    seen_titles.add(paper.normalized_title)
                papers.append(paper.to_dict())
                report["gold_papers"] += 1

            target.write(
                json.dumps(
                    {
                        "qid": data.get("qid"),
                        "question": data.get("question"),
                        "published_time": (
                            data.get("source_meta", {}).get("published_time")
                            if isinstance(data.get("source_meta"), dict)
                            else None
                        ),
                        "gold_papers": papers,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            report["queries"] += 1
    return report
