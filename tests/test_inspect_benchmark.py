from __future__ import annotations

import json
from pathlib import Path

from scholarpath.data.inspect_benchmark import (
    classify_answer_item,
    inspect_jsonl,
    report_to_markdown,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )


def test_pasa_like_schema(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    write_jsonl(
        path,
        [
            {
                "question": "Find papers about a synthetic topic.",
                "answer": ["A Synthetic Paper", "Another Synthetic Paper"],
                "source_meta": {"published_time": "20241001"},
            },
            {
                "question": "Find papers about another synthetic topic.",
                "answer": ["2401.01234"],
                "source_meta": {"published_time": "20241001"},
            },
        ],
    )

    report = inspect_jsonl(path)

    assert report["detected_schema"]["query_field"] == "question"
    assert report["detected_schema"]["answer_field"] == "answer"
    assert report["detected_schema"]["date_path"] == ["source_meta", "published_time"]
    assert report["record_counts"]["valid_json_objects"] == 2
    assert report["quality"]["issue_counts"].get("error", 0) == 0
    assert report["answer_analysis"]["identifier_kinds"]["title_string"] == 2
    assert report["answer_analysis"]["identifier_kinds"]["arxiv_id"] == 1


def test_invalid_query_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    write_jsonl(
        path,
        [
            {
                "question": "",
                "answer": [],
                "source_meta": {"published_time": "not-a-date"},
            }
        ],
    )

    report = inspect_jsonl(path)
    codes = {issue["code"] for issue in report["issues"]}

    assert "invalid_query" in codes
    assert "date_unparseable" in codes
    assert report["quality"]["issue_counts"]["error"] == 1


def test_answer_classifier() -> None:
    assert classify_answer_item("2401.01234v2") == "arxiv_id"
    assert classify_answer_item("10.1000/example") == "doi"
    assert classify_answer_item("A Paper Title") == "title_string"
    assert classify_answer_item({"title": "A Paper"}) == "dict_with_title"


def test_markdown_contains_gate(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    write_jsonl(
        path,
        [
            {
                "question": "A valid query",
                "answer": ["A Valid Paper"],
                "source_meta": {"published_time": "20241001"},
            }
        ],
    )
    report = inspect_jsonl(path)
    markdown = report_to_markdown(report)
    assert "是否允许进入下一步" in markdown
    assert "条件通过" in markdown
