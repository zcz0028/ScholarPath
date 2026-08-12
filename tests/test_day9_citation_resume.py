from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_day9_citation_artifacts import _load_existing_rows, _merge_rows


def test_load_existing_rows_missing_file(tmp_path: Path) -> None:
    assert _load_existing_rows(tmp_path / "missing.jsonl") == []


def test_resume_merge_preserves_existing_success() -> None:
    existing = [{"qid": "q1", "result_openalex_id": "W1", "expanded_openalex_id": "W2"}]
    incoming = [{"qid": "q1", "result_openalex_id": "W3", "expanded_openalex_id": "W4"}]
    merged = _merge_rows(existing, incoming)
    assert {row["result_openalex_id"] for row in merged} == {"W1", "W3"}


def test_resume_merge_replaces_same_result_with_new_success() -> None:
    existing = [{"qid": "q1", "result_openalex_id": "W1", "expanded_openalex_id": "OLD"}]
    incoming = [{"qid": "q1", "result_openalex_id": "W1", "expanded_openalex_id": "NEW"}]
    merged = _merge_rows(existing, incoming)
    assert len(merged) == 1
    assert merged[0]["expanded_openalex_id"] == "NEW"


def test_resume_merge_keeps_other_qids() -> None:
    existing = [{"qid": "q_other", "result_openalex_id": "W9", "expanded_openalex_id": "W10"}]
    incoming = [{"qid": "q1", "result_openalex_id": "W1", "expanded_openalex_id": "W2"}]
    merged = _merge_rows(existing, incoming)
    assert {row["qid"] for row in merged} == {"q_other", "q1"}
