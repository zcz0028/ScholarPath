from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholarpath.evaluation.io import load_gold, load_predictions


def test_load_raw_realscholarquery_style_gold(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "Find papers.",
                "answer": ["Paper A"],
                "answer_arxiv_id": ["2401.01234v2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_gold(path)
    assert loaded["q1"]["papers"][0].arxiv_id == "2401.01234"


def test_load_prediction_title_strings(tmp_path: Path) -> None:
    path = tmp_path / "pred.jsonl"
    path.write_text(
        json.dumps({"qid": "q1", "papers": ["Paper A"]}) + "\n",
        encoding="utf-8",
    )
    loaded = load_predictions(path)
    assert loaded["q1"]["papers"][0].title == "Paper A"


def test_duplicate_qid_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pred.jsonl"
    path.write_text(
        json.dumps({"qid": "q1", "papers": []})
        + "\n"
        + json.dumps({"qid": "q1", "papers": []})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate prediction qid"):
        load_predictions(path)
