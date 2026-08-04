import json
from pathlib import Path

from scholarpath.paper.gold import build_gold_records


def test_build_gold_records(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    target = tmp_path / "gold.jsonl"
    source.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "Find papers.",
                "answer": ["Paper One", "Paper Two"],
                "answer_arxiv_id": ["2401.01234v2", "cs/9901001"],
                "source_meta": {"published_time": "20241001"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_gold_records(source, target)
    assert report["queries"] == 1
    assert report["gold_papers"] == 2
    output = json.loads(target.read_text(encoding="utf-8").strip())
    assert output["gold_papers"][0]["arxiv_id"] == "2401.01234"
    assert output["gold_papers"][0]["canonical_id"] == "arxiv:2401.01234"
