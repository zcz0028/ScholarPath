from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scholarpath.paper.schema import PaperRecord


PREDICTION_LIST_FIELDS = ("papers", "predictions", "results", "answer")


def normalize_qid(value: Any) -> str:
    if value is None:
        raise ValueError("Each record must contain a non-null 'qid'.")
    text = str(value).strip()
    if not text:
        raise ValueError("Each record must contain a non-empty 'qid'.")
    return text


def paper_from_item(item: Any, default_source: str) -> PaperRecord:
    if isinstance(item, str):
        return PaperRecord(title=item, source=default_source)
    if isinstance(item, Mapping):
        return PaperRecord.from_mapping(item, default_source=default_source)
    raise ValueError(
        "Paper items must be title strings or JSON objects, "
        f"got {type(item).__name__}."
    )


def parse_gold_papers(record: Mapping[str, Any]) -> list[PaperRecord]:
    gold_papers = record.get("gold_papers")
    if isinstance(gold_papers, list):
        return [
            paper_from_item(item, default_source="gold")
            for item in gold_papers
        ]

    titles = record.get("answer")
    if not isinstance(titles, list):
        raise ValueError(
            "Gold record must contain 'gold_papers' or list-valued 'answer'."
        )

    arxiv_ids = record.get("answer_arxiv_id")
    if arxiv_ids is None:
        return [
            paper_from_item(item, default_source="gold")
            for item in titles
        ]
    if not isinstance(arxiv_ids, list):
        raise ValueError("'answer_arxiv_id' must be a list when present.")
    if len(titles) != len(arxiv_ids):
        raise ValueError(
            "'answer' and 'answer_arxiv_id' must have equal lengths."
        )

    papers: list[PaperRecord] = []
    for title, arxiv_id in zip(titles, arxiv_ids, strict=True):
        papers.append(
            PaperRecord(
                title=str(title or ""),
                arxiv_id=arxiv_id,
                source="gold",
            )
        )
    return papers


def parse_prediction_papers(record: Mapping[str, Any]) -> list[PaperRecord]:
    for field_name in PREDICTION_LIST_FIELDS:
        value = record.get(field_name)
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"'{field_name}' must be a list.")
            return [
                paper_from_item(item, default_source="prediction")
                for item in value
            ]
    raise ValueError(
        "Prediction record must contain one of: "
        + ", ".join(PREDICTION_LIST_FIELDS)
    )


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"File does not exist: {source_path}")

    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source_path}, line {line_no}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source_path}, line {line_no}: expected JSON object."
                )
            yield line_no, value


def load_gold(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_no, raw in read_jsonl(path):
        try:
            qid = normalize_qid(raw.get("qid"))
            papers = parse_gold_papers(raw)
        except ValueError as exc:
            raise ValueError(f"Gold line {line_no}: {exc}") from exc
        if qid in records:
            raise ValueError(f"Duplicate gold qid: {qid}")
        records[qid] = {
            "qid": qid,
            "question": raw.get("question"),
            "papers": papers,
        }
    return records


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_no, raw in read_jsonl(path):
        try:
            qid = normalize_qid(raw.get("qid"))
            papers = parse_prediction_papers(raw)
        except ValueError as exc:
            raise ValueError(f"Prediction line {line_no}: {exc}") from exc
        if qid in records:
            raise ValueError(f"Duplicate prediction qid: {qid}")
        records[qid] = {
            "qid": qid,
            "question": raw.get("question"),
            "papers": papers,
        }
    return records
