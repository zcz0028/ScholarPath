from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_PATH = Path("data/raw/RealScholarQuery/test.jsonl")


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def main() -> int:
    if not DATA_PATH.exists():
        print(f"[ERROR] File not found: {DATA_PATH}")
        return 1

    record_count = 0
    length_mismatch_count = 0
    missing_arxiv_field_count = 0
    empty_arxiv_id_count = 0
    answer_types: Counter[str] = Counter()
    arxiv_container_types: Counter[str] = Counter()
    arxiv_item_types: Counter[str] = Counter()

    first_records: list[dict[str, Any]] = []

    with DATA_PATH.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            record_count += 1

            answers = record.get("answer")
            answer_arxiv_ids = record.get("answer_arxiv_id")

            answer_types[value_type(answers)] += 1
            arxiv_container_types[value_type(answer_arxiv_ids)] += 1

            if "answer_arxiv_id" not in record:
                missing_arxiv_field_count += 1
                print(
                    f"[ERROR] Line {line_no}: "
                    "'answer_arxiv_id' field is missing."
                )
                continue

            if isinstance(answer_arxiv_ids, list):
                for arxiv_id in answer_arxiv_ids:
                    arxiv_item_types[value_type(arxiv_id)] += 1

                    if arxiv_id is None:
                        empty_arxiv_id_count += 1
                    elif isinstance(arxiv_id, str) and not arxiv_id.strip():
                        empty_arxiv_id_count += 1

            if isinstance(answers, list) and isinstance(answer_arxiv_ids, list):
                if len(answers) != len(answer_arxiv_ids):
                    length_mismatch_count += 1
                    print(
                        f"[ERROR] Line {line_no}: "
                        f"answer={len(answers)}, "
                        f"answer_arxiv_id={len(answer_arxiv_ids)}"
                    )

            if len(first_records) < 3:
                first_records.append(
                    {
                        "qid": record.get("qid"),
                        "question": record.get("question"),
                        "answer": answers,
                        "answer_arxiv_id": answer_arxiv_ids,
                        "published_time": (
                            record.get("source_meta", {})
                            .get("published_time")
                        ),
                    }
                )

    print("\n========== Gold ID Structure Report ==========")
    print(f"Records: {record_count}")
    print(f"Missing answer_arxiv_id fields: {missing_arxiv_field_count}")
    print(f"Length mismatches: {length_mismatch_count}")
    print(f"Empty arXiv IDs: {empty_arxiv_id_count}")
    print(f"answer field types: {dict(answer_types)}")
    print(
        "answer_arxiv_id container types: "
        f"{dict(arxiv_container_types)}"
    )
    print(
        "answer_arxiv_id item types: "
        f"{dict(arxiv_item_types)}"
    )

    print("\n========== First Three Records ==========")
    for index, record in enumerate(first_records, start=1):
        print(f"\n--- Record {index} ---")
        print(json.dumps(record, ensure_ascii=False, indent=2))

    blocking_errors = (
        missing_arxiv_field_count
        + length_mismatch_count
    )

    if blocking_errors == 0:
        print("\n[PASS] Gold title and arXiv ID fields are structurally usable.")
        return 0

    print("\n[FAIL] Gold fields require additional parsing rules.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())