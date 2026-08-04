from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.paper.resolver import PaperResolver, ResolutionConfig, write_resolution_outputs
from scholarpath.paper.schema import PaperRecord


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and deduplicate scholarly paper JSONL records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--default-source", default="unknown")
    parser.add_argument("--max-year-gap", type=int, default=2)
    parser.add_argument("--fuzzy-review-threshold", type=float, default=0.93)
    args = parser.parse_args()

    records: list[PaperRecord] = []
    with Path(args.input).open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] Line {line_no}: invalid JSON: {exc}")
                return 2
            if not isinstance(payload, dict):
                print(f"[ERROR] Line {line_no}: expected JSON object")
                return 2
            record = PaperRecord.from_mapping(payload, default_source=args.default_source)
            if not record.title:
                print(f"[WARNING] Line {line_no}: empty title")
            records.append(record)

    resolver = PaperResolver(
        ResolutionConfig(
            max_year_gap=args.max_year_gap,
            fuzzy_review_threshold=args.fuzzy_review_threshold,
        )
    )
    result = resolver.resolve(records)
    paths = write_resolution_outputs(result, args.output_dir)
    report = result.report()

    print(f"[OK] Input records: {report['input_records']}")
    print(f"[OK] Resolved entities: {report['resolved_entities']}")
    print(f"[OK] Removed duplicates: {report['removed_duplicates']}")
    print(f"[OK] Exact-ID merges: {report['exact_id_merges']}")
    print(f"[OK] Title/author/year merges: {report['title_author_year_merges']}")
    print(f"[INFO] Review candidate pairs: {report['review_candidate_pairs']}")
    for name, path in paths.items():
        print(f"[OUTPUT] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
