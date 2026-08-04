from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.paper.gold import build_gold_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build normalized RealScholarQuery gold records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = build_gold_records(args.input, args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Queries: {report['queries']}")
    print(f"[OK] Gold papers: {report['gold_papers']}")
    print(f"[OK] Length mismatches: {report['length_mismatches']}")
    print(f"[INFO] Empty arXiv IDs: {report['empty_arxiv_ids']}")
    print(f"[OK] Output: {report['output_path']}")
    print(f"[OK] Report: {report_path}")
    return 2 if report["length_mismatches"] or report["empty_titles"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
