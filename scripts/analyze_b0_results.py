from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.evaluation.error_analysis import B0AnalysisConfig, analyze_b0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive and analyze B0 OpenAlex baseline results."
    )
    parser.add_argument(
        "--b0-dir",
        default="outputs/b0_openalex",
        help="Directory containing B0 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/b0_openalex_analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--mode",
        default="strict",
        choices=["strict", "pasa_title"],
        help="Matching mode to analyze.",
    )
    parser.add_argument(
        "--focus-topk",
        type=int,
        default=50,
        help="Top-K setting used as the main B0 result.",
    )
    parser.add_argument(
        "--rank-source-topk",
        type=int,
        default=100,
        help="Per-query Top-K file used for rank-position analysis.",
    )
    parser.add_argument(
        "--top-n-queries",
        type=int,
        default=20,
        help="Number of high-FP queries to output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = analyze_b0(
        B0AnalysisConfig(
            b0_dir=Path(args.b0_dir),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            focus_topk=args.focus_topk,
            rank_source_topk=args.rank_source_topk,
            top_n_queries=args.top_n_queries,
        )
    )

    focus = summary.get("focus", {}).get("result") or {}
    errors = summary.get("error_analysis", {})
    print(f"[OK] Output dir: {args.output_dir}")
    print(
        "[OK] Focus result: "
        f"top{args.focus_topk}/{args.mode} "
        f"MacroF1={focus.get('macro_f1')}"
    )
    print(f"[OK] Zero-recall queries: {errors.get('zero_recall_queries')}")
    print(f"[OK] Late-recall queries: {errors.get('late_recall_queries')}")
    print(
        "[OK] Matched rank distribution: "
        + json.dumps(errors.get("rank_distribution"), ensure_ascii=False)
    )
    print(f"[OUTPUT] Summary: {Path(args.output_dir) / 'analysis_summary.json'}")
    print(f"[OUTPUT] Recommendations: {Path(args.output_dir) / 'b1_recommendations.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
