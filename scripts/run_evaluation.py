from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.metrics import (
    EvaluationConfig,
    evaluate_records,
    write_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scholarly paper retrieval predictions."
    )
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=("strict", "strict_v2", "pasa_title"),
        default="strict",
    )
    parser.add_argument(
        "--recall-at",
        nargs="+",
        type=int,
        default=[20, 50, 100],
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Do not remove exact duplicate predictions before scoring.",
    )
    parser.add_argument(
        "--strict-qids",
        action="store_true",
        help="Fail when prediction qids do not exactly match gold qids.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gold = load_gold(args.gold)
    predictions = load_predictions(args.predictions)

    result = evaluate_records(
        gold_records=gold,
        prediction_records=predictions,
        config=EvaluationConfig(
            mode=args.mode,
            recall_at=tuple(args.recall_at),
            deduplicate=not args.no_deduplicate,
        ),
    )

    if args.strict_qids and (
        result.missing_prediction_qids or result.extra_prediction_qids
    ):
        print("[ERROR] Prediction qids do not match gold qids.")
        print(
            "[ERROR] Missing qids: "
            + json.dumps(result.missing_prediction_qids, ensure_ascii=False)
        )
        print(
            "[ERROR] Extra qids: "
            + json.dumps(result.extra_prediction_qids, ensure_ascii=False)
        )
        return 2

    paths = write_evaluation_outputs(result, args.output_dir)
    summary = result.summary()
    macro = summary["macro"]
    micro = summary["micro"]

    print(f"[OK] Matching mode: {summary['matching_mode']}")
    print(f"[OK] Queries: {summary['query_count']}")
    print(
        "[OK] Macro P/R/F1: "
        f"{macro['precision']:.6f} / "
        f"{macro['recall']:.6f} / "
        f"{macro['f1']:.6f}"
    )
    print(
        "[OK] Micro P/R/F1: "
        f"{micro['precision']:.6f} / "
        f"{micro['recall']:.6f} / "
        f"{micro['f1']:.6f}"
    )
    print(
        "[INFO] Missing prediction qids: "
        f"{len(result.missing_prediction_qids)}"
    )
    print(
        "[INFO] Extra prediction qids: "
        f"{len(result.extra_prediction_qids)}"
    )
    for name, path in paths.items():
        print(f"[OUTPUT] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
