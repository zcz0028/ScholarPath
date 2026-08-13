from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "prepare_day12_qe_retrieval_ablation.py"
)
DAY4_RUNNER = (
    PROJECT_ROOT
    / "scripts"
    / "run_day4_rescue_retrieval.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day12_query_expansion_retrieval"
)
QE_PLANS = OUTPUT_DIR / "query_plans_qe_r9.jsonl"

FROZEN_DAY4_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "week2_day4_rescue"
    / "run_summary.json"
)

EXPERIMENT_SUMMARY = OUTPUT_DIR / "run_summary.json"
COMPARISON_PATH = OUTPUT_DIR / "qe_vs_frozen_comparison.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def strict_metric(
    summary: dict[str, Any],
    k: int,
) -> dict[str, float | int]:
    item = (summary.get("evaluation") or {}).get(f"top{k}_strict") or {}
    counts = item.get("counts") or {}
    macro = item.get("macro") or {}
    micro = item.get("micro") or {}
    return {
        "tp": int(counts.get("tp", 0)),
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "micro_precision": float(micro.get("precision", 0.0)),
        "micro_recall": float(micro.get("recall", 0.0)),
        "micro_f1": float(micro.get("f1", 0.0)),
    }


def compare_summaries(
    baseline: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    by_k: dict[str, Any] = {}

    for k in (20, 50, 100):
        before = strict_metric(baseline, k)
        after = strict_metric(experiment, k)

        by_k[str(k)] = {
            "baseline": before,
            "qe_r9": after,
            "delta": {
                key: after[key] - before[key]
                for key in before
            },
        }

    retrieval = experiment.get("retrieval") or {}
    additional_api_calls = int(retrieval.get("actual_api_calls", 0))
    cache_hits = int(retrieval.get("cache_hits", 0))
    retries = int(retrieval.get("retries", 0))
    estimated_cost = float(
        experiment.get("estimated_cost_usd", 0.0) or 0.0
    )
    latency = experiment.get("latency_ms") or {}

    top20_delta = by_k["20"]["delta"]
    top100_delta = by_k["100"]["delta"]

    adoption_checks = {
        "top100_tp_gain_positive": top100_delta["tp"] > 0,
        "top100_recall_gain_positive": (
            top100_delta["macro_recall"] > 0
        ),
        "top100_f1_gain_positive": (
            top100_delta["macro_f1"] > 0
        ),
        "top20_precision_not_materially_degraded": (
            top20_delta["macro_precision"] >= -0.002
        ),
    }

    adopt_candidate = all(adoption_checks.values())

    tp_gain = int(top100_delta["tp"])
    api_calls_per_added_tp = (
        additional_api_calls / tp_gain
        if tp_gain > 0
        else None
    )

    return {
        "schema_version": "day12.qe-r9.v1",
        "scope": "9_residual_retrieval_failures_only",
        "baseline": "frozen_day4_rescue",
        "experiment": "day4_plus_qe_r9",
        "ranking_note": (
            "This experiment reuses the Day4 fusion and semantic reranking "
            "path. It measures retrieval-stage QE impact and does not modify "
            "the frozen Day8 E3 artifacts. If adopted, E3 integration should "
            "be validated separately."
        ),
        "metrics_by_k": by_k,
        "efficiency": {
            "additional_api_calls": additional_api_calls,
            "cache_hits": cache_hits,
            "retries": retries,
            "estimated_cost_usd": estimated_cost,
            "latency_ms": latency,
            "api_calls_per_added_top100_tp": api_calls_per_added_tp,
        },
        "adoption_checks": adoption_checks,
        "adopt_candidate": adopt_candidate,
        "decision": (
            "candidate_for_e3_integration"
            if adopt_candidate
            else "keep_frozen_pipeline"
        ),
        "reporting_boundaries": [
            (
                "QE-R9 targets only residual retrieval failures; it is not "
                "a full 50-query query-expansion rollout."
            ),
            (
                "The experiment uses gold only during evaluation performed "
                "by the existing Day4 runner."
            ),
            (
                "Do not attribute QE-R9 results to the frozen Day8 E3 "
                "ranking until a separate E3 integration experiment passes."
            ),
        ],
    }


def write_comparison(value: dict[str, Any]) -> None:
    COMPARISON_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_comparison(value: dict[str, Any]) -> None:
    print()
    print("=== QE-R9 vs Frozen Day4 ===")
    for k in ("20", "50", "100"):
        row = value["metrics_by_k"][k]
        print(
            f"Top{k}: "
            f"TP {row['baseline']['tp']} -> {row['qe_r9']['tp']} "
            f"(delta={row['delta']['tp']:+d}), "
            f"MacroRecall {row['baseline']['macro_recall']:.6f} -> "
            f"{row['qe_r9']['macro_recall']:.6f}, "
            f"MacroF1 {row['baseline']['macro_f1']:.6f} -> "
            f"{row['qe_r9']['macro_f1']:.6f}"
        )

    efficiency = value["efficiency"]
    print()
    print("=== Efficiency ===")
    print("Additional API calls:", efficiency["additional_api_calls"])
    print("Cache hits:", efficiency["cache_hits"])
    print("Retries:", efficiency["retries"])
    print(
        "Estimated cost USD:",
        efficiency["estimated_cost_usd"],
    )
    print(
        "API calls / added Top100 TP:",
        efficiency["api_calls_per_added_top100_tp"],
    )

    print()
    print("=== Decision ===")
    for name, passed in value["adoption_checks"].items():
        print(f"{name}: {passed}")
    print("decision:", value["decision"])


def run_command(command: list[str]) -> int:
    print("[Day12-3-2] Running:")
    print(" ".join(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Day12-3-2 QE-R9 real OpenAlex retrieval ablation "
            "on residual retrieval failures only."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare plans and ask the Day4 runner for its execution manifest only.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--per-plan-page",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=9,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepare_rc = run_command(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
        ]
    )
    if prepare_rc != 0:
        return prepare_rc

    command = [
        sys.executable,
        str(DAY4_RUNNER),
        "--query-plans",
        str(QE_PLANS),
        "--base-pool",
        str(
            PROJECT_ROOT
            / "outputs"
            / "week2_day4_rescue"
            / "merged_candidates_before_rerank.jsonl"
        ),
        "--base-predictions-dir",
        str(
            PROJECT_ROOT
            / "outputs"
            / "week2_day4_rescue"
        ),
        "--output-dir",
        str(OUTPUT_DIR),
        "--per-plan-page",
        str(args.per_plan_page),
        "--rescue-top-n",
        "100",
        "--max-merged-candidates",
        "300",
        "--max-plans-per-query",
        "1",
        "--max-api-calls",
        str(args.max_api_calls),
        "--semantic-mode",
        "semantic_first",
        "--top-k",
        "20",
        "50",
        "100",
        "--cache-dir",
        str(
            PROJECT_ROOT
            / "data"
            / "cache"
            / "openalex_day12_qe_r9"
        ),
    ]

    if args.refresh_cache:
        command.append("--refresh-cache")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.dry_run:
        command.append("--dry-run")

    run_rc = run_command(command)
    if run_rc != 0:
        return run_rc

    if args.dry_run:
        print("[Day12-3-2] Dry run complete; no comparison generated.")
        return 0

    if not FROZEN_DAY4_SUMMARY.is_file():
        print(
            f"[Day12-3-2] Missing baseline summary: {FROZEN_DAY4_SUMMARY}",
            file=sys.stderr,
        )
        return 2
    if not EXPERIMENT_SUMMARY.is_file():
        print(
            f"[Day12-3-2] Missing experiment summary: {EXPERIMENT_SUMMARY}",
            file=sys.stderr,
        )
        return 2

    baseline = read_json(FROZEN_DAY4_SUMMARY)
    experiment = read_json(EXPERIMENT_SUMMARY)
    comparison = compare_summaries(baseline, experiment)
    write_comparison(comparison)
    print_comparison(comparison)

    print()
    print(f"[Day12-3-2] comparison = {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
