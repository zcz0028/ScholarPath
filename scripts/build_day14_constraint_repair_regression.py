from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.matching import match_papers
from scholarpath.query.constraints import ConstraintDecomposer, is_hard_fragment
from scholarpath.rerank.constraint_evidence import (
    build_canonical_constraints,
    build_constraint_evidence,
)

DEFAULT_GOLD = PROJECT_ROOT / "data/processed/realscholarquery_gold.jsonl"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/week2_day8_evidence_rerank/predictions_e3_selected.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/week2_day14_constraint_repair_regression_v2"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def by_qid(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): dict(row) for row in rows}

def mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0

def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0

def warning_reason(text: str, ctype: str) -> list[str]:
    import re
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    reasons: list[str] = []
    if len(tokens) <= 3 and ctype == "topic":
        reasons.append("short_topic_review")
    if len(tokens) >= 2 and tokens[-1] in {"model", "method", "task"}:
        reasons.append("generic_tail_review")
    return reasons

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    gold = load_gold(Path(args.gold))
    predictions = load_predictions(Path(args.predictions))
    raw = by_qid(read_jsonl(Path(args.predictions)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decomposer = ConstraintDecomposer()
    hard_fragments = total_constraints = hard_fragment_queries = 0
    warning_constraints = warning_queries = 0
    tp_cov: list[float] = []
    fp_cov: list[float] = []
    tp_generic: list[float] = []
    fp_generic: list[float] = []
    per_query: list[dict[str, Any]] = []

    for qid in sorted(gold):
        question = str(raw[qid].get("question") or "")
        decomposition = decomposer.decompose(question)
        canonical = build_canonical_constraints(question=question, constraints=decomposition.constraints)

        q_hard = q_warn = 0
        hard_texts: list[str] = []
        warning_texts: list[str] = []
        for item in canonical:
            total_constraints += 1
            if is_hard_fragment(item.canonical_text):
                hard_fragments += 1
                q_hard += 1
                hard_texts.append(item.canonical_text)
            reasons = warning_reason(item.canonical_text, item.constraint_type)
            if reasons:
                warning_constraints += 1
                q_warn += 1
                warning_texts.append(item.canonical_text)

        hard_fragment_queries += q_hard > 0
        warning_queries += q_warn > 0

        raw_papers = [dict(x) for x in (raw[qid].get("papers") or [])[:args.top_k]]
        eval_papers = list(predictions[qid].get("papers") or [])[:args.top_k]
        match = match_papers(predictions=eval_papers, gold=list(gold[qid].get("papers") or []), mode="strict")
        tp_indices = {p.prediction_index for p in match.pairs}

        for idx, paper in enumerate(raw_papers):
            evidence = build_constraint_evidence(canonical, paper)
            matched = [x for x in evidence if x.matched]
            total_weight = sum(max(0.0, float(c.weight)) for c in canonical)
            matched_ids = {x.constraint_id for x in matched}
            matched_weight = sum(max(0.0, float(c.weight)) for c in canonical if c.id in matched_ids)
            coverage = safe_divide(matched_weight, total_weight)
            generic = float(len(matched) == 1 and matched[0].constraint_type == "model_or_entity")
            if idx in tp_indices:
                tp_cov.append(coverage); tp_generic.append(generic)
            else:
                fp_cov.append(coverage); fp_generic.append(generic)

        per_query.append({
            "qid": qid,
            "question": question,
            "canonical_constraints": [x.to_dict() for x in canonical],
            "hard_fragment_count": q_hard,
            "hard_fragments": hard_texts,
            "warning_count": q_warn,
            "warnings": warning_texts,
        })

    tp_mean, fp_mean = mean(tp_cov), mean(fp_cov)
    ratio = tp_mean / fp_mean if fp_mean > 0 else None
    tp_generic_rate, fp_generic_rate = mean(tp_generic), mean(fp_generic)
    hard_rate = safe_divide(hard_fragments, total_constraints)
    hard_query_rate = safe_divide(hard_fragment_queries, len(gold))
    warning_rate = safe_divide(warning_constraints, total_constraints)
    warning_query_rate = safe_divide(warning_queries, len(gold))

    hard_pass = hard_rate <= 0.10 and hard_query_rate <= 0.20
    discrimination_pass = (
        tp_mean > fp_mean and ratio is not None and ratio >= 1.20
        and fp_generic_rate > tp_generic_rate
    )
    status = (
        "precise_constraint_repair_pass"
        if hard_pass and discrimination_pass
        else "hard_fragments_remain"
        if not hard_pass
        else "evidence_discrimination_regressed"
    )

    summary = {
        "schema_version": "day14.constraint-repair-v2",
        "query_count": len(gold),
        "top_k": args.top_k,
        "constraint_quality": {
            "constraint_count": total_constraints,
            "hard_fragment_count": hard_fragments,
            "hard_fragment_rate": hard_rate,
            "hard_fragment_query_count": hard_fragment_queries,
            "hard_fragment_query_rate": hard_query_rate,
            "warning_constraint_count": warning_constraints,
            "warning_constraint_rate": warning_rate,
            "warning_query_count": warning_queries,
            "warning_query_rate": warning_query_rate,
        },
        "evidence_discrimination": {
            "tp_mean_weighted_coverage": tp_mean,
            "fp_mean_weighted_coverage": fp_mean,
            "tp_to_fp_weighted_coverage_ratio": ratio,
            "tp_generic_entity_only_rate": tp_generic_rate,
            "fp_generic_entity_only_rate": fp_generic_rate,
        },
        "decision": {
            "status": status,
            "hard_fragment_pass": hard_pass,
            "evidence_discrimination_pass": discrimination_pass,
            "rules": {
                "hard_fragment_rate_max": 0.10,
                "hard_fragment_query_rate_max": 0.20,
                "tp_to_fp_weighted_coverage_ratio_min": 1.20,
                "fp_generic_entity_only_gt_tp": True,
            },
        },
        "boundary": {
            "api_calls": 0,
            "llm_calls": 0,
            "ranking_modified": False,
            "candidate_pool_modified": False,
            "gold_usage": "strict_offline_regression_only",
        },
    }

    (out / "repair_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (out / "optimization_decision.json").write_text(
        json.dumps(
            summary["decision"],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        out / "per_query_repaired_constraints.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in per_query:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print("[Day14-1C-2] Precise Constraint Repair Regression")
    print("API calls = 0")
    print("LLM calls = 0")
    print("ranking modified = False")
    print()
    print("hard_fragment_rate =", round(hard_rate, 6))
    print("hard_fragment_query_rate =", round(hard_query_rate, 6))
    print("warning_constraint_rate =", round(warning_rate, 6))
    print("warning_query_rate =", round(warning_query_rate, 6))
    print()
    print("TP weighted coverage =", round(tp_mean, 6))
    print("FP weighted coverage =", round(fp_mean, 6))
    print("TP/FP ratio =", None if ratio is None else round(ratio, 6))
    print("generic-only TP/FP =", round(tp_generic_rate, 6), "/", round(fp_generic_rate, 6))
    print()
    print("decision =", status)
    print("[OUTPUT]", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
