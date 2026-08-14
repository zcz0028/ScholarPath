from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.matching import match_papers

DEFAULT_GOLD = PROJECT_ROOT / "data" / "processed" / "realscholarquery_gold.jsonl"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "week2_day8_evidence_rerank" / "predictions_e3_selected.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week2_day14_constraint_evidence_audit"

_FRAGMENT_PRONE_ENDINGS = {
    "a", "an", "and", "between", "claim", "have", "inductive", "knowledgeable",
    "multiple", "of", "or", "supporting", "systematically", "the", "to", "with",
}
_DISCOURSE_PREFIXES = (
    "supporting the claim ", "show that ", "showing that ", "evidence that ",
    "papers that ", "paper that ",
)
_GENERIC_TOPIC_TOKENS = {
    "analysis", "approach", "data", "document", "learning", "method", "model",
    "network", "paper", "prediction", "result", "system", "task", "study",
    "research", "relationship", "relationships",
}
_DECISION_THRESHOLDS = {
    "max_suspicious_constraint_rate_for_score_ablation": 0.20,
    "max_queries_with_suspicious_constraints_rate": 0.30,
    "min_tp_minus_fp_weighted_coverage": 0.05,
    "min_tp_to_fp_weighted_coverage_ratio": 1.20,
    "min_fp_minus_tp_generic_entity_only_rate": 0.05,
}

@dataclass(slots=True)
class PaperAuditRow:
    qid: str
    question: str
    prediction_rank: int
    label: str
    title: str
    matched_count: int
    constraint_count: int
    raw_coverage: float
    weighted_coverage: float
    mean_matched_confidence: float
    mean_all_confidence: float
    generic_entity_only: bool
    title_only_evidence: bool
    matched_types: str
    matched_fields: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "prediction_rank": self.prediction_rank,
            "label": self.label,
            "title": self.title,
            "matched_count": self.matched_count,
            "constraint_count": self.constraint_count,
            "raw_coverage": self.raw_coverage,
            "weighted_coverage": self.weighted_coverage,
            "mean_matched_confidence": self.mean_matched_confidence,
            "mean_all_confidence": self.mean_all_confidence,
            "generic_entity_only": self.generic_entity_only,
            "title_only_evidence": self.title_only_evidence,
            "matched_types": self.matched_types,
            "matched_fields": self.matched_fields,
        }

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows

def records_by_qid(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if not qid:
            raise ValueError("Prediction row is missing qid")
        if qid in output:
            raise ValueError(f"Duplicate qid: {qid}")
        output[qid] = dict(row)
    return output

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default

def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def normalize_tokens(text: Any) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())

def suspicious_constraint_reasons(constraint: Mapping[str, Any]) -> list[str]:
    text = str(constraint.get("canonical_text") or constraint.get("text") or "").strip()
    ctype = str(constraint.get("constraint_type") or "unknown")
    tokens = normalize_tokens(text)
    reasons: list[str] = []
    if not tokens:
        return ["empty_constraint"]
    if len(tokens) == 1 and ctype == "topic":
        reasons.append("single_token_topic")
    if len(tokens) <= 3 and ctype == "topic":
        reasons.append("very_short_topic")
    if tokens[-1] in _FRAGMENT_PRONE_ENDINGS:
        reasons.append("fragment_prone_ending")
    lowered = " ".join(tokens)
    if any(lowered.startswith(prefix) for prefix in _DISCOURSE_PREFIXES):
        reasons.append("discourse_prefix")
    if len(tokens) >= 2 and all(token in _GENERIC_TOPIC_TOKENS for token in tokens):
        reasons.append("generic_topic_only")
    if "between" in tokens and tokens[-1] in {"between", "multiple"}:
        reasons.append("dangling_relation")
    return list(dict.fromkeys(reasons))

def canonical_constraints(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = record.get("day8_canonical_constraints") or []
    return [dict(item) for item in value if isinstance(item, Mapping)]

def evidence_rows(paper: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = paper.get("raw")
    if not isinstance(raw, Mapping):
        return []
    value = raw.get("day8_constraint_evidence") or []
    return [dict(item) for item in value if isinstance(item, Mapping)]

def weighted_coverage(constraints: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]]) -> float:
    weights = {
        str(item.get("id") or ""): max(0.0, safe_float(item.get("weight"), 0.0))
        for item in constraints if str(item.get("id") or "")
    }
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0
    matched_ids = {
        str(item.get("constraint_id") or "")
        for item in evidence if bool(item.get("matched"))
    }
    return safe_divide(sum(weight for cid, weight in weights.items() if cid in matched_ids), total)

def coverage_bucket(matched_count: int, constraint_count: int) -> str:
    if constraint_count <= 0:
        return "no_constraints"
    if matched_count <= 0:
        return "0/N"
    if matched_count == 1:
        return "1/N"
    if matched_count == 2:
        return "2/N"
    return "3+/N"

def summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    return {"count": len(values), "mean": fmean(values), "median": median(values)}

def summarize_boolean(values: Sequence[bool]) -> dict[str, float]:
    if not values:
        return {"count": 0, "true_count": 0, "rate": 0.0}
    true_count = sum(bool(value) for value in values)
    return {"count": len(values), "true_count": true_count, "rate": safe_divide(true_count, len(values))}

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_decision(*, suspicious_constraint_rate: float, suspicious_query_rate: float,
                   tp_weighted_coverage: float, fp_weighted_coverage: float,
                   tp_generic_only_rate: float, fp_generic_only_rate: float) -> dict[str, Any]:
    t = _DECISION_THRESHOLDS
    decomposition_unstable = (
        suspicious_constraint_rate > t["max_suspicious_constraint_rate_for_score_ablation"]
        or suspicious_query_rate > t["max_queries_with_suspicious_constraints_rate"]
    )
    coverage_delta = tp_weighted_coverage - fp_weighted_coverage
    coverage_ratio = (
        tp_weighted_coverage / fp_weighted_coverage if fp_weighted_coverage > 0
        else (float("inf") if tp_weighted_coverage > 0 else 1.0)
    )
    generic_delta = fp_generic_only_rate - tp_generic_only_rate
    evidence_discriminative = (
        coverage_delta >= t["min_tp_minus_fp_weighted_coverage"]
        and coverage_ratio >= t["min_tp_to_fp_weighted_coverage_ratio"]
        and generic_delta >= t["min_fp_minus_tp_generic_entity_only_rate"]
    )
    if decomposition_unstable:
        status = "repair_constraint_decomposition_first"
        reason = (
            "Constraint fragmentation/suspicion rate is above the audit guardrail. "
            "Do not increase evidence weight in production ranking until decomposition quality is repaired and re-audited."
        )
    elif evidence_discriminative:
        status = "proceed_to_comprehensive_score_ablation"
        reason = (
            "Strict true positives show materially stronger weighted constraint evidence than false positives, while "
            "generic-entity-only matches are more common among false positives. A protected offline comprehensive-score ablation is warranted."
        )
    else:
        status = "repair_constraint_evidence_first"
        reason = (
            "Constraint decomposition is not the dominant blocker, but current evidence does not separate strict TPs from FPs strongly enough "
            "to justify a larger production ranking weight."
        )
    return {
        "status": status,
        "reason": reason,
        "metrics": {
            "suspicious_constraint_rate": suspicious_constraint_rate,
            "suspicious_query_rate": suspicious_query_rate,
            "tp_mean_weighted_coverage": tp_weighted_coverage,
            "fp_mean_weighted_coverage": fp_weighted_coverage,
            "tp_minus_fp_weighted_coverage": coverage_delta,
            "tp_to_fp_weighted_coverage_ratio": coverage_ratio if math.isfinite(coverage_ratio) else "inf",
            "tp_generic_entity_only_rate": tp_generic_only_rate,
            "fp_generic_entity_only_rate": fp_generic_only_rate,
            "fp_minus_tp_generic_entity_only_rate": generic_delta,
        },
        "thresholds": t,
    }

def render_report(summary: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    tp = summary["paper_metrics"]["TP"]
    fp = summary["paper_metrics"]["FP"]
    cq = summary["constraint_quality"]
    return "\n".join([
        "# Day14-1B Constraint Quality + Evidence Sanity Audit",
        "",
        "## Boundary",
        "",
        "- Offline only: **true**",
        "- API calls: **0**",
        "- LLM calls: **0**",
        "- Gold usage: **strict offline audit only**",
        "- Production ranking modified: **false**",
        "- Candidate pool modified: **false**",
        "- Frontend modified: **false**",
        "",
        "## Scope",
        "",
        f"- Queries: **{summary['query_count']}**",
        f"- Top-K audited per query: **{summary['top_k']}**",
        f"- Papers audited: **{summary['paper_count']}**",
        "",
        "## Constraint quality",
        "",
        f"- Constraints: **{cq['constraint_count']}**",
        f"- Suspicious constraints: **{cq['suspicious_constraint_count']}**",
        f"- Suspicious constraint rate: **{cq['suspicious_constraint_rate']:.4f}**",
        f"- Queries with >=1 suspicious constraint: **{cq['queries_with_suspicious_constraints']}**",
        f"- Suspicious query rate: **{cq['suspicious_query_rate']:.4f}**",
        "",
        "## Strict TP vs FP evidence",
        "",
        "| Metric | TP | FP |",
        "|---|---:|---:|",
        f"| Paper count | {tp['paper_count']} | {fp['paper_count']} |",
        f"| Mean raw coverage | {tp['raw_coverage']['mean']:.4f} | {fp['raw_coverage']['mean']:.4f} |",
        f"| Mean weighted coverage | {tp['weighted_coverage']['mean']:.4f} | {fp['weighted_coverage']['mean']:.4f} |",
        f"| Mean matched confidence | {tp['matched_confidence']['mean']:.4f} | {fp['matched_confidence']['mean']:.4f} |",
        f"| Generic-entity-only rate | {tp['generic_entity_only']['rate']:.4f} | {fp['generic_entity_only']['rate']:.4f} |",
        f"| Title-only evidence rate | {tp['title_only_evidence']['rate']:.4f} | {fp['title_only_evidence']['rate']:.4f} |",
        "",
        "## Coverage distribution",
        "",
        f"- TP: `{tp['coverage_buckets']}`",
        f"- FP: `{fp['coverage_buckets']}`",
        "",
        "## Decision",
        "",
        f"**`{decision['status']}`**",
        "",
        str(decision["reason"]),
        "",
    ])

def main() -> int:
    parser = argparse.ArgumentParser(description="Day14-1B constraint quality and evidence sanity audit.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    gold_path = Path(args.gold)
    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    top_k = max(1, int(args.top_k))
    if not gold_path.is_file():
        print(f"[ERROR] Missing gold file: {gold_path}", file=sys.stderr)
        return 2
    if not prediction_path.is_file():
        print(f"[ERROR] Missing prediction file: {prediction_path}", file=sys.stderr)
        return 2

    raw_prediction_rows = read_jsonl(prediction_path)
    raw_by_qid = records_by_qid(raw_prediction_rows)
    gold_records = load_gold(gold_path)
    prediction_records = load_predictions(prediction_path)
    if set(gold_records) != set(prediction_records):
        missing = sorted(set(gold_records) - set(prediction_records))
        extra = sorted(set(prediction_records) - set(gold_records))
        raise ValueError(f"Gold/prediction qid mismatch. missing={missing} extra={extra}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_rows: list[PaperAuditRow] = []
    constraint_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    for qid in sorted(gold_records):
        gold_record = gold_records[qid]
        prediction_record = prediction_records[qid]
        raw_record = raw_by_qid[qid]
        question = str(raw_record.get("question") or "")
        raw_papers = [dict(item) for item in (raw_record.get("papers") or [])[:top_k] if isinstance(item, Mapping)]
        eval_papers = list(prediction_record.get("papers") or [])[:top_k]
        gold_papers = list(gold_record.get("papers") or [])
        if len(raw_papers) != len(eval_papers):
            raise ValueError(f"{qid}: raw/evaluation paper count mismatch ({len(raw_papers)} != {len(eval_papers)})")

        match = match_papers(predictions=eval_papers, gold=gold_papers, mode="strict")
        tp_indices = {pair.prediction_index for pair in match.pairs}
        constraints = canonical_constraints(raw_record)
        suspicious_constraints = 0

        for constraint in constraints:
            reasons = suspicious_constraint_reasons(constraint)
            suspicious = bool(reasons)
            suspicious_constraints += int(suspicious)
            constraint_rows.append({
                "qid": qid,
                "question": question,
                "constraint_id": str(constraint.get("id") or ""),
                "constraint_text": str(constraint.get("canonical_text") or constraint.get("text") or ""),
                "constraint_type": str(constraint.get("constraint_type") or ""),
                "weight": safe_float(constraint.get("weight"), 0.0),
                "token_count": len(normalize_tokens(constraint.get("canonical_text") or constraint.get("text") or "")),
                "suspicious": suspicious,
                "suspicious_reasons": "|".join(reasons),
            })

        query_paper_rows: list[PaperAuditRow] = []
        for index, paper in enumerate(raw_papers):
            evidence = evidence_rows(paper)
            matched = [item for item in evidence if bool(item.get("matched"))]
            matched_count = len(matched)
            constraint_count = len(evidence) or len(constraints)
            matched_confidences = [safe_float(item.get("confidence"), 0.0) for item in matched]
            all_confidences = [safe_float(item.get("confidence"), 0.0) for item in evidence]
            matched_types = [str(item.get("constraint_type") or "") for item in matched if str(item.get("constraint_type") or "")]
            matched_fields = [str(item.get("evidence_field") or "") for item in matched if str(item.get("evidence_field") or "")]
            row = PaperAuditRow(
                qid=qid,
                question=question,
                prediction_rank=index + 1,
                label="TP" if index in tp_indices else "FP",
                title=str(paper.get("title") or ""),
                matched_count=matched_count,
                constraint_count=constraint_count,
                raw_coverage=safe_divide(matched_count, constraint_count),
                weighted_coverage=weighted_coverage(constraints, evidence),
                mean_matched_confidence=fmean(matched_confidences) if matched_confidences else 0.0,
                mean_all_confidence=fmean(all_confidences) if all_confidences else 0.0,
                generic_entity_only=(matched_count == 1 and matched_types == ["model_or_entity"]),
                title_only_evidence=(matched_count > 0 and set(matched_fields) == {"title"}),
                matched_types="|".join(sorted(set(matched_types))),
                matched_fields="|".join(sorted(set(matched_fields))),
            )
            paper_rows.append(row)
            query_paper_rows.append(row)

        per_query_rows.append({
            "qid": qid,
            "question": question,
            "constraint_count": len(constraints),
            "suspicious_constraint_count": suspicious_constraints,
            "suspicious_constraint_rate": safe_divide(suspicious_constraints, len(constraints)),
            "topk_prediction_count": len(query_paper_rows),
            "strict_tp_at_topk": sum(row.label == "TP" for row in query_paper_rows),
            "mean_weighted_coverage_topk": fmean(row.weighted_coverage for row in query_paper_rows) if query_paper_rows else 0.0,
            "generic_entity_only_count": sum(row.generic_entity_only for row in query_paper_rows),
        })

    grouped = {
        "TP": [row for row in paper_rows if row.label == "TP"],
        "FP": [row for row in paper_rows if row.label == "FP"],
    }
    paper_metrics: dict[str, Any] = {}
    for label, rows in grouped.items():
        paper_metrics[label] = {
            "paper_count": len(rows),
            "raw_coverage": summarize_numeric([row.raw_coverage for row in rows]),
            "weighted_coverage": summarize_numeric([row.weighted_coverage for row in rows]),
            "matched_confidence": summarize_numeric([row.mean_matched_confidence for row in rows]),
            "all_confidence": summarize_numeric([row.mean_all_confidence for row in rows]),
            "generic_entity_only": summarize_boolean([row.generic_entity_only for row in rows]),
            "title_only_evidence": summarize_boolean([row.title_only_evidence for row in rows]),
            "coverage_buckets": dict(Counter(coverage_bucket(row.matched_count, row.constraint_count) for row in rows)),
        }

    constraint_count = len(constraint_rows)
    suspicious_constraint_count = sum(bool(row["suspicious"]) for row in constraint_rows)
    suspicious_query_count = sum(int(row["suspicious_constraint_count"]) > 0 for row in per_query_rows)

    summary = {
        "schema_version": "day14.constraint-evidence-audit.v1",
        "query_count": len(gold_records),
        "top_k": top_k,
        "paper_count": len(paper_rows),
        "constraint_quality": {
            "constraint_count": constraint_count,
            "suspicious_constraint_count": suspicious_constraint_count,
            "suspicious_constraint_rate": safe_divide(suspicious_constraint_count, constraint_count),
            "queries_with_suspicious_constraints": suspicious_query_count,
            "suspicious_query_rate": safe_divide(suspicious_query_count, len(gold_records)),
            "suspicious_reason_counts": dict(Counter(
                reason for row in constraint_rows
                for reason in str(row["suspicious_reasons"]).split("|") if reason
            )),
        },
        "paper_metrics": paper_metrics,
        "boundary": {
            "offline_only": True,
            "api_calls": 0,
            "llm_calls": 0,
            "gold_usage": "strict_offline_audit_only",
            "production_ranking_modified": False,
            "candidate_pool_modified": False,
            "frontend_modified": False,
        },
    }

    decision = build_decision(
        suspicious_constraint_rate=summary["constraint_quality"]["suspicious_constraint_rate"],
        suspicious_query_rate=summary["constraint_quality"]["suspicious_query_rate"],
        tp_weighted_coverage=paper_metrics["TP"]["weighted_coverage"]["mean"],
        fp_weighted_coverage=paper_metrics["FP"]["weighted_coverage"]["mean"],
        tp_generic_only_rate=paper_metrics["TP"]["generic_entity_only"]["rate"],
        fp_generic_only_rate=paper_metrics["FP"]["generic_entity_only"]["rate"],
    )

    write_json(output_dir / "audit_summary.json", summary)
    write_json(output_dir / "optimization_decision.json", decision)
    write_jsonl(output_dir / "per_query_audit.jsonl", per_query_rows)
    write_csv(output_dir / "paper_evidence_audit.csv", [row.to_dict() for row in paper_rows])
    write_csv(output_dir / "constraint_quality_audit.csv", constraint_rows)
    (output_dir / "day14_1b_report.md").write_text(render_report(summary, decision) + "\n", encoding="utf-8")

    print("[Day14-1B] Constraint Quality + Evidence Sanity Audit")
    print("[Day14-1B] API calls = 0")
    print("[Day14-1B] LLM calls = 0")
    print("[Day14-1B] production ranking modified = False")
    print()
    print("=== Constraint Quality ===")
    print("queries =", summary["query_count"])
    print("constraints =", constraint_count)
    print("suspicious_constraint_rate =", round(summary["constraint_quality"]["suspicious_constraint_rate"], 6))
    print("suspicious_query_rate =", round(summary["constraint_quality"]["suspicious_query_rate"], 6))
    print()
    print("=== Strict TP vs FP @TopK ===")
    for label in ("TP", "FP"):
        metrics = paper_metrics[label]
        print(
            f"{label}: papers={metrics['paper_count']} "
            f"weighted_cov={metrics['weighted_coverage']['mean']:.6f} "
            f"matched_conf={metrics['matched_confidence']['mean']:.6f} "
            f"generic_entity_only={metrics['generic_entity_only']['rate']:.6f}"
        )
        print(f"{label}: coverage_buckets={metrics['coverage_buckets']}")
    print()
    print("=== Decision ===")
    print("status =", decision["status"])
    print("reason =", decision["reason"])
    print("[OUTPUT]", output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
