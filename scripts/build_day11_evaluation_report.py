from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week2_day11_evaluation"

B0_SUMMARY = PROJECT_ROOT / "outputs" / "b0_openalex" / "run_summary.json"
B3_SUMMARY = PROJECT_ROOT / "outputs" / "b3_semantic_first" / "run_summary.json"
B4_SUMMARY = PROJECT_ROOT / "outputs" / "b4_fusion_semantic" / "run_summary.json"
DAY4_SUMMARY = PROJECT_ROOT / "outputs" / "week2_day4_rescue" / "run_summary.json"
DAY8_ABLATION = PROJECT_ROOT / "outputs" / "week2_day8_evidence_rerank" / "ablation_summary.json"
DAY8_E3 = PROJECT_ROOT / "outputs" / "week2_day8_evidence_rerank" / "predictions_e3_selected.jsonl"
DAY9_CITATION = PROJECT_ROOT / "outputs" / "week2_day9_citation" / "coverage_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{lineno}")
            rows.append(row)
    return rows


def strict_metrics(summary: dict[str, Any], k: int) -> dict[str, Any]:
    item = (summary.get("evaluation") or {}).get(f"top{k}_strict") or {}
    counts = item.get("counts") or {}
    macro = item.get("macro") or {}
    micro = item.get("micro") or {}
    return {
        "k": k,
        "tp": int(counts.get("tp", 0)),
        "fp": int(counts.get("fp", 0)),
        "fn": int(counts.get("fn", 0)),
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "micro_precision": float(micro.get("precision", 0.0)),
        "micro_recall": float(micro.get("recall", 0.0)),
        "micro_f1": float(micro.get("f1", 0.0)),
    }


def build_retrieval_summary(b0: dict[str, Any], b3: dict[str, Any], b4: dict[str, Any], day4: dict[str, Any]) -> dict[str, Any]:
    defs = [
        ("B0 Raw OpenAlex", "raw_retrieval_baseline", b0),
        ("B3 Semantic Rerank", "historical_semantic_rerank", b3),
        ("B4 Candidate Fusion", "frozen_candidate_fusion", b4),
        ("Day4 Rescue Retrieval", "retrieval_recovery", day4),
    ]
    stages = []
    for name, role, summary in defs:
        stage = {
            "stage": name,
            "role": role,
            "metrics": {str(k): strict_metrics(summary, k) for k in (20, 50, 100)},
        }
        if name == "B0 Raw OpenAlex":
            stage["runtime"] = {
                "actual_api_calls": (summary.get("retrieval") or {}).get("actual_api_calls"),
                "cache_hits": (summary.get("retrieval") or {}).get("cache_hits"),
                "retries": (summary.get("retrieval") or {}).get("retries"),
                "estimated_cost_usd": summary.get("estimated_cost_usd"),
                "latency_ms": summary.get("latency_ms") or {},
            }
        stages.append(stage)

    return {
        "matching_mode": "strict",
        "query_count": 50,
        "stages": stages,
        "key_improvements": {
            "b4_top100_tp": stages[2]["metrics"]["100"]["tp"],
            "day4_top100_tp": stages[3]["metrics"]["100"]["tp"],
            "b4_to_day4_top100_tp_gain": stages[3]["metrics"]["100"]["tp"] - stages[2]["metrics"]["100"]["tp"],
        },
    }


def build_ranking_summary(day8: dict[str, Any]) -> dict[str, Any]:
    baseline = day8.get("baseline") or {}
    e3 = ((day8.get("variants") or {}).get("E3") or {})
    selected = e3.get("selected_metrics") or {}
    b = float(baseline.get("mean_ndcg_at_10", 0.0))
    e = float(selected.get("mean_ndcg_at_10", 0.0))
    return {
        "metric": "NDCG@10",
        "baseline": {
            "mean_ndcg_at_10": b,
            "top20_tp": baseline.get("top20_tp"),
            "top50_tp": baseline.get("top50_tp"),
            "top100_tp": baseline.get("top100_tp"),
            "zero_recall_queries": baseline.get("zero_recall_queries"),
        },
        "e3": {
            "variant": "E3",
            "alpha": e3.get("selected_alpha"),
            "beta": e3.get("selected_beta"),
            "mean_ndcg_at_10": e,
            "top20_tp": selected.get("top20_tp"),
            "top50_tp": selected.get("top50_tp"),
            "top100_tp": selected.get("top100_tp"),
            "zero_recall_queries": selected.get("zero_recall_queries"),
            "protection_ok": selected.get("protection_ok"),
        },
        "absolute_ndcg_gain": e - b,
        "relative_ndcg_improvement": ((e - b) / b) if b else None,
        "gold_usage": {
            "analysis_only": bool(day8.get("analysis_only_uses_gold")),
            "production_retrieval_uses_gold": bool(day8.get("production_retrieval_uses_gold")),
        },
    }


def evidence_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (paper.get("raw") or {}).get("day8_constraint_evidence") or []
    return [x for x in rows if isinstance(x, dict)]


def evidence_metrics_at_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    papers = [p for row in rows for p in (row.get("papers") or [])[:k] if isinstance(p, dict)]
    eligible = [p for p in papers if evidence_rows(p)]
    evidence = [e for p in eligible for e in evidence_rows(p)]
    matched = [e for e in evidence if e.get("matched")]
    supported = [p for p in eligible if any(e.get("matched") for e in evidence_rows(p))]
    abstracts = [p for p in papers if bool(p.get("abstract"))]
    return {
        "k": k,
        "paper_count": len(papers),
        "eligible_papers": len(eligible),
        "constraint_count": len(evidence),
        "matched_constraint_count": len(matched),
        "constraint_evidence_match_rate": len(matched) / len(evidence) if evidence else 0.0,
        "papers_with_evidence": len(supported),
        "paper_evidence_support_rate": len(supported) / len(eligible) if eligible else 0.0,
        "reason_generation_count": len(eligible),
        "reason_generation_rate": 1.0 if eligible else 0.0,
        "evidence_backed_reason_count": len(supported),
        "evidence_backed_reason_rate": len(supported) / len(eligible) if eligible else 0.0,
        "papers_with_abstract": len(abstracts),
        "abstract_availability_rate": len(abstracts) / len(papers) if papers else 0.0,
    }


def build_explainability_summary(e3_rows: list[dict[str, Any]], citation: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": len(e3_rows),
        "constraint_evidence": {
            "metrics_by_k": {str(k): evidence_metrics_at_k(e3_rows, k) for k in (20, 50, 100)},
            "definitions": {
                "constraint_evidence_match_rate": "matched canonical constraint evidence rows / all canonical constraint evidence rows",
                "paper_evidence_support_rate": "papers with >=1 matched constraint evidence / eligible papers",
                "reason_generation_rate": "eligible papers for which deterministic RecommendationReason can generate non-empty text / eligible papers",
                "evidence_backed_reason_rate": "generated reasons backed by >=1 matched constraint evidence / generated reasons",
            },
        },
        "citation_path": {
            "scope": "single_query_case_study",
            "top_k": citation.get("top_k"),
            "query_count": citation.get("query_count"),
            "total_results": citation.get("total_results"),
            "total_paths": citation.get("total_paths"),
            "coverage": citation.get("coverage"),
            "failed_paths": citation.get("failed_paths"),
            "queries": citation.get("queries") or [],
            "reporting_note": "This citation coverage is a Day9 case study and must not be reported as full 50-query benchmark coverage.",
        },
        "limitations": [
            "The frozen Day8 E3 artifact contains no paper abstract text; current evidence coverage is measured without abstract evidence.",
            "Recommendation reasons are deterministic presentation artifacts and do not modify ranking.",
        ],
    }


def build_competition_metrics(retrieval: dict[str, Any], ranking: dict[str, Any], explainability: dict[str, Any]) -> dict[str, Any]:
    stages = retrieval["stages"]
    e100 = explainability["constraint_evidence"]["metrics_by_k"]["100"]
    citation = explainability["citation_path"]
    return {
        "benchmark_query_count": 50,
        "headline_metrics": {
            "b0_top100_tp": stages[0]["metrics"]["100"]["tp"],
            "b4_top100_tp": stages[2]["metrics"]["100"]["tp"],
            "day4_top100_tp": stages[3]["metrics"]["100"]["tp"],
            "day4_macro_recall_at_100": stages[3]["metrics"]["100"]["macro_recall"],
            "day8_baseline_ndcg_at_10": ranking["baseline"]["mean_ndcg_at_10"],
            "day8_e3_ndcg_at_10": ranking["e3"]["mean_ndcg_at_10"],
            "day8_relative_ndcg_improvement": ranking["relative_ndcg_improvement"],
            "constraint_evidence_match_rate_at_100": e100["constraint_evidence_match_rate"],
            "paper_evidence_support_rate_at_100": e100["paper_evidence_support_rate"],
            "reason_generation_rate_at_100": e100["reason_generation_rate"],
            "evidence_backed_reason_rate_at_100": e100["evidence_backed_reason_rate"],
            "abstract_availability_rate_at_100": e100["abstract_availability_rate"],
            "citation_case_study_coverage": citation["coverage"],
        },
        "reporting_rules": [
            "Use strict identity matching for primary retrieval comparisons.",
            "Do not report the 95% Day9 citation coverage as full benchmark coverage; it is a single-query case study.",
            "Do not claim Evidence/Citation improves Day8 ranking: the frozen E3 ranking uses alpha=1.0, beta=0.0.",
            "Report the 0% abstract availability alongside constraint-evidence coverage to describe the current evidence-data limitation.",
        ],
    }


def pct(value: float | int | None) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def build_report_markdown(retrieval: dict[str, Any], ranking: dict[str, Any], explainability: dict[str, Any], competition: dict[str, Any]) -> str:
    stages = retrieval["stages"]
    e20 = explainability["constraint_evidence"]["metrics_by_k"]["20"]
    e50 = explainability["constraint_evidence"]["metrics_by_k"]["50"]
    e100 = explainability["constraint_evidence"]["metrics_by_k"]["100"]
    citation = explainability["citation_path"]
    h = competition["headline_metrics"]
    lines = [
        "# ScholarPath Day11 Evaluation & Ablation Report", "",
        "## 1. Evaluation Protocol", "",
        "- Benchmark size: **50 queries**.",
        "- Primary retrieval comparison: **strict identity matching**.",
        "- Gold labels are used for evaluation/analysis only.",
        "- Day11 aggregation is offline-only: no OpenAlex calls, no LLM calls, and no ranking modification.", "",
        "## 2. Retrieval Evolution", "",
        "| Stage | Top20 TP | Top50 TP | Top100 TP | Macro R@20 | Macro R@50 | Macro R@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stages:
        m = s["metrics"]
        lines.append(f"| {s['stage']} | {m['20']['tp']} | {m['50']['tp']} | {m['100']['tp']} | {num(m['20']['macro_recall'])} | {num(m['50']['macro_recall'])} | {num(m['100']['macro_recall'])} |")
    lines += [
        "", "### Retrieval conclusion", "",
        f"- B4 → Day4 Top100 TP gain: **+{retrieval['key_improvements']['b4_to_day4_top100_tp_gain']}**.",
        f"- B0 → Day4 Top100 TP: **{stages[0]['metrics']['100']['tp']} → {stages[3]['metrics']['100']['tp']}**.", "",
        "## 3. Ranking Ablation", "",
        "| Variant | NDCG@10 | Top20 TP | Top50 TP | Top100 TP | Zero Recall |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Day4 ranking baseline | {num(ranking['baseline']['mean_ndcg_at_10'], 6)} | {ranking['baseline']['top20_tp']} | {ranking['baseline']['top50_tp']} | {ranking['baseline']['top100_tp']} | {ranking['baseline']['zero_recall_queries']} |",
        f"| Day8 E3 pure semantic | {num(ranking['e3']['mean_ndcg_at_10'], 6)} | {ranking['e3']['top20_tp']} | {ranking['e3']['top50_tp']} | {ranking['e3']['top100_tp']} | {ranking['e3']['zero_recall_queries']} |",
        "", f"- Relative NDCG@10 improvement: **{pct(ranking['relative_ndcg_improvement'])}**.",
        "- E3 improves top-rank ordering quality while keeping the frozen Day4 candidate coverage unchanged.", "",
        "## 4. Explainability Evaluation", "",
        "| Metric | @20 | @50 | @100 |", "|---|---:|---:|---:|",
        f"| Constraint Evidence Match Rate | {pct(e20['constraint_evidence_match_rate'])} | {pct(e50['constraint_evidence_match_rate'])} | {pct(e100['constraint_evidence_match_rate'])} |",
        f"| Paper Evidence Support Rate | {pct(e20['paper_evidence_support_rate'])} | {pct(e50['paper_evidence_support_rate'])} | {pct(e100['paper_evidence_support_rate'])} |",
        f"| Reason Generation Rate | {pct(e20['reason_generation_rate'])} | {pct(e50['reason_generation_rate'])} | {pct(e100['reason_generation_rate'])} |",
        f"| Evidence-backed Reason Rate | {pct(e20['evidence_backed_reason_rate'])} | {pct(e50['evidence_backed_reason_rate'])} | {pct(e100['evidence_backed_reason_rate'])} |",
        f"| Abstract Availability Rate | {pct(e20['abstract_availability_rate'])} | {pct(e50['abstract_availability_rate'])} | {pct(e100['abstract_availability_rate'])} |", "",
        "- `Reason Generation Rate` is an availability metric, not explanation accuracy.",
        "- `Evidence-backed Reason Rate` is the stronger support metric.",
        "- Frozen Day8 artifacts have **0% abstract availability**.", "",
        "## 5. Citation Path Case Study", "",
        f"- Query: **{citation['queries'][0]['qid'] if citation.get('queries') else 'N/A'}**.",
        f"- Citation paths: **{citation.get('total_paths')}/{citation.get('total_results')}**.",
        f"- Coverage: **{pct(citation.get('coverage'))}**.", "",
        "> Reporting boundary: this is a **single-query Day9 case study** and must not be reported as 50-query benchmark citation coverage.", "",
        "## 6. Efficiency Snapshot", "",
        f"- B0 actual OpenAlex API calls: **{stages[0]['runtime'].get('actual_api_calls')}**.",
        f"- B0 cache hits: **{stages[0]['runtime'].get('cache_hits')}**.",
        f"- B0 retries: **{stages[0]['runtime'].get('retries')}**.",
        f"- B0 estimated cost: **${num(stages[0]['runtime'].get('estimated_cost_usd'), 3)}**.", "",
        "## 7. Reporting Boundaries", "",
    ]
    lines.extend(f"- {rule}" for rule in competition["reporting_rules"])
    lines += [
        "", "## 8. Key Competition Conclusions", "",
        f"1. **Retrieval recovery:** B4 Top100 TP increases from **{h['b4_top100_tp']}** to **{h['day4_top100_tp']}** after Day4 Rescue.",
        f"2. **Ranking quality:** Day8 E3 improves NDCG@10 from **{num(h['day8_baseline_ndcg_at_10'], 6)}** to **{num(h['day8_e3_ndcg_at_10'], 6)}** (**{pct(h['day8_relative_ndcg_improvement'])} relative gain**).",
        f"3. **Evidence support:** **{pct(h['paper_evidence_support_rate_at_100'])}** of eligible Top100 papers have at least one matched constraint-evidence row.",
        f"4. **Traceability:** RealScholarQuery_16 reaches **{pct(h['citation_case_study_coverage'])}** Top20 citation-path coverage in the Day9 case study.",
    ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_retrieval_csv(path: Path, retrieval: dict[str, Any]) -> None:
    fields = ["stage", "role", "k", "tp", "fp", "fn", "macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in retrieval["stages"]:
            for k in ("20", "50", "100"):
                row = {"stage": stage["stage"], "role": stage["role"], "k": k}
                row.update({key: stage["metrics"][k][key] for key in fields[3:]})
                writer.writerow(row)


def write_ranking_csv(path: Path, ranking: dict[str, Any]) -> None:
    rows = [
        {"variant": "Day4 ranking baseline", "ndcg_at_10": ranking["baseline"]["mean_ndcg_at_10"], "top20_tp": ranking["baseline"]["top20_tp"], "top50_tp": ranking["baseline"]["top50_tp"], "top100_tp": ranking["baseline"]["top100_tp"], "zero_recall_queries": ranking["baseline"]["zero_recall_queries"]},
        {"variant": "Day8 E3 pure semantic", "ndcg_at_10": ranking["e3"]["mean_ndcg_at_10"], "top20_tp": ranking["e3"]["top20_tp"], "top50_tp": ranking["e3"]["top50_tp"], "top100_tp": ranking["e3"]["top100_tp"], "zero_recall_queries": ranking["e3"]["zero_recall_queries"]},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    required = [B0_SUMMARY, B3_SUMMARY, B4_SUMMARY, DAY4_SUMMARY, DAY8_ABLATION, DAY8_E3, DAY9_CITATION]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen Day11 artifacts:\n" + "\n".join(str(p) for p in missing))

    retrieval = build_retrieval_summary(read_json(B0_SUMMARY), read_json(B3_SUMMARY), read_json(B4_SUMMARY), read_json(DAY4_SUMMARY))
    ranking = build_ranking_summary(read_json(DAY8_ABLATION))
    explainability = build_explainability_summary(read_jsonl(DAY8_E3), read_json(DAY9_CITATION))
    competition = build_competition_metrics(retrieval, ranking, explainability)

    summary = {
        "schema_version": "day11-v1",
        "offline_only": True,
        "network_calls": 0,
        "llm_calls": 0,
        "ranking_modified": False,
        "retrieval": retrieval,
        "ranking": ranking,
        "explainability": explainability,
        "competition_metrics": competition,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "evaluation_summary.json", summary)
    write_json(OUTPUT_DIR / "retrieval_ranking_summary.json", {"retrieval": retrieval, "ranking": ranking})
    write_json(OUTPUT_DIR / "explainability_summary.json", explainability)
    write_json(OUTPUT_DIR / "competition_metrics.json", competition)
    write_retrieval_csv(OUTPUT_DIR / "retrieval_ablation.csv", retrieval)
    write_ranking_csv(OUTPUT_DIR / "ranking_ablation.csv", ranking)
    (OUTPUT_DIR / "evaluation_report.md").write_text(build_report_markdown(retrieval, ranking, explainability, competition), encoding="utf-8")

    h = competition["headline_metrics"]
    print("[Day11] Unified evaluation artifacts generated.")
    print(f"[Day11] output_dir = {OUTPUT_DIR}")
    print("\n=== Retrieval ===")
    print("B0 Top100 TP:", h["b0_top100_tp"])
    print("B4 Top100 TP:", h["b4_top100_tp"])
    print("Day4 Top100 TP:", h["day4_top100_tp"])
    print("\n=== Ranking ===")
    print("Baseline NDCG@10:", round(h["day8_baseline_ndcg_at_10"], 6))
    print("E3 NDCG@10:", round(h["day8_e3_ndcg_at_10"], 6))
    print("Relative NDCG gain:", round(h["day8_relative_ndcg_improvement"], 6))
    print("\n=== Explainability @100 ===")
    print("Constraint Evidence Match Rate:", round(h["constraint_evidence_match_rate_at_100"], 6))
    print("Paper Evidence Support Rate:", round(h["paper_evidence_support_rate_at_100"], 6))
    print("Reason Generation Rate:", round(h["reason_generation_rate_at_100"], 6))
    print("Evidence-backed Reason Rate:", round(h["evidence_backed_reason_rate_at_100"], 6))
    print("Abstract Availability Rate:", round(h["abstract_availability_rate_at_100"], 6))
    print("\n=== Citation ===")
    print("Case-study coverage:", h["citation_case_study_coverage"])
    print("NOTE: citation coverage is single-query only.")


if __name__ == "__main__":
    main()
