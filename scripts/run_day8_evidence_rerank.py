from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholarpath.query.constraints import ConstraintDecomposer
from scholarpath.rerank.evidence_aware_rerank import (
    DEFAULT_ALPHAS,
    EvidenceAwareConfig,
    annotate_and_evidence_rerank_record,
    ndcg_at_k,
    select_alpha_from_cv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day8-1C evidence-aware rerank + NDCG@10 ablation."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/week2_day8_evidence_rerank"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tie-tolerance", type=float, default=0.005)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qid_of(record: Mapping[str, Any]) -> str:
    return str(record.get("qid") or record.get("query_id") or "").strip()


def question_of(record: Mapping[str, Any]) -> str:
    return str(record.get("question") or record.get("query") or "").strip()


def papers_of(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("papers", "gold_papers", "results", "references"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def normalize_title(value: object | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: object | None) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.rstrip(" .")


def paper_keys(paper: Mapping[str, Any]) -> set[str]:
    raw = paper.get("raw") if isinstance(paper.get("raw"), Mapping) else {}
    keys: set[str] = set()
    for source in (paper, raw):
        canonical = str(source.get("canonical_id") or source.get("canonical_id_v2") or "").strip().casefold()
        if canonical:
            keys.add("canonical:" + canonical)
        doi = normalize_doi(source.get("doi"))
        if doi:
            keys.add("doi:" + doi)
        for field, prefix in (
            ("openalex_id", "openalex:"),
            ("semantic_scholar_id", "s2:"),
            ("arxiv_id", "arxiv:"),
        ):
            value = str(source.get(field) or "").strip().casefold()
            if value:
                keys.add(prefix + value)
    title = normalize_title(paper.get("normalized_title") or paper.get("title"))
    if title:
        keys.add("title:" + title)
    return keys


def relevance_vector(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> tuple[list[int], int]:
    gold_keys = [paper_keys(item) for item in gold]
    matched_gold: set[int] = set()
    relevance: list[int] = []
    for paper in predictions:
        keys = paper_keys(paper)
        match_index = next(
            (
                index
                for index, target in enumerate(gold_keys)
                if index not in matched_gold and keys and target and keys.intersection(target)
            ),
            None,
        )
        if match_index is None:
            relevance.append(0)
        else:
            relevance.append(1)
            matched_gold.add(match_index)
    return relevance, len(matched_gold)


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    gold_by_qid: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    per_query: dict[str, Any] = {}
    total_tp_20 = total_tp_50 = total_tp_100 = 0
    zero_recall = 0
    queries_with_gold = 0
    ndcgs: list[float] = []

    for record in records:
        qid = qid_of(record)
        gold = list(gold_by_qid.get(qid, ()))
        predictions = papers_of(record)
        relevance, _ = relevance_vector(predictions, gold)
        gold_count = len(gold)
        if gold_count:
            queries_with_gold += 1
        tp20 = sum(relevance[:20])
        tp50 = sum(relevance[:50])
        tp100 = sum(relevance[:100])
        if gold_count and tp100 == 0:
            zero_recall += 1
        ndcg10 = ndcg_at_k(relevance, k=10) if gold_count else 0.0
        if gold_count:
            ndcgs.append(ndcg10)
        total_tp_20 += tp20
        total_tp_50 += tp50
        total_tp_100 += tp100
        per_query[qid] = {
            "gold_count": gold_count,
            "tp20": tp20,
            "tp50": tp50,
            "tp100": tp100,
            "ndcg_at_10": ndcg10,
        }

    return {
        "query_count": len(records),
        "queries_with_gold": queries_with_gold,
        "queries_without_gold": len(records) - queries_with_gold,
        "mean_ndcg_at_10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "top20_tp": total_tp_20,
        "top50_tp": total_tp_50,
        "top100_tp": total_tp_100,
        "zero_recall_queries": zero_recall,
        "per_query": per_query,
        "matching_note": (
            "Binary relevance; deterministic exact identity via canonical ID/DOI/"
            "OpenAlex/S2/arXiv or exact normalized title. Gold is evaluation-only."
        ),
    }


def protection_ok(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return (
        int(metrics["top20_tp"]) >= int(baseline["top20_tp"])
        and int(metrics["top100_tp"]) >= int(baseline["top100_tp"])
        and int(metrics["zero_recall_queries"]) <= int(baseline["zero_recall_queries"])
    )


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")

    predictions = load_jsonl(args.predictions)
    gold_records = load_jsonl(args.gold)
    gold_by_qid = {qid_of(row): papers_of(row) for row in gold_records if qid_of(row)}
    decomposer = ConstraintDecomposer()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments: dict[str, dict[float, dict[str, Any]]] = {}
    reranked_cache: dict[tuple[str, float], list[dict[str, Any]]] = {}

    # E0 is the untouched current B3 order/score baseline.
    e0_records: list[dict[str, Any]] = []
    for record in predictions:
        question = question_of(record)
        decomposition = decomposer.decompose(question)
        e0_records.append(
            annotate_and_evidence_rerank_record(
                dict(record), decomposition, EvidenceAwareConfig("E0", 1.0)
            )
        )
    baseline = evaluate_records(e0_records, gold_by_qid)
    experiments["E0"] = {1.0: baseline}
    reranked_cache[("E0", 1.0)] = e0_records

    for variant in ("E1", "E2", "E3"):
        experiments[variant] = {}
        for alpha in DEFAULT_ALPHAS:
            rows: list[dict[str, Any]] = []
            for record in predictions:
                question = question_of(record)
                decomposition = decomposer.decompose(question)
                rows.append(
                    annotate_and_evidence_rerank_record(
                        dict(record),
                        decomposition,
                        EvidenceAwareConfig(variant=variant, alpha=alpha),
                    )
                )
            metrics = evaluate_records(rows, gold_by_qid)
            metrics["protection_ok"] = protection_ok(metrics, baseline)
            experiments[variant][alpha] = metrics
            reranked_cache[(variant, alpha)] = rows

    summary: dict[str, Any] = {
        "analysis_only_uses_gold": True,
        "production_retrieval_uses_gold": False,
        "predictions": str(args.predictions),
        "gold": str(args.gold),
        "baseline": {key: value for key, value in baseline.items() if key != "per_query"},
        "variants": {},
    }

    qids = [qid_of(row) for row in predictions if qid_of(row) in gold_by_qid and gold_by_qid[qid_of(row)]]

    for variant in ("E1", "E2", "E3"):
        eligible_alphas = [
            alpha for alpha, metrics in experiments[variant].items()
            if metrics.get("protection_ok")
        ]
        if not eligible_alphas:
            eligible_alphas = [1.0]

        per_query_ndcg = {
            alpha: {
                qid: float(values["ndcg_at_10"])
                for qid, values in experiments[variant][alpha]["per_query"].items()
                if qid in qids
            }
            for alpha in eligible_alphas
        }
        cv = select_alpha_from_cv(
            per_query_ndcg,
            qids,
            folds=args.folds,
            tie_tolerance=args.tie_tolerance,
        )
        selected_alpha = float(cv["selected_alpha"])
        selected_metrics = experiments[variant][selected_alpha]
        summary["variants"][variant] = {
            "selected_alpha": selected_alpha,
            "selected_beta": 1.0 - selected_alpha,
            "cv": cv,
            "selected_metrics": {
                key: value for key, value in selected_metrics.items() if key != "per_query"
            },
            "grid": [
                {
                    "alpha": alpha,
                    "beta": 1.0 - alpha,
                    **{key: value for key, value in metrics.items() if key not in {"per_query", "matching_note"}},
                }
                for alpha, metrics in sorted(experiments[variant].items(), reverse=True)
            ],
        }
        write_jsonl(
            output_dir / f"predictions_{variant.lower()}_selected.jsonl",
            reranked_cache[(variant, selected_alpha)],
        )

    # Recommend the best protected variant by selected held-out NDCG. E0 remains
    # available as the safe fallback if no variant improves it.
    candidates = []
    for variant, data in summary["variants"].items():
        metrics = data["selected_metrics"]
        if metrics.get("protection_ok"):
            candidates.append(
                (
                    float(metrics["mean_ndcg_at_10"]),
                    float(data["selected_alpha"]),
                    variant,
                )
            )
    if candidates:
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best_ndcg, best_alpha, best_variant = candidates[0]
        if best_ndcg > float(baseline["mean_ndcg_at_10"]):
            summary["recommended"] = {
                "variant": best_variant,
                "alpha": best_alpha,
                "beta": 1.0 - best_alpha,
                "mean_ndcg_at_10": best_ndcg,
            }
        else:
            summary["recommended"] = {
                "variant": "E0",
                "alpha": 1.0,
                "beta": 0.0,
                "mean_ndcg_at_10": baseline["mean_ndcg_at_10"],
                "reason": "No protected Day8 variant exceeded the B3 baseline NDCG@10.",
            }

    write_json(output_dir / "ablation_summary.json", summary)
    write_jsonl(output_dir / "predictions_e0_baseline.jsonl", e0_records)

    print("[Day8-1C] completed")
    print(f"[E0] NDCG@10={baseline['mean_ndcg_at_10']:.6f} "
          f"TP20={baseline['top20_tp']} TP100={baseline['top100_tp']} "
          f"zero={baseline['zero_recall_queries']}")
    for variant, data in summary["variants"].items():
        metrics = data["selected_metrics"]
        print(
            f"[{variant}] alpha={data['selected_alpha']:.1f} beta={data['selected_beta']:.1f} "
            f"NDCG@10={metrics['mean_ndcg_at_10']:.6f} "
            f"TP20={metrics['top20_tp']} TP100={metrics['top100_tp']} "
            f"zero={metrics['zero_recall_queries']} protected={metrics['protection_ok']}"
        )
    print(f"[OUTPUT] {output_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
