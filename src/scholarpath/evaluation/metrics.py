from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from scholarpath.paper.schema import PaperRecord

from .matching import (
    deduplicate_predictions,
    deduplicate_predictions_v2,
    match_papers,
)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


@dataclass(slots=True)
class EvaluationConfig:
    mode: str = "strict"
    recall_at: tuple[int, ...] = (20, 50, 100)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"strict", "strict_v2", "pasa_title"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        cleaned = sorted({int(k) for k in self.recall_at if int(k) > 0})
        self.recall_at = tuple(cleaned)


@dataclass(slots=True)
class QueryEvaluation:
    qid: str
    question: str | None
    gold_count: int
    raw_prediction_count: int
    prediction_count: int
    duplicates_removed: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    recall_at: dict[str, float]
    matched_pairs: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    false_negatives: list[dict[str, Any]] = field(default_factory=list)
    removed_duplicates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "counts": {
                "gold": self.gold_count,
                "raw_predictions": self.raw_prediction_count,
                "predictions_after_dedup": self.prediction_count,
                "duplicates_removed": self.duplicates_removed,
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                **self.recall_at,
            },
            "matched_pairs": self.matched_pairs,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "removed_duplicates": self.removed_duplicates,
        }


@dataclass(slots=True)
class EvaluationResult:
    config: EvaluationConfig
    per_query: list[QueryEvaluation]
    missing_prediction_qids: list[str] = field(default_factory=list)
    extra_prediction_qids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        query_count = len(self.per_query)
        total_tp = sum(item.tp for item in self.per_query)
        total_fp = sum(item.fp for item in self.per_query)
        total_fn = sum(item.fn for item in self.per_query)

        macro_precision = fmean(item.precision for item in self.per_query) if query_count else 0.0
        macro_recall = fmean(item.recall for item in self.per_query) if query_count else 0.0
        macro_f1 = fmean(item.f1 for item in self.per_query) if query_count else 0.0
        macro_pr_harmonic = f1_from_precision_recall(
            macro_precision,
            macro_recall,
        )

        micro_precision = safe_divide(total_tp, total_tp + total_fp)
        micro_recall = safe_divide(total_tp, total_tp + total_fn)
        micro_f1 = f1_from_precision_recall(micro_precision, micro_recall)

        macro_recall_at: dict[str, float] = {}
        for k in self.config.recall_at:
            key = f"recall@{k}"
            macro_recall_at[key] = (
                fmean(item.recall_at[key] for item in self.per_query)
                if query_count
                else 0.0
            )

        return {
            "matching_mode": self.config.mode,
            "prediction_deduplication": self.config.deduplicate,
            "query_count": query_count,
            "missing_prediction_qids": self.missing_prediction_qids,
            "extra_prediction_qids": self.extra_prediction_qids,
            "counts": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "gold_papers": sum(item.gold_count for item in self.per_query),
                "raw_predictions": sum(
                    item.raw_prediction_count for item in self.per_query
                ),
                "predictions_after_dedup": sum(
                    item.prediction_count for item in self.per_query
                ),
                "duplicates_removed": sum(
                    item.duplicates_removed for item in self.per_query
                ),
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "harmonic_of_macro_precision_recall": macro_pr_harmonic,
                **macro_recall_at,
            },
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
        }


def _evaluate_single_query(
    qid: str,
    question: str | None,
    gold: list[PaperRecord],
    predictions: list[PaperRecord],
    config: EvaluationConfig,
) -> QueryEvaluation:
    raw_prediction_count = len(predictions)
    if config.deduplicate:
        deduplicator = (
            deduplicate_predictions_v2
            if config.mode == "strict_v2"
            else deduplicate_predictions
        )
        evaluated_predictions, removed_duplicates = deduplicator(predictions)
    else:
        evaluated_predictions = list(predictions)
        removed_duplicates = []

    match_result = match_papers(
        predictions=evaluated_predictions,
        gold=gold,
        mode=config.mode,
    )
    tp = match_result.tp
    fp = len(evaluated_predictions) - tp
    fn = len(gold) - tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = f1_from_precision_recall(precision, recall)

    recall_at: dict[str, float] = {}
    for k in config.recall_at:
        top_k_match = match_papers(
            predictions=evaluated_predictions[:k],
            gold=gold,
            mode=config.mode,
        )
        recall_at[f"recall@{k}"] = safe_divide(top_k_match.tp, len(gold))

    matched_pairs: list[dict[str, Any]] = []
    for pair in match_result.pairs:
        matched_pairs.append(
            {
                "prediction_rank": pair.prediction_index + 1,
                "gold_index": pair.gold_index,
                "match_type": pair.match_type,
                "prediction": evaluated_predictions[
                    pair.prediction_index
                ].to_dict(),
                "gold": gold[pair.gold_index].to_dict(),
            }
        )

    false_positives = [
        {
            "prediction_rank": index + 1,
            "paper": evaluated_predictions[index].to_dict(),
        }
        for index in match_result.unmatched_prediction_indices
    ]
    false_negatives = [
        {
            "gold_index": index,
            "paper": gold[index].to_dict(),
        }
        for index in match_result.unmatched_gold_indices
    ]

    return QueryEvaluation(
        qid=qid,
        question=question,
        gold_count=len(gold),
        raw_prediction_count=raw_prediction_count,
        prediction_count=len(evaluated_predictions),
        duplicates_removed=len(removed_duplicates),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        recall_at=recall_at,
        matched_pairs=matched_pairs,
        false_positives=false_positives,
        false_negatives=false_negatives,
        removed_duplicates=removed_duplicates,
    )


def evaluate_records(
    gold_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, Any]],
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    effective_config = config or EvaluationConfig()
    gold_qids = set(gold_records)
    prediction_qids = set(prediction_records)
    missing = sorted(gold_qids - prediction_qids)
    extra = sorted(prediction_qids - gold_qids)

    per_query: list[QueryEvaluation] = []
    for qid, gold_record in gold_records.items():
        prediction_record = prediction_records.get(qid)
        predictions = (
            prediction_record["papers"]
            if prediction_record is not None
            else []
        )
        per_query.append(
            _evaluate_single_query(
                qid=qid,
                question=gold_record.get("question"),
                gold=gold_record["papers"],
                predictions=predictions,
                config=effective_config,
            )
        )

    return EvaluationResult(
        config=effective_config,
        per_query=per_query,
        missing_prediction_qids=missing,
        extra_prediction_qids=extra,
    )


def write_evaluation_outputs(
    result: EvaluationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "summary.json"
    per_query_path = output_path / "per_query.jsonl"
    errors_path = output_path / "errors.jsonl"

    summary_path.write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with per_query_path.open("w", encoding="utf-8") as handle:
        for query_result in result.per_query:
            handle.write(
                json.dumps(query_result.to_dict(), ensure_ascii=False) + "\n"
            )

    with errors_path.open("w", encoding="utf-8") as handle:
        for query_result in result.per_query:
            if query_result.fp == 0 and query_result.fn == 0:
                continue
            handle.write(
                json.dumps(
                    {
                        "qid": query_result.qid,
                        "question": query_result.question,
                        "false_positives": query_result.false_positives,
                        "false_negatives": query_result.false_negatives,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "summary": summary_path,
        "per_query": per_query_path,
        "errors": errors_path,
    }
