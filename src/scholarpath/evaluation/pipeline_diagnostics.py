from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scholarpath.evaluation.io import load_gold, load_predictions
from scholarpath.evaluation.matching import match_papers
from scholarpath.paper.normalizers import first_author_key, normalize_title
from scholarpath.paper.schema import PaperRecord
from scholarpath.query.anchor_extractor import AnchorExtractionResult, extract_anchors


@dataclass(slots=True)
class StageCoverage:
    stage_name: str
    candidate_count: int
    matched_gold_count: int
    matched_gold_ids: list[str] = field(default_factory=list)
    matched_gold_titles: list[str] = field(default_factory=list)
    matched_ranks: list[int] = field(default_factory=list)
    best_matched_rank: int | None = None
    match_types: list[str] = field(default_factory=list)

    @property
    def has_hit(self) -> bool:
        return self.matched_gold_count > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PossibleMatchingIssue:
    qid: str
    stage_name: str
    prediction_rank: int
    prediction_title: str
    gold_title: str
    title_similarity: float
    token_jaccard: float
    first_author_equal: bool
    prediction_year: int | None
    gold_year: int | None
    year_distance: int | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryDiagnosis:
    qid: str
    question: str
    gold_count: int
    target_stage: str
    stages: dict[str, StageCoverage]
    first_hit_stage: str | None
    last_hit_stage: str | None
    failure_type: str
    failure_stage: str | None
    priority: str
    suggested_actions: list[str]
    anchor_result: AnchorExtractionResult
    possible_matching_issue_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "gold_count": self.gold_count,
            "target_stage": self.target_stage,
            "stages": {name: value.to_dict() for name, value in self.stages.items()},
            "first_hit_stage": self.first_hit_stage,
            "last_hit_stage": self.last_hit_stage,
            "failure_type": self.failure_type,
            "failure_stage": self.failure_stage,
            "priority": self.priority,
            "suggested_actions": self.suggested_actions,
            "anchors": self.anchor_result.to_dict(),
            "possible_matching_issue_count": self.possible_matching_issue_count,
            "analysis_only_uses_gold": True,
            "production_retrieval_uses_gold": False,
        }


@dataclass(slots=True)
class DiagnosticSummary:
    query_count: int
    target_stage: str
    target_zero_recall_count: int
    target_query_hit_count: int
    target_total_tp: int
    failure_type_counts: dict[str, int]
    priority_counts: dict[str, int]
    stage_query_hit_counts: dict[str, int]
    stage_total_tp: dict[str, int]
    stage_candidate_counts: dict[str, int]
    possible_matching_issue_count: int
    matching_mode: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["analysis_only_uses_gold"] = True
        payload["production_retrieval_uses_gold"] = False
        payload["api_calls"] = 0
        return payload


DEFAULT_RETRIEVAL_STAGES = ("b0_top100", "b1_2_top100", "b2_top100")


def _coverage_for_stage(
    stage_name: str,
    predictions: Sequence[PaperRecord],
    gold: Sequence[PaperRecord],
    *,
    mode: str,
) -> StageCoverage:
    result = match_papers(list(predictions), list(gold), mode=mode)
    ranks = [pair.prediction_index + 1 for pair in result.pairs]
    matched_gold = [gold[pair.gold_index] for pair in result.pairs]
    return StageCoverage(
        stage_name=stage_name,
        candidate_count=len(predictions),
        matched_gold_count=result.tp,
        matched_gold_ids=[paper.canonical_id for paper in matched_gold],
        matched_gold_titles=[paper.title for paper in matched_gold],
        matched_ranks=ranks,
        best_matched_rank=min(ranks) if ranks else None,
        match_types=[pair.match_type for pair in result.pairs],
    )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def find_possible_matching_issues(
    qid: str,
    stage_name: str,
    predictions: Sequence[PaperRecord],
    gold: Sequence[PaperRecord],
    *,
    max_predictions: int = 200,
    min_similarity: float = 0.90,
) -> list[PossibleMatchingIssue]:
    """Find audit-only title/author/year candidates.

    These candidates never count as true positives. They are written to a
    separate audit file for Day 2 identifier and version-resolution work.
    """
    issues: list[PossibleMatchingIssue] = []
    exact_result = match_papers(list(predictions), list(gold), mode="strict")
    matched_prediction_indices = {pair.prediction_index for pair in exact_result.pairs}
    matched_gold_indices = {pair.gold_index for pair in exact_result.pairs}

    for prediction_index, prediction in enumerate(predictions[:max_predictions]):
        if prediction_index in matched_prediction_indices:
            continue
        prediction_title = normalize_title(prediction.title)
        if not prediction_title:
            continue

        best: PossibleMatchingIssue | None = None
        for gold_index, gold_paper in enumerate(gold):
            if gold_index in matched_gold_indices:
                continue
            gold_title = normalize_title(gold_paper.title)
            if not gold_title:
                continue

            similarity = SequenceMatcher(None, prediction_title, gold_title).ratio()
            jaccard = _token_jaccard(prediction.title, gold_paper.title)
            prediction_author = first_author_key(prediction.authors)
            gold_author = first_author_key(gold_paper.authors)
            author_equal = bool(prediction_author and gold_author and prediction_author == gold_author)
            year_distance = (
                abs(prediction.year - gold_paper.year)
                if prediction.year is not None and gold_paper.year is not None
                else None
            )
            year_close = year_distance is None or year_distance <= 1

            qualifies = (
                similarity >= 0.97
                or jaccard >= 0.94
                or (similarity >= min_similarity and author_equal and year_close)
            )
            if not qualifies:
                continue

            reason_parts = ["high_title_similarity"]
            if author_equal:
                reason_parts.append("same_first_author")
            if year_distance is not None and year_distance <= 1:
                reason_parts.append("near_year")

            candidate = PossibleMatchingIssue(
                qid=qid,
                stage_name=stage_name,
                prediction_rank=prediction_index + 1,
                prediction_title=prediction.title,
                gold_title=gold_paper.title,
                title_similarity=round(similarity, 4),
                token_jaccard=round(jaccard, 4),
                first_author_equal=author_equal,
                prediction_year=prediction.year,
                gold_year=gold_paper.year,
                year_distance=year_distance,
                reason="+".join(reason_parts),
            )
            if best is None or candidate.title_similarity > best.title_similarity:
                best = candidate

        if best is not None:
            issues.append(best)

    issues.sort(key=lambda item: (-item.title_similarity, item.prediction_rank))
    return issues


def _anchor_type_set(anchor_result: AnchorExtractionResult) -> set[str]:
    return {anchor.anchor_type for anchor in anchor_result.anchors}


def classify_failure(
    stages: Mapping[str, StageCoverage],
    *,
    target_stage: str,
    retrieval_stages: Sequence[str] = DEFAULT_RETRIEVAL_STAGES,
    possible_matching_issue_count: int = 0,
) -> tuple[str, str | None]:
    target = stages[target_stage]
    if target.has_hit:
        return "success", None

    if possible_matching_issue_count > 0:
        return "possible_matching_issue", target_stage

    pool = stages.get("b4_pool")
    if pool is not None and pool.has_hit:
        if target_stage == "b4_top20":
            return "ranking_loss_top20", target_stage
        if target_stage == "b4_top50":
            return "ranking_loss_top50", target_stage
        return "ranking_loss_top100", target_stage

    retrieval_hit = any(stages.get(name) and stages[name].has_hit for name in retrieval_stages)
    if retrieval_hit and pool is not None and not pool.has_hit:
        return "fusion_pool_miss", "b4_pool"

    # Diagnose post-B4 filtering when a later target stage is selected.
    b4_top100 = stages.get("b4_top100")
    if target_stage.startswith("b5_1") and b4_top100 and b4_top100.has_hit:
        return "guard_drop", target_stage
    if target_stage == "b5_precision" and b4_top100 and b4_top100.has_hit:
        return "selector_drop", target_stage
    if target_stage.startswith("b5_2") and b4_top100 and b4_top100.has_hit:
        return "guard_aware_drop", target_stage

    return "retrieval_miss", retrieval_stages[-1] if retrieval_stages else None


def recommend_actions(
    failure_type: str,
    anchor_result: AnchorExtractionResult,
) -> list[str]:
    types = _anchor_type_set(anchor_result)
    actions: list[str] = []

    if failure_type.startswith("ranking_loss"):
        actions.extend(
            [
                "inspect B4 semantic reranking features and score components",
                "increase exact title/entity anchor weight without adding API calls",
                "compare the gold candidate rank before and after semantic reranking",
            ]
        )
    elif failure_type == "fusion_pool_miss":
        actions.extend(
            [
                "inspect candidate identity normalization during B4 fusion",
                "check source caps and fusion truncation",
                "verify that a correct upstream candidate was not removed as a duplicate",
            ]
        )
    elif failure_type == "possible_matching_issue":
        actions.extend(
            [
                "audit DOI/arXiv/OpenAlex identifier normalization",
                "check preprint versus published-version alignment",
                "do not count the audit candidate as TP until Day 2 matching review",
            ]
        )
    elif failure_type in {"selector_drop", "guard_drop", "guard_aware_drop"}:
        actions.extend(
            [
                "compare the paper before and after filtering",
                "inspect relevance labels and hard-constraint evidence",
                "add a regression case before changing selector or guard thresholds",
            ]
        )
    else:
        if types & {"benchmark", "method", "model", "paper_title", "named_entity"}:
            actions.append("build anchor-aware rescue queries from explicit academic entities")
        if "comparison" in types:
            actions.append("convert comparison relations into paired benchmark or method queries")
        if "negative_constraint" in types:
            actions.append("apply negative constraints after recall rather than inside broad retrieval")
        actions.extend(
            [
                "generate at most two targeted rescue queries for this query",
                "consider controlled citation expansion only from a high-confidence seed",
            ]
        )

    # Stable de-duplication while preserving order.
    return list(dict.fromkeys(actions))


def assign_priority(
    failure_type: str,
    anchor_result: AnchorExtractionResult,
) -> str:
    types = _anchor_type_set(anchor_result)
    if failure_type.startswith("ranking_loss"):
        return "P0"
    if failure_type == "retrieval_miss" and types & {
        "benchmark",
        "method",
        "model",
        "paper_title",
        "named_entity",
    }:
        return "P1"
    if failure_type in {"fusion_pool_miss", "possible_matching_issue", "guard_drop", "selector_drop", "guard_aware_drop"}:
        return "P2"
    return "P3"


def validate_qids(
    gold_records: Mapping[str, Mapping[str, Any]],
    stage_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_query_count: int | None = 50,
) -> None:
    gold_qids = set(gold_records)
    if expected_query_count is not None and len(gold_qids) != expected_query_count:
        raise ValueError(
            f"Gold contains {len(gold_qids)} qids; expected {expected_query_count}."
        )

    for stage_name, records in stage_records.items():
        stage_qids = set(records)
        missing = sorted(gold_qids - stage_qids)
        extra = sorted(stage_qids - gold_qids)
        if missing or extra:
            raise ValueError(
                f"Stage '{stage_name}' qids do not match gold. "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        if expected_query_count is not None and len(stage_qids) != expected_query_count:
            raise ValueError(
                f"Stage '{stage_name}' contains {len(stage_qids)} qids; "
                f"expected {expected_query_count}."
            )


def diagnose_pipeline(
    gold_records: Mapping[str, Mapping[str, Any]],
    stage_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target_stage: str,
    mode: str = "strict",
    expected_query_count: int | None = 50,
) -> tuple[list[QueryDiagnosis], DiagnosticSummary, list[PossibleMatchingIssue]]:
    if target_stage not in stage_records:
        raise ValueError(f"Unknown target stage: {target_stage}")
    validate_qids(gold_records, stage_records, expected_query_count=expected_query_count)

    ordered_stage_names = list(stage_records)
    diagnoses: list[QueryDiagnosis] = []
    all_matching_issues: list[PossibleMatchingIssue] = []

    for qid, gold_record in gold_records.items():
        question = str(gold_record.get("question") or "")
        gold_papers = list(gold_record["papers"])
        stage_coverages: dict[str, StageCoverage] = {}

        for stage_name in ordered_stage_names:
            predictions = list(stage_records[stage_name][qid]["papers"])
            stage_coverages[stage_name] = _coverage_for_stage(
                stage_name,
                predictions,
                gold_papers,
                mode=mode,
            )

        target_predictions = list(stage_records[target_stage][qid]["papers"])
        matching_issues = find_possible_matching_issues(
            qid,
            target_stage,
            target_predictions,
            gold_papers,
        )
        all_matching_issues.extend(matching_issues)

        anchor_result = extract_anchors(question)
        hit_stages = [name for name in ordered_stage_names if stage_coverages[name].has_hit]
        failure_type, failure_stage = classify_failure(
            stage_coverages,
            target_stage=target_stage,
            possible_matching_issue_count=len(matching_issues),
        )
        diagnoses.append(
            QueryDiagnosis(
                qid=qid,
                question=question,
                gold_count=len(gold_papers),
                target_stage=target_stage,
                stages=stage_coverages,
                first_hit_stage=hit_stages[0] if hit_stages else None,
                last_hit_stage=hit_stages[-1] if hit_stages else None,
                failure_type=failure_type,
                failure_stage=failure_stage,
                priority=assign_priority(failure_type, anchor_result),
                suggested_actions=recommend_actions(failure_type, anchor_result),
                anchor_result=anchor_result,
                possible_matching_issue_count=len(matching_issues),
            )
        )

    target_zero_recall_count = sum(
        not diagnosis.stages[target_stage].has_hit for diagnosis in diagnoses
    )
    summary = DiagnosticSummary(
        query_count=len(diagnoses),
        target_stage=target_stage,
        target_zero_recall_count=target_zero_recall_count,
        target_query_hit_count=len(diagnoses) - target_zero_recall_count,
        target_total_tp=sum(
            diagnosis.stages[target_stage].matched_gold_count for diagnosis in diagnoses
        ),
        failure_type_counts=dict(Counter(item.failure_type for item in diagnoses)),
        priority_counts=dict(Counter(item.priority for item in diagnoses)),
        stage_query_hit_counts={
            stage_name: sum(item.stages[stage_name].has_hit for item in diagnoses)
            for stage_name in ordered_stage_names
        },
        stage_total_tp={
            stage_name: sum(item.stages[stage_name].matched_gold_count for item in diagnoses)
            for stage_name in ordered_stage_names
        },
        stage_candidate_counts={
            stage_name: sum(item.stages[stage_name].candidate_count for item in diagnoses)
            for stage_name in ordered_stage_names
        },
        possible_matching_issue_count=len(all_matching_issues),
        matching_mode=mode,
    )
    return diagnoses, summary, all_matching_issues


def load_pipeline_inputs(
    gold_path: str | Path,
    stage_paths: Mapping[str, str | Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    gold_records = load_gold(gold_path)
    stage_records = {
        stage_name: load_predictions(path) for stage_name, path in stage_paths.items()
    }
    return gold_records, stage_records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def _write_stage_matrix_csv(
    path: Path,
    diagnoses: Sequence[QueryDiagnosis],
    stage_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["qid", "gold_count", *stage_names, "failure_type", "priority"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for diagnosis in diagnoses:
            row: dict[str, Any] = {
                "qid": diagnosis.qid,
                "gold_count": diagnosis.gold_count,
                "failure_type": diagnosis.failure_type,
                "priority": diagnosis.priority,
            }
            for stage_name in stage_names:
                row[stage_name] = diagnosis.stages[stage_name].matched_gold_count
            writer.writerow(row)


def _write_stage_matrix_md(
    path: Path,
    diagnoses: Sequence[QueryDiagnosis],
    stage_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["qid", "gold", *stage_names, "failure", "priority"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for diagnosis in diagnoses:
        values = [
            diagnosis.qid,
            str(diagnosis.gold_count),
            *[
                str(diagnosis.stages[stage_name].matched_gold_count)
                for stage_name in stage_names
            ],
            diagnosis.failure_type,
            diagnosis.priority,
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_priority_markdown(path: Path, diagnoses: Sequence[QueryDiagnosis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptions = {
        "P0": "候选池中已有正确论文，但目标 Top-K 排序丢失。优先修排序，不增加 API。",
        "P1": "完全召回失败，但查询含明确数据集、方法、模型、标题或命名实体锚点。",
        "P2": "融合、匹配、Selector 或 Guard 过程可能丢失正确论文。",
        "P3": "语义较模糊的召回失败，需要任务别名或受控引文扩展。",
    }
    lines = ["# Week 2 Day 1：优先处理查询", ""]
    for priority in ("P0", "P1", "P2", "P3"):
        items = [item for item in diagnoses if item.priority == priority and item.failure_type != "success"]
        lines.extend([f"## {priority}", "", descriptions[priority], ""])
        if not items:
            lines.extend(["无。", ""])
            continue
        for item in items:
            anchor_text = ", ".join(anchor.text for anchor in item.anchor_result.anchors[:8]) or "无明确锚点"
            lines.extend(
                [
                    f"### {item.qid}",
                    "",
                    f"- 失败类型：`{item.failure_type}`",
                    f"- 问题：{item.question}",
                    f"- 锚点：{anchor_text}",
                    "- 建议：" + "；".join(item.suggested_actions),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_diagnostic_outputs(
    output_dir: str | Path,
    diagnoses: Sequence[QueryDiagnosis],
    summary: DiagnosticSummary,
    possible_matching_issues: Sequence[PossibleMatchingIssue],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stage_names = list(diagnoses[0].stages) if diagnoses else []

    _write_json(output_path / "diagnostic_summary.json", summary.to_dict())
    _write_jsonl(
        output_path / "query_stage_diagnostics.jsonl",
        (diagnosis.to_dict() for diagnosis in diagnoses),
    )
    zero_recall = [
        diagnosis for diagnosis in diagnoses if not diagnosis.stages[summary.target_stage].has_hit
    ]
    _write_jsonl(
        output_path / "zero_recall_taxonomy.jsonl",
        (
            {
                "qid": diagnosis.qid,
                "question": diagnosis.question,
                "gold_count": diagnosis.gold_count,
                "failure_type": diagnosis.failure_type,
                "failure_stage": diagnosis.failure_stage,
                "priority": diagnosis.priority,
                "anchor_types": sorted(_anchor_type_set(diagnosis.anchor_result)),
                "anchors": [anchor.to_dict() for anchor in diagnosis.anchor_result.anchors],
                "suggested_actions": diagnosis.suggested_actions,
            }
            for diagnosis in zero_recall
        ),
    )
    _write_jsonl(
        output_path / "anchor_inventory.jsonl",
        (
            {
                "qid": diagnosis.qid,
                "question": diagnosis.question,
                **diagnosis.anchor_result.to_dict(),
            }
            for diagnosis in diagnoses
        ),
    )
    _write_jsonl(
        output_path / "possible_matching_issues.jsonl",
        (issue.to_dict() for issue in possible_matching_issues),
    )
    _write_stage_matrix_csv(
        output_path / "stage_recall_matrix.csv", diagnoses, stage_names
    )
    _write_stage_matrix_md(
        output_path / "stage_recall_matrix.md", diagnoses, stage_names
    )
    _write_priority_markdown(output_path / "priority_queries.md", diagnoses)
