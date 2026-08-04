from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GuardConstraint:
    constraint_id: str
    constraint_type: str
    label: str
    required_groups: list[list[str]] = field(default_factory=list)
    prohibited_terms: list[str] = field(default_factory=list)
    severity: str = "medium"


@dataclass(slots=True)
class GuardConfig:
    mode: str
    reject_violations: bool
    filter_rejects: bool
    filter_downrank: bool
    min_guard_score: float
    max_output_per_query: int
    min_keep_per_query: int
    score_bonus_weight: float
    violation_penalty: float
    missing_penalty: float


@dataclass(slots=True)
class GuardResult:
    guard_score: float
    decision: str
    satisfied_constraints: list[str]
    missing_constraints: list[str]
    violation_tags: list[str]
    evidence_tags: list[str]
    reason: str


MODE_PRESETS = {
    "annotate": {
        "reject_violations": False,
        "filter_rejects": False,
        "filter_downrank": False,
        "min_guard_score": 0.0,
        "max_output_per_query": 100,
        "min_keep_per_query": 100,
        "score_bonus_weight": 0.20,
        "violation_penalty": 0.35,
        "missing_penalty": 0.08,
    },
    "balanced": {
        "reject_violations": True,
        "filter_rejects": True,
        "filter_downrank": False,
        "min_guard_score": 0.25,
        "max_output_per_query": 100,
        "min_keep_per_query": 15,
        "score_bonus_weight": 0.26,
        "violation_penalty": 0.45,
        "missing_penalty": 0.10,
    },
    "precision": {
        "reject_violations": True,
        "filter_rejects": True,
        "filter_downrank": True,
        "min_guard_score": 0.45,
        "max_output_per_query": 50,
        "min_keep_per_query": 0,
        "score_bonus_weight": 0.32,
        "violation_penalty": 0.60,
        "missing_penalty": 0.13,
    },
    "recall_safe": {
        "reject_violations": True,
        "filter_rejects": False,
        "filter_downrank": False,
        "min_guard_score": 0.12,
        "max_output_per_query": 100,
        "min_keep_per_query": 50,
        "score_bonus_weight": 0.18,
        "violation_penalty": 0.30,
        "missing_penalty": 0.06,
    },
}


def make_guard_config(mode: str) -> GuardConfig:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown guard mode: {mode}")
    return GuardConfig(mode=mode, **MODE_PRESETS[mode])


def normalize_text(value: object | None) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212-]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens_of(text: str) -> set[str]:
    return set(normalize_text(text).split())


def token_has_any(text: str, words: set[str]) -> bool:
    toks = tokens_of(text)
    for word in words:
        word = word.casefold()
        if word in toks:
            return True
        if word.endswith("y") and (word[:-1] + "ies") in toks:
            return True
        if not word.endswith("s") and (word + "s") in toks:
            return True
    return False


def contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    text_norm = normalize_text(text)
    if f" {phrase_norm} " in f" {text_norm} ":
        return True

    # Light plural-tolerant fallback for phrases.
    phrase_tokens = phrase_norm.split()
    text_tokens = text_norm.split()
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False

    def same_token(a: str, b: str) -> bool:
        if a == b:
            return True
        if a + "s" == b or b + "s" == a:
            return True
        if a.endswith("y") and a[:-1] + "ies" == b:
            return True
        if b.endswith("y") and b[:-1] + "ies" == a:
            return True
        return False

    n = len(phrase_tokens)
    for i in range(0, len(text_tokens) - n + 1):
        if all(same_token(phrase_tokens[j], text_tokens[i + j]) for j in range(n)):
            return True
    return False


def contains_any(text: str, terms: list[str]) -> bool:
    return any(contains_phrase(text, term) for term in terms)


def contains_all_groups(text: str, groups: list[list[str]]) -> tuple[bool, list[str], list[str]]:
    satisfied: list[str] = []
    missing: list[str] = []
    for group in groups:
        if contains_any(text, group):
            satisfied.append("/".join(group[:3]))
        else:
            missing.append("/".join(group[:3]))
    return len(missing) == 0, satisfied, missing


DATASET_ALIASES: dict[str, list[str]] = {
    "HotPotQA": ["hotpotqa", "hotpot qa"],
    "MS COCO": ["ms coco", "coco dataset", "microsoft coco"],
    "ActivityNet": ["activitynet", "activity net"],
    "WebVid": ["webvid", "web vid"],
    "AudioSet": ["audioset", "audio set"],
    "VGGSound": ["vggsound", "vgg sound"],
    "Ego4D": ["ego4d", "ego 4d"],
    "HowTo100M": ["howto100m", "how to 100m", "howto 100m"],
    "MSR-VTT": ["msr vtt", "msrvtt"],
    "ImageNet": ["imagenet", "image net"],
    "MMLU": ["mmlu"],
    "GSM8K": ["gsm8k", "gsm 8k"],
    "HumanEval": ["humaneval", "human eval"],
    "MBPP": ["mbpp"],
}


METHOD_ALIASES: dict[str, list[list[str]]] = {
    "RLHF": [["rlhf", "reinforcement learning from human feedback"]],
    "reward_shaping": [["reward shaping", "shaped reward", "reward shaping methods"]],
    "autoregressive_transformer": [["autoregressive", "auto regressive"], ["transformer", "transformers"]],
    "in_context_learning": [["in context learning", "in context capability", "icl"]],
    "pre_training": [["pre training", "pretraining", "pre train", "pretrained"]],
    "reinforcement_learning": [["reinforcement learning", "rl"]],
    "llm_agent": [["large language model", "large language models", "llm", "llms"], ["agent", "agents"]],
    "ranking_with_llm": [["rank", "ranking", "rerank", "reranking"], ["search result", "search results", "retrieval", "search"], ["llm", "llms", "large language model", "large language models"]],
}


SURVEY_EXCLUSION_QUERY_TERMS = [
    "exclude survey",
    "exclude surveys",
    "excluding survey",
    "not survey",
    "non survey",
    "non survey paper",
    "non survey papers",
    "exclude review",
    "excluding review",
    "not review",
]

SURVEY_PAPER_TERMS = [
    "survey",
    "review",
    "systematic review",
    "comprehensive survey",
    "literature review",
    "overview",
    "bibliometric",
]


def add_constraint_once(
    constraints: list[GuardConstraint],
    *,
    constraint_id: str,
    constraint_type: str,
    label: str,
    required_groups: list[list[str]] | None = None,
    prohibited_terms: list[str] | None = None,
    severity: str = "medium",
) -> None:
    if any(item.constraint_id == constraint_id for item in constraints):
        return
    constraints.append(
        GuardConstraint(
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            label=label,
            required_groups=required_groups or [],
            prohibited_terms=prohibited_terms or [],
            severity=severity,
        )
    )


def query_has_smaller_bigger_better(q: str) -> bool:
    has_small = (
        contains_any(q, ["smaller dataset", "smaller datasets", "small dataset", "small datasets", "less data", "limited data"])
        or (token_has_any(q, {"small", "smaller", "less", "limited"}) and token_has_any(q, {"dataset", "data"}))
    )
    has_big = (
        contains_any(q, ["bigger dataset", "bigger datasets", "larger dataset", "larger datasets", "more data"])
        or (token_has_any(q, {"big", "bigger", "large", "larger", "more"}) and token_has_any(q, {"dataset", "data"}))
    )
    has_better = token_has_any(q, {"better", "outperform", "outperforms", "improve", "improved", "superior"})
    return has_small and has_big and has_better


def query_has_video_generation(q: str) -> bool:
    return (
        contains_any(q, ["video generation", "video generations", "generate videos", "generating videos", "generate video", "generating video", "text to video", "text-to-video"])
        or (token_has_any(q, {"generate", "generating", "generation"}) and token_has_any(q, {"video", "videos"}))
    )


def query_has_image_video_description(q: str) -> bool:
    return (
        contains_any(q, ["image and video description", "image video description", "image and video caption", "image video caption"])
        or (
            token_has_any(q, {"image", "images", "visual"})
            and token_has_any(q, {"video", "videos"})
            and token_has_any(q, {"description", "descriptions", "caption", "captions", "captioning"})
        )
    )


def query_has_ranking_with_llm(q: str) -> bool:
    return (
        token_has_any(q, {"rank", "ranking", "rerank", "reranking"})
        and (contains_any(q, ["search result", "search results"]) or token_has_any(q, {"search", "retrieval", "results", "result"}))
        and token_has_any(q, {"llm", "llms"}) or
        (
            token_has_any(q, {"rank", "ranking", "rerank", "reranking"})
            and (contains_any(q, ["large language model", "large language models"]))
            and (contains_any(q, ["search result", "search results"]) or token_has_any(q, {"search", "retrieval", "results", "result"}))
        )
    )


def extract_guard_constraints(question: str) -> list[GuardConstraint]:
    q = normalize_text(question)
    constraints: list[GuardConstraint] = []

    if contains_any(q, SURVEY_EXCLUSION_QUERY_TERMS):
        add_constraint_once(
            constraints,
            constraint_id="exclude_survey_or_review",
            constraint_type="exclusion",
            label="排除综述/调研类论文",
            prohibited_terms=SURVEY_PAPER_TERMS,
            severity="high",
        )

    for dataset_name, aliases in DATASET_ALIASES.items():
        if contains_any(q, aliases):
            add_constraint_once(
                constraints,
                constraint_id=f"dataset_{normalize_text(dataset_name).replace(' ', '_')}",
                constraint_type="dataset",
                label=f"要求数据集：{dataset_name}",
                required_groups=[aliases],
                severity="high",
            )

    # Method constraints: multi-group constraints require all groups to appear in the query.
    for method_name, groups in METHOD_ALIASES.items():
        if method_name == "ranking_with_llm":
            should_trigger = query_has_ranking_with_llm(q)
        elif len(groups) >= 2:
            should_trigger, _, _ = contains_all_groups(q, groups)
        else:
            should_trigger = contains_any(q, groups[0] if groups else [])

        if should_trigger:
            severity = "high" if method_name in {"RLHF", "reward_shaping", "autoregressive_transformer", "llm_agent", "ranking_with_llm"} else "medium"
            add_constraint_once(
                constraints,
                constraint_id=f"method_{method_name}",
                constraint_type="method",
                label=f"要求方法/机制：{method_name}",
                required_groups=groups,
                severity=severity,
            )

    if (
        contains_any(q, ["large language model agent", "large language model agents", "llm agent", "llm agents"])
        or (
            (contains_any(q, ["large language model", "large language models"]) or token_has_any(q, {"llm", "llms"}))
            and token_has_any(q, {"agent", "agents"})
        )
    ):
        add_constraint_once(
            constraints,
            constraint_id="method_llm_agent",
            constraint_type="method",
            label="要求方法/机制：llm_agent",
            required_groups=METHOD_ALIASES["llm_agent"],
            severity="high",
        )

    if (contains_any(q, ["visual and audio", "visual audio", "audio visual", "audio and visual"])
            or (token_has_any(q, {"visual", "vision", "image", "video"}) and token_has_any(q, {"audio", "sound"}))):
        add_constraint_once(
            constraints,
            constraint_id="modality_visual_audio",
            constraint_type="modality",
            label="要求同时支持视觉与音频输入",
            required_groups=[
                ["visual", "vision", "image", "video"],
                ["audio", "sound", "acoustic"],
            ],
            severity="high",
        )

    if query_has_image_video_description(q):
        add_constraint_once(
            constraints,
            constraint_id="task_image_video_description",
            constraint_type="task",
            label="要求图像与视频描述/字幕任务",
            required_groups=[
                ["image", "images", "visual"],
                ["video", "videos"],
                ["description", "descriptions", "caption", "captions", "captioning"],
            ],
            severity="high",
        )

    if contains_any(q, ["long video", "long videos", "long form video", "long form videos"]):
        add_constraint_once(
            constraints,
            constraint_id="task_long_video",
            constraint_type="task",
            label="要求长视频场景",
            required_groups=[
                ["long video", "long videos", "long form video", "long form videos"],
            ],
            severity="medium",
        )

    if query_has_video_generation(q):
        add_constraint_once(
            constraints,
            constraint_id="task_video_generation",
            constraint_type="task",
            label="要求视频生成任务",
            required_groups=[
                ["video generation", "video generations", "generate video", "generate videos", "generating video", "generating videos", "text to video", "text-to-video"],
            ],
            severity="medium",
        )

    if contains_any(q, ["hallucination", "hallucinations"]):
        add_constraint_once(
            constraints,
            constraint_id="task_hallucination",
            constraint_type="task",
            label="要求幻觉问题相关",
            required_groups=[
                ["hallucination", "hallucinations", "hallucinate"],
            ],
            severity="medium",
        )

    if contains_any(q, ["foundation model", "foundation models"]):
        add_constraint_once(
            constraints,
            constraint_id="model_foundation_model",
            constraint_type="model",
            label="要求基础模型/多模态基础模型",
            required_groups=[
                ["foundation model", "foundation models"],
            ],
            severity="medium",
        )

    if query_has_smaller_bigger_better(q):
        add_constraint_once(
            constraints,
            constraint_id="relation_smaller_dataset_better",
            constraint_type="relation",
            label="要求小数据集优于大数据集的比较关系",
            required_groups=[
                ["smaller dataset", "smaller datasets", "small dataset", "small datasets", "less data", "limited data", "deduplicating", "deduplication"],
                ["bigger dataset", "bigger datasets", "larger dataset", "larger datasets", "more data", "scaling"],
                ["better", "better model", "better models", "outperform", "improve", "improved", "makes language models better"],
            ],
            severity="high",
        )

    return constraints


def raw_meta(paper: dict[str, Any]) -> dict[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, dict) else {}


def paper_text(paper: dict[str, Any]) -> tuple[str, str]:
    title = normalize_text(paper.get("title"))
    abstract = normalize_text(paper.get("abstract"))
    raw = raw_meta(paper)
    extra_parts = [
        raw.get("display_name"),
        raw.get("venue"),
        raw.get("host_venue"),
        raw.get("source"),
    ]
    extra = normalize_text(" ".join(str(x or "") for x in extra_parts))
    full = normalize_text(" ".join([title, abstract, extra]))
    return title, full


def base_rank_score(paper: dict[str, Any]) -> float:
    raw = raw_meta(paper)
    for key in ("b5_selector_score", "b3_final_score", "b4_pre_rerank_score"):
        value = raw.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def evaluate_paper_with_guard(
    paper: dict[str, Any],
    constraints: list[GuardConstraint],
    config: GuardConfig | None = None,
) -> GuardResult:
    config = config or make_guard_config("balanced")
    title_text, full_text = paper_text(paper)

    if not constraints:
        return GuardResult(
            guard_score=1.0,
            decision="pass",
            satisfied_constraints=[],
            missing_constraints=[],
            violation_tags=[],
            evidence_tags=["no_explicit_hard_constraint"],
            reason="未检测到明确硬约束，保持原排序证据。",
        )

    satisfied_constraints: list[str] = []
    missing_constraints: list[str] = []
    violation_tags: list[str] = []
    evidence_tags: list[str] = []

    total_weight = 0.0
    satisfied_weight = 0.0

    for constraint in constraints:
        weight = 1.5 if constraint.severity == "high" else 1.0
        total_weight += weight

        if constraint.constraint_type == "exclusion":
            title_hit = contains_any(title_text, constraint.prohibited_terms)
            full_hit = contains_any(full_text, constraint.prohibited_terms)
            if title_hit or full_hit:
                tag = f"violates_{constraint.constraint_id}"
                violation_tags.append(tag)
                missing_constraints.append(constraint.label)
            else:
                satisfied_constraints.append(constraint.label)
                evidence_tags.append(f"satisfy_{constraint.constraint_id}")
                satisfied_weight += weight
            continue

        ok, group_hits, group_misses = contains_all_groups(full_text, constraint.required_groups)
        if ok:
            satisfied_constraints.append(constraint.label)
            evidence_tags.append(f"satisfy_{constraint.constraint_id}")
            for hit in group_hits:
                evidence_tags.append(f"evidence_{constraint.constraint_id}:{hit}")
            satisfied_weight += weight
        else:
            missing_constraints.append(f"{constraint.label}｜缺失：{'; '.join(group_misses)}")

    guard_score = satisfied_weight / total_weight if total_weight else 1.0

    if violation_tags and config.reject_violations:
        decision = "reject"
    elif guard_score >= 0.75 and not missing_constraints:
        decision = "pass"
    elif guard_score >= config.min_guard_score:
        decision = "soft_pass"
    else:
        decision = "downrank"

    reason = build_guard_reason(
        decision=decision,
        guard_score=guard_score,
        satisfied=satisfied_constraints,
        missing=missing_constraints,
        violations=violation_tags,
    )
    return GuardResult(
        guard_score=guard_score,
        decision=decision,
        satisfied_constraints=satisfied_constraints,
        missing_constraints=missing_constraints,
        violation_tags=violation_tags,
        evidence_tags=evidence_tags,
        reason=reason,
    )


def build_guard_reason(
    *,
    decision: str,
    guard_score: float,
    satisfied: list[str],
    missing: list[str],
    violations: list[str],
) -> str:
    decision_text = {
        "pass": "通过硬约束校验",
        "soft_pass": "部分通过硬约束校验",
        "downrank": "硬约束证据不足，建议降权",
        "reject": "违反硬约束，建议过滤",
    }.get(decision, decision)

    parts = [f"{decision_text}，规则分={guard_score:.3f}"]
    if satisfied:
        parts.append("满足：" + "；".join(satisfied[:3]))
    if missing:
        parts.append("缺失：" + "；".join(missing[:3]))
    if violations:
        parts.append("违规：" + "；".join(violations[:3]))
    return "。".join(parts) + "。"


def annotate_paper_with_guard(
    paper: dict[str, Any],
    constraints: list[GuardConstraint],
    config: GuardConfig,
) -> dict[str, Any]:
    result = evaluate_paper_with_guard(paper, constraints, config)
    annotated = dict(paper)
    raw = dict(raw_meta(annotated))

    base_score = base_rank_score(annotated)
    final_score = base_score + config.score_bonus_weight * result.guard_score
    final_score -= config.violation_penalty * len(result.violation_tags)
    final_score -= config.missing_penalty * len(result.missing_constraints)

    raw["b5_1_guard_score"] = result.guard_score
    raw["b5_1_guard_decision"] = result.decision
    raw["b5_1_satisfied_constraints"] = result.satisfied_constraints
    raw["b5_1_missing_constraints"] = result.missing_constraints
    raw["b5_1_violation_tags"] = result.violation_tags
    raw["b5_1_evidence_tags"] = result.evidence_tags
    raw["b5_1_guard_reason"] = result.reason
    raw["b5_1_base_score"] = base_score
    raw["b5_1_final_score"] = final_score
    annotated["raw"] = raw
    return annotated


def constraint_to_dict(constraint: GuardConstraint) -> dict[str, Any]:
    return {
        "constraint_id": constraint.constraint_id,
        "constraint_type": constraint.constraint_type,
        "label": constraint.label,
        "required_groups": constraint.required_groups,
        "prohibited_terms": constraint.prohibited_terms,
        "severity": constraint.severity,
    }


def apply_guard_to_record(
    record: dict[str, Any],
    config: GuardConfig,
) -> dict[str, Any]:
    question = str(record.get("question") or "")
    constraints = extract_guard_constraints(question)
    papers = record.get("papers") or []
    if not isinstance(papers, list):
        papers = []

    annotated = [
        annotate_paper_with_guard(paper, constraints, config)
        for paper in papers
        if isinstance(paper, dict)
    ]

    if config.mode == "annotate":
        selected = annotated
    else:
        selected = []
        for paper in annotated:
            decision = raw_meta(paper).get("b5_1_guard_decision")
            guard_score = float(raw_meta(paper).get("b5_1_guard_score", 0.0))
            if config.filter_rejects and decision == "reject":
                continue
            if config.filter_downrank and decision == "downrank":
                continue
            if config.filter_downrank and constraints and guard_score < config.min_guard_score:
                continue
            selected.append(paper)

        if len(selected) < config.min_keep_per_query:
            selected_ids = {id(item) for item in selected}
            for paper in annotated:
                if id(paper) in selected_ids:
                    continue
                if config.filter_rejects and raw_meta(paper).get("b5_1_guard_decision") == "reject":
                    continue
                selected.append(paper)
                selected_ids.add(id(paper))
                if len(selected) >= config.min_keep_per_query:
                    break

    selected.sort(
        key=lambda paper: (
            -float(raw_meta(paper).get("b5_1_final_score", 0.0)),
            -float(raw_meta(paper).get("b5_1_guard_score", 0.0)),
            str(paper.get("title") or "").casefold(),
        )
    )
    selected = selected[: config.max_output_per_query]

    return {
        "qid": record.get("qid"),
        "question": record.get("question"),
        "papers": selected,
        "guard": {
            "strategy": "b5_1_constraint_guard",
            "config": {
                "mode": config.mode,
                "reject_violations": config.reject_violations,
                "filter_rejects": config.filter_rejects,
                "filter_downrank": config.filter_downrank,
                "min_guard_score": config.min_guard_score,
                "max_output_per_query": config.max_output_per_query,
                "min_keep_per_query": config.min_keep_per_query,
            },
            "constraints": [constraint_to_dict(item) for item in constraints],
        },
    }
