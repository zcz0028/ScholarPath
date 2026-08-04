from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scholarpath.selector.selector import (
    compute_selector_score,
    make_selector_config,
    raw_meta,
)


@dataclass(slots=True)
class GuardAwareConfig:
    mode: str
    selector_mode: str
    selector_weight: float
    guard_weight: float
    guard_final_weight: float
    pass_bonus: float
    soft_pass_bonus: float
    downrank_penalty: float
    reject_penalty: float
    missing_penalty: float
    violation_penalty: float
    min_final_score: float
    max_output_per_query: int
    min_keep_per_query: int
    filter_reject: bool
    filter_downrank: bool
    require_non_low_selector: bool


MODE_PRESETS = {
    "ranking": {
        "selector_mode": "balanced",
        "selector_weight": 0.58,
        "guard_weight": 0.25,
        "guard_final_weight": 0.17,
        "pass_bonus": 0.04,
        "soft_pass_bonus": 0.015,
        "downrank_penalty": 0.04,
        "reject_penalty": 0.20,
        "missing_penalty": 0.012,
        "violation_penalty": 0.08,
        "min_final_score": -999.0,
        "max_output_per_query": 100,
        "min_keep_per_query": 100,
        "filter_reject": False,
        "filter_downrank": False,
        "require_non_low_selector": False,
    },
    "balanced": {
        "selector_mode": "balanced",
        "selector_weight": 0.60,
        "guard_weight": 0.26,
        "guard_final_weight": 0.14,
        "pass_bonus": 0.045,
        "soft_pass_bonus": 0.018,
        "downrank_penalty": 0.06,
        "reject_penalty": 0.25,
        "missing_penalty": 0.014,
        "violation_penalty": 0.10,
        "min_final_score": 0.18,
        "max_output_per_query": 100,
        "min_keep_per_query": 20,
        "filter_reject": True,
        "filter_downrank": False,
        "require_non_low_selector": False,
    },
    "precision": {
        "selector_mode": "balanced",
        "selector_weight": 0.66,
        "guard_weight": 0.26,
        "guard_final_weight": 0.08,
        "pass_bonus": 0.06,
        "soft_pass_bonus": 0.02,
        "downrank_penalty": 0.12,
        "reject_penalty": 0.40,
        "missing_penalty": 0.022,
        "violation_penalty": 0.16,
        "min_final_score": 0.30,
        "max_output_per_query": 50,
        "min_keep_per_query": 0,
        "filter_reject": True,
        "filter_downrank": True,
        "require_non_low_selector": False,
    },
    "precision_safe": {
        "selector_mode": "balanced",
        "selector_weight": 0.64,
        "guard_weight": 0.24,
        "guard_final_weight": 0.12,
        "pass_bonus": 0.055,
        "soft_pass_bonus": 0.02,
        "downrank_penalty": 0.10,
        "reject_penalty": 0.35,
        "missing_penalty": 0.018,
        "violation_penalty": 0.14,
        "min_final_score": 0.26,
        "max_output_per_query": 60,
        "min_keep_per_query": 8,
        "filter_reject": True,
        "filter_downrank": True,
        "require_non_low_selector": False,
    },
}


def make_guard_aware_config(mode: str) -> GuardAwareConfig:
    if mode not in MODE_PRESETS:
        raise ValueError(f"Unknown guard-aware selector mode: {mode}")
    return GuardAwareConfig(mode=mode, **MODE_PRESETS[mode])


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_guard_final_score(value: float) -> float:
    # b5_1_final_score can be outside [0,1]. Compress to a bounded supportive signal.
    if value <= 0:
        return 0.0
    return min(1.0, value / 1.5)


def guard_decision_bonus(decision: str, config: GuardAwareConfig) -> float:
    if decision == "pass":
        return config.pass_bonus
    if decision == "soft_pass":
        return config.soft_pass_bonus
    if decision == "downrank":
        return -config.downrank_penalty
    if decision == "reject":
        return -config.reject_penalty
    return 0.0


def compute_guard_aware_score(
    paper: dict[str, Any],
    config: GuardAwareConfig,
) -> dict[str, Any]:
    raw = raw_meta(paper)
    selector_config = make_selector_config(config.selector_mode)
    selector_result = compute_selector_score(paper, selector_config)

    selector_score = safe_float(selector_result.get("selector_score"), 0.0)
    selector_label = str(selector_result.get("relevance_label") or "unknown")

    guard_score = safe_float(raw.get("b5_1_guard_score"), 1.0)
    guard_decision = str(raw.get("b5_1_guard_decision") or "pass")
    guard_final_signal = normalize_guard_final_score(
        safe_float(raw.get("b5_1_final_score"), 0.0)
    )
    missing_count = len(safe_list(raw.get("b5_1_missing_constraints")))
    violation_count = len(safe_list(raw.get("b5_1_violation_tags")))

    final_score = (
        config.selector_weight * selector_score
        + config.guard_weight * guard_score
        + config.guard_final_weight * guard_final_signal
        + guard_decision_bonus(guard_decision, config)
        - config.missing_penalty * missing_count
        - config.violation_penalty * violation_count
    )

    if selector_score < 0.10 and guard_decision != "pass":
        final_score -= 0.035
    if selector_score < 0.06:
        final_score -= 0.035

    if final_score >= 0.46 and guard_decision in {"pass", "soft_pass"}:
        decision = "strong_keep"
    elif final_score >= config.min_final_score and guard_decision != "reject":
        decision = "keep"
    elif guard_decision == "reject":
        decision = "reject"
    else:
        decision = "drop"

    reason_tags = []
    if selector_score >= 0.30:
        reason_tags.append("selector_score_high")
    elif selector_score >= 0.18:
        reason_tags.append("selector_score_medium")
    else:
        reason_tags.append("selector_score_low")

    if guard_decision == "pass":
        reason_tags.append("guard_pass")
    elif guard_decision == "soft_pass":
        reason_tags.append("guard_soft_pass")
    elif guard_decision == "downrank":
        reason_tags.append("guard_downrank")
    elif guard_decision == "reject":
        reason_tags.append("guard_reject")

    if missing_count:
        reason_tags.append(f"missing_constraints_{missing_count}")
    if violation_count:
        reason_tags.append(f"violations_{violation_count}")

    return {
        "selector_score": selector_score,
        "selector_label": selector_label,
        "guard_score": guard_score,
        "guard_decision": guard_decision,
        "guard_final_signal": guard_final_signal,
        "missing_count": missing_count,
        "violation_count": violation_count,
        "final_score": final_score,
        "decision": decision,
        "reason_tags": reason_tags,
    }


def build_guard_aware_reason(info: dict[str, Any], raw: dict[str, Any]) -> str:
    decision_text = {
        "strong_keep": "强保留",
        "keep": "保留",
        "drop": "丢弃",
        "reject": "过滤",
    }.get(str(info.get("decision")), str(info.get("decision")))

    guard_reason = str(raw.get("b5_1_guard_reason") or "")
    selector_label = str(info.get("selector_label") or "unknown")
    final_score = safe_float(info.get("final_score"), 0.0)
    selector_score = safe_float(info.get("selector_score"), 0.0)
    guard_score = safe_float(info.get("guard_score"), 0.0)

    parts = [
        f"{decision_text}：综合分={final_score:.3f}",
        f"Selector={selector_label}({selector_score:.3f})",
        f"Guard={info.get('guard_decision')}({guard_score:.3f})",
    ]
    if guard_reason:
        parts.append(f"规则原因：{guard_reason}")
    if info.get("missing_count"):
        parts.append(f"仍有{info.get('missing_count')}项硬约束缺失")
    if info.get("violation_count"):
        parts.append(f"存在{info.get('violation_count')}项违规约束")
    return "；".join(parts) + "。"


def annotate_paper_guard_aware(
    paper: dict[str, Any],
    config: GuardAwareConfig,
) -> dict[str, Any]:
    info = compute_guard_aware_score(paper, config)
    annotated = dict(paper)
    raw = dict(raw_meta(annotated))
    raw["b5_2_mode"] = config.mode
    raw["b5_2_selector_score"] = info["selector_score"]
    raw["b5_2_selector_label"] = info["selector_label"]
    raw["b5_2_guard_score"] = info["guard_score"]
    raw["b5_2_guard_decision"] = info["guard_decision"]
    raw["b5_2_guard_final_signal"] = info["guard_final_signal"]
    raw["b5_2_missing_count"] = info["missing_count"]
    raw["b5_2_violation_count"] = info["violation_count"]
    raw["b5_2_final_score"] = info["final_score"]
    raw["b5_2_decision"] = info["decision"]
    raw["b5_2_reason_tags"] = info["reason_tags"]
    raw["b5_2_reason_text"] = build_guard_aware_reason(info, raw)
    annotated["raw"] = raw
    return annotated


def select_guard_aware_papers(
    papers: list[dict[str, Any]],
    config: GuardAwareConfig,
) -> list[dict[str, Any]]:
    annotated = [
        annotate_paper_guard_aware(paper, config)
        for paper in papers
        if isinstance(paper, dict)
    ]

    if config.mode == "ranking":
        selected = annotated
    else:
        selected = []
        for paper in annotated:
            raw = raw_meta(paper)
            decision = str(raw.get("b5_2_decision") or "")
            guard_decision = str(raw.get("b5_2_guard_decision") or "")
            selector_label = str(raw.get("b5_2_selector_label") or "")
            final_score = safe_float(raw.get("b5_2_final_score"), 0.0)

            if config.filter_reject and guard_decision == "reject":
                continue
            if config.filter_downrank and guard_decision == "downrank":
                continue
            if config.require_non_low_selector and selector_label == "low_relevance":
                continue
            if decision in {"strong_keep", "keep"} or final_score >= config.min_final_score:
                selected.append(paper)

        if len(selected) < config.min_keep_per_query:
            selected_ids = {id(item) for item in selected}
            for paper in annotated:
                if id(paper) in selected_ids:
                    continue
                raw = raw_meta(paper)
                if config.filter_reject and raw.get("b5_2_guard_decision") == "reject":
                    continue
                selected.append(paper)
                selected_ids.add(id(paper))
                if len(selected) >= config.min_keep_per_query:
                    break

    selected.sort(
        key=lambda paper: (
            -safe_float(raw_meta(paper).get("b5_2_final_score"), 0.0),
            -safe_float(raw_meta(paper).get("b5_2_selector_score"), 0.0),
            -safe_float(raw_meta(paper).get("b5_2_guard_score"), 0.0),
            str(paper.get("title") or "").casefold(),
        )
    )
    return selected[: config.max_output_per_query]


def apply_guard_aware_selector_to_record(
    record: dict[str, Any],
    config: GuardAwareConfig,
) -> dict[str, Any]:
    papers = record.get("papers") or []
    if not isinstance(papers, list):
        papers = []
    selected = select_guard_aware_papers(papers, config)
    return {
        "qid": record.get("qid"),
        "question": record.get("question"),
        "papers": selected,
        "guard_aware_selector": {
            "strategy": "b5_2_guard_aware_selector",
            "config": {
                "mode": config.mode,
                "selector_mode": config.selector_mode,
                "selector_weight": config.selector_weight,
                "guard_weight": config.guard_weight,
                "guard_final_weight": config.guard_final_weight,
                "min_final_score": config.min_final_score,
                "max_output_per_query": config.max_output_per_query,
                "min_keep_per_query": config.min_keep_per_query,
                "filter_reject": config.filter_reject,
                "filter_downrank": config.filter_downrank,
            },
        },
    }
