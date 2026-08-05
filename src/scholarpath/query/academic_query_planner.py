from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from scholarpath.query.anchor_extractor import AnchorEvidence, extract_anchors
from scholarpath.query.rewriter import (
    clean_question_text,
    compact_query_terms,
    extract_keywords,
    normalize_space,
)


@dataclass(slots=True, frozen=True)
class AliasRule:
    name: str
    pattern: str
    aliases: tuple[str, ...]
    anchor_type: str
    plan_type: str
    priority: int
    reason: str


@dataclass(slots=True, frozen=True)
class PlannedQuery:
    text: str
    plan_type: str
    priority: int
    confidence: float
    reason: str
    anchor_texts: tuple[str, ...] = ()
    anchor_types: tuple[str, ...] = ()
    estimated_api_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["anchor_texts"] = list(self.anchor_texts)
        value["anchor_types"] = list(self.anchor_types)
        return value


@dataclass(slots=True)
class AcademicQueryPlan:
    qid: str
    question: str
    anchors: list[AnchorEvidence]
    derived_aliases: list[str]
    filters: dict[str, Any]
    planned_queries: list[PlannedQuery]
    planner_version: str = "anchor_planner_v1"
    analysis_only_uses_gold: bool = False
    production_retrieval_uses_gold: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "planner_version": self.planner_version,
            "anchors": [item.to_dict() for item in self.anchors],
            "anchors_by_type": _anchors_by_type(self.anchors),
            "derived_aliases": list(self.derived_aliases),
            "filters": dict(self.filters),
            "planned_queries": [item.to_dict() for item in self.planned_queries],
            "estimated_api_calls": sum(
                item.estimated_api_calls for item in self.planned_queries
            ),
            "analysis_only_uses_gold": self.analysis_only_uses_gold,
            "production_retrieval_uses_gold": self.production_retrieval_uses_gold,
        }


@dataclass(slots=True)
class PlannerConfig:
    max_queries: int = 2
    max_terms_per_query: int = 14
    include_keyword_fallback: bool = True


ALIAS_RULES: tuple[AliasRule, ...] = (
    AliasRule(
        name="icl_emergence",
        pattern=r"\bin[- ]context learning\b.*\b(?:pre[- ]?training|pretraining)\b|\b(?:pre[- ]?training|pretraining)\b.*\bin[- ]context learning\b",
        aliases=(
            "in-context learning emergence during pretraining",
            "induction heads mechanism of in-context learning",
            "implicit learning mechanism for in-context learning",
        ),
        anchor_type="method",
        plan_type="method_alias",
        priority=98,
        reason="Expands the mechanism-oriented terminology used for the emergence of in-context learning.",
    ),
    AliasRule(
        name="audio_visual_foundation_model",
        pattern=r"\bmultimodal\b.*\b(?:visual|vision)\b.*\baudio\b|\baudio[- ]visual\b.*\b(?:foundation model|pre[- ]?train)",
        aliases=(
            "audio visual multimodal foundation model pretraining",
            "vision audio foundation model large-scale pretraining",
        ),
        anchor_type="modality",
        plan_type="modality_composition",
        priority=96,
        reason="Preserves the joint visual, audio, and audio-visual modality composition.",
    ),
    AliasRule(
        name="long_video_description",
        pattern=r"\blong video (?:description|captioning)\b|\bvideos?\b.*\b(?:several minutes|minutes-long|long-form)\b",
        aliases=(
            "long-form video captioning minutes-long video",
            "dense video captioning long video description",
            "movie description long video",
        ),
        anchor_type="task",
        plan_type="task_alias",
        priority=96,
        reason="Maps informal long-video description wording to stable captioning task names.",
    ),
    AliasRule(
        name="transformer_3d_video",
        pattern=r"\btransformer\b.*\b3d\b.*\bvideo generation\b|\b3d\b.*\bvideo generation\b.*\btransformer\b",
        aliases=(
            "transformer 3D video generation",
            "transformer-based 4D dynamic scene generation",
        ),
        anchor_type="task",
        plan_type="task_method",
        priority=97,
        reason="Keeps transformer, 3D/4D, and video-generation constraints in one query.",
    ),
    AliasRule(
        name="trigger_free_event_extraction",
        pattern=r"\btrigger[- ]free\b.*\bevent extraction\b|\bevent extraction\b.*\b(?:without|no)\b.*\btrigger",
        aliases=(
            "trigger-free document-level event extraction",
            "document event extraction without trigger annotations",
            "event extraction without human-annotated triggers",
        ),
        anchor_type="method",
        plan_type="task_method",
        priority=99,
        reason="Promotes the trigger-free and no-trigger-annotation constraints instead of generic event extraction.",
    ),
    AliasRule(
        name="icl_vs_sft_information_extraction",
        pattern=r"\bin[- ]context learning\b.*\b(?:supervised fine[- ]tun(?:e|ed|ing)?|sft)\b.*\b(?:information extraction|ner|re|ee)\b",
        aliases=(
            "in-context learning versus supervised fine-tuning information extraction",
            "LLM ICL NER RE EE small model comparison",
        ),
        anchor_type="comparison",
        plan_type="comparison_aware",
        priority=98,
        reason="Builds the requested ICL-versus-SFT comparison for information extraction.",
    ),
    AliasRule(
        name="automated_literature_review",
        pattern=r"\bmultiple papers\b.*\b(?:survey|review)\b|\bwrite a survey\b|\brelationships between (?:multiple )?papers\b",
        aliases=(
            "LLM automated literature review survey generation",
            "large language model multi-paper synthesis scientific survey writing",
            "scientific literature synthesis with language models",
        ),
        anchor_type="task",
        plan_type="task_alias",
        priority=96,
        reason="Maps long natural-language descriptions to automated literature review and scientific synthesis.",
    ),
    AliasRule(
        name="multi_response_sft",
        pattern=r"\bsame prompt\b.*\bdifferent responses\b|\bmultiple responses\b.*\b(?:sft|fine[- ]tuning)\b",
        aliases=(
            "multiple responses per prompt supervised fine-tuning",
            "response diversity improves instruction tuning",
            "multi-response instruction tuning",
        ),
        anchor_type="method",
        plan_type="method_alias",
        priority=97,
        reason="Maps the same-prompt/multiple-response condition to response-diversity terminology.",
    ),
    AliasRule(
        name="commonsense_machine_translation",
        pattern=r"\bcommon ?sense\b.*\bmachine translation\b|\bmachine translation\b.*\bcommon ?sense\b",
        aliases=(
            "commonsense reasoning for machine translation",
            "commonsense-aware neural machine translation",
            "external knowledge machine translation commonsense",
        ),
        anchor_type="task",
        plan_type="task_alias",
        priority=96,
        reason="Expands common-sense translation into the terminology used by machine-translation papers.",
    ),
    AliasRule(
        name="rl_diffusion_video",
        pattern=r"\breinforcement learning\b.*\bdiffusion\b.*\bvideo generation\b",
        aliases=(
            "reinforcement learning optimize diffusion video generation",
            "reward fine-tuning diffusion model for text-to-video generation",
        ),
        anchor_type="method",
        plan_type="task_method",
        priority=98,
        reason="Keeps reinforcement learning, diffusion, and video generation in a constrained intersection query.",
    ),
    AliasRule(
        name="code_benchmark_difficulty",
        pattern=r"\b(?:harder|more difficult) than\b.*\b(?:humaneval|mbpp)\b.*\b(?:easier|less difficult) than\b.*\b(?:code[_ ]?contests)\b",
        aliases=(
            "code generation benchmark harder than HumanEval MBPP",
            "program synthesis benchmark between HumanEval and CodeContests difficulty",
        ),
        anchor_type="comparison",
        plan_type="benchmark_comparison",
        priority=100,
        reason="Uses benchmark names together with the requested difficulty interval.",
    ),
    AliasRule(
        name="neural_quantum_monte_carlo",
        pattern=r"\bneural network(?: based)? quantum monte carlo\b|\bneural quantum states?\b",
        aliases=(
            "neural network quantum Monte Carlo",
            "neural quantum states variational Monte Carlo",
            "deep neural quantum Monte Carlo",
        ),
        anchor_type="domain_entity",
        plan_type="exact_phrase",
        priority=99,
        reason="Preserves the domain phrase and adds the established neural-quantum-state alias.",
    ),
    AliasRule(
        name="3d_aigc",
        pattern=r"\b3d\b.*\b(?:aigc|generative)\b.*\b(?:foundation models?|scene understanding)\b|\b3d scene understanding\b.*\b3d aigc foundation models?\b",
        aliases=(
            "3D scene understanding generative foundation model",
            "3D AIGC foundation model scene understanding",
            "3D world model scene understanding",
        ),
        anchor_type="model",
        plan_type="model_alias",
        priority=97,
        reason="Expands AIGC into 3D generative foundation-model and world-model terminology.",
    ),
    AliasRule(
        name="quantized_pretraining",
        pattern=r"\b(?:llm|large language model)\b.*\bquantized pre[- ]?training\b|\bquantized pre[- ]?training\b.*\b(?:llm|large language model)\b",
        aliases=(
            "quantized pretraining large language model",
            "low-bit quantization-aware LLM pretraining",
            "low precision language model pretraining",
        ),
        anchor_type="method",
        plan_type="method_alias",
        priority=99,
        reason="Expands quantized pretraining to low-bit and quantization-aware pretraining.",
    ),
    AliasRule(
        name="identity_preserving_video",
        pattern=r"\bidentity (?:preservation|preserving|consistent)\b.*\bvideo generation\b",
        aliases=(
            "identity-preserving video generation",
            "subject-driven identity-consistent text-to-video generation",
            "personalized video generation identity consistency",
        ),
        anchor_type="task",
        plan_type="task_alias",
        priority=98,
        reason="Expands identity preservation to subject-driven and identity-consistent video generation.",
    ),
    AliasRule(
        name="image_encoding_distribution",
        pattern=r"\bimage encoding distributions?\b",
        aliases=(
            "image latent representation distribution encoding",
            "visual tokenization encoding distribution",
            "image representation distribution learning",
        ),
        anchor_type="task",
        plan_type="ambiguity_split",
        priority=92,
        reason="Splits the ambiguous phrase into latent-representation and visual-tokenization interpretations.",
    ),
    AliasRule(
        name="synthetic_long_reasoning",
        pattern=r"\bsynthetic data\b.*\b(?:long thought|reasoning)\b|\blong thought data\b",
        aliases=(
            "synthetic long chain-of-thought data generation for LLM",
            "automatic high-quality reasoning trace data synthesis",
            "diverse difficult synthetic reasoning data language model",
        ),
        anchor_type="method",
        plan_type="task_alias",
        priority=98,
        reason="Maps long-thought data to chain-of-thought and reasoning-trace synthesis.",
    ),
    AliasRule(
        name="frame_selection_video_understanding",
        pattern=r"\b(?:select frames?|frame selection)\b.*\bvideo understanding\b|\bvideo understanding\b.*\b(?:select frames?|frame selection)\b",
        aliases=(
            "adaptive frame selection for video understanding",
            "keyframe sampling temporal token selection video understanding",
            "temporal token pruning video understanding",
        ),
        anchor_type="method",
        plan_type="task_method",
        priority=98,
        reason="Expands frame selection to keyframe sampling and temporal-token selection.",
    ),
    AliasRule(
        name="crypto_private_learning",
        pattern=r"\bcrypto[- ]based private learning\b|\bprivacy[- ]preserving machine learning\b",
        aliases=(
            "cryptographic privacy-preserving machine learning",
            "secure multiparty computation private machine learning",
            "homomorphic encryption privacy-preserving machine learning",
        ),
        anchor_type="domain_entity",
        plan_type="domain_alias",
        priority=98,
        reason="Expands the broad phrase into cryptographic ML, MPC, and homomorphic-encryption terminology.",
    ),
    AliasRule(
        name="controllable_video_generation",
        pattern=r"\bcontrollability\b.*\bvideo generation\b|\bcontrollable video generation\b",
        aliases=(
            "controllable video generation motion camera trajectory layout",
            "motion controlled text-to-video generation",
            "camera trajectory controlled video generation",
        ),
        anchor_type="task",
        plan_type="task_decomposition",
        priority=95,
        reason="Decomposes controllability into motion, camera, trajectory, and layout controls.",
    ),
    AliasRule(
        name="robot_planning_benchmark",
        pattern=r"\brobot decision making\b.*\btask planning\b.*\b(?:datasets?|benchmarks?)\b",
        aliases=(
            "robot task planning decision making datasets benchmarks",
            "embodied AI long-horizon planning benchmark",
            "robot planning benchmark dataset decision making",
        ),
        anchor_type="benchmark",
        plan_type="benchmark_aware",
        priority=99,
        reason="Preserves dataset/benchmark intent and expands robot planning to embodied long-horizon planning.",
    ),
    AliasRule(
        name="financial_llm_agents",
        pattern=r"\bllm agents?\b.*\bfinancial tasks?\b|\bagents?\b.*\b(?:finance|financial)\b.*\b(?:evaluat|benchmark)",
        aliases=(
            "LLM agent benchmark financial tasks",
            "evaluation benchmark autonomous language agents for finance",
            "financial agent benchmark large language model",
        ),
        anchor_type="benchmark",
        plan_type="benchmark_aware",
        priority=99,
        reason="Keeps the agent requirement and constructs finance-specific evaluation queries.",
    ),
    AliasRule(
        name="financial_factor_mining",
        pattern=r"\b(?:mining|discovering) factors?\b.*\b(?:stock|financial|exchange)\b|\bstock exchange analysis\b.*\bfactors?\b",
        aliases=(
            "large language model alpha factor mining stock market",
            "LLM quantitative factor discovery financial signals",
            "language model alpha signal generation quantitative finance",
        ),
        anchor_type="domain_entity",
        plan_type="domain_alias",
        priority=98,
        reason="Maps factor mining to alpha-factor discovery and quantitative-signal generation.",
    ),
    AliasRule(
        name="vlm_game_agent",
        pattern=r"\bvision[- ]language models?\b.*\bagents?\b.*\b(?:pc )?games?\b|\bagents?\b.*\bplay\b.*\bgames?\b",
        aliases=(
            "vision-language agent PC game playing",
            "multimodal VLM agent computer game control",
            "vision-language-action agent game playing",
        ),
        anchor_type="task",
        plan_type="task_method",
        priority=99,
        reason="Preserves VLM, agent, game-playing, and computer-control constraints.",
    ),
)


def _normalize_key(text: str) -> str:
    text = text.casefold().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_space(value)
        key = _normalize_key(cleaned)
        if not key or key in seen:
            continue
        output.append(cleaned)
        seen.add(key)
    return output


def _anchors_by_type(anchors: Iterable[AnchorEvidence]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for anchor in anchors:
        bucket = output.setdefault(anchor.anchor_type, [])
        if anchor.text not in bucket:
            bucket.append(anchor.text)
    return output


def _filter_metadata(anchors: Iterable[AnchorEvidence]) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "negative_constraints": [],
        "temporal_constraints": [],
    }
    for anchor in anchors:
        if anchor.anchor_type == "negative_constraint":
            normalized = anchor.metadata.get("normalized_value", anchor.normalized_text)
            if normalized not in filters["negative_constraints"]:
                filters["negative_constraints"].append(normalized)
        elif anchor.anchor_type == "temporal_constraint":
            filters["temporal_constraints"].append(
                {
                    "text": anchor.text,
                    "relation": anchor.metadata.get("relation"),
                    "years": list(anchor.metadata.get("years", [])),
                }
            )
    return filters


class AcademicQueryPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        if self.config.max_queries < 1:
            raise ValueError("max_queries must be at least 1.")

    def plan(self, qid: str, question: str) -> AcademicQueryPlan:
        qid = str(qid or "").strip()
        question = normalize_space(question)
        if not qid:
            raise ValueError("qid must be non-empty.")
        if not question:
            raise ValueError("question must be non-empty.")

        extraction = extract_anchors(question)
        anchors = extraction.anchors
        candidates: list[PlannedQuery] = []
        derived_aliases: list[str] = []

        candidates.extend(self._title_candidates(anchors))
        candidates.extend(self._rule_candidates(question, anchors, derived_aliases))
        candidates.extend(self._anchor_composition_candidates(question, anchors))

        if self.config.include_keyword_fallback:
            fallback = self._keyword_fallback(question, anchors)
            if fallback is not None:
                candidates.append(fallback)

        selected = self._deduplicate_and_select(candidates)
        return AcademicQueryPlan(
            qid=qid,
            question=question,
            anchors=anchors,
            derived_aliases=_unique(derived_aliases),
            filters=_filter_metadata(anchors),
            planned_queries=selected,
        )

    def _title_candidates(self, anchors: list[AnchorEvidence]) -> list[PlannedQuery]:
        output: list[PlannedQuery] = []
        for anchor in anchors:
            if anchor.anchor_type != "paper_title":
                continue
            output.append(
                PlannedQuery(
                    text=anchor.text,
                    plan_type="title_anchor",
                    priority=110,
                    confidence=1.0,
                    reason="Uses an explicit paper-title anchor from the query.",
                    anchor_texts=(anchor.text,),
                    anchor_types=("paper_title",),
                )
            )
        return output

    def _rule_candidates(
        self,
        question: str,
        anchors: list[AnchorEvidence],
        derived_aliases: list[str],
    ) -> list[PlannedQuery]:
        output: list[PlannedQuery] = []
        for rule in ALIAS_RULES:
            if re.search(rule.pattern, question, flags=re.IGNORECASE) is None:
                continue
            aliases = _unique(rule.aliases)
            derived_aliases.extend(aliases)
            related = self._related_anchor_texts(anchors, rule.anchor_type)
            for offset, alias in enumerate(aliases[:2]):
                output.append(
                    PlannedQuery(
                        text=alias,
                        plan_type=rule.plan_type,
                        priority=rule.priority - offset,
                        confidence=max(0.75, 0.99 - 0.03 * offset),
                        reason=rule.reason,
                        anchor_texts=tuple(_unique(related + [alias])),
                        anchor_types=tuple(
                            _unique(
                                [rule.anchor_type]
                                + [anchor.anchor_type for anchor in anchors]
                            )
                        ),
                    )
                )
        return output

    @staticmethod
    def _related_anchor_texts(
        anchors: list[AnchorEvidence],
        preferred_type: str,
    ) -> list[str]:
        preferred = [
            anchor.text for anchor in anchors if anchor.anchor_type == preferred_type
        ]
        if preferred:
            return preferred
        return [
            anchor.text
            for anchor in anchors
            if anchor.anchor_type
            in {"benchmark", "method", "model", "task", "modality", "named_entity"}
        ][:6]

    def _anchor_composition_candidates(
        self,
        question: str,
        anchors: list[AnchorEvidence],
    ) -> list[PlannedQuery]:
        by_type = _anchors_by_type(anchors)
        benchmarks = by_type.get("benchmark", [])
        methods = by_type.get("method", [])
        models = by_type.get("model", [])
        tasks = by_type.get("task", [])
        modalities = by_type.get("modality", [])
        named = by_type.get("named_entity", [])
        comparisons = [
            anchor for anchor in anchors if anchor.anchor_type == "comparison"
        ]
        output: list[PlannedQuery] = []

        if comparisons and benchmarks:
            terms = benchmarks + [comparison.text for comparison in comparisons]
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="benchmark_comparison",
                        priority=94,
                        confidence=0.93,
                        reason="Combines explicit benchmark anchors with the comparison relation.",
                        anchor_texts=tuple(_unique(terms)),
                        anchor_types=("benchmark", "comparison"),
                    )
                )

        if benchmarks:
            keywords = extract_keywords(question, max_terms=8)
            terms = benchmarks + tasks + keywords
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="benchmark_aware",
                        priority=88,
                        confidence=0.88,
                        reason="Promotes explicit benchmark names and retains the core task terms.",
                        anchor_texts=tuple(_unique(benchmarks + tasks)),
                        anchor_types=tuple(
                            item for item in ("benchmark", "task") if by_type.get(item)
                        ),
                    )
                )

        if methods and tasks:
            terms = methods + tasks + models + modalities
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="task_method",
                        priority=86,
                        confidence=0.87,
                        reason="Keeps method and task anchors together instead of issuing broad independent searches.",
                        anchor_texts=tuple(_unique(terms)),
                        anchor_types=tuple(
                            item
                            for item in ("method", "task", "model", "modality")
                            if by_type.get(item)
                        ),
                    )
                )

        elif models and tasks:
            terms = models + tasks + modalities
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="model_task",
                        priority=84,
                        confidence=0.85,
                        reason="Combines model and task anchors into a constrained academic query.",
                        anchor_texts=tuple(_unique(terms)),
                        anchor_types=tuple(
                            item
                            for item in ("model", "task", "modality")
                            if by_type.get(item)
                        ),
                    )
                )

        elif tasks:
            keywords = extract_keywords(clean_question_text(question), max_terms=8)
            terms = tasks + modalities + named + keywords
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="task_focus",
                        priority=78,
                        confidence=0.79,
                        reason="Uses the strongest task and modality anchors with cleaned technical keywords.",
                        anchor_texts=tuple(_unique(tasks + modalities + named)),
                        anchor_types=tuple(
                            item
                            for item in ("task", "modality", "named_entity")
                            if by_type.get(item)
                        ),
                    )
                )

        elif methods or models:
            keywords = extract_keywords(clean_question_text(question), max_terms=8)
            terms = methods + models + named + keywords
            text = compact_query_terms(terms, limit=self.config.max_terms_per_query)
            if text:
                output.append(
                    PlannedQuery(
                        text=text,
                        plan_type="method_model",
                        priority=76,
                        confidence=0.77,
                        reason="Uses method/model anchors and removes prompt-style filler.",
                        anchor_texts=tuple(_unique(methods + models + named)),
                        anchor_types=tuple(
                            item
                            for item in ("method", "model", "named_entity")
                            if by_type.get(item)
                        ),
                    )
                )

        return output

    def _keyword_fallback(
        self,
        question: str,
        anchors: list[AnchorEvidence],
    ) -> PlannedQuery | None:
        cleaned = clean_question_text(question)
        keywords = extract_keywords(cleaned, max_terms=self.config.max_terms_per_query)
        anchor_terms = [
            anchor.text
            for anchor in anchors
            if anchor.anchor_type
            not in {"negative_constraint", "temporal_constraint", "comparison"}
        ]
        text = compact_query_terms(anchor_terms + keywords, limit=self.config.max_terms_per_query)
        if not text:
            return None
        return PlannedQuery(
            text=text,
            plan_type="keyword_fallback",
            priority=40,
            confidence=0.55,
            reason="Fallback query built only from cleaned technical keywords; used when no stronger plan fills the budget.",
            anchor_texts=tuple(_unique(anchor_terms)),
            anchor_types=tuple(
                _unique(anchor.anchor_type for anchor in anchors)
            ),
        )

    def _deduplicate_and_select(
        self,
        candidates: list[PlannedQuery],
    ) -> list[PlannedQuery]:
        ordered = sorted(
            candidates,
            key=lambda item: (-item.priority, -item.confidence, item.text.casefold()),
        )
        selected: list[PlannedQuery] = []
        seen: set[str] = set()
        for candidate in ordered:
            key = _normalize_key(candidate.text)
            if not key or key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= self.config.max_queries:
                break
        return selected


def summarize_plans(plans: Iterable[AcademicQueryPlan]) -> dict[str, Any]:
    plans = list(plans)
    plan_types = Counter(
        query.plan_type for plan in plans for query in plan.planned_queries
    )
    anchor_types = Counter(
        anchor.anchor_type for plan in plans for anchor in plan.anchors
    )
    query_counts = Counter(len(plan.planned_queries) for plan in plans)
    return {
        "query_count": len(plans),
        "planned_query_count": sum(len(plan.planned_queries) for plan in plans),
        "estimated_api_calls": sum(
            query.estimated_api_calls
            for plan in plans
            for query in plan.planned_queries
        ),
        "queries_with_zero_plans": query_counts.get(0, 0),
        "queries_with_one_plan": query_counts.get(1, 0),
        "queries_with_two_plans": query_counts.get(2, 0),
        "plan_type_counts": dict(sorted(plan_types.items())),
        "anchor_type_counts": dict(sorted(anchor_types.items())),
        "api_calls_executed": 0,
        "production_retrieval_uses_gold": False,
    }
