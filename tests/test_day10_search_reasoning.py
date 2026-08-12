from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.schemas import (
    CostSummary,
    PipelineSummary,
    SearchReasoning,
    SearchResponse,
)


def test_search_reasoning_defaults_are_isolated() -> None:
    first = SearchReasoning(
        original_query="papers about multimodal large language models",
        cleaned_query="multimodal large language models",
    )
    second = SearchReasoning(
        original_query="papers about event extraction",
        cleaned_query="event extraction",
    )

    first.constraints.append(
        {
            "id": "c1",
            "text": "multimodal",
            "constraint_type": "task_or_modality",
        }
    )
    first.derived_aliases.append("vision language model")
    first.filters["temporal_constraints"] = [{"relation": "after", "years": [2023]}]
    first.execution.append({"name": "query_planner", "status": "completed"})

    assert second.constraints == []
    assert second.candidate_subqueries == []
    assert second.academic_anchors == []
    assert second.derived_aliases == []
    assert second.filters == {}
    assert second.selected_plans == []
    assert second.execution == []


def test_search_reasoning_accepts_structured_trace() -> None:
    reasoning = SearchReasoning(
        original_query="Find papers about trigger-free event extraction",
        cleaned_query="trigger free event extraction",
        constraints=[
            {
                "id": "c1_trigger_free",
                "text": "trigger free",
                "constraint_type": "method_or_property",
                "terms": ["trigger", "free"],
                "weight": 1.25,
            }
        ],
        candidate_subqueries=[
            {
                "text": "trigger free event extraction",
                "subquery_type": "constraint_core",
                "constraint_ids": ["c1_trigger_free"],
                "reason": "Primary constraint combined with major secondary constraints.",
            }
        ],
        academic_anchors=[
            {
                "text": "trigger-free event extraction",
                "anchor_type": "method",
            }
        ],
        derived_aliases=[
            "trigger-free document-level event extraction",
        ],
        filters={
            "negative_constraints": [],
            "temporal_constraints": [],
        },
        selected_plans=[
            {
                "text": "trigger-free document-level event extraction",
                "plan_type": "task_method",
                "priority": 99,
                "confidence": 0.99,
                "reason": (
                    "Promotes the trigger-free and no-trigger-annotation "
                    "constraints instead of generic event extraction."
                ),
                "anchor_texts": ["trigger-free event extraction"],
                "anchor_types": ["method"],
                "estimated_api_calls": 1,
            }
        ],
        execution=[
            {
                "name": "academic_query_planning",
                "status": "completed",
                "count": 1,
            },
            {
                "name": "day8_e3_rerank",
                "status": "completed",
                "variant": "E3",
                "alpha": 1.0,
                "beta": 0.0,
            },
        ],
    )

    payload = reasoning.model_dump()

    assert payload["original_query"] == (
        "Find papers about trigger-free event extraction"
    )
    assert payload["cleaned_query"] == "trigger free event extraction"

    assert payload["constraints"][0]["constraint_type"] == "method_or_property"
    assert payload["candidate_subqueries"][0]["subquery_type"] == "constraint_core"

    assert payload["academic_anchors"][0]["anchor_type"] == "method"
    assert payload["derived_aliases"] == [
        "trigger-free document-level event extraction"
    ]

    assert payload["selected_plans"][0]["plan_type"] == "task_method"
    assert payload["selected_plans"][0]["confidence"] == 0.99

    assert payload["execution"][1]["name"] == "day8_e3_rerank"


def test_search_response_remains_backward_compatible_without_reasoning() -> None:
    response = SearchResponse(
        run_id="bench_test",
        query="papers about event extraction",
        qid="RealScholarQuery_test",
        mode="benchmark",
        parsed_constraints=[],
        academic_anchors=[],
        query_plan=[],
        results=[],
        pipeline=PipelineSummary(
            stages=[
                {
                    "name": "day8_e3_rerank",
                    "status": "completed",
                    "variant": "E3",
                }
            ],
            total_candidates=0,
            returned_results=0,
        ),
        cost=CostSummary(),
        latency_ms=0,
        warnings=[],
    )

    payload = response.model_dump()

    assert "reasoning" in payload
    assert payload["reasoning"] is None

    assert payload["parsed_constraints"] == []
    assert payload["academic_anchors"] == []
    assert payload["query_plan"] == []

    assert payload["pipeline"]["stages"][0]["name"] == "day8_e3_rerank"


def test_search_response_can_carry_reasoning_without_changing_legacy_fields() -> None:
    reasoning = SearchReasoning(
        original_query="papers about large language models",
        cleaned_query="large language models",
        constraints=[
            {
                "id": "c1_large_language_model",
                "text": "large language model",
                "constraint_type": "model_or_entity",
            }
        ],
        selected_plans=[
            {
                "text": "large language model",
                "plan_type": "method_model",
                "reason": "Uses method/model anchors and removes prompt-style filler.",
            }
        ],
        execution=[
            {
                "name": "day8_e3_rerank",
                "status": "completed",
                "variant": "E3",
            }
        ],
    )

    legacy_constraints = [
        {
            "id": "c1_large_language_model",
            "text": "large language model",
            "constraint_type": "model_or_entity",
        }
    ]
    legacy_plan = [
        {
            "text": "large language model",
            "plan_type": "method_model",
            "reason": "Uses method/model anchors and removes prompt-style filler.",
        }
    ]

    response = SearchResponse(
        run_id="live_test",
        query="papers about large language models",
        qid="live_test",
        mode="live",
        reasoning=reasoning,
        parsed_constraints=legacy_constraints,
        academic_anchors=[],
        query_plan=legacy_plan,
        results=[],
        pipeline=PipelineSummary(
            stages=list(reasoning.execution),
            total_candidates=0,
            returned_results=0,
        ),
        cost=CostSummary(api_calls=1),
        latency_ms=10,
        warnings=[],
    )

    payload = response.model_dump()

    assert payload["reasoning"] is not None
    assert payload["reasoning"]["constraints"] == payload["parsed_constraints"]
    assert payload["reasoning"]["selected_plans"] == payload["query_plan"]
    assert payload["reasoning"]["execution"] == payload["pipeline"]["stages"]