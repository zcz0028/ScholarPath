from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "prepare_day12_qe_retrieval_ablation.py"
)
RUNNER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_day12_query_expansion_retrieval_ablation.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_qe_plan_preserves_schema_and_replaces_text() -> None:
    module = load(PREPARE_SCRIPT, "qe_prepare")

    source = {
        "qid": "q1",
        "question": "image encoding distributions",
        "anchors": [{"text": "image encoding distributions"}],
        "planned_queries": [
            {
                "text": "old query",
                "plan_type": "anchor",
                "reason": "old",
                "estimated_api_calls": 1,
                "extra": "preserve",
            },
            {
                "text": "second query",
                "plan_type": "fallback",
                "reason": "second",
                "estimated_api_calls": 1,
            },
        ],
    }

    result = module.build_qe_plan_record(
        source,
        expanded_query="image compression latent representation",
    )

    assert result["qid"] == "q1"
    assert result["anchors"] == source["anchors"]
    assert len(result["planned_queries"]) == 1
    plan = result["planned_queries"][0]
    assert plan["text"] == "image compression latent representation"
    assert plan["plan_type"] == "day12_qe_residual"
    assert plan["estimated_api_calls"] == 1
    assert plan["extra"] == "preserve"


def test_prepare_qe_plans_targets_retrieval_misses_only() -> None:
    module = load(PREPARE_SCRIPT, "qe_prepare_scope")

    source_plans = [
        {
            "qid": "q1",
            "question": "one",
            "planned_queries": [{"text": "old"}],
        },
        {
            "qid": "q2",
            "question": "two",
            "planned_queries": [{"text": "old"}],
        },
    ]
    residual = [
        {"qid": "q1", "b4_failure_type": "retrieval_miss"},
        {"qid": "q2", "b4_failure_type": "ranking_loss_top100"},
    ]
    expansions = [
        {"qid": "q1", "expanded_query": "expanded one"},
        {"qid": "q2", "expanded_query": "expanded two"},
    ]

    result = module.prepare_qe_plans(
        source_plans=source_plans,
        residual_failures=residual,
        expansion_cases=expansions,
    )

    assert len(result) == 1
    assert result[0]["qid"] == "q1"
    assert result[0]["planned_queries"][0]["text"] == "expanded one"


def test_compare_summaries_detects_positive_candidate() -> None:
    module = load(RUNNER_SCRIPT, "qe_runner")

    def summary(tp20, tp50, tp100, p20, r100, f100):
        return {
            "evaluation": {
                "top20_strict": {
                    "counts": {"tp": tp20},
                    "macro": {
                        "precision": p20,
                        "recall": 0.08,
                        "f1": 0.04,
                    },
                    "micro": {
                        "precision": 0.02,
                        "recall": 0.07,
                        "f1": 0.03,
                    },
                },
                "top50_strict": {
                    "counts": {"tp": tp50},
                    "macro": {
                        "precision": 0.03,
                        "recall": 0.14,
                        "f1": 0.05,
                    },
                    "micro": {
                        "precision": 0.03,
                        "recall": 0.10,
                        "f1": 0.05,
                    },
                },
                "top100_strict": {
                    "counts": {"tp": tp100},
                    "macro": {
                        "precision": 0.02,
                        "recall": r100,
                        "f1": f100,
                    },
                    "micro": {
                        "precision": 0.02,
                        "recall": 0.13,
                        "f1": 0.036,
                    },
                },
            },
            "retrieval": {
                "actual_api_calls": 9,
                "cache_hits": 0,
                "retries": 0,
            },
            "estimated_cost_usd": 0.009,
            "latency_ms": {},
        }

    baseline = summary(
        58, 81, 105, 0.058, 0.1675, 0.0342
    )
    experiment = summary(
        60, 84, 112, 0.057, 0.1800, 0.0370
    )

    result = module.compare_summaries(
        baseline,
        experiment,
    )

    assert result["metrics_by_k"]["100"]["delta"]["tp"] == 7
    assert result["adoption_checks"]["top100_tp_gain_positive"] is True
    assert result["adoption_checks"]["top100_recall_gain_positive"] is True
    assert result["adoption_checks"]["top100_f1_gain_positive"] is True
    assert (
        result["adoption_checks"][
            "top20_precision_not_materially_degraded"
        ]
        is True
    )
    assert result["adopt_candidate"] is True
    assert result["decision"] == "candidate_for_e3_integration"


def test_compare_summaries_rejects_no_gain() -> None:
    module = load(RUNNER_SCRIPT, "qe_runner_reject")

    baseline = {
        "evaluation": {
            "top20_strict": {
                "counts": {"tp": 58},
                "macro": {
                    "precision": 0.058,
                    "recall": 0.08,
                    "f1": 0.04,
                },
                "micro": {
                    "precision": 0.02,
                    "recall": 0.07,
                    "f1": 0.03,
                },
            },
            "top50_strict": {
                "counts": {"tp": 81},
                "macro": {
                    "precision": 0.03,
                    "recall": 0.14,
                    "f1": 0.05,
                },
                "micro": {
                    "precision": 0.03,
                    "recall": 0.10,
                    "f1": 0.05,
                },
            },
            "top100_strict": {
                "counts": {"tp": 105},
                "macro": {
                    "precision": 0.02,
                    "recall": 0.1675,
                    "f1": 0.0342,
                },
                "micro": {
                    "precision": 0.02,
                    "recall": 0.13,
                    "f1": 0.036,
                },
            },
        }
    }
    experiment = {
        "evaluation": baseline["evaluation"],
        "retrieval": {
            "actual_api_calls": 9,
            "cache_hits": 0,
            "retries": 0,
        },
        "estimated_cost_usd": 0.009,
        "latency_ms": {},
    }

    result = module.compare_summaries(
        baseline,
        experiment,
    )

    assert result["adopt_candidate"] is False
    assert result["decision"] == "keep_frozen_pipeline"
    assert (
        result["efficiency"]["api_calls_per_added_top100_tp"]
        is None
    )
