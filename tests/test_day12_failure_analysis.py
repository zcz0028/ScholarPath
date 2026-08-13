from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_day12_failure_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_day12_failure_analysis",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_family_preserves_existing_taxonomy() -> None:
    module = load_module()

    assert module.failure_family("success") == "success"
    assert module.failure_family("retrieval_miss") == "retrieval"
    assert module.failure_family("fusion_pool_miss") == "fusion"
    assert module.failure_family("ranking_loss_top20") == "ranking"
    assert module.failure_family("ranking_loss_top50") == "ranking"
    assert module.failure_family("ranking_loss_top100") == "ranking"
    assert (
        module.failure_family("possible_matching_issue")
        == "matching_audit"
    )
    assert module.failure_family("guard_drop") == "post_ranking_filter"
    assert (
        module.failure_family("selector_drop")
        == "post_ranking_filter"
    )
    assert (
        module.failure_family("guard_aware_drop")
        == "post_ranking_filter"
    )


def test_safe_rate_handles_zero_denominator() -> None:
    module = load_module()

    assert module.safe_rate(5, 10) == 0.5
    assert module.safe_rate(0, 10) == 0.0
    assert module.safe_rate(1, 0) == 0.0


def test_recovery_summary() -> None:
    module = load_module()

    recovered = [
        {
            "qid": "q1",
            "baseline_tp": 0,
            "current_tp": 3,
            "tp_delta": 3,
        },
        {
            "qid": "q2",
            "baseline_tp": 0,
            "current_tp": 1,
            "tp_delta": 1,
        },
    ]

    unrecovered = [
        {
            "qid": "q3",
            "baseline_tp": 0,
            "current_tp": 0,
            "remaining_false_negatives": [{}, {}, {}],
        }
    ]

    result = module.build_recovery_summary(
        recovered,
        unrecovered,
    )

    assert result["baseline_zero_recall_query_count"] == 3
    assert result["recovered_query_count"] == 2
    assert result["unrecovered_query_count"] == 1
    assert result["query_recovery_rate"] == 2 / 3
    assert result["recovered_tp_gain"] == 4
    assert result["current_tp_on_recovered_queries"] == 4
    assert (
        result["remaining_false_negative_count_on_unrecovered_queries"]
        == 3
    )


def test_robustness_summary_keeps_reporting_boundaries() -> None:
    module = load_module()

    day8 = {
        "baseline": {"mean_ndcg_at_10": 0.19},
        "recommended": {
            "variant": "E3",
            "alpha": 1.0,
            "beta": 0.0,
            "mean_ndcg_at_10": 0.25,
        },
    }

    day9 = {
        "query_count": 1,
        "top_k": 20,
        "coverage": 0.95,
        "failed_paths": 1,
    }

    day11 = {
        "competition_metrics": {
            "headline_metrics": {
                "day8_relative_ndcg_improvement": 0.29,
                "constraint_evidence_match_rate_at_100": 0.13,
                "paper_evidence_support_rate_at_100": 0.48,
                "reason_generation_rate_at_100": 1.0,
                "evidence_backed_reason_rate_at_100": 0.48,
                "abstract_availability_rate_at_100": 0.0,
            }
        }
    }

    recovery = {
        "baseline_zero_recall_query_count": 10,
        "recovered_query_count": 7,
        "unrecovered_query_count": 3,
        "query_recovery_rate": 0.7,
    }

    result = module.build_robustness_summary(
        day8=day8,
        day9=day9,
        day11=day11,
        recovery_summary=recovery,
    )

    assert result["offline_only"] is True
    assert result["network_calls"] == 0
    assert result["llm_calls"] == 0
    assert result["ranking_modified"] is False

    assert result["ranking"]["selected_variant"] == "E3"
    assert result["ranking"]["selected_alpha"] == 1.0
    assert result["ranking"]["selected_beta"] == 0.0

    assert result["citation"]["scope"] == "single_query_case_study"
    assert result["citation"]["coverage"] == 0.95

    boundaries = " ".join(result["reporting_boundaries"])
    assert "single-query" in boundaries
    assert "alpha=1.0" in boundaries
    assert "beta=0.0" in boundaries


def test_report_contains_required_sections() -> None:
    module = load_module()

    failure = {
        "failure_type_counts": {
            "retrieval_miss": 3,
            "success": 47,
        },
        "failure_family_counts": {
            "retrieval": 3,
            "success": 47,
        },
    }

    recovery = {
        "baseline_zero_recall_query_count": 10,
        "recovered_query_count": 7,
        "unrecovered_query_count": 3,
        "query_recovery_rate": 0.7,
        "recovered_tp_gain": 12,
    }

    residual = [
        {
            "qid": "q1",
            "question": "example question",
            "b4_failure_type": "retrieval_miss",
            "priority": "P1",
            "remaining_false_negative_count": 5,
        }
    ]

    robustness = {
        "ranking": {
            "baseline_ndcg_at_10": 0.19,
            "selected_variant": "E3",
            "selected_ndcg_at_10": 0.25,
            "relative_ndcg_improvement": 0.29,
        },
        "explainability": {
            "constraint_evidence_match_rate_at_100": 0.13,
            "paper_evidence_support_rate_at_100": 0.48,
            "reason_generation_rate_at_100": 1.0,
            "evidence_backed_reason_rate_at_100": 0.48,
            "abstract_availability_rate_at_100": 0.0,
        },
        "citation": {
            "query_count": 1,
            "top_k": 20,
            "coverage": 0.95,
            "failed_paths": 1,
        },
    }

    report = module.build_report(
        failure,
        recovery,
        residual,
        robustness,
    )

    assert "Pipeline Failure Taxonomy" in report
    assert "Day4 Recovery" in report
    assert "Residual Failure Cases" in report
    assert "Robustness View" in report
    assert "Competition-safe Interpretation" in report
    assert "single-query case study" in report