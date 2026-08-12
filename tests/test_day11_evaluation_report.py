from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_day11_evaluation_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("day11_eval_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_all(module):
    retrieval = module.build_retrieval_summary(
        module.read_json(module.B0_SUMMARY),
        module.read_json(module.B3_SUMMARY),
        module.read_json(module.B4_SUMMARY),
        module.read_json(module.DAY4_SUMMARY),
    )
    ranking = module.build_ranking_summary(module.read_json(module.DAY8_ABLATION))
    explainability = module.build_explainability_summary(
        module.read_jsonl(module.DAY8_E3),
        module.read_json(module.DAY9_CITATION),
    )
    competition = module.build_competition_metrics(retrieval, ranking, explainability)
    return retrieval, ranking, explainability, competition


def test_day11_frozen_artifacts_exist() -> None:
    m = load_module()
    required = [m.B0_SUMMARY, m.B3_SUMMARY, m.B4_SUMMARY, m.DAY4_SUMMARY, m.DAY8_ABLATION, m.DAY8_E3, m.DAY9_CITATION]
    assert all(path.exists() for path in required)


def test_day11_headline_metrics_match_frozen_baselines() -> None:
    m = load_module()
    _, _, _, competition = build_all(m)
    h = competition["headline_metrics"]
    assert h["b0_top100_tp"] == 44
    assert h["b4_top100_tp"] == 61
    assert h["day4_top100_tp"] == 105
    assert abs(h["day8_baseline_ndcg_at_10"] - 0.19291929976021374) < 1e-12
    assert abs(h["day8_e3_ndcg_at_10"] - 0.2488522823567324) < 1e-12
    assert abs(h["day8_relative_ndcg_improvement"] - 0.2899294299017245) < 1e-12
    assert abs(h["constraint_evidence_match_rate_at_100"] - 0.13548223350253807) < 1e-12
    assert abs(h["paper_evidence_support_rate_at_100"] - 0.4882) < 1e-12
    assert h["reason_generation_rate_at_100"] == 1.0
    assert abs(h["evidence_backed_reason_rate_at_100"] - 0.4882) < 1e-12
    assert h["abstract_availability_rate_at_100"] == 0.0
    assert h["citation_case_study_coverage"] == 0.95


def test_day11_reporting_boundaries_are_preserved() -> None:
    m = load_module()
    _, _, explainability, competition = build_all(m)
    citation = explainability["citation_path"]
    assert citation["scope"] == "single_query_case_study"
    assert citation["query_count"] == 1
    assert citation["total_results"] == 20
    assert citation["total_paths"] == 19
    assert citation["coverage"] == 0.95
    assert "must not be reported" in citation["reporting_note"].lower()
    rules = " ".join(competition["reporting_rules"]).lower()
    assert "single-query case study" in rules
    assert "alpha=1.0, beta=0.0" in rules


def test_day11_report_contains_competition_conclusions() -> None:
    m = load_module()
    retrieval, ranking, explainability, competition = build_all(m)
    report = m.build_report_markdown(retrieval, ranking, explainability, competition)
    assert "B4 Top100 TP increases from **61** to **105**" in report
    assert "**28.99% relative gain**" in report
    assert "**48.82%**" in report
    assert "single-query Day9 case study" in report
    assert "0% abstract availability" in report.lower()


def test_day11_generated_output_is_self_consistent(tmp_path: Path) -> None:
    m = load_module()
    original = m.OUTPUT_DIR
    m.OUTPUT_DIR = tmp_path
    try:
        m.main()
    finally:
        m.OUTPUT_DIR = original
    expected = {
        "evaluation_summary.json",
        "retrieval_ranking_summary.json",
        "explainability_summary.json",
        "competition_metrics.json",
        "retrieval_ablation.csv",
        "ranking_ablation.csv",
        "evaluation_report.md",
    }
    assert expected == {p.name for p in tmp_path.iterdir() if p.is_file()}
    competition = json.loads((tmp_path / "competition_metrics.json").read_text(encoding="utf-8"))
    assert competition["benchmark_query_count"] == 50
    assert competition["headline_metrics"]["day4_top100_tp"] == 105
