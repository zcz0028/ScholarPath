from scholarpath.query.academic_query_planner import (
    AcademicQueryPlanner,
    PlannerConfig,
    summarize_plans,
)


def plan(question: str, max_queries: int = 2):
    planner = AcademicQueryPlanner(PlannerConfig(max_queries=max_queries))
    return planner.plan("q1", question)


def query_texts(result):
    return [item.text.casefold() for item in result.planned_queries]


def test_planner_limits_queries_to_two():
    result = plan(
        "Give me papers that share insights about how large language models "
        "gain in-context learning capability during pretraining."
    )
    assert len(result.planned_queries) == 2
    assert result.to_dict()["estimated_api_calls"] == 2


def test_planner_builds_benchmark_comparison_queries():
    result = plan(
        "Show me code evaluation datasets harder than HumanEval and MBPP, "
        "but easier than code_contests."
    )
    texts = " ".join(query_texts(result))
    assert "humaneval" in texts
    assert "mbpp" in texts
    assert "codecontests" in texts or "code_contests" in texts
    assert result.planned_queries[0].plan_type == "benchmark_comparison"


def test_planner_promotes_trigger_free_constraint():
    result = plan(
        "Find papers on trigger-free document-level event extraction methods "
        "that do not use human-annotated triggers."
    )
    texts = " ".join(query_texts(result))
    assert "trigger-free" in texts or "without trigger annotations" in texts
    assert "document" in texts
    assert "event extraction" in texts


def test_planner_expands_long_video_description():
    result = plan(
        "Show me research on long video description for videos lasting several minutes."
    )
    texts = " ".join(query_texts(result))
    assert "long-form video captioning" in texts
    assert "dense video captioning" in texts


def test_planner_expands_neural_quantum_monte_carlo():
    result = plan("Show me research on neural network based quantum Monte Carlo.")
    texts = " ".join(query_texts(result))
    assert "neural network quantum monte carlo" in texts
    assert "neural quantum states variational monte carlo" in texts


def test_planner_expands_financial_factor_mining():
    result = plan(
        "Papers using large language models for mining factors in stock exchange analysis."
    )
    texts = " ".join(query_texts(result))
    assert "alpha factor" in texts
    assert "quantitative factor" in texts


def test_negative_survey_constraint_is_filter_not_query_text():
    result = plan(
        "Find multimodal visual audio foundation model papers, but exclude surveys."
    )
    assert "exclude_survey" in result.filters["negative_constraints"]
    assert all("exclude survey" not in text for text in query_texts(result))


def test_planner_does_not_read_or_emit_gold():
    result = plan("Find papers about quantized pretraining for LLMs.")
    payload = result.to_dict()
    assert payload["production_retrieval_uses_gold"] is False
    assert payload["analysis_only_uses_gold"] is False
    assert "gold_papers" not in payload


def test_planner_is_deterministic():
    question = "How can LLM agents be evaluated and benchmarked for financial tasks?"
    first = plan(question).to_dict()
    second = plan(question).to_dict()
    assert first == second


def test_summary_counts_plans_and_api_budget():
    first = plan("Show me research on identity preservation video generation.")
    second = plan("Show me research on frame selection for video understanding.")
    summary = summarize_plans([first, second])
    assert summary["query_count"] == 2
    assert summary["planned_query_count"] == 4
    assert summary["estimated_api_calls"] == 4
    assert summary["api_calls_executed"] == 0
