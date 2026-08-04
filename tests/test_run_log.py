from scholarpath.observability.run_log import build_run_summary, percentile


def test_percentile_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 0.50) == 2
    assert percentile([1, 2, 3, 4], 0.95) == 4
    assert percentile([], 0.95) == 0.0


def test_build_run_summary() -> None:
    summary = build_run_summary(
        [
            {
                "qid": "q1",
                "actual_api_calls": 1,
                "cache_hits": 0,
                "retries": 0,
                "raw_result_count": 100,
                "response_bytes": 500,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_api_cost_usd": 0.001,
                "request_latency_ms": 100,
                "end_to_end_latency_ms": 110,
                "error": None,
            },
            {
                "qid": "q2",
                "actual_api_calls": 0,
                "cache_hits": 1,
                "retries": 0,
                "raw_result_count": 100,
                "response_bytes": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_api_cost_usd": 0.0,
                "request_latency_ms": 0,
                "end_to_end_latency_ms": 5,
                "error": None,
            },
        ],
        wall_clock_ms=120,
    )
    assert summary["queries"]["succeeded"] == 2
    assert summary["retrieval"]["actual_api_calls"] == 1
    assert summary["retrieval"]["cache_hits"] == 1
    assert summary["tokens"]["total"] == 0
    assert summary["estimated_cost_usd"] == 0.001
