from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Iterable, Mapping


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1.")
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[rank - 1]


def build_run_summary(
    query_logs: Iterable[Mapping[str, Any]],
    *,
    wall_clock_ms: float,
) -> dict[str, Any]:
    logs = list(query_logs)
    successful = [item for item in logs if not item.get("error")]
    failed = [item for item in logs if item.get("error")]
    latencies = [
        float(item.get("end_to_end_latency_ms", 0.0))
        for item in logs
    ]
    request_latencies = [
        float(item.get("request_latency_ms", 0.0))
        for item in logs
    ]

    return {
        "queries": {
            "total": len(logs),
            "succeeded": len(successful),
            "failed": len(failed),
        },
        "retrieval": {
            "actual_api_calls": sum(
                int(item.get("actual_api_calls", 0)) for item in logs
            ),
            "cache_hits": sum(int(item.get("cache_hits", 0)) for item in logs),
            "retries": sum(int(item.get("retries", 0)) for item in logs),
            "raw_results": sum(
                int(item.get("raw_result_count", 0)) for item in logs
            ),
            "response_bytes": sum(
                int(item.get("response_bytes", 0)) for item in logs
            ),
        },
        "tokens": {
            "input": sum(int(item.get("input_tokens", 0)) for item in logs),
            "output": sum(int(item.get("output_tokens", 0)) for item in logs),
            "total": sum(int(item.get("total_tokens", 0)) for item in logs),
            "note": "B0 uses no LLM, so token usage should remain zero.",
        },
        "estimated_cost_usd": sum(
            float(item.get("estimated_api_cost_usd", 0.0)) for item in logs
        ),
        "latency_ms": {
            "wall_clock": float(wall_clock_ms),
            "mean_per_query": fmean(latencies) if latencies else 0.0,
            "p50_per_query": percentile(latencies, 0.50),
            "p95_per_query": percentile(latencies, 0.95),
            "max_per_query": max(latencies, default=0.0),
            "mean_http_request": (
                fmean(request_latencies) if request_latencies else 0.0
            ),
        },
        "failed_qids": [str(item.get("qid")) for item in failed],
    }
