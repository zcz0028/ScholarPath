from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.settings import ApiSettings


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    write_jsonl(
        tmp_path / "data/raw/RealScholarQuery/test.jsonl",
        [
            {
                "qid": "RealScholarQuery_1",
                "question": "papers about large language model in-context learning",
                "answer": ["SECRET GOLD TITLE"],
                "answer_arxiv_id": ["9999.99999"],
                "source_meta": {"published_time": "20241001"},
            },
            {
                "qid": "RealScholarQuery_2",
                "question": "papers about video understanding",
                "answer": [],
                "source_meta": {"published_time": "20241001"},
            },
        ],
    )
    write_jsonl(
        tmp_path / "outputs/week2_day3_query_plans/query_plans.jsonl",
        [
            {
                "qid": "RealScholarQuery_1",
                "question": "papers about large language model in-context learning",
                "anchors": [{"text": "in-context learning", "anchor_type": "method"}],
                "planned_queries": [
                    {
                        "text": "large language model in-context learning",
                        "plan_type": "task_method",
                        "reason": "test",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        tmp_path / "outputs/week2_day1_diagnostics/anchor_inventory.jsonl",
        [{"qid": "RealScholarQuery_1", "anchors": [{"text": "in-context learning", "anchor_type": "method"}]}],
    )
    paper = {
        "title": "A Relevant Paper",
        "authors": ["Alice"],
        "year": 2023,
        "venue": "TestConf",
        "doi": "10.1000/test",
        "openalex_id": "W123",
        "url": "https://example.test/paper",
        "raw": {
            "b3_final_score": 0.82,
            "day4_reason_tags": ["anchor_rescue_source"],
            "b4_sources": ["b4_base", "day4_rescue"],
            "b5_1_guard_reason": "约束满足",
        },
    }
    for top_k in (20, 50, 100):
        write_jsonl(
            tmp_path / f"outputs/week2_day4_rescue/predictions_top{top_k}.jsonl",
            [{"qid": "RealScholarQuery_1", "question": "papers about large language model in-context learning", "papers": [paper]}],
        )
    write_json(
        tmp_path / "outputs/week2_day4_rescue/execution_manifest.json",
        {"target_qids": ["RealScholarQuery_1"]},
    )
    write_json(
        tmp_path / "outputs/week2_day4_rescue/run_summary.json",
        {
            "comparison_to_b4": {
                "baseline_top100_tp": 61,
                "day4_top100_tp": 105,
                "top100_tp_delta": 44,
                "baseline_zero_recall_count": 26,
                "day4_zero_recall_count": 11,
                "recovered_query_count": 15,
            }
        },
    )
    write_jsonl(tmp_path / "outputs/week2_day4_rescue/query_logs.jsonl", [])
    write_jsonl(
        tmp_path / "outputs/week2_day5_citation/selected_seeds.jsonl",
        [{"qid": "RealScholarQuery_1", "question": "x", "seeds": [{"paper": {"title": "Seed"}}]}],
    )
    write_jsonl(
        tmp_path / "outputs/week2_day5_citation/citation_paths.jsonl",
        [
            {
                "qid": "RealScholarQuery_1",
                "seed_openalex_id": "W9",
                "seed_title": "Seed",
                "expanded_openalex_id": "W123",
                "expanded_title": "A Relevant Paper",
                "edge_type": "cited_by",
                "hop": 1,
            }
        ],
    )
    write_json(
        tmp_path / "outputs/week2_day5_citation/run_summary.json",
        {
            "target_query_count": 9,
            "selected_seed_count": 16,
            "actual_api_calls": 23,
            "cache_hits": 17,
            "raw_citation_candidates": 350,
            "citation_paths": 350,
            "failed_expansions": 0,
            "skipped_due_to_budget": 0,
        },
    )
    write_json(
        tmp_path / "configs/week2/frozen_baselines.json",
        {
            "baselines": [
                {"name": "b5_1_guard_balanced_v4", "role": "primary_recommendation", "path": "outputs/b5_1_guard_balanced_v4"},
                {"name": "b5_on_b4_precision", "role": "high_precision", "path": "outputs/b5_on_b4_precision"},
                {"name": "b5_2_guard_aware_balanced", "role": "guard_aware_balanced", "path": "outputs/b5_2_guard_aware_balanced"},
            ]
        },
    )
    settings = ApiSettings(project_root=tmp_path)
    return TestClient(create_app(settings))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system(client: TestClient) -> None:
    payload = client.get("/api/system").json()
    assert payload["project"] == "ScholarPath"
    assert payload["benchmarks_loaded"] is True
    assert "benchmark" in payload["available_modes"]


def test_query_list_does_not_expose_gold(client: TestClient) -> None:
    response = client.get("/api/queries")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    serialized = json.dumps(payload)
    assert "SECRET GOLD TITLE" not in serialized
    assert "answer_arxiv_id" not in serialized


def test_benchmark_search(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={
            "query": "papers about large language model in-context learning",
            "qid": "RealScholarQuery_1",
            "mode": "benchmark",
            "top_k": 20,
            "enable_citation": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "benchmark"
    assert payload["results"][0]["title"] == "A Relevant Paper"
    assert payload["results"][0]["citation_path"]["seed_title"] == "Seed"
    assert payload["results"][0]["constraint_count"] >= 1
    assert isinstance(payload["results"][0]["constraint_evidence"], list)
    assert payload["results"][0]["reason_text"] is not None
    assert any(stage["name"] == "day8_e3_rerank" for stage in payload["pipeline"]["stages"])
    assert payload["cost"]["api_calls"] == 0
    assert "SECRET GOLD TITLE" not in json.dumps(payload)


def test_benchmark_search_requires_known_qid(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"query": "some academic query", "qid": "missing", "mode": "benchmark", "top_k": 20},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "QUERY_NOT_FOUND"


def test_invalid_top_k_returns_structured_422(client: TestClient) -> None:
    response = client.post(
        "/api/search",
        json={"query": "some academic query", "qid": "RealScholarQuery_1", "mode": "benchmark", "top_k": 10},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_diagnosis(client: TestClient) -> None:
    response = client.get("/api/queries/RealScholarQuery_1/diagnosis")
    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"]["day4_rescue_triggered"] is True
    assert payload["pipeline"]["day5_citation_triggered"] is True
    assert payload["pipeline"]["citation_path_count"] == 1
    assert "gold" not in json.dumps(payload).lower()


def test_unknown_diagnosis_returns_404(client: TestClient) -> None:
    response = client.get("/api/queries/unknown/diagnosis")
    assert response.status_code == 404


def test_experiments(client: TestClient) -> None:
    response = client.get("/api/experiments")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert "week2_day4_rescue" in ids
    assert "week2_day5_citation" in ids
    assert "b5_1_guard_balanced_v4" in ids


def test_missing_prediction_artifact_is_structured_error(client: TestClient, tmp_path: Path) -> None:
    # Existing client is already bound to a separate temp tree; this assertion
    # documents the error schema through an unknown Top-K query artifact path.
    response = client.post(
        "/api/search",
        json={"query": "papers about video understanding", "qid": "RealScholarQuery_2", "mode": "benchmark", "top_k": 20},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "QUERY_NOT_FOUND"


def test_live_search_uses_day8_e3_and_deterministic_reason(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from apps.api.services import search_service as module

    class FakePaper:
        def __init__(self, canonical_id: str, title: str) -> None:
            self.canonical_id_v2 = canonical_id
            self._title = title

        def to_dict(self, *, include_raw: bool = False) -> dict:
            payload = {
                "title": self._title,
                "authors": ["Test Author"],
                "year": 2025,
                "openalex_id": self.canonical_id_v2.split(":", 1)[-1],
                "url": "https://example.test/" + self.canonical_id_v2,
            }
            if include_raw:
                payload["raw"] = {"concepts": []}
            return payload

    class FakeRetriever:
        def __init__(self, _config: object) -> None:
            pass

        def search(self, _request: object) -> object:
            return SimpleNamespace(
                papers=[
                    FakePaper("openalex:W1", "A Survey of Large Language Models"),
                    FakePaper("openalex:W2", "Multimodal Large Language Models for Scientific Documents"),
                ],
                raw_result_count=2,
                stats=SimpleNamespace(
                    actual_api_calls=1,
                    cache_hits=0,
                    retries=0,
                    estimated_api_cost_usd=0.001,
                ),
            )

    monkeypatch.setattr(module, "OpenAlexRetriever", FakeRetriever)

    response = client.post(
        "/api/search",
        json={
            "query": "recent papers on multimodal large language models for scientific document understanding",
            "mode": "live",
            "top_k": 20,
            "enable_citation": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "live"
    assert payload["results"]

    first = payload["results"][0]
    assert first["title"] == "Multimodal Large Language Models for Scientific Documents"
    assert first["score"] is not None
    assert first["matched_count"] >= 2
    assert first["constraint_count"] >= first["matched_count"]
    assert "large language model" in first["matched_constraints"]
    assert "multimodal" in first["matched_constraints"]
    assert "model_or_entity_match" in first["reason_tags"]
    assert "task_or_modality_match" in first["reason_tags"]
    assert first["reason_text"]
    assert first["constraint_evidence"]
    assert first["citation_path"] is None

    stage_names = [stage["name"] for stage in payload["pipeline"]["stages"]]
    assert "day8_e3_rerank" in stage_names
    assert "day8_constraint_evidence" in stage_names
    assert "day8_recommendation_reason" in stage_names
