from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.services.artifact_service import ArtifactService
from apps.api.settings import ApiSettings


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_day9_citation_paths_are_indexed_by_qid(tmp_path: Path) -> None:
    output_dir = tmp_path / "week2_day9_citation"
    _write_jsonl(
        output_dir / "citation_paths.jsonl",
        [{
            "qid": "q1",
            "result_openalex_id": "W1",
            "result_rank": 1,
            "seed_openalex_id": "W1",
            "seed_title": "Seed",
            "expanded_openalex_id": "W2",
            "expanded_title": "Expanded",
            "edge_type": "references",
            "hop": 1,
        }],
    )
    settings = ApiSettings(
        project_root=tmp_path,
        day9_citation_dir=Path("week2_day9_citation"),
    )
    service = ArtifactService(settings)
    paths = service.day9_citation_paths()
    assert "q1" in paths
    assert paths["q1"][0]["result_openalex_id"] == "W1"


def test_day9_citation_paths_by_paper(tmp_path: Path) -> None:
    output_dir = tmp_path / "week2_day9_citation"
    _write_jsonl(
        output_dir / "citation_paths.jsonl",
        [{
            "qid": "q1",
            "result_openalex_id": "W1",
            "result_rank": 1,
            "seed_openalex_id": "W1",
            "seed_title": "Seed",
            "expanded_openalex_id": "W2",
            "expanded_title": "Expanded",
            "edge_type": "cited_by",
            "hop": 1,
        }],
    )
    settings = ApiSettings(
        project_root=tmp_path,
        day9_citation_dir=Path("week2_day9_citation"),
    )
    service = ArtifactService(settings)
    paths = service.day9_citation_paths_by_paper("q1")
    assert paths["W1"]["expanded_openalex_id"] == "W2"


def test_missing_day9_artifact_is_safe(tmp_path: Path) -> None:
    settings = ApiSettings(
        project_root=tmp_path,
        day9_citation_dir=Path("missing"),
    )
    service = ArtifactService(settings)
    assert service.day9_citation_paths() == {}
    assert service.day9_citation_paths_by_paper("q1") == {}


def test_day9_paths_override_day5_paths() -> None:
    day5 = {"W1": {"expanded_title": "Day5"}}
    day9 = {"W1": {"expanded_title": "Day9"}}
    merged = {**day5, **day9}
    assert merged["W1"]["expanded_title"] == "Day9"
