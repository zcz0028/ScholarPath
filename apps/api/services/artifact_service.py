from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.api.settings import ApiSettings


class ArtifactNotFoundError(FileNotFoundError):
    pass


@dataclass(slots=True)
class _CacheEntry:
    mtime_ns: int
    value: Any


class ArtifactService:
    """Loads Week-2 artifacts with mtime-aware in-memory caching."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._cache: dict[Path, _CacheEntry] = {}

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise ArtifactNotFoundError(str(path))
        return path

    def _read_json(self, path: Path, *, required: bool = True) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise ArtifactNotFoundError(str(path))
            return {}
        stat = path.stat()
        cached = self._cache.get(path)
        if cached and cached.mtime_ns == stat.st_mtime_ns:
            return cached.value
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        self._cache[path] = _CacheEntry(stat.st_mtime_ns, value)
        return value

    def _read_jsonl(self, path: Path, *, required: bool = True) -> list[dict[str, Any]]:
        if not path.exists():
            if required:
                raise ArtifactNotFoundError(str(path))
            return []
        stat = path.stat()
        cached = self._cache.get(path)
        if cached and cached.mtime_ns == stat.st_mtime_ns:
            return cached.value
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object JSONL record at {path}:{line_no}")
                rows.append(value)
        self._cache[path] = _CacheEntry(stat.st_mtime_ns, rows)
        return rows

    @staticmethod
    def _index_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            qid = str(row.get("qid") or "").strip()
            if qid and qid not in output:
                output[qid] = row
        return output

    def benchmark_queries(self) -> dict[str, dict[str, Any]]:
        # Raw benchmark data contains question/answer fields, but this method
        # deliberately exposes only qid/question/source_meta to the API layer.
        rows = self._read_jsonl(self.settings.resolved_benchmark_path)
        safe_rows = [
            {
                "qid": str(row.get("qid") or ""),
                "question": str(row.get("question") or "").strip(),
                "source_meta": dict(row.get("source_meta") or {}),
            }
            for row in rows
            if row.get("qid")
        ]
        return self._index_by_qid(safe_rows)

    def query_plans(self) -> dict[str, dict[str, Any]]:
        rows = self._read_jsonl(self.settings.resolved_query_plans_path, required=False)
        return self._index_by_qid(rows)

    def anchor_inventory(self) -> dict[str, dict[str, Any]]:
        rows = self._read_jsonl(self.settings.resolved_anchor_inventory_path, required=False)
        return self._index_by_qid(rows)

    def day4_predictions(self, top_k: int) -> dict[str, dict[str, Any]]:
        path = self.settings.resolved_day4_dir / f"predictions_top{top_k}.jsonl"
        return self._index_by_qid(self._read_jsonl(path))

    def day4_summary(self) -> dict[str, Any]:
        return self._read_json(self.settings.resolved_day4_dir / "run_summary.json", required=False)

    def day4_query_logs(self) -> dict[str, dict[str, Any]]:
        return self._index_by_qid(
            self._read_jsonl(self.settings.resolved_day4_dir / "query_logs.jsonl", required=False)
        )

    def day4_target_qids(self) -> set[str]:
        summary = self.day4_summary()
        values = summary.get("target_qids")
        if isinstance(values, list):
            return {str(x) for x in values}
        manifest = self._read_json(
            self.settings.resolved_day4_dir / "execution_manifest.json",
            required=False,
        )
        values = manifest.get("target_qids")
        return {str(x) for x in values} if isinstance(values, list) else set()

    def day5_summary(self) -> dict[str, Any]:
        return self._read_json(self.settings.resolved_day5_dir / "run_summary.json", required=False)

    def day5_manifest(self) -> dict[str, Any]:
        return self._read_json(
            self.settings.resolved_day5_dir / "execution_manifest.json",
            required=False,
        )

    def day5_seeds(self) -> dict[str, dict[str, Any]]:
        return self._index_by_qid(
            self._read_jsonl(self.settings.resolved_day5_dir / "selected_seeds.jsonl", required=False)
        )

    def day5_citation_paths(self) -> dict[str, list[dict[str, Any]]]:
        rows = self._read_jsonl(
            self.settings.resolved_day5_dir / "citation_paths.jsonl",
            required=False,
        )
        output: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            qid = str(row.get("qid") or "").strip()
            if qid:
                output.setdefault(qid, []).append(row)
        return output

    def day9_citation_paths(self) -> dict[str, list[dict[str, Any]]]:
        rows = self._read_jsonl(
            self.settings.resolved_day9_citation_dir / "citation_paths.jsonl",
            required=False,
        )
        output: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            qid = str(row.get("qid") or "").strip()
            if qid:
                output.setdefault(qid, []).append(row)
        return output

    def day9_citation_paths_by_paper(
        self,
        qid: str,
    ) -> dict[str, dict[str, Any]]:
        paths = self.day9_citation_paths().get(qid, [])
        output: dict[str, dict[str, Any]] = {}
        for path in paths:
            result_id = str(path.get("result_openalex_id") or "").strip()
            if result_id:
                output.setdefault(result_id, path)
        return output

    def frozen_baselines(self) -> dict[str, Any]:
        return self._read_json(self.settings.resolved_frozen_baselines_path, required=False)

    def artifact_health(self) -> dict[str, bool]:
        return {
            "benchmark": self.settings.resolved_benchmark_path.exists(),
            "query_plans": self.settings.resolved_query_plans_path.exists(),
            "anchor_inventory": self.settings.resolved_anchor_inventory_path.exists(),
            "day4_top20": (self.settings.resolved_day4_dir / "predictions_top20.jsonl").exists(),
            "day4_top50": (self.settings.resolved_day4_dir / "predictions_top50.jsonl").exists(),
            "day4_top100": (self.settings.resolved_day4_dir / "predictions_top100.jsonl").exists(),
            "day5_citation_paths": (self.settings.resolved_day5_dir / "citation_paths.jsonl").exists(),
            "day9_citation_paths": (self.settings.resolved_day9_citation_dir / "citation_paths.jsonl").exists(),
        }
    def citation_paths_by_paper(
    self,
    qid: str,
    ) -> dict[str, dict[str, Any]]:

        paths = self.day5_citation_paths().get(
            qid,
            [],
        )

        output = {}

        for path in paths:
            seed = path.get(
                "seed_openalex_id"
            )

            expanded = path.get(
                "expanded_openalex_id"
            )


            if seed:
                output.setdefault(
                    seed,
                    path,
                )

            if expanded:
                output.setdefault(
                    expanded,
                    path,
                )

        return output
