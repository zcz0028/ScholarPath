from __future__ import annotations

from typing import Any

from apps.api.schemas import ExperimentItem, ExperimentListResponse
from apps.api.services.artifact_service import ArtifactService
from apps.api.settings import ApiSettings


LABELS = {
    "b5_1_guard_balanced_v4": ("主推荐版本", "Top20/Top50 提升且 Top100 Recall 不损失"),
    "b5_on_b4_precision": ("高精度精选版本", "Macro F1 最高且 FP 最低"),
    "b5_2_guard_aware_balanced": ("可信均衡精筛版本", "Selector 与硬约束联合精筛"),
}


class ExperimentService:
    def __init__(self, artifacts: ArtifactService, settings: ApiSettings) -> None:
        self.artifacts = artifacts
        self.settings = settings

    def list_experiments(self) -> ExperimentListResponse:
        items: list[ExperimentItem] = []
        frozen = self.artifacts.frozen_baselines()
        baselines = frozen.get("baselines") if isinstance(frozen.get("baselines"), list) else []
        for row in baselines:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if name not in LABELS:
                continue
            label, description = LABELS[name]
            path = self.settings.resolve(__import__("pathlib").Path(str(row.get("path") or "")))
            items.append(
                ExperimentItem(
                    id=name,
                    label=label,
                    role=str(row.get("role") or "baseline"),
                    description=description,
                    available=path.exists(),
                    metrics=self._summary_if_present(path),
                )
            )

        day4 = self.artifacts.day4_summary()
        comparison = day4.get("comparison_to_b4") if isinstance(day4.get("comparison_to_b4"), dict) else {}
        items.append(
            ExperimentItem(
                id="week2_day4_rescue",
                label="定向救援召回",
                role="recall_rescue",
                description="失败触发式 Anchor Rescue；只重跑 retrieval-miss 查询。",
                available=bool(day4),
                metrics={
                    "baseline_top100_tp": comparison.get("baseline_top100_tp"),
                    "day4_top100_tp": comparison.get("day4_top100_tp"),
                    "top100_tp_delta": comparison.get("top100_tp_delta"),
                    "baseline_zero_recall_count": comparison.get("baseline_zero_recall_count"),
                    "day4_zero_recall_count": comparison.get("day4_zero_recall_count"),
                    "recovered_query_count": comparison.get("recovered_query_count"),
                },
            )
        )

        day5 = self.artifacts.day5_summary()
        items.append(
            ExperimentItem(
                id="week2_day5_citation",
                label="受控引文扩展",
                role="citation_discovery",
                description="高可信 Seed Gate + 一跳 references/cited-by + API 预算控制。",
                available=bool(day5),
                metrics={
                    "target_query_count": day5.get("target_query_count"),
                    "selected_seed_count": day5.get("selected_seed_count"),
                    "actual_api_calls": day5.get("actual_api_calls"),
                    "cache_hits": day5.get("cache_hits"),
                    "raw_citation_candidates": day5.get("raw_citation_candidates"),
                    "citation_paths": day5.get("citation_paths"),
                    "failed_expansions": day5.get("failed_expansions"),
                    "skipped_due_to_budget": day5.get("skipped_due_to_budget"),
                },
            )
        )
        return ExperimentListResponse(items=items)

    @staticmethod
    def _summary_if_present(path: Any) -> dict[str, Any]:
        summary_path = path / "run_summary.json"
        if not summary_path.exists():
            return {}
        import json
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
