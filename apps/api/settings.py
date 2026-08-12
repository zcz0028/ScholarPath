from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Filesystem and runtime settings for the competition API.

    Paths are resolved relative to ``project_root`` unless absolute.  The
    defaults match the Week-2 artifact layout used by ScholarPath.
    """

    model_config = SettingsConfigDict(
        env_prefix="SCHOLARPATH_API_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    environment: str = "development"
    service_version: str = "0.6.0"

    benchmark_path: Path = Path("data/raw/RealScholarQuery/test.jsonl")
    query_plans_path: Path = Path("outputs/week2_day3_query_plans/query_plans.jsonl")
    anchor_inventory_path: Path = Path("outputs/week2_day1_diagnostics/anchor_inventory.jsonl")

    day4_dir: Path = Path("outputs/week2_day4_rescue")
    day5_dir: Path = Path("outputs/week2_day5_citation")
    day9_citation_dir: Path = Path("outputs/week2_day9_citation")
    frozen_baselines_path: Path = Path("configs/week2/frozen_baselines.json")

    live_max_plans: int = 2
    live_per_plan: int = 50
    live_max_results: int = 100

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    @property
    def resolved_benchmark_path(self) -> Path:
        return self.resolve(self.benchmark_path)

    @property
    def resolved_query_plans_path(self) -> Path:
        return self.resolve(self.query_plans_path)

    @property
    def resolved_anchor_inventory_path(self) -> Path:
        return self.resolve(self.anchor_inventory_path)

    @property
    def resolved_day4_dir(self) -> Path:
        return self.resolve(self.day4_dir)

    @property
    def resolved_day5_dir(self) -> Path:
        return self.resolve(self.day5_dir)

    @property
    def resolved_day9_citation_dir(self) -> Path:
        return self.resolve(self.day9_citation_dir)

    @property
    def resolved_frozen_baselines_path(self) -> Path:
        return self.resolve(self.frozen_baselines_path)
