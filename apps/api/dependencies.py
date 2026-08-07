from __future__ import annotations

from functools import lru_cache

from apps.api.services.artifact_service import ArtifactService
from apps.api.services.diagnosis_service import DiagnosisService
from apps.api.services.experiment_service import ExperimentService
from apps.api.services.search_service import SearchService
from apps.api.settings import ApiSettings


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache(maxsize=1)
def get_artifact_service() -> ArtifactService:
    return ArtifactService(get_settings())


def get_search_service() -> SearchService:
    return SearchService(get_artifact_service(), get_settings())


def get_diagnosis_service() -> DiagnosisService:
    return DiagnosisService(get_artifact_service())


def get_experiment_service() -> ExperimentService:
    return ExperimentService(get_artifact_service(), get_settings())
