from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import fmean, median
from typing import Any, Mapping, Sequence

DEFAULT_KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _raw(paper: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = paper.get("raw")
    return raw if isinstance(raw, Mapping) else {}


def _score(paper: Mapping[str, Any]) -> float:
    raw = _raw(paper)
    return _float(raw.get("day8_final_score"), _float(raw.get("b3_final_score")))


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(_mean([(x - m) ** 2 for x in values]))


def _mean_prefix(values: Sequence[float], n: int) -> float:
    return _mean(list(values[:n]))


def _gap(values: Sequence[float], left_rank: int, right_rank: int) -> float:
    if len(values) < right_rank:
        return 0.0
    return values[left_rank - 1] - values[right_rank - 1]


def _paper_feature_mean(papers: Sequence[Mapping[str, Any]], key: str, n: int = 5) -> float:
    vals = [_float(_raw(p).get(key)) for p in papers[:n]]
    return _mean(vals)


def extract_cutoff_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Gold-free query/ranking features used by the Day13-2 cutoff policy."""
    question = str(record.get("question") or "")
    papers = [p for p in (record.get("papers") or []) if isinstance(p, Mapping)]
    scores = [_score(p) for p in papers]
    tokens = re.findall(r"[A-Za-z0-9]+", question.casefold())

    canonical = record.get("day8_canonical_constraints")
    constraint_count = len(canonical) if isinstance(canonical, list) else 0

    top5_cov = _paper_feature_mean(papers, "day8_canonical_constraint_coverage", 5)
    top10_cov = _paper_feature_mean(papers, "day8_canonical_constraint_coverage", 10)
    top5_raw_cov = _paper_feature_mean(papers, "day8_raw_constraint_coverage", 5)

    source_counts = []
    for p in papers[:10]:
        raw = _raw(p)
        source_counts.append(_float(raw.get("b4_source_count")))

    return {
        "query_token_count": float(len(tokens)),
        "query_char_count": float(len(question)),
        "constraint_count": float(constraint_count),
        "candidate_count": float(len(papers)),
        "score_top1": scores[0] if scores else 0.0,
        "score_top3_mean": _mean_prefix(scores, 3),
        "score_top5_mean": _mean_prefix(scores, 5),
        "score_top10_mean": _mean_prefix(scores, 10),
        "score_top20_mean": _mean_prefix(scores, 20),
        "score_top10_std": _std(scores[:10]),
        "score_gap_1_2": _gap(scores, 1, 2),
        "score_gap_1_5": _gap(scores, 1, 5),
        "score_gap_5_10": _gap(scores, 5, 10),
        "score_gap_10_20": _gap(scores, 10, 20),
        "top5_canonical_coverage": top5_cov,
        "top10_canonical_coverage": top10_cov,
        "top5_raw_coverage": top5_raw_cov,
        "top10_source_count_mean": _mean(source_counts),
    }


def snap_k(value: float, allowed_ks: Sequence[int] = DEFAULT_KS) -> int:
    ks = sorted({int(k) for k in allowed_ks if int(k) > 0})
    if not ks:
        raise ValueError("allowed_ks must contain at least one positive K")
    return min(ks, key=lambda k: (abs(k - value), k))


@dataclass(slots=True)
class RobustScaler:
    medians: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, float]]) -> "RobustScaler":
        keys = sorted({k for row in rows for k in row})
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        for key in keys:
            vals = sorted(_float(row.get(key)) for row in rows)
            med = float(median(vals)) if vals else 0.0
            deviations = sorted(abs(v - med) for v in vals)
            mad = float(median(deviations)) if deviations else 0.0
            medians[key] = med
            scales[key] = mad if mad > 1e-9 else 1.0
        return cls(medians=medians, scales=scales)

    def transform(self, row: Mapping[str, float]) -> dict[str, float]:
        return {k: (_float(row.get(k)) - med) / self.scales[k] for k, med in self.medians.items()}


@dataclass(slots=True)
class AdaptiveCutoffModel:
    scaler: RobustScaler
    train_features: list[dict[str, float]]
    train_ks: list[int]
    neighbor_count: int
    fallback_k: int
    allowed_ks: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        features: Sequence[Mapping[str, float]],
        target_ks: Sequence[int],
        *,
        neighbor_count: int = 7,
        fallback_k: int = 5,
        allowed_ks: Sequence[int] = DEFAULT_KS,
    ) -> "AdaptiveCutoffModel":
        if len(features) != len(target_ks) or not features:
            raise ValueError("features and target_ks must be non-empty and equal length")
        scaler = RobustScaler.fit(features)
        transformed = [scaler.transform(row) for row in features]
        return cls(
            scaler=scaler,
            train_features=transformed,
            train_ks=[int(k) for k in target_ks],
            neighbor_count=max(1, min(int(neighbor_count), len(features))),
            fallback_k=int(fallback_k),
            allowed_ks=tuple(sorted({int(k) for k in allowed_ks if int(k) > 0})),
        )

    def predict(self, features: Mapping[str, float]) -> tuple[int, dict[str, Any]]:
        x = self.scaler.transform(features)
        distances: list[tuple[float, int]] = []
        for train_x, train_k in zip(self.train_features, self.train_ks, strict=True):
            keys = self.scaler.medians.keys()
            distance = math.sqrt(_mean([(x[k] - train_x[k]) ** 2 for k in keys]))
            distances.append((distance, train_k))
        distances.sort(key=lambda item: (item[0], item[1]))
        neighbors = distances[: self.neighbor_count]
        weights = [1.0 / (d + 0.25) for d, _ in neighbors]
        weighted_k = sum(w * k for w, (_, k) in zip(weights, neighbors, strict=True)) / sum(weights)
        predicted = snap_k(weighted_k, self.allowed_ks)
        return predicted, {
            "weighted_k": weighted_k,
            "neighbor_ks": [k for _, k in neighbors],
            "neighbor_distances": [d for d, _ in neighbors],
            "fallback_k": self.fallback_k,
        }
