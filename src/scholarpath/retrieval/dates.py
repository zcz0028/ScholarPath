from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def parse_benchmark_cutoff(value: Any, offset_days: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = datetime.strptime(text, "%Y%m%d").date()
    return (parsed - timedelta(days=int(offset_days))).isoformat()
