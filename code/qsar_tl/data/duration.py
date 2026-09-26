from __future__ import annotations

from math import log1p


def normalize_duration_to_hours(value: float | None, unit: str | None) -> float | None:
    if value is None or unit is None:
        return None
    unit_norm = unit.strip().lower()
    multipliers = {
        "h": 1.0,
        "hr": 1.0,
        "hrs": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "d": 24.0,
        "day": 24.0,
        "days": 24.0,
        "min": 1.0 / 60.0,
        "minute": 1.0 / 60.0,
        "minutes": 1.0 / 60.0,
    }
    if unit_norm not in multipliers:
        return None
    return float(value) * multipliers[unit_norm]


def duration_features(duration_h: float | None) -> dict[str, float | None]:
    if duration_h is None:
        return {"duration_h": None, "log1p_duration_h": None}
    return {"duration_h": duration_h, "log1p_duration_h": log1p(duration_h)}

