from __future__ import annotations


def within_duration_tolerance(
    a_hours: float,
    b_hours: float,
    relative_tolerance: float = 0.002,
    minimum_tolerance_hours: float = 0.05,
) -> bool:
    tolerance = max(minimum_tolerance_hours, max(abs(a_hours), abs(b_hours)) * relative_tolerance)
    return abs(a_hours - b_hours) <= tolerance

