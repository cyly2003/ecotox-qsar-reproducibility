from __future__ import annotations

import math


def parse_publication_year(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4:
        return None
    for start in range(0, len(text) - 3):
        chunk = text[start : start + 4]
        if chunk.isdigit():
            year = int(chunk)
            if 1800 <= year <= 2100:
                return year
    return None


def reference_recency_weight(
    publication_year: object,
    *,
    latest_year: int | None,
    decay_per_5_years: float = 0.8,
    min_weight: float = 0.3,
) -> float:
    year = parse_publication_year(publication_year)
    if year is None or latest_year is None:
        return min_weight
    years_older = max(0, latest_year - year)
    weight = float(decay_per_5_years) ** (years_older / 5.0)
    return max(float(min_weight), min(1.0, weight))


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    if not values:
        raise ValueError("At least one value is required.")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / weight_sum


def weighted_std(values: list[float], weights: list[float], mean_value: float | None = None) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    if len(values) < 2:
        return 0.0
    mean = weighted_mean(values, weights) if mean_value is None else float(mean_value)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    else:
        variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / weight_sum
    return math.sqrt(max(variance, 0.0))
