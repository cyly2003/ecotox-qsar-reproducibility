from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesContext:
    taxonomy: dict[str, str | None]
    context: dict[str, str | float | None]


TAXONOMY_LEVELS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def build_species_context(row: dict[str, object]) -> SpeciesContext:
    taxonomy = {level: _to_optional_str(row.get(level)) for level in TAXONOMY_LEVELS}
    context = {
        "life_stage": _to_optional_str(row.get("life_stage")),
        "habitat": _to_optional_str(row.get("habitat")),
        "primary_medium": _to_optional_str(row.get("primary_medium")),
        "exposure_route": _to_optional_str(row.get("exposure_route")),
        "duration_h": _to_optional_float(row.get("duration_h")),
        "log1p_duration_h": _to_optional_float(row.get("log1p_duration_h")),
    }
    return SpeciesContext(taxonomy=taxonomy, context=context)


def _to_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

