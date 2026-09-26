from __future__ import annotations

import json
from dataclasses import dataclass


AQUATIC_MEDIA_CODES = {"FW", "SW", "AQU"}
SEDIMENT_MEDIA_CODES = {"SED", "SEDIMENT"}
SOIL_MEDIA_CODES = {"NAT", "ART", "UKS", "MIN", "HYP"}
TERRESTRIAL_NONSOIL_MEDIA_CODES = {"NONE", "FLT", "AGR", "LIT", "FAB", "POP"}
CULTURE_MEDIA_CODES = {"CUL"}

AQUEOUS_EXPOSURE_CODES = {"AQ", "CM", "MM", "HP", "NR"}
SOIL_EXPOSURE_CODES = {"PR", "SS"}

DOMAIN_PRIORITY = ("sediment", "aquatic", "soil", "terrestrial_nonsoil")


@dataclass(frozen=True)
class MediumClassification:
    medium_domain: str
    primary_medium_domain: str
    medium_domains: tuple[str, ...]
    medium_domain_detail: str
    medium_domain_reason: str
    medium_conflict_flag: bool

    def to_row(self) -> dict[str, object]:
        return {
            "medium_domain": self.medium_domain,
            "primary_medium_domain": self.primary_medium_domain,
            "medium_domains": json.dumps(list(self.medium_domains), ensure_ascii=False),
            "medium_domain_detail": self.medium_domain_detail,
            "medium_domain_reason": self.medium_domain_reason,
            "medium_conflict_flag": int(self.medium_conflict_flag),
        }


def classify_exposure_medium(
    *,
    organism_habitat: object = None,
    media_type: object = None,
    exposure_type: object = None,
    target_basis: object = None,
) -> MediumClassification:
    """Classify exposure medium from test conditions, not species ecology."""

    habitat = clean_text(organism_habitat)
    media = normalize_media_code(media_type)
    exposure = normalize_media_code(exposure_type)
    target = clean_text(target_basis)

    evidence: dict[str, list[str]] = {}
    detail: str | None = None
    primary: str | None = None

    if media in SEDIMENT_MEDIA_CODES:
        add_evidence(evidence, "sediment", f"media_type:{media}")
        primary = primary or "sediment"
        detail = detail or "sediment"
    elif media in AQUATIC_MEDIA_CODES:
        add_evidence(evidence, "aquatic", f"media_type:{media}")
        primary = primary or "aquatic"
        detail = detail or "aquatic"
    elif media in SOIL_MEDIA_CODES:
        add_evidence(evidence, "soil", f"media_type:{media}")
        primary = primary or "soil"
        detail = detail or "soil"
    elif media in CULTURE_MEDIA_CODES:
        if habitat == "water" or exposure in AQUEOUS_EXPOSURE_CODES:
            add_evidence(evidence, "aquatic", "media_type:CUL_culture_with_water_or_aqueous_exposure")
            primary = primary or "aquatic"
            detail = detail or "aquatic_culture"
        elif habitat == "soil" or exposure in SOIL_EXPOSURE_CODES:
            add_evidence(evidence, "soil", "media_type:CUL_culture_with_soil_exposure")
            primary = primary or "soil"
            detail = detail or "soil_culture_or_growth_medium"
        elif habitat in {"non-soil", "nonsoil", "non_soil"}:
            add_evidence(evidence, "terrestrial_nonsoil", "media_type:CUL_culture_with_non_soil_exposure")
            primary = primary or "terrestrial_nonsoil"
            detail = detail or "terrestrial_nonsoil_culture"
        else:
            detail = detail or "culture_unspecified"
    elif media in TERRESTRIAL_NONSOIL_MEDIA_CODES:
        add_evidence(evidence, "terrestrial_nonsoil", f"media_type:{media}")
        primary = primary or "terrestrial_nonsoil"
        detail = detail or "terrestrial_nonsoil_or_contact"

    if habitat == "water":
        add_evidence(evidence, "aquatic", "organism_habitat:Water")
        primary = primary or "aquatic"
        detail = detail or "aquatic"
    elif habitat == "soil":
        add_evidence(evidence, "soil", "organism_habitat:Soil")
        primary = primary or "soil"
        detail = detail or "soil"
    elif habitat in {"non-soil", "nonsoil", "non_soil"}:
        add_evidence(evidence, "terrestrial_nonsoil", "organism_habitat:Non-Soil")
        primary = primary or "terrestrial_nonsoil"
        detail = detail or "terrestrial_nonsoil"

    if exposure in AQUEOUS_EXPOSURE_CODES:
        add_evidence(evidence, "aquatic", f"exposure_type:{exposure}")
        primary = primary or "aquatic"
        detail = detail or "aquatic_exposure"
    elif exposure in SOIL_EXPOSURE_CODES:
        add_evidence(evidence, "soil", f"exposure_type:{exposure}")
        primary = primary or "soil"
        detail = detail or "soil_exposure"

    if "sediment" in target or target.startswith("mg/kg:sed"):
        add_evidence(evidence, "sediment", "target_basis:sediment")
        primary = primary or "sediment"
        detail = detail or "sediment"
    elif target.startswith("mg/kg:soil"):
        add_evidence(evidence, "soil", "target_basis:soil")
        primary = primary or "soil"
        detail = detail or "soil"

    domains = tuple(domain for domain in DOMAIN_PRIORITY if domain in evidence)
    if not domains:
        return MediumClassification(
            medium_domain="unknown",
            primary_medium_domain="unknown",
            medium_domains=("unknown",),
            medium_domain_detail=detail or "unknown",
            medium_domain_reason=f"unmapped_media:{media.lower()}" if media else "missing_medium",
            medium_conflict_flag=False,
        )

    primary = primary if primary in domains else domains[0]
    reasons = []
    for domain in domains:
        reasons.extend(evidence[domain])
    return MediumClassification(
        medium_domain=primary,
        primary_medium_domain=primary,
        medium_domains=domains,
        medium_domain_detail=detail or primary,
        medium_domain_reason=";".join(reasons),
        medium_conflict_flag=len(domains) > 1,
    )


def add_evidence(evidence: dict[str, list[str]], domain: str, reason: str) -> None:
    evidence.setdefault(domain, []).append(reason)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", "-")


def normalize_media_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    while text.endswith(("/", "*")):
        text = text[:-1]
    return text
