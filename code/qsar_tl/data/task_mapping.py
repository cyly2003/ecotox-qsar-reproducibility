from __future__ import annotations

import re
from dataclasses import dataclass


DOSE_RESPONSE_ENDPOINT_RE = re.compile(r"^(EC|LC|IC|AC|ED|LD|BMD|BMC)(\d+(?:\.\d+)?)$")

EXCLUDED_ENDPOINTS = {
    "LT50",
}

MORTALITY_CODES = {"MOR", "MORT", "SURV"}
GROWTH_CODES = {"GRO", "WGHT", "LGTH", "GGRO", "BMAS", "HGHT", "GAIN", "SIZE", "VGOR", "MMTE"}
REPRODUCTION_CODES = {"REP", "GERM", "PROG", "FCND", "GREP", "FERZ"}
POPULATION_CODES = {"POP", "ABND", "PGRT", "GPOP", "CNTL", "COVR", "CHLA", "DVRS", "DMTR", "DBMS"}
IMMOBILIZATION_CODES = {"ITX", "IMBL"}
DEVELOPMENT_CODES = {"DVP", "EMRG", "HTCH", "DVLP", "DFRM", "STGE", "ABNM"}
BEHAVIOR_CODES = {"BEH", "SWIM", "LOCO", "EQUL", "PHTR", "AVO", "CHEM"}
MORPHOLOGY_CODES = {"MPH", "SMIX"}
PHYSIOLOGY_CODES = {"PHY", "PSYN", "PSII", "HTRT", "HRM", "CEL"}
BIOCHEMICAL_CODES = {"ENZ", "BCM", "CTLS", "SODA", "GSTR", "ACHE", "GLPX", "MLDH", "PRCO", "GLTH"}
GENETIC_CODES = {"GEN", "DAMG", "GEXP", "ERAM"}
INJURY_CODES = {"INJ", "HIS", "GINJ"}
FEEDING_CODES = {"FDB", "FDNG", "FCNS"}
MOLTING_CODES = {"MLT", "MULT"}
ACCUMULATION_CODES = {"ACC", "RSDE", "GACC"}

MAIN_ENDPOINT_FAMILIES = {"ECx", "NOEC", "LOEC"}
BIOACCUMULATION_ENDPOINTS = {"BAF", "BCF", "BCFD"}
ENDPOINT_ALIASES = {
    "NOEL": "NOEC",
    "LOEL": "LOEC",
}


@dataclass(frozen=True)
class TaskMappingResult:
    task_head: str | None
    task_group: str | None
    task_family: str | None
    effect_family: str | None
    effect_level_x: float | None
    task_status: str
    task_excluded_reason: str | None


def normalize_code(value: object) -> str:
    """Normalize compact ECOTOX endpoint/effect/measurement codes."""

    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    text = text.lstrip("~")
    while text.endswith(("/", "*")):
        text = text[:-1]
    return text


def parse_effect_level_x(endpoint: object) -> float | None:
    endpoint_norm = normalize_code(endpoint)
    match = DOSE_RESPONSE_ENDPOINT_RE.match(endpoint_norm)
    if not match:
        return None
    return float(match.group(2))


def classify_endpoint(endpoint: object) -> tuple[str | None, str | None, float | None, str | None]:
    endpoint_norm = normalize_code(endpoint)
    if not endpoint_norm:
        return None, None, None, "missing_endpoint"

    if endpoint_norm.startswith("NR"):
        return None, None, None, "excluded_endpoint:NR"

    if endpoint_norm in EXCLUDED_ENDPOINTS:
        return None, None, None, f"excluded_endpoint:{endpoint_norm}"

    if endpoint_norm in BIOACCUMULATION_ENDPOINTS:
        return None, None, None, "bioaccumulation_endpoint_requires_factor_target"

    aliased = ENDPOINT_ALIASES.get(endpoint_norm, endpoint_norm)
    if aliased in {"NOEC", "LOEC"}:
        return aliased, "main_toxicity", None, None

    if endpoint_norm == "MATC":
        return "MATC", "toxicity_aux", None, None

    match = DOSE_RESPONSE_ENDPOINT_RE.match(endpoint_norm)
    effect_level = float(match.group(2)) if match else None
    if effect_level is not None:
        prefix = match.group(1)
        if prefix in {"EC", "LC"}:
            return "ECx", "main_toxicity", effect_level, None
        return f"{prefix}x", "toxicity_aux", effect_level, None

    return None, None, None, f"unsupported_endpoint:{endpoint_norm}"


def classify_effect(effect: object, measurement: object) -> tuple[str | None, str | None]:
    codes = {normalize_code(effect), normalize_code(measurement)}
    codes.discard("")

    if codes & MORTALITY_CODES:
        return "Mortality", None
    if codes & GROWTH_CODES:
        return "Growth", None
    if codes & REPRODUCTION_CODES:
        return "Reproduction", None
    if codes & POPULATION_CODES:
        return "Population", None
    if codes & IMMOBILIZATION_CODES:
        return "Immobilization", None
    if codes & DEVELOPMENT_CODES:
        return "Development", None
    if codes & BEHAVIOR_CODES:
        return "Behavior", None
    if codes & MORPHOLOGY_CODES:
        return "Morphology", None
    if codes & PHYSIOLOGY_CODES:
        return "Physiology", None
    if codes & BIOCHEMICAL_CODES:
        return "Biochemical", None
    if codes & GENETIC_CODES:
        return "GeneticDamage", None
    if codes & INJURY_CODES:
        return "Injury", None
    if codes & FEEDING_CODES:
        return "Feeding", None
    if codes & MOLTING_CODES:
        return "Molting", None
    if codes & ACCUMULATION_CODES:
        return "Accumulation", None

    if not codes:
        return None, "missing_effect_and_measurement"
    return None, "unsupported_effect_family"


def map_task_head(
    *,
    endpoint: object,
    effect: object,
    measurement: object,
    target_name: object = None,
    target_basis: object = None,
) -> TaskMappingResult:
    """Map an included target record to a first-batch ECOTOX-QSAR task head."""

    target_name_norm = normalize_code(target_name)
    target_basis_norm = normalize_code(target_basis)
    if target_name_norm == "NEG_LOG10_MG_KG_BW_DAY" or target_basis_norm == "MG/KG/DAY":
        return TaskMappingResult(
            task_head=None,
            task_group=None,
            task_family=None,
            effect_family=None,
            effect_level_x=None,
            task_status="excluded",
            task_excluded_reason="excluded_oral_target",
        )

    task_family, task_group, effect_level, endpoint_reason = classify_endpoint(endpoint)
    if endpoint_reason is not None:
        return TaskMappingResult(
            task_head=None,
            task_group=task_group,
            task_family=None,
            effect_family=None,
            effect_level_x=None,
            task_status="excluded",
            task_excluded_reason=endpoint_reason,
        )

    effect_family, effect_reason = classify_effect(effect, measurement)
    if effect_reason is not None:
        return TaskMappingResult(
            task_head=None,
            task_group=task_group,
            task_family=task_family,
            effect_family=None,
            effect_level_x=effect_level,
            task_status="excluded",
            task_excluded_reason=effect_reason,
        )

    return TaskMappingResult(
        task_head=f"{task_family}_{effect_family}",
        task_group=task_group,
        task_family=task_family,
        effect_family=effect_family,
        effect_level_x=effect_level,
        task_status="included",
        task_excluded_reason=None,
    )
