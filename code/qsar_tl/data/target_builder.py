from __future__ import annotations

from dataclasses import dataclass
from math import log10


@dataclass(frozen=True)
class ToxicityValueResult:
    value: float | None
    source: str
    imputed: bool
    value_quality: str
    excluded_reason: str | None = None


def choose_toxicity_value(
    mean_value: float | None,
    min_value: float | None,
    max_value: float | None,
    dose_group_count: int | None,
    min_dose_groups_for_midpoint: int = 3,
    mean_op: object = None,
    min_op: object = None,
    max_op: object = None,
) -> ToxicityValueResult:
    del dose_group_count, min_dose_groups_for_midpoint

    if mean_value is not None:
        mean_operator = normalize_operator(mean_op)
        if mean_operator in {"<", ">", "<=", ">="}:
            return ToxicityValueResult(
                value=float(mean_value),
                source="mean_censored",
                imputed=False,
                value_quality="censored",
                excluded_reason="censored_toxicity_value",
            )
        quality = "approx" if mean_operator == "~" else "exact"
        return ToxicityValueResult(value=float(mean_value), source="mean", imputed=False, value_quality=quality)

    if min_value is not None and max_value is not None:
        min_operator = normalize_operator(min_op)
        max_operator = normalize_operator(max_op)
        if min_operator in {"<", ">", "<=", ">="} or max_operator in {"<", ">", "<=", ">="}:
            return ToxicityValueResult(
                value=(float(min_value) + float(max_value)) / 2.0,
                source="min_max_midpoint_censored",
                imputed=True,
                value_quality="censored_midpoint",
                excluded_reason="censored_min_or_max_toxicity_value",
            )
        return ToxicityValueResult(
            value=(float(min_value) + float(max_value)) / 2.0,
            source="min_max_midpoint",
            imputed=True,
            value_quality="midpoint",
        )

    return ToxicityValueResult(
        value=None,
        source="excluded",
        imputed=False,
        value_quality="missing",
        excluded_reason="missing_mean_and_invalid_min_max_midpoint",
    )


def neg_log10(value: float) -> float:
    if value <= 0:
        raise ValueError("Cannot apply -log10 to non-positive values.")
    return -log10(value)


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NR", "NC", "NA", "N/A", "NONE", "NULL"}:
        return None
    while text.endswith(("*", "/")):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def normalize_operator(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_int(value: object) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


@dataclass(frozen=True)
class TargetBuildResult:
    target_value: float | None
    target_name: str | None
    target_family: str | None
    target_basis: str | None
    target_status: str
    tox_value: float | None
    tox_value_source: str
    tox_value_imputed: bool
    value_quality: str
    excluded_reason: str | None


def build_target_from_standardized_value(
    *,
    mean_value: object,
    min_value: object,
    max_value: object,
    dose_group_count: object,
    unit_family: str | None,
    standard_unit: str | None,
    molecular_weight_g_mol: object,
    medium: str | None,
    min_dose_groups_for_midpoint: int = 3,
    mean_op: object = None,
    min_op: object = None,
    max_op: object = None,
) -> TargetBuildResult:
    """Build the first-version modeling target from standardized ECOTOX fields.

    Rules:
    - mean is preferred.
    - if mean is missing, min/max midpoint is allowed without a dose-group-count threshold.
    - water mg/L is converted to mol/L using molecular weight.
    - water mol/L is used directly.
    - soil/sediment mg/kg is modeled as -log10(mg/kg).
    - oral mg/kg/d is modeled as -log10(mg/kg/d) for the auxiliary branch.
    """

    selected = choose_toxicity_value(
        mean_value=parse_float(mean_value),
        min_value=parse_float(min_value),
        max_value=parse_float(max_value),
        dose_group_count=parse_int(dose_group_count),
        min_dose_groups_for_midpoint=min_dose_groups_for_midpoint,
        mean_op=mean_op,
        min_op=min_op,
        max_op=max_op,
    )
    if selected.value is None:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_family=None,
            target_basis=None,
            target_status="excluded",
            tox_value=None,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            value_quality=selected.value_quality,
            excluded_reason=selected.excluded_reason,
        )
    if selected.excluded_reason is not None and selected.value_quality.startswith("censored"):
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_family=None,
            target_basis=None,
            target_status="excluded",
            tox_value=selected.value,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            value_quality=selected.value_quality,
            excluded_reason=selected.excluded_reason,
        )

    if selected.value <= 0:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_family=None,
            target_basis=None,
            target_status="excluded",
            tox_value=selected.value,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            value_quality=selected.value_quality,
            excluded_reason="non_positive_toxicity_value",
        )

    family = (unit_family or "").strip()
    unit = (standard_unit or "").strip()
    medium_norm = (medium or "").strip()

    try:
        if family == "water_mol_l" or unit == "mol/L":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="ptox_mol_l",
                target_family="aquatic_pTox_mol_L",
                target_basis="mol/L",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "water_mg_l" or unit == "mg/L":
            mw = parse_float(molecular_weight_g_mol)
            if mw is None or mw <= 0:
                return TargetBuildResult(
                    target_value=None,
                    target_name=None,
                    target_family=None,
                    target_basis=None,
                    target_status="excluded",
                    tox_value=selected.value,
                    tox_value_source=selected.source,
                    tox_value_imputed=selected.imputed,
                    value_quality=selected.value_quality,
                    excluded_reason="missing_or_invalid_molecular_weight_for_mg_l_to_mol_l",
                )
            mol_l = selected.value / 1000.0 / mw
            return TargetBuildResult(
                target_value=neg_log10(mol_l),
                target_name="ptox_mol_l",
                target_family="aquatic_pTox_mol_L",
                target_basis="mol/L_from_mg/L",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "soil_mg_kg" or unit == "mg/kg":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_mg_kg",
                target_family="solid_neglog_mg_kg",
                target_basis=f"mg/kg:{medium_norm or 'unknown_medium'}",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family in {"oral_mg_kg_d", "diet_mg_kg"} or unit in {"mg/kg/d", "mg/kg diet"}:
            target_name = "neg_log10_mg_kg_diet" if family == "diet_mg_kg" or unit == "mg/kg diet" else "neg_log10_mg_kg_bw_day"
            target_basis = "mg/kg diet" if target_name.endswith("_diet") else "mg/kg/day"
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name=target_name,
                target_family="diet_oral_neglog_mg_kg",
                target_basis=target_basis,
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "soil_g_ha" or unit == "g/ha":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_g_ha",
                target_family="application_neglog_g_ha",
                target_basis="g/ha",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "soil_l_ha" or unit == "L/ha":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_l_ha",
                target_family="application_neglog_l_ha",
                target_basis="L/ha",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "seed_g_kg" or unit == "g/kg seed":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_g_kg_seed",
                target_family="seed_treatment_neglog",
                target_basis="g/kg seed",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "seed_ml_kg" or unit == "mL/kg seed":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_ml_kg_seed",
                target_family="seed_treatment_neglog",
                target_basis="mL/kg seed",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "percent" or unit == "%":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_percent",
                target_family="percent_neglog",
                target_basis="percent",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "mass_per_organism_mg" or unit == "mg/organism":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_mg_per_organism",
                target_family="organism_dose_neglog",
                target_basis="mg/organism",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )

        if family == "mass_per_experimental_unit_mg" or unit == "mg/experimental_unit":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_mg_per_experimental_unit",
                target_family="organism_dose_neglog",
                target_basis="mg/experimental_unit",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                value_quality=selected.value_quality,
                excluded_reason=None,
            )
    except ValueError as exc:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_family=None,
            target_basis=None,
            target_status="excluded",
            tox_value=selected.value,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            value_quality=selected.value_quality,
            excluded_reason=str(exc),
        )

    return TargetBuildResult(
        target_value=None,
        target_name=None,
        target_family=None,
        target_basis=None,
        target_status="excluded",
        tox_value=selected.value,
        tox_value_source=selected.source,
        tox_value_imputed=selected.imputed,
        value_quality=selected.value_quality,
        excluded_reason=f"unsupported_unit_family:{family or unit or 'missing'}",
    )
