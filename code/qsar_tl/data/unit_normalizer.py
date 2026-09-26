from __future__ import annotations

import re
from dataclasses import dataclass

from qsar_tl.data.medium import classify_exposure_medium
from qsar_tl.data.target_builder import parse_float


@dataclass(frozen=True)
class UnitNormalizationResult:
    mean_value_v2: float | None
    min_value_v2: float | None
    max_value_v2: float | None
    unit_family_v2: str | None
    standard_unit_v2: str | None
    standard_value_mg_l: float | None
    standard_value_mol_l: float | None
    standard_value_mg_kg: float | None
    standard_value_g_ha: float | None
    standard_value_mg_kg_diet: float | None
    standard_value_mg_kg_bw_day: float | None
    unit_conversion_source: str
    unit_conversion_confidence: str
    unit_conversion_note: str | None
    conversion_path: str | None
    active_ingredient_basis: bool
    acid_equivalent_basis: bool


def normalize_concentration_values(
    *,
    standardized_mean: object,
    standardized_min: object,
    standardized_max: object,
    unit_family: object,
    standard_unit: object,
    raw_mean: object,
    raw_min: object,
    raw_max: object,
    raw_unit: object,
    molecular_weight_g_mol: object = None,
    organism_habitat: object = None,
    media_type: object = None,
    exposure_type: object = None,
) -> UnitNormalizationResult:
    family = clean_family(unit_family)
    unit = clean_unit(standard_unit)
    mean_std = parse_float(standardized_mean)
    min_std = parse_float(standardized_min)
    max_std = parse_float(standardized_max)
    raw_unit_norm = normalize_unit_text(raw_unit)
    medium = classify_exposure_medium(
        organism_habitat=organism_habitat,
        media_type=media_type,
        exposure_type=exposure_type,
    )
    if (
        family
        and family != "other"
        and raw_unit_norm not in CONTEXT_SENSITIVE_UNITS
        and (mean_std is not None or (min_std is not None and max_std is not None))
    ):
        standard_values = standard_value_payload(
            family=family,
            unit=unit or None,
            mean_value=mean_std,
            molecular_weight_g_mol=molecular_weight_g_mol,
        )
        return UnitNormalizationResult(
            mean_value_v2=mean_std,
            min_value_v2=min_std,
            max_value_v2=max_std,
            unit_family_v2=family,
            standard_unit_v2=unit or None,
            **standard_values,
            unit_conversion_source="source_standardized",
            unit_conversion_confidence="high",
            unit_conversion_note=None,
            conversion_path=f"source_standardized:{unit or family}",
            active_ingredient_basis=is_active_ingredient_unit(raw_unit),
            acid_equivalent_basis=is_acid_equivalent_unit(raw_unit),
        )

    raw_values = (parse_float(raw_mean), parse_float(raw_min), parse_float(raw_max))
    if not raw_unit_norm:
        return unsupported(raw_values, raw_unit, "missing_raw_unit")

    rule = conversion_rule(raw_unit_norm, molecular_weight_g_mol, medium_domain=medium.medium_domain)
    if rule is None:
        return unsupported(raw_values, raw_unit, "unsupported_raw_unit")

    unit_family_v2, standard_unit_v2, factor, confidence, note = rule
    mean_value = scale(raw_values[0], factor)
    standard_values = standard_value_payload(
        family=unit_family_v2,
        unit=standard_unit_v2,
        mean_value=mean_value,
        molecular_weight_g_mol=molecular_weight_g_mol,
    )
    return UnitNormalizationResult(
        mean_value_v2=mean_value,
        min_value_v2=scale(raw_values[1], factor),
        max_value_v2=scale(raw_values[2], factor),
        unit_family_v2=unit_family_v2,
        standard_unit_v2=standard_unit_v2,
        **standard_values,
        unit_conversion_source="raw_unit_rule",
        unit_conversion_confidence=confidence,
        unit_conversion_note=note,
        conversion_path=conversion_path(unit_family_v2, standard_unit_v2),
        active_ingredient_basis=is_active_ingredient_unit(raw_unit),
        acid_equivalent_basis=is_acid_equivalent_unit(raw_unit),
    )


def conversion_rule(
    unit: str,
    molecular_weight_g_mol: object = None,
    *,
    medium_domain: str | None = None,
) -> tuple[str, str, float, str, str | None] | None:
    mw = parse_float(molecular_weight_g_mol)
    medium = (medium_domain or "").strip().lower()

    soil_mass_factors = {
        "mg/kg": 1.0,
        "mg/kg soil": 1.0,
        "mg/kg dry soil": 1.0,
        "mg/kg dry wt": 1.0,
        "ai mg/kg": 1.0,
        "ai mg/kg soil": 1.0,
        "ai mg/kg dry soil": 1.0,
        "ug/g": 1.0,
        "ug/g soil": 1.0,
        "ug/g dry soil": 1.0,
        "ug/g dry wt": 1.0,
        "ug/g bdwt": 1.0,
        "g/kg soil": 1000.0,
        "g/kg dry soil": 1000.0,
        "ai g/kg": 1000.0,
        "mg/g soil": 1000.0,
        "ug/kg soil": 0.001,
        "ai ug/kg soil": 0.001,
        "ug/kg dry soil": 0.001,
        "ng/g soil": 0.001,
        "ng/g dw soil": 0.001,
    }
    if unit in soil_mass_factors:
        return ("soil_mg_kg", "mg/kg", soil_mass_factors[unit], "high", f"{unit}_to_mg/kg")
    if unit in {"ppm", "ppmw", "ai ppm", "ae ppm"}:
        if medium == "aquatic":
            return ("water_mg_l", "mg/L", 1.0, "medium", "ppm_assumed_mg_per_l_for_aqueous_matrix")
        if medium in {"soil", "sediment", "terrestrial_nonsoil"}:
            return ("soil_mg_kg", "mg/kg", 1.0, "medium", "ppm_assumed_mg_per_kg_for_solid_matrix")
        return None
    if unit in {"ppb", "ai ppb", "ae ppb"}:
        if medium == "aquatic":
            return ("water_mg_l", "mg/L", 0.001, "medium", "ppb_assumed_ug_per_l_for_aqueous_matrix")
        if medium in {"soil", "sediment", "terrestrial_nonsoil"}:
            return ("soil_mg_kg", "mg/kg", 0.001, "medium", "ppb_assumed_ug_per_kg_for_solid_matrix")
        return None
    if unit == "mmol/kg soil" and mw is not None and mw > 0:
        return ("soil_mg_kg", "mg/kg", mw, "medium", "mmol/kg_soil_to_mg/kg_using_molecular_weight")

    area_factors = {
        "kg/ha": 1000.0,
        "ai kg/ha": 1000.0,
        "ae kg/ha": 1000.0,
        "g/ha": 1.0,
        "ai g/ha": 1.0,
        "ae g/ha": 1.0,
        "lb/acre": 1120.85116,
        "ai lb/acre": 1120.85116,
        "ae lb/acre": 1120.85116,
        "oz/acre": 70.0531975,
        "ai oz/acre": 70.0531975,
        "g/m2": 10000.0,
        "ai g/m2": 10000.0,
        "mg/cm2": 100000.0,
        "ug/cm2": 100.0,
        "ai ug/cm2": 100.0,
    }
    if unit in area_factors:
        return ("soil_g_ha", "g/ha", area_factors[unit], "high", f"{unit}_to_g/ha")

    volume_area_factors = {
        "l/ha": 1.0,
        "ai l/ha": 1.0,
        "ml/ha": 0.001,
        "ai ml/ha": 0.001,
        "gal/acre": 9.353956,
        "fl oz/acre": 0.073078,
        "l/m2": 10000.0,
        "ml/m2": 10.0,
    }
    if unit in volume_area_factors:
        return ("soil_l_ha", "L/ha", volume_area_factors[unit], "medium", f"{unit}_to_L/ha")

    water_mass_factors = {
        "mg/l": 1.0,
        "mg/l diet": 1.0,
        "ai mg/l": 1.0,
        "ae mg/l": 1.0,
        "mg/dm3": 1.0,
        "ug/ml": 1.0,
        "ai ug/ml": 1.0,
        "g/l": 1000.0,
        "ai g/l": 1000.0,
        "ae g/l": 1000.0,
        "mg/ml": 1000.0,
        "ai mg/ml": 1000.0,
        "ug/l": 0.001,
        "ai ug/l": 0.001,
        "ae ug/l": 0.001,
        "ng/ml": 0.001,
        "ul/l": 0.001,
        "nl/l": 0.000001,
        "ng/l": 0.000001,
        "ai ng/l": 0.000001,
    }
    if unit in water_mass_factors:
        return ("water_mg_l", "mg/L", water_mass_factors[unit], "high", f"{unit}_to_mg/L")

    molar_factors = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "umol/l": 1e-6,
        "mmol/l": 1e-3,
        "nmol/l": 1e-9,
        "μm": 1e-6,
        "nm": 1e-9,
        "umol/ml": 1e-3,
        "nmol/ml": 1e-6,
        "mol/m3": 1e-3,
    }
    if unit in molar_factors:
        return ("water_mol_l", "mol/L", molar_factors[unit], "high", f"{unit}_to_mol/L")

    seed_factors = {
        "g/kg sd": 1.0,
        "ai g/kg sd": 1.0,
        "ai g/100 kg sd": 0.01,
        "g/100 kg sd": 0.01,
        "mg/kg sd": 0.001,
        "ml/kg sd": 1.0,
    }
    if unit in seed_factors:
        family = "seed_ml_kg" if unit == "ml/kg sd" else "seed_g_kg"
        standard = "mL/kg seed" if unit == "ml/kg sd" else "g/kg seed"
        return (family, standard, seed_factors[unit], "medium", f"{unit}_seed_treatment")

    diet_factors = {
        "mg/kg diet": 1.0,
        "ai mg/kg diet": 1.0,
        "ppm diet": 1.0,
        "ai ppm diet": 1.0,
        "ppb diet": 0.001,
        "ug/g diet": 1.0,
        "ug/g dry diet": 1.0,
        "ug/kg diet": 0.001,
    }
    if unit in diet_factors:
        return ("diet_mg_kg", "mg/kg diet", diet_factors[unit], "medium", f"{unit}_to_mg/kg_diet")

    oral_factors = {
        "mg/kg/d": 1.0,
        "mg/kg bdwt/d": 1.0,
        "mg/kg bw/d": 1.0,
        "mg/kg/day": 1.0,
        "ug/kg/d": 0.001,
        "ug/kg bdwt/d": 0.001,
        "mg/kg bdwt": 1.0,
        "mg/kg bw": 1.0,
    }
    if unit in oral_factors:
        return ("oral_mg_kg_d", "mg/kg/d", oral_factors[unit], "medium", f"{unit}_to_mg/kg/day_or_bw")

    organism_factors = {
        "ug/org": 0.001,
        "mg/org": 1.0,
        "ug": 0.001,
        "mg": 1.0,
        "g": 1000.0,
    }
    if unit in organism_factors:
        return ("mass_per_organism_mg", "mg/organism", organism_factors[unit], "low", f"{unit}_to_mg/organism")

    experimental_unit_factors = {
        "mg/eu": 1.0,
        "ai mg/eu": 1.0,
        "g/eu": 1000.0,
        "ai g/eu": 1000.0,
    }
    if unit in experimental_unit_factors:
        return ("mass_per_experimental_unit_mg", "mg/experimental_unit", experimental_unit_factors[unit], "low", f"{unit}_to_mg/eu")

    if unit in {"%", "ai %", "% v/v", "% w/v"}:
        return ("percent", "%", 1.0, "low", f"{unit}_kept_as_percent")

    return None


def unsupported(values: tuple[float | None, float | None, float | None], raw_unit: object, reason: str) -> UnitNormalizationResult:
    return UnitNormalizationResult(
        mean_value_v2=values[0],
        min_value_v2=values[1],
        max_value_v2=values[2],
        unit_family_v2=None,
        standard_unit_v2=None,
        standard_value_mg_l=None,
        standard_value_mol_l=None,
        standard_value_mg_kg=None,
        standard_value_g_ha=None,
        standard_value_mg_kg_diet=None,
        standard_value_mg_kg_bw_day=None,
        unit_conversion_source="unsupported",
        unit_conversion_confidence="none",
        unit_conversion_note=f"{reason}:{raw_unit if raw_unit is not None else ''}",
        conversion_path=None,
        active_ingredient_basis=is_active_ingredient_unit(raw_unit),
        acid_equivalent_basis=is_acid_equivalent_unit(raw_unit),
    )


def normalize_unit_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"", "nr", "nc", "na", "n/a", "none", "null", "--"}:
        return ""
    text = text.replace("μ", "u")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_family(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_unit(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def scale(value: float | None, factor: float) -> float | None:
    if value is None:
        return None
    return float(value) * factor


def is_active_ingredient_unit(value: object) -> bool:
    return normalize_unit_text(value).startswith("ai ") or normalize_unit_text(value).startswith("ae ")


def is_acid_equivalent_unit(value: object) -> bool:
    return normalize_unit_text(value).startswith("ae ")


CONTEXT_SENSITIVE_UNITS = {"ppm", "ppmw", "ai ppm", "ae ppm", "ppb", "ai ppb", "ae ppb"}


def standard_value_payload(
    *,
    family: str | None,
    unit: str | None,
    mean_value: float | None,
    molecular_weight_g_mol: object,
) -> dict[str, float | None]:
    values = {
        "standard_value_mg_l": None,
        "standard_value_mol_l": None,
        "standard_value_mg_kg": None,
        "standard_value_g_ha": None,
        "standard_value_mg_kg_diet": None,
        "standard_value_mg_kg_bw_day": None,
    }
    if mean_value is None:
        return values
    if family == "water_mg_l" or unit == "mg/L":
        values["standard_value_mg_l"] = mean_value
        mw = parse_float(molecular_weight_g_mol)
        if mw is not None and mw > 0:
            values["standard_value_mol_l"] = mean_value / 1000.0 / mw
    elif family == "water_mol_l" or unit == "mol/L":
        values["standard_value_mol_l"] = mean_value
    elif family == "soil_mg_kg" or unit == "mg/kg":
        values["standard_value_mg_kg"] = mean_value
    elif family == "soil_g_ha" or unit == "g/ha":
        values["standard_value_g_ha"] = mean_value
    elif family == "diet_mg_kg" or unit == "mg/kg diet":
        values["standard_value_mg_kg_diet"] = mean_value
    elif family == "oral_mg_kg_d" or unit == "mg/kg/d":
        values["standard_value_mg_kg_bw_day"] = mean_value
    return values


def conversion_path(family: str | None, unit: str | None) -> str | None:
    if family == "water_mg_l" and unit == "mg/L":
        return "raw_to_mg_L_to_mol_L_to_pTox"
    if family == "water_mol_l" and unit == "mol/L":
        return "raw_to_mol_L_to_pTox"
    if family == "soil_mg_kg" and unit == "mg/kg":
        return "raw_to_mg_kg_to_neglog"
    if family == "soil_g_ha" and unit == "g/ha":
        return "raw_to_g_ha_to_neglog"
    if family in {"diet_mg_kg", "oral_mg_kg_d"}:
        return "raw_to_diet_or_oral_mg_kg_to_neglog"
    if family:
        return f"raw_to_{family}_to_neglog"
    return None
