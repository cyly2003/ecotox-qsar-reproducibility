from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from qsar_tl.config import load_config


TOXICITY_BIN_MISSING_INDEX = -1
DEFAULT_TOXICITY_BIN_SCHEME = "authority_v1"


@dataclass(frozen=True)
class ToxicityBinningConfig:
    enabled: bool = False
    scheme: str = DEFAULT_TOXICITY_BIN_SCHEME
    mode: str = "aux_classification"
    loss_weight: float = 0.05
    boundary_policy: str = "hard"
    boundary_tolerance: float = 0.05
    threshold_multiplier: float = 1.0
    require_active_bin_for_regression: bool = False

    def active(self) -> bool:
        return self.enabled and self.mode in {"aux_classification", "ordinal", "soft_expert"} and self.loss_weight > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scheme": self.scheme,
            "mode": self.mode,
            "loss_weight": self.loss_weight,
            "boundary_policy": self.boundary_policy,
            "boundary_tolerance": self.boundary_tolerance,
            "threshold_multiplier": self.threshold_multiplier,
            "require_active_bin_for_regression": self.require_active_bin_for_regression,
        }


@dataclass(frozen=True)
class ToxicityBinAssignment:
    index: int = TOXICITY_BIN_MISSING_INDEX
    label: str = ""
    scheme: str = DEFAULT_TOXICITY_BIN_SCHEME
    source: str = ""
    boundary_flag: bool = False
    status: str = "not_configured"
    value: float | None = None
    value_unit: str = ""
    conversion: str = ""

    def as_sample_fields(self) -> dict[str, Any]:
        return {
            "toxicity_bin_index": int(self.index),
            "toxicity_bin_label": self.label,
            "toxicity_bin_scheme": self.scheme,
            "toxicity_bin_source": self.source,
            "toxicity_bin_boundary_flag": int(bool(self.boundary_flag)),
            "toxicity_bin_status": self.status,
            "toxicity_bin_value": "" if self.value is None else float(self.value),
            "toxicity_bin_value_unit": self.value_unit,
            "toxicity_bin_conversion": self.conversion,
        }


def load_toxicity_bin_scheme(scheme: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(scheme, Mapping):
        return dict(scheme)
    scheme_name = str(scheme or DEFAULT_TOXICITY_BIN_SCHEME).strip()
    if scheme_name in {"", DEFAULT_TOXICITY_BIN_SCHEME, "authority"}:
        path = Path("configs/toxicity_bins.authority_v1.yaml")
    else:
        path = Path(scheme_name)
    data = load_config(path)
    data.setdefault("scheme", DEFAULT_TOXICITY_BIN_SCHEME)
    return data


def toxicity_bin_class_count(scheme: Mapping[str, Any]) -> int:
    classes = scheme.get("classes", [])
    indices = []
    if isinstance(classes, list):
        for item in classes:
            if isinstance(item, Mapping) and _optional_float(item.get("index")) is not None:
                indices.append(int(float(item["index"])))
    if indices:
        return max(indices) + 1
    rules = scheme.get("rules", {})
    if isinstance(rules, Mapping):
        for rule in rules.values():
            if not isinstance(rule, Mapping):
                continue
            for bin_rule in rule.get("bins", []) or []:
                if isinstance(bin_rule, Mapping) and _optional_float(bin_rule.get("index")) is not None:
                    indices.append(int(float(bin_rule["index"])))
    return max(indices) + 1 if indices else 0


def assign_toxicity_bin(
    row: Mapping[str, Any],
    scheme: Mapping[str, Any],
    *,
    config: ToxicityBinningConfig | None = None,
    descriptor_mol_weight: float | None = None,
) -> ToxicityBinAssignment:
    cfg = config or ToxicityBinningConfig(enabled=True)
    scheme_name = str(scheme.get("scheme", cfg.scheme) or cfg.scheme)
    unit_family = str(row.get("unit_family_v2", "") or "").strip()
    unsupported = {str(value) for value in scheme.get("unsupported_unit_families", []) or []}
    if unit_family in unsupported:
        return ToxicityBinAssignment(scheme=scheme_name, status="no_authoritative_threshold")

    rules = scheme.get("rules", {}) if isinstance(scheme.get("rules", {}), Mapping) else {}
    rule = rules.get(unit_family)
    if not isinstance(rule, Mapping):
        return ToxicityBinAssignment(scheme=scheme_name, status="no_authoritative_threshold")
    if not bool(rule.get("enabled_for_aux_loss", False)):
        return ToxicityBinAssignment(scheme=scheme_name, status="disabled_for_aux_loss")

    value, conversion, status = _resolve_standard_value(
        row,
        unit_family=unit_family,
        rule=rule,
        descriptor_mol_weight=descriptor_mol_weight,
    )
    if value is None:
        return ToxicityBinAssignment(
            scheme=scheme_name,
            source=str(rule.get("source", "")),
            status=status,
            value_unit=str(rule.get("value_unit", "")),
            conversion=conversion,
        )
    if value <= 0 or not math.isfinite(value):
        return ToxicityBinAssignment(
            scheme=scheme_name,
            source=str(rule.get("source", "")),
            status="non_positive_or_invalid_standard_value",
            value=value,
            value_unit=str(rule.get("value_unit", "")),
            conversion=conversion,
        )

    threshold_multiplier = max(float(cfg.threshold_multiplier), 1e-12)
    bin_rule = _match_bin(value, rule.get("bins", []) or [], threshold_multiplier=threshold_multiplier)
    if bin_rule is None:
        return ToxicityBinAssignment(
            scheme=scheme_name,
            source=str(rule.get("source", "")),
            status="no_matching_bin",
            value=value,
            value_unit=str(rule.get("value_unit", "")),
            conversion=conversion,
        )
    return ToxicityBinAssignment(
        index=int(float(bin_rule["index"])),
        label=str(bin_rule.get("label", "")),
        scheme=scheme_name,
        source=str(rule.get("source", "")),
        boundary_flag=_is_near_boundary(
            value,
            rule.get("bins", []) or [],
            tolerance=max(0.0, float(cfg.boundary_tolerance)),
            threshold_multiplier=threshold_multiplier,
        ),
        status="active",
        value=value,
        value_unit=str(rule.get("value_unit", "")),
        conversion=conversion,
    )


def summarize_toxicity_bins(samples: list[Mapping[str, Any]], config: ToxicityBinningConfig, class_count: int) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    conversion_counts: dict[str, int] = {}
    boundary_count = 0
    eligible_count = 0
    for sample in samples:
        status = str(sample.get("toxicity_bin_status", "not_configured") or "not_configured")
        status_counts[status] = status_counts.get(status, 0) + 1
        label = str(sample.get("toxicity_bin_label", "") or "")
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        conversion = str(sample.get("toxicity_bin_conversion", "") or "")
        if conversion:
            conversion_counts[conversion] = conversion_counts.get(conversion, 0) + 1
        if int(sample.get("toxicity_bin_index", TOXICITY_BIN_MISSING_INDEX) or TOXICITY_BIN_MISSING_INDEX) >= 0:
            eligible_count += 1
        if int(sample.get("toxicity_bin_boundary_flag", 0) or 0) == 1:
            boundary_count += 1
    return {
        **config.to_manifest(),
        "applied": bool(config.enabled),
        "class_count": int(class_count),
        "eligible_samples": eligible_count,
        "ineligible_samples": max(len(samples) - eligible_count, 0),
        "boundary_samples": boundary_count,
        "status_counts": dict(sorted(status_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "conversion_counts": dict(sorted(conversion_counts.items())),
    }


def _resolve_standard_value(
    row: Mapping[str, Any],
    *,
    unit_family: str,
    rule: Mapping[str, Any],
    descriptor_mol_weight: float | None,
) -> tuple[float | None, str, str]:
    mg_l = _optional_float(row.get("standard_value_mg_l"))
    mg_kg = _optional_float(row.get("standard_value_mg_kg"))
    if unit_family == "water_mg_l":
        return (mg_l, "direct_standard_value_mg_l", "missing_standard_value_mg_l") if mg_l is not None else (None, "", "missing_standard_value_mg_l")
    if unit_family == "soil_mg_kg":
        return (mg_kg, "direct_standard_value_mg_kg", "missing_standard_value_mg_kg") if mg_kg is not None else (None, "", "missing_standard_value_mg_kg")
    if unit_family == "water_mol_l":
        if mg_l is not None:
            return mg_l, "direct_standard_value_mg_l", ""
        mol_l = _optional_float(row.get("standard_value_mol_l"))
        if mol_l is None:
            return None, "", "missing_standard_value_mol_l"
        mw = _molecular_weight(row, descriptor_mol_weight=descriptor_mol_weight)
        if mw is None or mw <= 0:
            return None, "mol_l_to_mg_l_using_molecular_weight", "missing_molecular_weight"
        return mol_l * mw * 1000.0, "mol_l_to_mg_l_using_molecular_weight", ""
    if unit_family == "oral_mg_kg_d":
        value = _optional_float(row.get("standard_value_mg_kg_bw_day"))
        if value is None:
            return None, "", "missing_standard_value_mg_kg_bw_day"
        task_head = str(row.get("task_head", "") or "").lower()
        target_name = str(row.get("target_name", "") or "").lower()
        if "ld50" not in task_head and "oral" not in target_name:
            return None, "direct_standard_value_mg_kg_bw_day", "endpoint_context_not_supported"
        return value, "direct_standard_value_mg_kg_bw_day", ""
    for column in rule.get("value_columns", []) or []:
        value = _optional_float(row.get(str(column)))
        if value is not None:
            return value, f"direct_{column}", ""
    return None, "", "missing_standard_value"


def _molecular_weight(row: Mapping[str, Any], *, descriptor_mol_weight: float | None) -> float | None:
    for key in ("molecular_weight_g_mol", "molecular_weight_rdkit_g_mol"):
        value = _optional_float(row.get(key))
        if value is not None and value > 0:
            return value
    value = _optional_float(descriptor_mol_weight)
    if value is not None and value > 0:
        return value
    smiles = str(row.get("smiles", "") or "").strip()
    if smiles.casefold() in {"", "nan", "none", "null", "na", "n/a", "<na>"}:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    weight = float(Descriptors.MolWt(mol))
    return weight if math.isfinite(weight) and weight > 0 else None


def _match_bin(value: float, bins: list[Any], *, threshold_multiplier: float) -> Mapping[str, Any] | None:
    for bin_rule in bins:
        if not isinstance(bin_rule, Mapping):
            continue
        upper = _optional_float(bin_rule.get("upper"))
        if upper is not None and value <= upper * threshold_multiplier:
            return bin_rule
        lower_exclusive = _optional_float(bin_rule.get("lower_exclusive"))
        if lower_exclusive is not None and value > lower_exclusive * threshold_multiplier:
            return bin_rule
    return None


def _is_near_boundary(value: float, bins: list[Any], *, tolerance: float, threshold_multiplier: float) -> bool:
    for bin_rule in bins:
        if not isinstance(bin_rule, Mapping):
            continue
        threshold = _optional_float(bin_rule.get("upper"))
        if threshold is None:
            threshold = _optional_float(bin_rule.get("lower_exclusive"))
        if threshold is None or threshold <= 0:
            continue
        boundary = threshold * threshold_multiplier
        if abs(value - boundary) <= boundary * tolerance + 1e-12:
            return True
    return False


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
