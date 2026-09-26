from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from qsar_tl.data.task_mapping import map_task_head


CENSOR_OPERATORS = {"<", ">", "<=", ">="}


@dataclass(frozen=True)
class CensoredAuditRow:
    endpoint: str
    unit_family: str
    medium_domain: str
    task_family: str
    task_head: str
    operator: str
    value_quality: str
    target_status: str
    excluded_reason: str
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "unit_family": self.unit_family,
            "medium_domain": self.medium_domain,
            "task_family": self.task_family,
            "task_head": self.task_head,
            "operator": self.operator,
            "value_quality": self.value_quality,
            "target_status": self.target_status,
            "excluded_reason": self.excluded_reason,
            "n": int(self.n),
        }


def censor_operator(row: Mapping[str, Any]) -> str:
    for column in ("conc1_mean_op", "conc1_min_op", "conc1_max_op"):
        operator = normalize_operator(row.get(column))
        if operator in CENSOR_OPERATORS:
            return operator
    return ""


def normalize_operator(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text in CENSOR_OPERATORS else ""


def censored_direction(operator: str) -> str:
    normalized = normalize_operator(operator)
    if normalized in {">", ">="}:
        return "right"
    if normalized in {"<", "<="}:
        return "left"
    return ""


def censored_hinge_loss(predictions: Any, bounds: Any, directions: Iterable[Any], *, margin: float = 0.0) -> Any:
    """One-sided hinge loss for pTox-scale censored bounds.

    For right-censored concentration values (`true concentration > bound`), the
    pTox target is below the pTox bound. Predictions above that bound are
    penalized. Left-censored concentration values use the opposite inequality.
    """

    import torch

    pred = predictions.float()
    bound = bounds.to(device=pred.device, dtype=pred.dtype)
    direction_ids = direction_id_tensor(directions, device=pred.device)
    if direction_ids.numel() != pred.numel():
        raise ValueError("directions length must match predictions length.")
    right = direction_ids > 0
    left = direction_ids < 0
    valid = right | left
    if not bool(valid.any()):
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    values = torch.zeros_like(pred)
    values[right] = torch.relu(pred[right] - bound[right] + float(margin))
    values[left] = torch.relu(bound[left] - pred[left] + float(margin))
    values = values[valid]
    return torch.mean(values.pow(2))


def direction_id_tensor(directions: Iterable[Any], *, device: Any) -> Any:
    import torch

    if torch.is_tensor(directions):
        return directions.to(device=device, dtype=torch.long)
    values = []
    for direction in directions:
        if direction in (1, "1", "right"):
            values.append(1)
        elif direction in (-1, "-1", "left"):
            values.append(-1)
        else:
            values.append(0)
    return torch.tensor(values, dtype=torch.long, device=device)


def censored_audit_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, str, str, str, str], int] = {}
    for row in rows:
        operator = censor_operator(row)
        if not operator:
            continue
        mapping = map_task_head(
            endpoint=row.get("endpoint"),
            effect=row.get("effect"),
            measurement=row.get("measurement"),
            target_name=row.get("target_name"),
            target_basis=row.get("target_basis"),
        )
        key = (
            str(row.get("endpoint", "") or ""),
            str(row.get("unit_family_v2", row.get("conc1_unit_family", "")) or ""),
            str(row.get("medium_domain", "") or ""),
            str(mapping.task_family or ""),
            str(mapping.task_head or ""),
            operator,
            str(row.get("value_quality", "") or ""),
            str(row.get("target_status", "") or ""),
            str(row.get("excluded_reason", "") or ""),
        )
        buckets[key] = buckets.get(key, 0) + 1
    return [
        CensoredAuditRow(*key, n=count).as_dict()
        for key, count in sorted(buckets.items(), key=lambda item: (-item[1], item[0]))
    ]
