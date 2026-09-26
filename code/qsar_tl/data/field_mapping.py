from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldMapping:
    tables: dict[str, str]
    fields: dict[str, str]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FieldMapping":
        tables = config.get("tables", {})
        fields = config.get("fields", {})
        if not isinstance(tables, dict) or not isinstance(fields, dict):
            raise ValueError("Field mapping config must contain 'tables' and 'fields' mappings.")
        return cls(tables=tables, fields=fields)

