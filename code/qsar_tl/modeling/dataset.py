from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CATEGORICAL_FIELDS = (
    "species_id",
    "taxon_id",
    "kingdom_id",
    "phylum_id",
    "class_id",
    "order_id",
    "family_id",
    "genus_id",
    "species_context_id",
    "life_stage_id",
    "habitat_id",
    "primary_medium_id",
    "exposure_route_id",
)


@dataclass(frozen=True)
class DatasetFieldConfig:
    """Column names accepted by AggregatedTaskDataset."""

    molecular_numeric_fields: tuple[str, ...] = (
        "molecular_numeric",
        "numeric_features",
        "descriptor_values",
        "rdkit_descriptors",
    )
    fingerprint_fields: tuple[str, ...] = (
        "fingerprint",
        "morgan_fingerprint",
        "morgan_bits",
    )
    categorical_id_field: str = "categorical_ids"
    categorical_fields: tuple[str, ...] = DEFAULT_CATEGORICAL_FIELDS
    adapter_id_field: str = "adapter_id"
    adapter_name_field: str = "adapter_name"
    task_head_field: str = "task_head"
    target_field: str = "target_value"
    sample_id_fields: tuple[str, ...] = ("sample_id", "record_id", "aggregate_id", "id")


@dataclass(frozen=True)
class AggregatedTaskSample:
    molecular_numeric: list[float]
    fingerprint: list[float]
    categorical_ids: dict[str, int]
    task_head: str
    target_value: float
    adapter_id: int = 0
    adapter_name: str | None = None
    sample_id: str | int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "molecular_numeric": self.molecular_numeric,
            "fingerprint": self.fingerprint,
            "categorical_ids": self.categorical_ids,
            "task_head": self.task_head,
            "target_value": self.target_value,
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "sample_id": self.sample_id,
        }
        for key, value in self.raw.items():
            if key not in payload:
                payload[str(key)] = value
        return payload


class AggregatedTaskDataset:
    """Dataset for condition-aggregated ECOTOX-QSAR task rows.

    The class is intentionally torch-free. PyTorch DataLoader can still consume
    it through the standard ``__len__`` and ``__getitem__`` protocol, while
    importing this module remains safe in environments without torch installed.
    """

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]] | None = None,
        *,
        sqlite_path: str | Path | None = None,
        table_name: str = "aggregated_task_records",
        query: str | None = None,
        field_config: DatasetFieldConfig | None = None,
        numeric_feature_names: Sequence[str] | None = None,
        fingerprint_size: int | None = None,
    ) -> None:
        if samples is None and sqlite_path is None:
            raise ValueError("Provide either samples or sqlite_path.")
        if samples is not None and sqlite_path is not None:
            raise ValueError("Use only one data source: samples or sqlite_path.")

        self.field_config = field_config or DatasetFieldConfig()
        self.numeric_feature_names = tuple(numeric_feature_names or ())
        self.fingerprint_size = fingerprint_size

        source_rows = (
            list(samples)
            if samples is not None
            else self._load_sqlite_rows(Path(sqlite_path or ""), table_name, query)
        )
        self._samples = [self._normalize_row(row) for row in source_rows]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index].as_dict()

    @property
    def samples(self) -> tuple[AggregatedTaskSample, ...]:
        return tuple(self._samples)

    def task_heads(self) -> tuple[str, ...]:
        return tuple(sorted({sample.task_head for sample in self._samples}))

    def numeric_dim(self) -> int:
        return len(self._samples[0].molecular_numeric) if self._samples else 0

    def fingerprint_dim(self) -> int:
        return len(self._samples[0].fingerprint) if self._samples else 0

    def categorical_fields(self) -> tuple[str, ...]:
        fields: set[str] = set()
        for sample in self._samples:
            fields.update(sample.categorical_ids)
        return tuple(sorted(fields))

    def _load_sqlite_rows(
        self,
        sqlite_path: Path,
        table_name: str,
        query: str | None,
    ) -> list[dict[str, Any]]:
        if not sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        sql = query or f"SELECT * FROM {table_name}"
        with sqlite3.connect(sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def _normalize_row(self, row: Mapping[str, Any]) -> AggregatedTaskSample:
        numeric = self._extract_numeric_features(row)
        fingerprint = self._extract_fingerprint(row)
        categorical_ids = self._extract_categorical_ids(row)

        task_head = row.get(self.field_config.task_head_field)
        if task_head is None or str(task_head).strip() == "":
            raise ValueError(f"Missing task head field '{self.field_config.task_head_field}'.")

        target_value = row.get(self.field_config.target_field)
        if target_value is None:
            raise ValueError(f"Missing target field '{self.field_config.target_field}'.")

        return AggregatedTaskSample(
            molecular_numeric=numeric,
            fingerprint=fingerprint,
            categorical_ids=categorical_ids,
            task_head=str(task_head),
            target_value=float(target_value),
            adapter_id=self._coerce_adapter_id(row),
            adapter_name=self._coerce_adapter_name(row),
            sample_id=self._first_present(row, self.field_config.sample_id_fields),
            raw=row,
        )

    def _extract_numeric_features(self, row: Mapping[str, Any]) -> list[float]:
        if self.numeric_feature_names:
            return [float(row.get(name, 0.0) or 0.0) for name in self.numeric_feature_names]
        value = self._first_present(row, self.field_config.molecular_numeric_fields)
        return _coerce_float_vector(value)

    def _extract_fingerprint(self, row: Mapping[str, Any]) -> list[float]:
        value = self._first_present(row, self.field_config.fingerprint_fields)
        fingerprint = _coerce_float_vector(value)
        if self.fingerprint_size is None:
            return fingerprint
        if len(fingerprint) > self.fingerprint_size:
            return fingerprint[: self.fingerprint_size]
        return fingerprint + [0.0] * (self.fingerprint_size - len(fingerprint))

    def _extract_categorical_ids(self, row: Mapping[str, Any]) -> dict[str, int]:
        encoded = row.get(self.field_config.categorical_id_field)
        if encoded is not None:
            return _coerce_int_mapping(encoded)
        result: dict[str, int] = {}
        for field_name in self.field_config.categorical_fields:
            if field_name in row and row[field_name] is not None:
                result[field_name] = int(row[field_name])
        return result

    def _coerce_adapter_id(self, row: Mapping[str, Any]) -> int:
        raw = row.get(self.field_config.adapter_id_field, 0)
        if raw is None or str(raw).strip() == "":
            return 0
        return int(raw)

    def _coerce_adapter_name(self, row: Mapping[str, Any]) -> str | None:
        raw = row.get(self.field_config.adapter_name_field)
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @staticmethod
    def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return None


def _coerce_float_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            parsed = json.loads(text)
            if isinstance(parsed, Mapping):
                return [float(parsed[key]) for key in sorted(parsed)]
            return [float(item) for item in parsed]
        delimiter = "," if "," in text else ";" if ";" in text else "|"
        return [float(item.strip()) for item in text.split(delimiter) if item.strip()]
    if isinstance(value, bytes):
        return _coerce_float_vector(value.decode("utf-8"))
    if isinstance(value, Mapping):
        return [float(value[key]) for key in sorted(value)]
    return [float(item) for item in value]


def _coerce_int_mapping(value: Any) -> dict[str, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("categorical_ids must be a mapping or JSON object string.")
    return {str(key): int(item) for key, item in value.items()}
