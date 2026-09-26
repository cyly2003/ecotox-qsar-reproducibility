from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence


UNKNOWN_TOKEN = "<UNK>"


@dataclass(frozen=True)
class CategoryVocab:
    """Stable integer vocabulary for taxonomy and experiment context fields."""

    field_name: str
    token_to_id: dict[str, int]
    unknown_token: str = UNKNOWN_TOKEN

    @property
    def unknown_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, value: Any) -> int:
        token = normalize_category_token(value)
        return self.token_to_id.get(token, self.unknown_id)


@dataclass(frozen=True)
class NumericStandardizationParams:
    """Mean and scale values used to standardize numeric molecular descriptors."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.mean):
            raise ValueError(
                f"Expected {len(self.mean)} numeric features, got {len(values)}."
            )
        return [
            (float(value) - center) / divisor
            for value, center, divisor in zip(values, self.mean, self.scale)
        ]


def normalize_category_token(value: Any) -> str:
    if value is None:
        return UNKNOWN_TOKEN
    text = str(value).strip()
    return text if text else UNKNOWN_TOKEN


def build_category_vocab(
    samples: Iterable[Mapping[str, Any]],
    field_name: str,
    *,
    min_count: int = 1,
    unknown_token: str = UNKNOWN_TOKEN,
) -> CategoryVocab:
    counts: Counter[str] = Counter()
    for sample in samples:
        counts[normalize_category_token(sample.get(field_name))] += 1

    tokens = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count and token != unknown_token
    ]
    token_to_id = {unknown_token: 0}
    token_to_id.update({token: index for index, token in enumerate(tokens, start=1)})
    return CategoryVocab(
        field_name=field_name,
        token_to_id=token_to_id,
        unknown_token=unknown_token,
    )


def build_category_vocabs(
    samples: Sequence[Mapping[str, Any]],
    categorical_fields: Sequence[str],
    *,
    min_count: int = 1,
) -> dict[str, CategoryVocab]:
    return {
        field_name: build_category_vocab(samples, field_name, min_count=min_count)
        for field_name in categorical_fields
    }


def encode_category_fields(
    sample: Mapping[str, Any],
    vocabs: Mapping[str, CategoryVocab],
) -> dict[str, int]:
    return {
        field_name: vocab.encode(sample.get(field_name))
        for field_name, vocab in vocabs.items()
    }


def one_hot_encode_id(category_id: int, size: int) -> list[float]:
    if size <= 0:
        raise ValueError("One-hot size must be positive.")
    if category_id < 0 or category_id >= size:
        raise ValueError(f"Category id {category_id} is outside one-hot size {size}.")
    values = [0.0] * size
    values[category_id] = 1.0
    return values


def fit_numeric_standardizer(
    rows: Sequence[Sequence[float]],
    *,
    feature_names: Sequence[str] | None = None,
    epsilon: float = 1e-8,
) -> NumericStandardizationParams:
    if not rows:
        raise ValueError("Cannot fit numeric standardizer on an empty matrix.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All numeric rows must have the same feature count.")

    names = tuple(feature_names or [f"x{i}" for i in range(width)])
    if len(names) != width:
        raise ValueError("feature_names length must match numeric row width.")

    matrix = [[float(value) for value in row] for row in rows]
    mean = tuple(sum(row[column] for row in matrix) / len(matrix) for column in range(width))
    scale_values: list[float] = []
    for column, center in enumerate(mean):
        variance = sum((row[column] - center) ** 2 for row in matrix) / len(matrix)
        scale_values.append(max(sqrt(variance), epsilon))
    return NumericStandardizationParams(
        feature_names=names,
        mean=mean,
        scale=tuple(scale_values),
    )


def standardize_numeric_rows(
    rows: Sequence[Sequence[float]],
    params: NumericStandardizationParams,
) -> list[list[float]]:
    return [params.transform(row) for row in rows]


def encode_samples_with_vocabs(
    samples: Sequence[Mapping[str, Any]],
    vocabs: Mapping[str, CategoryVocab],
    *,
    output_field: str = "categorical_ids",
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        row[output_field] = encode_category_fields(sample, vocabs)
        encoded.append(row)
    return encoded
