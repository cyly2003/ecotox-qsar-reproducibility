"""Identity-safe helpers for frozen final-encoder tree-model experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd

from qsar_tl.training.traditional_domain_comparison import PreparedDomainData


EmbeddingCondition = Literal["raw", "raw_plus_final_encoder", "final_encoder_only"]
LATENT_PREFIX = "z_final_"
IDENTITY_COLUMNS = ("stable_record_id", "analysis_split", "model_head")


def latent_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(column for column in frame.columns if str(column).startswith(LATENT_PREFIX))
    if not columns:
        raise ValueError(f"No frozen latent columns prefixed {LATENT_PREFIX!r} were found.")
    return columns


def align_latent_matrix(data: PreparedDomainData, latents: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return latent rows in exactly the PreparedDomainData row order.

    Split and model-head identity are included in the join key deliberately: a
    matching aggregate/strict record identifier alone would not justify reuse
    across a different route or task representation context.
    """

    missing = sorted(set(IDENTITY_COLUMNS) - set(latents.columns))
    if missing:
        raise ValueError(f"Latent table lacks identity columns: {missing}")
    feature_columns = latent_feature_columns(latents)
    if latents.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Latent table has duplicate record/split/model-head identities.")
    key_frame = data.frame.loc[:, IDENTITY_COLUMNS].copy()
    if key_frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Prepared data has duplicate record/split/model-head identities.")
    indexed = latents.set_index(list(IDENTITY_COLUMNS), verify_integrity=True)
    expected = pd.MultiIndex.from_frame(key_frame)
    if not expected.isin(indexed.index).all():
        missing_count = int((~expected.isin(indexed.index)).sum())
        raise ValueError(f"Frozen latent table is missing {missing_count} prepared-data rows.")
    if not indexed.index.isin(expected).all():
        extra_count = int((~indexed.index.isin(expected)).sum())
        raise ValueError(f"Frozen latent table has {extra_count} rows outside the prepared-data boundary.")
    aligned = indexed.loc[expected, list(feature_columns)].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(aligned).all():
        raise ValueError("Frozen latent matrix contains non-finite values.")
    return aligned, feature_columns


def condition_data(
    data: PreparedDomainData,
    *,
    latents: pd.DataFrame,
    condition: EmbeddingCondition,
) -> PreparedDomainData:
    """Create a feature-contract variant without changing rows, targets, or splits."""

    if condition == "raw":
        return data
    matrix, feature_columns = align_latent_matrix(data, latents)
    if condition == "raw_plus_final_encoder":
        return replace(
            data,
            molecular_matrix=np.concatenate([data.molecular_matrix, matrix], axis=1).astype(
                np.float32, copy=False
            ),
            molecular_feature_names=(*data.molecular_feature_names, *feature_columns),
            audit={
                **data.audit,
                "embedding_condition": condition,
                "frozen_embedding_dim": int(matrix.shape[1]),
            },
        )
    if condition == "final_encoder_only":
        return replace(
            data,
            molecular_matrix=matrix,
            context_numeric=np.zeros((len(data.frame), 0), dtype=np.float32),
            molecular_feature_names=feature_columns,
            context_numeric_names=(),
            categorical_names=(),
            effect_level_numeric_indices=(),
            effect_level_categorical_names=(),
            audit={
                **data.audit,
                "embedding_condition": condition,
                "frozen_embedding_dim": int(matrix.shape[1]),
            },
        )
    raise ValueError(f"Unsupported embedding condition: {condition}")
