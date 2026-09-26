from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from qsar_tl.training.baseline import load_split_frame
from qsar_tl.training.deep_experiment import MolecularFeatureBuilder


DEFAULT_TAXON_COLUMNS = ("kingdom", "phylum", "class_name", "tax_order", "family")


@dataclass(frozen=True)
class ApplicationDomainConfig:
    fingerprint_size: int = 512
    molecular_cache_path: str | Path | None = None
    pca_components: int = 32
    tanimoto_threshold: float = 0.5
    taxon_columns: tuple[str, ...] = DEFAULT_TAXON_COLUMNS
    taxon_similarity_threshold: float = 0.8


@dataclass(frozen=True)
class ApplicationDomainResult:
    report_path: Path
    manifest_path: Path
    rows: int
    train_rows: int
    williams_critical_h: float
    pca_components_used: int


def build_application_domain_report(
    db_path: str | Path,
    *,
    split_name: str,
    out_path: str | Path,
    source_table: str | None = None,
    limit: int | None = None,
    config: ApplicationDomainConfig | None = None,
) -> ApplicationDomainResult:
    cfg = config or ApplicationDomainConfig()
    frame = load_split_frame(db_path, split_name=split_name, source_table=source_table, limit=limit)
    if frame.empty:
        raise ValueError("No rows available for application-domain report.")
    if "split_part" not in frame.columns:
        raise ValueError("Application-domain report requires split_part.")

    train_mask = frame["split_part"].astype("string").str.lower() == "train"
    if not bool(train_mask.any()):
        raise ValueError("Application-domain report requires at least one train row.")

    descriptor_matrix, fingerprint_matrix, encoder_source = build_molecular_matrices(frame, cfg)
    leverage, critical_h, pca_components_used = williams_leverage(
        descriptor_matrix,
        fingerprint_matrix,
        train_mask.to_numpy(dtype=bool),
        pca_components=cfg.pca_components,
    )
    tanimoto = max_tanimoto_to_train(fingerprint_matrix, train_mask.to_numpy(dtype=bool))
    taxon_similarity = max_taxon_similarity_to_train(frame, train_mask.to_numpy(dtype=bool), cfg.taxon_columns)
    taxon_distance = 1.0 - taxon_similarity
    species_context = species_context_flags(frame, train_mask.to_numpy(dtype=bool))

    output = frame[metadata_columns(frame)].copy()
    output["williams_leverage"] = leverage
    output["williams_critical_h"] = critical_h
    output["chemical_in_domain_williams"] = leverage <= critical_h
    output["max_tanimoto_to_train"] = tanimoto
    output["chemical_in_domain_tanimoto"] = tanimoto >= cfg.tanimoto_threshold
    output["chemical_in_domain_either"] = output["chemical_in_domain_williams"] | output["chemical_in_domain_tanimoto"]
    output["max_taxon_similarity_to_train"] = taxon_similarity
    output["taxon_distance_to_train"] = taxon_distance
    output["species_in_domain_taxon"] = taxon_similarity >= cfg.taxon_similarity_threshold
    for column, values in species_context.items():
        output[column] = values
    output["overall_in_domain"] = output["chemical_in_domain_either"] & output["species_in_domain_taxon"]
    output["ad_warning"] = [
        ad_warning(chemical_ok, species_ok)
        for chemical_ok, species_ok in zip(output["chemical_in_domain_either"], output["species_in_domain_taxon"])
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    manifest = {
        "split_name": split_name,
        "source_table": source_table,
        "limit": limit,
        "rows": int(len(output)),
        "train_rows": int(train_mask.sum()),
        "fingerprint_size": cfg.fingerprint_size,
        "encoder_source": encoder_source,
        "pca_components_requested": int(cfg.pca_components),
        "pca_components_used": int(pca_components_used),
        "williams_critical_h": float(critical_h),
        "tanimoto_threshold": float(cfg.tanimoto_threshold),
        "taxon_columns": list(cfg.taxon_columns),
        "taxon_similarity_threshold": float(cfg.taxon_similarity_threshold),
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ApplicationDomainResult(
        report_path=out,
        manifest_path=manifest_path,
        rows=int(len(output)),
        train_rows=int(train_mask.sum()),
        williams_critical_h=float(critical_h),
        pca_components_used=int(pca_components_used),
    )


def build_molecular_matrices(
    frame: pd.DataFrame,
    config: ApplicationDomainConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    encoder = MolecularFeatureBuilder(
        fingerprint_size=config.fingerprint_size,
        cache_path=config.molecular_cache_path,
    )
    descriptors: list[list[float]] = []
    fingerprints: list[list[float]] = []
    for smiles in frame.get("smiles", pd.Series([""] * len(frame))):
        descriptor_row, fingerprint_row = encoder.encode(smiles)
        descriptors.append([safe_float(value) for value in descriptor_row])
        fingerprints.append([1.0 if safe_float(value) > 0.0 else 0.0 for value in fingerprint_row])
    return (
        np.asarray(descriptors, dtype=float),
        np.asarray(fingerprints, dtype=float),
        encoder.source,
    )


def williams_leverage(
    descriptor_matrix: np.ndarray,
    fingerprint_matrix: np.ndarray,
    train_mask: np.ndarray,
    *,
    pca_components: int,
) -> tuple[np.ndarray, float, int]:
    if descriptor_matrix.shape[0] != fingerprint_matrix.shape[0]:
        raise ValueError("Descriptor and fingerprint matrices must have the same row count.")
    train_count = int(train_mask.sum())
    if train_count < 2:
        return np.zeros(descriptor_matrix.shape[0], dtype=float), 0.0, 0

    descriptor_clean = np.nan_to_num(descriptor_matrix, nan=0.0)
    fingerprint_clean = np.nan_to_num(fingerprint_matrix, nan=0.0)
    descriptor_scaled = standardize_from_train(descriptor_clean, train_mask, with_mean=True)
    fingerprint_scaled = standardize_from_train(fingerprint_clean, train_mask, with_mean=False)
    train_fp = fingerprint_scaled[train_mask]
    max_components = min(int(pca_components), train_count - 1, fingerprint_matrix.shape[1])
    if max_components > 0 and train_fp.shape[1] > 0:
        fingerprint_features, components_used = pca_transform_from_train(
            fingerprint_scaled,
            train_mask,
            n_components=max_components,
        )
    else:
        fingerprint_features = np.zeros((fingerprint_matrix.shape[0], 0), dtype=float)
        components_used = 0

    feature_matrix = np.hstack([descriptor_scaled, fingerprint_features])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    design_matrix = np.hstack([np.ones((feature_matrix.shape[0], 1)), feature_matrix])
    train_design = design_matrix[train_mask]
    xtx_inverse = np.linalg.pinv(train_design.T @ train_design)
    leverage = np.einsum("ij,jk,ik->i", design_matrix, xtx_inverse, design_matrix)
    leverage = np.maximum(leverage, 0.0)
    parameter_count = train_design.shape[1]
    critical_h = 3.0 * parameter_count / train_count
    return leverage, float(critical_h), components_used


def standardize_from_train(matrix: np.ndarray, train_mask: np.ndarray, *, with_mean: bool) -> np.ndarray:
    train = matrix[train_mask]
    if train.size == 0:
        return np.zeros_like(matrix, dtype=float)
    mean = train.mean(axis=0) if with_mean else np.zeros(matrix.shape[1], dtype=float)
    std = train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (matrix - mean) / std


def pca_transform_from_train(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, int]:
    train = matrix[train_mask]
    if train.size == 0 or n_components <= 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), 0
    center = train.mean(axis=0)
    train_centered = train - center
    try:
        _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((matrix.shape[0], 0), dtype=float), 0
    components_used = min(int(n_components), vt.shape[0])
    if components_used <= 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), 0
    components = vt[:components_used].T
    return (matrix - center) @ components, components_used


def max_tanimoto_to_train(fingerprint_matrix: np.ndarray, train_mask: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    train = fingerprint_matrix[train_mask].astype(bool)
    query = fingerprint_matrix.astype(bool)
    if train.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=float)

    train_counts = train.sum(axis=1).astype(float)
    maxima = np.zeros(query.shape[0], dtype=float)
    for start in range(0, query.shape[0], batch_size):
        batch = query[start : start + batch_size]
        intersections = batch.astype(float) @ train.astype(float).T
        batch_counts = batch.sum(axis=1).astype(float)[:, None]
        denominators = batch_counts + train_counts[None, :] - intersections
        similarities = np.divide(intersections, denominators, out=np.zeros_like(intersections), where=denominators > 0)
        maxima[start : start + batch_size] = similarities.max(axis=1)
    return maxima


def max_taxon_similarity_to_train(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    taxon_columns: Sequence[str],
) -> np.ndarray:
    profiles = [taxon_profile(row, taxon_columns) for _, row in frame.iterrows()]
    train_profiles = [profile for profile, is_train in zip(profiles, train_mask) if is_train]
    if not train_profiles:
        return np.zeros(len(profiles), dtype=float)
    result = []
    for profile in profiles:
        result.append(max(taxon_prefix_similarity(profile, train_profile) for train_profile in train_profiles))
    return np.asarray(result, dtype=float)


def taxon_profile(row: pd.Series, taxon_columns: Sequence[str]) -> tuple[str, ...]:
    values = []
    for column in taxon_columns:
        raw = row.get(column, "")
        text = "" if pd.isna(raw) else str(raw).strip().lower()
        values.append(text)
    return tuple(values)


def taxon_prefix_similarity(profile: Sequence[str], train_profile: Sequence[str]) -> float:
    if not profile or not train_profile:
        return 0.0
    max_levels = min(len(profile), len(train_profile))
    matched = 0
    for query_value, train_value in zip(profile[:max_levels], train_profile[:max_levels]):
        if not query_value or not train_value or query_value != train_value:
            break
        matched += 1
    return matched / max_levels if max_levels else 0.0


def metadata_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "aggregate_id",
        "record_id",
        "split_name",
        "split_part",
        "task_head",
        "target_name",
        "task_family",
        "target_basis",
        "medium_domain",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
        "species_number",
        "latin_name",
        *DEFAULT_TAXON_COLUMNS,
        "genus",
        "species",
        "organism_lifestage",
    ]
    return [column for column in preferred if column in frame.columns]


def species_context_flags(frame: pd.DataFrame, train_mask: np.ndarray) -> dict[str, list[bool]]:
    train_frame = frame.loc[train_mask]
    latin_seen = normalized_set(train_frame, "latin_name")
    genus_seen = normalized_set(train_frame, "genus")
    family_seen = normalized_set(train_frame, "family")
    order_seen = normalized_set(train_frame, "tax_order")
    lifestage_seen = normalized_set(train_frame, "organism_lifestage")
    species_task_seen = {
        (normalize_text(row.get("latin_name")), normalize_text(row.get("task_family")))
        for _, row in train_frame.iterrows()
        if normalize_text(row.get("latin_name")) and normalize_text(row.get("task_family"))
    }
    return {
        "species_seen_train": [normalize_text(value) in latin_seen for value in column_values(frame, "latin_name")],
        "genus_seen_train": [normalize_text(value) in genus_seen for value in column_values(frame, "genus")],
        "family_seen_train": [normalize_text(value) in family_seen for value in column_values(frame, "family")],
        "order_seen_train": [normalize_text(value) in order_seen for value in column_values(frame, "tax_order")],
        "life_stage_seen_train": [
            normalize_text(value) in lifestage_seen for value in column_values(frame, "organism_lifestage")
        ],
        "species_task_family_seen_train": [
            (normalize_text(row.get("latin_name")), normalize_text(row.get("task_family"))) in species_task_seen
            for _, row in frame.iterrows()
        ],
    }


def column_values(frame: pd.DataFrame, column: str) -> list[object]:
    if column not in frame.columns:
        return [None] * len(frame)
    return list(frame[column])


def normalized_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    return {normalize_text(value) for value in frame[column] if normalize_text(value)}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().lower()


def ad_warning(chemical_in_domain: Any, species_in_domain: Any) -> str:
    chemical_ok = bool(chemical_in_domain)
    species_ok = bool(species_in_domain)
    if chemical_ok and species_ok:
        return "in_domain"
    if not chemical_ok and not species_ok:
        return "chemical_and_species_extrapolation"
    if not chemical_ok:
        return "chemical_extrapolation"
    return "species_extrapolation"


def safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0
