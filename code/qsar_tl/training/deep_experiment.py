from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qsar_tl.evaluation.metrics import regression_metrics
from qsar_tl.modeling.dataset import AggregatedTaskDataset
from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork, TOXICITY_BIN_LOGITS_KEY
from qsar_tl.training.censored_loss import censored_direction, censored_hinge_loss
from qsar_tl.training.baseline import (
    EVAL_SPLIT_PARTS,
    add_duration_nonlinear_features,
    load_split_frame,
    task_skip_reason,
)
from qsar_tl.training.deep_train import (
    DeepTrainingConfig,
    build_optimizer,
    collate_aggregated_task_batch,
    dataloader_runtime_options,
    graph_to_device,
    set_torch_seed,
)
from qsar_tl.training.ordinal_binning import ordinal_softmax_loss
from qsar_tl.training.toxicity_binning import (
    ToxicityBinningConfig,
    assign_toxicity_bin,
    load_toxicity_bin_scheme,
    summarize_toxicity_bins,
    toxicity_bin_class_count,
)


MOLECULAR_DESCRIPTOR_NAMES = (
    "MolWt",
    "TPSA",
    "MolLogP",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "RingCount",
    "RotatableBonds",
)
MOLECULAR_SIZE_RELATED_DESCRIPTOR_NAMES = (
    "MolWt",
    "TPSA",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "RingCount",
    "RotatableBonds",
)
EFFECT_LEVEL_NUMERIC_COLUMNS = (
    "effect_level_x",
    "effect_level_x_fraction",
    "effect_level_x_log1p",
    "effect_level_x_present",
)
DURATION_NUMERIC_COLUMNS = (
    "duration_bin_h",
    "duration_log1p_h",
    "duration_sqrt_h",
    "duration_inv_log1p_h",
    "duration_rbf_24h",
    "duration_rbf_48h",
    "duration_rbf_96h",
    "duration_rbf_168h",
    "duration_rbf_336h",
    "duration_rbf_720h",
)
CONTEXT_NUMERIC_COLUMNS = EFFECT_LEVEL_NUMERIC_COLUMNS + DURATION_NUMERIC_COLUMNS
RAW_NUMERIC_CLIP_ABS = 1.0e12
SOURCE_WEIGHT_CACHE_SCHEMA = "source_weight_v1"
CATEGORICAL_COLUMNS = (
    "latin_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
    "target_basis",
    "effect_family",
)
MISSING_CATEGORY_TOKEN = "<missing>"
UNKNOWN_CATEGORY_TOKEN = "<unknown>"
RARE_CATEGORY_TOKEN = "<rare>"
GLOBAL_TARGET_SCALE_KEY = "__global__"
PREDICTION_METADATA_COLUMNS = (
    "sample_id",
    "aggregate_id",
    "record_id",
    "result_ids",
    "split_name",
    "split_part",
    "task_head",
    "base_task_head",
    "model_head",
    "task_group",
    "task_family",
    "effect_level_x",
    "target_name",
    "target_family",
    "target_basis",
    "target_column",
    "target_scale_key",
    "unit_family_v2",
    "standard_unit_v2",
    "standard_value_mg_l",
    "standard_value_mol_l",
    "standard_value_mg_kg",
    "standard_value_g_ha",
    "standard_value_mg_kg_diet",
    "standard_value_mg_kg_bw_day",
    "standard_value_mol_kg",
    "molecular_weight_g_mol_used",
    "target_transform",
    "parent_target_name",
    "parent_target_family",
    "parent_target_basis",
    "conversion_path",
    "original_split_part",
    "medium_domain",
    "primary_medium",
    "media_type",
    "cas_number",
    "dtxsid",
    "chemical_name",
    "smiles",
    "species_number",
    "latin_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "organism_lifestage",
    "effect_family",
    "adapter_name",
    "adapter_id",
    "toxicity_bin_index",
    "toxicity_bin_label",
    "toxicity_bin_scheme",
    "toxicity_bin_source",
    "toxicity_bin_boundary_flag",
    "toxicity_bin_status",
    "toxicity_bin_value",
    "toxicity_bin_value_unit",
    "toxicity_bin_conversion",
    "perturbation_replicate",
    "perturbation_numeric_noise_std",
)


def bounded_numeric_value(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return max(-RAW_NUMERIC_CLIP_ABS, min(RAW_NUMERIC_CLIP_ABS, number))


def bounded_numeric_matrix(matrix: np.ndarray) -> np.ndarray:
    bounded = np.nan_to_num(
        matrix.astype(float, copy=False),
        nan=0.0,
        posinf=RAW_NUMERIC_CLIP_ABS,
        neginf=-RAW_NUMERIC_CLIP_ABS,
    )
    return np.clip(bounded, -RAW_NUMERIC_CLIP_ABS, RAW_NUMERIC_CLIP_ABS)


@dataclass(frozen=True)
class DeepExperimentResult:
    out_dir: Path
    metrics_path: Path
    history_path: Path
    manifest_path: Path
    preprocessing_path: Path
    predictions_path: Path
    best_model_path: Path | None
    encoder_source: str
    trained_tasks: tuple[str, ...]
    rows: int
    ablation: str
    best_epoch: int
    early_stopping_enabled: bool


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_descriptors: bool = True
    use_fingerprint: bool = True
    use_molecular_graph: bool = False
    use_context_numeric: bool = True
    use_effect_level_features: bool = True
    use_duration_features: bool = True
    use_species_lifestage: bool = True
    use_other_categorical_context: bool = True
    use_molecular_residual: bool = True
    use_medium_adapter: bool = True
    masked_descriptor_names: tuple[str, ...] = ()
    species_lifestage_columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TargetScaler:
    mode: str
    target_column: str
    fit_split_parts: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    def transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return (float(value) - float(stat["mean"])) / float(stat["std"])

    def inverse_transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return float(value) * float(stat["std"]) + float(stat["mean"])

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_column": self.target_column,
            "fit_split_parts": list(self.fit_split_parts),
            "stats": self.stats,
        }


@dataclass(frozen=True)
class FeatureNoiseConfig:
    train_replicates: int = 1
    finetune_replicates: int = 1
    numeric_noise_std: float = 0.0
    target_noise_std: float = 0.0
    seed: int = 42

    def active_for_train(self) -> bool:
        return self.train_replicates > 1 and (self.numeric_noise_std > 0 or self.target_noise_std > 0)

    def active_for_finetune(self) -> bool:
        return self.finetune_replicates > 1 and (self.numeric_noise_std > 0 or self.target_noise_std > 0)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "train_replicates": self.train_replicates,
            "finetune_replicates": self.finetune_replicates,
            "numeric_noise_std": self.numeric_noise_std,
            "target_noise_std": self.target_noise_std,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PerturbationConfig:
    enabled: bool = False
    split_parts: tuple[str, ...] = ("test",)
    replicates: int = 1
    numeric_noise_std: float = 0.0
    seed: int = 42

    def active(self) -> bool:
        return self.enabled and self.replicates > 1 and self.numeric_noise_std > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "split_parts": list(self.split_parts),
            "replicates": self.replicates,
            "numeric_noise_std": self.numeric_noise_std,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SourceWeightingConfig:
    enabled: bool = False
    method: str = "none"
    alpha: float = 0.0
    source_split_parts: tuple[str, ...] = ("train",)
    target_split_parts: tuple[str, ...] = ("finetune",)
    source_domains: tuple[str, ...] = ("aquatic",)
    target_domains: tuple[str, ...] = ("soil",)
    min_weight: float = 0.25
    max_weight: float = 2.0

    def active(self) -> bool:
        return self.enabled and self.method not in {"", "none", "off", "false"} and self.alpha > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "alpha": self.alpha,
            "source_split_parts": list(self.source_split_parts),
            "target_split_parts": list(self.target_split_parts),
            "source_domains": list(self.source_domains),
            "target_domains": list(self.target_domains),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class EffectLevelWeightingConfig:
    enabled: bool = False
    beta: float = 0.0
    split_parts: tuple[str, ...] = ("train", "finetune")
    task_prefixes: tuple[str, ...] = ("ECx", "LCx", "ICx", "LDx")
    min_weight: float = 0.5
    max_weight: float = 3.0

    def active(self) -> bool:
        return self.enabled and self.beta > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "beta": self.beta,
            "split_parts": list(self.split_parts),
            "task_prefixes": list(self.task_prefixes),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class DomainAlignmentConfig:
    enabled: bool = False
    method: str = "none"
    weight: float = 0.0
    source_split_parts: tuple[str, ...] = ("train",)
    target_split_parts: tuple[str, ...] = ("finetune",)
    source_domains: tuple[str, ...] = ("aquatic",)
    target_domains: tuple[str, ...] = ("soil",)
    phases: tuple[str, ...] = ("pretrain",)

    def active(self) -> bool:
        return self.enabled and self.method == "coral" and self.weight > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "weight": self.weight,
            "source_split_parts": list(self.source_split_parts),
            "target_split_parts": list(self.target_split_parts),
            "source_domains": list(self.source_domains),
            "target_domains": list(self.target_domains),
            "phases": list(self.phases),
        }


@dataclass(frozen=True)
class CensoredLossConfig:
    enabled: bool = False
    method: str = "hinge"
    weight: float = 0.0
    margin: float = 0.0
    split_parts: tuple[str, ...] = ("train", "finetune")
    include_ops: tuple[str, ...] = ("<", "<=", ">", ">=")

    def active(self) -> bool:
        return self.enabled and self.method == "hinge" and self.weight > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "weight": self.weight,
            "margin": self.margin,
            "split_parts": list(self.split_parts),
            "include_ops": list(self.include_ops),
        }


@dataclass(frozen=True)
class SwaConfig:
    enabled: bool = False
    phase: str = "finetune"
    start_epoch: int = 15

    def active(self) -> bool:
        return self.enabled and self.start_epoch > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "start_epoch": self.start_epoch,
        }


@dataclass(frozen=True)
class ZScoreCorrectionConfig:
    enabled: bool = True
    threshold: float = 6.0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": "clip_standardized_zscore",
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ZScoreCorrection:
    enabled: bool
    threshold: float
    feature_names: tuple[str, ...]
    fit_split_parts: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    def transform(self, index: int, value: float) -> float:
        if not self.enabled:
            return float(value)
        if not math.isfinite(float(value)):
            return 0.0
        return max(-self.threshold, min(self.threshold, float(value)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": "clip_standardized_zscore",
            "threshold": self.threshold,
            "fit_split_parts": list(self.fit_split_parts),
            "feature_names": list(self.feature_names),
            "stats": self.stats,
        }


ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": AblationSpec("full"),
    "no_fingerprint": AblationSpec("no_fingerprint", use_fingerprint=False),
    "no_descriptors": AblationSpec("no_descriptors", use_descriptors=False),
    # Preserve all experimental context while retaining only the named molecular view.
    # The older `descriptors_only` preset intentionally removes context and remains
    # unchanged for historical comparability.
    "descriptors_with_context": AblationSpec("descriptors_with_context", use_fingerprint=False),
    "fingerprint_with_context": AblationSpec("fingerprint_with_context", use_descriptors=False),
    "no_molecular_input": AblationSpec(
        "no_molecular_input",
        use_descriptors=False,
        use_fingerprint=False,
    ),
    "no_molecular_size_descriptors": AblationSpec(
        "no_molecular_size_descriptors",
        masked_descriptor_names=MOLECULAR_SIZE_RELATED_DESCRIPTOR_NAMES,
    ),
    "no_molwt": AblationSpec(
        "no_molwt",
        masked_descriptor_names=("MolWt",),
    ),
    "descriptors_only": AblationSpec(
        "descriptors_only",
        use_fingerprint=False,
        use_context_numeric=False,
        use_duration_features=False,
        use_species_lifestage=False,
        use_other_categorical_context=False,
        use_medium_adapter=False,
    ),
    "fingerprint_only": AblationSpec(
        "fingerprint_only",
        use_descriptors=False,
        use_context_numeric=False,
        use_duration_features=False,
        use_species_lifestage=False,
        use_other_categorical_context=False,
        use_medium_adapter=False,
    ),
    "no_effect_level": AblationSpec("no_effect_level", use_effect_level_features=False),
    "graph_only_molecule": AblationSpec(
        "graph_only_molecule",
        use_descriptors=False,
        use_fingerprint=False,
        use_molecular_graph=True,
    ),
    "no_species_lifestage": AblationSpec("no_species_lifestage", use_species_lifestage=False),
    "taxonomy_identity_only": AblationSpec(
        "taxonomy_identity_only",
        species_lifestage_columns=("latin_name", "organism_lifestage"),
    ),
    "taxonomy_coarse_only": AblationSpec(
        "taxonomy_coarse_only",
        species_lifestage_columns=(
            "kingdom",
            "phylum",
            "class_name",
            "tax_order",
            "family",
            "organism_lifestage",
        ),
    ),
    "no_duration": AblationSpec("no_duration", use_duration_features=False),
    "no_other_categorical_context": AblationSpec(
        "no_other_categorical_context",
        use_other_categorical_context=False,
    ),
    "no_context": AblationSpec(
        "no_context",
        use_context_numeric=False,
        use_duration_features=False,
        use_species_lifestage=False,
        use_other_categorical_context=False,
        use_medium_adapter=False,
    ),
    "no_molecular_residual": AblationSpec("no_molecular_residual", use_molecular_residual=False),
    "no_medium_adapter": AblationSpec("no_medium_adapter", use_medium_adapter=False),
}

SPECIES_LIFESTAGE_COLUMNS = frozenset(
    (
        "latin_name",
        "kingdom",
        "phylum",
        "class_name",
        "tax_order",
        "family",
        "genus",
        "species",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "organism_lifestage",
    )
)
DURATION_CONTEXT_COLUMNS = frozenset(DURATION_NUMERIC_COLUMNS)


def run_deep_experiment(
    db_path: str | Path,
    *,
    split_name: str,
    out_dir: str | Path,
    config: Mapping[str, Any],
    limit: int | None = None,
    seed: int = 42,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    scheduler: str | None = None,
    device: str | None = None,
    ablation: str = "full",
    source_table: str | None = None,
    early_stopping: bool | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    validation_fraction: float | None = None,
    validation_seed: int | None = None,
    monitor_split: str | None = None,
    finetune_epochs: int | None = None,
    finetune_learning_rate: float | None = None,
    finetune_batch_size: int | None = None,
    finetune_scheduler: str | None = None,
    finetune_freeze: str | None = None,
    finetune_validation_fraction: float | None = None,
    finetune_validation_seed: int | None = None,
    finetune_mgkg_epochs: int | None = None,
    finetune_mgkg_learning_rate: float | None = None,
    finetune_mgkg_batch_size: int | None = None,
    finetune_mgkg_scheduler: str | None = None,
    finetune_mgkg_freeze: str | None = None,
    finetune_mgkg_validation_fraction: float | None = None,
    finetune_mgkg_validation_seed: int | None = None,
    finetune_mgkg_monitor_split: str | None = None,
    finetune_mgkg_early_stopping: bool | None = None,
    finetune_mgkg_head_only_epochs: int | None = None,
    finetune_mgkg_trunk_learning_rate: float | None = None,
    finetune_mgkg_replay_fraction: float | None = None,
    finetune_mgkg_toxicity_bin_loss_weight: float | None = None,
    finetune_mgkg_mse_loss_weight: float | None = None,
    finetune_mgkg_target_bin_sampling: bool | None = None,
    finetune_mgkg_target_bins: int | None = None,
    finetune_mgkg_sampling_min_weight: float | None = None,
    finetune_mgkg_sampling_max_weight: float | None = None,
    finetune_mgkg_hierarchical_head: bool | None = None,
    finetune_mgkg_hierarchical_family_tau: float | None = None,
    finetune_mgkg_hierarchical_task_tau: float | None = None,
    finetune_mgkg_init_checkpoint: str | Path | None = None,
    export_finetune_mgkg_init_checkpoint: str | Path | None = None,
    mgkg_residual_adapter: bool | None = None,
    mgkg_residual_adapter_bottleneck: int | None = None,
    head_routing: str | None = None,
    allow_mixed_target_dimensions: bool | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    target_standardization: str | None = None,
    augment_train_replicates: int | None = None,
    augment_finetune_replicates: int | None = None,
    augment_numeric_noise_std: float | None = None,
    augment_target_noise_std: float | None = None,
    test_noise_replicates: int | None = None,
    test_noise_numeric_std: float | None = None,
    feature_zscore_correction: bool | None = None,
    feature_zscore_threshold: float | None = None,
    metric_min_n: int | None = None,
    source_weighting_method: str | None = None,
    source_weighting_alpha: float | None = None,
    source_weight_cache_dir: str | Path | None = None,
    effect_level_weighting_enabled: bool | None = None,
    effect_level_weighting_beta: float | None = None,
    toxicity_binning_enabled: bool | None = None,
    toxicity_binning_mode: str | None = None,
    toxicity_binning_loss_weight: float | None = None,
    toxicity_binning_scheme: str | None = None,
    censored_loss_enabled: bool | None = None,
    censored_loss_weight: float | None = None,
    censored_loss_margin: float | None = None,
    domain_alignment_method: str | None = None,
    domain_alignment_weight: float | None = None,
    swa_enabled: bool | None = None,
    swa_start_epoch: int | None = None,
    swa_phase: str | None = None,
    evaluation_checkpoint: str | Path | None = None,
    prediction_split_parts: tuple[str, ...] | None = None,
) -> DeepExperimentResult:
    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_spec = get_ablation_spec(ablation)
    if not bool(config.get("model", {}).get("use_medium_adapters", True)):
        ablation_spec = replace(ablation_spec, use_medium_adapter=False)

    train_cfg = config.get("training", {})
    mixed_target_dimensions_allowed = bool(
        allow_mixed_target_dimensions
        if allow_mixed_target_dimensions is not None
        else train_cfg.get("allow_mixed_target_dimensions", False)
    )
    experiment_cfg = (
        config.get("experiment", {})
        if isinstance(config.get("experiment", {}), Mapping)
        else {}
    )
    development_only = bool(experiment_cfg.get("development_only", False))
    development_split_parts = (
        tuple(experiment_cfg.get("development_split_parts", ("train", "finetune", "finetune_mgkg", "valid")))
        if development_only
        else None
    )
    frame = load_split_frame(
        db_path,
        split_name=split_name,
        source_table=source_table,
        limit=limit,
        allow_mixed_target_dimensions=mixed_target_dimensions_allowed,
        split_parts=development_split_parts,
    )
    split_join_audit = dict(frame.attrs.get("split_join_audit", {}))
    frame = add_duration_nonlinear_features(frame)
    target_column = "target_value_median" if "target_value_median" in frame.columns else "target_value"
    frame = frame[frame[target_column].notna()].copy()
    if frame.empty:
        raise ValueError("No target rows available for deep training.")

    head_routing_mode = normalize_head_routing(
        head_routing if head_routing is not None else train_cfg.get("head_routing", "task")
    )
    frame = apply_head_routing(frame, mode=head_routing_mode)
    allowed_task_heads = tuple(
        str(value)
        for value in experiment_cfg.get("allowed_task_heads", ())
        if str(value).strip()
    )
    if allowed_task_heads:
        frame = frame[frame["task_head"].astype(str).isin(set(allowed_task_heads))].copy()
        if frame.empty:
            raise ValueError("experiment.allowed_task_heads removed every development row.")

    filter_cfg = config.get("experiment", {}).get("task_filter", {})
    min_total = int(filter_cfg.get("min_total", 0))
    min_train = int(filter_cfg.get("min_train", 1))
    min_eval = int(filter_cfg.get("min_eval", 1))
    skipped_tasks: dict[str, str] = {}
    kept_frames = []
    for task_head, task_frame in frame.groupby("task_head", dropna=False):
        task_label = "default" if task_head is None else str(task_head)
        # A task introduced only in the third stage has no stage-1 `train`
        # rows. For thresholding, its dedicated finetune set is its training
        # population; the actual stage ordering remains unchanged below.
        filter_frame = task_frame
        if "finetune_mgkg" in set(task_frame.get("split_part", [])):
            filter_frame = task_frame.copy()
            filter_frame["split_part"] = filter_frame["split_part"].replace(
                {"finetune_mgkg": "train"}
            )
        reason = task_skip_reason(
            filter_frame,
            min_total=min_total,
            min_train=min_train,
            min_eval=min_eval,
        )
        if reason is not None:
            skipped_tasks[task_label] = reason
            continue
        kept_frames.append(task_frame)
    if not kept_frames:
        raise ValueError(f"No task heads passed sample thresholds. Skipped: {skipped_tasks}")
    frame = _concat_frames(kept_frames)

    reporting_cfg = config.get("reporting", {}) if isinstance(config.get("reporting", {}), dict) else {}
    metric_min_group_n = max(1, int(metric_min_n if metric_min_n is not None else reporting_cfg.get("min_metric_group_n", 5)))
    finetune_cfg = train_cfg.get("finetune", {}) if isinstance(train_cfg.get("finetune", {}), dict) else {}
    finetune_mgkg_cfg = (
        train_cfg.get("finetune_mgkg", {})
        if isinstance(train_cfg.get("finetune_mgkg", {}), dict)
        else {}
    )
    augmentation_cfg = _feature_noise_config(
        train_cfg,
        seed=seed,
        train_replicates_override=augment_train_replicates,
        finetune_replicates_override=augment_finetune_replicates,
        numeric_noise_std_override=augment_numeric_noise_std,
        target_noise_std_override=augment_target_noise_std,
    )
    perturbation_cfg = _perturbation_config(
        train_cfg,
        seed=seed,
        replicates_override=test_noise_replicates,
        numeric_noise_std_override=test_noise_numeric_std,
    )
    source_weighting_cfg = _source_weighting_config(
        train_cfg,
        method_override=source_weighting_method,
        alpha_override=source_weighting_alpha,
    )
    effect_level_weighting_cfg = _effect_level_weighting_config(
        train_cfg,
        enabled_override=effect_level_weighting_enabled,
        beta_override=effect_level_weighting_beta,
    )
    toxicity_binning_cfg = _toxicity_binning_config(
        train_cfg,
        enabled_override=toxicity_binning_enabled,
        mode_override=toxicity_binning_mode,
        loss_weight_override=toxicity_binning_loss_weight,
        scheme_override=toxicity_binning_scheme,
    )
    toxicity_bin_scheme = load_toxicity_bin_scheme(toxicity_binning_cfg.scheme)
    toxicity_bin_count = toxicity_bin_class_count(toxicity_bin_scheme) if toxicity_binning_cfg.enabled else 0
    censored_loss_cfg = _censored_loss_config(
        train_cfg,
        enabled_override=censored_loss_enabled,
        weight_override=censored_loss_weight,
        margin_override=censored_loss_margin,
    )
    domain_alignment_cfg = _domain_alignment_config(
        train_cfg,
        method_override=domain_alignment_method,
        weight_override=domain_alignment_weight,
    )
    swa_cfg = _swa_config(
        train_cfg,
        enabled_override=swa_enabled,
        start_epoch_override=swa_start_epoch,
        phase_override=swa_phase,
    )
    requested_stage1_epochs = int(
        epochs if epochs is not None else train_cfg.get("epochs", 5)
    )
    requested_finetune_epochs = int(
        finetune_epochs if finetune_epochs is not None else finetune_cfg.get("epochs", 0)
    )
    finetune_requested = requested_finetune_epochs > 0
    finetune_freeze_mode = str(
        finetune_freeze if finetune_freeze is not None else finetune_cfg.get("freeze", "none")
    ).strip().lower()
    requested_finetune_mgkg_epochs = int(
        finetune_mgkg_epochs
        if finetune_mgkg_epochs is not None
        else finetune_mgkg_cfg.get("epochs", 0)
    )
    finetune_mgkg_requested = requested_finetune_mgkg_epochs > 0
    finetune_mgkg_freeze_mode = str(
        finetune_mgkg_freeze
        if finetune_mgkg_freeze is not None
        else finetune_mgkg_cfg.get("freeze", "none")
    ).strip().lower()
    finetune_mgkg_head_only_epoch_count = int(
        finetune_mgkg_head_only_epochs
        if finetune_mgkg_head_only_epochs is not None
        else finetune_mgkg_cfg.get("head_only_epochs", 0)
    )
    if finetune_mgkg_head_only_epoch_count < 0:
        raise ValueError("finetune_mgkg head_only_epochs must be non-negative.")
    if finetune_mgkg_head_only_epoch_count > requested_finetune_mgkg_epochs:
        raise ValueError("finetune_mgkg head_only_epochs cannot exceed total epochs.")
    finetune_mgkg_trunk_lr = float(
        finetune_mgkg_trunk_learning_rate
        if finetune_mgkg_trunk_learning_rate is not None
        else finetune_mgkg_cfg.get("trunk_learning_rate", 0.0)
    )
    finetune_mgkg_replay_ratio = float(
        finetune_mgkg_replay_fraction
        if finetune_mgkg_replay_fraction is not None
        else finetune_mgkg_cfg.get("soil_ptox_replay_fraction", 0.0)
    )
    if not 0.0 <= finetune_mgkg_replay_ratio <= 0.5:
        raise ValueError("finetune_mgkg soil_ptox_replay_fraction must be in [0, 0.5].")
    if finetune_mgkg_replay_ratio > 0 and head_routing_mode != "task_target":
        raise ValueError("Soil pTox replay requires task_target head routing.")
    target_bin_sampling_enabled = bool(
        finetune_mgkg_target_bin_sampling
        if finetune_mgkg_target_bin_sampling is not None
        else finetune_mgkg_cfg.get("target_bin_sampling", False)
    )
    target_bin_sampling_bins = int(
        finetune_mgkg_target_bins
        if finetune_mgkg_target_bins is not None
        else finetune_mgkg_cfg.get("target_bins", 10)
    )
    target_bin_sampling_min_weight = float(
        finetune_mgkg_sampling_min_weight
        if finetune_mgkg_sampling_min_weight is not None
        else finetune_mgkg_cfg.get("target_bin_sampling_min_weight", 0.5)
    )
    target_bin_sampling_max_weight = float(
        finetune_mgkg_sampling_max_weight
        if finetune_mgkg_sampling_max_weight is not None
        else finetune_mgkg_cfg.get("target_bin_sampling_max_weight", 2.0)
    )
    if target_bin_sampling_bins < 2:
        raise ValueError("finetune_mgkg target_bins must be at least 2.")
    if not (
        0 < target_bin_sampling_min_weight
        <= 1.0
        <= target_bin_sampling_max_weight
    ):
        raise ValueError(
            "finetune_mgkg target-bin sampling weights require 0 < min <= 1 <= max "
            "so each task's expected sampling mass can remain unchanged."
        )
    if target_bin_sampling_enabled and finetune_mgkg_replay_ratio > 0:
        raise ValueError(
            "Stage-3 equal-width target-bin sampling cannot be combined with soil pTox replay."
        )
    hierarchical_head_enabled = bool(
        finetune_mgkg_hierarchical_head
        if finetune_mgkg_hierarchical_head is not None
        else finetune_mgkg_cfg.get("hierarchical_head", False)
    )
    hierarchical_family_tau = float(
        finetune_mgkg_hierarchical_family_tau
        if finetune_mgkg_hierarchical_family_tau is not None
        else finetune_mgkg_cfg.get("hierarchical_family_tau", 128.0)
    )
    hierarchical_task_tau = float(
        finetune_mgkg_hierarchical_task_tau
        if finetune_mgkg_hierarchical_task_tau is not None
        else finetune_mgkg_cfg.get("hierarchical_task_tau", 64.0)
    )
    if hierarchical_family_tau < 0 or hierarchical_task_tau < 0:
        raise ValueError("Hierarchical residual-scale tau values must be non-negative.")
    if hierarchical_head_enabled and finetune_mgkg_replay_ratio > 0:
        raise ValueError("The hierarchical soil head cannot be combined with pTox replay.")
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    use_mgkg_residual_adapter = bool(
        mgkg_residual_adapter
        if mgkg_residual_adapter is not None
        else model_cfg.get("use_mgkg_residual_adapter", False)
    )
    mgkg_adapter_bottleneck = int(
        mgkg_residual_adapter_bottleneck
        if mgkg_residual_adapter_bottleneck is not None
        else model_cfg.get("mgkg_residual_adapter_bottleneck", 32)
    )
    if mgkg_adapter_bottleneck <= 0:
        raise ValueError("mgkg_residual_adapter_bottleneck must be positive.")

    fingerprint_size = int(config.get("features", {}).get("molecule", {}).get("morgan_n_bits", 512))
    cache_path = _molecular_cache_path(config)
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size, cache_path=cache_path)
    graph_cache_path = _molecular_graph_cache_path(config)
    graph_encoder = MolecularGraphFeatureBuilder(cache_path=graph_cache_path) if ablation_spec.use_molecular_graph else None
    if ablation_spec.use_molecular_graph and (graph_encoder is None or not graph_encoder.cache):
        raise ValueError("graph_only_molecule requires features.molecule.graph_cache with cached molecular graphs.")
    graph_encoder_cfg = _graph_encoder_config(config)
    descriptor_count = _descriptor_count(frame, encoder, ablation_spec)
    descriptor_names = encoder.descriptor_names(descriptor_count)
    numeric_feature_names = build_numeric_feature_names(descriptor_count, descriptor_names=descriptor_names)
    descriptor_encoder_cfg = _descriptor_encoder_config(config, descriptor_names)

    early_cfg = _early_stopping_config(
        config,
        enabled_override=early_stopping,
        patience_override=early_stopping_patience,
        min_delta_override=early_stopping_min_delta,
        validation_fraction_override=validation_fraction,
        monitor_split_override=monitor_split,
    )
    requested_stage1_monitor_split = str(early_cfg["monitor_split"])
    early_cfg["monitor_split"] = resolve_stage1_monitor_split(
        requested_stage1_monitor_split,
        staged_training=bool(finetune_requested or finetune_mgkg_requested),
    )
    split_probe_samples = [
        {"split_part": str(row.get("split_part")), "task_head": str(row.get("task_head"))}
        for _, row in frame.iterrows()
    ]
    train_indices = [idx for idx, sample in enumerate(split_probe_samples) if sample["split_part"] == "train"]
    actual_train_indices, validation_indices, validation_source = resolve_stage1_training_boundary(
        split_probe_samples,
        train_indices=train_indices,
        seed=int(seed if validation_seed is None else validation_seed),
        validation_fraction=early_cfg["validation_fraction"],
        monitor_split=early_cfg["monitor_split"],
        requested_epochs=requested_stage1_epochs,
        downstream_training_requested=bool(
            finetune_requested or finetune_mgkg_requested or evaluation_checkpoint is not None
        ),
    )
    early_enabled = bool(early_cfg["enabled"] and validation_indices)
    if not early_enabled:
        actual_train_indices = train_indices
        validation_indices = []
        validation_source = ""

    finetune_indices = [idx for idx, sample in enumerate(split_probe_samples) if sample["split_part"] == "finetune"]
    finetune_train_indices, finetune_validation_indices, finetune_validation_source = split_finetune_validation_indices(
        split_probe_samples,
        finetune_indices=finetune_indices,
        seed=int(seed if finetune_validation_seed is None else finetune_validation_seed),
        validation_fraction=float(
            finetune_validation_fraction
            if finetune_validation_fraction is not None
            else finetune_cfg.get("validation_fraction", 0.0)
        ),
        monitor_split=str(finetune_cfg.get("monitor_split", "auto")),
    )
    finetune_mgkg_indices = [
        idx
        for idx, sample in enumerate(split_probe_samples)
        if sample["split_part"] == "finetune_mgkg"
    ]
    finetune_mgkg_train_indices, finetune_mgkg_validation_indices, finetune_mgkg_validation_source = (
        split_finetune_validation_indices(
            split_probe_samples,
            finetune_indices=finetune_mgkg_indices,
            seed=int(seed + 20_000 if finetune_mgkg_validation_seed is None else finetune_mgkg_validation_seed),
            validation_fraction=float(
                finetune_mgkg_validation_fraction
                if finetune_mgkg_validation_fraction is not None
                else finetune_mgkg_cfg.get("validation_fraction", 0.0)
            ),
            monitor_split=str(
                finetune_mgkg_monitor_split
                if finetune_mgkg_monitor_split is not None
                else finetune_mgkg_cfg.get("monitor_split", "auto")
            ),
        )
    )
    preprocessing_indices = resolve_preprocessing_fit_indices(
        actual_train_indices=actual_train_indices,
        finetune_train_indices=finetune_train_indices,
        finetune_mgkg_train_indices=finetune_mgkg_train_indices,
        finetune_requested=bool(finetune_requested),
        finetune_mgkg_requested=bool(finetune_mgkg_requested),
        evaluation_checkpoint_requested=evaluation_checkpoint is not None,
    )
    preprocessing_frame = frame.iloc[sorted(set(preprocessing_indices))].copy()
    category_min_count = int(
        train_cfg.get(
            "categorical_min_count",
            config.get("features", {}).get("categorical_min_count", 1),
        )
    )
    categorical_maps = fit_categorical_maps(
        preprocessing_frame,
        ablation=ablation_spec,
        min_count=category_min_count,
    )
    adapter_map = fit_adapter_map(preprocessing_frame, ablation=ablation_spec)
    numeric_stats = fit_numeric_stats(
        preprocessing_frame,
        encoder,
        descriptor_names=descriptor_names,
        ablation=ablation_spec,
    )
    zscore_cfg = _zscore_correction_config(
        train_cfg,
        enabled_override=feature_zscore_correction,
        threshold_override=feature_zscore_threshold,
    )
    zscore_correction = fit_zscore_correction(
        preprocessing_frame,
        encoder,
        numeric_stats=numeric_stats,
        feature_names=numeric_feature_names,
        descriptor_names=descriptor_names,
        config=zscore_cfg,
        ablation=ablation_spec,
    )
    target_scaler = fit_target_scaler(
        frame,
        target_column=target_column,
        mode=str(target_standardization or train_cfg.get("target_standardization", "per_task_target")),
        fit_indices=preprocessing_indices,
    )
    samples = build_deep_samples(
        frame,
        encoder=encoder,
        descriptor_names=descriptor_names,
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        target_column=target_column,
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        ablation=ablation_spec,
        graph_encoder=graph_encoder,
        toxicity_binning_config=toxicity_binning_cfg,
        toxicity_bin_scheme=toxicity_bin_scheme,
    )
    censored_summary = {
        **censored_loss_cfg.to_manifest(),
        "candidate_rows": 0,
        "usable_rows": 0,
        "train_rows": 0,
        "finetune_rows": 0,
        "skipped_rows": {},
    }
    # Preserve the Stage-1 early-stopping holdout as an explicit prediction
    # label.  This permits validation-only hyperparameter selection without
    # exporting or inspecting the immutable outer-test rows.
    mark_internal_validation_samples(
        samples,
        indices=validation_indices,
        split_part="validation",
    )
    mark_internal_validation_samples(
        samples,
        indices=finetune_validation_indices,
        split_part="finetune_validation",
    )
    mark_internal_validation_samples(
        samples,
        indices=finetune_mgkg_validation_indices,
        split_part="finetune_mgkg_validation",
    )
    if censored_loss_cfg.active():
        censored_samples, censored_summary = build_censored_training_samples(
            db_path,
            frame,
            encoder=encoder,
            categorical_maps=categorical_maps,
            adapter_map=adapter_map,
            numeric_stats=numeric_stats,
            target_column=target_column,
            target_scaler=target_scaler,
            zscore_correction=zscore_correction,
            descriptor_names=descriptor_names,
            ablation=ablation_spec,
            graph_encoder=graph_encoder,
            config=censored_loss_cfg,
            kept_task_heads=tuple(sorted(set(str(sample.get("task_head")) for sample in samples))),
            split_parts=tuple(sorted({str(sample.get("split_part")) for sample in samples})),
            head_routing_mode=head_routing_mode,
        )
        censored_start = len(samples)
        samples.extend(censored_samples)
        for offset, sample in enumerate(censored_samples):
            idx = censored_start + offset
            split_part = str(sample.get("split_part", "")).lower()
            if split_part == "train":
                actual_train_indices.append(idx)
                train_indices.append(idx)
            elif split_part == "finetune":
                finetune_indices.append(idx)
                finetune_train_indices.append(idx)
            elif split_part == "finetune_mgkg":
                finetune_mgkg_indices.append(idx)
                finetune_mgkg_train_indices.append(idx)
    source_weighting_summary = apply_source_similarity_weights(
        samples,
        source_weighting_cfg,
        descriptor_names=descriptor_names,
        cache_dir=source_weight_cache_dir,
    )
    effect_level_weighting_summary = apply_effect_level_frequency_weights(
        samples,
        effect_level_weighting_cfg,
        train_indices=(
            actual_train_indices
            + finetune_train_indices
            + finetune_mgkg_train_indices
        ),
    )
    weighting_history_fields = sample_weighting_history_fields(
        source_weighting_summary,
        effect_level_weighting_summary,
    )
    target_bin_sampling_weights, target_bin_sampling_audit = (
        build_task_equal_width_target_bin_sampling_spec(
            samples,
            train_indices=finetune_mgkg_train_indices,
            enabled=target_bin_sampling_enabled,
            bins=target_bin_sampling_bins,
            min_weight=target_bin_sampling_min_weight,
            max_weight=target_bin_sampling_max_weight,
        )
    )
    toxicity_binning_summary = summarize_toxicity_bins(samples, toxicity_binning_cfg, toxicity_bin_count)
    dataset = AggregatedTaskDataset(samples=samples, fingerprint_size=fingerprint_size)

    train_dataset = build_noisy_index_dataset(
        dataset,
        actual_train_indices,
        replicates=augmentation_cfg.train_replicates,
        numeric_noise_std=augmentation_cfg.numeric_noise_std,
        target_noise_std=augmentation_cfg.target_noise_std,
        seed=augmentation_cfg.seed,
    )
    validation_dataset = _IndexDataset(dataset, validation_indices) if validation_indices else None
    task_heads = tuple(allowed_task_heads) if allowed_task_heads else dataset.task_heads()
    observed_task_heads = set(dataset.task_heads())
    if not observed_task_heads.issubset(set(task_heads)):
        raise ValueError("Development samples include a task head outside the locked task-head contract.")
    hierarchical_head_spec = build_mgkg_hierarchical_head_spec(
        samples,
        train_indices=finetune_mgkg_train_indices,
        enabled=hierarchical_head_enabled,
        family_tau=hierarchical_family_tau,
        task_tau=hierarchical_task_tau,
    )
    mgkg_adapter_heads = tuple(
        sorted(
            {
                str(sample.get("task_head", ""))
                for sample in samples
                if str(sample.get("target_name", "")).strip() == "neg_log10_mg_kg"
            }
        )
    )
    if use_mgkg_residual_adapter:
        invalid_mgkg_families = sorted(
            {
                str(sample.get("target_family", "")).strip()
                for sample in samples
                if str(sample.get("target_name", "")).strip() == "neg_log10_mg_kg"
                and str(sample.get("target_family", "")).strip() != "solid_neglog_mg_kg"
            }
        )
        if not mgkg_adapter_heads:
            raise ValueError("mg/kg residual adapter requires at least one neg_log10_mg_kg task head.")
        if invalid_mgkg_families:
            raise ValueError(
                "mg/kg residual adapter found unexpected target families: "
                f"{invalid_mgkg_families}"
            )
        if not set(mgkg_adapter_heads).issubset(set(task_heads)):
            raise ValueError("mg/kg residual adapter heads must be a subset of model task heads.")
    categorical_cardinalities = {
        column: max(mapping.values(), default=0) + 1
        for column, mapping in categorical_maps.items()
    }
    # Seed before module construction so each matrix cell has reproducible and
    # paired shared-parameter initialization. A second reset below keeps data
    # loader randomness invariant to optional architecture modules.
    set_torch_seed(seed)
    hidden_dims = _hidden_dims(config)
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    deep_model_config = DeepModelConfig(
        numeric_dim=dataset.numeric_dim(),
        fingerprint_dim=dataset.fingerprint_dim(),
        categorical_cardinalities=categorical_cardinalities,
        adapter_count=max(adapter_map.values(), default=0) + 1 if adapter_map else 0,
        effect_level_numeric_indices=effect_level_feature_indices(numeric_feature_names),
        descriptor_count=descriptor_count,
        descriptor_encoder_mode=descriptor_encoder_cfg["mode"],
        descriptor_head_dim=descriptor_encoder_cfg["head_dim"],
        descriptor_group_head_dim=descriptor_encoder_cfg["group_head_dim"],
        descriptor_group_indices=descriptor_encoder_cfg["group_indices"],
        graph_atom_feature_dim=0 if graph_encoder is None else graph_encoder.atom_feature_dim,
        graph_edge_feature_dim=0 if graph_encoder is None else graph_encoder.edge_feature_dim,
        graph_embedding_dim=0 if graph_encoder is None else graph_encoder_cfg["embedding_dim"],
        graph_message_steps=graph_encoder_cfg["message_steps"],
        task_heads=task_heads,
        hidden_dims=hidden_dims,
        fusion_mode=str(model_cfg.get("fusion_mode", "concat")),
        molecular_hidden_dims=_dimension_tuple(model_cfg.get("molecular_hidden_dims", ())),
        context_hidden_dims=_dimension_tuple(model_cfg.get("context_hidden_dims", ())),
        film_context_hidden_dim=int(model_cfg.get("film_context_hidden_dim", 0) or 0),
        dropout=float(
            dropout
            if dropout is not None
            else config.get("model", {}).get("dropout", 0.15)
        ),
        use_molecular_residual=ablation_spec.use_molecular_residual,
        use_adapters=ablation_spec.use_medium_adapter,
        toxicity_bin_count=toxicity_bin_count,
        toxicity_binning_mode=(
            toxicity_binning_cfg.mode if toxicity_binning_cfg.enabled else "none"
        ),
        use_mgkg_residual_adapter=use_mgkg_residual_adapter,
        mgkg_residual_adapter_bottleneck=mgkg_adapter_bottleneck,
        mgkg_residual_adapter_heads=mgkg_adapter_heads,
        mgkg_hierarchical_heads=tuple(hierarchical_head_spec["heads"]),
        mgkg_hierarchical_head_families=hierarchical_head_spec["head_families"],
        mgkg_hierarchical_family_scales=hierarchical_head_spec["family_scales"],
        mgkg_hierarchical_task_scales=hierarchical_head_spec["task_scales"],
    )
    model = EcotoxMultiTaskNetwork(deep_model_config)
    loss_cfg = train_cfg.get("loss", {}) if isinstance(train_cfg.get("loss", {}), dict) else {}
    task_weights = resolve_task_weights(
        samples,
        train_indices=actual_train_indices,
        train_cfg=train_cfg,
        task_heads=task_heads,
    )
    train_config = DeepTrainingConfig(
        epochs=requested_stage1_epochs,
        batch_size=int(batch_size or train_cfg.get("batch_size", 256)),
        learning_rate=float(learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 3e-4)),
        huber_delta=float(loss_cfg.get("delta", 1.0)),
        mse_loss_weight=float(loss_cfg.get("mse_weight", 0.0)),
        task_weights=task_weights,
        optimizer=str(train_cfg.get("optimizer", "adamw")),
        weight_decay=float(weight_decay if weight_decay is not None else train_cfg.get("weight_decay", 1e-4)),
        gradient_clip_norm=_optional_float(train_cfg.get("gradient_clip_norm", 5.0)),
        scheduler=str(scheduler if scheduler is not None else train_cfg.get("scheduler", "none")),
        device=_resolve_device(device or train_cfg.get("device", "cpu")),
        seed=seed,
        num_workers=int(train_cfg.get("num_workers", 0)),
        toxicity_bin_loss_weight=toxicity_binning_cfg.loss_weight if toxicity_binning_cfg.active() else 0.0,
        toxicity_binning_mode=toxicity_binning_cfg.mode,
        censored_loss_weight=censored_loss_cfg.weight if censored_loss_cfg.active() else 0.0,
        censored_loss_margin=censored_loss_cfg.margin,
    )
    finetune_config = DeepTrainingConfig(
        epochs=requested_finetune_epochs,
        batch_size=int(finetune_batch_size or finetune_cfg.get("batch_size", train_config.batch_size)),
        learning_rate=float(
            finetune_learning_rate
            if finetune_learning_rate is not None
            else finetune_cfg.get("learning_rate", train_config.learning_rate * 0.2)
        ),
        huber_delta=train_config.huber_delta,
        mse_loss_weight=train_config.mse_loss_weight,
        task_weights=train_config.task_weights,
        optimizer=str(finetune_cfg.get("optimizer", train_config.optimizer)),
        weight_decay=float(finetune_cfg.get("weight_decay", train_config.weight_decay)),
        gradient_clip_norm=_optional_float(finetune_cfg.get("gradient_clip_norm", train_config.gradient_clip_norm)),
        scheduler=str(finetune_scheduler if finetune_scheduler is not None else finetune_cfg.get("scheduler", train_config.scheduler)),
        device=train_config.device,
        seed=seed,
        num_workers=train_config.num_workers,
        toxicity_bin_loss_weight=train_config.toxicity_bin_loss_weight,
        toxicity_binning_mode=train_config.toxicity_binning_mode,
        censored_loss_weight=train_config.censored_loss_weight,
        censored_loss_margin=train_config.censored_loss_margin,
    )
    finetune_mgkg_config = DeepTrainingConfig(
        epochs=requested_finetune_mgkg_epochs,
        batch_size=int(
            finetune_mgkg_batch_size
            or finetune_mgkg_cfg.get("batch_size", finetune_config.batch_size)
        ),
        learning_rate=float(
            finetune_mgkg_learning_rate
            if finetune_mgkg_learning_rate is not None
            else finetune_mgkg_cfg.get("learning_rate", finetune_config.learning_rate * 0.5)
        ),
        huber_delta=train_config.huber_delta,
        mse_loss_weight=float(
            finetune_mgkg_mse_loss_weight
            if finetune_mgkg_mse_loss_weight is not None
            else finetune_mgkg_cfg.get("mse_loss_weight", train_config.mse_loss_weight)
        ),
        task_weights=train_config.task_weights,
        optimizer=str(finetune_mgkg_cfg.get("optimizer", finetune_config.optimizer)),
        weight_decay=float(finetune_mgkg_cfg.get("weight_decay", finetune_config.weight_decay)),
        gradient_clip_norm=_optional_float(
            finetune_mgkg_cfg.get("gradient_clip_norm", finetune_config.gradient_clip_norm)
        ),
        scheduler=str(
            finetune_mgkg_scheduler
            if finetune_mgkg_scheduler is not None
            else finetune_mgkg_cfg.get("scheduler", finetune_config.scheduler)
        ),
        device=train_config.device,
        seed=seed,
        num_workers=train_config.num_workers,
        toxicity_bin_loss_weight=float(
            finetune_mgkg_toxicity_bin_loss_weight
            if finetune_mgkg_toxicity_bin_loss_weight is not None
            else finetune_mgkg_cfg.get(
                "toxicity_bin_loss_weight",
                train_config.toxicity_bin_loss_weight,
            )
        ),
        toxicity_binning_mode=train_config.toxicity_binning_mode,
        censored_loss_weight=train_config.censored_loss_weight,
        censored_loss_margin=train_config.censored_loss_margin,
    )
    checkpoint_preprocessing = build_preprocessing_manifest(
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        ablation=ablation_spec,
        fingerprint_size=fingerprint_size,
        encoder_source=encoder.source,
        cache_path=cache_path,
        descriptor_count=descriptor_count,
        descriptor_names=descriptor_names,
        descriptor_encoder=descriptor_encoder_cfg["manifest"],
        graph_encoder=(
            {}
            if graph_encoder is None
            else {**graph_encoder.to_manifest(), **graph_encoder_cfg}
        ),
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        categorical_min_count=category_min_count,
    )
    base_architecture = asdict(deep_model_config)
    for hierarchy_key in (
        "mgkg_hierarchical_heads",
        "mgkg_hierarchical_head_families",
        "mgkg_hierarchical_family_scales",
        "mgkg_hierarchical_task_scales",
    ):
        base_architecture.pop(hierarchy_key, None)
    finetune_mgkg_checkpoint_contract = stage3_init_checkpoint_contract(
        data_identity={
            **database_source_identity(
                db_path,
                source_table=str(
                    source_table or split_join_audit.get("source_table", "")
                ),
                split_name=split_name,
            ),
            "split_join_audit": split_join_audit,
            "all_split_scientific_identity_sha256": sample_scientific_identity_sha256(
                samples, list(range(len(samples))), include_split_part=True
            ),
        },
        preprocessing={
            "manifest": checkpoint_preprocessing,
            "categorical_maps_sha256": canonical_sha256(categorical_maps),
            "numeric_feature_names_sha256": canonical_sha256(
                list(numeric_feature_names)
            ),
            "preprocessing_fit_samples": sample_indices_contract(
                samples, preprocessing_indices
            ),
        },
        base_architecture={
            "head_routing": head_routing_mode,
            "ablation": asdict(ablation_spec),
            "model_config_without_g3_hierarchy": base_architecture,
            "runtime_code_identity": stage12_runtime_code_identity(),
            "descriptor_encoder": descriptor_encoder_cfg,
            "graph_encoder": (
                {}
                if graph_encoder is None
                else {**graph_encoder.to_manifest(), **graph_encoder_cfg}
            ),
        },
        weighting_and_auxiliary={
            "task_weighting": str(train_cfg.get("task_weighting", "balanced")),
            "task_weights": task_weights,
            # Cache provenance remains available in the experiment manifest,
            # but a cold computation and a warm cache replay must describe the
            # same scientific stage-1/2 initialization contract.
            "source_weighting": scientific_summary_contract(
                source_weighting_summary
            ),
            "effect_level_weighting": scientific_summary_contract(
                effect_level_weighting_summary
            ),
            "actual_training_sample_weights": {
                "stage1": sample_weight_identity_contract(
                    samples, actual_train_indices
                ),
                "stage2": sample_weight_identity_contract(
                    samples, finetune_train_indices
                ),
            },
            "toxicity_binning_config": toxicity_binning_cfg.to_manifest(),
            "toxicity_binning_scheme": toxicity_bin_scheme,
            "toxicity_binning_summary": scientific_summary_contract(
                toxicity_binning_summary
            ),
            "censored_loss": scientific_summary_contract(censored_summary),
            "domain_alignment": domain_alignment_cfg.to_manifest(),
        },
        stage12_protocol={
            "seed": int(seed),
            "stage1": {
                "training_config": asdict(train_config),
                "early_stopping": {
                    **early_cfg,
                    "enabled_effective": early_enabled,
                    "validation_source": validation_source,
                    "validation_seed": int(
                        seed if validation_seed is None else validation_seed
                    ),
                },
                "train_samples": sample_indices_contract(
                    samples, actual_train_indices
                ),
                "validation_samples": sample_indices_contract(
                    samples, validation_indices
                ),
            },
            "stage2": {
                "training_config": asdict(finetune_config),
                "freeze": finetune_freeze_mode,
                "early_stopping": {
                    "enabled_effective": bool(
                        finetune_validation_indices
                        and finetune_cfg.get("early_stopping", True)
                    ),
                    "patience": int(
                        finetune_cfg.get(
                            "early_stopping_patience", early_cfg["patience"]
                        )
                    ),
                    "min_delta": float(
                        finetune_cfg.get(
                            "early_stopping_min_delta", early_cfg["min_delta"]
                        )
                    ),
                    "validation_source": finetune_validation_source,
                    "validation_seed": int(
                        seed
                        if finetune_validation_seed is None
                        else finetune_validation_seed
                    ),
                    "validation_fraction": float(
                        finetune_validation_fraction
                        if finetune_validation_fraction is not None
                        else finetune_cfg.get("validation_fraction", 0.0)
                    ),
                    "monitor_split": str(
                        finetune_cfg.get("monitor_split", "auto")
                    ),
                },
                "train_samples": sample_indices_contract(
                    samples, finetune_train_indices
                ),
                "validation_samples": sample_indices_contract(
                    samples, finetune_validation_indices
                ),
            },
            "swa": swa_cfg.to_manifest(),
            "augmentation": augmentation_cfg.to_manifest(),
        },
    )

    set_torch_seed(seed)
    torch_device = torch.device(train_config.device)
    model.to(torch_device)
    evaluation_checkpoint_audit: dict[str, Any] = {
        "requested": evaluation_checkpoint is not None,
        "loaded": False,
        "path": "" if evaluation_checkpoint is None else str(evaluation_checkpoint),
        "sha256": "",
        "strict": True,
    }
    if evaluation_checkpoint is not None:
        if any(
            value > 0
            for value in (
                requested_stage1_epochs,
                requested_finetune_epochs,
                requested_finetune_mgkg_epochs,
            )
        ):
            raise ValueError("evaluation_checkpoint is read-only and requires every training epoch count to be zero.")
        if finetune_mgkg_init_checkpoint is not None:
            raise ValueError("evaluation_checkpoint cannot be combined with finetune_mgkg_init_checkpoint.")
        checkpoint_path = Path(evaluation_checkpoint)
        state = load_plain_model_state_dict(checkpoint_path, map_location=torch_device)
        model.load_state_dict(state, strict=True)
        evaluation_checkpoint_audit.update(
            {
                "loaded": True,
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "parameter_tensor_count": len(state),
            }
        )
    finetune_mgkg_checkpoint_audit: dict[str, Any] = {
        "format": "qsar_stage3_init_v1",
        "contract_sha256": finetune_mgkg_checkpoint_contract["sha256"],
        "contract_schema_version": int(
            finetune_mgkg_checkpoint_contract.get("schema_version", 0)
        ),
        "contract_hash_recomputed": True,
        "loaded": False,
        "exported": False,
        "stage1_stage2_skipped": False,
    }
    finetune_mgkg_checkpoint_loaded = False
    if finetune_mgkg_init_checkpoint is not None:
        load_audit = load_stage3_init_checkpoint(
            model,
            finetune_mgkg_init_checkpoint,
            expected_contract=finetune_mgkg_checkpoint_contract,
            reset_hierarchical_heads=tuple(hierarchical_head_spec["heads"]),
        )
        finetune_mgkg_checkpoint_audit.update(load_audit)
        finetune_mgkg_checkpoint_audit["stage1_stage2_skipped"] = True
        finetune_mgkg_checkpoint_loaded = True
        model.to(torch_device)
    trajectory_cfg = resolve_trajectory_config(train_cfg)
    trajectory_source_indices = (
        finetune_mgkg_train_indices
        if finetune_mgkg_requested and finetune_mgkg_train_indices
        else finetune_train_indices
        if finetune_requested and finetune_train_indices
        else actual_train_indices
    )
    trajectory_anchor_indices = (
        select_task_stratified_anchor(
            samples,
            trajectory_source_indices,
            size=int(trajectory_cfg["anchor_size"]),
            seed=int(trajectory_cfg["selection_seed"]),
        )
        if trajectory_cfg["enabled"]
        else []
    )
    trajectory_dir = output_dir / "trajectory"
    trajectory_snapshots: list[dict[str, Any]] = []
    if trajectory_cfg["enabled"]:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        write_trajectory_anchor_manifest(
            trajectory_dir / "anchor_manifest.json",
            samples=samples,
            indices=trajectory_anchor_indices,
            config=trajectory_cfg,
        )
        trajectory_snapshots.append(
            capture_trajectory_snapshot(
                model,
                dataset,
                samples,
                trajectory_anchor_indices,
                out_path=trajectory_dir / "epoch_0000_initialization.npz",
                phase="initialization",
                epoch=0,
                global_epoch=0,
                batch_size=train_config.batch_size,
                device=torch_device,
                target_scaler=target_scaler,
            )
        )
    optimizer = build_optimizer(model.parameters(), train_config)
    scheduler = build_scheduler(optimizer, train_config, train_config.epochs)
    loss_fn = build_regression_loss(train_config)
    finetune_loss_fn = build_regression_loss(finetune_config)
    finetune_mgkg_loss_fn = build_regression_loss(finetune_mgkg_config)
    swa_model: Any | None = None
    swa_updates = 0
    swa_last_global_epoch = 0
    swa_applied = False
    dataloader = (
        DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=collate_aggregated_task_batch,
            **dataloader_runtime_options(torch_device, train_config.num_workers),
        )
        if actual_train_indices
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_aggregated_task_batch,
            **dataloader_runtime_options(torch_device, train_config.num_workers),
        )
        if validation_dataset is not None
        else None
    )
    domain_alignment_summary = {
        **domain_alignment_cfg.to_manifest(),
        "pretrain_reference_samples": 0,
        "finetune_reference_samples": 0,
        "pretrain_alignment_steps": 0,
        "finetune_alignment_steps": 0,
    }
    pretrain_alignment_indices = domain_alignment_reference_indices(
        samples,
        domain_alignment_cfg,
        phase="pretrain",
    )
    domain_alignment_summary["pretrain_reference_samples"] = len(pretrain_alignment_indices)
    pretrain_alignment_batches = (
        cycle_dataloader(
            DataLoader(
                _IndexDataset(dataset, pretrain_alignment_indices),
                batch_size=train_config.batch_size,
                shuffle=True,
                collate_fn=collate_aggregated_task_batch,
                **dataloader_runtime_options(torch_device, train_config.num_workers),
            )
        )
        if pretrain_alignment_indices
        else None
    )

    history = []
    best_epoch = 0
    best_monitor_loss = float("inf")
    best_state: dict[str, Any] | None = (
        clone_state_dict(model)
        if finetune_mgkg_checkpoint_loaded or evaluation_checkpoint_audit["loaded"]
        else None
    )
    no_improve_epochs = 0
    pretrain_epochs = 0 if finetune_mgkg_checkpoint_loaded else train_config.epochs
    for epoch in range(1, pretrain_epochs + 1):
        if dataloader is None:
            raise RuntimeError("Stage-1 epochs were requested without a Stage-1 dataloader.")
        epoch_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            train_config,
            torch_device,
            alignment_batches=pretrain_alignment_batches,
            alignment_weight=domain_alignment_cfg.weight if pretrain_alignment_batches is not None else 0.0,
        )
        domain_alignment_summary["pretrain_alignment_steps"] += int(epoch_loss.get("alignment_steps", 0))
        validation_loss = (
            evaluate_loss(model, validation_loader, loss_fn, train_config, torch_device)
            if validation_loader is not None
            else None
        )
        monitor_loss = validation_loss["mean_loss"] if validation_loss is not None else epoch_loss["mean_loss"]
        if early_enabled:
            improved = monitor_loss < best_monitor_loss - float(early_cfg["min_delta"])
            if improved:
                best_monitor_loss = monitor_loss
                best_epoch = epoch
                no_improve_epochs = 0
                best_state = clone_state_dict(model)
            else:
                no_improve_epochs += 1
        else:
            best_monitor_loss = monitor_loss
            best_epoch = epoch
            no_improve_epochs = 0
            best_state = clone_state_dict(model)
        row = {
            "phase": "pretrain",
            "epoch": epoch,
            "global_epoch": epoch,
            **weighting_history_fields,
            **epoch_loss,
            "validation_loss": "" if validation_loss is None else validation_loss["mean_loss"],
            "validation_task_loss": "" if validation_loss is None else validation_loss.get("mean_task_loss", ""),
            "validation_toxicity_bin_loss": "" if validation_loss is None else validation_loss.get("mean_toxicity_bin_loss", ""),
            "validation_toxicity_bin_samples": 0 if validation_loss is None else validation_loss.get("toxicity_bin_samples", 0),
            "validation_censored_loss": "" if validation_loss is None else validation_loss.get("mean_censored_loss", ""),
            "validation_censored_samples": 0 if validation_loss is None else validation_loss.get("censored_samples", 0),
            "validation_samples": 0 if validation_loss is None else validation_loss["samples"],
            "monitor_loss": monitor_loss,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "learning_rate": current_learning_rate(optimizer),
            "swa_updates": swa_updates,
        }
        if should_update_swa(swa_cfg, phase="pretrain", epoch=epoch):
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)
            swa_updates += 1
            swa_last_global_epoch = epoch
            row["swa_updates"] = swa_updates
        history.append(row)
        if trajectory_cfg["enabled"]:
            trajectory_snapshots.append(
                capture_trajectory_snapshot(
                    model,
                    dataset,
                    samples,
                    trajectory_anchor_indices,
                    out_path=trajectory_dir / f"epoch_{epoch:04d}_pretrain.npz",
                    phase="pretrain",
                    epoch=epoch,
                    global_epoch=epoch,
                    batch_size=train_config.batch_size,
                    device=torch_device,
                    target_scaler=target_scaler,
                )
            )
        step_scheduler(scheduler, monitor_loss)
        validation_msg = "" if validation_loss is None else f" validation_loss={validation_loss['mean_loss']:.6f}"
        print(
            f"[epoch {epoch}] mean_loss={epoch_loss['mean_loss']:.6f}{validation_msg} "
            f"samples={epoch_loss['samples']} best_epoch={best_epoch}",
            flush=True,
        )
        if early_enabled and no_improve_epochs >= int(early_cfg["patience"]):
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"monitor_loss={best_monitor_loss:.6f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    finetune_ran = 0
    finetune_best_epoch = 0
    finetune_best_monitor_loss = float("inf")
    finetune_no_improve_epochs = 0
    finetune_early_enabled = bool(
        finetune_validation_indices and bool(finetune_cfg.get("early_stopping", True))
    )
    finetune_patience = int(finetune_cfg.get("early_stopping_patience", early_cfg["patience"]))
    finetune_min_delta = float(finetune_cfg.get("early_stopping_min_delta", early_cfg["min_delta"]))
    if (
        not finetune_mgkg_checkpoint_loaded
        and finetune_requested
        and finetune_train_indices
    ):
        trainable_parameters = apply_finetune_freeze(model, finetune_freeze_mode)
        finetune_dataset = build_noisy_index_dataset(
            dataset,
            finetune_train_indices,
            replicates=augmentation_cfg.finetune_replicates,
            numeric_noise_std=augmentation_cfg.numeric_noise_std,
            target_noise_std=augmentation_cfg.target_noise_std,
            seed=augmentation_cfg.seed + 100_000,
        )
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=finetune_config.batch_size,
            shuffle=True,
            collate_fn=collate_aggregated_task_batch,
            **dataloader_runtime_options(torch_device, finetune_config.num_workers),
        )
        finetune_validation_loader = (
            DataLoader(
                _IndexDataset(dataset, finetune_validation_indices),
                batch_size=finetune_config.batch_size,
                shuffle=False,
                collate_fn=collate_aggregated_task_batch,
                **dataloader_runtime_options(torch_device, finetune_config.num_workers),
            )
            if finetune_validation_indices
            else None
        )
        finetune_alignment_indices = domain_alignment_reference_indices(
            samples,
            domain_alignment_cfg,
            phase="finetune",
        )
        domain_alignment_summary["finetune_reference_samples"] = len(finetune_alignment_indices)
        finetune_alignment_batches = (
            cycle_dataloader(
                DataLoader(
                    _IndexDataset(dataset, finetune_alignment_indices),
                    batch_size=finetune_config.batch_size,
                    shuffle=True,
                    collate_fn=collate_aggregated_task_batch,
                    **dataloader_runtime_options(torch_device, finetune_config.num_workers),
                )
            )
            if finetune_alignment_indices
            else None
        )
        optimizer = build_optimizer(trainable_parameters, finetune_config)
        scheduler = build_scheduler(optimizer, finetune_config, finetune_config.epochs)
        finetune_best_state: dict[str, Any] | None = None
        for epoch in range(1, finetune_config.epochs + 1):
            epoch_loss = train_one_epoch(
                model,
                finetune_loader,
                optimizer,
                finetune_loss_fn,
                finetune_config,
                torch_device,
                alignment_batches=finetune_alignment_batches,
                alignment_weight=domain_alignment_cfg.weight if finetune_alignment_batches is not None else 0.0,
            )
            domain_alignment_summary["finetune_alignment_steps"] += int(epoch_loss.get("alignment_steps", 0))
            finetune_ran = epoch
            global_epoch = len(history) + 1
            finetune_validation_loss = (
                evaluate_loss(
                    model,
                    finetune_validation_loader,
                    finetune_loss_fn,
                    finetune_config,
                    torch_device,
                )
                if finetune_validation_loader is not None
                else None
            )
            monitor_loss = (
                finetune_validation_loss["mean_loss"]
                if finetune_validation_loss is not None
                else epoch_loss["mean_loss"]
            )
            if finetune_early_enabled:
                improved = monitor_loss < finetune_best_monitor_loss - finetune_min_delta
                if improved:
                    finetune_best_monitor_loss = monitor_loss
                    finetune_best_epoch = global_epoch
                    finetune_no_improve_epochs = 0
                    finetune_best_state = clone_state_dict(model)
                else:
                    finetune_no_improve_epochs += 1
            else:
                finetune_best_monitor_loss = monitor_loss
                finetune_best_epoch = global_epoch
                finetune_no_improve_epochs = 0
                finetune_best_state = clone_state_dict(model)
            row = {
                "phase": "finetune",
                "epoch": epoch,
                "global_epoch": global_epoch,
                **weighting_history_fields,
                **epoch_loss,
                "validation_loss": "" if finetune_validation_loss is None else finetune_validation_loss["mean_loss"],
                "validation_task_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_task_loss", ""),
                "validation_toxicity_bin_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_toxicity_bin_loss", ""),
                "validation_toxicity_bin_samples": 0 if finetune_validation_loss is None else finetune_validation_loss.get("toxicity_bin_samples", 0),
                "validation_censored_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_censored_loss", ""),
                "validation_censored_samples": 0 if finetune_validation_loss is None else finetune_validation_loss.get("censored_samples", 0),
                "validation_samples": 0 if finetune_validation_loss is None else finetune_validation_loss["samples"],
                "monitor_loss": monitor_loss,
                "best_epoch": finetune_best_epoch,
                "no_improve_epochs": finetune_no_improve_epochs,
                "learning_rate": current_learning_rate(optimizer),
                "swa_updates": swa_updates,
            }
            if should_update_swa(swa_cfg, phase="finetune", epoch=epoch):
                if swa_model is None:
                    swa_model = torch.optim.swa_utils.AveragedModel(model)
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_last_global_epoch = global_epoch
                row["swa_updates"] = swa_updates
            history.append(row)
            if trajectory_cfg["enabled"]:
                trajectory_snapshots.append(
                    capture_trajectory_snapshot(
                        model,
                        dataset,
                        samples,
                        trajectory_anchor_indices,
                        out_path=trajectory_dir / f"epoch_{global_epoch:04d}_finetune.npz",
                        phase="finetune",
                        epoch=epoch,
                        global_epoch=global_epoch,
                        batch_size=finetune_config.batch_size,
                        device=torch_device,
                        target_scaler=target_scaler,
                    )
                )
            step_scheduler(scheduler, monitor_loss)
            best_epoch = finetune_best_epoch
            best_monitor_loss = finetune_best_monitor_loss
            best_state = finetune_best_state
            validation_msg = (
                ""
                if finetune_validation_loss is None
                else f" validation_loss={finetune_validation_loss['mean_loss']:.6f}"
            )
            print(
                f"[finetune epoch {epoch}] mean_loss={epoch_loss['mean_loss']:.6f}{validation_msg} "
                f"samples={epoch_loss['samples']} best_epoch={best_epoch}",
                flush=True,
            )
            if finetune_early_enabled and finetune_no_improve_epochs >= finetune_patience:
                print(
                    f"[finetune early-stop] epoch={epoch} best_epoch={finetune_best_epoch} "
                    f"monitor_loss={finetune_best_monitor_loss:.6f}",
                    flush=True,
                )
                break
        if finetune_best_state is not None:
            model.load_state_dict(finetune_best_state)

    if export_finetune_mgkg_init_checkpoint is not None:
        if (
            not finetune_mgkg_checkpoint_loaded
            and not (finetune_requested and finetune_train_indices)
        ):
            raise ValueError(
                "Exporting a stage-3 init checkpoint requires a completed stage-2 "
                "finetune phase or an already validated stage-3 init checkpoint."
            )
        export_audit = export_stage3_init_checkpoint(
            model,
            export_finetune_mgkg_init_checkpoint,
            contract=finetune_mgkg_checkpoint_contract,
        )
        finetune_mgkg_checkpoint_audit.update(export_audit)

    finetune_mgkg_ran = 0
    finetune_mgkg_best_epoch = 0
    finetune_mgkg_best_monitor_loss = float("inf")
    finetune_mgkg_no_improve_epochs = 0
    finetune_mgkg_target_samples = 0
    finetune_mgkg_replay_samples = 0
    finetune_mgkg_target_per_full_batch = 0
    finetune_mgkg_replay_per_full_batch = 0
    finetune_mgkg_replay_audit: dict[str, Any] = {}
    finetune_mgkg_early_requested = bool(
        finetune_mgkg_cfg.get("early_stopping", True)
        if finetune_mgkg_early_stopping is None
        else finetune_mgkg_early_stopping
    )
    finetune_mgkg_early_enabled = bool(
        finetune_mgkg_validation_indices and finetune_mgkg_early_requested
    )
    finetune_mgkg_patience = int(
        finetune_mgkg_cfg.get("early_stopping_patience", finetune_patience)
    )
    finetune_mgkg_min_delta = float(
        finetune_mgkg_cfg.get("early_stopping_min_delta", finetune_min_delta)
    )
    if finetune_mgkg_requested and finetune_mgkg_train_indices:
        if finetune_mgkg_replay_ratio > 0 and not finetune_train_indices:
            raise ValueError(
                "Soil pTox replay was requested, but stage-2 has no training-only rows."
            )
        if finetune_mgkg_replay_ratio > 0:
            finetune_mgkg_replay_audit = audit_mgkg_replay_boundary(
                samples,
                replay_indices=finetune_train_indices,
                mgkg_indices=finetune_mgkg_train_indices,
            )
        # The last stage is the deliverable target. Do not let a previous-stage
        # SWA snapshot overwrite its dedicated mg/kg calibration.
        swa_model = None
        swa_updates = 0
        swa_last_global_epoch = 0
        finetune_mgkg_target_dataset = build_noisy_index_dataset(
            dataset,
            finetune_mgkg_train_indices,
            replicates=augmentation_cfg.finetune_replicates,
            numeric_noise_std=augmentation_cfg.numeric_noise_std,
            target_noise_std=augmentation_cfg.target_noise_std,
            seed=augmentation_cfg.seed + 200_000,
        )
        finetune_mgkg_replay_dataset = build_noisy_index_dataset(
            dataset,
            finetune_train_indices,
            replicates=augmentation_cfg.finetune_replicates,
            numeric_noise_std=augmentation_cfg.numeric_noise_std,
            target_noise_std=augmentation_cfg.target_noise_std,
            seed=augmentation_cfg.seed + 300_000,
        )
        finetune_mgkg_pool_dataset = _StagePoolDataset(
            finetune_mgkg_target_dataset,
            finetune_mgkg_replay_dataset,
        )
        finetune_mgkg_target_samples = finetune_mgkg_pool_dataset.target_count
        finetune_mgkg_weighted_sampler = None
        if target_bin_sampling_enabled:
            from torch.utils.data import WeightedRandomSampler

            expanded_sampling_weights = sampling_weights_for_stage_dataset(
                finetune_mgkg_target_dataset,
                source_weights=target_bin_sampling_weights,
            )
            sampling_generator = torch.Generator()
            sampling_generator.manual_seed(int(seed) + 510_031)
            finetune_mgkg_weighted_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(expanded_sampling_weights, dtype=torch.double),
                num_samples=len(finetune_mgkg_target_dataset),
                replacement=True,
                generator=sampling_generator,
            )
            target_bin_sampling_audit["sampler_seed"] = int(seed) + 510_031
            target_bin_sampling_audit["samples_per_epoch"] = int(
                len(finetune_mgkg_target_dataset)
            )
            target_bin_sampling_audit["augmentation_replicates"] = int(
                augmentation_cfg.finetune_replicates
            )
        finetune_mgkg_target_loader = DataLoader(
            finetune_mgkg_target_dataset,
            batch_size=finetune_mgkg_config.batch_size,
            shuffle=finetune_mgkg_weighted_sampler is None,
            sampler=finetune_mgkg_weighted_sampler,
            collate_fn=collate_aggregated_task_batch,
            **dataloader_runtime_options(torch_device, finetune_mgkg_config.num_workers),
        )
        finetune_mgkg_replay_sampler = (
            _TargetReplayBatchSampler(
                target_count=finetune_mgkg_pool_dataset.target_count,
                replay_count=finetune_mgkg_pool_dataset.replay_pool_count,
                batch_size=finetune_mgkg_config.batch_size,
                replay_fraction=finetune_mgkg_replay_ratio,
                seed=seed + 400_000,
            )
            if finetune_mgkg_replay_ratio > 0
            else None
        )
        finetune_mgkg_replay_loader = (
            DataLoader(
                finetune_mgkg_pool_dataset,
                batch_sampler=finetune_mgkg_replay_sampler,
                collate_fn=collate_aggregated_task_batch,
                **dataloader_runtime_options(torch_device, finetune_mgkg_config.num_workers),
            )
            if finetune_mgkg_replay_sampler is not None
            else None
        )
        finetune_mgkg_replay_samples = (
            finetune_mgkg_replay_sampler.replay_samples_per_epoch
            if finetune_mgkg_replay_sampler is not None
            else 0
        )
        if finetune_mgkg_replay_sampler is not None:
            finetune_mgkg_target_per_full_batch = (
                finetune_mgkg_replay_sampler.target_per_full_batch
            )
            finetune_mgkg_replay_per_full_batch = (
                finetune_mgkg_replay_sampler.replay_per_full_batch
            )
        finetune_mgkg_validation_loader = (
            DataLoader(
                _IndexDataset(dataset, finetune_mgkg_validation_indices),
                batch_size=finetune_mgkg_config.batch_size,
                shuffle=False,
                collate_fn=collate_aggregated_task_batch,
                **dataloader_runtime_options(torch_device, finetune_mgkg_config.num_workers),
            )
            if finetune_mgkg_validation_indices
            else None
        )
        initial_freeze_mode = (
            "heads_only"
            if finetune_mgkg_head_only_epoch_count > 0
            else finetune_mgkg_freeze_mode
        )
        trainable_parameters = build_finetune_parameter_groups(
            model,
            freeze_mode=initial_freeze_mode,
            head_learning_rate=finetune_mgkg_config.learning_rate,
            trunk_learning_rate=(
                0.0 if initial_freeze_mode == "heads_only" else finetune_mgkg_trunk_lr
            ),
        )
        optimizer = build_optimizer(trainable_parameters, finetune_mgkg_config)
        initial_phase_epochs = (
            finetune_mgkg_head_only_epoch_count
            if finetune_mgkg_head_only_epoch_count > 0
            else finetune_mgkg_config.epochs
        )
        scheduler = build_scheduler(optimizer, finetune_mgkg_config, initial_phase_epochs)
        finetune_mgkg_loader = (
            finetune_mgkg_replay_loader
            if initial_freeze_mode != "heads_only" and finetune_mgkg_replay_loader is not None
            else finetune_mgkg_target_loader
        )
        finetune_mgkg_best_state: dict[str, Any] | None = None
        for epoch in range(1, finetune_mgkg_config.epochs + 1):
            optimizer_reset_at_unfreeze = False
            if (
                finetune_mgkg_head_only_epoch_count > 0
                and epoch == finetune_mgkg_head_only_epoch_count + 1
            ):
                trainable_parameters = build_finetune_parameter_groups(
                    model,
                    freeze_mode=finetune_mgkg_freeze_mode,
                    head_learning_rate=finetune_mgkg_config.learning_rate,
                    trunk_learning_rate=finetune_mgkg_trunk_lr,
                )
                optimizer = build_optimizer(trainable_parameters, finetune_mgkg_config)
                scheduler = build_scheduler(
                    optimizer,
                    finetune_mgkg_config,
                    finetune_mgkg_config.epochs - finetune_mgkg_head_only_epoch_count,
                )
                finetune_mgkg_no_improve_epochs = 0
                optimizer_reset_at_unfreeze = True
                if finetune_mgkg_replay_loader is not None:
                    finetune_mgkg_loader = finetune_mgkg_replay_loader
            mgkg_subphase = (
                "heads_only"
                if epoch <= finetune_mgkg_head_only_epoch_count
                else finetune_mgkg_freeze_mode
            )
            if finetune_mgkg_loader is finetune_mgkg_replay_loader:
                finetune_mgkg_replay_sampler.set_epoch(epoch)
            epoch_loss = train_one_epoch(
                model,
                finetune_mgkg_loader,
                optimizer,
                finetune_mgkg_loss_fn,
                finetune_mgkg_config,
                torch_device,
            )
            finetune_mgkg_ran = epoch
            global_epoch = len(history) + 1
            finetune_mgkg_validation_loss = (
                evaluate_loss(
                    model,
                    finetune_mgkg_validation_loader,
                    finetune_mgkg_loss_fn,
                    finetune_mgkg_config,
                    torch_device,
                )
                if finetune_mgkg_validation_loader is not None
                else None
            )
            monitor_loss = (
                finetune_mgkg_validation_loss["mean_loss"]
                if finetune_mgkg_validation_loss is not None
                else epoch_loss["mean_loss"]
            )
            if finetune_mgkg_early_enabled:
                improved = monitor_loss < finetune_mgkg_best_monitor_loss - finetune_mgkg_min_delta
                if improved:
                    finetune_mgkg_best_monitor_loss = monitor_loss
                    finetune_mgkg_best_epoch = global_epoch
                    finetune_mgkg_no_improve_epochs = 0
                    finetune_mgkg_best_state = clone_state_dict(model)
                else:
                    finetune_mgkg_no_improve_epochs += 1
            else:
                finetune_mgkg_best_monitor_loss = monitor_loss
                finetune_mgkg_best_epoch = global_epoch
                finetune_mgkg_no_improve_epochs = 0
                finetune_mgkg_best_state = clone_state_dict(model)
            group_learning_rates = optimizer_learning_rates(optimizer)
            row = {
                "phase": "finetune_mgkg",
                "mgkg_subphase": mgkg_subphase,
                "epoch": epoch,
                "global_epoch": global_epoch,
                **weighting_history_fields,
                **epoch_loss,
                "validation_loss": "" if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss["mean_loss"],
                "validation_task_loss": "" if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss.get("mean_task_loss", ""),
                "validation_toxicity_bin_loss": "" if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss.get("mean_toxicity_bin_loss", ""),
                "validation_toxicity_bin_samples": 0 if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss.get("toxicity_bin_samples", 0),
                "validation_censored_loss": "" if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss.get("mean_censored_loss", ""),
                "validation_censored_samples": 0 if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss.get("censored_samples", 0),
                "validation_samples": 0 if finetune_mgkg_validation_loss is None else finetune_mgkg_validation_loss["samples"],
                "monitor_loss": monitor_loss,
                "best_epoch": finetune_mgkg_best_epoch,
                "no_improve_epochs": finetune_mgkg_no_improve_epochs,
                "learning_rate": current_learning_rate(optimizer),
                "head_learning_rate": group_learning_rates.get(
                    "head", current_learning_rate(optimizer)
                ),
                "trunk_learning_rate": group_learning_rates.get("trunk", ""),
                "optimizer_reset_at_unfreeze": optimizer_reset_at_unfreeze,
                "mgkg_target_samples": finetune_mgkg_target_samples,
                "soil_ptox_replay_samples": (
                    finetune_mgkg_replay_samples
                    if finetune_mgkg_loader is finetune_mgkg_replay_loader
                    else 0
                ),
                "swa_updates": swa_updates,
            }
            if should_update_swa(swa_cfg, phase="finetune_mgkg", epoch=epoch):
                if swa_model is None:
                    swa_model = torch.optim.swa_utils.AveragedModel(model)
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_last_global_epoch = global_epoch
                row["swa_updates"] = swa_updates
            history.append(row)
            if trajectory_cfg["enabled"]:
                trajectory_snapshots.append(
                    capture_trajectory_snapshot(
                        model,
                        dataset,
                        samples,
                        trajectory_anchor_indices,
                        out_path=trajectory_dir / f"epoch_{global_epoch:04d}_finetune_mgkg.npz",
                        phase="finetune_mgkg",
                        epoch=epoch,
                        global_epoch=global_epoch,
                        batch_size=finetune_mgkg_config.batch_size,
                        device=torch_device,
                        target_scaler=target_scaler,
                    )
                )
            step_scheduler(scheduler, monitor_loss)
            best_epoch = finetune_mgkg_best_epoch
            best_monitor_loss = finetune_mgkg_best_monitor_loss
            best_state = finetune_mgkg_best_state
            validation_msg = (
                ""
                if finetune_mgkg_validation_loss is None
                else f" validation_loss={finetune_mgkg_validation_loss['mean_loss']:.6f}"
            )
            print(
                f"[finetune_mgkg epoch {epoch} subphase={mgkg_subphase}] "
                f"mean_loss={epoch_loss['mean_loss']:.6f}{validation_msg} "
                f"samples={epoch_loss['samples']} best_epoch={best_epoch}",
                flush=True,
            )
            can_early_stop = (
                finetune_mgkg_head_only_epoch_count <= 0
                or epoch > finetune_mgkg_head_only_epoch_count
                or finetune_mgkg_head_only_epoch_count >= finetune_mgkg_config.epochs
            )
            if (
                finetune_mgkg_early_enabled
                and can_early_stop
                and finetune_mgkg_no_improve_epochs >= finetune_mgkg_patience
            ):
                print(
                    f"[finetune_mgkg early-stop] epoch={epoch} best_epoch={finetune_mgkg_best_epoch} "
                    f"monitor_loss={finetune_mgkg_best_monitor_loss:.6f}",
                    flush=True,
                )
                break
        if finetune_mgkg_best_state is not None:
            model.load_state_dict(finetune_mgkg_best_state)

    if swa_model is not None and swa_updates > 0:
        model.load_state_dict(swa_model.module.state_dict())
        best_state = clone_state_dict(model)
        best_epoch = swa_last_global_epoch or best_epoch
        swa_applied = True

    trajectory_summary = summarize_trajectory_snapshots(
        trajectory_snapshots,
        best_epoch=best_epoch,
        enabled=bool(trajectory_cfg["enabled"]),
    )
    if trajectory_cfg["enabled"]:
        (trajectory_dir / "trajectory_index.json").write_text(
            json.dumps(trajectory_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    requested_prediction_parts = normalize_prediction_split_parts(prediction_split_parts)
    prediction_indices = (
        [
            index
            for index, sample in enumerate(samples)
            if str(sample.get("split_part", "")).strip().lower()
            in requested_prediction_parts
        ]
        if requested_prediction_parts
        else list(range(len(samples)))
    )
    if not prediction_indices:
        raise ValueError(
            "prediction_split_parts selected no rows before model inference: "
            f"requested={sorted(requested_prediction_parts)}"
        )
    prediction_dataset = _IndexDataset(dataset, prediction_indices)
    prediction_samples = [samples[index] for index in prediction_indices]
    representation_cfg = resolve_final_representation_config(train_cfg)
    final_representation: dict[str, Any] = {"enabled": bool(representation_cfg["enabled"])}
    if representation_cfg["enabled"]:
        representation_dir = output_dir / "representations"
        representation_dir.mkdir(parents=True, exist_ok=True)
        final_representation.update(
            capture_trajectory_snapshot(
                model,
                dataset,
                samples,
                prediction_indices,
                out_path=representation_dir / "final_development_representation.npz",
                phase="final_best",
                epoch=best_epoch,
                global_epoch=best_epoch,
                batch_size=int(representation_cfg["batch_size"] or train_config.batch_size),
                device=torch_device,
                target_scaler=target_scaler,
            )
        )
    predictions = predict_all(
        model,
        prediction_dataset,
        prediction_samples,
        batch_size=train_config.batch_size,
        device=torch_device,
        num_workers=train_config.num_workers,
        target_scaler=target_scaler,
    )
    metrics_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        min_n_for_summary=metric_min_group_n,
    )
    filtered_metric_rows = filter_summary_metric_rows(metrics_rows)
    effect_level_metric_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        group_columns=("split_part", "task_head", "target_name", "medium_domain", "effect_level_x"),
        min_n_for_summary=metric_min_group_n,
    )
    filtered_effect_level_metric_rows = filter_summary_metric_rows(effect_level_metric_rows)
    toxicity_bin_metric_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        group_columns=(
            "split_part",
            "task_head",
            "target_name",
            "medium_domain",
            "toxicity_bin_status",
            "toxicity_bin_label",
            "toxicity_bin_boundary_flag",
        ),
        min_n_for_summary=metric_min_group_n,
    )
    toxicity_bin_boundary_audit_rows = build_toxicity_bin_boundary_audit_rows(predictions)
    split_medium_audit_rows = build_split_medium_audit_rows(frame, split_join_audit=split_join_audit)
    perturbation_predictions: list[dict[str, Any]] = []
    perturbation_summary_rows: list[dict[str, Any]] = []
    perturbation_metric_rows: list[dict[str, Any]] = []
    if perturbation_cfg.active():
        perturbation_predictions = predict_with_feature_perturbation(
            model,
            dataset,
            samples,
            split_parts=perturbation_cfg.split_parts,
            replicates=perturbation_cfg.replicates,
            numeric_noise_std=perturbation_cfg.numeric_noise_std,
            seed=perturbation_cfg.seed,
            batch_size=train_config.batch_size,
            device=torch_device,
            target_scaler=target_scaler,
        )
        perturbation_summary_rows = summarize_perturbation_predictions(perturbation_predictions)
        perturbation_metric_rows = metrics_by_group(
            perturbation_mean_predictions(perturbation_summary_rows),
            huber_delta=train_config.huber_delta,
            min_n_for_summary=metric_min_group_n,
        )

    history_path = output_dir / "history.csv"
    metrics_path = output_dir / "metrics.csv"
    metrics_filtered_path = output_dir / "metrics_filtered.csv"
    effect_level_metrics_path = output_dir / "effect_level_metrics.csv"
    effect_level_metrics_filtered_path = output_dir / "effect_level_metrics_filtered.csv"
    toxicity_bin_metrics_path = output_dir / "toxicity_bin_metrics.csv"
    toxicity_bin_boundary_audit_path = output_dir / "toxicity_bin_boundary_audit.csv"
    split_medium_audit_path = output_dir / "split_medium_audit.csv"
    predictions_path = output_dir / "predictions.csv"
    perturbation_predictions_path = output_dir / "test_noise_predictions.csv"
    perturbation_summary_path = output_dir / "test_noise_summary.csv"
    perturbation_metrics_path = output_dir / "test_noise_metrics.csv"
    manifest_path = output_dir / "manifest.json"
    preprocessing_path = output_dir / "preprocessing.json"
    best_model_path = output_dir / "best_model.pt"
    write_rows(history_path, history)
    write_rows(metrics_path, metrics_rows)
    write_rows(metrics_filtered_path, filtered_metric_rows)
    write_rows(effect_level_metrics_path, effect_level_metric_rows)
    write_rows(effect_level_metrics_filtered_path, filtered_effect_level_metric_rows)
    write_rows(toxicity_bin_metrics_path, toxicity_bin_metric_rows)
    write_rows(toxicity_bin_boundary_audit_path, toxicity_bin_boundary_audit_rows)
    write_rows(split_medium_audit_path, split_medium_audit_rows)
    write_rows(predictions_path, predictions)
    if perturbation_predictions:
        write_rows(perturbation_predictions_path, perturbation_predictions)
        write_rows(perturbation_summary_path, perturbation_summary_rows)
        write_rows(perturbation_metrics_path, perturbation_metric_rows)
    if best_state is not None:
        torch.save(best_state, best_model_path)
    else:
        best_model_path = None
    manifest = {
        "architecture_schema_version": 3,
        "runtime_config_sha256": canonical_sha256(config),
        "seed": seed,
        "split_name": split_name,
        "data_source": {
            "modeling_tables_db": str(Path(db_path)),
            "source_table": source_table or split_join_audit.get("source_table", ""),
        },
        "rows": len(samples),
        "development_only": development_only,
        "loaded_split_parts": sorted({str(sample.get("split_part", "")) for sample in samples}),
        "outer_test_rows_loaded": int(
            sum(str(sample.get("split_part", "")).strip().lower() == "test" for sample in samples)
        ),
        "outer_test_rows_predicted": int(
            sum(
                str(samples[index].get("split_part", "")).strip().lower() == "test"
                for index in prediction_indices
            )
        ),
        "trajectory": {
            **trajectory_cfg,
            **trajectory_summary,
            "anchor_manifest": str(trajectory_dir / "anchor_manifest.json")
            if trajectory_cfg["enabled"]
            else "",
            "index_path": str(trajectory_dir / "trajectory_index.json")
            if trajectory_cfg["enabled"]
            else "",
        },
        "final_representation": final_representation,
        "head_routing": head_routing_mode,
        "allowed_task_heads": list(allowed_task_heads),
        "allow_mixed_target_dimensions": mixed_target_dimensions_allowed,
        "train_rows": len(train_indices),
        "actual_train_rows": len(actual_train_indices),
        "finetune_rows": len(finetune_indices),
        "finetune_train_rows": len(finetune_train_indices),
        "finetune_validation_rows": len(finetune_validation_indices),
        "finetune_validation_source": finetune_validation_source,
        "finetune_mgkg_rows": len(finetune_mgkg_indices),
        "finetune_mgkg_train_rows": len(finetune_mgkg_train_indices),
        "finetune_mgkg_validation_rows": len(finetune_mgkg_validation_indices),
        "finetune_mgkg_validation_source": finetune_mgkg_validation_source,
        "prediction_output_split_parts": (
            sorted(requested_prediction_parts) if requested_prediction_parts else ["all"]
        ),
        "validation_rows": len(validation_indices),
        "validation_source": validation_source,
        "validation_seed": int(seed if validation_seed is None else validation_seed),
        "finetune_validation_seed": int(
            seed if finetune_validation_seed is None else finetune_validation_seed
        ),
        "finetune_mgkg_validation_seed": int(
            seed + 20_000
            if finetune_mgkg_validation_seed is None
            else finetune_mgkg_validation_seed
        ),
        "task_heads": list(task_heads),
        "skipped_tasks": skipped_tasks,
        "task_filter": {
            "min_total": min_total,
            "min_train": min_train,
            "min_eval": min_eval,
        },
        "fingerprint_size": fingerprint_size,
        "encoder_source": encoder.source,
        "molecular_descriptor_names": list(descriptor_names),
        "descriptor_encoder": descriptor_encoder_cfg["manifest"],
        "device": train_config.device,
        "epochs": train_config.epochs,
        "finetune_epochs": finetune_config.epochs,
        "finetune_mgkg_epochs": finetune_mgkg_config.epochs,
        "finetune_epochs_ran": finetune_ran,
        "finetune_mgkg_epochs_ran": finetune_mgkg_ran,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_monitor_loss": best_monitor_loss if math.isfinite(best_monitor_loss) else None,
        "early_stopping": {
            "enabled": early_enabled,
            "requested": bool(early_cfg["enabled"]),
            "patience": int(early_cfg["patience"]),
            "min_delta": float(early_cfg["min_delta"]),
            "validation_fraction": float(early_cfg["validation_fraction"]),
            "requested_monitor_split": requested_stage1_monitor_split,
            "monitor_split": early_cfg["monitor_split"],
        },
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "dropout": deep_model_config.dropout,
        "model_architecture": {
            "config": asdict(deep_model_config),
            "resolved_hidden_dims": list(deep_model_config.hidden_dims),
            "resolved_molecular_hidden_dims": list(deep_model_config.molecular_hidden_dims),
            "resolved_context_hidden_dims": list(deep_model_config.context_hidden_dims),
            "fusion_mode": deep_model_config.fusion_mode,
            "fusion_output_dim": int(model.fusion_output_dim),
            "molecular_core_input_dim": int(model.molecular_core_input_dim),
            "context_input_dim": int(model.context_input_dim),
            "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "config_sha256": canonical_sha256(asdict(deep_model_config)),
        },
        "evaluation_checkpoint": evaluation_checkpoint_audit,
        "optimizer": train_config.optimizer,
        "weight_decay": train_config.weight_decay,
        "gradient_clip_norm": train_config.gradient_clip_norm,
        "scheduler": train_config.scheduler,
        "regression_loss": {
            "kind": "huber_mse_hybrid" if train_config.mse_loss_weight > 0 else "huber",
            "huber_delta": train_config.huber_delta,
            "mse_weight": train_config.mse_loss_weight,
        },
        "target_standardization": target_scaler.to_manifest(),
        "feature_zscore_correction": zscore_correction.to_manifest(),
        "metric_reporting": {
            "min_metric_group_n": metric_min_group_n,
            "filtered_metrics_path": str(metrics_filtered_path),
            "effect_level_metrics_path": str(effect_level_metrics_path),
            "effect_level_metrics_filtered_path": str(effect_level_metrics_filtered_path),
            "toxicity_bin_metrics_path": str(toxicity_bin_metrics_path),
            "toxicity_bin_boundary_audit_path": str(toxicity_bin_boundary_audit_path),
        },
        "split_join_audit": split_join_audit,
        "split_medium_audit_path": str(split_medium_audit_path),
        "augmentation": augmentation_cfg.to_manifest(),
        "test_perturbation": perturbation_cfg.to_manifest(),
        "source_weighting": source_weighting_summary,
        "effect_level_weighting": effect_level_weighting_summary,
        "toxicity_binning": toxicity_binning_summary,
        "censored_loss": censored_summary,
        "domain_alignment": domain_alignment_summary,
        "swa": {
            **swa_cfg.to_manifest(),
            "updates": swa_updates,
            "last_global_epoch": swa_last_global_epoch,
            "applied": swa_applied,
        },
        "task_weighting": str(train_cfg.get("task_weighting", "balanced")),
        "task_weights": train_config.task_weights,
        "categorical_min_count": category_min_count,
        "finetune_batch_size": finetune_config.batch_size,
        "finetune_learning_rate": finetune_config.learning_rate,
        "finetune": {
            "requested": finetune_requested,
            "enabled": bool(finetune_requested and finetune_indices),
            "epochs": finetune_config.epochs,
            "epochs_ran": finetune_ran,
            "rows": len(finetune_indices),
            "train_rows": len(finetune_train_indices),
            "validation_rows": len(finetune_validation_indices),
            "validation_source": finetune_validation_source,
            "early_stopping": finetune_early_enabled,
            "early_stopping_patience": finetune_patience,
            "early_stopping_min_delta": finetune_min_delta,
            "learning_rate": finetune_config.learning_rate,
            "batch_size": finetune_config.batch_size,
            "freeze": finetune_freeze_mode,
        },
        "finetune_mgkg": {
            "requested": finetune_mgkg_requested,
            "enabled": bool(finetune_mgkg_requested and finetune_mgkg_indices),
            "epochs": finetune_mgkg_config.epochs,
            "epochs_ran": finetune_mgkg_ran,
            "rows": len(finetune_mgkg_indices),
            "train_rows": len(finetune_mgkg_train_indices),
            "validation_rows": len(finetune_mgkg_validation_indices),
            "validation_source": finetune_mgkg_validation_source,
            "early_stopping": finetune_mgkg_early_enabled,
            "early_stopping_requested": finetune_mgkg_early_requested,
            "early_stopping_patience": finetune_mgkg_patience,
            "early_stopping_min_delta": finetune_mgkg_min_delta,
            "learning_rate": finetune_mgkg_config.learning_rate,
            "head_learning_rate": finetune_mgkg_config.learning_rate,
            "trunk_learning_rate": (
                finetune_mgkg_trunk_lr if finetune_mgkg_trunk_lr > 0 else None
            ),
            "batch_size": finetune_mgkg_config.batch_size,
            "freeze": finetune_mgkg_freeze_mode,
            "head_only_epochs": finetune_mgkg_head_only_epoch_count,
            "post_unfreeze_epochs": max(
                finetune_mgkg_config.epochs - finetune_mgkg_head_only_epoch_count,
                0,
            ),
            "optimizer_reset_at_unfreeze": bool(
                finetune_mgkg_head_only_epoch_count > 0
                and finetune_mgkg_head_only_epoch_count < finetune_mgkg_config.epochs
            ),
            "soil_ptox_replay_fraction_requested": finetune_mgkg_replay_ratio,
            "soil_ptox_replay_start_epoch": (
                finetune_mgkg_head_only_epoch_count + 1
                if finetune_mgkg_replay_ratio > 0
                else None
            ),
            "soil_ptox_replay_source": "finetune_training_only",
            "soil_ptox_replay_sampling": "fixed_row_fraction_per_batch",
            "soil_ptox_replay_objective_weighting": "existing_task_balanced_loss",
            "soil_ptox_replay_candidate_rows": len(finetune_train_indices),
            "mgkg_target_samples_per_epoch": finetune_mgkg_target_samples,
            "soil_ptox_replay_samples_per_epoch": finetune_mgkg_replay_samples,
            "mgkg_target_rows_per_full_batch": finetune_mgkg_target_per_full_batch,
            "soil_ptox_replay_rows_per_full_batch": finetune_mgkg_replay_per_full_batch,
            "soil_ptox_replay_fraction_realized": (
                finetune_mgkg_replay_samples
                / max(finetune_mgkg_target_samples + finetune_mgkg_replay_samples, 1)
            ),
            "toxicity_bin_loss_weight": finetune_mgkg_config.toxicity_bin_loss_weight,
            "regression_loss": {
                "kind": (
                    "huber_mse_hybrid"
                    if finetune_mgkg_config.mse_loss_weight > 0
                    else "huber"
                ),
                "huber_delta": finetune_mgkg_config.huber_delta,
                "mse_weight": finetune_mgkg_config.mse_loss_weight,
            },
            "target_bin_sampling": target_bin_sampling_audit,
            "hierarchical_head": hierarchical_head_spec,
            "stage3_init_checkpoint": finetune_mgkg_checkpoint_audit,
            "soil_ptox_replay_boundary_audit": finetune_mgkg_replay_audit,
        },
        "mgkg_residual_adapter": {
            "enabled": use_mgkg_residual_adapter,
            "kind": "zero_initialized_residual_bottleneck",
            "target_name": "neg_log10_mg_kg",
            "target_family": "solid_neglog_mg_kg",
            "task_heads": list(mgkg_adapter_heads),
            "bottleneck_dim": mgkg_adapter_bottleneck,
            "zero_initialized": True,
        },
        "numeric_dim": dataset.numeric_dim(),
        "fingerprint_dim": dataset.fingerprint_dim(),
        "graph_atom_feature_dim": 0 if graph_encoder is None else graph_encoder.atom_feature_dim,
        "graph_edge_feature_dim": 0 if graph_encoder is None else graph_encoder.edge_feature_dim,
        "graph_embedding_dim": 0 if graph_encoder is None else graph_encoder_cfg["embedding_dim"],
        "categorical_cardinalities": categorical_cardinalities,
        "adapter_cardinality": max(adapter_map.values(), default=0) + 1 if adapter_map else 0,
        "adapter_map": adapter_map,
        "ablation": ablation_spec.name,
        "ablation_features": {
            "use_descriptors": ablation_spec.use_descriptors,
            "use_fingerprint": ablation_spec.use_fingerprint,
            "use_molecular_graph": ablation_spec.use_molecular_graph,
            "use_context_numeric": ablation_spec.use_context_numeric,
            "use_effect_level_features": ablation_spec.use_effect_level_features,
            "use_duration_features": ablation_spec.use_duration_features,
            "use_species_lifestage": ablation_spec.use_species_lifestage,
            "use_other_categorical_context": ablation_spec.use_other_categorical_context,
            "use_molecular_residual": ablation_spec.use_molecular_residual,
            "use_medium_adapter": ablation_spec.use_medium_adapter,
            "masked_descriptor_names": list(ablation_spec.masked_descriptor_names),
        },
    }
    preprocessing = checkpoint_preprocessing
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing_path.write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")

    return DeepExperimentResult(
        out_dir=output_dir,
        metrics_path=metrics_path,
        history_path=history_path,
        manifest_path=manifest_path,
        preprocessing_path=preprocessing_path,
        predictions_path=predictions_path,
        best_model_path=best_model_path,
        encoder_source=encoder.source,
        trained_tasks=task_heads,
        rows=len(samples),
        ablation=ablation_spec.name,
        best_epoch=best_epoch,
        early_stopping_enabled=early_enabled,
    )


def get_ablation_spec(name: str | None) -> AblationSpec:
    key = (name or "full").strip().lower()
    if key not in ABLATION_SPECS:
        allowed = ", ".join(sorted(ABLATION_SPECS))
        raise ValueError(f"Unknown ablation '{name}'. Allowed values: {allowed}")
    return ABLATION_SPECS[key]


def normalize_head_routing(value: Any) -> str:
    normalized = str(value or "task").strip().lower()
    aliases = {
        "task": "task",
        "task_only": "task",
        "task_target": "task_target",
        "task_target_family": "task_target",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported head routing '{value}'. Allowed values: {allowed}")
    return aliases[normalized]


def apply_head_routing(frame: Any, *, mode: str) -> Any:
    """Route output heads while retaining the scientific task label for reports."""
    if "task_head" not in frame.columns:
        raise ValueError("Head routing requires a task_head column.")
    routed = frame.copy()
    base = routed["task_head"].map(_category_value)
    routed["base_task_head"] = base
    if mode == "task":
        routed["model_head"] = base
    elif mode == "task_target":
        if "target_family" in routed.columns:
            target = routed["target_family"].map(_category_value)
        elif "target_name" in routed.columns:
            target = routed["target_name"].map(_category_value)
        else:
            raise ValueError("task_target head routing requires target_family or target_name.")
        routed["model_head"] = base + "__" + target
    else:  # pragma: no cover - normalize_head_routing guards this boundary.
        raise ValueError(f"Unsupported normalized head routing mode: {mode}")
    routed["task_head"] = routed["model_head"]
    return routed


def build_preprocessing_manifest(
    *,
    categorical_maps: dict[str, dict[str, int]],
    adapter_map: dict[str, int] | None = None,
    numeric_stats: dict[str, tuple[float, float]],
    ablation: AblationSpec,
    fingerprint_size: int,
    encoder_source: str,
    cache_path: str | None,
    descriptor_count: int | None = None,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    descriptor_encoder: Mapping[str, Any] | None = None,
    graph_encoder: Mapping[str, Any] | None = None,
    target_scaler: TargetScaler | None = None,
    zscore_correction: ZScoreCorrection | None = None,
    categorical_min_count: int = 1,
) -> dict[str, Any]:
    descriptor_names = tuple(
        descriptor_names
        if descriptor_names is not None
        else _descriptor_feature_names(descriptor_count or len(MOLECULAR_DESCRIPTOR_NAMES))
    )
    numeric_feature_names = build_numeric_feature_names(
        descriptor_count or len(descriptor_names) or len(MOLECULAR_DESCRIPTOR_NAMES),
        descriptor_names=descriptor_names,
    )
    numeric_feature_names = numeric_feature_names[: len(numeric_stats)]
    masked_indices = descriptor_mask_indices(len(descriptor_names), ablation, descriptor_names=descriptor_names)
    return {
        "schema_version": 2,
        "ablation": ablation.name,
        "fingerprint_size": int(fingerprint_size),
        "encoder_source": encoder_source,
        "molecular_feature_cache": cache_path or "",
        "molecular_graph_encoder": dict(graph_encoder or {}),
        "target_standardization": {} if target_scaler is None else target_scaler.to_manifest(),
        "feature_zscore_correction": {} if zscore_correction is None else zscore_correction.to_manifest(),
        "numeric_feature_names": numeric_feature_names,
        "molecular_descriptor_names": descriptor_names,
        "descriptor_encoder": dict(descriptor_encoder or {"mode": "raw"}),
        "masked_descriptor_names": list(ablation.masked_descriptor_names),
        "masked_descriptor_indices": list(masked_indices),
        "context_numeric_columns": list(CONTEXT_NUMERIC_COLUMNS),
        "effect_level_numeric_columns": list(EFFECT_LEVEL_NUMERIC_COLUMNS),
        "effect_level_numeric_indices": list(effect_level_feature_indices(numeric_feature_names)),
        "duration_context_columns": sorted(DURATION_CONTEXT_COLUMNS),
        "categorical_columns": list(active_categorical_columns(ablation)),
        "categorical_min_count": int(categorical_min_count),
        "categorical_special_tokens": {
            "missing": MISSING_CATEGORY_TOKEN,
            "unknown": UNKNOWN_CATEGORY_TOKEN,
            "rare": RARE_CATEGORY_TOKEN,
        },
        "categorical_maps": categorical_maps,
        "adapter_columns": ["medium_domain", "target_family_or_target_name"],
        "adapter_map": adapter_map or {},
        "adapter_cardinality": max((adapter_map or {}).values(), default=0) + 1 if adapter_map else 0,
        "numeric_stats": {
            name: {
                "index": idx,
                "mean": float(numeric_stats[str(idx)][0]),
                "std": float(numeric_stats[str(idx)][1]),
            }
            for idx, name in enumerate(numeric_feature_names)
        },
        "ablation_features": {
            "use_descriptors": ablation.use_descriptors,
            "use_fingerprint": ablation.use_fingerprint,
            "use_molecular_graph": ablation.use_molecular_graph,
            "use_context_numeric": ablation.use_context_numeric,
            "use_effect_level_features": ablation.use_effect_level_features,
            "use_duration_features": ablation.use_duration_features,
            "use_species_lifestage": ablation.use_species_lifestage,
            "use_other_categorical_context": ablation.use_other_categorical_context,
            "use_molecular_residual": ablation.use_molecular_residual,
            "use_medium_adapter": ablation.use_medium_adapter,
            "masked_descriptor_names": list(ablation.masked_descriptor_names),
        },
    }


def _descriptor_count(frame: Any, encoder: MolecularFeatureBuilder, ablation: AblationSpec) -> int:
    for value in frame.get("smiles", []):
        descriptors = masked_descriptors(encoder.encode(value)[0], ablation)
        return len(descriptors)
    return len(MOLECULAR_DESCRIPTOR_NAMES)


def _descriptor_feature_names(count: int) -> list[str]:
    if count == len(MOLECULAR_DESCRIPTOR_NAMES):
        return list(MOLECULAR_DESCRIPTOR_NAMES)
    return [f"descriptor_{idx}" for idx in range(count)]


def build_numeric_feature_names(
    descriptor_count: int,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    names = list(descriptor_names or _descriptor_feature_names(descriptor_count))
    if len(names) != int(descriptor_count):
        names = _descriptor_feature_names(descriptor_count)
    return tuple(names + list(CONTEXT_NUMERIC_COLUMNS))


def _descriptor_encoder_config(config: Mapping[str, Any], descriptor_names: tuple[str, ...]) -> dict[str, Any]:
    molecule_cfg = config.get("features", {}).get("molecule", {})
    if not isinstance(molecule_cfg, Mapping):
        molecule_cfg = {}
    encoder_cfg = molecule_cfg.get("descriptor_head", {})
    if not isinstance(encoder_cfg, Mapping):
        encoder_cfg = {}
    mode = str(
        encoder_cfg.get(
            "mode",
            molecule_cfg.get("descriptor_encoder_mode", "raw"),
        )
        or "raw"
    ).strip().lower()
    cluster_file = str(
        encoder_cfg.get(
            "cluster_file",
            molecule_cfg.get("descriptor_cluster_file", ""),
        )
        or ""
    ).strip()
    clustered_modes = {"prior_clustered", "prior_clustered_heads", "clustered", "clustered_heads"}
    group_indices = (
        _load_descriptor_group_indices(cluster_file, descriptor_names)
        if cluster_file and mode in clustered_modes
        else {}
    )
    head_dim = int(encoder_cfg.get("head_dim", molecule_cfg.get("descriptor_head_dim", 64)))
    group_head_dim = int(encoder_cfg.get("group_head_dim", molecule_cfg.get("descriptor_group_head_dim", 16)))
    manifest = {
        "mode": mode,
        "head_dim": head_dim,
        "group_head_dim": group_head_dim,
        "cluster_file": cluster_file,
        "group_count": len(group_indices),
        "groups": {key: list(value) for key, value in group_indices.items()},
    }
    return {
        "mode": mode,
        "head_dim": head_dim,
        "group_head_dim": group_head_dim,
        "group_indices": group_indices,
        "manifest": manifest,
    }


def _load_descriptor_group_indices(
    cluster_file: str,
    descriptor_names: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    from qsar_tl.features.descriptor_groups import load_descriptor_group_indices

    return load_descriptor_group_indices(cluster_file, descriptor_names)


def _flatten_descriptor_groups(raw: Any, name_to_index: Mapping[str, int], prefix: str = "") -> dict[str, tuple[int, ...]]:
    if isinstance(raw, Mapping):
        result: dict[str, tuple[int, ...]] = {}
        for key, value in raw.items():
            group_name = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_descriptor_groups(value, name_to_index, group_name))
        return result
    if isinstance(raw, list):
        indices = tuple(
            sorted(
                {
                    int(name_to_index[str(name)])
                    for name in raw
                    if str(name) in name_to_index
                }
            )
        )
        return {prefix: indices} if indices else {}
    return {}


def effect_level_feature_indices(feature_names: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    effect_columns = set(EFFECT_LEVEL_NUMERIC_COLUMNS)
    return tuple(idx for idx, name in enumerate(feature_names) if name in effect_columns)


def apply_finetune_freeze(model: Any, mode: str) -> list[Any]:
    normalized = (mode or "none").strip().lower()
    if normalized in {"", "none", "all"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    for parameter in model.parameters():
        parameter.requires_grad = False

    residual_adapter = getattr(model, "mgkg_residual_adapter", None)
    if normalized == "heads_only":
        modules = model_head_modules(model)
        if residual_adapter is not None:
            modules.append(residual_adapter)
    elif normalized == "last_trunk":
        modules = [*model_head_modules(model), _last_parameterized_trunk_module(model)]
        if residual_adapter is not None:
            modules.append(residual_adapter)
    elif normalized == "heads_embeddings":
        modules = [*model_head_modules(model), model.embeddings, model.adapters]
        if residual_adapter is not None:
            modules.append(residual_adapter)
    else:
        allowed = "none, heads_only, last_trunk, heads_embeddings"
        raise ValueError(f"Unsupported finetune freeze mode '{mode}'. Allowed values: {allowed}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters for finetune freeze mode '{mode}'.")
    return trainable


def model_head_modules(model: Any) -> list[Any]:
    """Return every prediction-head module, including optional stage-3 heads."""

    modules = [model.heads]
    hierarchical_shared = getattr(model, "mgkg_hierarchical_shared_head", None)
    if hierarchical_shared is not None:
        modules.append(hierarchical_shared)
    hierarchical_families = getattr(model, "mgkg_hierarchical_family_heads", None)
    if hierarchical_families is not None:
        modules.append(hierarchical_families)
    return modules


def _last_parameterized_trunk_module(model: Any) -> Any:
    trunk = getattr(model, "trunk", None)
    if trunk is None:
        raise ValueError("last_trunk freeze mode requires model.trunk.")
    candidates = [
        module
        for module in trunk.modules()
        if module is not trunk and any(True for _ in module.parameters(recurse=False))
    ]
    if not candidates:
        raise ValueError("last_trunk freeze mode found no parameterized trunk layer.")
    return candidates[-1]


def build_finetune_parameter_groups(
    model: Any,
    *,
    freeze_mode: str,
    head_learning_rate: float,
    trunk_learning_rate: float,
) -> list[Any]:
    """Freeze the requested modules and optionally assign a lower LR off-head."""

    trainable = apply_finetune_freeze(model, freeze_mode)
    if trunk_learning_rate <= 0:
        return trainable

    head_parameter_ids = {
        id(parameter)
        for module in model_head_modules(model)
        for parameter in module.parameters()
    }
    residual_adapter = getattr(model, "mgkg_residual_adapter", None)
    if residual_adapter is not None:
        head_parameter_ids.update(id(parameter) for parameter in residual_adapter.parameters())
    head_parameters = [parameter for parameter in trainable if id(parameter) in head_parameter_ids]
    trunk_parameters = [parameter for parameter in trainable if id(parameter) not in head_parameter_ids]
    groups: list[Any] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": float(head_learning_rate), "name": "head"})
    if trunk_parameters:
        groups.append({"params": trunk_parameters, "lr": float(trunk_learning_rate), "name": "trunk"})
    return groups or trainable


def optimizer_learning_rates(optimizer: Any) -> dict[str, float]:
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        rates[name] = float(group.get("lr", 0.0))
    return rates


def audit_mgkg_replay_boundary(
    samples: list[dict[str, Any]],
    *,
    replay_indices: list[int],
    mgkg_indices: list[int],
) -> dict[str, Any]:
    """Fail closed when source replay reaches any validation/test identity."""

    replay_set = set(replay_indices)
    evaluation_indices = [
        index
        for index, sample in enumerate(samples)
        if str(sample.get("split_part", "")).strip().lower()
        in {"validation", "valid", "test", "finetune_validation", "finetune_mgkg_validation"}
    ]
    if replay_set & set(evaluation_indices):
        raise ValueError("Soil pTox replay indices overlap validation/test indices.")

    invalid_replay = [
        index
        for index in replay_indices
        if str(samples[index].get("target_name", "")).strip() != "ptox_mol_l"
        or str(samples[index].get("target_family", "")).strip() != "aquatic_pTox_mol_L"
        or str(samples[index].get("medium_domain", "")).strip().lower() != "soil"
    ]
    invalid_mgkg = [
        index
        for index in mgkg_indices
        if str(samples[index].get("target_name", "")).strip() != "neg_log10_mg_kg"
        or str(samples[index].get("target_family", "")).strip() != "solid_neglog_mg_kg"
        or str(samples[index].get("medium_domain", "")).strip().lower() != "soil"
    ]
    if invalid_replay:
        raise ValueError(f"Replay pool contains non-soil-pTox rows: {invalid_replay[:5]}")
    if invalid_mgkg:
        raise ValueError(f"mg/kg target pool contains invalid rows: {invalid_mgkg[:5]}")

    replay_aggregates = _sample_aggregate_ids(samples, replay_indices)
    evaluation_aggregates = _sample_aggregate_ids(samples, evaluation_indices)
    replay_results = _sample_result_ids(samples, replay_indices)
    evaluation_results = _sample_result_ids(samples, evaluation_indices)
    aggregate_overlap = replay_aggregates & evaluation_aggregates
    result_overlap = replay_results & evaluation_results
    if aggregate_overlap or result_overlap:
        raise ValueError(
            "Soil pTox replay has exact source overlap with validation/test rows: "
            f"aggregate_id={len(aggregate_overlap)}, result_id={len(result_overlap)}"
        )
    return {
        "replay_candidate_rows": len(replay_indices),
        "evaluation_rows_checked": len(evaluation_indices),
        "aggregate_id_overlap": 0,
        "result_id_overlap": 0,
        "replay_aggregate_hash": _stable_values_hash(replay_aggregates),
        "replay_result_id_hash": _stable_values_hash(replay_results),
    }


def _sample_aggregate_ids(samples: list[dict[str, Any]], indices: list[int]) -> set[str]:
    return {
        str(samples[index].get("aggregate_id", "")).strip()
        for index in indices
        if str(samples[index].get("aggregate_id", "")).strip()
    }


def _sample_result_ids(samples: list[dict[str, Any]], indices: list[int]) -> set[str]:
    result: set[str] = set()
    for index in indices:
        raw = samples[index].get("result_ids")
        if raw in (None, ""):
            continue
        if isinstance(raw, str):
            try:
                values = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid result_ids JSON at sample index {index}.") from exc
        else:
            values = raw
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f"result_ids must be a sequence at sample index {index}.")
        result.update(str(value).strip() for value in values if str(value).strip())
    return result


def _stable_values_hash(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MolecularFeatureBuilder:
    def __init__(self, *, fingerprint_size: int = 512, cache_path: str | Path | None = None) -> None:
        self.fingerprint_size = fingerprint_size
        self.cache = load_molecular_feature_cache(cache_path, fingerprint_size=fingerprint_size) if cache_path else {}
        self.cache_descriptor_names = (
            load_molecular_feature_cache_descriptor_names(cache_path, fingerprint_size=fingerprint_size)
            if cache_path
            else ()
        )
        self.cache_source = (
            load_molecular_feature_cache_source(cache_path, fingerprint_size=fingerprint_size)
            if cache_path
            else ""
        )
        self._rdkit_available = self._check_rdkit()
        if self.cache:
            self.source = self.cache_source or "molecular_feature_cache"
        elif self._rdkit_available:
            self.source = "rdkit"
        else:
            self.source = "stable_smiles_fallback"

    @staticmethod
    def _check_rdkit() -> bool:
        try:
            import rdkit  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _normalize_smiles(smiles: object) -> str:
        if smiles is None:
            return ""
        try:
            if isinstance(smiles, (float, np.floating)) and not math.isfinite(float(smiles)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(smiles).strip()
        if text.lower() in {"", "nan", "none", "null", "na", "n/a", "<na>"}:
            return ""
        return text

    def encode(self, smiles: object) -> tuple[list[float], list[float]]:
        text = self._normalize_smiles(smiles)
        cached = self.cache.get(text)
        if cached is not None:
            return self._align_to_cache_schema(cached)
        if self.cache_descriptor_names:
            return self._encode_cache_miss(text)
        if self._rdkit_available and text.strip():
            try:
                return self._encode_rdkit(text)
            except Exception:
                return self._encode_fallback(text)
        return self._encode_fallback(text)

    def _encode_rdkit(self, smiles: str) -> tuple[list[float], list[float]]:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._encode_fallback(smiles)
        descriptors = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
        ]
        generator = GetMorganGenerator(radius=2, fpSize=self.fingerprint_size)
        fingerprint = generator.GetFingerprint(mol)
        return descriptors, [float(bit) for bit in fingerprint.ToBitString()]

    def _align_to_cache_schema(self, features: tuple[list[float], list[float]]) -> tuple[list[float], list[float]]:
        descriptors, fingerprint = features
        if self.cache_descriptor_names:
            descriptor_count = len(self.cache_descriptor_names)
            descriptors = [float(value) for value in descriptors[:descriptor_count]]
            if len(descriptors) < descriptor_count:
                descriptors.extend([0.0] * (descriptor_count - len(descriptors)))
        fingerprint = [float(value) for value in fingerprint[: self.fingerprint_size]]
        if len(fingerprint) < self.fingerprint_size:
            fingerprint.extend([0.0] * (self.fingerprint_size - len(fingerprint)))
        return descriptors, fingerprint

    def _encode_cache_miss(self, smiles: str) -> tuple[list[float], list[float]]:
        descriptors = [0.0] * len(self.cache_descriptor_names)
        return descriptors, self._fingerprint_from_smiles(smiles)

    def _fingerprint_from_smiles(self, smiles: str) -> list[float]:
        text = smiles or ""
        if self._rdkit_available and text.strip():
            try:
                from rdkit import Chem
                from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

                mol = Chem.MolFromSmiles(text)
                if mol is not None:
                    generator = GetMorganGenerator(radius=2, fpSize=self.fingerprint_size)
                    return [float(bit) for bit in generator.GetFingerprint(mol).ToBitString()]
            except Exception:
                pass
        bits = [0.0] * self.fingerprint_size
        for ngram in _smiles_ngrams(text):
            digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).hexdigest()
            bits[int(digest, 16) % self.fingerprint_size] = 1.0
        return bits

    def descriptor_names(self, descriptor_count: int | None = None) -> tuple[str, ...]:
        count = int(descriptor_count or 0)
        if self.cache_descriptor_names and (count <= 0 or len(self.cache_descriptor_names) == count):
            return tuple(self.cache_descriptor_names)
        if count == len(MOLECULAR_DESCRIPTOR_NAMES) and self.source in {"rdkit", "rdkit_cache"}:
            return tuple(MOLECULAR_DESCRIPTOR_NAMES)
        if count > 0:
            return tuple(f"descriptor_{idx}" for idx in range(count))
        return tuple(MOLECULAR_DESCRIPTOR_NAMES)

    def _encode_fallback(self, smiles: str) -> tuple[list[float], list[float]]:
        text = smiles or ""
        counts = Counter(text)
        descriptors = [
            float(len(text)),
            float(sum(ch.isupper() for ch in text)),
            float(sum(ch.islower() for ch in text)),
            float(counts.get("C", 0)),
            float(counts.get("N", 0)),
            float(counts.get("O", 0)),
            float(counts.get("S", 0)),
            float(counts.get("P", 0)),
        ]
        bits = [0.0] * self.fingerprint_size
        for ngram in _smiles_ngrams(text):
            digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).hexdigest()
            bits[int(digest, 16) % self.fingerprint_size] = 1.0
        return descriptors, bits


class MolecularGraphFeatureBuilder:
    def __init__(self, *, cache_path: str | Path | None = None) -> None:
        from qsar_tl.features.molecular_graph import (
            ATOM_FEATURE_NAMES,
            BOND_FEATURE_NAMES,
            empty_molecular_graph,
            load_molecular_graph_cache,
        )

        self.cache_path = "" if cache_path is None else str(cache_path)
        self.cache = load_molecular_graph_cache(cache_path) if cache_path else {}
        self.atom_feature_names = tuple(ATOM_FEATURE_NAMES)
        self.bond_feature_names = tuple(BOND_FEATURE_NAMES)
        if self.cache:
            first = next(iter(self.cache.values()))
            self.atom_feature_names = tuple(first.get("atom_feature_names") or self.atom_feature_names)
            self.bond_feature_names = tuple(first.get("bond_feature_names") or self.bond_feature_names)
        self.atom_feature_dim = len(self.atom_feature_names)
        self.edge_feature_dim = len(self.bond_feature_names)
        self._empty_graph = empty_molecular_graph

    def encode(self, smiles: object) -> dict[str, Any]:
        text = MolecularFeatureBuilder._normalize_smiles(smiles)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        return self._empty_graph(text)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "cache_path": self.cache_path,
            "cache_rows": len(self.cache),
            "atom_feature_dim": self.atom_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "atom_feature_names": list(self.atom_feature_names),
            "bond_feature_names": list(self.bond_feature_names),
        }


def _molecular_cache_path(config: Mapping[str, Any]) -> str | None:
    value = config.get("experiment", {}).get("molecular_feature_cache")
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _molecular_graph_cache_path(config: Mapping[str, Any]) -> str | None:
    molecule_cfg = config.get("features", {}).get("molecule", {})
    if not isinstance(molecule_cfg, Mapping):
        return None
    value = molecule_cfg.get("graph_cache")
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _graph_encoder_config(config: Mapping[str, Any]) -> dict[str, int]:
    molecule_cfg = config.get("features", {}).get("molecule", {})
    if not isinstance(molecule_cfg, Mapping):
        molecule_cfg = {}
    graph_cfg = molecule_cfg.get("graph_encoder", {})
    if not isinstance(graph_cfg, Mapping):
        graph_cfg = {}
    return {
        "embedding_dim": int(graph_cfg.get("embedding_dim", molecule_cfg.get("graph_embedding_dim", 64))),
        "message_steps": int(graph_cfg.get("message_steps", molecule_cfg.get("graph_message_steps", 2))),
    }


def load_molecular_feature_cache(
    cache_path: str | Path | None,
    *,
    fingerprint_size: int,
) -> dict[str, tuple[list[float], list[float]]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    cache: dict[str, tuple[list[float], list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row.get("smiles", ""))
            descriptors = [float(value) for value in row.get("descriptors", [])]
            fingerprint = [float(value) for value in row.get("fingerprint", [])]
            if len(fingerprint) != fingerprint_size:
                continue
            cache[smiles] = (descriptors, fingerprint)
    return cache


def load_molecular_feature_cache_descriptor_names(
    cache_path: str | Path | None,
    *,
    fingerprint_size: int,
) -> tuple[str, ...]:
    row = _first_valid_molecular_cache_row(cache_path, fingerprint_size=fingerprint_size)
    if not row:
        return ()
    names = row.get("descriptor_names") or row.get("molecular_descriptor_names") or []
    descriptors = row.get("descriptors", [])
    if isinstance(names, list) and len(names) == len(descriptors):
        return tuple(str(name) for name in names)
    return tuple(f"descriptor_{idx}" for idx in range(len(descriptors)))


def load_molecular_feature_cache_source(
    cache_path: str | Path | None,
    *,
    fingerprint_size: int,
) -> str:
    row = _first_valid_molecular_cache_row(cache_path, fingerprint_size=fingerprint_size)
    if not row:
        return ""
    for key in ("feature_source", "encoder_source", "source"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    names = row.get("descriptor_names") or []
    if names and len(names) != len(MOLECULAR_DESCRIPTOR_NAMES):
        return "descriptor_cache"
    return "rdkit_cache"


def _first_valid_molecular_cache_row(
    cache_path: str | Path | None,
    *,
    fingerprint_size: int,
) -> dict[str, Any]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fingerprint = row.get("fingerprint", [])
            if len(fingerprint) == fingerprint_size:
                return row
    return {}


def build_molecular_feature_cache(
    db_path: str | Path,
    *,
    out_path: str | Path,
    fingerprint_size: int = 512,
    limit: int | None = None,
    source_table: str = "aggregated_task_records",
) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT smiles, COUNT(*) AS n
            FROM "{source_table}"
            WHERE smiles IS NOT NULL
              AND TRIM(CAST(smiles AS TEXT)) <> ''
            GROUP BY smiles
            ORDER BY n DESC, smiles
            {limit_clause}
            """
        ).fetchall()
    written = 0
    fallback_count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for smiles, count in rows:
            descriptors, fingerprint = encoder.encode(smiles)
            if encoder.source != "rdkit" or len(descriptors) != len(MOLECULAR_DESCRIPTOR_NAMES):
                fallback_count += 1
            payload = {
                "smiles": smiles,
                "source_count": int(count),
                "descriptor_names": list(MOLECULAR_DESCRIPTOR_NAMES),
                "descriptors": descriptors,
                "fingerprint_size": fingerprint_size,
                "fingerprint": fingerprint,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    manifest = {
        "out_path": str(out),
        "source_table": source_table,
        "fingerprint_size": fingerprint_size,
        "encoder_source": encoder.source,
        "unique_smiles": len(rows),
        "written": written,
        "fallback_count": fallback_count,
    }
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _smiles_ngrams(text: str) -> list[str]:
    if not text:
        return ["<missing>"]
    grams = [text[idx : idx + width] for width in (1, 2, 3) for idx in range(max(len(text) - width + 1, 0))]
    return grams or [text]


def fit_categorical_maps(
    frame: Any,
    *,
    ablation: AblationSpec | None = None,
    min_count: int = 1,
) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for column in active_categorical_columns(ablation or ABLATION_SPECS["full"]):
        if column not in frame.columns:
            continue
        counts = Counter(_category_value(value) for value in frame[column])
        mapping = {
            MISSING_CATEGORY_TOKEN: 1,
            UNKNOWN_CATEGORY_TOKEN: 2,
            RARE_CATEGORY_TOKEN: 3,
        }
        values = sorted(
            value
            for value, count in counts.items()
            if value != MISSING_CATEGORY_TOKEN and count >= max(int(min_count), 1)
        )
        mapping.update({value: idx + 4 for idx, value in enumerate(values)})
        rare_values = sorted(
            value
            for value, count in counts.items()
            if value != MISSING_CATEGORY_TOKEN and count < max(int(min_count), 1)
        )
        mapping.update({value: mapping[RARE_CATEGORY_TOKEN] for value in rare_values})
        maps[column] = mapping
    return maps


def fit_adapter_map(frame: Any, *, ablation: AblationSpec | None = None) -> dict[str, int]:
    spec = ablation or ABLATION_SPECS["full"]
    if not spec.use_medium_adapter:
        return {}
    values = sorted({adapter_name(row) for _, row in frame.iterrows()})
    return {value: idx + 1 for idx, value in enumerate(values)}


def raw_numeric_matrix(
    frame: Any,
    encoder: MolecularFeatureBuilder,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    ablation: AblationSpec | None = None,
) -> np.ndarray:
    spec = ablation or ABLATION_SPECS["full"]
    names = tuple(descriptor_names or encoder.descriptor_names())
    molecular_rows = [
        masked_descriptors(encoder.encode(value)[0], spec, descriptor_names=names)
        for value in frame.get("smiles", [])
    ]
    context_rows = [_context_numeric(row, spec) for _, row in frame.iterrows()]
    matrix = np.array([mol + ctx for mol, ctx in zip(molecular_rows, context_rows)], dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((1, len(MOLECULAR_DESCRIPTOR_NAMES) + len(CONTEXT_NUMERIC_COLUMNS)))
    return bounded_numeric_matrix(matrix)


def fit_numeric_stats(
    frame: Any,
    encoder: MolecularFeatureBuilder,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    ablation: AblationSpec | None = None,
) -> dict[str, tuple[float, float]]:
    matrix = raw_numeric_matrix(frame, encoder, descriptor_names=descriptor_names, ablation=ablation)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stats: dict[str, tuple[float, float]] = {}
    for idx, (mean, std) in enumerate(zip(means, stds)):
        mean_value = float(mean) if math.isfinite(float(mean)) else 0.0
        std_value = float(std) if math.isfinite(float(std)) and float(std) > 1e-12 else 1.0
        stats[str(idx)] = (mean_value, std_value)
    return stats


def fit_zscore_correction(
    frame: Any,
    encoder: MolecularFeatureBuilder,
    *,
    numeric_stats: dict[str, tuple[float, float]],
    feature_names: tuple[str, ...],
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    config: ZScoreCorrectionConfig,
    ablation: AblationSpec | None = None,
) -> ZScoreCorrection:
    split_parts = tuple(sorted({_category_value(value) for value in frame.get("split_part", [])}))
    threshold = max(float(config.threshold), 1.0)
    if not config.enabled:
        return ZScoreCorrection(
            enabled=False,
            threshold=threshold,
            feature_names=feature_names,
            fit_split_parts=split_parts,
            stats={},
        )
    matrix = raw_numeric_matrix(frame, encoder, descriptor_names=descriptor_names, ablation=ablation)
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(feature_names[: matrix.shape[1]]):
        mean, std = numeric_stats[str(idx)]
        std = std if std > 1e-12 else 1.0
        values = np.nan_to_num(matrix[:, idx].astype(float), nan=float(mean))
        z_values = (values - mean) / std
        clipped = np.abs(z_values) > threshold
        stats[name] = {
            "index": float(idx),
            "mean": float(mean),
            "std": float(std),
            "lower_raw": float(mean - threshold * std),
            "upper_raw": float(mean + threshold * std),
            "clipped_fit_count": float(np.count_nonzero(clipped)),
            "clipped_fit_fraction": float(np.count_nonzero(clipped) / max(len(values), 1)),
        }
    return ZScoreCorrection(
        enabled=True,
        threshold=threshold,
        feature_names=feature_names,
        fit_split_parts=split_parts,
        stats=stats,
    )


def build_deep_samples(
    frame: Any,
    *,
    encoder: MolecularFeatureBuilder,
    categorical_maps: dict[str, dict[str, int]],
    numeric_stats: dict[str, tuple[float, float]],
    target_column: str,
    target_scaler: TargetScaler | None = None,
    zscore_correction: ZScoreCorrection | None = None,
    adapter_map: dict[str, int] | None = None,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    ablation: AblationSpec | None = None,
    graph_encoder: MolecularGraphFeatureBuilder | None = None,
    toxicity_binning_config: ToxicityBinningConfig | None = None,
    toxicity_bin_scheme: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = ablation or ABLATION_SPECS["full"]
    descriptor_names = tuple(descriptor_names or encoder.descriptor_names())
    samples: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        descriptors, fingerprint = encoder.encode(row.get("smiles"))
        numeric = masked_descriptors(descriptors, spec, descriptor_names=descriptor_names) + _context_numeric(row, spec)
        fingerprint = masked_fingerprint(fingerprint, spec)
        normalized = []
        for idx, value in enumerate(numeric):
            mean, std = numeric_stats[str(idx)]
            value = bounded_numeric_value(value)
            std = float(std) if math.isfinite(float(std)) and float(std) > 1e-12 else 1.0
            z_value = (value - float(mean)) / std
            if not math.isfinite(float(z_value)):
                z_value = 0.0
            normalized.append(zscore_correction.transform(idx, z_value) if zscore_correction is not None else z_value)
        categorical_ids = {
            column: encode_category_id(row.get(column), mapping)
            for column, mapping in categorical_maps.items()
        }
        row_adapter_name = adapter_name(row)
        adapter_id = (adapter_map or {}).get(row_adapter_name, 0) if spec.use_medium_adapter else 0
        scale_key = target_scale_key(row, "none" if target_scaler is None else target_scaler.mode)
        raw_target = float(row[target_column])
        scaled_target = raw_target if target_scaler is None else target_scaler.transform(scale_key, raw_target)
        metadata = sample_metadata(row, target_column=target_column, scale_key=scale_key)
        toxicity_fields: dict[str, Any] = {}
        if toxicity_binning_config is not None and toxicity_bin_scheme is not None:
            descriptor_mol_weight = descriptor_value_by_name(
                descriptors,
                descriptor_names,
                ("MolWt", "MolecularWeight", "Molecular_Weight", "MW", "MWt"),
            )
            if getattr(encoder, "source", "") == "stable_smiles_fallback":
                descriptor_mol_weight = None
            toxicity_fields = assign_toxicity_bin(
                row,
                toxicity_bin_scheme,
                config=toxicity_binning_config,
                descriptor_mol_weight=descriptor_mol_weight,
            ).as_sample_fields()
        sample = {
            "sample_id": row.get("aggregate_id"),
            "molecular_numeric": normalized,
            "fingerprint": fingerprint,
            "categorical_ids": categorical_ids,
            "adapter_name": row_adapter_name,
            "adapter_id": adapter_id,
            "task_head": str(row.get("task_head")),
            "target_value": float(scaled_target),
            "target_value_raw": raw_target,
            "target_value_scaled": float(scaled_target),
            "split_part": str(row.get("split_part")),
            **metadata,
            **toxicity_fields,
        }
        if spec.use_molecular_graph:
            if graph_encoder is None:
                raise ValueError("Molecular graph ablation requires a graph encoder.")
            sample["molecular_graph"] = graph_encoder.encode(row.get("smiles"))
        samples.append(sample)
    return samples


def build_censored_training_samples(
    db_path: str | Path,
    reference_frame: Any,
    *,
    encoder: MolecularFeatureBuilder,
    categorical_maps: dict[str, dict[str, int]],
    adapter_map: dict[str, int] | None,
    numeric_stats: dict[str, tuple[float, float]],
    target_column: str,
    target_scaler: TargetScaler | None,
    zscore_correction: ZScoreCorrection | None,
    ablation: AblationSpec,
    config: CensoredLossConfig,
    kept_task_heads: tuple[str, ...],
    split_parts: tuple[str, ...],
    head_routing_mode: str = "task",
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    graph_encoder: MolecularGraphFeatureBuilder | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    split_part_set = {part.lower() for part in config.split_parts}
    kept_tasks = {str(task) for task in kept_task_heads}
    reference_lookup = censored_reference_split_lookup(reference_frame)
    fallback_aquatic_train = any(
        str(row.get("medium_domain", "")).lower() == "aquatic"
        and str(row.get("split_part", "")).lower() == "train"
        for _, row in reference_frame.iterrows()
    )
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    candidates = 0
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for raw in conn.execute(
            """
            SELECT *
            FROM target_records
            WHERE value_quality LIKE 'censored%'
              AND target_status = 'excluded'
            """
        ):
            candidates += 1
            row = dict(raw)
            operator = first_censor_operator(row)
            if operator not in config.include_ops:
                skipped["unsupported_operator"] += 1
                continue
            direction = censored_direction(operator)
            direction_id = 1 if direction == "right" else -1 if direction == "left" else 0
            if direction_id == 0:
                skipped["missing_direction"] += 1
                continue
            task = map_censored_task(row)
            if not task:
                skipped["unmapped_task"] += 1
                continue
            task = route_censored_task_head(task, row, mode=head_routing_mode)
            if task["task_head"] not in kept_tasks:
                skipped["task_not_trained"] += 1
                continue
            bound = censored_bound_value(row)
            if bound is None:
                skipped["missing_or_unsupported_bound"] += 1
                continue
            split_part = resolve_censored_split_part(
                row,
                task_head=task["task_head"],
                reference_lookup=reference_lookup,
                fallback_aquatic_train=fallback_aquatic_train,
            )
            if not split_part:
                skipped["unmatched_split"] += 1
                continue
            if split_part not in split_part_set:
                skipped["split_not_trainable"] += 1
                continue
            payload = dict(row)
            payload.update(task)
            payload["aggregate_id"] = f"censored:{payload.get('record_id', len(rows))}"
            payload["split_part"] = split_part
            payload["target_value"] = bound
            payload["target_value_median"] = bound
            payload["censored_direction"] = direction
            payload["censored_direction_id"] = direction_id
            payload["censored_operator"] = operator
            payload["censored_bound_raw"] = bound
            payload["censored_value_quality"] = payload.get("value_quality", "")
            rows.append(payload)
    if not rows:
        return [], {
            **config.to_manifest(),
            "candidate_rows": candidates,
            "usable_rows": 0,
            "train_rows": 0,
            "finetune_rows": 0,
            "skipped_rows": dict(sorted(skipped.items())),
        }
    censored_frame = add_duration_nonlinear_features(pd.DataFrame(rows))
    samples = build_deep_samples(
        censored_frame,
        encoder=encoder,
        descriptor_names=descriptor_names,
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        target_column=target_column,
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        ablation=ablation,
        graph_encoder=graph_encoder,
    )
    for sample, row in zip(samples, rows):
        sample["censored_direction"] = row["censored_direction"]
        sample["censored_direction_id"] = row["censored_direction_id"]
        sample["censored_operator"] = row["censored_operator"]
        sample["censored_bound_raw"] = row["censored_bound_raw"]
        sample["censored_value_quality"] = row["censored_value_quality"]
    split_counts = Counter(str(sample.get("split_part", "")).lower() for sample in samples)
    return samples, {
        **config.to_manifest(),
        "candidate_rows": candidates,
        "usable_rows": len(samples),
        "train_rows": int(split_counts.get("train", 0)),
        "finetune_rows": int(split_counts.get("finetune", 0)),
        "skipped_rows": dict(sorted(skipped.items())),
    }


def censored_reference_split_lookup(frame: Any) -> dict[tuple[str, str, str, str, str, str], set[str]]:
    lookup: dict[tuple[str, str, str, str, str, str], set[str]] = {}
    for _, row in frame.iterrows():
        for identifier in (row.get("cas_number"), row.get("dtxsid")):
            key = censored_match_key(
                identifier,
                row.get("species_number"),
                row.get("task_head"),
                row.get("target_name"),
                row.get("target_basis"),
                row.get("medium_domain"),
            )
            if key[0]:
                lookup.setdefault(key, set()).add(str(row.get("split_part", "")).lower())
    return lookup


def resolve_censored_split_part(
    row: Mapping[str, Any],
    *,
    task_head: str,
    reference_lookup: dict[tuple[str, str, str, str, str, str], set[str]],
    fallback_aquatic_train: bool,
) -> str:
    parts: set[str] = set()
    for identifier in (row.get("cas_number"), row.get("dtxsid")):
        key = censored_match_key(
            identifier,
            row.get("species_number"),
            task_head,
            row.get("target_name"),
            row.get("target_basis"),
            row.get("medium_domain"),
        )
        parts.update(reference_lookup.get(key, set()))
    if len(parts) == 1:
        return next(iter(parts))
    if not parts and fallback_aquatic_train and str(row.get("medium_domain", "")).lower() == "aquatic":
        return "train"
    return ""


def censored_match_key(
    identifier: Any,
    species_number: Any,
    task_head: Any,
    target_name: Any,
    target_basis: Any,
    medium_domain: Any,
) -> tuple[str, str, str, str, str, str]:
    return (
        clean_match_value(identifier),
        clean_match_value(species_number),
        clean_match_value(task_head),
        clean_match_value(target_name),
        clean_match_value(target_basis),
        clean_match_value(medium_domain),
    )


def clean_match_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def first_censor_operator(row: Mapping[str, Any]) -> str:
    for column in ("conc1_mean_op", "conc1_min_op", "conc1_max_op"):
        value = str(row.get(column, "") or "").strip()
        if value in {"<", "<=", ">", ">="}:
            return value
    return ""


def map_censored_task(row: Mapping[str, Any]) -> dict[str, str]:
    from qsar_tl.data.task_mapping import map_task_head

    mapping = map_task_head(
        endpoint=row.get("endpoint"),
        effect=row.get("effect"),
        measurement=row.get("measurement"),
        target_name=row.get("target_name"),
        target_basis=row.get("target_basis"),
    )
    if not mapping.task_head:
        return {}
    return {
        "task_head": str(mapping.task_head),
        "task_family": str(mapping.task_family or ""),
        "effect_family": str(mapping.effect_family or ""),
    }


def route_censored_task_head(
    task: Mapping[str, str],
    row: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, str]:
    routed = dict(task)
    base_task_head = str(routed.get("task_head", ""))
    routed["base_task_head"] = base_task_head
    if mode == "task":
        routed["model_head"] = base_task_head
        return routed
    if mode != "task_target":
        raise ValueError(f"Unsupported normalized head routing mode: {mode}")
    target_family = str(row.get("target_family", "") or "").strip()
    if not target_family:
        target_name = str(row.get("target_name", "") or "").strip()
        target_family = {
            "ptox_mol_l": "aquatic_pTox_mol_L",
            "neg_log10_mg_kg": "solid_neglog_mg_kg",
        }.get(target_name, target_name)
    model_head = f"{base_task_head}__{_category_value(target_family)}"
    routed["task_head"] = model_head
    routed["model_head"] = model_head
    return routed


def censored_bound_value(row: Mapping[str, Any]) -> float | None:
    target_name = str(row.get("target_name", "") or "").strip()
    candidates = []
    if target_name == "ptox_mol_l":
        candidates = ("standard_value_mol_l",)
    elif target_name == "neg_log10_mg_kg":
        candidates = ("standard_value_mg_kg",)
    elif target_name == "neg_log10_g_ha":
        candidates = ("standard_value_g_ha",)
    elif target_name == "neg_log10_mg_kg_diet":
        candidates = ("standard_value_mg_kg_diet",)
    elif target_name == "neg_log10_mg_kg_bw_day":
        candidates = ("standard_value_mg_kg_bw_day",)
    else:
        unit_family = str(row.get("unit_family_v2", "") or "").strip()
        if unit_family == "water_mol_l":
            candidates = ("standard_value_mol_l",)
        elif unit_family == "soil_mg_kg":
            candidates = ("standard_value_mg_kg",)
    for column in candidates:
        value = optional_float(row.get(column))
        if value is not None and value > 0:
            return -math.log10(value)
    return None


def active_categorical_columns(ablation: AblationSpec) -> tuple[str, ...]:
    columns: list[str] = []
    allowed_species_columns = (
        None
        if ablation.species_lifestage_columns is None
        else set(ablation.species_lifestage_columns)
    )
    for column in CATEGORICAL_COLUMNS:
        if column in SPECIES_LIFESTAGE_COLUMNS and not ablation.use_species_lifestage:
            continue
        if (
            column in SPECIES_LIFESTAGE_COLUMNS
            and allowed_species_columns is not None
            and column not in allowed_species_columns
        ):
            continue
        if column not in SPECIES_LIFESTAGE_COLUMNS and not ablation.use_other_categorical_context:
            continue
        columns.append(column)
    return tuple(columns)


def fit_target_scaler(
    frame: Any,
    *,
    target_column: str,
    mode: str = "per_task_target",
    fit_indices: list[int] | tuple[int, ...] | None = None,
) -> TargetScaler:
    normalized_mode = (mode or "per_task_target").strip().lower()
    allowed_modes = {
        "none",
        "identity",
        "global",
        "per_task",
        "per_target",
        "per_task_target",
        "per_adapter",
        "per_task_adapter",
    }
    if normalized_mode not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValueError(f"Unsupported target_standardization '{mode}'. Allowed values: {allowed}")
    if fit_indices is None:
        fit_frame = frame[frame["split_part"].astype("string").str.lower() == "train"].copy()
    else:
        fit_frame = frame.iloc[list(fit_indices)].copy()
    if fit_frame.empty:
        raise ValueError("Cannot fit target scaler without training rows.")
    split_parts = tuple(sorted({_category_value(value) for value in fit_frame.get("split_part", [])}))
    if normalized_mode in {"none", "identity"}:
        return TargetScaler(
            mode=normalized_mode,
            target_column=target_column,
            fit_split_parts=split_parts,
            stats={GLOBAL_TARGET_SCALE_KEY: _target_stats([0.0])},
        )
    stats: dict[str, dict[str, float]] = {
        GLOBAL_TARGET_SCALE_KEY: _target_stats([safe_number(value) for value in fit_frame[target_column]])
    }
    if normalized_mode == "global":
        return TargetScaler(
            mode=normalized_mode,
            target_column=target_column,
            fit_split_parts=split_parts,
            stats=stats,
        )
    grouped_values: dict[str, list[float]] = {}
    for _, row in fit_frame.iterrows():
        key = target_scale_key(row, normalized_mode)
        grouped_values.setdefault(key, []).append(safe_number(row[target_column]))
    for key, values in grouped_values.items():
        stats[key] = _target_stats(values)
    return TargetScaler(
        mode=normalized_mode,
        target_column=target_column,
        fit_split_parts=split_parts,
        stats=stats,
    )


def target_scale_key(row: Mapping[str, Any], mode: str) -> str:
    normalized_mode = (mode or "per_task_target").strip().lower()
    task = _category_value(row.get("task_head"))
    target = _target_dimension_value(row)
    if normalized_mode in {"none", "identity", "global"}:
        return GLOBAL_TARGET_SCALE_KEY
    if normalized_mode == "per_task":
        return task
    if normalized_mode == "per_target":
        return target
    if normalized_mode == "per_adapter":
        return adapter_name(row)
    if normalized_mode == "per_task_adapter":
        return f"{task}|{adapter_name(row)}"
    return f"{task}|{target}"


def _target_stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        finite = [0.0]
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "count": float(len(finite)),
        "mean": mean,
        "std": std if std > 1e-12 else 1.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def build_mgkg_hierarchical_head_spec(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    enabled: bool,
    family_tau: float,
    task_tau: float,
) -> dict[str, Any]:
    """Build a train-only hierarchy for native soil mol/kg task heads."""

    empty = {
        "enabled": False,
        "kind": "shared_plus_task_family_plus_exact_task_residual",
        "target_family": "solid_neglog_mol_kg",
        "heads": [],
        "head_families": {},
        "family_labels": {},
        "family_counts": {},
        "task_counts": {},
        "family_scales": {},
        "task_scales": {},
        "family_tau": float(family_tau),
        "task_tau": float(task_tau),
        "count_source": "finetune_mgkg_training_only",
        "train_identity_sha256": sample_indices_sha256(samples, train_indices),
        "non_train_rows_used": 0,
        "residual_contribution_scale_formula": "alpha=n/(n+tau)",
        "scale_interpretation": "effective_initial_regularization_not_a_hard_constraint",
        "shared_head_zero_initialized": False,
        "family_residual_zero_initialized": True,
        "exact_task_residual_zero_initialized": True,
    }
    if not enabled:
        return empty

    eligible = [
        samples[index]
        for index in train_indices
        if str(samples[index].get("target_family", "")).strip()
        == "solid_neglog_mol_kg"
    ]
    if not eligible:
        raise ValueError(
            "Hierarchical stage-3 head requires solid_neglog_mol_kg training rows."
        )
    task_counts = Counter(str(sample.get("task_head", "")).strip() for sample in eligible)
    task_counts.pop("", None)
    if not task_counts:
        raise ValueError("Hierarchical stage-3 head found no routed task heads.")

    task_family_labels: dict[str, str] = {}
    for sample in eligible:
        task_head = str(sample.get("task_head", "")).strip()
        if not task_head:
            continue
        family_label = str(sample.get("task_family", "")).strip()
        if not family_label or family_label == MISSING_CATEGORY_TOKEN:
            base_head = str(sample.get("base_task_head", task_head)).strip()
            family_label = base_head.split("_", 1)[0] or "other"
        previous = task_family_labels.setdefault(task_head, family_label)
        if previous != family_label:
            raise ValueError(
                f"Hierarchical task head maps to multiple families: {task_head}"
            )

    family_keys = {
        label: f"family_{position}"
        for position, label in enumerate(sorted(set(task_family_labels.values())))
    }
    head_families = {
        task_head: family_keys[label]
        for task_head, label in sorted(task_family_labels.items())
    }
    family_counts: Counter[str] = Counter()
    for task_head, count in task_counts.items():
        family_counts[head_families[task_head]] += int(count)

    def residual_contribution_scale(count: int, tau: float) -> float:
        return 1.0 if tau <= 0 else float(count) / (float(count) + float(tau))

    return {
        "enabled": True,
        "kind": "shared_plus_task_family_plus_exact_task_residual",
        "target_family": "solid_neglog_mol_kg",
        "heads": sorted(task_counts),
        "head_families": head_families,
        "family_labels": {key: label for label, key in family_keys.items()},
        "family_counts": dict(sorted(family_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "family_scales": {
            family: residual_contribution_scale(count, family_tau)
            for family, count in sorted(family_counts.items())
        },
        "task_scales": {
            task_head: residual_contribution_scale(count, task_tau)
            for task_head, count in sorted(task_counts.items())
        },
        "family_tau": float(family_tau),
        "task_tau": float(task_tau),
        "count_source": "finetune_mgkg_training_only",
        "train_identity_sha256": sample_indices_sha256(samples, train_indices),
        "non_train_rows_used": 0,
        "residual_contribution_scale_formula": "alpha=n/(n+tau)",
        "scale_interpretation": "effective_initial_regularization_not_a_hard_constraint",
        "shared_head_zero_initialized": False,
        "family_residual_zero_initialized": True,
        "exact_task_residual_zero_initialized": True,
    }


def build_task_equal_width_target_bin_sampling_spec(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    enabled: bool,
    bins: int,
    min_weight: float,
    max_weight: float,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Build bounded, train-only inverse-stratum weights without changing task mass.

    Equal-width raw-target bins are fitted independently inside each routed task
    using only stage-3 training rows. After inverse-bin-frequency weighting, a
    bounded scalar projection restores each task's expected sampling mass to its
    original row count.
    """

    audit: dict[str, Any] = {
        "enabled": bool(enabled),
        "kind": "within_task_equal_width_target_bin_inverse_frequency",
        "sampler": "torch.utils.data.WeightedRandomSampler",
        "replacement": True,
        "count_source": "finetune_mgkg_training_only",
        "target_source": "target_value_raw",
        "requested_bins": int(bins),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "preserve_task_expected_mass": True,
        "train_rows": int(len(train_indices)),
        "train_identity_sha256": sample_indices_sha256(samples, train_indices),
        "non_train_rows_used": 0,
        "tasks": {},
    }
    if not enabled:
        return {}, audit
    if not train_indices:
        raise ValueError("Equal-width target-bin sampling requires stage-3 training rows.")

    by_task: dict[str, list[int]] = {}
    for index in train_indices:
        task_head = str(samples[index].get("task_head", "")).strip()
        if not task_head:
            raise ValueError("Target-bin sampling found a stage-3 row without task_head.")
        target = optional_float(samples[index].get("target_value_raw"))
        if target is None:
            raise ValueError("Target-bin sampling found a non-finite stage-3 raw target.")
        by_task.setdefault(task_head, []).append(index)

    weights: dict[int, float] = {}
    for task_head, indices in sorted(by_task.items()):
        values = np.asarray(
            [float(samples[index]["target_value_raw"]) for index in indices],
            dtype=float,
        )
        target_min = float(values.min())
        target_max = float(values.max())
        if target_max <= target_min:
            bin_ids = np.zeros(len(indices), dtype=int)
            bin_width = 0.0
        else:
            edges = np.linspace(target_min, target_max, int(bins) + 1)
            bin_width = float((target_max - target_min) / int(bins))
            # Values at the maximum remain in the last bin; every other
            # boundary value is assigned deterministically to the upper bin.
            bin_ids = np.searchsorted(edges[1:-1], values, side="right")
        bin_counts = Counter(int(value) for value in bin_ids.tolist())
        effective_bins = max(len(bin_counts), 1)
        raw = np.asarray(
            [len(indices) / (effective_bins * bin_counts[int(bin_id)]) for bin_id in bin_ids],
            dtype=float,
        )
        bounded = _project_sampling_weights_to_task_mass(
            raw,
            target_mass=float(len(indices)),
            min_weight=float(min_weight),
            max_weight=float(max_weight),
        )
        for index, value in zip(indices, bounded.tolist()):
            weights[index] = float(value)
        expected_mass = float(bounded.sum())
        audit["tasks"][task_head] = {
            "rows": int(len(indices)),
            "occupied_bins": int(effective_bins),
            "target_min": target_min,
            "target_max": target_max,
            "bin_width": bin_width,
            "bin_counts": {
                str(key): int(value) for key, value in sorted(bin_counts.items())
            },
            "weight_min": float(bounded.min()),
            "weight_max": float(bounded.max()),
            "weight_mean": float(bounded.mean()),
            "expected_sampling_mass": expected_mass,
            "expected_mass_error": expected_mass - float(len(indices)),
        }
    audit["task_count"] = int(len(by_task))
    audit["weight_min_realized"] = float(min(weights.values()))
    audit["weight_max_realized"] = float(max(weights.values()))
    audit["expected_total_sampling_mass"] = float(sum(weights.values()))
    return weights, audit


def _project_sampling_weights_to_task_mass(
    raw_weights: np.ndarray,
    *,
    target_mass: float,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Scale then clip positive weights while satisfying a feasible sum."""

    raw = np.asarray(raw_weights, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Sampling weights must be a finite, positive vector.")
    feasible_min = float(raw.size) * float(min_weight)
    feasible_max = float(raw.size) * float(max_weight)
    if target_mass < feasible_min - 1e-9 or target_mass > feasible_max + 1e-9:
        raise ValueError("Requested task sampling mass is outside the clipping bounds.")
    low = 0.0
    high = max(1.0, float(target_mass / max(raw.sum(), 1e-12)))
    while float(np.clip(high * raw, min_weight, max_weight).sum()) < target_mass:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        mass = float(np.clip(middle * raw, min_weight, max_weight).sum())
        if mass < target_mass:
            low = middle
        else:
            high = middle
    projected = np.clip(((low + high) / 2.0) * raw, min_weight, max_weight)
    if abs(float(projected.sum()) - float(target_mass)) > 1e-7:
        raise RuntimeError("Could not preserve task expected sampling mass after clipping.")
    return projected


def sampling_weights_for_stage_dataset(
    stage_dataset: Any,
    *,
    source_weights: Mapping[int, float],
) -> list[float]:
    """Expand source-row weights over optional deterministic augmentation replicas."""

    expanded: list[float] = []
    for position in range(len(stage_dataset)):
        if isinstance(stage_dataset, _NoisyIndexDataset):
            source_index, _ = stage_dataset.source(position)
        elif isinstance(stage_dataset, _IndexDataset):
            source_index = stage_dataset.indices[position]
        else:  # pragma: no cover - stage construction owns the supported types.
            raise TypeError(
                "Target-bin sampling supports only _IndexDataset or _NoisyIndexDataset."
            )
        if source_index not in source_weights:
            raise ValueError(
                f"Target-bin sampling is missing a train-only source weight: {source_index}"
            )
        expanded.append(float(source_weights[source_index]))
    return expanded


def resolve_task_weights(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    train_cfg: Mapping[str, Any],
    task_heads: tuple[str, ...],
) -> dict[str, float]:
    raw_manual = train_cfg.get("task_weights", {}) or {}
    manual = {str(key): float(value) for key, value in raw_manual.items()}
    mode = str(train_cfg.get("task_weighting", "balanced")).strip().lower()
    group_by_task: dict[str, str] = {}
    for idx in train_indices:
        task_head = str(samples[idx].get("task_head", ""))
        task_group = str(samples[idx].get("task_group", "") or "")
        if task_head and task_head not in group_by_task:
            group_by_task[task_head] = task_group
    main_weight = float(train_cfg.get("main_task_weight", 1.0))
    toxicity_aux_weight = float(train_cfg.get("toxicity_aux_task_weight", 0.35))
    bioaccum_aux_weight = float(train_cfg.get("bioaccumulation_aux_task_weight", 0.2))
    weights = {}
    for task_head in task_heads:
        task_group = group_by_task.get(task_head, "")
        if task_group == "toxicity_aux":
            base_weight = toxicity_aux_weight
        elif task_group == "bioaccumulation_aux":
            base_weight = bioaccum_aux_weight
        else:
            base_weight = main_weight
        weights[task_head] = base_weight * float(manual.get(task_head, 1.0))
    if mode in {"none", "manual"}:
        return weights
    if mode != "balanced":
        raise ValueError("training.task_weighting must be 'balanced', 'manual', or 'none'.")
    counts = Counter(str(samples[idx]["task_head"]) for idx in train_indices)
    if not counts:
        return weights
    reference = float(np.median([count for count in counts.values()]))
    exponent = float(train_cfg.get("task_weight_exponent", 0.5))
    min_weight = float(train_cfg.get("task_weight_min", 0.25))
    max_weight = float(train_cfg.get("task_weight_max", 4.0))
    for task_head in task_heads:
        count = max(int(counts.get(task_head, 0)), 1)
        balanced = (reference / count) ** exponent
        weights[task_head] = max(min_weight, min(max_weight, balanced * weights[task_head]))
    return weights


def encode_category_id(value: Any, mapping: Mapping[str, int]) -> int:
    token = _category_value(value)
    if token in mapping:
        return int(mapping[token])
    return int(mapping.get(UNKNOWN_CATEGORY_TOKEN, 0))


def sample_metadata(row: Mapping[str, Any], *, target_column: str, scale_key: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column in PREDICTION_METADATA_COLUMNS:
        if column == "sample_id":
            metadata[column] = clean_metadata_value(row.get("aggregate_id", row.get("sample_id", "")))
        elif column == "target_column":
            metadata[column] = target_column
        elif column == "target_scale_key":
            metadata[column] = scale_key
        elif column in row:
            metadata[column] = clean_metadata_value(row.get(column))
    return metadata


def clean_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return value


def safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def adapter_name(row: Mapping[str, Any]) -> str:
    medium = _category_value(
        row.get("medium_domain")
        if row.get("medium_domain") is not None
        else row.get("primary_medium")
    )
    target = _target_dimension_value(row)
    return f"{medium}|{target}"


def _target_dimension_value(row: Mapping[str, Any]) -> str:
    value = row.get("target_family")
    if value is None or str(value).strip() == "":
        value = row.get("target_name")
    return _category_value(value)


def masked_descriptors(
    descriptors: list[float],
    ablation: AblationSpec,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
) -> list[float]:
    if not ablation.use_descriptors:
        return [0.0] * len(descriptors)
    if not ablation.masked_descriptor_names:
        return descriptors
    masked = list(descriptors)
    for idx in descriptor_mask_indices(len(masked), ablation, descriptor_names=descriptor_names):
        masked[idx] = 0.0
    return masked


def descriptor_mask_indices(
    descriptor_count: int,
    ablation: AblationSpec,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[int, ...]:
    if not ablation.masked_descriptor_names:
        return ()
    names = tuple(descriptor_names or _descriptor_feature_names(descriptor_count))
    if len(names) != descriptor_count:
        names = tuple(_descriptor_feature_names(descriptor_count))
    masked_names = set(ablation.masked_descriptor_names)
    return tuple(idx for idx, name in enumerate(names) if name in masked_names)


def descriptor_index_by_name(
    descriptor_names: tuple[str, ...] | list[str],
    aliases: tuple[str, ...] | list[str],
) -> int | None:
    normalized = {normalize_descriptor_name(name): idx for idx, name in enumerate(descriptor_names)}
    for alias in aliases:
        idx = normalized.get(normalize_descriptor_name(alias))
        if idx is not None:
            return idx
    return None


def descriptor_value_by_name(
    descriptors: list[float],
    descriptor_names: tuple[str, ...] | list[str],
    aliases: tuple[str, ...] | list[str],
) -> float | None:
    idx = descriptor_index_by_name(descriptor_names, aliases)
    if idx is None or idx < 0 or idx >= len(descriptors):
        return None
    try:
        value = float(descriptors[idx])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def normalize_descriptor_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def masked_fingerprint(fingerprint: list[float], ablation: AblationSpec) -> list[float]:
    if ablation.use_fingerprint:
        return fingerprint
    return [0.0] * len(fingerprint)


def _context_numeric(row: Mapping[str, Any], ablation: AblationSpec | None = None) -> list[float]:
    spec = ablation or ABLATION_SPECS["full"]
    effect_level = optional_float(row.get("effect_level_x"))
    has_effect_level = effect_level is not None
    effect_level_value = float(effect_level or 0.0)
    values = []
    for column in CONTEXT_NUMERIC_COLUMNS:
        if not spec.use_context_numeric:
            values.append(0.0)
            continue
        if column in EFFECT_LEVEL_NUMERIC_COLUMNS and not spec.use_effect_level_features:
            values.append(0.0)
            continue
        if column == "effect_level_x":
            values.append(effect_level_value)
            continue
        if column == "effect_level_x_fraction":
            values.append(effect_level_value / 100.0)
            continue
        if column == "effect_level_x_log1p":
            values.append(math.log1p(max(effect_level_value, 0.0)))
            continue
        if column == "effect_level_x_present":
            values.append(1.0 if has_effect_level else 0.0)
            continue
        if column in DURATION_CONTEXT_COLUMNS and not spec.use_duration_features:
            values.append(0.0)
            continue
        raw = row.get(column, 0.0)
        try:
            values.append(float(raw) if raw is not None and raw == raw else 0.0)
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _category_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def _concat_frames(frames: list[Any]) -> Any:
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


class _IndexDataset:
    def __init__(self, dataset: AggregatedTaskDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


class _StagePoolDataset:
    """Address target and replay pools through one index space for a batch sampler."""

    def __init__(self, target_dataset: Any, replay_dataset: Any) -> None:
        self.target_dataset = target_dataset
        self.replay_dataset = replay_dataset
        self.target_count = len(target_dataset)
        self.replay_pool_count = len(replay_dataset)

    def __len__(self) -> int:
        return self.target_count + self.replay_pool_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < self.target_count:
            source = "mgkg"
            source_index = index
            dataset = self.target_dataset
        else:
            source = "soil_ptox_replay"
            source_index = index - self.target_count
            dataset = self.replay_dataset
        sample = dict(dataset[source_index])
        sample["finetune_mgkg_source"] = source
        return sample


class _TargetReplayBatchSampler:
    """Yield batches with a fixed replay sample fraction and full target coverage."""

    def __init__(
        self,
        *,
        target_count: int,
        replay_count: int,
        batch_size: int,
        replay_fraction: float,
        seed: int,
    ) -> None:
        if target_count <= 0:
            raise ValueError("Replay batch sampler requires target samples.")
        if replay_count <= 0:
            raise ValueError("Replay batch sampler requires replay samples.")
        if batch_size < 2:
            raise ValueError("Replay batch sampler requires batch_size >= 2.")
        self.target_count = int(target_count)
        self.replay_count = int(replay_count)
        self.batch_size = int(batch_size)
        self.replay_fraction = float(replay_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self.replay_per_full_batch = min(
            max(int(round(self.batch_size * self.replay_fraction)), 1),
            self.batch_size - 1,
        )
        self.target_per_full_batch = self.batch_size - self.replay_per_full_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(math.ceil(self.target_count / self.target_per_full_batch))

    @property
    def replay_samples_per_epoch(self) -> int:
        total = 0
        for start in range(0, self.target_count, self.target_per_full_batch):
            target_batch_count = min(self.target_per_full_batch, self.target_count - start)
            total += max(1, int(round(target_batch_count * self.replay_fraction / (1.0 - self.replay_fraction))))
        return total

    def __iter__(self) -> Any:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        targets = rng.permutation(self.target_count).tolist()
        replay_cycle: list[int] = []
        replay_position = 0

        def take_replay(count: int) -> list[int]:
            nonlocal replay_cycle, replay_position
            selected: list[int] = []
            while len(selected) < count:
                if replay_position >= len(replay_cycle):
                    replay_cycle = rng.permutation(self.replay_count).tolist()
                    replay_position = 0
                take = min(count - len(selected), len(replay_cycle) - replay_position)
                selected.extend(replay_cycle[replay_position : replay_position + take])
                replay_position += take
            return [self.target_count + index for index in selected]

        for start in range(0, self.target_count, self.target_per_full_batch):
            target_batch = targets[start : start + self.target_per_full_batch]
            replay_batch_count = max(
                1,
                int(round(len(target_batch) * self.replay_fraction / (1.0 - self.replay_fraction))),
            )
            batch = target_batch + take_replay(replay_batch_count)
            rng.shuffle(batch)
            yield batch


class _NoisyIndexDataset:
    def __init__(
        self,
        dataset: AggregatedTaskDataset,
        indices: list[int],
        *,
        replicates: int,
        numeric_noise_std: float,
        target_noise_std: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.replicates = max(int(replicates), 1)
        self.numeric_noise_std = max(float(numeric_noise_std), 0.0)
        self.target_noise_std = max(float(target_noise_std), 0.0)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices) * self.replicates

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index, replicate_id = self.source(index)
        sample = dict(self.dataset[source_index])
        sample["categorical_ids"] = dict(sample.get("categorical_ids", {}))
        sample["molecular_numeric"] = list(sample.get("molecular_numeric", []))
        sample["fingerprint"] = list(sample.get("fingerprint", []))
        if replicate_id > 0:
            rng = np.random.default_rng(self.seed + source_index * 1009 + replicate_id * 9176)
            if self.numeric_noise_std > 0 and sample["molecular_numeric"]:
                noise = rng.normal(0.0, self.numeric_noise_std, size=len(sample["molecular_numeric"]))
                sample["molecular_numeric"] = [
                    float(value) + float(delta)
                    for value, delta in zip(sample["molecular_numeric"], noise)
                ]
            if self.target_noise_std > 0:
                sample["target_value"] = float(sample["target_value"]) + float(
                    rng.normal(0.0, self.target_noise_std)
                )
        sample["perturbation_replicate"] = replicate_id
        sample["perturbation_numeric_noise_std"] = self.numeric_noise_std
        return sample

    def source(self, index: int) -> tuple[int, int]:
        if not self.indices:
            raise IndexError("Cannot index an empty noisy dataset.")
        base_position = index // self.replicates
        replicate_id = index % self.replicates
        return self.indices[base_position], replicate_id


def build_noisy_index_dataset(
    dataset: AggregatedTaskDataset,
    indices: list[int],
    *,
    replicates: int,
    numeric_noise_std: float,
    target_noise_std: float,
    seed: int,
) -> Any:
    if max(int(replicates), 1) <= 1 or (float(numeric_noise_std) <= 0 and float(target_noise_std) <= 0):
        return _IndexDataset(dataset, indices)
    return _NoisyIndexDataset(
        dataset,
        indices,
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        target_noise_std=target_noise_std,
        seed=seed,
    )


def mark_internal_validation_samples(
    samples: list[dict[str, Any]],
    *,
    indices: list[int],
    split_part: str,
) -> None:
    for idx in indices:
        if idx < 0 or idx >= len(samples):
            continue
        sample = dict(samples[idx])
        sample.setdefault("original_split_part", sample.get("split_part", ""))
        sample["split_part"] = split_part
        samples[idx] = sample


def apply_source_similarity_weights(
    samples: list[dict[str, Any]],
    config: SourceWeightingConfig,
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    for sample in samples:
        sample["sample_weight"] = 1.0
    summary: dict[str, Any] = {
        **config.to_manifest(),
        "applied": False,
        "weighted_samples": 0,
        "target_reference_samples": 0,
        "cache_enabled": cache_dir is not None,
        "cache_hit": False,
        "cache_schema": SOURCE_WEIGHT_CACHE_SCHEMA,
        "cache_key": "",
        "cache_path": "",
    }
    if not config.active():
        return summary
    allowed_methods = {
        "tanimoto",
        "tanimoto_to_target",
        "tanimoto_to_finetune",
        "proxy_distance_to_finetune",
        "tanimoto_proxy_to_finetune",
    }
    if config.method not in allowed_methods:
        raise ValueError(
            "training.source_weighting.method must be 'none', 'tanimoto', "
            "'tanimoto_to_target', 'tanimoto_to_finetune', "
            "'proxy_distance_to_finetune', or 'tanimoto_proxy_to_finetune'."
        )
    source_indices = select_samples_by_domain_and_split(
        samples,
        split_parts=config.source_split_parts,
        domains=config.source_domains,
    )
    target_indices = select_samples_by_domain_and_split(
        samples,
        split_parts=config.target_split_parts,
        domains=config.target_domains,
    )
    summary["weighted_samples"] = len(source_indices)
    summary["target_reference_samples"] = len(target_indices)
    if not source_indices or not target_indices:
        return summary
    similarities: np.ndarray | None = None
    distances: np.ndarray | None = None
    raw_weights: np.ndarray | None = None
    source_fp: np.ndarray | None = None
    target_fp: np.ndarray | None = None
    source_proxy: np.ndarray | None = None
    target_proxy: np.ndarray | None = None
    if config.method in {"tanimoto", "tanimoto_to_target", "tanimoto_to_finetune", "tanimoto_proxy_to_finetune"}:
        source_fp = np.asarray([samples[idx].get("fingerprint", []) for idx in source_indices], dtype=np.float32)
        target_fp = np.asarray([samples[idx].get("fingerprint", []) for idx in target_indices], dtype=np.float32)
        if source_fp.ndim != 2 or target_fp.ndim != 2 or source_fp.shape[1] != target_fp.shape[1]:
            return summary
    if config.method in {"proxy_distance_to_finetune", "tanimoto_proxy_to_finetune"}:
        source_proxy = proxy_descriptor_matrix(samples, source_indices, descriptor_names=descriptor_names)
        target_proxy = proxy_descriptor_matrix(samples, target_indices, descriptor_names=descriptor_names)

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key = source_weight_cache_key(
            samples,
            source_indices=source_indices,
            target_indices=target_indices,
            config=config,
            descriptor_names=descriptor_names,
            source_fp=source_fp,
            target_fp=target_fp,
            source_proxy=source_proxy,
            target_proxy=target_proxy,
        )
        cache_path = Path(cache_dir) / f"{cache_key}.npz"
        summary.update({"cache_key": cache_key, "cache_path": str(cache_path)})
        if cache_path.exists():
            clipped, cached_summary = load_source_weight_cache(
                cache_path,
                expected_rows=len(source_indices),
                expected_key=cache_key,
            )
            for idx, weight in zip(source_indices, clipped):
                samples[idx]["sample_weight"] = float(weight)
            summary.update(cached_summary)
            summary.update(
                {
                    **config.to_manifest(),
                    "weighted_samples": len(source_indices),
                    "target_reference_samples": len(target_indices),
                    "cache_enabled": True,
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "cache_path": str(cache_path),
                }
            )
            return summary

    if source_fp is not None and target_fp is not None:
        similarities = max_tanimoto_similarity(source_fp, target_fp)
        raw_weights = 1.0 + float(config.alpha) * similarities
    if source_proxy is not None and target_proxy is not None:
        distances = min_proxy_distance(source_proxy, target_proxy)
        proxy_weights = np.exp(-float(config.alpha) * distances)
        raw_weights = proxy_weights if raw_weights is None else raw_weights * proxy_weights
    if raw_weights is None:
        return summary
    mean_weight = float(np.mean(raw_weights)) if raw_weights.size else 1.0
    if mean_weight > 0 and math.isfinite(mean_weight):
        raw_weights = raw_weights / mean_weight
    lower = min(float(config.min_weight), float(config.max_weight))
    upper = max(float(config.min_weight), float(config.max_weight))
    clipped = np.clip(raw_weights, lower, upper)
    for idx, weight in zip(source_indices, clipped):
        samples[idx]["sample_weight"] = float(weight)
    summary.update(
        {
            "applied": True,
            "weight_min": float(np.min(clipped)),
            "weight_mean": float(np.mean(clipped)),
            "weight_max": float(np.max(clipped)),
        }
    )
    if similarities is not None and similarities.size:
        summary.update(
            {
                "similarity_min": float(np.min(similarities)),
                "similarity_mean": float(np.mean(similarities)),
                "similarity_max": float(np.max(similarities)),
            }
        )
    if distances is not None and distances.size:
        summary.update(
            {
                "proxy_distance_min": float(np.min(distances)),
                "proxy_distance_mean": float(np.mean(distances)),
                "proxy_distance_max": float(np.max(distances)),
            }
        )
    if cache_path is not None:
        write_source_weight_cache(cache_path, clipped, summary)
    return summary


def source_weight_cache_key(
    samples: list[dict[str, Any]],
    *,
    source_indices: list[int],
    target_indices: list[int],
    config: SourceWeightingConfig,
    descriptor_names: tuple[str, ...] | list[str] | None,
    source_fp: np.ndarray | None,
    target_fp: np.ndarray | None,
    source_proxy: np.ndarray | None,
    target_proxy: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    contract = {
        "schema": SOURCE_WEIGHT_CACHE_SCHEMA,
        "config": config.to_manifest(),
        "descriptor_names": [str(value) for value in (descriptor_names or ())],
    }
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for label, indices in (("source", source_indices), ("target", target_indices)):
        digest.update(label.encode("ascii"))
        for idx in indices:
            sample = samples[idx]
            identity = [
                idx,
                sample.get("sample_id", ""),
                sample.get("aggregate_id", ""),
                sample.get("split_part", ""),
                sample.get("medium_domain", ""),
                sample.get("target_name", ""),
                sample.get("target_family", ""),
                sample.get("target_basis", ""),
                sample.get("target_column", ""),
                sample.get("target_scale_key", ""),
                sample.get("unit_family_v2", ""),
                sample.get("task_head", ""),
                sample.get("model_head", ""),
            ]
            digest.update(
                json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
    for label, matrix in (
        ("source_fp", source_fp),
        ("target_fp", target_fp),
        ("source_proxy", source_proxy),
        ("target_proxy", target_proxy),
    ):
        digest.update(label.encode("ascii"))
        if matrix is None:
            digest.update(b"none")
            continue
        contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_source_weight_cache(
    path: Path,
    *,
    expected_rows: int,
    expected_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            weights = np.asarray(payload["weights"], dtype=np.float32)
            summary = json.loads(str(payload["summary_json"].item()))
            cache_key = str(payload["cache_key"].item())
            schema_version = str(payload["schema_version"].item())
            weights_sha256 = str(payload["weights_sha256"].item())
            stored_row_count = int(payload["row_count"].item())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Source-weight cache is corrupt or incompatible: {path}") from exc
    if weights.ndim != 1 or int(weights.shape[0]) != int(expected_rows):
        raise ValueError(
            "Source-weight cache row count mismatch: "
            f"path={path}, expected={expected_rows}, observed={weights.shape}"
        )
    if not isinstance(summary, dict):
        raise ValueError(f"Source-weight cache summary must be a JSON object: {path}")
    observed_checksum = hashlib.sha256(np.ascontiguousarray(weights).tobytes(order="C")).hexdigest()
    if (
        cache_key != expected_key
        or schema_version != SOURCE_WEIGHT_CACHE_SCHEMA
        or stored_row_count != expected_rows
        or weights_sha256 != observed_checksum
    ):
        raise ValueError(
            "Source-weight cache contract mismatch: "
            f"path={path}, expected_key={expected_key}, stored_key={cache_key}, "
            f"schema={schema_version}, rows={stored_row_count}, checksum_ok={weights_sha256 == observed_checksum}"
        )
    return weights, summary


def write_source_weight_cache(path: Path, weights: np.ndarray, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    normalized_weights = np.asarray(weights, dtype=np.float32)
    weights_sha256 = hashlib.sha256(
        np.ascontiguousarray(normalized_weights).tobytes(order="C")
    ).hexdigest()
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                weights=normalized_weights,
                summary_json=np.asarray(json.dumps(dict(summary), sort_keys=True)),
                cache_key=np.asarray(path.stem),
                schema_version=np.asarray(SOURCE_WEIGHT_CACHE_SCHEMA),
                weights_sha256=np.asarray(weights_sha256),
                row_count=np.asarray(int(normalized_weights.shape[0]), dtype=np.int64),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def proxy_descriptor_matrix(
    samples: list[dict[str, Any]],
    indices: list[int],
    *,
    descriptor_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    proxy_indices = descriptor_proxy_indices(descriptor_names)
    rows: list[list[float]] = []
    for idx in indices:
        values = list(samples[idx].get("molecular_numeric", []))
        rows.append([safe_proxy_float(values[pos]) if pos < len(values) else 0.0 for pos in proxy_indices])
    return np.asarray(rows, dtype=np.float32)


def descriptor_proxy_indices(descriptor_names: tuple[str, ...] | list[str] | None = None) -> tuple[int, ...]:
    names = [str(name) for name in (descriptor_names or MOLECULAR_DESCRIPTOR_NAMES)]
    aliases = (
        ("MolWt", "MolecularWeight", "Molecular_Weight", "MW", "MWt"),
        ("TPSA", "TopoPSA", "TopologicalPolarSurfaceArea"),
        ("MolLogP", "ALogP", "XLogP", "MLogP", "LogP"),
    )
    indices: list[int] = []
    for group in aliases:
        idx = descriptor_index_by_name(names, group)
        if idx is not None:
            indices.append(idx)
    if len(indices) == len(aliases):
        return tuple(indices)
    return tuple(range(min(3, len(names))))


def safe_proxy_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def min_proxy_distance(source_proxy: np.ndarray, target_proxy: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
    if source_proxy.ndim != 2 or target_proxy.ndim != 2 or source_proxy.shape[1] != target_proxy.shape[1]:
        return np.full(source_proxy.shape[0] if source_proxy.ndim >= 1 else 0, np.inf, dtype=np.float32)
    if source_proxy.shape[0] == 0 or target_proxy.shape[0] == 0:
        return np.full(source_proxy.shape[0], np.inf, dtype=np.float32)
    pooled = np.vstack([source_proxy, target_proxy])
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    source_scaled = ((np.nan_to_num(source_proxy, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32)
    target_scaled = ((np.nan_to_num(target_proxy, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32)
    target_norm = np.sum(target_scaled * target_scaled, axis=1, dtype=np.float32)[None, :]
    out = np.empty(source_scaled.shape[0], dtype=np.float32)
    chunk = max(1, int(batch_size))
    for start in range(0, source_scaled.shape[0], chunk):
        batch = source_scaled[start : start + chunk]
        batch_norm = np.sum(batch * batch, axis=1, dtype=np.float32)[:, None]
        distances_sq = batch_norm + target_norm - 2.0 * (batch @ target_scaled.T)
        out[start : start + chunk] = np.sqrt(np.maximum(np.min(distances_sq, axis=1), 0.0))
    return out


def apply_effect_level_frequency_weights(
    samples: list[dict[str, Any]],
    config: EffectLevelWeightingConfig,
    *,
    train_indices: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **config.to_manifest(),
        "applied": False,
        "training_reference_samples": len(set(train_indices)),
        "weighted_samples": 0,
        "eligible_levels": 0,
        "level_counts": {},
    }
    if not config.active():
        return summary
    lower = min(float(config.min_weight), float(config.max_weight))
    upper = max(float(config.min_weight), float(config.max_weight))
    split_set = {str(value).strip().lower() for value in config.split_parts if str(value).strip()}
    prefix_set = tuple(str(value).strip().lower() for value in config.task_prefixes if str(value).strip())
    eligible: list[tuple[int, str]] = []
    seen_indices = sorted({int(idx) for idx in train_indices if 0 <= int(idx) < len(samples)})
    for idx in seen_indices:
        sample = samples[idx]
        split_part = str(sample.get("split_part", "")).strip().lower()
        if split_set and split_part not in split_set:
            continue
        task_head = str(sample.get("task_head", "")).strip().lower()
        if prefix_set and not task_head.startswith(prefix_set):
            continue
        effect_level = optional_float(sample.get("effect_level_x"))
        if effect_level is None:
            continue
        eligible.append((idx, effect_level_label(effect_level)))
    counts = Counter(level for _, level in eligible)
    summary["weighted_samples"] = len(eligible)
    summary["eligible_levels"] = len(counts)
    summary["level_counts"] = dict(sorted(counts.items()))
    if not eligible or not counts:
        return summary
    raw_weights = np.asarray([float(counts[level]) ** (-float(config.beta)) for _, level in eligible], dtype=np.float32)
    raw_mean = float(np.mean(raw_weights)) if raw_weights.size else 1.0
    normalized = raw_weights / raw_mean if raw_mean > 0 and math.isfinite(raw_mean) else raw_weights
    clipped = normalize_clipped_weights(normalized, lower, upper)
    final_weights: list[float] = []
    for (idx, _), effect_weight in zip(eligible, clipped):
        current = float(samples[idx].get("sample_weight", 1.0) or 1.0)
        final_weight = current * float(effect_weight)
        samples[idx]["sample_weight"] = final_weight
        final_weights.append(final_weight)
    final_array = np.asarray(final_weights, dtype=np.float32)
    summary.update(
        {
            "applied": True,
            "raw_weight_min": float(np.min(raw_weights)),
            "raw_weight_mean": raw_mean,
            "raw_weight_max": float(np.max(raw_weights)),
            "effect_weight_min": float(np.min(clipped)),
            "effect_weight_mean": float(np.mean(clipped)),
            "effect_weight_max": float(np.max(clipped)),
            "final_weight_min": float(np.min(final_array)),
            "final_weight_mean": float(np.mean(final_array)),
            "final_weight_max": float(np.max(final_array)),
        }
    )
    return summary


def normalize_clipped_weights(weights: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if values.size == 0:
        return values
    raw_lower = float(lower)
    raw_upper = float(upper)
    lower = min(raw_lower, raw_upper)
    upper = max(raw_lower, raw_upper)
    if lower > 1.0 or upper < 1.0:
        return np.clip(values, lower, upper)

    def clipped_mean(scale: float) -> float:
        return float(np.mean(np.clip(values * scale, lower, upper)))

    low = 0.0
    high = 1.0
    while clipped_mean(high) < 1.0 and high < 1_000_000.0:
        high *= 2.0
    for _ in range(48):
        mid = (low + high) / 2.0
        if clipped_mean(mid) < 1.0:
            low = mid
        else:
            high = mid
    return np.clip(values * high, lower, upper)


def sample_weighting_history_fields(
    source_weighting_summary: Mapping[str, Any],
    effect_level_weighting_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_weighting_applied": bool(source_weighting_summary.get("applied", False)),
        "source_weighting_alpha": source_weighting_summary.get("alpha", ""),
        "source_weighted_samples": source_weighting_summary.get("weighted_samples", 0),
        "source_weight_min": source_weighting_summary.get("weight_min", ""),
        "source_weight_mean": source_weighting_summary.get("weight_mean", ""),
        "source_weight_max": source_weighting_summary.get("weight_max", ""),
        "effect_level_weighting_applied": bool(effect_level_weighting_summary.get("applied", False)),
        "effect_level_weighting_beta": effect_level_weighting_summary.get("beta", ""),
        "effect_level_weighted_samples": effect_level_weighting_summary.get("weighted_samples", 0),
        "effect_level_weight_min": effect_level_weighting_summary.get("effect_weight_min", ""),
        "effect_level_weight_mean": effect_level_weighting_summary.get("effect_weight_mean", ""),
        "effect_level_weight_max": effect_level_weighting_summary.get("effect_weight_max", ""),
        "final_sample_weight_min": effect_level_weighting_summary.get(
            "final_weight_min",
            source_weighting_summary.get("weight_min", ""),
        ),
        "final_sample_weight_mean": effect_level_weighting_summary.get(
            "final_weight_mean",
            source_weighting_summary.get("weight_mean", ""),
        ),
        "final_sample_weight_max": effect_level_weighting_summary.get(
            "final_weight_max",
            source_weighting_summary.get("weight_max", ""),
        ),
    }


def select_samples_by_domain_and_split(
    samples: list[dict[str, Any]],
    *,
    split_parts: tuple[str, ...],
    domains: tuple[str, ...],
) -> list[int]:
    split_set = {str(value).strip().lower() for value in split_parts if str(value).strip()}
    domain_set = {str(value).strip().lower() for value in domains if str(value).strip()}
    indices: list[int] = []
    for idx, sample in enumerate(samples):
        split_part = str(sample.get("split_part", "")).strip().lower()
        medium_domain = str(sample.get("medium_domain", "")).strip().lower()
        if split_set and split_part not in split_set:
            continue
        if domain_set and medium_domain not in domain_set:
            continue
        indices.append(idx)
    return indices


def max_tanimoto_similarity(source_fp: np.ndarray, target_fp: np.ndarray, *, chunk_size: int = 2048) -> np.ndarray:
    source_binary = (source_fp > 0).astype(np.uint8, copy=False)
    target_binary = (target_fp > 0).astype(np.uint8, copy=False)
    if source_binary.shape[0] == 0 or target_binary.shape[0] == 0:
        return np.zeros(source_binary.shape[0], dtype=np.float32)
    source_unique, source_inverse = np.unique(source_binary, axis=0, return_inverse=True)
    target_unique = np.unique(target_binary, axis=0)
    source = source_unique.astype(np.float32, copy=False)
    target = target_unique.astype(np.float32, copy=False)
    target_sums = target.sum(axis=1, keepdims=True).T
    unique_scores = np.zeros(source.shape[0], dtype=np.float32)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size]
        intersections = chunk @ target.T
        denominators = chunk.sum(axis=1, keepdims=True) + target_sums - intersections
        sims = np.divide(
            intersections,
            np.maximum(denominators, 1.0),
            out=np.zeros_like(intersections, dtype=np.float32),
            where=denominators > 0,
        )
        unique_scores[start : start + chunk.shape[0]] = sims.max(axis=1)
    return unique_scores[source_inverse]


def domain_alignment_reference_indices(
    samples: list[dict[str, Any]],
    config: DomainAlignmentConfig,
    *,
    phase: str,
) -> list[int]:
    if not config.active() or phase not in set(config.phases):
        return []
    if phase == "pretrain":
        return select_samples_by_domain_and_split(
            samples,
            split_parts=config.target_split_parts,
            domains=config.target_domains,
        )
    if phase == "finetune":
        return select_samples_by_domain_and_split(
            samples,
            split_parts=config.source_split_parts,
            domains=config.source_domains,
        )
    return []


def cycle_dataloader(dataloader: Any) -> Any:
    while True:
        for batch in dataloader:
            yield batch


def should_update_swa(config: SwaConfig, *, phase: str, epoch: int) -> bool:
    return config.active() and config.phase == phase and int(epoch) >= int(config.start_epoch)


def _hidden_dims(config: Mapping[str, Any]) -> tuple[int, ...]:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    if "hidden_dims" in model_cfg and model_cfg.get("hidden_dims") is not None:
        resolved = _dimension_tuple(model_cfg.get("hidden_dims"))
        if not resolved:
            raise ValueError("model.hidden_dims must include at least one positive dimension.")
        return resolved
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    return (hidden_dim, max(32, hidden_dim // 2))


def _dimension_tuple(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("Model dimension fields must be sequences of positive integers.")
    values = tuple(int(value) for value in raw)
    if any(value <= 0 for value in values):
        raise ValueError("Model dimensions must be positive integers.")
    return values


def _resolve_device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _early_stopping_config(
    config: Mapping[str, Any],
    *,
    enabled_override: bool | None,
    patience_override: int | None,
    min_delta_override: float | None,
    validation_fraction_override: float | None,
    monitor_split_override: str | None,
) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    raw = train_cfg.get("early_stopping", {}) if isinstance(train_cfg.get("early_stopping", {}), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False) if enabled_override is None else enabled_override),
        "patience": int(patience_override if patience_override is not None else raw.get("patience", 15)),
        "min_delta": float(min_delta_override if min_delta_override is not None else raw.get("min_delta", 0.0)),
        "validation_fraction": float(
            validation_fraction_override if validation_fraction_override is not None else raw.get("validation_fraction", 0.1)
        ),
        "monitor_split": str(monitor_split_override if monitor_split_override is not None else raw.get("monitor_split", "auto")),
    }


def _feature_noise_config(
    train_cfg: Mapping[str, Any],
    *,
    seed: int,
    train_replicates_override: int | None = None,
    finetune_replicates_override: int | None = None,
    numeric_noise_std_override: float | None = None,
    target_noise_std_override: float | None = None,
) -> FeatureNoiseConfig:
    raw = train_cfg.get("augmentation", {}) if isinstance(train_cfg.get("augmentation", {}), dict) else {}
    return FeatureNoiseConfig(
        train_replicates=max(
            1,
            int(train_replicates_override if train_replicates_override is not None else raw.get("train_replicates", 1)),
        ),
        finetune_replicates=max(
            1,
            int(
                finetune_replicates_override
                if finetune_replicates_override is not None
                else raw.get("finetune_replicates", raw.get("train_replicates", 1))
            ),
        ),
        numeric_noise_std=max(
            0.0,
            float(
                numeric_noise_std_override
                if numeric_noise_std_override is not None
                else raw.get("numeric_noise_std", 0.0)
            ),
        ),
        target_noise_std=max(
            0.0,
            float(
                target_noise_std_override
                if target_noise_std_override is not None
                else raw.get("target_noise_std", 0.0)
            ),
        ),
        seed=int(raw.get("seed", seed)),
    )


def _zscore_correction_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    threshold_override: float | None = None,
) -> ZScoreCorrectionConfig:
    raw = train_cfg.get("feature_zscore_correction", {}) if isinstance(train_cfg.get("feature_zscore_correction", {}), dict) else {}
    enabled = bool(raw.get("enabled", True) if enabled_override is None else enabled_override)
    threshold = float(threshold_override if threshold_override is not None else raw.get("threshold", 6.0))
    return ZScoreCorrectionConfig(enabled=enabled, threshold=max(threshold, 1.0))


def _perturbation_config(
    train_cfg: Mapping[str, Any],
    *,
    seed: int,
    replicates_override: int | None = None,
    numeric_noise_std_override: float | None = None,
) -> PerturbationConfig:
    raw = train_cfg.get("test_perturbation", {}) if isinstance(train_cfg.get("test_perturbation", {}), dict) else {}
    split_parts_raw = raw.get("split_parts", ["test"])
    if isinstance(split_parts_raw, str):
        split_parts = tuple(part.strip() for part in split_parts_raw.split(",") if part.strip())
    else:
        split_parts = tuple(str(part).strip() for part in split_parts_raw if str(part).strip())
    replicates = max(1, int(replicates_override if replicates_override is not None else raw.get("replicates", 1)))
    numeric_noise_std = max(
        0.0,
        float(
            numeric_noise_std_override
            if numeric_noise_std_override is not None
            else raw.get("numeric_noise_std", 0.0)
        ),
    )
    return PerturbationConfig(
        enabled=bool(raw.get("enabled", False) or replicates_override is not None or numeric_noise_std_override is not None),
        split_parts=split_parts or ("test",),
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        seed=int(raw.get("seed", seed + 200_000)),
    )


def _source_weighting_config(
    train_cfg: Mapping[str, Any],
    *,
    method_override: str | None = None,
    alpha_override: float | None = None,
) -> SourceWeightingConfig:
    raw = train_cfg.get("source_weighting", {}) if isinstance(train_cfg.get("source_weighting", {}), dict) else {}
    method = str(method_override if method_override is not None else raw.get("method", "none")).strip().lower()
    alpha = max(0.0, float(alpha_override if alpha_override is not None else raw.get("alpha", 0.0)))
    return SourceWeightingConfig(
        enabled=bool(raw.get("enabled", False) or method_override is not None or alpha_override is not None),
        method=method,
        alpha=alpha,
        source_split_parts=_string_tuple(raw.get("source_split_parts", ("train",)), lower=True) or ("train",),
        target_split_parts=_string_tuple(raw.get("target_split_parts", ("finetune",)), lower=True) or ("finetune",),
        source_domains=_string_tuple(raw.get("source_domains", ("aquatic",)), lower=True) or ("aquatic",),
        target_domains=_string_tuple(raw.get("target_domains", ("soil",)), lower=True) or ("soil",),
        min_weight=max(0.0, float(raw.get("min_weight", 0.25))),
        max_weight=max(0.0, float(raw.get("max_weight", 2.0))),
    )


def _effect_level_weighting_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    beta_override: float | None = None,
) -> EffectLevelWeightingConfig:
    raw = (
        train_cfg.get("effect_level_weighting", {})
        if isinstance(train_cfg.get("effect_level_weighting", {}), dict)
        else {}
    )
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if beta_override is not None:
        enabled = True
    beta = max(0.0, float(beta_override if beta_override is not None else raw.get("beta", 0.0)))
    return EffectLevelWeightingConfig(
        enabled=enabled,
        beta=beta,
        split_parts=_string_tuple(raw.get("split_parts", ("train", "finetune")), lower=True)
        or ("train", "finetune"),
        task_prefixes=_string_tuple(raw.get("task_prefixes", ("ECx", "LCx", "ICx", "LDx")), lower=False)
        or ("ECx", "LCx", "ICx", "LDx"),
        min_weight=max(0.0, float(raw.get("min_weight", 0.5))),
        max_weight=max(0.0, float(raw.get("max_weight", 3.0))),
    )


def _toxicity_binning_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    mode_override: str | None = None,
    loss_weight_override: float | None = None,
    scheme_override: str | None = None,
) -> ToxicityBinningConfig:
    raw = (
        train_cfg.get("toxicity_binning", {})
        if isinstance(train_cfg.get("toxicity_binning", {}), dict)
        else {}
    )
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if mode_override is not None or loss_weight_override is not None or scheme_override is not None:
        enabled = True
    mode = str(mode_override if mode_override is not None else raw.get("mode", "aux_classification")).strip().lower()
    loss_weight = max(0.0, float(loss_weight_override if loss_weight_override is not None else raw.get("loss_weight", 0.05)))
    return ToxicityBinningConfig(
        enabled=enabled,
        scheme=str(scheme_override if scheme_override is not None else raw.get("scheme", "authority_v1")).strip()
        or "authority_v1",
        mode=mode,
        loss_weight=loss_weight,
        boundary_policy=str(raw.get("boundary_policy", "hard")).strip().lower() or "hard",
        boundary_tolerance=max(0.0, float(raw.get("boundary_tolerance", 0.05))),
        threshold_multiplier=max(1e-12, float(raw.get("threshold_multiplier", 1.0))),
        require_active_bin_for_regression=bool(raw.get("require_active_bin_for_regression", False)),
    )


def _censored_loss_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    weight_override: float | None = None,
    margin_override: float | None = None,
) -> CensoredLossConfig:
    raw = train_cfg.get("censored_loss", {}) if isinstance(train_cfg.get("censored_loss", {}), dict) else {}
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if weight_override is not None or margin_override is not None:
        enabled = True
    method = str(raw.get("method", "hinge")).strip().lower() or "hinge"
    weight = max(0.0, float(weight_override if weight_override is not None else raw.get("weight", 0.0)))
    margin = max(0.0, float(margin_override if margin_override is not None else raw.get("margin", 0.0)))
    return CensoredLossConfig(
        enabled=enabled,
        method=method,
        weight=weight,
        margin=margin,
        split_parts=_string_tuple(raw.get("split_parts", ("train", "finetune")), lower=True)
        or ("train", "finetune"),
        include_ops=_string_tuple(raw.get("include_ops", ("<", "<=", ">", ">=")), lower=False)
        or ("<", "<=", ">", ">="),
    )


def _domain_alignment_config(
    train_cfg: Mapping[str, Any],
    *,
    method_override: str | None = None,
    weight_override: float | None = None,
) -> DomainAlignmentConfig:
    raw = train_cfg.get("domain_alignment", {}) if isinstance(train_cfg.get("domain_alignment", {}), dict) else {}
    method = str(method_override if method_override is not None else raw.get("method", "none")).strip().lower()
    weight = max(0.0, float(weight_override if weight_override is not None else raw.get("weight", 0.0)))
    return DomainAlignmentConfig(
        enabled=bool(raw.get("enabled", False) or method_override is not None or weight_override is not None),
        method=method,
        weight=weight,
        source_split_parts=_string_tuple(raw.get("source_split_parts", ("train",)), lower=True) or ("train",),
        target_split_parts=_string_tuple(raw.get("target_split_parts", ("finetune",)), lower=True) or ("finetune",),
        source_domains=_string_tuple(raw.get("source_domains", ("aquatic",)), lower=True) or ("aquatic",),
        target_domains=_string_tuple(raw.get("target_domains", ("soil",)), lower=True) or ("soil",),
        phases=_string_tuple(raw.get("phases", ("pretrain",)), lower=True) or ("pretrain",),
    )


def _swa_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    start_epoch_override: int | None = None,
    phase_override: str | None = None,
) -> SwaConfig:
    raw = train_cfg.get("swa", {}) if isinstance(train_cfg.get("swa", {}), dict) else {}
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    start_epoch = max(1, int(start_epoch_override if start_epoch_override is not None else raw.get("start_epoch", 15)))
    return SwaConfig(
        enabled=enabled,
        phase=str(
            phase_override if phase_override is not None else raw.get("phase", "finetune")
        ).strip().lower()
        or "finetune",
        start_epoch=start_epoch,
    )


def _string_tuple(value: Any, *, lower: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    if lower:
        return tuple(item.lower() for item in items if item)
    return tuple(item for item in items if item)


def split_training_validation_indices(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    seed: int,
    validation_fraction: float,
    monitor_split: str = "auto",
) -> tuple[list[int], list[int], str]:
    requested = (monitor_split or "auto").strip().lower()
    force_internal = requested in {"internal_train_fraction", "train_fraction"}
    explicit_parts = [] if requested == "auto" or force_internal else [requested]
    preferred_parts = [] if force_internal else (explicit_parts or ["finetune", "validation", "val"])
    for part in preferred_parts:
        indices = [idx for idx, sample in enumerate(samples) if str(sample.get("split_part", "")).lower() == part]
        if indices:
            return list(train_indices), indices, part

    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(train_indices) < 2:
        return list(train_indices), [], ""

    rng = np.random.default_rng(seed)
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for idx in train_indices:
        task = str(samples[idx].get("task_head", ""))
        by_task.setdefault(task, []).append(idx)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    actual_train = [idx for idx in train_indices if idx not in validation_set]
    return actual_train, sorted(validation), "internal_train_fraction" if validation else ""


def resolve_stage1_training_boundary(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    seed: int,
    validation_fraction: float,
    monitor_split: str,
    requested_epochs: int,
    downstream_training_requested: bool,
) -> tuple[list[int], list[int], str]:
    """Allow a strict Stage-3-only run without inventing Stage-1 samples.

    A missing Stage-1 boundary remains an error whenever Stage 1 has positive
    epochs or neither a downstream phase nor a read-only checkpoint evaluation
    is requested. This keeps ordinary deep experiments fail-closed while
    permitting Stage-3-only runs and zero-epoch evaluation of a locked model.
    """

    if train_indices:
        return split_training_validation_indices(
            samples,
            train_indices=train_indices,
            seed=seed,
            validation_fraction=validation_fraction,
            monitor_split=monitor_split,
        )
    if int(requested_epochs) > 0 or not downstream_training_requested:
        raise ValueError("No train samples found for deep experiment.")
    return [], [], ""


def resolve_preprocessing_fit_indices(
    *,
    actual_train_indices: Sequence[int],
    finetune_train_indices: Sequence[int],
    finetune_mgkg_train_indices: Sequence[int],
    finetune_requested: bool,
    finetune_mgkg_requested: bool,
    evaluation_checkpoint_requested: bool,
) -> list[int]:
    """Resolve training-only preprocessing rows, including zero-epoch evaluation.

    A read-only M00 checkpoint evaluation has no Stage-1 ``train`` rows and no
    requested optimization epochs. It must nevertheless reconstruct the
    checkpoint's preprocessing state from the Stage-3 training partition, not
    from validation or outer-test rows.
    """

    indices = list(int(index) for index in actual_train_indices)
    if finetune_requested:
        indices.extend(int(index) for index in finetune_train_indices)
    if finetune_mgkg_requested:
        indices.extend(int(index) for index in finetune_mgkg_train_indices)
    if evaluation_checkpoint_requested and not indices:
        if finetune_mgkg_train_indices:
            indices.extend(int(index) for index in finetune_mgkg_train_indices)
        elif finetune_train_indices:
            indices.extend(int(index) for index in finetune_train_indices)
    return list(dict.fromkeys(indices))


def resolve_stage1_monitor_split(
    monitor_split: str | None,
    *,
    staged_training: bool,
) -> str:
    """Keep stage-1 scheduling and model selection independent of downstream rows."""

    requested = (monitor_split or "auto").strip().lower()
    if not staged_training:
        return requested
    if requested in {"", "auto"}:
        return "internal_train_fraction"
    if requested in {"finetune", "finetune_mgkg"}:
        raise ValueError(
            "Stage-1 model selection cannot monitor downstream fine-tuning rows. "
            "Use monitor_split='internal_train_fraction' or a dedicated stage-1 validation split."
        )
    return requested


def split_finetune_validation_indices(
    samples: list[dict[str, Any]],
    *,
    finetune_indices: list[int],
    seed: int,
    validation_fraction: float,
    monitor_split: str = "auto",
) -> tuple[list[int], list[int], str]:
    requested = (monitor_split or "auto").strip().lower()
    if requested not in {"", "auto", "internal", "internal_finetune_fraction", "finetune_fraction"}:
        validation = [
            idx
            for idx, sample in enumerate(samples)
            if str(sample.get("split_part", "")).lower() == requested
        ]
        if validation:
            return list(finetune_indices), validation, requested

    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(finetune_indices) < 2:
        return list(finetune_indices), [], ""

    rng = np.random.default_rng(seed + 17_031)
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for idx in finetune_indices:
        task = str(samples[idx].get("task_head", ""))
        by_task.setdefault(task, []).append(idx)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    finetune_train = [idx for idx in finetune_indices if idx not in validation_set]
    if not finetune_train:
        return list(finetune_indices), [], ""
    return finetune_train, sorted(validation), "internal_finetune_fraction" if validation else ""


def normalize_prediction_split_parts(
    split_parts: tuple[str, ...] | None,
) -> frozenset[str]:
    if not split_parts:
        return frozenset()
    normalized = {
        str(part).strip().lower()
        for part in split_parts
        if str(part).strip()
    }
    if not normalized:
        raise ValueError("prediction_split_parts must include at least one non-empty split label.")
    return frozenset(normalized)


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def stage3_init_checkpoint_contract(
    *,
    data_identity: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    base_architecture: Mapping[str, Any],
    weighting_and_auxiliary: Mapping[str, Any],
    stage12_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the complete compatibility boundary for reusable stage-1/2 weights."""

    contract = {
        "schema_version": 2,
        "data_identity": dict(data_identity),
        "preprocessing": dict(preprocessing),
        "base_architecture": dict(base_architecture),
        # Enforce the scientific/runtime boundary at the contract constructor
        # as well as at the experiment assembly call site.  This prevents a
        # future caller from accidentally sealing cache provenance.
        "weighting_and_auxiliary": scientific_summary_contract(
            weighting_and_auxiliary
        ),
        "stage12_protocol": dict(stage12_protocol),
    }
    return seal_stage3_init_contract(contract)


def canonical_contract_value(value: Any) -> Any:
    """Convert nested scientific/config objects into deterministic JSON values."""

    if isinstance(value, Mapping):
        return {
            str(key): canonical_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return canonical_contract_value(value.tolist())
    if isinstance(value, np.generic):
        return canonical_contract_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [canonical_contract_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_contract_value(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        if math.isinf(value):
            return {"__nonfinite_float__": "inf" if value > 0 else "-inf"}
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return canonical_contract_value(value.detach().cpu().tolist())
    raise TypeError(
        f"Unsupported value in stage-3 init checkpoint contract: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_contract_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_summary_contract(value: Any) -> Any:
    """Remove runtime cache provenance from a scientific checkpoint contract.

    Cache hits, paths, creation flags, schemas, and cache keys describe how a
    deterministic result was obtained, not which result was obtained.  They
    remain in the normal experiment manifest for traceability.  Recursively
    excluding every ``cache_*`` key here makes cold and warm executions
    compatible while preserving scientific configuration and statistics.
    """

    if isinstance(value, Mapping):
        return {
            str(key): scientific_summary_contract(item)
            for key, item in value.items()
            if not str(key).strip().lower().startswith("cache_")
        }
    if isinstance(value, list):
        return [scientific_summary_contract(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scientific_summary_contract(item) for item in value)
    return value


def stage3_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("sha256", None)
    return canonical_sha256(payload)


def seal_stage3_init_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and hash a contract, never preserving a caller-supplied digest."""

    normalized = canonical_contract_value(contract)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by type.
        raise TypeError("Stage-3 init checkpoint contract must be a mapping.")
    normalized["sha256"] = stage3_contract_sha256(normalized)
    return normalized


def verify_stage3_init_contract(contract: Mapping[str, Any], *, label: str) -> str:
    """Recompute a payload digest and reject stale or tampered self-reported hashes."""

    claimed = str(contract.get("sha256", ""))
    recomputed = stage3_contract_sha256(contract)
    if not claimed or claimed != recomputed:
        raise ValueError(
            f"{label} stage-3 init checkpoint contract hash is invalid: "
            f"claimed={claimed or '<missing>'} recomputed={recomputed}"
        )
    return recomputed


def database_source_identity(
    db_path: str | Path, *, source_table: str, split_name: str
) -> dict[str, Any]:
    path = Path(db_path).resolve()
    stat = path.stat()
    return {
        "modeling_tables_db": str(path),
        "database_size_bytes": int(stat.st_size),
        "database_mtime_ns": int(stat.st_mtime_ns),
        "source_table": str(source_table),
        "split_name": str(split_name),
        "identity_method": "path_size_mtime_plus_selected_sample_content_v1",
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plain_model_state_dict(path: str | Path, *, map_location: Any) -> dict[str, Any]:
    """Load a plain model checkpoint for read-only evaluation with strict key matching."""

    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, Mapping) and "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")
    return {str(key): value for key, value in payload.items()}


def stage12_runtime_code_identity() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "qsar_tl/modeling/network.py",
        "qsar_tl/modeling/dataset.py",
        "qsar_tl/training/deep_experiment.py",
        "qsar_tl/training/deep_train.py",
        "qsar_tl/training/censored_loss.py",
        "qsar_tl/training/toxicity_binning.py",
    )
    return {
        relative_path: file_sha256(project_root / relative_path)
        for relative_path in relative_paths
    }


def sample_scientific_identity_sha256(
    samples: list[dict[str, Any]],
    indices: list[int] | tuple[int, ...],
    *,
    include_split_part: bool = False,
) -> str:
    records = []
    for index in indices:
        sample = samples[index]
        record = {
            "sample_id": sample.get("sample_id", ""),
            "aggregate_id": sample.get("aggregate_id", ""),
            "record_id": sample.get("record_id", ""),
            "result_ids": sample.get("result_ids", ""),
            "task_head": sample.get("task_head", ""),
            "base_task_head": sample.get("base_task_head", ""),
            "target_name": sample.get("target_name", ""),
            "target_family": sample.get("target_family", ""),
            "target_value_raw": sample.get("target_value_raw", None),
            "target_value_scaled": sample.get("target_value_scaled", None),
        }
        if include_split_part:
            record["split_part"] = sample.get("split_part", "")
            record["original_split_part"] = sample.get("original_split_part", "")
        records.append(record)
    records.sort(key=canonical_json)
    return canonical_sha256(records)


def sample_preprocessing_input_sha256(
    samples: list[dict[str, Any]], indices: list[int] | tuple[int, ...]
) -> str:
    records: list[dict[str, Any]] = []
    for index in indices:
        sample = samples[index]
        record = {
            "sample_id": sample.get("sample_id", ""),
            "molecular_numeric": sample.get("molecular_numeric", []),
            "fingerprint": sample.get("fingerprint", []),
            "molecular_graph": sample.get("molecular_graph", {}),
            "categorical_ids": sample.get("categorical_ids", {}),
            "adapter_name": sample.get("adapter_name", ""),
            "adapter_id": sample.get("adapter_id", 0),
            "target_scale_key": sample.get("target_scale_key", ""),
            "target_value": sample.get("target_value", None),
            "sample_weight": sample.get("sample_weight", 1.0),
            "toxicity_bin_index": sample.get("toxicity_bin_index", -1),
            "censored_direction_id": sample.get("censored_direction_id", 0),
        }
        records.append(record)
    digest = hashlib.sha256()
    for record in sorted(records, key=canonical_json):
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_weight_identity_contract(
    samples: list[dict[str, Any]], indices: list[int] | tuple[int, ...]
) -> dict[str, Any]:
    """Hash the exact per-sample weights consumed by one training stage."""

    records: list[dict[str, Any]] = []
    for index in indices:
        sample = samples[index]
        records.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "aggregate_id": sample.get("aggregate_id", ""),
                "task_head": sample.get("task_head", ""),
                "split_part": sample.get("split_part", ""),
                "sample_weight": sample.get("sample_weight", 1.0),
            }
        )
    records.sort(key=canonical_json)
    return {
        "rows": int(len(indices)),
        "sha256": canonical_sha256(records),
    }


def sample_indices_contract(
    samples: list[dict[str, Any]], indices: list[int] | tuple[int, ...]
) -> dict[str, Any]:
    return {
        "rows": int(len(indices)),
        "scientific_identity_sha256": sample_scientific_identity_sha256(
            samples, indices
        ),
        "preprocessing_input_sha256": sample_preprocessing_input_sha256(
            samples, indices
        ),
    }


def sample_indices_sha256(
    samples: list[dict[str, Any]], indices: list[int] | tuple[int, ...]
) -> str:
    """Hash canonical scientific row identities used by a checkpoint phase."""

    return sample_scientific_identity_sha256(samples, indices)


def export_stage3_init_checkpoint(
    model: Any,
    path: str | Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Export the restored stage-2 model before any stage-3 optimizer step."""

    import torch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_contract = seal_stage3_init_contract(contract)
    payload = {
        "format": "qsar_stage3_init_v1",
        "contract": sealed_contract,
        "state_dict": clone_state_dict(model),
    }
    torch.save(payload, checkpoint_path)
    return {
        "exported": True,
        "path": str(checkpoint_path),
        "contract_sha256": sealed_contract["sha256"],
        "contract_schema_version": int(sealed_contract.get("schema_version", 0)),
        "contract_hash_recomputed": True,
        "parameter_tensors": int(len(payload["state_dict"])),
    }


def load_stage3_init_checkpoint(
    model: Any,
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
    reset_hierarchical_heads: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load a same-seed stage-2 checkpoint, resetting G3 residual branches."""

    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-3 init checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != "qsar_stage3_init_v1":
        raise ValueError("Stage-3 init checkpoint has an unsupported or missing format.")
    contract = payload.get("contract")
    state = payload.get("state_dict")
    if not isinstance(contract, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Stage-3 init checkpoint is missing contract or state_dict.")
    expected_sha = verify_stage3_init_contract(
        expected_contract, label="Expected"
    )
    actual_sha = verify_stage3_init_contract(contract, label="Checkpoint payload")
    if actual_sha != expected_sha:
        raise ValueError(
            "Stage-3 init checkpoint contract mismatch; same seed, split, preprocessing, "
            "and base architecture are required. "
            f"expected={expected_sha or '<missing>'} actual={actual_sha or '<missing>'}"
        )

    target_state = model.state_dict()
    registered_heads = frozenset(getattr(model, "mgkg_hierarchical_heads", ()))
    requested_reset_heads = frozenset(reset_hierarchical_heads)
    if requested_reset_heads != registered_heads:
        raise ValueError(
            "G3 checkpoint loading may reset only the model's pre-registered "
            "hierarchical heads: "
            f"registered={sorted(registered_heads)} requested={sorted(requested_reset_heads)}"
        )
    registered_hierarchy_keys = {
        key
        for key in target_state
        if key.startswith("mgkg_hierarchical_shared_head.")
        or key.startswith("mgkg_hierarchical_family_heads.")
        or any(key.startswith(f"heads.{head}.") for head in registered_heads)
    }
    reset_hierarchy_keys = {
        key
        for key in registered_hierarchy_keys
        if key.startswith("mgkg_hierarchical_family_heads.")
        or any(key.startswith(f"heads.{head}.") for head in registered_heads)
    }
    compatible: dict[str, Any] = {}
    skipped_reset: list[str] = []
    for key, value in state.items():
        key = str(key)
        if key in reset_hierarchy_keys:
            skipped_reset.append(key)
            continue
        if key not in target_state:
            raise ValueError(f"Stage-3 init checkpoint has an unexpected parameter: {key}")
        if tuple(value.shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"Stage-3 init checkpoint parameter shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} model={tuple(target_state[key].shape)}"
            )
        compatible[key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected parameters during stage-3 checkpoint load: {unexpected}")
    allowed_missing = set(registered_hierarchy_keys)
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if disallowed_missing:
        raise ValueError(
            "Stage-3 init checkpoint is missing non-hierarchical parameters: "
            f"{disallowed_missing}"
        )
    # Both family and exact-task residuals must start at exactly zero even when
    # the source checkpoint was itself hierarchical.
    family_heads = getattr(model, "mgkg_hierarchical_family_heads", None)
    if family_heads is not None:
        for module in family_heads.values():
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)
    for head in reset_hierarchical_heads:
        module = model.heads[head]
        torch.nn.init.zeros_(module.weight)
        torch.nn.init.zeros_(module.bias)
    return {
        "loaded": True,
        "path": str(checkpoint_path),
        "contract_sha256": actual_sha,
        "contract_schema_version": int(contract.get("schema_version", 0)),
        "contract_hash_recomputed": True,
        "loaded_parameter_tensors": int(len(compatible)),
        "reset_parameter_tensors": int(len(skipped_reset)),
        "allowed_missing_hierarchy_tensors": int(
            len(set(missing) & registered_hierarchy_keys)
        ),
        "hierarchical_residuals_reset": bool(reset_hierarchical_heads),
    }


def build_scheduler(optimizer: Any, config: DeepTrainingConfig, epochs: int) -> Any | None:
    scheduler_name = (config.scheduler or "none").strip().lower()
    if scheduler_name in {"", "none", "off", "false"}:
        return None
    import torch

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    raise ValueError("training.scheduler must be 'none', 'cosine', or 'reduce_on_plateau'.")


def step_scheduler(scheduler: Any | None, metric: float) -> None:
    if scheduler is None:
        return
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(float(metric))
    else:
        scheduler.step()


def current_learning_rate(optimizer: Any) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0


def build_regression_loss(config: DeepTrainingConfig) -> Any:
    """Build the stage-local standardized-target regression objective."""

    import torch

    mse_weight = float(config.mse_loss_weight)
    if not 0.0 <= mse_weight <= 1.0:
        raise ValueError("mse_loss_weight must be in [0, 1].")

    def loss_fn(predictions: Any, targets: Any) -> Any:
        losses = elementwise_regression_loss(predictions, targets, config=config)
        return losses.mean()

    return loss_fn


def elementwise_regression_loss(
    predictions: Any,
    targets: Any,
    *,
    config: DeepTrainingConfig,
) -> Any:
    """Return Huber/MSE loss per row so sample weighting remains exact."""

    import torch

    mse_weight = float(config.mse_loss_weight)
    huber = torch.nn.functional.huber_loss(
        predictions,
        targets,
        reduction="none",
        delta=float(config.huber_delta),
    )
    if mse_weight <= 0:
        return huber
    mse = torch.nn.functional.mse_loss(predictions, targets, reduction="none")
    return (1.0 - mse_weight) * huber + mse_weight * mse


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "false"}:
        return None
    return float(value)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    config: DeepTrainingConfig,
    device: Any,
    *,
    alignment_batches: Any | None = None,
    alignment_weight: float = 0.0,
) -> dict[str, Any]:
    import torch

    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_toxicity_bin_loss = 0.0
    total_toxicity_bin_samples = 0
    total_censored_loss = 0.0
    total_censored_samples = 0
    total_alignment_loss = 0.0
    alignment_steps = 0
    total_samples = 0
    max_grad_norm = 0.0
    for batch in dataloader:
        molecular_numeric = batch["molecular_numeric"].to(device, non_blocking=True)
        fingerprint = batch["fingerprint"].to(device, non_blocking=True)
        molecular_graph = graph_to_device(batch.get("molecular_graph"), device, non_blocking=True)
        categorical_ids = {key: value.to(device, non_blocking=True) for key, value in batch["categorical_ids"].items()}
        adapter_ids = batch.get("adapter_id")
        adapter_ids = adapter_ids.to(device, non_blocking=True) if adapter_ids is not None else None
        targets = batch["target_value"].to(device, non_blocking=True)
        sample_weights = batch.get("sample_weight")
        sample_weights = sample_weights.to(device, non_blocking=True) if sample_weights is not None else None
        toxicity_bin_index = batch.get("toxicity_bin_index")
        toxicity_bin_index = toxicity_bin_index.to(device, non_blocking=True) if toxicity_bin_index is not None else None
        censored_direction_id = batch.get("censored_direction_id")
        censored_direction_id = censored_direction_id.to(device, non_blocking=True) if censored_direction_id is not None else None
        task_heads = list(batch["task_head"])
        optimizer.zero_grad(set_to_none=True)
        loss_components = batch_weighted_loss(
            model,
            molecular_numeric,
            fingerprint,
            categorical_ids,
            adapter_ids,
            targets,
            task_heads,
            loss_fn,
            config,
            device,
            sample_weights=sample_weights,
            toxicity_bin_index=toxicity_bin_index,
            toxicity_bin_loss_weight=config.toxicity_bin_loss_weight,
            censored_direction_id=censored_direction_id,
            censored_loss_weight=config.censored_loss_weight,
            censored_loss_margin=config.censored_loss_margin,
            molecular_graph=molecular_graph,
            return_components=True,
        )
        task_loss = loss_components["regression_loss"]
        loss = loss_components["loss"]
        alignment_loss = None
        if alignment_batches is not None and alignment_weight > 0:
            reference_batch = next(alignment_batches)
            alignment_loss = coral_batch_loss(
                model,
                batch,
                reference_batch,
                device=device,
            )
            loss = loss + float(alignment_weight) * alignment_loss
        loss.backward()
        if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            max_grad_norm = max(max_grad_norm, float(grad_norm.detach().cpu()))
        optimizer.step()
        batch_size = len(task_heads)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_task_loss += float(task_loss.detach().cpu()) * batch_size
        toxicity_bin_samples = int(loss_components.get("toxicity_bin_samples", 0))
        if toxicity_bin_samples > 0:
            total_toxicity_bin_loss += float(loss_components["toxicity_bin_loss"].detach().cpu()) * toxicity_bin_samples
            total_toxicity_bin_samples += toxicity_bin_samples
        censored_samples = int(loss_components.get("censored_samples", 0))
        if censored_samples > 0:
            total_censored_loss += float(loss_components["censored_loss"].detach().cpu()) * censored_samples
            total_censored_samples += censored_samples
        if alignment_loss is not None:
            total_alignment_loss += float(alignment_loss.detach().cpu())
            alignment_steps += 1
        total_samples += batch_size
    return {
        "mean_loss": total_loss / max(total_samples, 1),
        "mean_task_loss": total_task_loss / max(total_samples, 1),
        "mean_toxicity_bin_loss": total_toxicity_bin_loss / max(total_toxicity_bin_samples, 1),
        "toxicity_bin_samples": total_toxicity_bin_samples,
        "toxicity_bin_loss_weight": config.toxicity_bin_loss_weight,
        "mean_censored_loss": total_censored_loss / max(total_censored_samples, 1),
        "censored_samples": total_censored_samples,
        "censored_loss_weight": config.censored_loss_weight,
        "mean_alignment_loss": total_alignment_loss / max(alignment_steps, 1) if alignment_steps else 0.0,
        "alignment_steps": alignment_steps,
        "samples": total_samples,
        "max_grad_norm": max_grad_norm,
    }


def evaluate_loss(model: Any, dataloader: Any, loss_fn: Any, config: DeepTrainingConfig, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_toxicity_bin_loss = 0.0
    total_toxicity_bin_samples = 0
    total_censored_loss = 0.0
    total_censored_samples = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            molecular_numeric = batch["molecular_numeric"].to(device, non_blocking=True)
            fingerprint = batch["fingerprint"].to(device, non_blocking=True)
            molecular_graph = graph_to_device(batch.get("molecular_graph"), device, non_blocking=True)
            categorical_ids = {key: value.to(device, non_blocking=True) for key, value in batch["categorical_ids"].items()}
            adapter_ids = batch.get("adapter_id")
            adapter_ids = adapter_ids.to(device, non_blocking=True) if adapter_ids is not None else None
            targets = batch["target_value"].to(device, non_blocking=True)
            task_heads = list(batch["task_head"])
            toxicity_bin_index = batch.get("toxicity_bin_index")
            toxicity_bin_index = toxicity_bin_index.to(device, non_blocking=True) if toxicity_bin_index is not None else None
            censored_direction_id = batch.get("censored_direction_id")
            censored_direction_id = censored_direction_id.to(device, non_blocking=True) if censored_direction_id is not None else None
            loss_components = batch_weighted_loss(
                model,
                molecular_numeric,
                fingerprint,
                categorical_ids,
                adapter_ids,
                targets,
                task_heads,
                loss_fn,
                config,
                device,
                sample_weights=None,
                toxicity_bin_index=toxicity_bin_index,
                toxicity_bin_loss_weight=config.toxicity_bin_loss_weight,
                censored_direction_id=censored_direction_id,
                censored_loss_weight=config.censored_loss_weight,
                censored_loss_margin=config.censored_loss_margin,
                molecular_graph=molecular_graph,
                return_components=True,
            )
            loss = loss_components["loss"]
            batch_size = len(task_heads)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_task_loss += float(loss_components["regression_loss"].detach().cpu()) * batch_size
            toxicity_bin_samples = int(loss_components.get("toxicity_bin_samples", 0))
            if toxicity_bin_samples > 0:
                total_toxicity_bin_loss += float(loss_components["toxicity_bin_loss"].detach().cpu()) * toxicity_bin_samples
                total_toxicity_bin_samples += toxicity_bin_samples
            censored_samples = int(loss_components.get("censored_samples", 0))
            if censored_samples > 0:
                total_censored_loss += float(loss_components["censored_loss"].detach().cpu()) * censored_samples
                total_censored_samples += censored_samples
            total_samples += batch_size
    return {
        "mean_loss": total_loss / max(total_samples, 1),
        "mean_task_loss": total_task_loss / max(total_samples, 1),
        "mean_toxicity_bin_loss": total_toxicity_bin_loss / max(total_toxicity_bin_samples, 1),
        "toxicity_bin_samples": total_toxicity_bin_samples,
        "mean_censored_loss": total_censored_loss / max(total_censored_samples, 1),
        "censored_samples": total_censored_samples,
        "samples": total_samples,
    }


def batch_weighted_loss(
    model: Any,
    molecular_numeric: Any,
    fingerprint: Any,
    categorical_ids: dict[str, Any],
    adapter_ids: Any,
    targets: Any,
    task_heads: list[str],
    loss_fn: Any,
    config: DeepTrainingConfig,
    device: Any,
    sample_weights: Any | None = None,
    toxicity_bin_index: Any | None = None,
    toxicity_bin_loss_weight: float = 0.0,
    censored_direction_id: Any | None = None,
    censored_loss_weight: float = 0.0,
    censored_loss_margin: float = 0.0,
    molecular_graph: Any | None = None,
    return_components: bool = False,
) -> Any:
    import torch

    model_kwargs = {"adapter_ids": adapter_ids}
    if molecular_graph is not None:
        model_kwargs["molecular_graph"] = molecular_graph
    outputs = model(molecular_numeric, fingerprint, categorical_ids, **model_kwargs)
    censored_ids = (
        torch.zeros(targets.shape[0], dtype=torch.long, device=device)
        if censored_direction_id is None
        else censored_direction_id.to(device=device, dtype=torch.long)
    )
    censored_mask = censored_ids != 0
    losses = []
    weight_sum = 0.0
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        task_mask = torch.tensor([head == task_head for head in task_heads], dtype=torch.bool, device=device)
        mask = task_mask & ~censored_mask
        if not bool(mask.any()):
            continue
        weight = float(config.task_weights.get(task_head, 1.0))
        weight_sum += weight
        losses.append(
            weight
            * masked_huber_loss(
                outputs[task_head][mask],
                targets[mask],
                loss_fn=loss_fn,
                config=config,
                sample_weights=None if sample_weights is None else sample_weights[mask],
            )
        )
    if not losses:
        regression_loss = torch.tensor(0.0, device=device)
    else:
        regression_loss = torch.stack(losses).sum() / max(weight_sum, 1e-12)
    toxicity_bin_loss, toxicity_bin_samples = toxicity_bin_auxiliary_loss(
        outputs,
        toxicity_bin_index,
        mode=config.toxicity_binning_mode,
    )
    censored_loss, censored_samples = censored_multitask_loss(
        outputs,
        targets=targets,
        task_heads=task_heads,
        censored_direction_id=censored_ids,
        margin=censored_loss_margin,
    )
    if toxicity_bin_loss is not None and float(toxicity_bin_loss_weight) > 0:
        total_loss = regression_loss + float(toxicity_bin_loss_weight) * toxicity_bin_loss
    else:
        total_loss = regression_loss
        if toxicity_bin_loss is None:
            toxicity_bin_loss = torch.tensor(0.0, device=device)
    if censored_loss is not None and float(censored_loss_weight) > 0:
        total_loss = total_loss + float(censored_loss_weight) * censored_loss
    elif censored_loss is None:
        censored_loss = torch.tensor(0.0, device=device)
    if return_components:
        return {
            "loss": total_loss,
            "regression_loss": regression_loss,
            "toxicity_bin_loss": toxicity_bin_loss,
            "toxicity_bin_samples": toxicity_bin_samples,
            "censored_loss": censored_loss,
            "censored_samples": censored_samples,
        }
    return total_loss


def censored_multitask_loss(
    outputs: Mapping[str, Any],
    *,
    targets: Any,
    task_heads: list[str],
    censored_direction_id: Any,
    margin: float = 0.0,
) -> tuple[Any | None, int]:
    import torch

    losses = []
    count = 0
    device = targets.device
    direction_ids = censored_direction_id.to(device=device, dtype=censored_direction_id.dtype)
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        mask = torch.tensor([head == task_head for head in task_heads], dtype=torch.bool, device=device)
        mask = mask & (direction_ids != 0)
        task_count = int(mask.sum().detach().cpu())
        if task_count <= 0:
            continue
        losses.append(censored_hinge_loss(outputs[task_head][mask], targets[mask], direction_ids[mask], margin=margin))
        count += task_count
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), count


def toxicity_bin_auxiliary_loss(
    outputs: Mapping[str, Any],
    toxicity_bin_index: Any | None,
    *,
    mode: str = "aux_classification",
) -> tuple[Any | None, int]:
    import torch

    logits = outputs.get(TOXICITY_BIN_LOGITS_KEY)
    if logits is None or toxicity_bin_index is None:
        return None, 0
    targets = toxicity_bin_index.to(device=logits.device, dtype=torch.long)
    mask = targets >= 0
    count = int(mask.sum().detach().cpu())
    if count <= 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    if str(mode or "").strip().lower() == "ordinal":
        return ordinal_softmax_loss(logits, targets), count
    return torch.nn.functional.cross_entropy(logits[mask], targets[mask]), count


def masked_huber_loss(
    predictions: Any,
    targets: Any,
    *,
    loss_fn: Any,
    config: DeepTrainingConfig,
    sample_weights: Any | None = None,
) -> Any:
    import torch

    if sample_weights is None:
        return loss_fn(predictions, targets)
    losses = elementwise_regression_loss(predictions, targets, config=config)
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype).clamp(min=0.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0:
        return losses.mean()
    return (losses * weights).sum() / weight_sum


def coral_batch_loss(
    model: Any,
    batch: Mapping[str, Any],
    reference_batch: Mapping[str, Any],
    *,
    device: Any,
) -> Any:
    main_shared = shared_representation(model, batch, device=device)
    reference_shared = shared_representation(model, reference_batch, device=device)
    return coral_loss(main_shared, reference_shared)


def shared_representation(model: Any, batch: Mapping[str, Any], *, device: Any) -> Any:
    molecular_numeric = batch["molecular_numeric"].to(device, non_blocking=True)
    fingerprint = batch["fingerprint"].to(device, non_blocking=True)
    molecular_graph = graph_to_device(batch.get("molecular_graph"), device, non_blocking=True)
    categorical_ids = {key: value.to(device, non_blocking=True) for key, value in batch["categorical_ids"].items()}
    adapter_ids = batch.get("adapter_id")
    adapter_ids = adapter_ids.to(device, non_blocking=True) if adapter_ids is not None else None
    return model.encode_shared(
        molecular_numeric=molecular_numeric,
        fingerprint=fingerprint,
        categorical_ids=categorical_ids,
        adapter_ids=adapter_ids,
        molecular_graph=molecular_graph,
    )


def coral_loss(source: Any, target: Any) -> Any:
    import torch

    if source.shape[0] < 2 or target.shape[0] < 2:
        return torch.zeros((), dtype=source.dtype, device=source.device)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T.matmul(source_centered) / max(source.shape[0] - 1, 1)
    target_cov = target_centered.T.matmul(target_centered) / max(target.shape[0] - 1, 1)
    dim = max(int(source.shape[1]), 1)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * dim * dim)


def resolve_trajectory_config(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = train_cfg.get("trajectory", {}) if isinstance(train_cfg.get("trajectory", {}), Mapping) else {}
    enabled = bool(raw.get("enabled", False))
    anchor_size = int(raw.get("anchor_size", 0) or 0)
    selection_seed = int(raw.get("selection_seed", 71_000))
    if enabled and anchor_size <= 0:
        raise ValueError("training.trajectory.anchor_size must be positive when trajectory capture is enabled.")
    return {
        "enabled": enabled,
        "anchor_size": anchor_size,
        "selection_seed": selection_seed,
        "selection_method": "task_stratified_sha256_v1",
        "snapshot_policy": "initialization_and_every_epoch",
    }


def resolve_final_representation_config(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        train_cfg.get("representation_export", {})
        if isinstance(train_cfg.get("representation_export", {}), Mapping)
        else {}
    )
    return {
        "enabled": bool(raw.get("enabled", False)),
        "batch_size": int(raw.get("batch_size", 0) or 0),
        "boundary": "requested_development_prediction_rows_only",
    }


def select_task_stratified_anchor(
    samples: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
    *,
    size: int,
    seed: int,
) -> list[int]:
    """Select a fixed task-stratified anchor without consuming any RNG stream."""

    unique_indices = list(dict.fromkeys(int(index) for index in candidate_indices))
    target_size = min(max(int(size), 0), len(unique_indices))
    if target_size <= 0:
        return []
    by_task: dict[str, list[int]] = {}
    for index in unique_indices:
        task = str(samples[index].get("task_head", ""))
        by_task.setdefault(task, []).append(index)
    for task, indices in by_task.items():
        indices.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}|{task}|{_trajectory_sample_identity(samples[index], index)}".encode("utf-8")
            ).hexdigest()
        )

    tasks = sorted(by_task)
    quotas = {task: 0 for task in tasks}
    if target_size >= len(tasks):
        for task in tasks:
            quotas[task] = 1
    remaining = target_size - sum(quotas.values())
    capacities = {task: len(by_task[task]) - quotas[task] for task in tasks}
    while remaining > 0:
        available = [task for task in tasks if capacities[task] > 0]
        if not available:
            break
        total_capacity = sum(capacities[task] for task in available)
        raw = {task: remaining * capacities[task] / total_capacity for task in available}
        additions = {
            task: min(capacities[task], int(math.floor(raw[task]))) for task in available
        }
        allocated = sum(additions.values())
        if allocated == 0:
            ranked = sorted(
                available,
                key=lambda task: (
                    -(raw[task] - math.floor(raw[task])),
                    hashlib.sha256(f"{seed}|quota|{task}".encode("utf-8")).hexdigest(),
                ),
            )
            additions[ranked[0]] = 1
            allocated = 1
        for task, count in additions.items():
            quotas[task] += count
            capacities[task] -= count
        remaining -= allocated

    selected = [index for task in tasks for index in by_task[task][: quotas[task]]]
    selected.sort(
        key=lambda index: hashlib.sha256(
            f"{seed}|global|{_trajectory_sample_identity(samples[index], index)}".encode("utf-8")
        ).hexdigest()
    )
    return selected


def _trajectory_sample_identity(sample: Mapping[str, Any], fallback_index: int) -> str:
    primary = ""
    for key in ("stage_sample_record_id", "sample_id", "aggregate_id", "record_id"):
        value = str(sample.get(key, "")).strip()
        if value:
            primary = value
            break
    if not primary:
        primary = f"row:{fallback_index}"
    return "|".join(
        (
            primary,
            str(sample.get("task_head", "")),
            str(sample.get("target_name", "")),
            str(sample.get("target_family", "")),
        )
    )


def write_trajectory_anchor_manifest(
    path: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    config: Mapping[str, Any],
) -> None:
    identities = [_trajectory_sample_identity(samples[index], index) for index in indices]
    task_counts = Counter(str(samples[index].get("task_head", "")) for index in indices)
    payload = {
        "schema_version": "v1.2.71_anchor_v1",
        "config": dict(config),
        "anchor_rows": len(indices),
        "anchor_identity_sha256": canonical_sha256(identities),
        "scientific_identity_sha256": sample_scientific_identity_sha256(samples, indices),
        "task_counts": dict(sorted(task_counts.items())),
        "split_parts": sorted({str(samples[index].get("split_part", "")) for index in indices}),
        "sample_identities": identities,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def capture_trajectory_snapshot(
    model: Any,
    dataset: AggregatedTaskDataset,
    samples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    out_path: Path,
    phase: str,
    epoch: int,
    global_epoch: int,
    batch_size: int,
    device: Any,
    target_scaler: TargetScaler | None,
) -> dict[str, Any]:
    """Capture latent/prediction/embedding state and restore all training state."""

    import torch
    from torch.utils.data import DataLoader

    if not indices:
        raise ValueError("Trajectory capture requires a non-empty anchor.")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    was_training = bool(model.training)
    latents: list[np.ndarray] = []
    scaled_predictions: list[np.ndarray] = []
    try:
        model.eval()
        loader = DataLoader(
            _IndexDataset(dataset, list(indices)),
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            collate_fn=collate_aggregated_task_batch,
            **dataloader_runtime_options(device, 0),
        )
        with torch.no_grad():
            for batch in loader:
                molecular_numeric = batch["molecular_numeric"].to(device, non_blocking=True)
                fingerprint = batch["fingerprint"].to(device, non_blocking=True)
                molecular_graph = graph_to_device(batch.get("molecular_graph"), device, non_blocking=True)
                categorical_ids = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch["categorical_ids"].items()
                }
                adapter_ids = batch.get("adapter_id")
                adapter_ids = adapter_ids.to(device, non_blocking=True) if adapter_ids is not None else None
                kwargs: dict[str, Any] = {"adapter_ids": adapter_ids}
                if molecular_graph is not None:
                    kwargs["molecular_graph"] = molecular_graph
                shared = model.encode_shared(
                    molecular_numeric,
                    fingerprint,
                    categorical_ids,
                    **kwargs,
                )
                outputs = model(molecular_numeric, fingerprint, categorical_ids, **kwargs)
                routed = torch.empty(len(batch["task_head"]), dtype=shared.dtype, device=shared.device)
                for row_index, task_head in enumerate(batch["task_head"]):
                    if task_head not in outputs:
                        raise ValueError(f"Trajectory output is missing task head: {task_head}")
                    routed[row_index] = outputs[task_head][row_index]
                latents.append(shared.detach().cpu().numpy().astype(np.float32, copy=False))
                scaled_predictions.append(routed.detach().cpu().numpy().astype(np.float32, copy=False))
        latent = np.concatenate(latents, axis=0)
        y_pred_scaled = np.concatenate(scaled_predictions, axis=0)
        y_pred = y_pred_scaled.astype(np.float64, copy=True)
        if target_scaler is not None:
            for row_index, sample_index in enumerate(indices):
                scale_key = str(samples[sample_index].get("target_scale_key", GLOBAL_TARGET_SCALE_KEY))
                y_pred[row_index] = target_scaler.inverse_transform(scale_key, float(y_pred_scaled[row_index]))
        arrays: dict[str, Any] = {
            "latent": latent,
            "y_pred_scaled": y_pred_scaled,
            "y_pred": y_pred.astype(np.float32),
            "sample_identity": np.asarray(
                [_trajectory_sample_identity(samples[index], index) for index in indices], dtype=str
            ),
            "task_head": np.asarray([str(samples[index].get("task_head", "")) for index in indices], dtype=str),
            "base_task_head": np.asarray(
                [str(samples[index].get("base_task_head", samples[index].get("task_head", ""))) for index in indices],
                dtype=str,
            ),
            "split_part": np.asarray([str(samples[index].get("split_part", "")) for index in indices], dtype=str),
            "target_family": np.asarray([str(samples[index].get("target_family", "")) for index in indices], dtype=str),
            "y_true": np.asarray(
                [float(samples[index].get("target_value_raw", samples[index].get("target_value", np.nan))) for index in indices],
                dtype=np.float32,
            ),
        }
        for field_name, embedding in model.embeddings.items():
            arrays[f"embedding__{field_name}"] = embedding.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"category_id__{field_name}"] = np.asarray(
                [int((samples[index].get("categorical_ids") or {}).get(field_name, 0)) for index in indices],
                dtype=np.int64,
            )
        np.savez_compressed(out_path, **arrays)
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    return {
        "phase": str(phase),
        "epoch": int(epoch),
        "global_epoch": int(global_epoch),
        "path": str(out_path),
        "rows": len(indices),
        "latent_dim": int(model.fusion_output_dim),
        "sha256": file_sha256(out_path),
    }


def summarize_trajectory_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"snapshot_count": 0, "milestones": [], "trajectory_information": "disabled"}
    ordered = sorted(snapshots, key=lambda row: (int(row["global_epoch"]), str(row["phase"])))
    by_epoch = {int(row["global_epoch"]): dict(row) for row in ordered}
    target_epochs = [int(round(float(best_epoch) * fraction)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
    available_epochs = sorted(by_epoch)
    milestones: list[dict[str, Any]] = []
    used_epochs: set[int] = set()
    for percentage, target_epoch in zip((0, 25, 50, 75, 100), target_epochs):
        resolved_epoch = min(available_epochs, key=lambda value: (abs(value - target_epoch), value))
        if resolved_epoch in used_epochs:
            continue
        used_epochs.add(resolved_epoch)
        milestones.append({"percentage": percentage, "target_epoch": target_epoch, **by_epoch[resolved_epoch]})
    return {
        "snapshot_count": len(ordered),
        "snapshots": [dict(row) for row in ordered],
        "best_epoch": int(best_epoch),
        "milestones": milestones,
        "milestone_duplicates_removed": 5 - len(milestones),
        "trajectory_information": "insufficient" if int(best_epoch) < 4 else "sufficient",
    }


def predict_all(
    model: Any,
    dataset: AggregatedTaskDataset,
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    device: Any,
    num_workers: int = 0,
    target_scaler: TargetScaler | None = None,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_aggregated_task_batch,
        **dataloader_runtime_options(device, num_workers),
    )
    model.eval()
    predictions: list[dict[str, Any]] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            molecular_numeric = batch["molecular_numeric"].to(device, non_blocking=True)
            fingerprint = batch["fingerprint"].to(device, non_blocking=True)
            molecular_graph = graph_to_device(batch.get("molecular_graph"), device, non_blocking=True)
            categorical_ids = {key: value.to(device, non_blocking=True) for key, value in batch["categorical_ids"].items()}
            adapter_ids = batch.get("adapter_id")
            adapter_ids = adapter_ids.to(device, non_blocking=True) if adapter_ids is not None else None
            model_kwargs = {"adapter_ids": adapter_ids}
            if molecular_graph is not None:
                model_kwargs["molecular_graph"] = molecular_graph
            outputs = model(molecular_numeric, fingerprint, categorical_ids, **model_kwargs)
            task_heads = list(batch["task_head"])
            targets = batch["target_value"].tolist()
            if not task_heads:
                continue
            indices_by_head: dict[str, list[int]] = {}
            for idx, model_head in enumerate(task_heads):
                indices_by_head.setdefault(model_head, []).append(idx)
            first_head = task_heads[0]
            if first_head not in outputs:
                raise ValueError(f"Prediction output is missing task head: {first_head}")
            first_output = outputs[first_head]
            if first_output.ndim != 1 or int(first_output.shape[0]) != len(task_heads):
                raise ValueError(
                    "Prediction task-head output must be one-dimensional and batch aligned: "
                    f"head={first_head}, shape={tuple(first_output.shape)}, batch={len(task_heads)}"
                )
            selected_predictions = torch.empty_like(first_output)
            written = torch.zeros(len(task_heads), dtype=torch.bool, device=first_output.device)
            for model_head, head_indices in indices_by_head.items():
                if model_head not in outputs:
                    raise ValueError(f"Prediction output is missing task head: {model_head}")
                head_output = outputs[model_head]
                if head_output.ndim != 1 or int(head_output.shape[0]) != len(task_heads):
                    raise ValueError(
                        "Prediction task-head output must be one-dimensional and batch aligned: "
                        f"head={model_head}, shape={tuple(head_output.shape)}, batch={len(task_heads)}"
                    )
                index_tensor = torch.as_tensor(head_indices, dtype=torch.long, device=device)
                selected_predictions.index_copy_(
                    0,
                    index_tensor,
                    head_output.index_select(0, index_tensor),
                )
                written.index_fill_(0, index_tensor, True)
            if not bool(written.all().item()):
                raise ValueError("Prediction routing did not assign every row exactly once.")
            scaled_predictions = selected_predictions.detach().cpu().tolist()
            for row_idx, model_head in enumerate(task_heads):
                sample_meta = samples[cursor + row_idx]
                y_pred_scaled = float(scaled_predictions[row_idx])
                scale_key = str(sample_meta.get("target_scale_key", GLOBAL_TARGET_SCALE_KEY))
                y_pred = (
                    y_pred_scaled
                    if target_scaler is None
                    else target_scaler.inverse_transform(scale_key, y_pred_scaled)
                )
                y_true = float(sample_meta.get("target_value_raw", targets[row_idx]))
                prediction_row = {
                    column: sample_meta.get(column, "")
                    for column in PREDICTION_METADATA_COLUMNS
                    if column in sample_meta
                }
                prediction_row.update(
                    {
                        "split_part": sample_meta["split_part"],
                        "task_head": sample_meta.get("base_task_head", model_head),
                        "model_head": model_head,
                        "target_name": sample_meta.get("target_name", ""),
                        "medium_domain": sample_meta.get("medium_domain", ""),
                        "y_true": y_true,
                        "y_pred": float(y_pred),
                        "y_true_scaled": float(targets[row_idx]),
                        "y_pred_scaled": y_pred_scaled,
                        "target_standardization": "" if target_scaler is None else target_scaler.mode,
                        "residual": y_true - float(y_pred),
                        "abs_error": abs(y_true - float(y_pred)),
                    }
                )
                predictions.append(
                    prediction_row
                )
            cursor += len(task_heads)
    return predictions


def predict_with_feature_perturbation(
    model: Any,
    dataset: AggregatedTaskDataset,
    samples: list[dict[str, Any]],
    *,
    split_parts: tuple[str, ...],
    replicates: int,
    numeric_noise_std: float,
    seed: int,
    batch_size: int,
    device: Any,
    target_scaler: TargetScaler | None = None,
) -> list[dict[str, Any]]:
    selected_parts = {part.lower() for part in split_parts}
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if str(sample.get("split_part", "")).lower() in selected_parts
    ]
    if not indices:
        return []
    noisy_dataset = _NoisyIndexDataset(
        dataset,
        indices,
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        target_noise_std=0.0,
        seed=seed,
    )
    noisy_samples: list[dict[str, Any]] = []
    for noisy_idx in range(len(noisy_dataset)):
        source_index, replicate_id = noisy_dataset.source(noisy_idx)
        sample = dict(samples[source_index])
        sample["perturbation_replicate"] = replicate_id
        sample["perturbation_numeric_noise_std"] = float(numeric_noise_std)
        noisy_samples.append(sample)
    return predict_all(
        model,
        noisy_dataset,
        noisy_samples,
        batch_size=batch_size,
        device=device,
        target_scaler=target_scaler,
    )


def summarize_perturbation_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (
            str(row.get("sample_id", "")),
            str(row.get("split_part", "")),
            str(row.get("task_head", "")),
            str(row.get("target_name", "")),
            str(row.get("medium_domain", "")),
            effect_level_label(row.get("effect_level_x", "")),
        )
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (sample_id, split_part, task_head, target_name, medium_domain, effect_level_x), items in sorted(grouped.items()):
        y_true = float(items[0]["y_true"])
        values = np.array([float(item["y_pred"]) for item in items], dtype=float)
        rows.append(
            {
                "sample_id": sample_id,
                "split_part": split_part,
                "task_head": task_head,
                "target_name": target_name,
                "medium_domain": medium_domain,
                "effect_level_x": effect_level_x,
                "n_replicates": len(items),
                "y_true": y_true,
                "y_pred_mean": float(np.mean(values)),
                "y_pred_std": float(np.std(values)),
                "abs_error_mean_prediction": abs(y_true - float(np.mean(values))),
                "mean_abs_error_across_replicates": float(np.mean(np.abs(y_true - values))),
            }
        )
    return rows


def perturbation_mean_predictions(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "split_part": row["split_part"],
            "task_head": row["task_head"],
            "target_name": row["target_name"],
            "medium_domain": row["medium_domain"],
            "effect_level_x": row.get("effect_level_x", ""),
            "y_true": row["y_true"],
            "y_pred": row["y_pred_mean"],
        }
        for row in summary_rows
    ]


def metrics_by_group(
    predictions: list[dict[str, Any]],
    *,
    huber_delta: float,
    group_columns: tuple[str, ...] = ("split_part", "task_head", "target_name", "medium_domain"),
    min_n_for_summary: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for row in predictions:
        key = tuple(metric_group_value(row, column) for column in group_columns)
        bucket = grouped.setdefault(key, {"y_true": [], "y_pred": []})
        bucket["y_true"].append(float(row["y_true"]))
        bucket["y_pred"].append(float(row["y_pred"]))
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        metrics = regression_metrics(values["y_true"], values["y_pred"], huber_delta=huber_delta)
        n = len(values["y_true"])
        row = {column: value for column, value in zip(group_columns, key)}
        row.update({"n": n, **metrics})
        row.update(metric_summary_status(row, min_n_for_summary=min_n_for_summary))
        rows.append(row)
    return rows


def metric_group_value(row: Mapping[str, Any], column: str) -> str:
    if column == "effect_level_x":
        return effect_level_label(row.get("effect_level_x", ""))
    return str(row.get(column, ""))


def effect_level_label(value: Any) -> str:
    number = optional_float(value)
    if number is None:
        return "none"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def metric_summary_status(row: Mapping[str, Any], *, min_n_for_summary: int) -> dict[str, Any]:
    n = int(float(row.get("n", 0) or 0))
    r2 = optional_float(row.get("r2"))
    if n < int(min_n_for_summary):
        return {
            "metric_valid_for_summary": 0,
            "metric_exclusion_reason": f"n_below_{int(min_n_for_summary)}",
        }
    if r2 is None:
        return {
            "metric_valid_for_summary": 0,
            "metric_exclusion_reason": "undefined_r2",
        }
    return {"metric_valid_for_summary": 1, "metric_exclusion_reason": ""}


def filter_summary_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("metric_valid_for_summary", 0) or 0) == 1]


def build_toxicity_bin_boundary_audit_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], int] = {}
    for row in predictions:
        key = (
            str(row.get("split_part", "")),
            str(row.get("medium_domain", "")),
            str(row.get("unit_family_v2", "")),
            str(row.get("toxicity_bin_status", "")),
            str(row.get("toxicity_bin_boundary_flag", "")),
        )
        grouped[key] = grouped.get(key, 0) + 1
    rows = [
        {
            "split_part": split_part,
            "medium_domain": medium_domain,
            "unit_family_v2": unit_family,
            "toxicity_bin_status": status,
            "toxicity_bin_boundary_flag": boundary_flag,
            "n": count,
        }
        for (split_part, medium_domain, unit_family, status, boundary_flag), count in sorted(grouped.items())
    ]
    return rows or [{"n": 0}]


def build_split_medium_audit_rows(frame: Any, *, split_join_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "split_part" in frame.columns and "medium_domain" in frame.columns:
        counts = frame.groupby(["split_part", "medium_domain"], dropna=False).size().reset_index(name="n")
        for _, item in counts.iterrows():
            rows.append(
                {
                    "audit_section": "loaded_rows_by_split_medium",
                    "split_part": clean_metadata_value(item["split_part"]),
                    "medium_domain": clean_metadata_value(item["medium_domain"]),
                    "n": int(item["n"]),
                }
            )
    removed_rows = split_join_audit.get("removed_rows", []) if isinstance(split_join_audit, Mapping) else []
    for item in removed_rows:
        rows.append({"audit_section": "removed_by_split_medium_contract", **dict(item)})
    if not rows:
        rows.append({"audit_section": "loaded_rows_by_split_medium", "n": int(len(frame))})
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
