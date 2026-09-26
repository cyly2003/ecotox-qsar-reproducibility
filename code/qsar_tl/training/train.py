from __future__ import annotations

import argparse
import re
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.training.deep_experiment import run_deep_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ECOTOX-QSAR transfer model")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--db", default=None, help="Override derived SQLite database path")
    parser.add_argument("--split-name", default=None, help="Split assignment name")
    parser.add_argument("--source-table", default=None, help="Override modeling source table")
    parser.add_argument(
        "--task-filter-min-total",
        type=int,
        default=None,
        help="Override experiment.task_filter.min_total without modifying the base config",
    )
    parser.add_argument(
        "--task-filter-min-train",
        type=int,
        default=None,
        help="Override experiment.task_filter.min_train without modifying the base config",
    )
    parser.add_argument(
        "--task-filter-min-eval",
        type=int,
        default=None,
        help="Override experiment.task_filter.min_eval without modifying the base config",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed for reproducible split sampling")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke training")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay")
    parser.add_argument("--scheduler", default=None, choices=["none", "cosine", "reduce_on_plateau"])
    parser.add_argument("--dropout", type=float, default=None, help="Override model dropout")
    parser.add_argument("--medium-adapters", dest="medium_adapters", action="store_true", default=None)
    parser.add_argument("--no-medium-adapters", dest="medium_adapters", action="store_false")
    parser.add_argument(
        "--target-standardization",
        default=None,
        choices=[
            "none",
            "identity",
            "global",
            "per_task",
            "per_target",
            "per_task_target",
            "per_adapter",
            "per_task_adapter",
        ],
    )
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--run-version", default=None, help="Override experiment.version for the output folder")
    parser.add_argument("--run-name-zh", default=None, help="Override experiment.name_zh for the output folder")
    parser.add_argument("--ablation", default="full", help="Deep model ablation name, default: full")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=None)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument(
        "--monitor-split",
        default=None,
        help=(
            "Split used for stage-1 model selection. Staged transfer runs should use "
            "'internal_train_fraction' so downstream fine-tuning rows remain unseen."
        ),
    )
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--finetune-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-batch-size", type=int, default=None)
    parser.add_argument("--finetune-scheduler", default=None, choices=["none", "cosine", "reduce_on_plateau"])
    parser.add_argument("--finetune-freeze", default=None, choices=["none", "heads_only", "heads_embeddings"])
    parser.add_argument("--finetune-validation-fraction", type=float, default=None)
    parser.add_argument("--finetune-validation-seed", type=int, default=None)
    parser.add_argument("--finetune-mgkg-epochs", type=int, default=None)
    parser.add_argument("--finetune-mgkg-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-mgkg-batch-size", type=int, default=None)
    parser.add_argument("--finetune-mgkg-scheduler", default=None, choices=["none", "cosine", "reduce_on_plateau"])
    parser.add_argument(
        "--finetune-mgkg-freeze",
        default=None,
        choices=["none", "heads_only", "last_trunk", "heads_embeddings"],
    )
    parser.add_argument("--finetune-mgkg-validation-fraction", type=float, default=None)
    parser.add_argument("--finetune-mgkg-validation-seed", type=int, default=None)
    parser.add_argument(
        "--finetune-mgkg-monitor-split",
        default=None,
        help=(
            "Optional explicit split used for stage-3 model selection. This is used by "
            "OOF runs to keep the held-out fold completely outside stage-3 fitting."
        ),
    )
    parser.add_argument(
        "--finetune-mgkg-early-stopping",
        dest="finetune_mgkg_early_stopping",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-finetune-mgkg-early-stopping",
        dest="finetune_mgkg_early_stopping",
        action="store_false",
    )
    parser.add_argument("--finetune-mgkg-head-only-epochs", type=int, default=None)
    parser.add_argument("--finetune-mgkg-trunk-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-mgkg-replay-fraction", type=float, default=None)
    parser.add_argument("--finetune-mgkg-toxicity-bin-loss-weight", type=float, default=None)
    parser.add_argument("--finetune-mgkg-mse-loss-weight", type=float, default=None)
    parser.add_argument(
        "--finetune-mgkg-target-bin-sampling",
        dest="finetune_mgkg_target_bin_sampling",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-finetune-mgkg-target-bin-sampling",
        dest="finetune_mgkg_target_bin_sampling",
        action="store_false",
    )
    parser.add_argument("--finetune-mgkg-target-bins", type=int, default=None)
    parser.add_argument("--finetune-mgkg-sampling-min-weight", type=float, default=None)
    parser.add_argument("--finetune-mgkg-sampling-max-weight", type=float, default=None)
    parser.add_argument(
        "--finetune-mgkg-hierarchical-head",
        dest="finetune_mgkg_hierarchical_head",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-finetune-mgkg-hierarchical-head",
        dest="finetune_mgkg_hierarchical_head",
        action="store_false",
    )
    parser.add_argument("--finetune-mgkg-hierarchical-family-tau", type=float, default=None)
    parser.add_argument("--finetune-mgkg-hierarchical-task-tau", type=float, default=None)
    parser.add_argument("--finetune-mgkg-init-checkpoint", default=None)
    parser.add_argument("--export-finetune-mgkg-init-checkpoint", default=None)
    parser.add_argument(
        "--mgkg-residual-adapter",
        dest="mgkg_residual_adapter",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mgkg-residual-adapter",
        dest="mgkg_residual_adapter",
        action="store_false",
    )
    parser.add_argument("--mgkg-residual-adapter-bottleneck", type=int, default=None)
    parser.add_argument("--head-routing", default=None, choices=["task", "task_target"])
    parser.add_argument(
        "--allow-mixed-target-dimensions",
        dest="allow_mixed_target_dimensions",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-allow-mixed-target-dimensions",
        dest="allow_mixed_target_dimensions",
        action="store_false",
    )
    parser.add_argument("--augment-train-replicates", type=int, default=None)
    parser.add_argument("--augment-finetune-replicates", type=int, default=None)
    parser.add_argument("--augment-numeric-noise-std", type=float, default=None)
    parser.add_argument("--augment-target-noise-std", type=float, default=None)
    parser.add_argument("--test-noise-replicates", type=int, default=None)
    parser.add_argument("--test-noise-numeric-std", type=float, default=None)
    parser.add_argument("--feature-zscore-correction", dest="feature_zscore_correction", action="store_true", default=None)
    parser.add_argument("--no-feature-zscore-correction", dest="feature_zscore_correction", action="store_false")
    parser.add_argument("--feature-zscore-threshold", type=float, default=None)
    parser.add_argument("--metric-min-n", type=int, default=None)
    parser.add_argument(
        "--source-weighting-method",
        default=None,
        choices=[
            "none",
            "tanimoto",
            "tanimoto_to_target",
            "tanimoto_to_finetune",
            "proxy_distance_to_finetune",
            "tanimoto_proxy_to_finetune",
        ],
    )
    parser.add_argument("--source-weighting-alpha", type=float, default=None)
    parser.add_argument(
        "--source-weight-cache-dir",
        default=None,
        help="Optional reusable cache directory keyed by exact source/target identities and feature matrices.",
    )
    parser.add_argument("--effect-level-weighting", dest="effect_level_weighting_enabled", action="store_true", default=None)
    parser.add_argument("--no-effect-level-weighting", dest="effect_level_weighting_enabled", action="store_false")
    parser.add_argument("--effect-level-weighting-beta", type=float, default=None)
    parser.add_argument("--toxicity-binning", dest="toxicity_binning_enabled", action="store_true", default=None)
    parser.add_argument("--no-toxicity-binning", dest="toxicity_binning_enabled", action="store_false")
    parser.add_argument("--toxicity-binning-mode", default=None, choices=["aux_classification", "ordinal", "soft_expert"])
    parser.add_argument("--toxicity-binning-loss-weight", type=float, default=None)
    parser.add_argument("--toxicity-binning-scheme", default=None)
    parser.add_argument("--censored-loss", dest="censored_loss_enabled", action="store_true", default=None)
    parser.add_argument("--no-censored-loss", dest="censored_loss_enabled", action="store_false")
    parser.add_argument("--censored-loss-weight", type=float, default=None)
    parser.add_argument("--censored-loss-margin", type=float, default=None)
    parser.add_argument("--domain-alignment-method", default=None, choices=["none", "coral"])
    parser.add_argument("--domain-alignment-weight", type=float, default=None)
    parser.add_argument("--swa", dest="swa_enabled", action="store_true", default=None)
    parser.add_argument("--no-swa", dest="swa_enabled", action="store_false")
    parser.add_argument("--swa-start-epoch", type=int, default=None)
    parser.add_argument(
        "--swa-phase",
        default=None,
        choices=["pretrain", "finetune", "finetune_mgkg"],
    )
    parser.add_argument(
        "--evaluation-checkpoint",
        type=Path,
        default=None,
        help="Strictly load an existing checkpoint for zero-epoch development-only evaluation.",
    )
    parser.add_argument(
        "--prediction-split-parts",
        nargs="+",
        default=None,
        help=(
            "Optionally write predictions only for these post-routing split parts. "
            "Training and model selection are unchanged."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    task_filter_overrides = {
        "min_total": args.task_filter_min_total,
        "min_train": args.task_filter_min_train,
        "min_eval": args.task_filter_min_eval,
    }
    if any(value is not None for value in task_filter_overrides.values()):
        config = dict(config)
        experiment_cfg = dict(config.get("experiment", {}))
        task_filter_cfg = dict(experiment_cfg.get("task_filter", {}))
        task_filter_cfg.update(
            {key: int(value) for key, value in task_filter_overrides.items() if value is not None}
        )
        experiment_cfg["task_filter"] = task_filter_cfg
        config["experiment"] = experiment_cfg
    if args.medium_adapters is not None:
        config = dict(config)
        model_cfg = dict(config.get("model", {}))
        model_cfg["use_medium_adapters"] = bool(args.medium_adapters)
        config["model"] = model_cfg
    output_dir = Path(args.out_dir or config.get("paths", {}).get("output_dir", "outputs/experiments/default"))
    split_name = args.split_name or config.get("experiment", {}).get("deep_training", {}).get("split_name", "B_random_8_2")
    db_path = args.db or config.get("data", {}).get("modeling_tables_db")
    if not db_path:
        raise ValueError("Missing data.modeling_tables_db or --db.")
    if args.run_version is not None or args.run_name_zh is not None:
        config = dict(config)
        experiment_cfg = dict(config.get("experiment", {}))
        if args.run_version is not None:
            experiment_cfg["version"] = args.run_version
        if args.run_name_zh is not None:
            experiment_cfg["name_zh"] = args.run_name_zh
        config["experiment"] = experiment_cfg
    run_dir = build_run_dir(output_dir, config=config, ablation=args.ablation, split_name=split_name)
    print(f"Loaded config: {Path(args.config).resolve()}")
    print(f"Output directory: {run_dir.resolve()}")
    result = run_deep_experiment(
        db_path,
        split_name=split_name,
        out_dir=run_dir,
        config=config,
        limit=args.limit,
        seed=int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        dropout=args.dropout,
        target_standardization=args.target_standardization,
        device=args.device,
        ablation=args.ablation,
        source_table=args.source_table,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        monitor_split=args.monitor_split,
        finetune_epochs=args.finetune_epochs,
        finetune_learning_rate=args.finetune_learning_rate,
        finetune_batch_size=args.finetune_batch_size,
        finetune_scheduler=args.finetune_scheduler,
        finetune_freeze=args.finetune_freeze,
        finetune_validation_fraction=args.finetune_validation_fraction,
        finetune_validation_seed=args.finetune_validation_seed,
        finetune_mgkg_epochs=args.finetune_mgkg_epochs,
        finetune_mgkg_learning_rate=args.finetune_mgkg_learning_rate,
        finetune_mgkg_batch_size=args.finetune_mgkg_batch_size,
        finetune_mgkg_scheduler=args.finetune_mgkg_scheduler,
        finetune_mgkg_freeze=args.finetune_mgkg_freeze,
        finetune_mgkg_validation_fraction=args.finetune_mgkg_validation_fraction,
        finetune_mgkg_validation_seed=args.finetune_mgkg_validation_seed,
        finetune_mgkg_monitor_split=args.finetune_mgkg_monitor_split,
        finetune_mgkg_early_stopping=args.finetune_mgkg_early_stopping,
        finetune_mgkg_head_only_epochs=args.finetune_mgkg_head_only_epochs,
        finetune_mgkg_trunk_learning_rate=args.finetune_mgkg_trunk_learning_rate,
        finetune_mgkg_replay_fraction=args.finetune_mgkg_replay_fraction,
        finetune_mgkg_toxicity_bin_loss_weight=args.finetune_mgkg_toxicity_bin_loss_weight,
        finetune_mgkg_mse_loss_weight=args.finetune_mgkg_mse_loss_weight,
        finetune_mgkg_target_bin_sampling=args.finetune_mgkg_target_bin_sampling,
        finetune_mgkg_target_bins=args.finetune_mgkg_target_bins,
        finetune_mgkg_sampling_min_weight=args.finetune_mgkg_sampling_min_weight,
        finetune_mgkg_sampling_max_weight=args.finetune_mgkg_sampling_max_weight,
        finetune_mgkg_hierarchical_head=args.finetune_mgkg_hierarchical_head,
        finetune_mgkg_hierarchical_family_tau=args.finetune_mgkg_hierarchical_family_tau,
        finetune_mgkg_hierarchical_task_tau=args.finetune_mgkg_hierarchical_task_tau,
        finetune_mgkg_init_checkpoint=args.finetune_mgkg_init_checkpoint,
        export_finetune_mgkg_init_checkpoint=args.export_finetune_mgkg_init_checkpoint,
        mgkg_residual_adapter=args.mgkg_residual_adapter,
        mgkg_residual_adapter_bottleneck=args.mgkg_residual_adapter_bottleneck,
        head_routing=args.head_routing,
        allow_mixed_target_dimensions=args.allow_mixed_target_dimensions,
        augment_train_replicates=args.augment_train_replicates,
        augment_finetune_replicates=args.augment_finetune_replicates,
        augment_numeric_noise_std=args.augment_numeric_noise_std,
        augment_target_noise_std=args.augment_target_noise_std,
        test_noise_replicates=args.test_noise_replicates,
        test_noise_numeric_std=args.test_noise_numeric_std,
        feature_zscore_correction=args.feature_zscore_correction,
        feature_zscore_threshold=args.feature_zscore_threshold,
        metric_min_n=args.metric_min_n,
        source_weighting_method=args.source_weighting_method,
        source_weighting_alpha=args.source_weighting_alpha,
        source_weight_cache_dir=args.source_weight_cache_dir,
        effect_level_weighting_enabled=args.effect_level_weighting_enabled,
        effect_level_weighting_beta=args.effect_level_weighting_beta,
        toxicity_binning_enabled=args.toxicity_binning_enabled,
        toxicity_binning_mode=args.toxicity_binning_mode,
        toxicity_binning_loss_weight=args.toxicity_binning_loss_weight,
        toxicity_binning_scheme=args.toxicity_binning_scheme,
        censored_loss_enabled=args.censored_loss_enabled,
        censored_loss_weight=args.censored_loss_weight,
        censored_loss_margin=args.censored_loss_margin,
        domain_alignment_method=args.domain_alignment_method,
        domain_alignment_weight=args.domain_alignment_weight,
        swa_enabled=args.swa_enabled,
        swa_start_epoch=args.swa_start_epoch,
        swa_phase=args.swa_phase,
        evaluation_checkpoint=args.evaluation_checkpoint,
        prediction_split_parts=(
            None
            if args.prediction_split_parts is None
            else tuple(args.prediction_split_parts)
        ),
    )
    print(f"Deep training complete: {result.out_dir.resolve()}")
    print(f"Metrics: {result.metrics_path.resolve()}")
    print(f"History: {result.history_path.resolve()}")
    print(f"Encoder source: {result.encoder_source}")
    print(f"Ablation: {result.ablation}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Early stopping: {result.early_stopping_enabled}")
    print(f"Tasks: {', '.join(result.trained_tasks)}")

def build_run_dir(
    output_dir: Path,
    *,
    config: dict,
    ablation: str,
    split_name: str,
) -> Path:
    experiment_cfg = config.get("experiment", {})
    version = str(experiment_cfg.get("version", "v1.0.0")).strip() or "v1.0.0"
    run_name = str(
        experiment_cfg.get(
            "name_zh",
            experiment_cfg.get("run_name_zh", "训练优化重构_默认实验"),
        )
    ).strip()
    group_name = sanitize_path_part(f"{version}_{run_name}")
    return output_dir / group_name / "deep" / sanitize_path_part(ablation) / sanitize_path_part(split_name)


def sanitize_path_part(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip(" ._")
    return text or "unnamed"


if __name__ == "__main__":
    main()
