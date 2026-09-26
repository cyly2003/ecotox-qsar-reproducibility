from __future__ import annotations

import argparse
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.data.modeling_tables import build_modeling_tables, summarize_target_table
from qsar_tl.data.task_tables import build_task_tables, summarize_task_table
from qsar_tl.evaluation.splits import MediumTransferRules, generate_and_write_experiment_splits, generate_and_write_split
from qsar_tl.training.runner import build_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECOTOX-QSAR transfer learning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate an experiment config")
    validate.add_argument("--config", required=True, help="Path to YAML or JSON config")

    run = subparsers.add_parser("run", help="Run an experiment through the configured executor")
    run.add_argument("--config", required=True, help="Path to YAML or JSON config")
    run.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    build_tables = subparsers.add_parser(
        "build-modeling-tables",
        help="Build wide_records and target_records from the configured SQLite database",
    )
    build_tables.add_argument("--config", required=True, help="Path to YAML or JSON config")
    build_tables.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")

    build_tasks = subparsers.add_parser(
        "build-task-tables",
        help="Build task_records and aggregated_task_records in the configured derived SQLite database",
    )
    build_tasks.add_argument("--config", required=True, help="Path to YAML or JSON config")
    build_tasks.add_argument("--db", default=None, help="Override derived SQLite database path")
    build_tasks.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")

    build_qc_tasks = subparsers.add_parser(
        "build-qc-task-tables",
        help="Build task_records_qc and aggregated_task_records_qc with outlier QC and reference-aware aggregation",
    )
    build_qc_tasks.add_argument("--config", required=True, help="Path to YAML or JSON config")
    build_qc_tasks.add_argument("--db", default=None, help="Override derived SQLite database path")
    build_qc_tasks.add_argument("--source-table", default="task_records")
    build_qc_tasks.add_argument("--qc-task-table", default="task_records_qc")
    build_qc_tasks.add_argument("--aggregate-table", default="aggregated_task_records_qc")
    build_qc_tasks.add_argument("--min-outlier-group-n", type=int, default=50)
    build_qc_tasks.add_argument("--robust-z-threshold", type=float, default=4.0)
    build_qc_tasks.add_argument("--conflict-threshold-log-unit", type=float, default=0.3)

    feature_cache = subparsers.add_parser(
        "build-molecule-feature-cache",
        help="Build RDKit descriptor + Morgan fingerprint cache from unique SMILES",
    )
    feature_cache.add_argument("--config", required=True, help="Path to YAML or JSON config")
    feature_cache.add_argument("--db", default=None, help="Override derived SQLite database path")
    feature_cache.add_argument("--source-table", default="aggregated_task_records", help="Source table containing SMILES")
    feature_cache.add_argument("--out", default=None, help="Output JSONL cache path")
    feature_cache.add_argument("--limit", type=int, default=None)

    split = subparsers.add_parser("generate-split", help="Generate split_assignments in a derived SQLite database")
    split.add_argument("--config", required=True, help="Path to YAML or JSON config")
    split.add_argument("--db", default=None, help="Override derived SQLite database path")
    split.add_argument("--source-table", default=None, help="Override modeling source table")
    split.add_argument("--split-name", required=True, help="Name for this split assignment set")
    split.add_argument(
        "--split-type",
        required=True,
        choices=[
            "random_split",
            "chemical_group_split",
            "species_group_split",
            "chemical_species_group_split",
            "medium_transfer_split",
        ],
    )
    split.add_argument("--seed", type=int, default=None)
    split.add_argument("--limit", type=int, default=None)
    split.add_argument("--medium-train-domains", default=None, help="Comma-separated train domains for medium_transfer_split")
    split.add_argument("--medium-test-domains", default=None, help="Comma-separated test domains for medium_transfer_split")
    split.add_argument("--medium-unknown-part", default="valid", choices=["train", "valid", "test", "excluded"])

    experiment_splits = subparsers.add_parser(
        "generate-experiment-splits",
        help="Generate agreed A-F experiment splits in a derived SQLite database",
    )
    experiment_splits.add_argument("--config", required=True, help="Path to YAML or JSON config")
    experiment_splits.add_argument("--db", default=None, help="Override derived SQLite database path")
    experiment_splits.add_argument("--source-table", default=None, help="Override modeling source table")
    experiment_splits.add_argument("--split-name-prefix", default="", help="Prefix prepended to generated split names")
    experiment_splits.add_argument(
        "--split-codes",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        help="Experiment split codes to generate, e.g. A B C D E F",
    )
    experiment_splits.add_argument("--seed", type=int, default=None)
    experiment_splits.add_argument("--limit", type=int, default=None)

    baseline = subparsers.add_parser("run-baseline", help="Run a scikit-learn baseline on a saved split")
    baseline.add_argument("--config", required=True, help="Path to YAML or JSON config")
    baseline.add_argument("--db", default=None, help="Override derived SQLite database path")
    baseline.add_argument("--source-table", default=None, help="Override modeling source table")
    baseline.add_argument("--split-name", required=True)
    baseline.add_argument(
        "--model",
        default="random_forest",
        choices=[
            "random_forest",
            "rf",
            "xgboost",
            "xgb",
            "lightgbm",
            "lgbm",
            "pls",
            "extra_trees",
            "extratress",
            "elastic_net",
            "mlp",
            "hist_gradient_boosting",
            "hgb",
        ],
    )
    baseline.add_argument("--out", required=True, help="Output metrics CSV or JSON path")
    baseline.add_argument("--limit", type=int, default=None)
    baseline.add_argument("--seed", type=int, default=None)
    baseline.add_argument("--min-total", type=int, default=None)
    baseline.add_argument("--min-train", type=int, default=None)
    baseline.add_argument("--min-eval", type=int, default=None)

    baseline_matrix = subparsers.add_parser(
        "run-baseline-matrix",
        help="Run configured baseline models over one or more split names",
    )
    baseline_matrix.add_argument("--config", required=True, help="Path to YAML or JSON config")
    baseline_matrix.add_argument("--db", default=None, help="Override derived SQLite database path")
    baseline_matrix.add_argument("--source-table", default=None, help="Override modeling source table")
    baseline_matrix.add_argument("--split-name", action="append", dest="split_names", default=[])
    baseline_matrix.add_argument("--model", action="append", dest="models", default=[])
    baseline_matrix.add_argument("--out-dir", required=True)
    baseline_matrix.add_argument("--limit", type=int, default=None)
    baseline_matrix.add_argument("--seed", type=int, default=None)
    baseline_matrix.add_argument("--continue-on-error", action="store_true")

    ad_report = subparsers.add_parser(
        "build-ad-report",
        help="Build chemical and species application-domain report for a saved split",
    )
    ad_report.add_argument("--config", required=True, help="Path to YAML or JSON config")
    ad_report.add_argument("--db", default=None, help="Override derived SQLite database path")
    ad_report.add_argument("--source-table", default=None, help="Override modeling source table")
    ad_report.add_argument("--split-name", required=True)
    ad_report.add_argument("--out", required=True, help="Output AD CSV path")
    ad_report.add_argument("--limit", type=int, default=None)
    ad_report.add_argument("--molecular-cache", default=None, help="Optional RDKit descriptor/fingerprint cache JSONL")
    ad_report.add_argument("--pca-components", type=int, default=32)
    ad_report.add_argument("--tanimoto-threshold", type=float, default=0.5)
    ad_report.add_argument(
        "--taxon-columns",
        default="kingdom,phylum,class_name,tax_order,family",
        help="Comma-separated taxon levels used for species-domain distance",
    )
    ad_report.add_argument("--taxon-similarity-threshold", type=float, default=0.8)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "validate-config":
        print(f"Config OK: {Path(args.config).resolve()}")
        return

    if args.command == "run":
        runner = build_runner(config)
        runner.run(config_path=Path(args.config), dry_run=args.dry_run)
        return

    if args.command == "build-modeling-tables":
        paths = config.get("paths", {})
        data = config.get("data", {})
        targets = config.get("targets", {}).get("toxicity", {})
        source_db = _required_config_value(paths, "sqlite_db", "paths.sqlite_db")
        output_db = _required_config_value(data, "modeling_tables_db", "data.modeling_tables_db")
        stats = build_modeling_tables(
            source_db=source_db,
            output_db=output_db,
            limit=args.limit,
            batch_size=int(data.get("build_batch_size", 10000)),
            min_dose_groups_for_midpoint=int(targets.get("min_dose_groups_for_midpoint", 3)),
        )
        print(f"Wrote modeling tables: {Path(output_db).resolve()}")
        print(stats)
        for row in summarize_target_table(output_db):
            print(row)
        return

    if args.command == "build-task-tables":
        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        stats = build_task_tables(
            db,
            limit=args.limit,
            batch_size=int(config.get("data", {}).get("build_batch_size", 10000)),
        )
        print(f"Wrote task tables: {Path(db).resolve()}")
        print(stats)
        for row in summarize_task_table(db):
            print(row)
        return

    if args.command == "build-qc-task-tables":
        from qsar_tl.data.qc_aggregation import build_qc_task_tables

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        stats = build_qc_task_tables(
            db,
            source_table=args.source_table,
            qc_task_table=args.qc_task_table,
            aggregate_table=args.aggregate_table,
            min_outlier_group_n=args.min_outlier_group_n,
            robust_z_threshold=args.robust_z_threshold,
            conflict_threshold_log_unit=args.conflict_threshold_log_unit,
        )
        print(f"Wrote QC task tables: {Path(db).resolve()}")
        print(stats)
        return

    if args.command == "build-molecule-feature-cache":
        from qsar_tl.training.deep_experiment import build_molecular_feature_cache

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        fingerprint_size = int(config.get("features", {}).get("molecule", {}).get("morgan_n_bits", 512))
        out_path = args.out or config.get("experiment", {}).get(
            "molecular_feature_cache",
            f"outputs/features/molecular_features_morgan{fingerprint_size}.jsonl",
        )
        manifest = build_molecular_feature_cache(
            db,
            out_path=out_path,
            fingerprint_size=fingerprint_size,
            source_table=args.source_table,
            limit=args.limit,
        )
        print(manifest)
        return

    if args.command == "generate-split":
        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        summary = generate_and_write_split(
            db,
            split_name=args.split_name,
            split_type=args.split_type,
            seed=seed,
            table_name=args.source_table,
            limit=args.limit,
            medium_rules=_medium_transfer_rules_from_args(args),
        )
        print(f"Wrote split assignments: {Path(db).resolve()}")
        print(summary)
        return

    if args.command == "generate-experiment-splits":
        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        summaries = generate_and_write_experiment_splits(
            db,
            seed=seed,
            table_name=args.source_table,
            limit=args.limit,
            min_records=0 if args.limit is not None else _minimum_modeling_rows(config),
            split_codes=args.split_codes,
            split_name_prefix=args.split_name_prefix,
        )
        print(f"Wrote experiment split assignments: {Path(db).resolve()}")
        for split_name, summary in summaries.items():
            print(split_name, summary)
        return

    if args.command == "run-baseline":
        from qsar_tl.training.baseline import run_baseline

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        result = run_baseline(
            db,
            split_name=args.split_name,
            model_name=args.model,
            out_path=args.out,
            limit=args.limit,
            source_table=args.source_table,
            seed=seed,
            min_total=_task_filter_value(config, args.min_total, "min_total"),
            min_train=_task_filter_value(config, args.min_train, "min_train"),
            min_eval=_task_filter_value(config, args.min_eval, "min_eval"),
        )
        print(f"Wrote baseline report: {result.report_path.resolve()}")
        print(f"Predictions: {result.prediction_count}")
        print(f"Task heads: {', '.join(result.task_heads)}")
        print(f"Max feature count: {result.feature_count}")
        if result.skipped_tasks:
            print(f"Skipped tasks: {result.skipped_tasks}")
        return

    if args.command == "run-baseline-matrix":
        from qsar_tl.training.baseline import run_baseline

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        baseline_cfg = config.get("experiment", {}).get("baselines", {})
        models = args.models or list(baseline_cfg.get("models", ["random_forest"]))
        split_names = args.split_names or _configured_split_names(config)
        if not split_names:
            raise ValueError("No split names supplied and no configured A-F split names found.")
        out_dir = Path(args.out_dir)
        for split_name in split_names:
            for model_name in models:
                out_path = out_dir / split_name / f"{model_name}_metrics.csv"
                try:
                    result = run_baseline(
                        db,
                        split_name=split_name,
                        model_name=model_name,
                        out_path=out_path,
                        limit=args.limit,
                        source_table=args.source_table,
                        seed=seed,
                        huber_delta=float(baseline_cfg.get("huber_delta", 1.0)),
                        min_total=_task_filter_value(config, None, "min_total"),
                        min_train=_task_filter_value(config, None, "min_train"),
                        min_eval=_task_filter_value(config, None, "min_eval"),
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    print(f"[skip] split={split_name} model={model_name}: {exc}", flush=True)
                    continue
                print(
                    f"[ok] split={split_name} model={model_name} "
                    f"predictions={result.prediction_count} out={result.report_path}"
                , flush=True)
        return

    if args.command == "build-ad-report":
        from qsar_tl.evaluation.application_domain import ApplicationDomainConfig, build_application_domain_report

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        feature_cfg = config.get("features", {}).get("molecule", {})
        fingerprint_size = int(feature_cfg.get("morgan_n_bits", 512))
        cache_path = args.molecular_cache or config.get("experiment", {}).get("molecular_feature_cache")
        result = build_application_domain_report(
            db,
            split_name=args.split_name,
            source_table=args.source_table,
            limit=args.limit,
            out_path=args.out,
            config=ApplicationDomainConfig(
                fingerprint_size=fingerprint_size,
                molecular_cache_path=cache_path,
                pca_components=args.pca_components,
                tanimoto_threshold=args.tanimoto_threshold,
                taxon_columns=_csv_tuple(args.taxon_columns),
                taxon_similarity_threshold=args.taxon_similarity_threshold,
            ),
        )
        print(f"Wrote application-domain report: {result.report_path.resolve()}")
        print(f"Wrote manifest: {result.manifest_path.resolve()}")
        print(
            f"Rows: {result.rows}; train rows: {result.train_rows}; "
            f"Williams h*: {result.williams_critical_h:.6g}; PCA components: {result.pca_components_used}"
        )
        return

    parser.error(f"Unsupported command: {args.command}")


def _required_config_value(section: dict, key: str, dotted_name: str) -> str:
    value = section.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required config value: {dotted_name}")
    return str(value)


def _medium_transfer_rules_from_args(args: argparse.Namespace) -> MediumTransferRules | None:
    if getattr(args, "split_type", "") != "medium_transfer_split":
        return None
    if args.medium_train_domains is None and args.medium_test_domains is None and args.medium_unknown_part == "valid":
        return None
    default_rules = MediumTransferRules()
    train_domains = _csv_tuple(args.medium_train_domains) or default_rules.train_domains
    test_domains = _csv_tuple(args.medium_test_domains) or default_rules.test_domains
    return MediumTransferRules(
        train_domains=train_domains,
        test_domains=test_domains,
        unknown_part=args.medium_unknown_part,
    )


def _csv_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _task_filter_value(config: dict, override: int | None, key: str) -> int:
    if override is not None:
        return int(override)
    return int(config.get("experiment", {}).get("task_filter", {}).get(key, 0 if key == "min_total" else 1))


def _minimum_modeling_rows(config: dict) -> int:
    return int(config.get("targets", {}).get("medium_units", {}).get("minimum_modeling_rows", 0))


def _configured_split_names(config: dict) -> list[str]:
    splits = config.get("experiment", {}).get("splits", {})
    names: list[str] = []
    for code in ("A", "B", "C", "F"):
        if code in splits:
            names.append(f"{code}_{splits[code].get('name', '').strip()}")
    for code in ("D", "E"):
        if code in splits:
            base = f"{code}_{splits[code].get('name', '').strip()}"
            folds = int(splits[code].get("folds", 5))
            names.extend(f"{base}_fold{idx}" for idx in range(1, folds + 1))
    return [name for name in names if not name.endswith("_")]


if __name__ == "__main__":
    main()
