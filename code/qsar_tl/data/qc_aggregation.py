from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

from qsar_tl.data.medium import classify_exposure_medium
from qsar_tl.data.reference_weighting import (
    parse_publication_year,
    reference_recency_weight,
    weighted_mean,
    weighted_std,
)
from qsar_tl.data.task_tables import AGGREGATION_KEY_COLUMNS, parse_float


def append_missing(columns: Iterable[str], additions: Iterable[str]) -> list[str]:
    result = list(columns)
    present = set(result)
    for column in additions:
        if column not in present:
            result.append(column)
            present.add(column)
    return result


TAXON_COLUMNS = [
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
]

QC_TASK_COLUMNS = [
    "medium_domain",
    "tox_qc_status",
    "tox_qc_reason",
    "robust_z",
    "qc_group_n",
    "reference_weight",
]

QC_AGGREGATION_KEY_COLUMNS = append_missing(
    [
        column
        for column in AGGREGATION_KEY_COLUMNS
        if column
        not in {
            "unit_conversion_source",
            "unit_conversion_confidence",
            "unit_conversion_note",
        }
    ],
    ["medium_domain", *TAXON_COLUMNS],
)

QC_AGGREGATED_TASK_COLUMNS = append_missing(
    [
        "aggregate_id",
        *QC_AGGREGATION_KEY_COLUMNS,
        "target_value_median",
        "target_value_mean",
        "target_value_weighted_mean",
        "target_value_unweighted_median",
        "target_value_std",
        "target_value_weighted_std",
        "target_value_count",
        "target_value_min",
        "target_value_max",
        "aggregation_weight_sum",
        "mean_reference_weight",
        "cross_reference_conflict_flag",
        "cross_reference_range",
        "source_reference_count",
        "source_test_count",
        "publication_year_latest",
        "publication_years",
        "reference_numbers",
        "test_ids",
        "result_ids",
    ],
    TAXON_COLUMNS,
)


@dataclass
class QcRow:
    payload: dict[str, object]
    target_value: float
    medium_domain: str
    qc_group: tuple[str, str, str]
    robust_z: float | None = None
    qc_group_n: int = 0
    tox_qc_status: str = "included"
    tox_qc_reason: str | None = None
    reference_weight: float = 1.0


@dataclass
class QcAggregationBucket:
    key: tuple[object, ...]
    rows: list[QcRow] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceTestUnit:
    reference_number: object
    test_id: object
    publication_year: object
    target_value: float
    reference_weight: float


def build_qc_task_tables(
    db: str | Path,
    *,
    source_table: str = "task_records",
    qc_task_table: str = "task_records_qc",
    aggregate_table: str = "aggregated_task_records_qc",
    min_outlier_group_n: int = 50,
    robust_z_threshold: float = 4.0,
    conflict_threshold_log_unit: float = 0.3,
) -> dict[str, int]:
    db_path = Path(db)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        source_columns = table_columns(conn, source_table)
        qc_rows = load_and_qc_rows(
            conn,
            source_table=source_table,
            source_columns=source_columns,
            min_outlier_group_n=min_outlier_group_n,
            robust_z_threshold=robust_z_threshold,
        )
        write_qc_task_table(conn, qc_task_table, source_columns, qc_rows)
        aggregate_rows = aggregate_qc_rows(
            qc_rows,
            conflict_threshold_log_unit=conflict_threshold_log_unit,
        )
        write_aggregate_table(conn, aggregate_table, aggregate_rows)
        write_qc_manifest(
            conn,
            stats={
                "qc_source_records": len(qc_rows),
                "qc_included_records": sum(row.tox_qc_status == "included" for row in qc_rows),
                "qc_excluded_records": sum(row.tox_qc_status == "excluded" for row in qc_rows),
                "aggregated_task_records_qc": len(aggregate_rows),
            },
            db_path=db_path,
            source_table=source_table,
            qc_task_table=qc_task_table,
            aggregate_table=aggregate_table,
            min_outlier_group_n=min_outlier_group_n,
            robust_z_threshold=robust_z_threshold,
            conflict_threshold_log_unit=conflict_threshold_log_unit,
        )
        conn.commit()
        return {
            "qc_source_records": len(qc_rows),
            "qc_included_records": sum(row.tox_qc_status == "included" for row in qc_rows),
            "qc_excluded_records": sum(row.tox_qc_status == "excluded" for row in qc_rows),
            "aggregated_task_records_qc": len(aggregate_rows),
        }


def load_and_qc_rows(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    source_columns: list[str],
    min_outlier_group_n: int,
    robust_z_threshold: float,
) -> list[QcRow]:
    raw_rows = conn.execute(
        f"""
        SELECT *
        FROM {quote_identifier(source_table)}
        WHERE task_status = 'included'
          AND target_value IS NOT NULL
        ORDER BY result_id
        """
    ).fetchall()
    qc_rows: list[QcRow] = []
    groups: dict[tuple[str, str, str], list[QcRow]] = defaultdict(list)
    for row in raw_rows:
        value = parse_float(row["target_value"])
        if value is None:
            continue
        payload = {column: row[column] for column in source_columns}
        medium_domain = classify_medium_domain(payload)
        group_key = (
            str(payload.get("target_name") or ""),
            medium_domain,
            str(payload.get("task_head") or ""),
        )
        qc_row = QcRow(payload=payload, target_value=value, medium_domain=medium_domain, qc_group=group_key)
        qc_rows.append(qc_row)
        groups[group_key].append(qc_row)

    for rows in groups.values():
        values = [row.target_value for row in rows]
        robust_z = robust_z_scores(values)
        auto_outlier = len(rows) >= min_outlier_group_n
        for row, z_value in zip(rows, robust_z):
            row.robust_z = z_value
            row.qc_group_n = len(rows)
            if auto_outlier and z_value is not None and abs(z_value) > robust_z_threshold:
                row.tox_qc_status = "excluded"
                row.tox_qc_reason = f"robust_z_gt_{robust_z_threshold:g}"
            elif not auto_outlier:
                row.tox_qc_status = "included"
                row.tox_qc_reason = "small_group_manual_audit"
            else:
                row.tox_qc_status = "included"
                row.tox_qc_reason = None

    assign_reference_weights(qc_rows)
    return qc_rows


def assign_reference_weights(rows: list[QcRow]) -> None:
    groups: dict[tuple[object, ...], list[QcRow]] = defaultdict(list)
    for row in rows:
        key = tuple(
            row.medium_domain if column == "medium_domain" else row.payload.get(column)
            for column in QC_AGGREGATION_KEY_COLUMNS
        )
        groups[key].append(row)
    for grouped_rows in groups.values():
        years = [parse_publication_year(row.payload.get("publication_year")) for row in grouped_rows]
        known_years = [year for year in years if year is not None]
        latest_year = max(known_years) if known_years else None
        for row in grouped_rows:
            row.reference_weight = reference_recency_weight(row.payload.get("publication_year"), latest_year=latest_year)


def aggregate_qc_rows(
    rows: list[QcRow],
    *,
    conflict_threshold_log_unit: float,
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], QcAggregationBucket] = {}
    for row in rows:
        if row.tox_qc_status != "included":
            continue
        key = tuple(row.payload.get(column) if column != "medium_domain" else row.medium_domain for column in QC_AGGREGATION_KEY_COLUMNS)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = QcAggregationBucket(key=key)
            buckets[key] = bucket
        bucket.rows.append(row)

    aggregate_rows: list[dict[str, object]] = []
    for aggregate_id, bucket in enumerate(buckets.values(), start=1):
        raw_values = [row.target_value for row in bucket.rows]
        reference_units = build_reference_test_units(bucket.rows)
        unit_values = [unit.target_value for unit in reference_units]
        unit_weights = [unit.reference_weight for unit in reference_units]
        if not raw_values or not unit_values:
            continue
        target_mean = weighted_mean(unit_values, unit_weights)
        value_range = max(unit_values) - min(unit_values)
        reference_numbers = ordered_unique(row.payload.get("reference_number") for row in bucket.rows)
        test_ids = ordered_unique(row.payload.get("test_id") for row in bucket.rows)
        publication_years = ordered_unique(row.payload.get("publication_year") for row in bucket.rows)
        known_years = [year for year in (parse_publication_year(item) for item in publication_years) if year is not None]
        output_row = dict(zip(QC_AGGREGATION_KEY_COLUMNS, bucket.key, strict=True))
        output_row.update(
            {
                "aggregate_id": aggregate_id,
                # Downstream training currently resolves target_value_median.
                # In QC tables this field intentionally carries the weighted
                # reference/test-aware mean so old training code uses the new target.
                "target_value_median": target_mean,
                "target_value_mean": sum(raw_values) / len(raw_values),
                "target_value_weighted_mean": target_mean,
                "target_value_unweighted_median": median(raw_values),
                "target_value_std": sample_std(raw_values),
                "target_value_weighted_std": weighted_std(unit_values, unit_weights, mean_value=target_mean),
                "target_value_count": len(raw_values),
                "target_value_min": min(raw_values),
                "target_value_max": max(raw_values),
                "aggregation_weight_sum": sum(unit_weights),
                "mean_reference_weight": sum(unit_weights) / len(unit_weights),
                "cross_reference_conflict_flag": int(len(reference_numbers) > 1 and value_range > conflict_threshold_log_unit),
                "cross_reference_range": value_range,
                "source_reference_count": len(reference_numbers),
                "source_test_count": len(test_ids),
                "publication_year_latest": max(known_years) if known_years else None,
                "publication_years": json.dumps(publication_years, ensure_ascii=False),
                "reference_numbers": json.dumps(reference_numbers, ensure_ascii=False),
                "test_ids": json.dumps(test_ids, ensure_ascii=False),
                "result_ids": json.dumps([row.payload.get("result_id") for row in bucket.rows], ensure_ascii=False),
            }
        )
        aggregate_rows.append(output_row)
    return aggregate_rows


def build_reference_test_units(rows: list[QcRow]) -> list[ReferenceTestUnit]:
    grouped: dict[tuple[str, str], list[QcRow]] = defaultdict(list)
    for row in rows:
        reference_key = "" if row.payload.get("reference_number") is None else str(row.payload.get("reference_number"))
        test_key = "" if row.payload.get("test_id") is None else str(row.payload.get("test_id"))
        grouped[(reference_key, test_key)].append(row)

    units: list[ReferenceTestUnit] = []
    for grouped_rows in grouped.values():
        values = [row.target_value for row in grouped_rows]
        weights = [row.reference_weight for row in grouped_rows]
        publication_years = [parse_publication_year(row.payload.get("publication_year")) for row in grouped_rows]
        latest_year = max([year for year in publication_years if year is not None], default=None)
        representative = grouped_rows[0]
        units.append(
            ReferenceTestUnit(
                reference_number=representative.payload.get("reference_number"),
                test_id=representative.payload.get("test_id"),
                publication_year=latest_year,
                target_value=median(values),
                reference_weight=sum(weights) / len(weights),
            )
        )
    return units


def write_qc_task_table(
    conn: sqlite3.Connection,
    table_name: str,
    source_columns: list[str],
    rows: list[QcRow],
) -> None:
    output_columns = append_missing(source_columns, QC_TASK_COLUMNS)
    create_table(conn, table_name, output_columns)
    payloads = []
    for row in rows:
        payload = dict(row.payload)
        payload.update(
            {
                "medium_domain": row.medium_domain,
                "tox_qc_status": row.tox_qc_status,
                "tox_qc_reason": row.tox_qc_reason,
                "robust_z": row.robust_z,
                "qc_group_n": row.qc_group_n,
                "reference_weight": row.reference_weight,
            }
        )
        payloads.append(payload)
    insert_rows(conn, table_name, output_columns, payloads)


def write_aggregate_table(conn: sqlite3.Connection, table_name: str, rows: list[dict[str, object]]) -> None:
    create_table(conn, table_name, QC_AGGREGATED_TASK_COLUMNS)
    insert_rows(conn, table_name, QC_AGGREGATED_TASK_COLUMNS, rows)


def write_qc_manifest(
    conn: sqlite3.Connection,
    *,
    stats: dict[str, int],
    db_path: Path,
    source_table: str,
    qc_task_table: str,
    aggregate_table: str,
    min_outlier_group_n: int,
    robust_z_threshold: float,
    conflict_threshold_log_unit: float,
) -> None:
    conn.execute('DROP TABLE IF EXISTS "qc_task_build_manifest"')
    conn.execute('CREATE TABLE "qc_task_build_manifest" ("key", "value")')
    conn.executemany(
        'INSERT INTO "qc_task_build_manifest" ("key", "value") VALUES (?, ?)',
        [
            ("db", str(db_path)),
            ("source_table", source_table),
            ("qc_task_table", qc_task_table),
            ("aggregate_table", aggregate_table),
            ("min_outlier_group_n", str(min_outlier_group_n)),
            ("robust_z_threshold", str(robust_z_threshold)),
            ("conflict_threshold_log_unit", str(conflict_threshold_log_unit)),
            ("stats_json", json.dumps(stats, ensure_ascii=False, sort_keys=True)),
        ],
    )


def robust_z_scores(values: list[float]) -> list[float | None]:
    if not values:
        return []
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad = median(deviations)
    if mad > 1e-12:
        return [0.6745 * (value - center) / mad for value in values]
    std = sample_std(values)
    if std > 1e-12:
        mean_value = sum(values) / len(values)
        return [(value - mean_value) / std for value in values]
    return [0.0 for _ in values]


def classify_medium_domain(row: dict[str, object]) -> str:
    existing = clean_text(row.get("medium_domain"))
    if existing and existing != "unknown":
        return existing
    return classify_exposure_medium(
        organism_habitat=row.get("organism_habitat"),
        media_type=row.get("media_type"),
        target_basis=row.get("target_basis"),
    ).medium_domain


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    if not rows:
        raise ValueError(f"Table not found: {table_name}")
    return [row[1] for row in rows]


def create_table(conn: sqlite3.Connection, table_name: str, columns: Iterable[str]) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)}")
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    conn.execute(f"CREATE TABLE {quote_identifier(table_name)} ({column_sql})")


def insert_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[str],
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    values = [[row.get(column) for column in columns] for row in rows]
    conn.executemany(
        f"INSERT INTO {quote_identifier(table_name)} ({column_sql}) VALUES ({placeholders})",
        values,
    )


def ordered_unique(values: Iterable[object]) -> list[object]:
    seen = set()
    result = []
    for value in values:
        key = "" if value is None else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", "-")


def normalize_media_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    while text.endswith(("/", "*")):
        text = text[:-1]
    return text


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
