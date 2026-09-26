from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Iterable

from qsar_tl.data.task_mapping import TaskMappingResult, map_task_head


TASK_COLUMNS = [
    "task_head",
    "task_group",
    "task_family",
    "effect_family",
    "effect_level_x",
    "duration_bin_h",
    "duration_bin_rule",
    "task_status",
    "task_excluded_reason",
]

AGGREGATED_TASK_COLUMNS = [
    "aggregate_id",
    "cas_number",
    "dtxsid",
    "chemical_name",
    "smiles",
    "species_number",
    "latin_name",
    "common_name",
    "task_head",
    "task_group",
    "task_family",
    "effect_family",
    "effect_level_x",
    "target_name",
    "target_family",
    "target_basis",
    "value_quality",
    "unit_family_v2",
    "standard_unit_v2",
    "standard_value_mg_l",
    "standard_value_mol_l",
    "standard_value_mg_kg",
    "standard_value_g_ha",
    "standard_value_mg_kg_diet",
    "standard_value_mg_kg_bw_day",
    "unit_conversion_source",
    "unit_conversion_confidence",
    "unit_conversion_note",
    "conversion_path",
    "active_ingredient_basis",
    "acid_equivalent_basis",
    "medium_domain",
    "primary_medium_domain",
    "medium_domains",
    "medium_domain_detail",
    "medium_domain_reason",
    "medium_conflict_flag",
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
    "duration_bin_h",
    "duration_bin_rule",
    "target_value_median",
    "target_value_mean",
    "target_value_std",
    "target_value_count",
    "target_value_min",
    "target_value_max",
    "result_ids",
]

AGGREGATION_KEY_COLUMNS = [
    # Current aggregation behavior:
    # Records are merged only when every key below is identical. This keeps
    # EC10/EC20/EC50 separate because effect_level_x is part of the key, and
    # keeps target scales separate because target_name/target_basis/unit fields
    # are also part of the key.
    #
    # Scientific risk to audit before formal soil modeling:
    # reference_number, test_id, and publication_year are not currently part of
    # this key. Therefore, records from different papers or test batches can be
    # merged when the chemical/species/endpoint/medium/duration descriptors
    # match. The redesigned pipeline should add reference-aware QC, year-based
    # weighting, and cross-reference conflict flags before aggregation.
    "cas_number",
    "dtxsid",
    "chemical_name",
    "smiles",
    "species_number",
    "latin_name",
    "common_name",
    "task_head",
    "task_group",
    "task_family",
    "effect_family",
    "effect_level_x",
    "target_name",
    "target_family",
    "target_basis",
    "value_quality",
    "unit_family_v2",
    "standard_unit_v2",
    "standard_value_mg_l",
    "standard_value_mol_l",
    "standard_value_mg_kg",
    "standard_value_g_ha",
    "standard_value_mg_kg_diet",
    "standard_value_mg_kg_bw_day",
    "unit_conversion_source",
    "unit_conversion_confidence",
    "unit_conversion_note",
    "conversion_path",
    "active_ingredient_basis",
    "acid_equivalent_basis",
    "medium_domain",
    "primary_medium_domain",
    "medium_domains",
    "medium_domain_detail",
    "medium_domain_reason",
    "medium_conflict_flag",
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
    "duration_bin_h",
    "duration_bin_rule",
]


@dataclass
class AggregationBucket:
    key: tuple[object, ...]
    values: list[float] = field(default_factory=list)
    result_ids: list[object] = field(default_factory=list)


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    if not rows:
        raise ValueError(f"Table not found or empty schema: {table_name}")
    return [row[1] for row in rows]


def create_table(conn: sqlite3.Connection, table_name: str, columns: Iterable[str]) -> None:
    column_sql = ", ".join(f'"{column}"' for column in columns)
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({column_sql})')


def insert_rows(conn: sqlite3.Connection, table_name: str, columns: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    values = [[row.get(column) for column in columns] for row in rows]
    conn.executemany(
        f'INSERT INTO "{table_name}" ({column_sql}) VALUES ({placeholders})',
        values,
    )


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def duration_bin_hours(duration_h: object) -> tuple[float | None, str]:
    duration = parse_float(duration_h)
    if duration is None:
        return None, "missing_duration"
    if abs(duration) >= 24.0:
        return round(duration * 2.0) / 2.0, "round_0.5h_ge_24h"
    if abs(duration) >= 1.0:
        return round(duration, 1), "round_0.1h_1_to_24h"
    return round(duration, 2), "round_0.01h_lt_1h"


def task_record_from_target(row: sqlite3.Row, source_columns: list[str]) -> dict[str, object]:
    mapping = map_task_head(
        endpoint=row["endpoint"],
        effect=row["effect"],
        measurement=row["measurement"],
        target_name=row["target_name"],
        target_basis=row["target_basis"],
    )
    duration_bin, duration_rule = duration_bin_hours(row["exposure_duration_mean_h"])
    output = {column: row[column] for column in source_columns}
    output.update(_mapping_to_row(mapping, duration_bin, duration_rule))
    return output


def _mapping_to_row(
    mapping: TaskMappingResult,
    duration_bin: float | None,
    duration_rule: str,
) -> dict[str, object]:
    return {
        "task_head": mapping.task_head,
        "task_group": mapping.task_group,
        "task_family": mapping.task_family,
        "effect_family": mapping.effect_family,
        "effect_level_x": mapping.effect_level_x,
        "duration_bin_h": duration_bin,
        "duration_bin_rule": duration_rule,
        "task_status": mapping.task_status,
        "task_excluded_reason": mapping.task_excluded_reason,
    }


def build_task_records(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    batch_size: int = 10000,
) -> tuple[list[str], dict[str, int]]:
    source_columns = table_columns(conn, "target_records")
    output_columns = source_columns + TASK_COLUMNS
    create_table(conn, "task_records", output_columns)

    stats = {
        "source_included_targets": 0,
        "task_records": 0,
        "included_task_records": 0,
        "excluded_task_records": 0,
    }

    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    cursor = conn.execute(
        f"""
        SELECT *
        FROM target_records
        WHERE target_status = 'included'
        ORDER BY result_id
        {limit_clause}
        """
    )

    while True:
        fetched = cursor.fetchmany(batch_size)
        if not fetched:
            break

        rows = [task_record_from_target(row, source_columns) for row in fetched]
        insert_rows(conn, "task_records", output_columns, rows)

        stats["source_included_targets"] += len(rows)
        stats["task_records"] += len(rows)
        for row in rows:
            if row["task_status"] == "included":
                stats["included_task_records"] += 1
            else:
                stats["excluded_task_records"] += 1
        conn.commit()

    return output_columns, stats


def aggregate_task_records(conn: sqlite3.Connection) -> dict[str, int]:
    # This is the first-pass strict-equality aggregation used by the current
    # experiments. It intentionally does not perform outlier removal, reference
    # weighting, or conflict resolution. The redesign plan replaces this with a
    # QC-first aggregation step while keeping this function as a reproducible
    # baseline for comparing old and new sample counts.
    create_table(conn, "aggregated_task_records", AGGREGATED_TASK_COLUMNS)

    buckets: dict[tuple[object, ...], AggregationBucket] = {}
    cursor = conn.execute(
        """
        SELECT *
        FROM task_records
        WHERE task_status = 'included'
          AND target_value IS NOT NULL
        ORDER BY result_id
        """
    )
    for row in cursor:
        key = tuple(row[column] if column in row.keys() else None for column in AGGREGATION_KEY_COLUMNS)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = AggregationBucket(key=key)
            buckets[key] = bucket

        value = parse_float(row["target_value"])
        if value is None:
            continue
        bucket.values.append(value)
        bucket.result_ids.append(row["result_id"])

    rows: list[dict[str, object]] = []
    for aggregate_id, bucket in enumerate(buckets.values(), start=1):
        values = bucket.values
        if not values:
            continue
        count = len(values)
        mean_value = sum(values) / count
        if count > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / (count - 1)
            std_value = sqrt(variance)
        else:
            std_value = 0.0

        row = dict(zip(AGGREGATION_KEY_COLUMNS, bucket.key, strict=True))
        row.update(
            {
                "aggregate_id": aggregate_id,
                "target_value_median": median(values),
                "target_value_mean": mean_value,
                "target_value_std": std_value,
                "target_value_count": count,
                "target_value_min": min(values),
                "target_value_max": max(values),
                "result_ids": json.dumps(bucket.result_ids, ensure_ascii=False),
            }
        )
        rows.append(row)

    insert_rows(conn, "aggregated_task_records", AGGREGATED_TASK_COLUMNS, rows)
    conn.commit()
    return {
        "aggregated_task_records": len(rows),
        "aggregated_source_records": sum(len(bucket.values) for bucket in buckets.values()),
    }


def write_task_manifest(conn: sqlite3.Connection, stats: dict[str, int], *, db_path: Path, limit: int | None) -> None:
    conn.execute('DROP TABLE IF EXISTS "task_build_manifest"')
    conn.execute('CREATE TABLE "task_build_manifest" ("key", "value")')
    conn.executemany(
        'INSERT INTO "task_build_manifest" ("key", "value") VALUES (?, ?)',
        [
            ("db", str(db_path)),
            ("limit", "" if limit is None else str(limit)),
            ("stats_json", json.dumps(stats, ensure_ascii=False, sort_keys=True)),
        ],
    )
    conn.commit()


def build_task_tables(
    db: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 10000,
) -> dict[str, int]:
    db_path = Path(db)
    stats: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        build_stats = build_task_records(conn, limit=limit, batch_size=batch_size)[1]
        stats.update(build_stats)
        stats.update(aggregate_task_records(conn))
        write_task_manifest(conn, stats, db_path=db_path, limit=limit)
    return stats


def summarize_task_table(db: str | Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(db) as conn:
        return conn.execute(
            """
            SELECT task_status, task_head, task_excluded_reason, COUNT(*) AS n
            FROM task_records
            GROUP BY task_status, task_head, task_excluded_reason
            ORDER BY n DESC
            """
        ).fetchall()
