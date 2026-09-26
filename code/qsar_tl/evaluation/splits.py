from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 42
DEFAULT_RANDOM_FRACTIONS = (0.8, 0.1, 0.1)
DEFAULT_TABLE_CANDIDATES = ("aggregated_task_records", "target_records")
EXPERIMENT_SPLIT_CODES = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class SplitAssignment:
    split_name: str
    split_part: str
    seed: int
    split_type: str
    source_table: str
    record_id: str | None = None
    aggregate_id: str | None = None
    group_key: str | None = None


@dataclass(frozen=True)
class MediumTransferRules:
    train_domains: tuple[str, ...] = (
        "water",
        "aquatic",
        "freshwater",
        "saltwater",
        "marine",
        "non-soil",
        "non_soil",
        "nonsoil",
        "fw",
        "fw/",
        "sw",
        "sw/",
        "aqu",
        "aqu/",
    )
    test_domains: tuple[str, ...] = (
        "soil",
        "sediment",
        "nat",
        "nat/",
        "art",
        "art/",
        "uks",
        "uks/",
        "lit",
        "lit/",
        "min",
        "min/",
    )
    unknown_part: str = "valid"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')]


def resolve_source_table(conn: sqlite3.Connection, table_name: str | None = None) -> str:
    if table_name:
        if not table_exists(conn, table_name):
            raise ValueError(f"Source table does not exist: {table_name}")
        return table_name
    for candidate in DEFAULT_TABLE_CANDIDATES:
        if table_exists(conn, candidate):
            return candidate
    raise ValueError(
        "No modeling source table found. Expected aggregated_task_records or target_records."
    )


def resolve_id_column(columns: Iterable[str], preferred: str | None = None) -> str:
    available = set(columns)
    if preferred:
        if preferred not in available:
            raise ValueError(f"ID column does not exist: {preferred}")
        return preferred
    for candidate in ("aggregate_id", "record_id", "result_id", "test_id"):
        if candidate in available:
            return candidate
    raise ValueError("No usable ID column found. Expected aggregate_id, record_id, result_id, or test_id.")


def read_split_records(
    conn: sqlite3.Connection,
    *,
    table_name: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    source_table = resolve_source_table(conn, table_name)
    columns = table_columns(conn, source_table)
    resolved_id = resolve_id_column(columns, id_column)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    rows = conn.execute(f'SELECT * FROM "{source_table}" ORDER BY "{resolved_id}"{limit_clause}').fetchall()
    return source_table, resolved_id, [dict(row) for row in rows]


def create_split_assignments_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS split_assignments (
            split_name TEXT NOT NULL,
            record_id TEXT,
            aggregate_id TEXT,
            split_part TEXT NOT NULL,
            seed INTEGER NOT NULL,
            split_type TEXT NOT NULL,
            source_table TEXT NOT NULL,
            group_key TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_split_assignments_name_part
        ON split_assignments (split_name, split_part)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_split_assignments_record
        ON split_assignments (split_name, record_id, aggregate_id)
        """
    )


def write_split_assignments(conn: sqlite3.Connection, assignments: list[SplitAssignment]) -> None:
    if not assignments:
        raise ValueError("No split assignments to write.")
    create_split_assignments_table(conn)
    split_name = assignments[0].split_name
    conn.execute("DELETE FROM split_assignments WHERE split_name = ?", (split_name,))
    conn.executemany(
        """
        INSERT INTO split_assignments (
            split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item.split_name,
                item.record_id,
                item.aggregate_id,
                item.split_part,
                item.seed,
                item.split_type,
                item.source_table,
                item.group_key,
            )
            for item in assignments
        ],
    )


def generate_and_write_split(
    db_path: str | Path,
    *,
    split_name: str,
    split_type: str,
    seed: int = DEFAULT_SEED,
    table_name: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
    random_fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
    medium_rules: MediumTransferRules | None = None,
) -> dict[str, int]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        source_table, resolved_id, records = read_split_records(
            conn,
            table_name=table_name,
            id_column=id_column,
            limit=limit,
        )
        assignments = build_split_assignments(
            records,
            split_name=split_name,
            split_type=split_type,
            source_table=source_table,
            id_column=resolved_id,
            seed=seed,
            random_fractions=random_fractions,
            medium_rules=medium_rules,
        )
        write_split_assignments(conn, assignments)
        conn.commit()
    return summarize_assignments(assignments)


def generate_and_write_experiment_splits(
    db_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    table_name: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
    min_records: int = 0,
    split_codes: Iterable[str] = EXPERIMENT_SPLIT_CODES,
    split_name_prefix: str = "",
) -> dict[str, dict[str, int]]:
    """Generate the agreed A-F experiment split suite.

    A: 7:2:1 train:finetune:test random split, unrestricted chemicals.
    B: 8:2 train:test random split, unrestricted chemicals.
    C: 8:2 train:test strict chemical holdout.
    D: 5-fold CV with chemicals kept in one fold only.
    E: 5-fold random CV.
    F: 7:2:1 train:finetune:test strict chemical grouping.
    """

    requested = tuple(code.strip().upper() for code in split_codes)
    unknown = sorted(set(requested) - set(EXPERIMENT_SPLIT_CODES))
    if unknown:
        raise ValueError(f"Unsupported experiment split code(s): {', '.join(unknown)}")

    summaries: dict[str, dict[str, int]] = {}
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        source_table, resolved_id, records = read_split_records(
            conn,
            table_name=table_name,
            id_column=id_column,
            limit=limit,
        )
        validate_min_records(records, source_table=source_table, min_records=min_records)
        for code in requested:
            split_sets = build_experiment_split_sets(
                records,
                code=code,
                source_table=source_table,
                id_column=resolved_id,
                seed=seed,
            )
            for split_name, assignments in split_sets.items():
                final_name = f"{split_name_prefix}{split_name}"
                final_assignments = (
                    [replace(item, split_name=final_name) for item in assignments]
                    if split_name_prefix
                    else assignments
                )
                write_split_assignments(conn, final_assignments)
                summaries[final_name] = summarize_assignments(final_assignments)
        conn.commit()
    return summaries


def validate_min_records(records: list[dict[str, Any]], *, source_table: str, min_records: int) -> None:
    if min_records <= 0:
        return
    count = len(records)
    if count < min_records:
        raise ValueError(
            f"Source table {source_table!r} has {count} rows, below minimum_modeling_rows={min_records}. "
            "Exclude this low-sample target dimension from formal modeling or lower the threshold explicitly "
            "for an external-observation analysis."
        )


def build_experiment_split_sets(
    records: list[dict[str, Any]],
    *,
    code: str,
    source_table: str,
    id_column: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[SplitAssignment]]:
    normalized = code.strip().upper()
    if normalized == "A":
        return {
            "A_random_adapt_7_2_1": build_custom_split_assignments(
                records,
                split_name="A_random_adapt_7_2_1",
                split_type="random_adapt_split",
                source_table=source_table,
                id_column=id_column,
                parts=assign_random_named_parts(
                    len(records),
                    seed=seed,
                    fractions={"train": 0.7, "finetune": 0.2, "test": 0.1},
                ),
                group_keys=[None] * len(records),
                seed=seed,
            )
        }
    if normalized == "B":
        return {
            "B_random_8_2": build_custom_split_assignments(
                records,
                split_name="B_random_8_2",
                split_type="random_holdout_split",
                source_table=source_table,
                id_column=id_column,
                parts=assign_random_named_parts(
                    len(records),
                    seed=seed,
                    fractions={"train": 0.8, "test": 0.2},
                ),
                group_keys=[None] * len(records),
                seed=seed,
            )
        }
    if normalized == "C":
        parts, group_keys = assign_group_named_parts(
            records,
            ("cas_number",),
            seed=seed,
            fractions={"train": 0.8, "test": 0.2},
        )
        return {
            "C_chemical_holdout_8_2": build_custom_split_assignments(
                records,
                split_name="C_chemical_holdout_8_2",
                split_type="chemical_holdout_split",
                source_table=source_table,
                id_column=id_column,
                parts=parts,
                group_keys=group_keys,
                seed=seed,
            )
        }
    if normalized == "D":
        return build_kfold_split_sets(
            records,
            split_prefix="D_chemical_group_5fold",
            split_type="chemical_group_kfold",
            source_table=source_table,
            id_column=id_column,
            seed=seed,
            n_folds=5,
            group_columns=("cas_number",),
        )
    if normalized == "E":
        return build_kfold_split_sets(
            records,
            split_prefix="E_random_5fold",
            split_type="random_kfold",
            source_table=source_table,
            id_column=id_column,
            seed=seed,
            n_folds=5,
            group_columns=None,
        )
    if normalized == "F":
        parts, group_keys = assign_group_named_parts(
            records,
            ("cas_number",),
            seed=seed,
            fractions={"train": 0.7, "finetune": 0.2, "test": 0.1},
        )
        return {
            "F_chemical_adapt_7_2_1": build_custom_split_assignments(
                records,
                split_name="F_chemical_adapt_7_2_1",
                split_type="chemical_adapt_split",
                source_table=source_table,
                id_column=id_column,
                parts=parts,
                group_keys=group_keys,
                seed=seed,
            )
        }
    raise ValueError(f"Unsupported experiment split code: {code}")


def build_split_assignments(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    split_type: str,
    source_table: str,
    id_column: str,
    seed: int = DEFAULT_SEED,
    random_fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
    medium_rules: MediumTransferRules | None = None,
) -> list[SplitAssignment]:
    if not records:
        raise ValueError("At least one record is required to build a split.")
    normalized_type = split_type.strip().lower()
    if normalized_type == "random_split":
        parts = assign_random_parts(len(records), seed=seed, fractions=random_fractions)
        group_keys = [None] * len(records)
    elif normalized_type == "chemical_group_split":
        parts, group_keys = assign_group_parts(records, ("cas_number",), seed=seed)
    elif normalized_type == "species_group_split":
        parts, group_keys = assign_group_parts(records, ("species_number",), seed=seed)
    elif normalized_type == "chemical_species_group_split":
        parts, group_keys = assign_group_parts(records, ("cas_number", "species_number"), seed=seed)
    elif normalized_type == "medium_transfer_split":
        parts, group_keys = assign_medium_transfer_parts(records, rules=medium_rules or MediumTransferRules())
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")

    assignments: list[SplitAssignment] = []
    for record, part, group_key in zip(records, parts, group_keys):
        record_value = str(record.get(id_column)) if id_column != "aggregate_id" else None
        aggregate_value = str(record.get(id_column)) if id_column == "aggregate_id" else None
        assignments.append(
            SplitAssignment(
                split_name=split_name,
                record_id=record_value,
                aggregate_id=aggregate_value,
                split_part=part,
                seed=seed,
                split_type=normalized_type,
                source_table=source_table,
                group_key=group_key,
            )
        )
    return assignments


def build_custom_split_assignments(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    split_type: str,
    source_table: str,
    id_column: str,
    parts: list[str],
    group_keys: list[str | None],
    seed: int,
) -> list[SplitAssignment]:
    if len(records) != len(parts) or len(records) != len(group_keys):
        raise ValueError("records, parts, and group_keys must have the same length.")
    assignments: list[SplitAssignment] = []
    for record, part, group_key in zip(records, parts, group_keys):
        record_value = str(record.get(id_column)) if id_column != "aggregate_id" else None
        aggregate_value = str(record.get(id_column)) if id_column == "aggregate_id" else None
        assignments.append(
            SplitAssignment(
                split_name=split_name,
                record_id=record_value,
                aggregate_id=aggregate_value,
                split_part=part,
                seed=seed,
                split_type=split_type,
                source_table=source_table,
                group_key=group_key,
            )
        )
    return assignments


def assign_random_parts(
    n_records: int,
    *,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
) -> list[str]:
    train_fraction, valid_fraction, test_fraction = fractions
    if min(fractions) < 0 or abs(train_fraction + valid_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Random split fractions must be non-negative and sum to 1.0.")
    indices = list(range(n_records))
    random.Random(seed).shuffle(indices)
    n_train = int(n_records * train_fraction)
    n_valid = int(n_records * valid_fraction)
    parts = ["test"] * n_records
    for idx in indices[:n_train]:
        parts[idx] = "train"
    for idx in indices[n_train : n_train + n_valid]:
        parts[idx] = "valid"
    return parts


def assign_random_named_parts(
    n_records: int,
    *,
    seed: int,
    fractions: dict[str, float],
) -> list[str]:
    validate_named_fractions(fractions)
    names = tuple(fractions)
    indices = list(range(n_records))
    random.Random(seed).shuffle(indices)
    parts = [names[-1]] * n_records
    offset = 0
    for name in names[:-1]:
        count = int(n_records * fractions[name])
        for idx in indices[offset : offset + count]:
            parts[idx] = name
        offset += count
    for idx in indices[offset:]:
        parts[idx] = names[-1]
    return parts


def validate_named_fractions(fractions: dict[str, float]) -> None:
    if len(fractions) < 2:
        raise ValueError("At least two split parts are required.")
    total = sum(float(value) for value in fractions.values())
    if min(float(value) for value in fractions.values()) < 0 or abs(total - 1.0) > 1e-9:
        raise ValueError("Split fractions must be non-negative and sum to 1.0.")


def assign_group_parts(
    records: list[dict[str, Any]],
    group_columns: tuple[str, ...],
    *,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
) -> tuple[list[str], list[str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = make_group_key(record, group_columns)
        grouped_indices[key].append(idx)

    groups = list(grouped_indices)
    group_parts = assign_random_parts(len(groups), seed=seed, fractions=fractions)
    parts = ["test"] * len(records)
    group_keys = [""] * len(records)
    for group_key, group_part in zip(groups, group_parts):
        for idx in grouped_indices[group_key]:
            parts[idx] = group_part
            group_keys[idx] = group_key
    return parts, group_keys


def assign_group_named_parts(
    records: list[dict[str, Any]],
    group_columns: tuple[str, ...],
    *,
    seed: int,
    fractions: dict[str, float],
) -> tuple[list[str], list[str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = make_group_key(record, group_columns)
        grouped_indices[key].append(idx)

    groups = list(grouped_indices)
    group_parts = assign_random_named_parts(len(groups), seed=seed, fractions=fractions)
    parts = [next(iter(fractions))] * len(records)
    group_keys = [""] * len(records)
    for group_key, group_part in zip(groups, group_parts):
        for idx in grouped_indices[group_key]:
            parts[idx] = group_part
            group_keys[idx] = group_key
    return parts, group_keys


def build_kfold_split_sets(
    records: list[dict[str, Any]],
    *,
    split_prefix: str,
    split_type: str,
    source_table: str,
    id_column: str,
    seed: int,
    n_folds: int,
    group_columns: tuple[str, ...] | None,
) -> dict[str, list[SplitAssignment]]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if group_columns is None:
        unit_to_indices = {str(idx): [idx] for idx in range(len(records))}
    else:
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            grouped[make_group_key(record, group_columns)].append(idx)
        unit_to_indices = dict(grouped)

    units = list(unit_to_indices)
    random.Random(seed).shuffle(units)
    unit_folds = {unit: idx % n_folds for idx, unit in enumerate(units)}
    split_sets: dict[str, list[SplitAssignment]] = {}
    for fold in range(n_folds):
        parts = ["train"] * len(records)
        group_keys: list[str | None] = [None] * len(records)
        for unit, indices in unit_to_indices.items():
            part = "test" if unit_folds[unit] == fold else "train"
            for idx in indices:
                parts[idx] = part
                group_keys[idx] = None if group_columns is None else unit
        split_name = f"{split_prefix}_fold{fold + 1}"
        split_sets[split_name] = build_custom_split_assignments(
            records,
            split_name=split_name,
            split_type=split_type,
            source_table=source_table,
            id_column=id_column,
            parts=parts,
            group_keys=group_keys,
            seed=seed,
        )
    return split_sets


def make_group_key(record: dict[str, Any], group_columns: tuple[str, ...]) -> str:
    values = []
    for column in group_columns:
        value = record.get(column)
        values.append("<missing>" if value is None or str(value).strip() == "" else str(value).strip())
    return "||".join(values)


def assign_medium_transfer_parts(
    records: list[dict[str, Any]],
    *,
    rules: MediumTransferRules,
) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    group_keys: list[str] = []
    for record in records:
        medium_text = normalize_medium_text(record)
        domain = classify_medium_domain(medium_text, rules)
        if domain == "test":
            part = "test"
        elif domain == "train":
            part = "train"
        else:
            part = rules.unknown_part
        if part not in {"train", "valid", "test", "excluded"}:
            raise ValueError("unknown_part must be one of train, valid, test, or excluded.")
        parts.append(part)
        group_keys.append(medium_text or "<missing>")
    return parts, group_keys


def normalize_medium_text(record: dict[str, Any]) -> str:
    curated_domain = record.get("medium_domain")
    if curated_domain is not None:
        domain = str(curated_domain).strip().lower()
        if domain in {"aquatic", "soil", "sediment", "solid", "terrestrial_nonsoil"}:
            return domain
    candidates = (
        record.get("domain"),
        record.get("medium_domain"),
        record.get("primary_medium"),
        record.get("media_type"),
        record.get("organism_habitat"),
        record.get("target_basis"),
    )
    return " ".join(str(value).strip().lower() for value in candidates if value is not None and str(value).strip())


def classify_medium_domain(medium_text: str, rules: MediumTransferRules) -> str:
    if not medium_text:
        return "unknown"
    tokens = set(medium_text.replace("_", "-").split())
    expanded_tokens: set[str] = set(tokens)
    for token in tokens:
        expanded_tokens.update(part for part in token.replace(":", " ").replace("/", " /").split() if part)
    test_domains = {term.lower() for term in rules.test_domains}
    train_domains = {term.lower() for term in rules.train_domains}
    if tokens & test_domains or expanded_tokens & test_domains:
        return "test"
    if tokens & train_domains or expanded_tokens & train_domains:
        return "train"
    if any(term in medium_text for term in test_domains):
        return "test"
    if any(term in medium_text for term in train_domains):
        return "train"
    if "water" in medium_text or "aquatic" in medium_text:
        return "train"
    if "soil" in medium_text or "sediment" in medium_text:
        return "test"
    return "unknown"


def summarize_assignments(assignments: list[SplitAssignment]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for assignment in assignments:
        summary[assignment.split_part] = summary.get(assignment.split_part, 0) + 1
    return dict(sorted(summary.items()))
