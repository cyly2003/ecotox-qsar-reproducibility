from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from qsar_tl.web.schemas import EffectOption, EndpointFamily, SpeciesRecord, TaxonomyInput


TAXONOMY_LEVEL_TO_COLUMN = {
    "kingdom": "kingdom",
    "phylum": "phylum",
    "class": "class_name",
    "class_name": "class_name",
    "order": "tax_order",
    "tax_order": "tax_order",
    "family": "family",
    "genus": "genus",
    "species": "species",
}
ENDPOINT_TO_TASK_FAMILY: dict[str, str] = {
    "EC": "ECx",
    "LOEC": "LOEC",
    "NOEC": "NOEC",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SpeciesRepository:
    db_path: Path
    source_table: str

    def __post_init__(self) -> None:
        if not IDENTIFIER_RE.match(self.source_table):
            raise ValueError(f"Unsafe SQLite table name: {self.source_table}")

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        db_uri = f"{self.db_path.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(db_uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def search_species(
        self,
        query: str,
        *,
        medium_domain: str | None = None,
        species_mae: Mapping[str, float] | None = None,
        species_r2: Mapping[str, float] | None = None,
        limit: int = 20,
    ) -> list[SpeciesRecord]:
        text = query.strip().lower()
        if not text:
            return []
        pattern = f"%{text}%"
        params: list[Any] = [pattern, pattern, pattern, pattern]
        context_clauses, context_params = self.medium_species_filter(medium_domain)
        context_sql = "".join(f"\n              AND {clause}" for clause in context_clauses)
        params.extend(context_params)
        has_species_metrics = bool(species_mae) or bool(species_r2)
        query_limit = max(int(limit) * 5, 250) if has_species_metrics else int(limit)
        params.extend([text, query_limit])
        sql = f"""
            SELECT
                COALESCE(CAST(species_number AS TEXT), '') AS species_number,
                COALESCE(latin_name, '') AS latin_name,
                COALESCE(common_name, '') AS common_name,
                COALESCE(kingdom, '') AS kingdom,
                COALESCE(phylum, '') AS phylum,
                COALESCE(class_name, '') AS class_name,
                COALESCE(tax_order, '') AS tax_order,
                COALESCE(family, '') AS family,
                COALESCE(genus, '') AS genus,
                COALESCE(species, '') AS species,
                COUNT(*) AS record_count
            FROM "{self.source_table}"
            WHERE latin_name IS NOT NULL
              AND TRIM(latin_name) <> ''
              AND (
                LOWER(latin_name) LIKE ?
                OR LOWER(COALESCE(common_name, '')) LIKE ?
                OR LOWER(COALESCE(genus, '')) LIKE ?
                OR LOWER(COALESCE(species, '')) LIKE ?
              )
              {context_sql}
            GROUP BY
                species_number, latin_name, common_name, kingdom, phylum,
                class_name, tax_order, family, genus, species
            ORDER BY
                CASE WHEN LOWER(latin_name) = ? THEN 0 ELSE 1 END,
                record_count DESC,
                latin_name
            LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        records = [
            species_record_from_row(row, species_mae=species_mae, species_r2=species_r2)
            for row in rows
        ]
        if has_species_metrics:
            records = sorted(
                records,
                key=lambda item: (
                    str(item.latin_name or "").strip().lower() != text,
                    item.species_r2 is None,
                    -float(item.species_r2) if item.species_r2 is not None else 0.0,
                    item.species_mae is None,
                    float(item.species_mae) if item.species_mae is not None else float("inf"),
                    -int(item.record_count),
                    str(item.latin_name or "").lower(),
                ),
            )
        return records[: int(limit)]

    def species_options(
        self,
        *,
        medium_domain: str | None = None,
        species_mae: Mapping[str, float] | None = None,
        species_r2: Mapping[str, float] | None = None,
        limit: int = 200,
    ) -> list[SpeciesRecord]:
        context_clauses, context_params = self.medium_species_filter(medium_domain)
        context_sql = "".join(f"\n              AND {clause}" for clause in context_clauses)
        params: list[Any] = [*context_params]
        limit_sql = ""
        has_species_metrics = bool(species_mae) or bool(species_r2)
        if not has_species_metrics:
            params.append(int(limit))
            limit_sql = "LIMIT ?"
        sql = f"""
            SELECT
                COALESCE(CAST(species_number AS TEXT), '') AS species_number,
                COALESCE(latin_name, '') AS latin_name,
                COALESCE(common_name, '') AS common_name,
                COALESCE(kingdom, '') AS kingdom,
                COALESCE(phylum, '') AS phylum,
                COALESCE(class_name, '') AS class_name,
                COALESCE(tax_order, '') AS tax_order,
                COALESCE(family, '') AS family,
                COALESCE(genus, '') AS genus,
                COALESCE(species, '') AS species,
                COUNT(*) AS record_count
            FROM "{self.source_table}"
            WHERE latin_name IS NOT NULL
              AND TRIM(latin_name) <> ''
              {context_sql}
            GROUP BY
                species_number, latin_name, common_name, kingdom, phylum,
                class_name, tax_order, family, genus, species
            ORDER BY record_count DESC, latin_name
            {limit_sql}
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        records = [
            species_record_from_row(row, species_mae=species_mae, species_r2=species_r2)
            for row in rows
        ]
        if has_species_metrics:
            records = sort_species_records_by_r2(records)
        return records[: int(limit)]

    def taxonomy_options(
        self,
        level: str,
        *,
        filters: Mapping[str, str] | None = None,
        medium_domain: str | None = None,
        search: str = "",
        limit: int = 100,
    ) -> list[str]:
        column = taxonomy_column(level)
        clauses = [f"{column} IS NOT NULL", f"TRIM({column}) <> ''"]
        params: list[Any] = []
        context_clauses, context_params = self.medium_species_filter(medium_domain)
        clauses.extend(context_clauses)
        params.extend(context_params)
        for raw_level, raw_value in (filters or {}).items():
            value = str(raw_value or "").strip()
            if not value:
                continue
            filter_column = taxonomy_column(raw_level)
            clauses.append(f"{filter_column} = ?")
            params.append(value)
        if search.strip():
            clauses.append(f"LOWER({column}) LIKE ?")
            params.append(f"%{search.strip().lower()}%")
        params.append(int(limit))
        sql = f"""
            SELECT {column} AS value, COUNT(*) AS record_count
            FROM "{self.source_table}"
            WHERE {' AND '.join(clauses)}
            GROUP BY {column}
            ORDER BY record_count DESC, value
            LIMIT ?
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [str(row["value"]) for row in rows if str(row["value"]).strip()]

    def effects_for_endpoint(
        self,
        endpoint: EndpointFamily,
        *,
        medium_domain: str | None = None,
    ) -> list[EffectOption]:
        task_family = ENDPOINT_TO_TASK_FAMILY[str(endpoint)]
        clauses = ["task_family = ?", "effect_family IS NOT NULL", "TRIM(effect_family) <> ''"]
        params: list[Any] = [task_family]
        if medium_domain:
            clauses.append("medium_domain = ?")
            params.append(str(medium_domain))
        sql = f"""
            SELECT effect_family, task_head, COUNT(*) AS record_count
            FROM "{self.source_table}"
            WHERE {' AND '.join(clauses)}
            GROUP BY effect_family, task_head
            ORDER BY record_count DESC, effect_family
        """
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            EffectOption(
                endpoint=endpoint,
                effect_family=str(row["effect_family"]),
                task_head=str(row["task_head"]),
                record_count=int(row["record_count"]),
            )
            for row in rows
        ]

    def task_reference_count(
        self,
        *,
        latin_name: str,
        task_family: str,
        effect_family: str,
        medium_domain: str,
    ) -> int:
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE latin_name = ?
              AND task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
        """
        with self.connect() as connection:
            row = connection.execute(sql, (latin_name, task_family, effect_family, medium_domain)).fetchone()
        return int(row["n"] if row is not None else 0)

    def task_count(self, *, task_family: str, effect_family: str, medium_domain: str) -> int:
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
        """
        with self.connect() as connection:
            row = connection.execute(sql, (task_family, effect_family, medium_domain)).fetchone()
        return int(row["n"] if row is not None else 0)

    def species_reference_count(self, *, latin_name: str, medium_domain: str) -> int:
        clauses = ["latin_name = ?"]
        params: list[Any] = [latin_name]
        context_clauses, context_params = self.medium_species_filter(medium_domain)
        clauses.extend(context_clauses)
        params.extend(context_params)
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE {' AND '.join(clauses)}
        """
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return int(row["n"] if row is not None else 0)

    def max_species_reference_count(self, *, medium_domain: str) -> int:
        clauses = ["latin_name IS NOT NULL", "TRIM(latin_name) <> ''"]
        params: list[Any] = []
        context_clauses, context_params = self.medium_species_filter(medium_domain)
        clauses.extend(context_clauses)
        params.extend(context_params)
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE {' AND '.join(clauses)}
            GROUP BY latin_name
            ORDER BY n DESC
            LIMIT 1
        """
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return int(row["n"] if row is not None else 0)

    def max_species_task_count(self, *, task_family: str, effect_family: str, medium_domain: str) -> int:
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
              AND latin_name IS NOT NULL
              AND TRIM(latin_name) <> ''
            GROUP BY latin_name
            ORDER BY n DESC
            LIMIT 1
        """
        with self.connect() as connection:
            row = connection.execute(sql, (task_family, effect_family, medium_domain)).fetchone()
        return int(row["n"] if row is not None else 0)

    def total_reference_count(self, *, medium_domain: str) -> int:
        sql = f"""
            SELECT COUNT(*) AS n
            FROM "{self.source_table}"
            WHERE medium_domain = ?
        """
        with self.connect() as connection:
            row = connection.execute(sql, (medium_domain,)).fetchone()
        return int(row["n"] if row is not None else 0)

    def target_context(self, *, task_family: str, effect_family: str, medium_domain: str) -> dict[str, str]:
        sql = f"""
            SELECT
                COALESCE(target_family, '') AS target_family,
                COALESCE(target_name, '') AS target_name,
                COALESCE(target_basis, '') AS target_basis,
                COUNT(*) AS record_count
            FROM "{self.source_table}"
            WHERE task_family = ?
              AND effect_family = ?
              AND medium_domain = ?
              AND target_family IS NOT NULL
              AND TRIM(target_family) <> ''
            GROUP BY target_family, target_name, target_basis
            ORDER BY record_count DESC
            LIMIT 1
        """
        with self.connect() as connection:
            row = connection.execute(sql, (task_family, effect_family, medium_domain)).fetchone()
        if row is None:
            return {
                "target_family": "aquatic_pTox_mol_L",
                "target_name": "ptox_mol_l",
                "target_basis": "",
            }
        return {
            "target_family": str(row["target_family"] or ""),
            "target_name": str(row["target_name"] or ""),
            "target_basis": str(row["target_basis"] or ""),
        }

    def medium_species_filter(self, medium_domain: str | None) -> tuple[list[str], list[Any]]:
        if not medium_domain:
            return [], []
        normalized = str(medium_domain).strip().lower()
        clauses = ["medium_domain = ?"]
        params: list[Any] = [normalized]
        if "habitat_labels" not in self.available_columns():
            return clauses, params
        habitat_patterns = {
            "aquatic": ["%aquatic%"],
            "soil": ["%soil%", "%terrestrial%"],
        }.get(normalized)
        if habitat_patterns:
            clauses.append(
                "("
                + " OR ".join(
                    "LOWER(COALESCE(habitat_labels, '')) LIKE ?"
                    for _ in habitat_patterns
                )
                + ")"
            )
            params.extend(habitat_patterns)
        return clauses, params

    def available_columns(self) -> set[str]:
        sql = f'PRAGMA table_info("{self.source_table}")'
        with self.connect() as connection:
            rows = connection.execute(sql).fetchall()
        return {str(row["name"]) for row in rows}


def taxonomy_column(level: str) -> str:
    key = str(level).strip().lower()
    if key not in TAXONOMY_LEVEL_TO_COLUMN:
        allowed = ", ".join(sorted(TAXONOMY_LEVEL_TO_COLUMN))
        raise ValueError(f"Unsupported taxonomy level '{level}'. Allowed values: {allowed}")
    return TAXONOMY_LEVEL_TO_COLUMN[key]


def species_record_from_row(
    row: Mapping[str, Any],
    *,
    species_mae: Mapping[str, float] | None = None,
    species_r2: Mapping[str, float] | None = None,
) -> SpeciesRecord:
    latin_name = str(row["latin_name"] or "")
    taxonomy = TaxonomyInput(
        kingdom=str(row["kingdom"] or ""),
        phylum=str(row["phylum"] or ""),
        class_name=str(row["class_name"] or ""),
        tax_order=str(row["tax_order"] or ""),
        family=str(row["family"] or ""),
        genus=str(row["genus"] or ""),
        species=str(row["species"] or ""),
        latin_name=latin_name,
    )
    return SpeciesRecord(
        species_number=str(row["species_number"] or ""),
        latin_name=latin_name,
        common_name=str(row["common_name"] or ""),
        taxonomy=taxonomy,
        record_count=int(row["record_count"] or 0),
        species_mae=lookup_species_mae(latin_name, species_mae),
        species_r2=lookup_species_metric(latin_name, species_r2),
    )


def lookup_species_mae(latin_name: str, species_mae: Mapping[str, float] | None) -> float | None:
    return lookup_species_metric(latin_name, species_mae)


def lookup_species_metric(latin_name: str, values: Mapping[str, float] | None) -> float | None:
    if not values:
        return None
    value = values.get(str(latin_name or "").strip())
    if value is None:
        value = values.get(str(latin_name or "").strip().lower())
    return float(value) if value is not None else None


def sort_species_records_by_mae(records: list[SpeciesRecord]) -> list[SpeciesRecord]:
    return sorted(
        records,
        key=lambda item: (
            item.species_mae is None,
            float(item.species_mae) if item.species_mae is not None else float("inf"),
            -int(item.record_count),
            str(item.latin_name or "").lower(),
        ),
    )


def sort_species_records_by_r2(records: list[SpeciesRecord]) -> list[SpeciesRecord]:
    return sorted(
        records,
        key=lambda item: (
            item.species_r2 is None,
            -float(item.species_r2) if item.species_r2 is not None else 0.0,
            item.species_mae is None,
            float(item.species_mae) if item.species_mae is not None else float("inf"),
            -int(item.record_count),
            str(item.latin_name or "").lower(),
        ),
    )
