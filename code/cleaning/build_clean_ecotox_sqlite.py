from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_SQLITE = Path("inputs/ECOTOX_ASCII.sqlite")
DEFAULT_SMILES_DICTIONARY = Path("inputs/compound_toxicity_master.csv")
DEFAULT_OUTPUT_SQLITE = Path("outputs/databases/ecotox_clean.sqlite")
DEFAULT_REPORT_JSON = Path("outputs/reports/ecotox_clean_build_report.json")


RESULTS_COLUMNS = [
    "result_id",
    "test_id",
    "obs_duration_mean_op",
    "obs_duration_mean",
    "obs_duration_unit",
    "conc1_type",
    "conc1_mean_op",
    "conc1_mean",
    "conc1_unit",
    "conc1_min_op",
    "conc1_min",
    "conc1_max_op",
    "conc1_max",
    "endpoint",
    "endpoint_comments",
    "trend",
    "effect",
    "measurement",
    "response_site_comments",
]

TESTS_COLUMNS = [
    "test_id",
    "reference_number",
    "test_cas",
    "species_number",
    "test_purity_mean_op",
    "test_purity_mean",
    "test_purity_min_op",
    "test_purity_min",
    "test_purity_max_op",
    "test_purity_max",
    "organism_habitat",
    "organism_lifestage",
    "exposure_duration_mean_op",
    "exposure_duration_mean",
    "exposure_duration_unit",
    "exposure_duration_min_op",
    "exposure_duration_min",
    "exposure_duration_max_op",
    "exposure_duration_max",
    "media_type",
    "num_doses_mean_op",
    "num_doses_mean",
    "num_doses_min_op",
    "num_doses_min",
    "num_doses_max_op",
    "num_doses_max",
]

FULL_COPY_TABLES = ["species", "references"]


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    not_null: bool
    default_value: str | None
    pk: int


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def qualified_identifier(schema: str, name: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(name)}"


def normalize_cas_key(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    return digits


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def get_source_columns(conn: sqlite3.Connection, table: str) -> dict[str, ColumnInfo]:
    rows = conn.execute(f"PRAGMA src.table_info({quote_identifier(table)})").fetchall()
    return {
        row[1]: ColumnInfo(
            name=row[1],
            data_type=row[2] or "TEXT",
            not_null=bool(row[3]),
            default_value=row[4],
            pk=int(row[5]),
        )
        for row in rows
    }


def require_columns(
    conn: sqlite3.Connection, table: str, required_columns: Iterable[str]
) -> list[ColumnInfo]:
    source_columns = get_source_columns(conn, table)
    missing = [column for column in required_columns if column not in source_columns]
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")
    return [source_columns[column] for column in required_columns]


def column_definition(column: ColumnInfo) -> str:
    parts = [quote_identifier(column.name), column.data_type or "TEXT"]
    if column.pk:
        parts.append("PRIMARY KEY")
    if column.not_null and not column.pk:
        parts.append("NOT NULL")
    if column.default_value is not None:
        parts.append(f"DEFAULT {column.default_value}")
    return " ".join(parts)


def create_table(conn: sqlite3.Connection, table: str, columns: list[ColumnInfo]) -> None:
    column_sql = ", ".join(column_definition(column) for column in columns)
    target = qualified_identifier("main", table)
    conn.execute(f"DROP TABLE IF EXISTS {target}")
    conn.execute(f"CREATE TABLE {target} ({column_sql})")


def copy_selected_table(
    conn: sqlite3.Connection, table: str, selected_columns: list[str]
) -> None:
    columns = require_columns(conn, table, selected_columns)
    create_table(conn, table, columns)
    col_sql = ", ".join(quote_identifier(column) for column in selected_columns)
    conn.execute(
        f"""
        INSERT INTO {qualified_identifier("main", table)} ({col_sql})
        SELECT {col_sql}
        FROM {qualified_identifier("src", table)}
        """
    )


def copy_full_table(conn: sqlite3.Connection, table: str) -> None:
    columns = list(get_source_columns(conn, table).values())
    if not columns:
        raise ValueError(f"Source table not found or empty schema: {table}")
    selected_columns = [column.name for column in columns]
    create_table(conn, table, columns)
    col_sql = ", ".join(quote_identifier(column) for column in selected_columns)
    conn.execute(
        f"""
        INSERT INTO {qualified_identifier("main", table)} ({col_sql})
        SELECT {col_sql}
        FROM {qualified_identifier("src", table)}
        """
    )


def detect_csv_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                handle.readline()
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Cannot detect CSV encoding")


def load_smiles_dictionary(
    conn: sqlite3.Connection, dictionary_path: Path
) -> dict[str, int | str]:
    encoding = detect_csv_encoding(dictionary_path)
    conn.execute("DROP TABLE IF EXISTS main.chemical_smiles_dictionary")
    conn.execute(
        """
        CREATE TABLE main.chemical_smiles_dictionary (
            cas_number TEXT PRIMARY KEY,
            casrn TEXT,
            chemical_name TEXT,
            dtxsid TEXT,
            smiles TEXT,
            connectivity_smiles TEXT,
            inchikey TEXT,
            smiles_source TEXT,
            smiles_match_method TEXT,
            query_status TEXT,
            remarks TEXT
        )
        """
    )

    best_by_cas: dict[str, dict[str, str | None]] = {}
    stats: dict[str, int | str] = {
        "encoding": encoding,
        "input_rows": 0,
        "rows_with_cas_key": 0,
        "rows_with_smiles": 0,
        "duplicate_cas_keys": 0,
        "conflicting_nonempty_smiles": 0,
    }

    with dictionary_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["input_rows"] = int(stats["input_rows"]) + 1
            cas_number = normalize_cas_key(
                row.get("cas_number_raw") or row.get("cas_number") or row.get("casrn")
            )
            if not cas_number:
                continue

            stats["rows_with_cas_key"] = int(stats["rows_with_cas_key"]) + 1
            smiles = clean_text(row.get("smiles"))
            if smiles:
                stats["rows_with_smiles"] = int(stats["rows_with_smiles"]) + 1

            record = {
                "cas_number": cas_number,
                "casrn": clean_text(row.get("casrn")),
                "chemical_name": clean_text(row.get("chemical_name")),
                "dtxsid": clean_text(row.get("dtxsid")),
                "smiles": smiles,
                "connectivity_smiles": clean_text(row.get("connectivity_smiles")),
                "inchikey": clean_text(row.get("inchikey")),
                "smiles_source": clean_text(row.get("smiles_source")),
                "smiles_match_method": clean_text(row.get("smiles_match_method")),
                "query_status": clean_text(row.get("query_status")),
                "remarks": clean_text(row.get("remarks")),
            }

            existing = best_by_cas.get(cas_number)
            if existing is None:
                best_by_cas[cas_number] = record
                continue

            stats["duplicate_cas_keys"] = int(stats["duplicate_cas_keys"]) + 1
            existing_smiles = existing.get("smiles")
            if existing_smiles and smiles and existing_smiles != smiles:
                stats["conflicting_nonempty_smiles"] = (
                    int(stats["conflicting_nonempty_smiles"]) + 1
                )
            if not existing_smiles and smiles:
                best_by_cas[cas_number] = record

    conn.executemany(
        """
        INSERT INTO chemical_smiles_dictionary (
            cas_number, casrn, chemical_name, dtxsid, smiles, connectivity_smiles,
            inchikey, smiles_source, smiles_match_method, query_status, remarks
        )
        VALUES (
            :cas_number, :casrn, :chemical_name, :dtxsid, :smiles,
            :connectivity_smiles, :inchikey, :smiles_source,
            :smiles_match_method, :query_status, :remarks
        )
        """,
        best_by_cas.values(),
    )
    stats["unique_cas_keys"] = len(best_by_cas)
    stats["unique_cas_keys_with_smiles"] = sum(
        1 for record in best_by_cas.values() if record.get("smiles")
    )
    return stats


def copy_chemicals_with_smiles(conn: sqlite3.Connection) -> None:
    source_columns = list(get_source_columns(conn, "chemicals").values())
    if not source_columns:
        raise ValueError("Source table not found or empty schema: chemicals")
    if any(column.name == "smiles" for column in source_columns):
        raise ValueError("Source chemicals table already contains a smiles column")

    columns = source_columns + [
        ColumnInfo(
            name="smiles",
            data_type="TEXT",
            not_null=False,
            default_value=None,
            pk=0,
        )
    ]
    create_table(conn, "chemicals", columns)
    source_col_sql = ", ".join(f"c.{quote_identifier(column.name)}" for column in source_columns)
    insert_col_sql = ", ".join(quote_identifier(column.name) for column in columns)
    conn.execute(
        f"""
        INSERT INTO main.chemicals ({insert_col_sql})
        SELECT {source_col_sql}, d.smiles
        FROM src.chemicals AS c
        LEFT JOIN main.chemical_smiles_dictionary AS d
            ON CAST(c.cas_number AS TEXT) = d.cas_number
        """
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    index_statements = [
        "CREATE INDEX IF NOT EXISTS main.idx_results_test_id ON results(test_id)",
        "CREATE INDEX IF NOT EXISTS main.idx_tests_reference_number ON tests(reference_number)",
        "CREATE INDEX IF NOT EXISTS main.idx_tests_test_cas ON tests(test_cas)",
        "CREATE INDEX IF NOT EXISTS main.idx_tests_species_number ON tests(species_number)",
        "CREATE INDEX IF NOT EXISTS main.idx_chemicals_dtxsid ON chemicals(dtxsid)",
        "CREATE INDEX IF NOT EXISTS main.idx_chemicals_smiles ON chemicals(smiles)",
        "CREATE INDEX IF NOT EXISTS main.idx_species_latin_name ON species(latin_name)",
        "CREATE INDEX IF NOT EXISTS main.idx_references_doi ON \"references\"(doi)",
        "CREATE INDEX IF NOT EXISTS main.idx_smiles_dictionary_smiles ON chemical_smiles_dictionary(smiles)",
    ]
    for statement in index_statements:
        conn.execute(statement)


def create_joined_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS main.ecotox_toxicity_joined")
    conn.execute(
        """
        CREATE VIEW main.ecotox_toxicity_joined AS
        SELECT
            r.result_id,
            r.test_id,
            t.reference_number,
            t.test_cas,
            c.cas_number,
            c.chemical_name,
            c.ecotox_group AS chemical_ecotox_group,
            c.dtxsid,
            c.smiles,
            t.species_number,
            s.common_name,
            s.latin_name,
            s.kingdom,
            s.phylum_division,
            s.class,
            s.tax_order,
            s.family,
            s.genus,
            s.species,
            s.ecotox_group AS species_ecotox_group,
            s.ncbi_taxid,
            r.endpoint,
            r.endpoint_comments,
            r.trend,
            r.effect,
            r.measurement,
            r.response_site_comments,
            r.conc1_type,
            r.conc1_mean_op,
            r.conc1_mean,
            r.conc1_unit,
            r.conc1_min_op,
            r.conc1_min,
            r.conc1_max_op,
            r.conc1_max,
            r.obs_duration_mean_op,
            r.obs_duration_mean,
            r.obs_duration_unit,
            t.test_purity_mean_op,
            t.test_purity_mean,
            t.test_purity_min_op,
            t.test_purity_min,
            t.test_purity_max_op,
            t.test_purity_max,
            t.organism_habitat,
            t.organism_lifestage,
            t.exposure_duration_mean_op,
            t.exposure_duration_mean,
            t.exposure_duration_unit,
            t.exposure_duration_min_op,
            t.exposure_duration_min,
            t.exposure_duration_max_op,
            t.exposure_duration_max,
            t.media_type,
            t.num_doses_mean_op,
            t.num_doses_mean,
            t.num_doses_min_op,
            t.num_doses_min,
            t.num_doses_max_op,
            t.num_doses_max,
            ref.reference_type,
            ref.author,
            ref.title,
            ref.source,
            ref.publication_year,
            ref.doi
        FROM results AS r
        LEFT JOIN tests AS t
            ON r.test_id = t.test_id
        LEFT JOIN chemicals AS c
            ON t.test_cas = c.cas_number
        LEFT JOIN species AS s
            ON t.species_number = s.species_number
        LEFT JOIN "references" AS ref
            ON t.reference_number = ref.reference_number
        """
    )


def scalar(conn: sqlite3.Connection, query: str) -> int:
    return int(conn.execute(query).fetchone()[0])


def build_report(
    conn: sqlite3.Connection,
    source_sqlite: Path,
    smiles_dictionary: Path,
    output_sqlite: Path,
    smiles_stats: dict[str, int | str],
    elapsed_seconds: float,
) -> dict[str, object]:
    table_counts = {
        table: scalar(conn, f"SELECT COUNT(*) FROM {quote_identifier(table)}")
        for table in [
            "results",
            "tests",
            "chemicals",
            "species",
            "references",
            "chemical_smiles_dictionary",
        ]
    }
    quality_checks = {
        "chemicals_with_smiles": scalar(
            conn,
            "SELECT COUNT(*) FROM chemicals WHERE smiles IS NOT NULL AND TRIM(smiles) <> ''",
        ),
        "chemicals_without_smiles": scalar(
            conn,
            "SELECT COUNT(*) FROM chemicals WHERE smiles IS NULL OR TRIM(smiles) = ''",
        ),
        "results_without_matching_test": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM results AS r
            LEFT JOIN tests AS t ON r.test_id = t.test_id
            WHERE t.test_id IS NULL
            """,
        ),
        "tests_without_matching_chemical": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM tests AS t
            LEFT JOIN chemicals AS c
                ON t.test_cas = c.cas_number
            WHERE t.test_cas IS NOT NULL
                AND c.cas_number IS NULL
            """,
        ),
        "tests_without_matching_species": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM tests AS t
            LEFT JOIN species AS s ON t.species_number = s.species_number
            WHERE t.species_number IS NOT NULL
                AND s.species_number IS NULL
            """,
        ),
        "tests_without_matching_reference": scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM tests AS t
            LEFT JOIN "references" AS ref ON t.reference_number = ref.reference_number
            WHERE t.reference_number IS NOT NULL
                AND ref.reference_number IS NULL
            """,
        ),
    }
    return {
        "source_sqlite": str(source_sqlite),
        "smiles_dictionary": str(smiles_dictionary),
        "output_sqlite": str(output_sqlite),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "selected_columns": {
            "results": RESULTS_COLUMNS,
            "tests": TESTS_COLUMNS,
            "chemicals": "all source columns plus smiles",
            "species": "all source columns",
            "references": "all source columns",
        },
        "table_counts": table_counts,
        "smiles_dictionary_stats": smiles_stats,
        "quality_checks": quality_checks,
        "notes": [
            "ECOTOX tests table uses num_doses_* field names; these are retained instead of renaming to number_doses_*.",
            "Operator fields ending in _op are retained for numeric groups where inequality qualifiers can affect toxicological interpretation.",
            "The joined view ecotox_toxicity_joined is provided for analysis convenience; normalized source tables remain available.",
        ],
    }


def build_clean_database(
    source_sqlite: Path,
    smiles_dictionary: Path,
    output_sqlite: Path,
    report_json: Path,
    overwrite: bool,
) -> dict[str, object]:
    if not source_sqlite.exists():
        raise FileNotFoundError(f"Source SQLite not found: {source_sqlite}")
    if not smiles_dictionary.exists():
        raise FileNotFoundError(f"SMILES dictionary CSV not found: {smiles_dictionary}")
    if output_sqlite.exists() and not overwrite:
        raise FileExistsError(f"Output SQLite already exists: {output_sqlite}")

    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_sqlite.with_suffix(output_sqlite.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    start = time.perf_counter()
    conn = sqlite3.connect(temp_output)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("ATTACH DATABASE ? AS src", (str(source_sqlite),))
        conn.execute("BEGIN")
        smiles_stats = load_smiles_dictionary(conn, smiles_dictionary)
        copy_selected_table(conn, "results", RESULTS_COLUMNS)
        copy_selected_table(conn, "tests", TESTS_COLUMNS)
        copy_chemicals_with_smiles(conn)
        for table in FULL_COPY_TABLES:
            copy_full_table(conn, table)
        create_indexes(conn)
        create_joined_view(conn)
        conn.commit()

        report = build_report(
            conn=conn,
            source_sqlite=source_sqlite,
            smiles_dictionary=smiles_dictionary,
            output_sqlite=output_sqlite,
            smiles_stats=smiles_stats,
            elapsed_seconds=0.0,
        )
        report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if output_sqlite.exists():
        output_sqlite.unlink()
    temp_output.replace(output_sqlite)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact ECOTOX SQLite database for QSAR/toxicity analysis."
    )
    parser.add_argument("--source-sqlite", type=Path, default=DEFAULT_SOURCE_SQLITE)
    parser.add_argument(
        "--smiles-dictionary", type=Path, default=DEFAULT_SMILES_DICTIONARY
    )
    parser.add_argument("--output-sqlite", type=Path, default=DEFAULT_OUTPUT_SQLITE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_clean_database(
        source_sqlite=args.source_sqlite,
        smiles_dictionary=args.smiles_dictionary,
        output_sqlite=args.output_sqlite,
        report_json=args.report_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
