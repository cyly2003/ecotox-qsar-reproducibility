from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ID_COLUMNS = (
    "Name",
    "name",
    "ID",
    "id",
    "Title",
    "title",
    "SMILES",
    "smiles",
    "CanonicalSMILES",
)


@dataclass(frozen=True)
class PadelCacheResult:
    out_path: Path
    manifest_path: Path
    descriptor_names: tuple[str, ...]
    rows_read: int
    rows_written: int
    missing_descriptor_values: int
    nonfinite_descriptor_values: int
    morgan_fingerprint_failures: int
    descriptor_generation_failures: int = 0


def build_padel_command(
    *,
    jar_path: str | Path,
    input_path: str | Path,
    output_csv: str | Path,
    threads: int = 1,
    max_runtime_ms: int | None = None,
) -> list[str]:
    command = [
        "java",
        "-jar",
        str(jar_path),
        "-dir",
        str(input_path),
        "-file",
        str(output_csv),
        "-2d",
        "-threads",
        str(max(int(threads), 1)),
    ]
    if max_runtime_ms is not None:
        command.extend(["-maxruntime", str(int(max_runtime_ms))])
    return command


def run_padel_jar(
    *,
    jar_path: str | Path,
    input_path: str | Path,
    output_csv: str | Path,
    threads: int = 1,
    max_runtime_ms: int | None = None,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = build_padel_command(
        jar_path=jar_path,
        input_path=input_path,
        output_csv=output_csv,
        threads=threads,
        max_runtime_ms=max_runtime_ms,
    )
    return subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout_s)


def write_padel_feature_cache_from_csv(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    smiles_column: str = "smiles",
    descriptor_columns: Sequence[str] | None = None,
    fingerprint_size: int = 512,
    morgan_radius: int = 2,
    source_table: str = "",
    feature_source: str = "padel_descriptor_morgan",
) -> PadelCacheResult:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_written = 0
    missing_descriptor_values = 0
    nonfinite_descriptor_values = 0
    morgan_fingerprint_failures = 0
    descriptor_schema_hash = ""
    descriptor_names: tuple[str, ...] = ()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"PaDEL CSV has no header: {csv_path}")
        descriptor_names = tuple(
            descriptor_columns
            if descriptor_columns is not None
            else [
                field
                for field in reader.fieldnames
                if field not in set(DEFAULT_ID_COLUMNS) and field != smiles_column
            ]
        )
        descriptor_schema_hash = hashlib.sha256(
            "\n".join(descriptor_names).encode("utf-8")
        ).hexdigest()
        with out_path.open("w", encoding="utf-8", newline="\n") as dst:
            for row in reader:
                rows_read += 1
                smiles = str(row.get(smiles_column) or row.get("SMILES") or row.get("Name") or "").strip()
                if not smiles:
                    continue
                descriptors: list[float] = []
                for name in descriptor_names:
                    value, status = coerce_descriptor_value(row.get(name))
                    if status == "missing":
                        missing_descriptor_values += 1
                    elif status == "nonfinite":
                        nonfinite_descriptor_values += 1
                    descriptors.append(value)
                fingerprint = morgan_fingerprint(smiles, fingerprint_size=fingerprint_size, radius=morgan_radius)
                if not any(fingerprint):
                    morgan_fingerprint_failures += 1
                payload = {
                    "schema_version": 2,
                    "smiles": smiles,
                    "feature_source": feature_source,
                    "source_table": source_table,
                    "descriptor_schema_hash": descriptor_schema_hash,
                    "descriptor_names": list(descriptor_names),
                    "descriptors": descriptors,
                    "fingerprint_size": int(fingerprint_size),
                    "fingerprint": fingerprint,
                }
                dst.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows_written += 1

    manifest = {
        "schema_version": 2,
        "input_csv": str(csv_path),
        "out_path": str(out_path),
        "feature_source": feature_source,
        "source_table": source_table,
        "descriptor_count": len(descriptor_names),
        "descriptor_names": list(descriptor_names),
        "descriptor_schema_hash": descriptor_schema_hash,
        "fingerprint_size": int(fingerprint_size),
        "morgan_radius": int(morgan_radius),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "missing_descriptor_values": missing_descriptor_values,
        "nonfinite_descriptor_values": nonfinite_descriptor_values,
        "morgan_fingerprint_failures": morgan_fingerprint_failures,
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return PadelCacheResult(
        out_path=out_path,
        manifest_path=manifest_path,
        descriptor_names=descriptor_names,
        rows_read=rows_read,
        rows_written=rows_written,
        missing_descriptor_values=missing_descriptor_values,
        nonfinite_descriptor_values=nonfinite_descriptor_values,
        morgan_fingerprint_failures=morgan_fingerprint_failures,
    )


def write_padel_feature_cache_from_smiles(
    smiles_values: Iterable[str],
    out_path: str | Path,
    *,
    fingerprint_size: int = 512,
    morgan_radius: int = 2,
    source_table: str = "",
    feature_source: str = "padel_descriptor_morgan",
    batch_size: int = 64,
    timeout_s: int = 300,
    maxruntime_s: int = -1,
    threads: int = 1,
    fallback_single: bool = True,
) -> PadelCacheResult:
    """Calculate PaDEL 2D descriptors from SMILES and write the deep-training cache."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    if manifest_path.exists():
        manifest_path.unlink()
    unique_smiles = _unique_nonempty_smiles(smiles_values)
    pending_rows: list[tuple[str, Mapping[str, Any] | None]] = []
    descriptor_generation_failures = 0
    descriptor_names: tuple[str, ...] = ()
    descriptor_schema_hash = ""
    missing_descriptor_values = 0
    nonfinite_descriptor_values = 0
    morgan_fingerprint_failures = 0
    rows_written = 0

    def write_cache_row(dst: Any, smiles: str, descriptor_row: Mapping[str, Any] | None) -> None:
        nonlocal missing_descriptor_values
        nonlocal nonfinite_descriptor_values
        nonlocal morgan_fingerprint_failures
        nonlocal rows_written

        descriptors: list[float] = []
        for name in descriptor_names:
            value = None if descriptor_row is None else descriptor_row.get(name)
            number, status = coerce_descriptor_value(value)
            if status == "missing":
                missing_descriptor_values += 1
            elif status == "nonfinite":
                nonfinite_descriptor_values += 1
            descriptors.append(number)
        fingerprint = morgan_fingerprint(smiles, fingerprint_size=fingerprint_size, radius=morgan_radius)
        if not any(fingerprint):
            morgan_fingerprint_failures += 1
        payload = {
            "schema_version": 2,
            "smiles": smiles,
            "feature_source": feature_source,
            "source_table": source_table,
            "descriptor_schema_hash": descriptor_schema_hash,
            "descriptor_names": list(descriptor_names),
            "descriptors": descriptors,
            "fingerprint_size": int(fingerprint_size),
            "fingerprint": fingerprint,
        }
        dst.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        rows_written += 1

    with out_path.open("w", encoding="utf-8", newline="\n") as dst:
        for start in range(0, len(unique_smiles), max(int(batch_size), 1)):
            batch = unique_smiles[start : start + max(int(batch_size), 1)]
            try:
                descriptor_rows = _padel_from_smiles_batch(
                    batch,
                    timeout_s=timeout_s,
                    maxruntime_s=maxruntime_s,
                    threads=threads,
                )
            except Exception:
                if not fallback_single:
                    raise
                descriptor_rows = []
                for smiles in batch:
                    try:
                        descriptor_rows.extend(
                            _padel_from_smiles_batch(
                                [smiles],
                                timeout_s=timeout_s,
                                maxruntime_s=maxruntime_s,
                                threads=threads,
                            )
                        )
                    except Exception:
                        descriptor_rows.append(None)
            if len(descriptor_rows) != len(batch):
                descriptor_generation_failures += len(batch)
                descriptor_rows = [None] * len(batch)
            for smiles, descriptor_row in zip(batch, descriptor_rows):
                if descriptor_row is None:
                    descriptor_generation_failures += 1
                elif not descriptor_names:
                    descriptor_names = tuple(
                        key for key in descriptor_row.keys() if key not in set(DEFAULT_ID_COLUMNS)
                    )
                    descriptor_schema_hash = hashlib.sha256(
                        "\n".join(descriptor_names).encode("utf-8")
                    ).hexdigest()
                    for pending_smiles, pending_row in pending_rows:
                        write_cache_row(dst, pending_smiles, pending_row)
                    pending_rows = []
                if descriptor_names:
                    write_cache_row(dst, smiles, descriptor_row)
                else:
                    pending_rows.append((smiles, descriptor_row))
            print(
                json.dumps(
                    {
                        "event": "padel_batch_done",
                        "processed": min(start + len(batch), len(unique_smiles)),
                        "total": len(unique_smiles),
                        "batch_size": len(batch),
                        "rows_written": rows_written,
                        "descriptor_generation_failures": descriptor_generation_failures,
                        "descriptor_count": len(descriptor_names),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if pending_rows:
            descriptor_schema_hash = hashlib.sha256(b"").hexdigest()
            for pending_smiles, pending_row in pending_rows:
                write_cache_row(dst, pending_smiles, pending_row)

    manifest = {
        "schema_version": 2,
        "input_source": "padelpy.padeldescriptor_2d",
        "out_path": str(out_path),
        "feature_source": feature_source,
        "source_table": source_table,
        "descriptor_count": len(descriptor_names),
        "descriptor_names": list(descriptor_names),
        "descriptor_schema_hash": descriptor_schema_hash,
        "fingerprint_size": int(fingerprint_size),
        "morgan_radius": int(morgan_radius),
        "batch_size": int(batch_size),
        "timeout_s": int(timeout_s),
        "maxruntime_s": int(maxruntime_s),
        "threads": int(threads),
        "rows_read": len(unique_smiles),
        "rows_written": rows_written,
        "missing_descriptor_values": missing_descriptor_values,
        "nonfinite_descriptor_values": nonfinite_descriptor_values,
        "morgan_fingerprint_failures": morgan_fingerprint_failures,
        "descriptor_generation_failures": descriptor_generation_failures,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return PadelCacheResult(
        out_path=out_path,
        manifest_path=manifest_path,
        descriptor_names=descriptor_names,
        rows_read=len(unique_smiles),
        rows_written=rows_written,
        missing_descriptor_values=missing_descriptor_values,
        nonfinite_descriptor_values=nonfinite_descriptor_values,
        morgan_fingerprint_failures=morgan_fingerprint_failures,
        descriptor_generation_failures=descriptor_generation_failures,
    )


def coerce_descriptor_value(value: Any) -> tuple[float, str]:
    if value is None:
        return 0.0, "missing"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null", "inf", "-inf", "infinity"}:
        return 0.0, "missing"
    try:
        number = float(text)
    except (TypeError, ValueError):
        return 0.0, "missing"
    if not math.isfinite(number):
        return 0.0, "nonfinite"
    return number, "ok"


def morgan_fingerprint(smiles: str, *, fingerprint_size: int, radius: int) -> list[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except Exception:
        return stable_smiles_fingerprint(smiles, fingerprint_size=fingerprint_size)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return stable_smiles_fingerprint(smiles, fingerprint_size=fingerprint_size)
    generator = GetMorganGenerator(radius=int(radius), fpSize=int(fingerprint_size))
    return [float(bit) for bit in generator.GetFingerprint(mol).ToBitString()]


def stable_smiles_fingerprint(smiles: str, *, fingerprint_size: int) -> list[float]:
    bits = [0.0] * int(fingerprint_size)
    for gram in smiles_ngrams(smiles):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest()
        bits[int(digest, 16) % int(fingerprint_size)] = 1.0
    return bits


def smiles_ngrams(text: str) -> Iterable[str]:
    text = text or ""
    yielded = False
    for width in (1, 2, 3):
        for idx in range(max(len(text) - width + 1, 0)):
            yielded = True
            yield text[idx : idx + width]
    if not yielded:
        yield "<missing>"


def _unique_nonempty_smiles(smiles_values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for value in smiles_values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _padel_from_smiles_batch(
    smiles_values: Sequence[str],
    *,
    timeout_s: int,
    maxruntime_s: int,
    threads: int,
) -> list[Mapping[str, Any] | None]:
    import padelpy

    with tempfile.TemporaryDirectory(prefix="qsar-padel-") as tmpdir:
        tmp = Path(tmpdir)
        smi_path = tmp / "molecules.smi"
        csv_path = tmp / "padel_2d.csv"
        ids = [f"QSAR_{idx:06d}" for idx in range(len(smiles_values))]
        smi_path.write_text(
            "\n".join(f"{smiles}\t{name}" for smiles, name in zip(smiles_values, ids)) + "\n",
            encoding="utf-8",
        )
        maxruntime_ms = -1 if int(maxruntime_s) < 0 else int(maxruntime_s) * 1000
        jar_path = Path(padelpy.__file__).parent / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar"
        command = [
            "java",
            "-Djava.awt.headless=true",
            "-jar",
            str(jar_path),
            "-maxruntime",
            str(maxruntime_ms),
            "-waitingjobs",
            "-1",
            "-threads",
            str(max(int(threads), 1)),
            "-maxcpdperfile",
            "0",
            "-2d",
            "-dir",
            str(smi_path),
            "-file",
            str(csv_path),
            "-retainorder",
        ]
        effective_timeout = _padel_subprocess_timeout(
            timeout_s=timeout_s,
            maxruntime_s=maxruntime_s,
            molecule_count=len(smiles_values),
            threads=threads,
        )
        subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=effective_timeout,
        )
        if not csv_path.exists():
            return [None] * len(smiles_values)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            by_name = {str(row.get("Name", "")).strip().strip('"'): row for row in reader}
        return [by_name.get(name) for name in ids]


def _padel_subprocess_timeout(
    *,
    timeout_s: int,
    maxruntime_s: int,
    molecule_count: int,
    threads: int,
) -> int | None:
    if int(timeout_s) <= 0:
        return None
    if int(maxruntime_s) <= 0:
        return int(timeout_s)
    parallel_slots = max(int(threads), 1)
    waves = math.ceil(max(int(molecule_count), 1) / parallel_slots)
    estimated = int(maxruntime_s) * waves + 45
    return max(60, min(int(timeout_s), estimated))
