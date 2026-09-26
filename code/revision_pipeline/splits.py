"""Prediction-blind revision holdouts; no model imports or fit operations.

All boundaries are assigned to the same full (exact + censored) universe.
Source-result components are indivisible in every boundary, including random.
Chemical and combination grouping is global across routes. Consequently the
random boundary is source-group-aware record random, not iid row random.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

REQUIRED = (
    "route", "stable_record_id", "canonical_parent", "latin_name",
    "model_head", "source_result_ids_json", "is_censored",
)
BOUNDARIES = ("record_random", "parent_disjoint", "combination_holdout")
PARTS = ("train", "valid", "test")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, ...]]]:
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing split fields: {missing}")
    # Explicit allowlist: response and prediction columns never reach builder.
    data = frame.loc[:, list(REQUIRED)].copy()
    if data.empty:
        raise ValueError("Cannot split an empty universe")
    for col in REQUIRED[:5]:
        if data[col].isna().any():
            raise ValueError(f"Missing {col}")
        data[col] = data[col].astype(str).str.strip()
        if data[col].eq("").any():
            raise ValueError(f"Empty {col}; resolve identity before splitting")
    if data.duplicated(["route", "stable_record_id"]).any():
        raise ValueError("Duplicate route/stable_record_id identity")
    data = data.sort_values(["route", "stable_record_id"], kind="stable").reset_index(drop=True)
    source_ids = []
    for raw in data["source_result_ids_json"]:
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed source_result_ids_json") from exc
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Every row requires a nonempty source-result ID list")
        if any(v is None or isinstance(v, (list, dict, bool)) for v in values):
            raise ValueError("Source-result IDs must be scalar non-null identities")
        keys = tuple(sorted(set(str(v).strip() for v in values)))
        if any(not k for k in keys):
            raise ValueError("Blank source-result identity")
        source_ids.append(keys)
    flags = []
    for value in data["is_censored"]:
        if pd.isna(value) or str(value).lower() not in ("true", "false", "0", "1"):
            raise ValueError("is_censored must be boolean or 0/1")
        flags.append(str(value).lower() in ("true", "1"))
    data["is_censored"] = flags
    data["source_result_ids_json"] = [json.dumps(v, ensure_ascii=False) for v in source_ids]
    return data, source_ids


def _groups(data: pd.DataFrame, sources: list[tuple[str, ...]], boundary: str) -> dict[str, list[int]]:
    union = _UnionFind(len(data))
    first_source: dict[str, int] = {}
    first_key: dict[Any, int] = {}
    parents = data["canonical_parent"].tolist()
    species = data["latin_name"].tolist()
    heads = data["model_head"].tolist()
    identities = [f"{route}\t{record}" for route, record in
                  zip(data["route"].tolist(), data["stable_record_id"].tolist())]
    for i in range(len(data)):
        for source in sources[i]:
            if source in first_source:
                union.union(i, first_source[source])
            else:
                first_source[source] = i
        key = None
        if boundary == "parent_disjoint":
            key = parents[i]
        elif boundary == "combination_holdout":
            key = (parents[i], species[i], heads[i])
        if key is not None:
            if key in first_key:
                union.union(i, first_key[key])
            else:
                first_key[key] = i
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(data)):
        components[union.find(i)].append(i)
    result = {}
    for rows in components.values():
        group_identities = [identities[i] for i in rows]
        result[_hash("\n".join(sorted(group_identities)))] = rows
    return result


def _assign(data: pd.DataFrame, groups: dict[str, list[int]], seed: int,
            fractions: tuple[float, float, float], attempts: int) -> tuple[dict[str, str], int]:
    strata = list(zip(data["route"], data["model_head"]))
    totals = Counter(strata)
    profiles = {key: Counter(strata[i] for i in rows) for key, rows in groups.items()}
    targets = {s: [totals[s] * f for f in fractions] for s in totals}
    best = None
    for attempt in range(attempts):
        # Fixed, label-blind small order perturbations reduce greedy artifacts.
        def ordering(key: str) -> tuple[float, str]:
            h = _hash(f"{seed}|{attempt}|{key}")
            jitter = 0.9 + 0.2 * int(h[:16], 16) / 2**64
            return (-len(groups[key]) * jitter, h)
        counts = {s: [0, 0, 0] for s in totals}
        assignment = {}
        for key in sorted(groups, key=ordering):
            def insertion_cost(part: int) -> tuple[float, str]:
                delta = 0.0
                for s, n in profiles[key].items():
                    old, target = counts[s][part], targets[s][part]
                    delta += ((old + n - target)**2 - (old - target)**2) / max(target, 1.0)
                return delta, _hash(f"{seed}|{attempt}|{key}|{part}")
            part = min(range(3), key=insertion_cost)
            assignment[key] = PARTS[part]
            for s, n in profiles[key].items():
                counts[s][part] += n
        cost = sum((counts[s][p] - targets[s][p])**2 / max(targets[s][p], 1.0)
                   for s in totals for p in range(3))
        score = (cost, attempt)
        if best is None or score < best[0]:
            best = (score, assignment, attempt)
    assert best is not None
    return best[1], best[2]


def build_revision_splits(
    frame: pd.DataFrame, *, seed: int = 42,
    fractions: tuple[float, float, float] = (0.64, 0.16, 0.20),
    attempts: int = 16, minimum_counts: tuple[int, int, int] = (35, 5, 10),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return three-boundary assignments and JSON-serializable audit.

    Caller must join returned rows to labels by route/stable_record_id only.
    Exact-only and censor-aware fits reuse these assignments without resplit.
    Test evaluation must use a common prelocked exact-test identity subset.
    No head is dropped here; support flags are descriptive admission checks.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions)-1) > 1e-10:
        raise ValueError("Require three positive fractions summing to one")
    if attempts < 1 or len(minimum_counts) != 3 or any(n < 0 for n in minimum_counts):
        raise ValueError("Invalid fixed candidate budget or support minima")
    data, sources = _normalize(frame)
    outputs, boundary_audits = [], []
    for boundary in BOUNDARIES:
        groups = _groups(data, sources, boundary)
        assignment, chosen = _assign(data, groups, seed, fractions, attempts)
        result = data.copy()
        result["boundary_id"] = boundary
        result["split_seed"] = seed
        # Per-group .loc writes can repeatedly copy pandas 3 Arrow string
        # columns. Populate ordinary lists, then construct each column once.
        assigned_parts = [""] * len(data)
        assigned_groups = [""] * len(data)
        for group, rows in groups.items():
            part = assignment[group]
            for i in rows:
                assigned_parts[i] = part
                assigned_groups[i] = group
        result["assigned_split"] = assigned_parts
        result["split_group_id"] = assigned_groups
        # Check every inherited source identity, even across routes.
        seen: dict[str, str] = {}
        for i, keys in enumerate(sources):
            part = assigned_parts[i]
            for key in keys:
                if key in seen and seen[key] != part:
                    raise AssertionError("Source-result identity crosses split")
                seen[key] = part
        if boundary != "record_random":
            columns = ["canonical_parent"] if boundary == "parent_disjoint" else ["canonical_parent", "latin_name", "model_head"]
            if result.groupby(columns)["assigned_split"].nunique().max() > 1:
                raise AssertionError("Requested global group crosses split")
        support = []
        for (route, head), head_rows in result.groupby(["route", "model_head"], sort=True):
            row = {"route": route, "model_head": head}
            for part in PARTS:
                subset = head_rows.loc[head_rows["assigned_split"].eq(part)]
                row[f"{part}_all_n"] = len(subset)
                row[f"{part}_exact_n"] = int((~subset["is_censored"]).sum())
                row[f"{part}_censored_n"] = int(subset["is_censored"].sum())
                row[f"{part}_parent_n"] = int(subset["canonical_parent"].nunique())
            # Conservative common exact-only evaluation/admission support.
            row["common_exact_support_eligible"] = all(row[f"{p}_exact_n"] >= n for p, n in zip(PARTS, minimum_counts))
            row["ineligible_reasons"] = [f"{p}_exact_n<{n}" for p, n in zip(PARTS, minimum_counts) if row[f"{p}_exact_n"] < n]
            support.append(row)
        digest_rows = result[["route", "stable_record_id", "assigned_split", "split_group_id"]].to_dict("records")
        boundary_audits.append({
            "boundary_id": boundary, "selected_candidate": chosen,
            "rows": len(result), "groups": len(groups),
            "largest_group_rows": max(map(len, groups.values())),
            "actual_split_rows": {p: int(result["assigned_split"].eq(p).sum()) for p in PARTS},
            "assignment_sha256": _hash(json.dumps(digest_rows, ensure_ascii=False, sort_keys=True)),
            "source_result_overlap": 0,
            "requested_global_group_overlap": 0 if boundary != "record_random" else None,
            "support": support,
        })
        outputs.append(result)
    audit = {
        "schema": "revision_single_holdout_splits_v1", "seed": seed,
        "seed_count": 1, "outer_cv_folds": 0, "holdouts_per_boundary": 1,
        "target_fractions": dict(zip(PARTS, fractions)), "candidate_budget": attempts,
        "minimum_exact_counts": dict(zip(PARTS, minimum_counts)),
        "rounding": "indivisible global source/group components; proportions are targets, no row rounding",
        "assignment_inputs": [c for c in REQUIRED if c != "is_censored"],
        "censor_flag_used_for_assignment": False, "responses_used": False,
        "record_random_definition": "source-result-component-aware head-balanced random holdout",
        "global_group_scope": "all input routes together",
        "empty_parent_policy": "fail_closed", "unsupported_heads_dropped": False,
        "boundaries": boundary_audits,
    }
    return pd.concat(outputs, ignore_index=True), audit

