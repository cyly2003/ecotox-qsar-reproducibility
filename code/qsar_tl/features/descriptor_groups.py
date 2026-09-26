from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


def load_descriptor_group_indices(
    path: str | Path,
    descriptor_names: list[str] | tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Load prior descriptor groups and resolve them to descriptor indices.

    The YAML file may use either a simple nested-list format:

    descriptor_groups:
      size:
        - MolWt
        - HeavyAtomCount

    or an auditable rule format:

    descriptor_groups:
      size:
        names: [MolWt]
        prefixes: [nAtom]
        contains: [weight]
        regex: ["^AATS.*m$"]
    """

    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups = raw.get("descriptor_groups", raw)
    return resolve_descriptor_group_indices(groups, descriptor_names)


def resolve_descriptor_group_indices(
    groups: Any,
    descriptor_names: list[str] | tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    name_to_index = {str(name): idx for idx, name in enumerate(descriptor_names)}
    normalized_to_index = {
        normalize_descriptor_name(name): idx for idx, name in enumerate(descriptor_names)
    }
    return _resolve_groups(groups, descriptor_names, name_to_index, normalized_to_index)


def descriptor_group_manifest(
    group_indices: Mapping[str, tuple[int, ...]],
    descriptor_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "group_count": len(group_indices),
        "groups": {
            str(group): {
                "indices": list(indices),
                "descriptor_names": [
                    str(descriptor_names[idx])
                    for idx in indices
                    if 0 <= idx < len(descriptor_names)
                ],
            }
            for group, indices in sorted(group_indices.items())
        },
    }


def normalize_descriptor_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_groups(
    raw: Any,
    descriptor_names: list[str] | tuple[str, ...],
    name_to_index: Mapping[str, int],
    normalized_to_index: Mapping[str, int],
    prefix: str = "",
) -> dict[str, tuple[int, ...]]:
    if isinstance(raw, Mapping):
        if _looks_like_rule_group(raw):
            indices = _indices_from_rules(raw, descriptor_names, name_to_index, normalized_to_index)
            return {prefix: indices} if indices else {}
        result: dict[str, tuple[int, ...]] = {}
        for key, value in raw.items():
            group_name = f"{prefix}/{key}" if prefix else str(key)
            result.update(
                _resolve_groups(value, descriptor_names, name_to_index, normalized_to_index, group_name)
            )
        return result
    if isinstance(raw, list):
        indices = tuple(
            sorted(
                {
                    normalized_to_index[normalize_descriptor_name(name)]
                    for name in raw
                    if normalize_descriptor_name(name) in normalized_to_index
                }
            )
        )
        return {prefix: indices} if indices else {}
    return {}


def _looks_like_rule_group(raw: Mapping[str, Any]) -> bool:
    return any(key in raw for key in ("names", "prefixes", "contains", "regex"))


def _indices_from_rules(
    raw: Mapping[str, Any],
    descriptor_names: list[str] | tuple[str, ...],
    name_to_index: Mapping[str, int],
    normalized_to_index: Mapping[str, int],
) -> tuple[int, ...]:
    indices: set[int] = set()
    for name in raw.get("names", []) or []:
        normalized = normalize_descriptor_name(name)
        if normalized in normalized_to_index:
            indices.add(int(normalized_to_index[normalized]))
        elif str(name) in name_to_index:
            indices.add(int(name_to_index[str(name)]))
    for prefix in raw.get("prefixes", []) or []:
        text = str(prefix).lower()
        indices.update(idx for idx, name in enumerate(descriptor_names) if str(name).lower().startswith(text))
    for fragment in raw.get("contains", []) or []:
        text = str(fragment).lower()
        indices.update(idx for idx, name in enumerate(descriptor_names) if text in str(name).lower())
    for pattern in raw.get("regex", []) or []:
        compiled = re.compile(str(pattern))
        indices.update(idx for idx, name in enumerate(descriptor_names) if compiled.search(str(name)))
    return tuple(sorted(indices))
