from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "should_drop_nosology_node",
    "NOSOLOGY_NODE_TYPES",
    "NOSOLOGY_NAME_KEYWORDS",
]

NOSOLOGY_NODE_TYPES = {"disease", "disorder", "diagnosis"}
NOSOLOGY_NAME_KEYWORDS = {
    "disorder",
    "disease",
    "syndrome",
    "diagnosis",
    "illness",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return False


def should_drop_nosology_node(attrs: Mapping[str, Any]) -> bool:
    node_type = str(attrs.get("node_type", "")).strip().lower()
    if node_type in NOSOLOGY_NODE_TYPES:
        return True
    for flag in ("ontology_flag", "group_flag", "is_psychiatric"):
        if _parse_bool(attrs.get(flag)):
            return True
    name = str(attrs.get("name", "")).lower()
    if name and any(keyword in name for keyword in NOSOLOGY_NAME_KEYWORDS):
        return True
    return False
