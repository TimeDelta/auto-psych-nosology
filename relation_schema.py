"""Resolve stored relation names separately from model-specific encodings."""

from __future__ import annotations

import math
from typing import Any, Mapping


def resolve_edge_relation(attributes: Mapping[str, Any]) -> str:
    """Accept legacy aliases and reject ambiguous, silently discarded relations."""
    raw_names = {
        str(attributes[field_name]).strip()
        for field_name in ("relation", "relation_type")
        if attributes.get(field_name) not in (None, "")
        and str(attributes[field_name]).strip()
    }
    if len(raw_names) > 1:
        raise ValueError(f"Conflicting raw relation aliases: {sorted(raw_names)}")

    model_relation = str(attributes.get("model_relation") or "").strip()
    predicate = str(attributes.get("predicate") or "").strip()
    if model_relation:
        if predicate and predicate != model_relation:
            raise ValueError("predicate differs from the explicit model_relation")
        return model_relation
    if predicate:
        if raw_names and predicate not in raw_names:
            raise ValueError("predicate conflicts with raw relation aliases")
        return predicate
    return next(iter(raw_names), "rel")


def sign_category(sign: Any) -> str:
    """Zero denotes an unsigned edge; missing denotes an unknown sign."""
    if sign is None or (isinstance(sign, str) and not sign.strip()):
        return "unknown"
    try:
        numeric_sign = float(sign)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid edge sign: {sign!r}") from error
    if math.isnan(numeric_sign):
        return "unknown"
    if not math.isfinite(numeric_sign) or numeric_sign not in (-1.0, 0.0, 1.0):
        raise ValueError(f"Edge sign must be -1, 0, 1 or missing: {sign!r}")
    return {-1.0: "negative", 0.0: "unsigned", 1.0: "positive"}[numeric_sign]


def encode_model_relation(relation_type: str, sign: Any, mode: str) -> str:
    if mode == "typed":
        return relation_type
    if mode == "typed-signed":
        return f"{relation_type}::sign={sign_category(sign)}"
    raise ValueError(f"Unknown relation mode: {mode}")
