"""Checkpoint persistence helpers for long-running extraction jobs."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Type, TypeVar

from pydantic import BaseModel, ValidationError

from models import PaperExtraction

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model regardless of major version."""

    if hasattr(model, "_model_dump"):
        return model._model_dump(exclude_none=True)
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _model_validate(model_cls: Type[ModelT], data: Mapping[str, Any]) -> ModelT:
    """Instantiate a Pydantic model regardless of major version."""

    if hasattr(model_cls, "_model_validate"):
        return model_cls._model_validate(data)
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


CHECKPOINT_ARG_KEYS = [
    "query",
    "filters",
    "out_prefix",
    "n_top_cited",
    "n_most_recent",
    "fetch_buffer",
    "project_to_weighted",
]


@dataclass
class CheckpointState:
    records: Optional[List[Dict[str, Any]]] = None
    extractions: List[PaperExtraction] = field(default_factory=list)
    completed_ids: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _identifier_from_values(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
        elif value not in (None, ""):
            candidate = str(value).strip()
            if candidate:
                return candidate
    return ""


def normalize_for_checkpoint(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize_for_checkpoint(v) for v in value]
    if isinstance(value, list):
        return [normalize_for_checkpoint(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def checkpoint_record_id(meta: Mapping[str, Any]) -> str:
    base = _identifier_from_values(
        meta.get("id"),
        meta.get("paper_id"),
        meta.get("doi"),
        meta.get("title"),
    )
    if base:
        return base
    try:
        fallback = json.dumps(meta, sort_keys=True, default=str)
    except TypeError:
        fallback = repr(meta)
    return hashlib.sha1(fallback.encode("utf-8")).hexdigest()


def checkpoint_extraction_id(extraction: PaperExtraction) -> str:
    base = _identifier_from_values(
        extraction.paper_id,
        extraction.doi,
        extraction.title,
    )
    if base:
        return base
    fallback = json.dumps(
        _model_dump(extraction),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(fallback.encode("utf-8")).hexdigest()


def _coerce_relation_aliases(
    relations: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    coerced: List[Dict[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if "object" not in relation and "obj" in relation:
            relation = dict(relation)
            relation["object"] = relation.pop("obj")
        coerced.append(relation)
    return coerced


def load_checkpoint_state(path: pathlib.Path) -> Optional[CheckpointState]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[warn] Failed to read checkpoint at {path}: {exc}.")
        return None

    extractions: List[PaperExtraction] = []
    raw_extractions = payload.get("extractions") or []
    for index, data in enumerate(raw_extractions):
        try:
            extractions.append(_model_validate(PaperExtraction, data))
        except ValidationError as exc:
            if isinstance(data, dict) and "relations" in data:
                fixed = dict(data)
                try:
                    fixed_relations = _coerce_relation_aliases(
                        list(fixed.get("relations") or [])
                    )
                    fixed["relations"] = fixed_relations
                    extractions.append(_model_validate(PaperExtraction, fixed))
                    continue
                except Exception:
                    pass
            print(
                f"[warn] Skipping extraction #{index} in checkpoint due to validation error: {exc}"
            )

    completed_ids = set(payload.get("completed_ids") or [])
    for extraction in extractions:
        completed_ids.add(checkpoint_extraction_id(extraction))

    records_obj = payload.get("records")
    records: Optional[List[Dict[str, Any]]] = None
    if isinstance(records_obj, list):
        records = records_obj  # type: ignore[assignment]

    metadata = payload.get("metadata") or {}
    return CheckpointState(
        records=records,
        extractions=extractions,
        completed_ids=completed_ids,
        metadata=metadata,
    )


def save_checkpoint_state(path: pathlib.Path, state: CheckpointState) -> None:
    metadata = dict(state.metadata or {})
    metadata["last_saved"] = datetime.utcnow().isoformat() + "Z"
    payload = {
        "metadata": metadata,
        "records": state.records,
        "extractions": [_model_dump(extraction) for extraction in state.extractions],
        "completed_ids": sorted(state.completed_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


__all__ = [
    "CHECKPOINT_ARG_KEYS",
    "CheckpointState",
    "checkpoint_extraction_id",
    "checkpoint_record_id",
    "load_checkpoint_state",
    "normalize_for_checkpoint",
    "save_checkpoint_state",
]
