"""Core data models for the auto-psych nosology knowledge graph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

NODE_TYPES: Set[str] = {
    "Symptom",
    "Diagnosis",
    "RDoC_Construct",
    "HiTOP_Component",
    "Biomarker",
    "Treatment",
    "Task",
    "Measure",
}

REL_TYPES: Set[str] = {
    "supports",
    "contradicts",
    "predicts",
    "co_occurs",
    "treats",
    "biomarker_for",
    "measure_of",
}


class NodeRecord(BaseModel):
    """Normalized node extracted from a document."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str = Field(
        ..., description="Primary name for the entity as used in the paper."
    )
    lemma: str = Field(
        ..., description="Singular, lowercase canonical key used for deduplication."
    )
    node_type: str = Field(..., description=f"One of {sorted(list(NODE_TYPES))}")
    synonyms: List[str] = Field(default_factory=list)
    normalizations: Dict[str, Any] = Field(
        default_factory=dict,
        description='UMLS or other dictionary normalizations, e.g. {"UMLS": "C0005586"}',
    )


class RelationRecord(BaseModel):
    """Relation between two extracted nodes."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., description="Canonical name of subject node")
    predicate: str = Field(..., description=f"One of {sorted(list(REL_TYPES))}")
    obj: str = Field(..., alias="object", description="Canonical name of object node")
    directionality: str = Field(default="directed", description="directed|undirected")
    evidence_span: str = Field(
        default="", description="Short quote from text supporting this"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    qualifiers: Dict[str, Any] = Field(default_factory=dict)


class PaperExtraction(BaseModel):
    """Container with nodes and relations extracted from a single document."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    doi: Optional[str] = None
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    nodes: List[NodeRecord]
    relations: List[RelationRecord]


__all__ = [
    "NODE_TYPES",
    "REL_TYPES",
    "NodeRecord",
    "RelationRecord",
    "PaperExtraction",
]
