"""Core data models for the auto-psych nosology knowledge graph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

NODE_TYPES: Set[str] = {
    "Symptom",
    "Diagnosis",
    "Biomarker",
    "Treatment",
    "Measure",
    "Species",
}

REL_TYPES: Set[str] = {
    "supports",
    "contradicts",
    "replicates",
    "null_reported",
    "predicts",
    "co_occurs",
    "treats",
    "biomarker_for",
    "measure_of",
}

EVIDENCE_RELATIONS: Set[str] = {
    "supports",
    "contradicts",
    "replicates",
    "null_reported",
}


class ClaimDescriptor(BaseModel):
    """Structured description of the claim an evidence edge is about."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "causal",
        "association",
        "mechanistic",
        "measurement",
        "ontological",
        "unknown",
    ] = Field(
        default="unknown",
        description=(
            "High-level category of the claim being supported, e.g. association or causal."
        ),
    )
    statement: str = Field(
        default="",
        description="Normalized natural-language rendering of the claim between the nodes.",
    )
    measure: Optional[str] = Field(
        default=None,
        description="Name of the statistical measure or comparison backing the claim.",
    )
    effect_size: Optional[float] = Field(
        default=None,
        description="Numeric effect size (if available) associated with the claim.",
    )
    direction: Literal["positive", "negative", "null", "complex", "unknown"] = Field(
        default="unknown",
        description="Direction of the reported effect or association.",
    )
    target_relation_id: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the substantive relation this evidence edge is supporting."
        ),
    )
    evidence_type: Optional[str] = Field(
        default=None,
        description="Type of evidence (e.g. textual_quote, statistical_result, replication).",
    )
    population: Optional[str] = Field(
        default=None,
        description="Population or sample characteristics tied to the claim.",
    )
    temporal_window: Optional[str] = Field(
        default=None,
        description="Temporal window covered by the claim (e.g. 24h, 6-month follow-up).",
    )
    measurement_context: Optional[str] = Field(
        default=None,
        description="Additional context about measurement instruments or paradigms.",
    )


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
    claim: Optional[ClaimDescriptor] = Field(
        default=None,
        description=(
            "Structured description of the claim tied to an evidence relation."
        ),
    )
    qualifiers: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_evidence_claim(self) -> "RelationRecord":
        if self.predicate in EVIDENCE_RELATIONS:
            existing_qualifiers = dict(self.qualifiers)
            qualifier_claim = existing_qualifiers.get("claim")
            if self.claim is None and qualifier_claim is not None:
                self.claim = ClaimDescriptor.model_validate(qualifier_claim)
            if self.claim is None:
                raise ValueError(
                    "Evidence relations require a claim descriptor describing the supported claim."
                )
            if not self.claim.statement.strip():
                raise ValueError("Evidence relation claim statement must not be empty.")
            existing_qualifiers.setdefault(
                "claim", self.claim.model_dump(exclude_none=True)
            )
            object.__setattr__(self, "qualifiers", existing_qualifiers)
        return self


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
    "EVIDENCE_RELATIONS",
    "ClaimDescriptor",
    "NodeRecord",
    "RelationRecord",
    "PaperExtraction",
]
