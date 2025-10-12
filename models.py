"""Core data models for the auto-psych nosology knowledge graph pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Type, TypeVar

try:  # pragma: no cover - import guard for Pydantic v1 compatibility
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Extra, Field, root_validator

    ConfigDict = None  # type: ignore[assignment]
    _PYDANTIC_V2 = False

    def model_validator(*, mode: str):  # type: ignore[override]
        if mode != "after":
            raise NotImplementedError(
                "Only mode='after' is supported when using Pydantic < 2."
            )

        def _decorator(func):
            def _wrapper(cls, values):
                instance = cls.construct(**values)
                result = func(instance)
                if isinstance(result, cls):
                    return result.dict()
                if result is None:
                    return values
                return result

            return root_validator(pre=False, allow_reuse=True)(_wrapper)

        return _decorator


ModelT = TypeVar("ModelT", bound="StrictBaseModel")


if _PYDANTIC_V2:

    class StrictBaseModel(BaseModel):
        """Base model that forbids extra fields across Pydantic versions."""

        model_config = ConfigDict(extra="forbid")  # type: ignore[arg-type]

        @classmethod
        def _model_validate(cls: Type[ModelT], data: Any) -> ModelT:
            return cls.model_validate(data)

        def _model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return super().model_dump(*args, **kwargs)

else:

    class StrictBaseModel(BaseModel):
        """Base model shim that emulates Pydantic v2 APIs for v1 installs."""

        class Config:
            extra = Extra.forbid

        @classmethod
        def model_validate(cls: Type[ModelT], data: Any) -> ModelT:  # type: ignore[override]
            return cls.parse_obj(data)

        @classmethod
        def _model_validate(cls: Type[ModelT], data: Any) -> ModelT:
            return cls.model_validate(data)

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            return self.dict(*args, **kwargs)

        def _model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return self.model_dump(*args, **kwargs)


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


class ClaimDescriptor(StrictBaseModel):
    """Structured description of the claim an evidence edge is about."""

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


class NodeRecord(StrictBaseModel):
    """Normalized node extracted from a document."""

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


class RelationRecord(StrictBaseModel):
    """Relation between two extracted nodes."""

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


class PaperExtraction(StrictBaseModel):
    """Container with nodes and relations extracted from a single document."""

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
