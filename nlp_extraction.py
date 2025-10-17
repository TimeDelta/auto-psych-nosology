"""BioMedKG ingestion and psychiatric subgraph extraction utilities."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import polars as pl

from psychiatry_scoring import (
    PsychiatricRelevanceScorer,
    PsychiatricScoringConfig,
    build_default_scoring_config,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_DEFAULT_PATTERNS: tuple[str, ...] = (
    r"(?i)psychi",
    r"(?i)mental",
    r"(?i)depress",
    r"(?i)bipolar",
    r"(?i)schizo",
    r"(?i)anxiety",
    r"(?i)adhd",
    r"(?i)attention deficit",
    r"(?i)autism",
    r"(?i)asperger",
    r"(?i)obsessive(?:-)?compulsive",
    r"(?i)ocd",
    r"(?i)post[- ]traumatic",
    r"(?i)ptsd",
    r"(?i)mood disorder",
    r"(?i)personality disorder",
    r"(?i)eating disorder",
    r"(?i)suicid",
    r"(?i)panic disorder",
    r"(?i)psychosis",
)

_DEFAULT_RELATION_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "drug_disease": ({"drug"}, {"disease"}),
    "disease_drug": ({"disease"}, {"drug"}),
    "disease_gene": ({"disease"}, {"gene", "protein", "gene/protein"}),
    "gene_disease": ({"gene", "protein", "gene/protein"}, {"disease"}),
    "drug_gene": ({"drug"}, {"gene", "protein", "gene/protein"}),
    "gene_drug": ({"gene", "protein", "gene/protein"}, {"drug"}),
    "drug_sideeffect": ({"drug"}, {"sideeffect", "symptom"}),
    "sideeffect_drug": ({"sideeffect", "symptom"}, {"drug"}),
}


def _truncate_text(expr: pl.Expr, limit: int, *, alias: str) -> pl.Expr:
    return (
        pl.when(expr.is_null())
        .then(pl.lit(None))
        .otherwise(expr.cast(pl.Utf8).str.slice(0, limit))
        .alias(alias)
    )


def _ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")


@dataclass
class ExtractionConfig:
    """Configuration for building a psychiatric subgraph from BioMedKG."""

    kg_path: Path = Path("data/primekg_kg.csv")
    data_dir: Path = Path("data")
    disease_features_path: Path | None = None
    drug_features_path: Path | None = None
    protein_features_path: Path | None = None
    dna_features_path: Path | None = None
    allowed_relations: set[str] | None = None
    psychiatric_patterns: Sequence[str] = field(
        default_factory=lambda: _DEFAULT_PATTERNS
    )
    metadata_truncate: int = 750
    neighbor_hops: int = 1
    include_reverse_edges: bool = False
    relation_role_constraints: Mapping[
        str, tuple[Sequence[str], Sequence[str]]
    ] | None = None

    psychiatric_mondo_ids: Sequence[str] | None = None
    psychiatric_group_labels: Sequence[str] | None = None
    psychiatric_drug_categories: Sequence[str] | None = None
    psychiatric_text_prototypes: Sequence[str] | None = None
    psychiatric_score_weights: Mapping[str, float] | None = None
    psychiatric_score_threshold: float = 0.6

    @property
    def resolved_disease_features_path(self) -> Path:
        if self.disease_features_path is not None:
            return self.disease_features_path
        return self.data_dir / "modalities" / "disease_feature_base.csv"

    @property
    def resolved_drug_features_path(self) -> Path:
        if self.drug_features_path is not None:
            return self.drug_features_path
        return self.data_dir / "modalities" / "drug_feature_base.csv"

    @property
    def resolved_protein_features_path(self) -> Path:
        if self.protein_features_path is not None:
            return self.protein_features_path
        return self.data_dir / "modalities" / "protein_aminoacid_sequence.csv"

    @property
    def resolved_dna_features_path(self) -> Path:
        if self.dna_features_path is not None:
            return self.dna_features_path
        return self.data_dir / "modalities" / "protein_dna_sequence.csv"


class EntityRelationExtractor:
    """Build a psychiatric-focused knowledge graph slice from BioMedKG."""

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config
        _ensure_file(self.config.kg_path, "PrimeKG edge list")
        _ensure_file(
            self.config.resolved_disease_features_path, "Disease modality table"
        )
        if self.config.resolved_drug_features_path.exists():
            logger.debug(
                "Using drug features from %s", self.config.resolved_drug_features_path
            )
        if self.config.resolved_protein_features_path.exists():
            logger.debug(
                "Using protein features from %s",
                self.config.resolved_protein_features_path,
            )
        if self.config.resolved_dna_features_path.exists():
            logger.debug(
                "Using DNA features from %s", self.config.resolved_dna_features_path
            )

        base_scoring = build_default_scoring_config(
            threshold=self.config.psychiatric_score_threshold
        )
        scoring_config = PsychiatricScoringConfig(
            mondo_ids=self.config.psychiatric_mondo_ids or base_scoring.mondo_ids,
            group_labels=self.config.psychiatric_group_labels
            or base_scoring.group_labels,
            drug_category_keywords=self.config.psychiatric_drug_categories
            or base_scoring.drug_category_keywords,
            text_prototypes=self.config.psychiatric_text_prototypes
            or base_scoring.text_prototypes,
            weights=self.config.psychiatric_score_weights or base_scoring.weights,
            threshold=self.config.psychiatric_score_threshold,
        )
        self.scorer = PsychiatricRelevanceScorer(config=scoring_config)
        default_constraints = {
            relation: (
                {token.lower() for token in subject_tokens},
                {token.lower() for token in object_tokens},
            )
            for relation, (
                subject_tokens,
                object_tokens,
            ) in _DEFAULT_RELATION_CONSTRAINTS.items()
        }
        if self.config.relation_role_constraints:
            for relation, pair in self.config.relation_role_constraints.items():
                subj_tokens = {token.lower() for token in pair[0]}
                obj_tokens = {token.lower() for token in pair[1]}
                default_constraints[relation.lower()] = (subj_tokens, obj_tokens)
        self.relation_constraints = default_constraints

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_subgraph(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return nodes and edges restricted to psychiatric domains."""

        disease_features = self._load_disease_features()
        drug_features = self._load_drug_features()
        protein_features = self._load_protein_features()
        dna_features = self._load_dna_features()

        base_scores = self.scorer.score_diseases(
            disease_features,
            pl.DataFrame(),
            drug_features,
        )
        candidate_frame = base_scores.filter(
            pl.col("ontology_flag")
            | pl.col("group_flag")
            | (pl.col("text_score") > 0.25)
        )
        psych_indices = candidate_frame.select("node_index").to_series().to_list()
        if not psych_indices:
            logger.warning("No psychiatric disease nodes passed hybrid scoring.")
            return self._empty_nodes(), self._empty_edges()

        edges_df = self._collect_relevant_edges(psych_indices)
        if edges_df.is_empty():
            logger.warning(
                "No edges incident to psychiatric nodes were found in the KG."
            )
            return self._empty_nodes(), self._empty_edges()

        refined_scores = self.scorer.score_diseases(
            disease_features,
            edges_df,
            drug_features,
        )
        disease_features = disease_features.join(
            refined_scores, on="node_index", how="left"
        )

        nodes_df = self._build_nodes_table(
            edges_df,
            disease_features,
            drug_features,
            protein_features,
            dna_features,
        )
        return nodes_df, edges_df

    def to_networkx(
        self, nodes_df: pl.DataFrame, edges_df: pl.DataFrame
    ) -> nx.MultiDiGraph:
        """Convert node/edge tables into a NetworkX MultiDiGraph."""

        graph = nx.MultiDiGraph()
        if nodes_df.is_empty() or edges_df.is_empty():
            return graph

        metadata_cols = [
            column
            for column in (
                "disease_metadata",
                "drug_metadata",
                "protein_metadata",
                "dna_metadata",
            )
            if column in nodes_df.columns
        ]

        for row in nodes_df.to_dicts():
            node_id = str(row["node_index"])
            merged_metadata: dict[str, str] = {}
            for column in metadata_cols:
                payload = row.get(column)
                if payload:
                    try:
                        data = json.loads(payload)
                        for key, value in data.items():
                            if value not in (None, ""):
                                merged_metadata[key] = value
                    except json.JSONDecodeError:
                        merged_metadata[column] = str(payload)
            graph.add_node(
                node_id,
                name=row.get("name", ""),
                node_type=row.get("node_type", ""),
                source=row.get("source", ""),
                node_index=row.get("node_index"),
                node_identifier=row.get("node_id"),
                is_psychiatric=bool(row.get("is_psychiatric", False)),
                metadata=json.dumps(merged_metadata, ensure_ascii=False),
            )

        for row in edges_df.to_dicts():
            src = str(row["source_index"])
            dst = str(row["target_index"])
            attributes = {
                "relation": row.get("relation", ""),
                "display_relation": row.get("display_relation", ""),
                "source_type": row.get("source_type", ""),
                "target_type": row.get("target_type", ""),
                "source_name": row.get("source_name", ""),
                "target_name": row.get("target_name", ""),
            }
            graph.add_edge(src, dst, **attributes)
            if self.config.include_reverse_edges:
                graph.add_edge(dst, src, **attributes)
        return graph

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_disease_features(self) -> pl.DataFrame:
        path = self.config.resolved_disease_features_path
        df = pl.read_csv(path)
        if "" in df.columns and "node_index" not in df.columns:
            df = df.rename({"": "node_index"})
        elif "" in df.columns:
            df = df.drop("")
        if "node_index" not in df.columns:
            raise ValueError("Disease feature table must include a node_index column")
        df = df.with_columns(pl.col("node_index").cast(pl.Int64))

        textual_columns = [
            "mondo_definition",
            "umls_description",
            "orphanet_definition",
            "orphanet_clinical_description",
            "mayo_symptoms",
            "mayo_causes",
            "mayo_risk_factors",
            "mayo_complications",
            "mayo_prevention",
            "mayo_see_doc",
        ]
        available_text_cols = [col for col in textual_columns if col in df.columns]

        agg_exprs: list[pl.Expr] = [
            pl.col("mondo_id").drop_nulls().first().alias("mondo_id"),
            pl.col("mondo_name").drop_nulls().first().alias("mondo_name"),
            pl.col("group_name_bert").drop_nulls().first().alias("group_name_bert"),
        ]
        agg_exprs.extend(
            [pl.col(col).drop_nulls().first().alias(col) for col in available_text_cols]
        )

        grouped = df.group_by("node_index").agg(agg_exprs)

        truncated_cols = [
            _truncate_text(pl.col(col), self.config.metadata_truncate, alias=col)
            for col in available_text_cols
        ]
        grouped = grouped.with_columns(truncated_cols)

        metadata_struct = pl.struct([pl.col(col) for col in available_text_cols])
        grouped = grouped.with_columns(
            metadata_struct.apply(
                lambda s: json.dumps(
                    {k: v for k, v in s.items() if v not in (None, "")},
                    ensure_ascii=False,
                )
                if any(v not in (None, "") for v in s.values())
                else None,
                return_dtype=pl.Utf8,
            ).alias("disease_metadata")
        )
        grouped = grouped.drop(available_text_cols)
        return grouped

    def _load_drug_features(self) -> pl.DataFrame:
        path = self.config.resolved_drug_features_path
        if not path.exists():
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = pl.read_csv(path)
        if "" in df.columns and "node_index" not in df.columns:
            df = df.rename({"": "node_index"})
        elif "" in df.columns:
            df = df.drop("")
        if "node_index" not in df.columns:
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = df.with_columns(pl.col("node_index").cast(pl.Int64))
        textual_columns = [
            "description",
            "indication",
            "mechanism_of_action",
            "protein_binding",
            "pharmacodynamics",
            "state",
            "category",
            "group",
            "pathway",
        ]
        agg_exprs = [pl.col("drugbank_ids").drop_nulls().first().alias("drugbank_ids")]
        agg_exprs.append(
            pl.col("generic_name").drop_nulls().first().alias("generic_name")
        )
        agg_exprs.extend(
            [pl.col(col).drop_nulls().first().alias(col) for col in textual_columns]
        )
        agg_exprs.append(pl.col("smiles").drop_nulls().first().alias("smiles"))

        grouped = df.group_by("node_index").agg(agg_exprs)
        truncated_cols = [
            _truncate_text(pl.col(col), self.config.metadata_truncate, alias=col)
            for col in textual_columns
        ]
        grouped = grouped.with_columns(truncated_cols)
        metadata_struct = pl.struct([pl.col(col) for col in textual_columns])
        grouped = grouped.with_columns(
            metadata_struct.apply(
                lambda s: json.dumps(
                    {k: v for k, v in s.items() if v not in (None, "")},
                    ensure_ascii=False,
                )
                if any(v not in (None, "") for v in s.values())
                else None,
                return_dtype=pl.Utf8,
            ).alias("drug_metadata")
        )
        grouped = grouped.drop(textual_columns)
        return grouped

    def _load_protein_features(self) -> pl.DataFrame:
        path = self.config.resolved_protein_features_path
        if not path.exists():
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = pl.read_csv(path)
        if "" in df.columns and "node_index" not in df.columns:
            df = df.rename({"": "node_index"})
        elif "" in df.columns:
            df = df.drop("")
        if "node_index" not in df.columns:
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = df.with_columns(pl.col("node_index").cast(pl.Int64))
        df = df.with_columns(
            pl.col("protein_seq").cast(pl.Utf8).str.len_chars().alias("protein_seq_len")
        )
        agg_exprs = [
            pl.col("protein_id").drop_nulls().first().alias("protein_id"),
            pl.col("protein_name").drop_nulls().first().alias("protein_name"),
            pl.col("fasta_id").drop_nulls().first().alias("fasta_id"),
            pl.col("fasta_description").drop_nulls().first().alias("fasta_description"),
            pl.col("ncbi_summary").drop_nulls().first().alias("ncbi_summary"),
            pl.col("protein_seq_len").drop_nulls().first().alias("protein_seq_len"),
        ]
        grouped = df.group_by("node_index").agg(agg_exprs)
        grouped = grouped.with_columns(
            _truncate_text(
                pl.col("fasta_description"),
                self.config.metadata_truncate,
                alias="fasta_description",
            ),
            _truncate_text(
                pl.col("ncbi_summary"),
                self.config.metadata_truncate,
                alias="ncbi_summary",
            ),
        )
        metadata_struct = pl.struct(
            [
                pl.col("fasta_description"),
                pl.col("ncbi_summary"),
                pl.col("protein_seq_len"),
            ]
        )
        grouped = grouped.with_columns(
            metadata_struct.apply(
                lambda s: json.dumps(
                    {k: v for k, v in s.items() if v not in (None, "")},
                    ensure_ascii=False,
                )
                if any(v not in (None, "") for v in s.values())
                else None,
                return_dtype=pl.Utf8,
            ).alias("protein_metadata")
        )
        grouped = grouped.drop(["fasta_description", "ncbi_summary", "protein_seq_len"])
        return grouped

    def _load_dna_features(self) -> pl.DataFrame:
        path = self.config.resolved_dna_features_path
        if not path.exists():
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = pl.read_csv(path)
        if "" in df.columns and "node_index" not in df.columns:
            df = df.rename({"": "node_index"})
        elif "" in df.columns:
            df = df.drop("")
        if "node_index" not in df.columns:
            return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})
        df = df.with_columns(pl.col("node_index").cast(pl.Int64))
        if "dna_sequence" in df.columns:
            df = df.with_columns(
                pl.col("dna_sequence")
                .cast(pl.Utf8)
                .str.len_chars()
                .alias("dna_seq_len")
            )
        else:
            df = df.with_columns(pl.lit(None).alias("dna_seq_len"))
        agg_exprs = []
        value_cols: list[str] = []
        if "gene_symbol" in df.columns:
            agg_exprs.append(
                pl.col("gene_symbol").drop_nulls().first().alias("gene_symbol")
            )
            value_cols.append("gene_symbol")
        if "dna_seq_len" in df.columns:
            agg_exprs.append(
                pl.col("dna_seq_len").drop_nulls().first().alias("dna_seq_len")
            )
            value_cols.append("dna_seq_len")

        grouped = (
            df.groupby("node_index").agg(agg_exprs)
            if agg_exprs
            else df.groupby("node_index").agg([])
        )

        if value_cols:
            metadata_struct = pl.struct([pl.col(col) for col in value_cols])
            grouped = grouped.with_columns(
                metadata_struct.apply(
                    lambda s: json.dumps(
                        {k: v for k, v in s.items() if v not in (None, "")},
                        ensure_ascii=False,
                    )
                    if any(v not in (None, "") for v in s.values())
                    else None,
                    return_dtype=pl.Utf8,
                ).alias("dna_metadata")
            )
            grouped = grouped.drop(value_cols)
        else:
            grouped = grouped.with_columns(pl.lit(None).alias("dna_metadata"))
        return grouped

    def _collect_relevant_edges(self, psych_indices: Iterable[int]) -> pl.DataFrame:
        psych_series = pl.Series(list(psych_indices), dtype=pl.Int64)
        edges_lazy = pl.scan_csv(
            self.config.kg_path,
            infer_schema_length=0,
            dtypes={
                "x_id": pl.Utf8,
                "y_id": pl.Utf8,
                "x_name": pl.Utf8,
                "y_name": pl.Utf8,
                "x_type": pl.Utf8,
                "y_type": pl.Utf8,
                "x_source": pl.Utf8,
                "y_source": pl.Utf8,
                "relation": pl.Utf8,
                "display_relation": pl.Utf8,
            },
        )
        filter_expr = pl.col("x_index").cast(pl.Int64).is_in(psych_series) | pl.col(
            "y_index"
        ).cast(pl.Int64).is_in(psych_series)
        if self.config.allowed_relations:
            filter_expr = filter_expr & pl.col("relation").is_in(
                sorted(self.config.allowed_relations)
            )
        collected = (
            edges_lazy.filter(filter_expr)
            .with_columns(
                pl.col("x_index").cast(pl.Int64),
                pl.col("y_index").cast(pl.Int64),
            )
            .collect(streaming=True)
        )
        if collected.is_empty():
            return self._empty_edges()
        renamed = collected.rename(
            {
                "x_index": "source_index",
                "x_id": "source_id",
                "x_type": "source_type",
                "x_name": "source_name",
                "x_source": "source_dataset",
                "y_index": "target_index",
                "y_id": "target_id",
                "y_type": "target_type",
                "y_name": "target_name",
                "y_source": "target_dataset",
            }
        )
        mask = [
            self._relation_allowed(
                row.get("relation"),
                row.get("source_type"),
                row.get("target_type"),
            )
            for row in renamed.iter_rows(named=True)
        ]
        if not mask or not any(mask):
            return self._empty_edges()
        renamed = renamed.with_columns(pl.Series("_allowed_mask", mask))
        renamed = renamed.filter(pl.col("_allowed_mask")).drop("_allowed_mask")
        return renamed

    def _build_nodes_table(
        self,
        edges_df: pl.DataFrame,
        disease_features: pl.DataFrame,
        drug_features: pl.DataFrame,
        protein_features: pl.DataFrame,
        dna_features: pl.DataFrame,
    ) -> pl.DataFrame:
        src_nodes = edges_df.select(
            pl.col("source_index").alias("node_index"),
            pl.col("source_id").alias("node_id"),
            pl.col("source_name").alias("name"),
            pl.col("source_type").alias("node_type"),
            pl.col("source_dataset").alias("source"),
        )
        dst_nodes = edges_df.select(
            pl.col("target_index").alias("node_index"),
            pl.col("target_id").alias("node_id"),
            pl.col("target_name").alias("name"),
            pl.col("target_type").alias("node_type"),
            pl.col("target_dataset").alias("source"),
        )
        nodes = pl.concat([src_nodes, dst_nodes]).unique(
            subset=["node_index"], keep="first"
        )

        nodes = nodes.join(disease_features, on="node_index", how="left")
        if not drug_features.is_empty():
            nodes = nodes.join(drug_features, on="node_index", how="left")
        if not protein_features.is_empty():
            nodes = nodes.join(protein_features, on="node_index", how="left")
        if not dna_features.is_empty():
            nodes = nodes.join(dna_features, on="node_index", how="left")

        default_columns = {
            "is_psychiatric": pl.lit(False),
            "psy_score": pl.lit(0.0),
            "psy_evidence": pl.lit("[]"),
            "ontology_flag": pl.lit(False),
            "group_flag": pl.lit(False),
            "drug_flag": pl.lit(False),
            "text_score": pl.lit(0.0),
        }
        for column, expr in default_columns.items():
            if column not in nodes.columns:
                nodes = nodes.with_columns(expr.alias(column))
            else:
                if column == "psy_evidence":
                    nodes = nodes.with_columns(pl.col(column).fill_null("[]"))
                elif column in {"psy_score", "text_score"}:
                    nodes = nodes.with_columns(pl.col(column).fill_null(0.0))
                else:
                    nodes = nodes.with_columns(pl.col(column).fill_null(False))

        return nodes.sort("node_index")

    def _relation_allowed(
        self, relation: str | None, src_type: str | None, dst_type: str | None
    ) -> bool:
        if relation is None:
            return True
        constraint = self.relation_constraints.get(relation.lower())
        if not constraint:
            return True
        subj_allowed, obj_allowed = constraint
        return self._type_matches(src_type, subj_allowed) and self._type_matches(
            dst_type, obj_allowed
        )

    @staticmethod
    def _type_matches(node_type: str | None, allowed: set[str]) -> bool:
        if not allowed:
            return True
        if not node_type:
            return False
        node_type = node_type.lower()
        if node_type in allowed:
            return True
        tokens = {node_type}
        tokens.update(part for part in re.split(r"[\s/_,;:-]+", node_type) if part)
        return any(token in allowed for token in tokens)

    # ------------------------------------------------------------------
    # Empty-table helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_nodes() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "node_index": pl.Series([], dtype=pl.Int64),
                "node_id": pl.Series([], dtype=pl.Utf8),
                "name": pl.Series([], dtype=pl.Utf8),
                "node_type": pl.Series([], dtype=pl.Utf8),
                "source": pl.Series([], dtype=pl.Utf8),
                "is_psychiatric": pl.Series([], dtype=pl.Boolean),
                "psy_score": pl.Series([], dtype=pl.Float64),
                "psy_evidence": pl.Series([], dtype=pl.Utf8),
                "ontology_flag": pl.Series([], dtype=pl.Boolean),
                "group_flag": pl.Series([], dtype=pl.Boolean),
                "drug_flag": pl.Series([], dtype=pl.Boolean),
                "text_score": pl.Series([], dtype=pl.Float64),
            }
        )

    @staticmethod
    def _empty_edges() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "relation": pl.Series([], dtype=pl.Utf8),
                "display_relation": pl.Series([], dtype=pl.Utf8),
                "source_index": pl.Series([], dtype=pl.Int64),
                "source_id": pl.Series([], dtype=pl.Utf8),
                "source_type": pl.Series([], dtype=pl.Utf8),
                "source_name": pl.Series([], dtype=pl.Utf8),
                "source_dataset": pl.Series([], dtype=pl.Utf8),
                "target_index": pl.Series([], dtype=pl.Int64),
                "target_id": pl.Series([], dtype=pl.Utf8),
                "target_type": pl.Series([], dtype=pl.Utf8),
                "target_name": pl.Series([], dtype=pl.Utf8),
                "target_dataset": pl.Series([], dtype=pl.Utf8),
            }
        )


__all__ = ["ExtractionConfig", "EntityRelationExtractor"]
