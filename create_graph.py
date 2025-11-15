"""Pipeline entry point for constructing a psychiatric BioMedKG subgraph."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree.ElementTree import Element, ElementTree, SubElement

import networkx as nx
import polars as pl

from augment_graph_with_ontologies import (
    DEFAULT_PSYCH_KEYWORDS as AUGMENT_DEFAULT_PSYCH_KEYWORDS,
    GraphAugmenter,
    load_ontology_bundle,
)
from nosology_filters import should_drop_nosology_node
from psychiatry_scoring import (
    PsychiatricRelevanceScorer,
    PsychiatricScoringConfig,
    build_default_scoring_config,
)
from remove_stranded_nodes import remove_stranded_nodes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_TEXT_TOKEN_LIMIT_PER_FIELD = 32
_TEXT_TOKEN_LIMIT_TOTAL = 128


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


def _tokenize_text(value: str) -> List[str]:
    if not value:
        return []
    return _TOKEN_RE.findall(value.lower())


def _unique_tokens(tokens: Iterable[str], limit: int) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def _extract_text_tokens(raw: str, per_field_limit: int) -> List[str]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw

    if isinstance(payload, dict):
        source_iter: Iterable[Any] = payload.values()
    elif isinstance(payload, list):
        source_iter = payload
    else:
        source_iter = [payload]

    candidates: List[str] = []
    for value in source_iter:
        if isinstance(value, str):
            candidates.extend(_tokenize_text(value))
    return _unique_tokens(candidates, per_field_limit)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value in (None, ""):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


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


def _parse_name_value_arg(pair: str) -> tuple[str, str]:
    if "=" not in pair:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=VALUE format, received '{pair}'."
        )
    key, value = pair.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise argparse.ArgumentTypeError(
            f"Both name and value must be non-empty for '{pair}'."
        )
    return key, value


def _pairs_to_path_dict(pairs: Sequence[tuple[str, str]]) -> Dict[str, Path]:
    return {name: Path(path_value).expanduser() for name, path_value in pairs}


def _pairs_to_str_dict(pairs: Sequence[tuple[str, str]]) -> Dict[str, str]:
    return {name: value for name, value in pairs}


def iter_json_array(path: Path, *, chunk_size: int = 4 * 1024 * 1024):
    """Stream very large JSON arrays without loading the entire file."""

    with path.open("r", encoding="utf-8") as handle:
        started = False
        depth = 0
        in_string = False
        escaped = False
        current: list[str] = []
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if not started:
                    if ch in " \t\r\n":
                        continue
                    if ch != "[":
                        raise ValueError(f"Expected '[' at start of {path}")
                    started = True
                    continue
                if depth == 0:
                    if ch in " \t\r\n,":
                        continue
                    if ch == "]":
                        return
                    if ch == "{":
                        depth = 1
                        current = [ch]
                        continue
                    raise ValueError(
                        f"Unexpected character '{ch}' while looking for object in {path}"
                    )
                # depth > 0 -> inside an object
                current.append(ch)
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                    continue
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        payload = "".join(current)
                        current = []
                        yield json.loads(payload)
                # other characters handled via append
        if depth != 0:
            raise ValueError(f"JSON array in {path} ended mid-object")


@dataclass
class ExtractionConfig:
    """Configuration for building a psychiatric subgraph from BioMedKG."""

    kg_path: Path = Path("data/primekg_kg.csv")
    data_dir: Path = Path("data")
    ikraph_root: Path | None = None
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
        self._ikraph_root = self._resolve_ikraph_root()
        self._using_ikraph = self._ikraph_root is not None
        self._ikraph_nodes_path: Path | None = None
        self._ikraph_db_path: Path | None = None
        self._ikraph_pubmed_path: Path | None = None
        self._ikraph_relations: dict[str, Mapping[str, object]] | None = None
        if self._using_ikraph:
            self._init_ikraph_resources()
        else:
            self._validate_primekg_inputs()

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
    # Dataset detection helpers
    # ------------------------------------------------------------------
    def _resolve_ikraph_root(self) -> Path | None:
        if self.config.ikraph_root is not None:
            return self.config.ikraph_root
        kg_path = self.config.kg_path
        candidate_dirs: list[Path] = []
        if kg_path.is_dir():
            candidate_dirs.append(kg_path)
        else:
            name = kg_path.name.lower()
            if name in {
                "ner_id_dict_cap_final.json",
                "dbrelations.json",
                "pubmedlist.json",
                "reltypeint.json",
            }:
                candidate_dirs.append(kg_path.parent)
        for directory in candidate_dirs:
            nodes_path = directory / "NER_ID_dict_cap_final.json"
            if nodes_path.exists():
                return directory
        return None

    def _init_ikraph_resources(self) -> None:
        assert self._ikraph_root is not None
        nodes_path = self._ikraph_root / "NER_ID_dict_cap_final.json"
        relation_path = self._ikraph_root / "RelTypeInt.json"
        _ensure_file(nodes_path, "iKraph node table")
        _ensure_file(relation_path, "iKraph relation schema")
        self._ikraph_nodes_path = nodes_path
        db_path = self._ikraph_root / "DBRelations.json"
        if db_path.exists():
            self._ikraph_db_path = db_path
        else:
            logger.warning(
                "iKraph DBRelations.json not found at %s; database edges will be skipped",
                db_path,
            )
        pubmed_path = self._ikraph_root / "PubMedList.json"
        if pubmed_path.exists():
            self._ikraph_pubmed_path = pubmed_path
        else:
            logger.warning(
                "iKraph PubMedList.json not found at %s; literature edges will be skipped",
                pubmed_path,
            )
        self._ikraph_relations = self._load_ikraph_relation_index(relation_path)
        logger.info("Using iKraph knowledge graph located at %s", self._ikraph_root)

    def _validate_primekg_inputs(self) -> None:
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

    @staticmethod
    def _load_ikraph_relation_index(path: Path) -> dict[str, Mapping[str, object]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        relation_map: dict[str, Mapping[str, object]] = {}
        for row in payload:
            relation_id = str(row.get("intRep"))
            relation_map[relation_id] = {
                "name": row.get("relType", relation_id),
                "corType": row.get("corType", []),
                "precision": row.get("relPrec"),
            }
        return relation_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_subgraph(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return nodes and edges restricted to psychiatric domains."""

        if self._using_ikraph:
            return self._build_ikraph_subgraph()
        return self._build_primekg_subgraph()

    def _build_primekg_subgraph(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        disease_features = self._load_disease_features()
        drug_features = self._load_drug_features()
        protein_features = self._load_protein_features()
        dna_features = self._load_dna_features()

        base_scores = self.scorer.score_diseases(
            disease_features,
            pl.DataFrame(),
            drug_features,
        )
        psych_indices = self._extract_psych_indices(
            base_scores, disease_features.height
        )
        if not psych_indices:
            logger.warning("No psychiatric disease nodes passed hybrid scoring.")
            return self._empty_nodes(), self._empty_edges()

        edges_df = self._collect_relevant_edges(psych_indices)
        logger.debug(
            "Collected %d edges incident to psychiatric seeds", edges_df.height
        )
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
        nodes_df = self._propagate_psychiatric_scores(nodes_df, edges_df)
        logger.debug(
            "Built node table with %d nodes prior to nosology filtering",
            nodes_df.height,
        )
        nodes_df = self._filter_nosology_nodes(nodes_df)
        logger.debug("Nodes after nosology filter: %d", nodes_df.height)
        return nodes_df, edges_df

    # ------------------------------------------------------------------
    # iKraph ingestion
    # ------------------------------------------------------------------
    def _build_ikraph_subgraph(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        if self._ikraph_nodes_path is None:
            logger.error("iKraph nodes path was not initialised")
            return self._empty_nodes(), self._empty_edges()

        disease_features, drug_features = self._load_ikraph_modalities()
        if disease_features.is_empty():
            logger.warning("No disease entries were found in the iKraph node table.")
            return self._empty_nodes(), self._empty_edges()

        base_scores = self.scorer.score_diseases(
            disease_features,
            pl.DataFrame(),
            drug_features,
        )
        psych_indices = self._extract_psych_indices(
            base_scores, disease_features.height
        )
        if not psych_indices:
            logger.warning(
                "iKraph ingestion produced zero psychiatric candidates; aborting."
            )
            return self._empty_nodes(), self._empty_edges()

        edges_df, required_node_ids = self._collect_ikraph_edges(psych_indices)
        if edges_df.is_empty():
            logger.warning(
                "No edges incident to the psychiatric iKraph nodes were discovered."
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

        name_lookup, type_lookup, dataset_lookup = self._load_ikraph_node_metadata(
            required_node_ids
        )
        edges_df = self._enrich_edges_with_metadata(
            edges_df, name_lookup, type_lookup, dataset_lookup
        )
        edges_df = self._enforce_relation_constraints(edges_df)
        if edges_df.is_empty():
            logger.warning(
                "Relation-role constraints removed all candidate edges for iKraph."
            )
            return self._empty_nodes(), self._empty_edges()

        protein_features = self._empty_feature_table()
        dna_features = self._empty_feature_table()
        nodes_df = self._build_nodes_table(
            edges_df,
            disease_features,
            drug_features,
            protein_features,
            dna_features,
        )
        nodes_df = self._propagate_psychiatric_scores(nodes_df, edges_df)
        nodes_df = self._filter_nosology_nodes(nodes_df)
        logger.debug(
            "iKraph pipeline retained %d nodes and %d edges after filtering",
            nodes_df.height,
            edges_df.height,
        )
        return nodes_df, edges_df

    def _extract_psych_indices(
        self, scores: pl.DataFrame, total_diseases: int
    ) -> list[int]:
        if scores.is_empty():
            return []
        candidate_frame = scores.filter(
            pl.col("ontology_flag")
            | pl.col("group_flag")
            | (pl.col("text_score") > 0.25)
        )
        psych_indices = candidate_frame.select("node_index").to_series().to_list()
        logger.debug(
            "Hybrid scoring produced %d candidate diseases (total diseases: %d)",
            len(psych_indices),
            total_diseases,
        )
        return psych_indices

    @staticmethod
    def _empty_feature_table() -> pl.DataFrame:
        return pl.DataFrame({"node_index": pl.Series([], dtype=pl.Int64)})

    def _load_ikraph_modalities(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        assert self._ikraph_nodes_path is not None
        disease_rows: list[dict[str, object]] = []
        drug_rows: list[dict[str, object]] = []
        for node in iter_json_array(self._ikraph_nodes_path):
            node_index = self._safe_int(node.get("biokdeid"))
            if node_index is None:
                continue
            node_type = self._normalized_value(node.get("type"))
            node_subtype = self._normalized_value(node.get("subtype"))
            official = self._normalized_value(node.get("official name"))
            common = self._normalized_value(node.get("common name"))
            mondo_id = self._normalized_value(node.get("id"))
            species = node.get("species") or []
            species_id = (
                self._normalized_value(species[0]) if len(species) > 0 else None
            )
            species_name = (
                self._normalized_value(species[1]) if len(species) > 1 else None
            )
            metadata = self._build_metadata_blob(
                {
                    "type": node_type,
                    "subtype": node_subtype,
                    "official_name": official,
                    "common_name": common,
                    "species_id": species_id,
                    "species_name": species_name,
                }
            )
            display_name = common or official or f"node_{node_index}"
            if (node_type or "").lower() == "disease":
                disease_rows.append(
                    {
                        "node_index": node_index,
                        "node_id": mondo_id or str(node_index),
                        "name": display_name,
                        "mondo_id": mondo_id
                        if mondo_id and mondo_id.startswith("MONDO")
                        else None,
                        "mondo_name": display_name,
                        "group_name_bert": node_subtype,
                        "mondo_definition": official,
                        "umls_description": common,
                        "disease_metadata": metadata,
                    }
                )
            if self._is_drug_like(node_type, node_subtype):
                drug_rows.append(
                    {
                        "node_index": node_index,
                        "generic_name": display_name,
                        "description": official or common,
                        "indication": common,
                        "mechanism_of_action": None,
                        "category": node_subtype or node_type,
                        "group": node_type,
                        "pathway": None,
                        "drug_metadata": metadata,
                    }
                )
        if disease_rows:
            disease_df = pl.DataFrame(disease_rows)
        else:
            disease_df = self._empty_feature_table()
        if drug_rows:
            drug_df = pl.DataFrame(drug_rows)
        else:
            drug_df = pl.DataFrame(
                {
                    "node_index": pl.Series([], dtype=pl.Int64),
                    "generic_name": pl.Series([], dtype=pl.Utf8),
                    "description": pl.Series([], dtype=pl.Utf8),
                    "indication": pl.Series([], dtype=pl.Utf8),
                    "mechanism_of_action": pl.Series([], dtype=pl.Utf8),
                    "category": pl.Series([], dtype=pl.Utf8),
                    "group": pl.Series([], dtype=pl.Utf8),
                    "pathway": pl.Series([], dtype=pl.Utf8),
                    "drug_metadata": pl.Series([], dtype=pl.Utf8),
                }
            )
        return disease_df, drug_df

    def _propagate_psychiatric_scores(
        self, nodes_df: pl.DataFrame, edges_df: pl.DataFrame
    ) -> pl.DataFrame:
        if nodes_df.is_empty() or edges_df.is_empty():
            return nodes_df
        if "psy_score" not in nodes_df.columns:
            return nodes_df
        positive_scores = nodes_df.select("node_index", "psy_score").filter(
            pl.col("psy_score") > 0
        )
        if positive_scores.is_empty():
            return nodes_df

        source_contrib = (
            edges_df.join(
                positive_scores,
                left_on="source_index",
                right_on="node_index",
                how="inner",
            )
            .select(pl.col("target_index").alias("node_index"), pl.col("psy_score"))
            .with_columns(pl.col("psy_score").cast(pl.Float64))
        )
        target_contrib = (
            edges_df.join(
                positive_scores,
                left_on="target_index",
                right_on="node_index",
                how="inner",
            )
            .select(pl.col("source_index").alias("node_index"), pl.col("psy_score"))
            .with_columns(pl.col("psy_score").cast(pl.Float64))
        )
        contributions = pl.concat([source_contrib, target_contrib])
        if contributions.is_empty():
            return nodes_df

        aggregated = contributions.group_by("node_index").agg(
            pl.col("psy_score").mean().alias("neighbor_psy_mean"),
            pl.col("psy_score").max().alias("neighbor_psy_max"),
            pl.len().alias("neighbor_psy_count"),
        )
        nodes_df = nodes_df.join(aggregated, on="node_index", how="left")

        neighbor_threshold = min(0.2, float(self.config.psychiatric_score_threshold))
        nodes_df = nodes_df.with_columns(
            pl.when(pl.col("psy_score") > 0)
            .then(pl.col("psy_score"))
            .otherwise(pl.col("neighbor_psy_mean").fill_null(0.0))
            .alias("psy_score"),
            (
                pl.col("is_psychiatric")
                | (
                    pl.col("psy_score")
                    >= float(self.config.psychiatric_score_threshold)
                )
                | (pl.col("neighbor_psy_max").fill_null(0.0) >= neighbor_threshold)
            ).alias("is_psychiatric"),
        )
        nodes_df = nodes_df.drop(
            [
                column
                for column in (
                    "neighbor_psy_mean",
                    "neighbor_psy_max",
                    "neighbor_psy_count",
                )
                if column in nodes_df.columns
            ]
        )
        return nodes_df

    def _collect_ikraph_edges(
        self, psych_indices: Iterable[int]
    ) -> tuple[pl.DataFrame, set[int]]:
        psych_set = {int(idx) for idx in psych_indices if idx is not None}
        touched_nodes: set[int] = set(psych_set)
        if not psych_set:
            return self._empty_edges(), touched_nodes

        relation_lookup = self._ikraph_relations or {}
        allowed_relations = (
            {label.lower() for label in self.config.allowed_relations}
            if self.config.allowed_relations
            else None
        )
        edge_rows: list[dict[str, object]] = []

        if self._ikraph_db_path is not None:
            for entry in iter_json_array(self._ikraph_db_path):
                src = self._safe_int(entry.get("node_one_id"))
                dst = self._safe_int(entry.get("node_two_id"))
                if src is None or dst is None:
                    continue
                if src not in psych_set and dst not in psych_set:
                    continue
                relation_id = str(entry.get("relationship_type"))
                relation_info = relation_lookup.get(relation_id, {})
                relation_name = str(relation_info.get("name", relation_id))
                relation_key = relation_name.lower()
                if allowed_relations and relation_key not in allowed_relations:
                    continue
                edge_rows.append(
                    {
                        "relation": relation_key,
                        "display_relation": relation_name,
                        "source_index": src,
                        "source_id": entry.get("node_one_id") or str(src),
                        "source_type": entry.get("node_one_type"),
                        "source_name": entry.get("node_one_name"),
                        "source_dataset": entry.get("source") or "iKraph_DB",
                        "target_index": dst,
                        "target_id": entry.get("node_two_id") or str(dst),
                        "target_type": entry.get("node_two_type"),
                        "target_name": entry.get("node_two_name"),
                        "target_dataset": entry.get("source") or "iKraph_DB",
                        "probability": entry.get("prob"),
                        "score": entry.get("score"),
                        "edge_source": entry.get("source") or "iKraph_DB",
                        "direction": entry.get("direction"),
                    }
                )
                touched_nodes.update((src, dst))

        if self._ikraph_pubmed_path is not None:
            for entry in iter_json_array(self._ikraph_pubmed_path):
                composite_id = entry.get("id") or ""
                parts = composite_id.split(".")
                if len(parts) < 6:
                    continue
                src = self._safe_int(parts[0])
                dst = self._safe_int(parts[1])
                if src is None or dst is None:
                    continue
                if src not in psych_set and dst not in psych_set:
                    continue
                relation_id = parts[2]
                relation_info = relation_lookup.get(relation_id, {})
                relation_name = str(relation_info.get("name", relation_id))
                relation_key = relation_name.lower()
                if allowed_relations and relation_key not in allowed_relations:
                    continue
                stats = entry.get("list") or []
                score_values = [float(item[0]) for item in stats if item]
                probability_values = [
                    float(item[2])
                    for item in stats
                    if isinstance(item, list) and len(item) > 2
                ]
                novelty_hits = sum(
                    1
                    for item in stats
                    if isinstance(item, list) and len(item) > 3 and item[3]
                )
                edge_rows.append(
                    {
                        "relation": relation_key,
                        "display_relation": relation_name,
                        "source_index": src,
                        "source_id": parts[0],
                        "source_type": None,
                        "source_name": None,
                        "source_dataset": "iKraph_PubMed",
                        "target_index": dst,
                        "target_id": parts[1],
                        "target_type": None,
                        "target_name": None,
                        "target_dataset": "iKraph_PubMed",
                        "probability": max(probability_values)
                        if probability_values
                        else None,
                        "score": max(score_values) if score_values else None,
                        "edge_source": "iKraph_PubMed",
                        "direction": parts[4],
                        "correlation": parts[3],
                        "method": parts[5],
                        "evidence_count": len(stats),
                        "novelty_hits": novelty_hits,
                    }
                )
                touched_nodes.update((src, dst))

        if not edge_rows:
            return self._empty_edges(), touched_nodes
        edges_df = pl.DataFrame(edge_rows).with_columns(
            pl.col("source_index").cast(pl.Int64),
            pl.col("target_index").cast(pl.Int64),
        )
        return edges_df, touched_nodes

    def _load_ikraph_node_metadata(
        self, node_ids: set[int]
    ) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
        if not node_ids:
            return {}, {}, {}
        assert self._ikraph_nodes_path is not None
        remaining = set(node_ids)
        name_lookup: dict[int, str] = {}
        type_lookup: dict[int, str] = {}
        dataset_lookup: dict[int, str] = {}
        for node in iter_json_array(self._ikraph_nodes_path):
            node_index = self._safe_int(node.get("biokdeid"))
            if node_index is None or node_index not in remaining:
                continue
            node_type = self._normalized_value(node.get("type"))
            node_subtype = self._normalized_value(node.get("subtype"))
            official = self._normalized_value(node.get("official name"))
            common = self._normalized_value(node.get("common name"))
            name_lookup[node_index] = common or official or f"node_{node_index}"
            type_lookup[node_index] = node_type or node_subtype or ""
            dataset_lookup[node_index] = node_subtype or node_type or "iKraph"
            remaining.remove(node_index)
            if not remaining:
                break
        if remaining:
            preview = ", ".join(str(value) for value in list(sorted(remaining))[:5])
            logger.debug(
                "Missing metadata for %d iKraph nodes (e.g., %s)",
                len(remaining),
                preview,
            )
        return name_lookup, type_lookup, dataset_lookup

    def _enrich_edges_with_metadata(
        self,
        edges_df: pl.DataFrame,
        name_lookup: Mapping[int, str],
        type_lookup: Mapping[int, str],
        dataset_lookup: Mapping[int, str],
    ) -> pl.DataFrame:
        if edges_df.is_empty():
            return edges_df
        edges_df = self._fill_missing_edge_values(
            edges_df, "source_name", "source_index", name_lookup
        )
        edges_df = self._fill_missing_edge_values(
            edges_df, "target_name", "target_index", name_lookup
        )
        edges_df = self._fill_missing_edge_values(
            edges_df, "source_type", "source_index", type_lookup
        )
        edges_df = self._fill_missing_edge_values(
            edges_df, "target_type", "target_index", type_lookup
        )
        edges_df = self._fill_missing_edge_values(
            edges_df, "source_dataset", "source_index", dataset_lookup
        )
        edges_df = self._fill_missing_edge_values(
            edges_df, "target_dataset", "target_index", dataset_lookup
        )
        return edges_df

    @staticmethod
    def _fill_missing_edge_values(
        df: pl.DataFrame,
        column: str,
        index_column: str,
        lookup: Mapping[int, str],
    ) -> pl.DataFrame:
        if not lookup or column not in df.columns:
            return df
        mapped = pl.col(index_column).map_dict(lookup, return_dtype=pl.Utf8)
        return df.with_columns(
            pl.when(pl.col(column).is_null() | (pl.col(column) == ""))
            .then(mapped)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def _enforce_relation_constraints(self, edges_df: pl.DataFrame) -> pl.DataFrame:
        if edges_df.is_empty():
            return edges_df
        mask = [
            self._relation_allowed(
                row.get("relation"),
                row.get("source_type"),
                row.get("target_type"),
            )
            for row in edges_df.iter_rows(named=True)
        ]
        if not mask or not any(mask):
            return self._empty_edges()
        return (
            edges_df.with_columns(pl.Series("_allowed_mask", mask))
            .filter(pl.col("_allowed_mask"))
            .drop("_allowed_mask")
        )

    @staticmethod
    def _safe_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalized_value(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.upper() == "NA":
            return None
        return text

    @staticmethod
    def _build_metadata_blob(entries: Mapping[str, object]) -> str | None:
        cleaned = {
            key: value
            for key, value in entries.items()
            if value not in (None, "", [], {})
        }
        if not cleaned:
            return None
        return json.dumps(cleaned, ensure_ascii=False)

    @staticmethod
    def _is_drug_like(node_type: str | None, node_subtype: str | None) -> bool:
        for token in (node_type, node_subtype):
            if token and "drug" in token.lower():
                return True
        return False

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

        scalar_attrs = (
            "psy_score",
            "psy_evidence",
            "ontology_flag",
            "group_flag",
            "drug_flag",
            "text_score",
            "name_keyword_flag",
        )
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
            extra = {
                key: row.get(key) for key in scalar_attrs if key in nodes_df.columns
            }
            if "psy_score" in extra and extra["psy_score"] is None:
                extra["psy_score"] = 0.0
            if "text_score" in extra and extra["text_score"] is None:
                extra["text_score"] = 0.0
            graph.add_node(
                node_id,
                name=row.get("name", ""),
                node_type=row.get("node_type", ""),
                source=row.get("source", ""),
                node_index=row.get("node_index"),
                node_identifier=row.get("node_id"),
                is_psychiatric=bool(row.get("is_psychiatric", False)),
                metadata=json.dumps(merged_metadata, ensure_ascii=False),
                **extra,
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
            metadata_struct.map_elements(
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
            metadata_struct.map_elements(
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
            metadata_struct.map_elements(
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
            df.group_by("node_index").agg(agg_exprs)
            if agg_exprs
            else df.group_by("node_index").agg([])
        )

        if value_cols:
            metadata_struct = pl.struct([pl.col(col) for col in value_cols])
            grouped = grouped.with_columns(
                metadata_struct.map_elements(
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
        return self._enforce_relation_constraints(renamed)

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
            "name_keyword_flag": pl.lit(False),
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

    def _filter_nosology_nodes(self, nodes_df: pl.DataFrame) -> pl.DataFrame:
        if nodes_df.is_empty():
            return nodes_df
        struct_expr = pl.struct([pl.col(col) for col in nodes_df.columns]).map_elements(
            lambda row: not should_drop_nosology_node(row), return_dtype=pl.Boolean
        )
        filtered = nodes_df.filter(struct_expr)
        removed = nodes_df.height - filtered.height
        if removed:
            logger.debug("Nosology filter removed %d nodes", removed)
        return filtered

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


RELATION_PRIOR_DEFAULT: Dict[str, float] = {
    "drug_disease": 1.0,
    "disease_drug": 1.0,
    "disease_gene": 0.6,
    "gene_disease": 0.6,
    "drug_gene": 0.5,
    "gene_drug": 0.5,
    "drug_sideeffect": 0.7,
    "sideeffect_drug": 0.7,
}


def _clean_value(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.replace("\x00", " ")
    return json.dumps(value, ensure_ascii=False)


def sanitize_graph(graph: nx.Graph) -> None:
    for key, value in list(graph.graph.items()):
        graph.graph[key] = _clean_value(value)
    for _, data in graph.nodes(data=True):
        for key in list(data.keys()):
            data[key] = _clean_value(data[key])
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = graph.edges(data=True)
    for edge in edge_iter:
        data = edge[-1]
        for key in list(data.keys()):
            data[key] = _clean_value(data[key])


@dataclass
class PipelineConfig:
    """Run configuration for the knowledge graph pipeline."""

    kg_path: Path = Path("data/primekg_kg.csv")
    data_dir: Path = Path("data")
    ikraph_dir: Path | None = None
    output_prefix: str = "psychiatric_biomedkg"
    output_dir: Path = Path("data")
    allowed_relations: Sequence[str] | None = None
    psychiatric_patterns: Sequence[str] | None = None
    metadata_truncate: int = 750
    neighbor_hops: int = 1
    include_reverse_edges: bool = False
    relation_priors: Mapping[str, float] | None = None
    ontology_terms: Dict[str, Path] = field(default_factory=dict)
    ontology_annotations: Dict[str, Path] = field(default_factory=dict)
    ontology_term_id_columns: Dict[str, str] = field(default_factory=dict)
    ontology_term_name_columns: Dict[str, str] = field(default_factory=dict)
    ontology_parent_columns: Dict[str, str] = field(default_factory=dict)
    ontology_synonym_columns: Dict[str, str] = field(default_factory=dict)
    ontology_annotation_entity_columns: Dict[str, str] = field(default_factory=dict)
    ontology_annotation_term_columns: Dict[str, str] = field(default_factory=dict)
    ontology_node_type: Optional[str] = "ontology"
    ontology_similarity_relation: str = "shared_ontology_term"
    ontology_entity_id_column: Optional[str] = None
    ontology_psych_keywords: Sequence[str] | None = None
    ontology_psy_score_threshold: float = 0.0
    ontology_allow_non_psy_entities: bool = False
    ontology_output_suffix: Optional[str] = ".augmented.graphml"
    ontology_weighted_suffix: Optional[str] = ".augmented.weighted.graphml"

    def to_extraction_config(self) -> ExtractionConfig:
        patterns = (
            tuple(self.psychiatric_patterns) if self.psychiatric_patterns else None
        )
        return ExtractionConfig(
            kg_path=self.kg_path,
            data_dir=self.data_dir,
            ikraph_root=self.ikraph_dir,
            allowed_relations=set(self.allowed_relations)
            if self.allowed_relations
            else None,
            psychiatric_patterns=patterns or tuple(),
            metadata_truncate=self.metadata_truncate,
            neighbor_hops=self.neighbor_hops,
            include_reverse_edges=self.include_reverse_edges,
        )

    @property
    def resolved_output_prefix(self) -> Path:
        return self.output_dir / self.output_prefix

    def wants_ontology_augmentation(self) -> bool:
        return bool(self.ontology_terms)


def _graphml_attr_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "long"
    if isinstance(value, float):
        return "double"
    return "string"


def _format_graphml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_graphml(graph: nx.Graph, path: Path) -> None:
    sanitize_graph(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_attr_types = {}
    edge_attr_types = {}
    for _, data in graph.nodes(data=True):
        for key, value in data.items():
            if key not in node_attr_types and value not in (None, ""):
                node_attr_types[key] = _graphml_attr_type(value)
    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = graph.edges(data=True)
    for edge in edge_iter:
        data = edge[-1]
        for key, value in data.items():
            if key not in edge_attr_types and value not in (None, ""):
                edge_attr_types[key] = _graphml_attr_type(value)

    root = Element("graphml", attrib={"xmlns": "http://graphml.graphdrawing.org/xmlns"})
    for key, attr_type in node_attr_types.items():
        SubElement(
            root,
            "key",
            attrib={
                "id": f"n_{key}",
                "for": "node",
                "attr.name": key,
                "attr.type": attr_type,
            },
        )
    for key, attr_type in edge_attr_types.items():
        SubElement(
            root,
            "key",
            attrib={
                "id": f"e_{key}",
                "for": "edge",
                "attr.name": key,
                "attr.type": attr_type,
            },
        )

    graph_elem = SubElement(
        root,
        "graph",
        attrib={
            "id": "G",
            "edgedefault": "directed" if graph.is_directed() else "undirected",
        },
    )
    for node, data in graph.nodes(data=True):
        node_elem = SubElement(graph_elem, "node", attrib={"id": str(node)})
        for key, value in data.items():
            if value in (None, ""):
                continue
            SubElement(
                node_elem, "data", attrib={"key": f"n_{key}"}
            ).text = _format_graphml_value(value)

    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = graph.edges(data=True)
    for edge in edge_iter:
        if graph.is_multigraph():
            u, v, _, data = edge
        else:
            u, v, data = edge
        edge_elem = SubElement(
            graph_elem, "edge", attrib={"source": str(u), "target": str(v)}
        )
        for key, value in data.items():
            if value in (None, ""):
                continue
            SubElement(
                edge_elem, "data", attrib={"key": f"e_{key}"}
            ).text = _format_graphml_value(value)

    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class KnowledgeGraphPipeline:
    """Orchestrate BioMedKG ingestion and artifact writing."""

    def __init__(self, extractor: EntityRelationExtractor | None = None) -> None:
        self._extractor = extractor

    def run(self, config: PipelineConfig) -> Dict[str, int]:
        extractor = self._extractor or EntityRelationExtractor(
            config.to_extraction_config()
        )
        nodes_df, edges_df = extractor.build_subgraph()
        summary = {"n_nodes": nodes_df.height, "n_edges": edges_df.height}
        if nodes_df.is_empty() or edges_df.is_empty():
            logger.warning("No subgraph produced; skipping serialization.")
            return summary

        graph = extractor.to_networkx(nodes_df, edges_df)

        relation_priors = {
            **RELATION_PRIOR_DEFAULT,
            **{k.lower(): float(v) for k, v in (config.relation_priors or {}).items()},
        }
        weighted = self._build_weighted_projection(graph, relation_priors)
        self._write_outputs(
            config,
            nodes_df,
            edges_df,
            graph,
            weighted,
            relation_priors,
        )
        return summary

    @staticmethod
    def _build_weighted_projection(
        graph: nx.MultiDiGraph, relation_priors: Mapping[str, float]
    ) -> nx.Graph:
        weighted = nx.Graph()
        for node, data in graph.nodes(data=True):
            weighted.add_node(node, **data)
        for u, v, data in graph.edges(data=True):
            relation = (data.get("relation") or "").lower()
            prior = relation_priors.get(relation, 1.0)
            psy_u = float(graph.nodes[u].get("psy_score", 0.0) or 0.0)
            psy_v = float(graph.nodes[v].get("psy_score", 0.0) or 0.0)
            avg_relevance = (psy_u + psy_v) / 2.0
            relevance_factor = max(avg_relevance, 0.1)
            weight_value = prior * relevance_factor
            if weighted.has_edge(u, v):
                weighted[u][v]["weight"] += weight_value
            else:
                weighted.add_edge(
                    u,
                    v,
                    weight=weight_value,
                    relation=data.get("relation", ""),
                )
        return weighted

    def _write_outputs(
        self,
        config: PipelineConfig,
        nodes_df: pl.DataFrame,
        edges_df: pl.DataFrame,
        graph: nx.MultiDiGraph,
        weighted: nx.Graph,
        relation_priors: Mapping[str, float],
    ) -> None:
        output_prefix = config.resolved_output_prefix
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        nodes_path = output_prefix.with_suffix(".nodes.parquet")
        edges_path = output_prefix.with_suffix(".rels.parquet")
        graph_path = output_prefix.with_suffix(".graphml")
        weighted_path = output_prefix.with_suffix(".weighted.graphml")

        nodes_df.write_parquet(nodes_path)
        edges_df.write_parquet(edges_path)
        write_graphml(graph, graph_path)
        self._prune_graphml_file(graph_path)
        write_graphml(weighted, weighted_path)
        self._prune_graphml_file(weighted_path)
        logger.info(
            "Wrote %d nodes, %d edges to %s",
            nodes_df.height,
            edges_df.height,
            output_prefix,
        )
        if config.wants_ontology_augmentation():
            self._write_ontology_outputs(
                config,
                graph,
                relation_priors,
                output_prefix,
            )

    def _write_ontology_outputs(
        self,
        config: PipelineConfig,
        base_graph: nx.MultiDiGraph,
        relation_priors: Mapping[str, float],
        output_prefix: Path,
    ) -> None:
        augmented_graph = base_graph.copy()
        stats = self._augment_graph_with_ontologies(augmented_graph, config)
        suffix = config.ontology_output_suffix or ""
        if suffix:
            augmented_path = output_prefix.with_suffix(suffix)
            write_graphml(augmented_graph, augmented_path)
            self._prune_graphml_file(augmented_path)
            logger.info(
                "Ontology augmentation wrote %s (added %d nodes / %d hierarchy / %d entity / %d similarity edges)",
                augmented_path,
                stats["nodes"],
                stats["hierarchy_edges"],
                stats["entity_edges"],
                stats["similarity_edges"],
            )
        if config.ontology_weighted_suffix:
            augmented_weighted = self._build_weighted_projection(
                augmented_graph, relation_priors
            )
            weighted_path = output_prefix.with_suffix(config.ontology_weighted_suffix)
            write_graphml(augmented_weighted, weighted_path)
            self._prune_graphml_file(weighted_path)
            logger.info("Ontology-weighted projection wrote %s", weighted_path)

    def _augment_graph_with_ontologies(
        self, graph: nx.MultiDiGraph, config: PipelineConfig
    ) -> Dict[str, int]:
        psych_keywords = (
            [kw.strip().lower() for kw in config.ontology_psych_keywords if kw.strip()]
            if config.ontology_psych_keywords is not None
            else sorted(AUGMENT_DEFAULT_PSYCH_KEYWORDS)
        )
        augmenter = GraphAugmenter(
            graph,
            psych_keywords=psych_keywords,
            psy_score_threshold=config.ontology_psy_score_threshold,
            allow_non_psy_entities=config.ontology_allow_non_psy_entities,
        )
        stats = {
            "nodes": 0,
            "hierarchy_edges": 0,
            "entity_edges": 0,
            "similarity_edges": 0,
        }
        for name, terms_path in config.ontology_terms.items():
            annotations_path = config.ontology_annotations.get(name)
            if annotations_path is None:
                logger.warning(
                    "Skipping ontology '%s' because annotations were not provided",
                    name,
                )
                continue
            bundle = load_ontology_bundle(
                name,
                terms_path,
                annotations_path,
                config.ontology_term_id_columns.get(name, "id"),
                config.ontology_term_name_columns.get(name, "name"),
                config.ontology_parent_columns.get(name) or None,
                config.ontology_synonym_columns.get(name) or None,
                config.ontology_annotation_entity_columns.get(name, "entity"),
                config.ontology_annotation_term_columns.get(name, "term"),
            )
            added_terms = augmenter.add_ontology_terms(
                bundle, config.ontology_node_type
            )
            stats["nodes"] += len(added_terms)
            stats["hierarchy_edges"] += augmenter.connect_hierarchy(
                bundle.iter_parent_edges(), relation=f"{name}_is_a"
            )
            entity_to_terms = augmenter.map_entities(
                bundle,
                added_terms,
                entity_id_col=config.ontology_entity_id_column,
            )
            stats["entity_edges"] += sum(len(v) for v in entity_to_terms.values())
            stats["similarity_edges"] += augmenter.add_similarity_edges(
                entity_to_terms, relation=config.ontology_similarity_relation
            )
        return stats

    @staticmethod
    def _prune_graphml_file(path: Path) -> None:
        if not path.exists():
            return
        try:
            original, removed, remaining = remove_stranded_nodes(
                path, output_path=path, keep_original=False
            )
            if removed:
                logger.info(
                    "Pruned %d stranded nodes from %s (now %d)",
                    removed,
                    path,
                    remaining,
                )
        except Exception as exc:
            logger.warning("Failed to prune stranded nodes in %s (%s)", path, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a psychiatric BioMedKG subgraph"
    )
    parser.add_argument("--kg-path", type=Path, default=PipelineConfig.kg_path)
    parser.add_argument("--data-dir", type=Path, default=PipelineConfig.data_dir)
    parser.add_argument(
        "--ikraph-dir",
        type=Path,
        default=None,
        help="Optional path to the iKraph_full directory; overrides automatic detection.",
    )
    parser.add_argument(
        "--output-prefix",
        default=PipelineConfig.output_prefix,
        help="File name prefix for parquet/graphml outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PipelineConfig.output_dir,
        help="Directory for outputs (default: data)",
    )
    parser.add_argument(
        "--allowed-relation",
        action="append",
        dest="allowed_relations",
        help="Limit relations to the provided list (can be repeated)",
    )
    parser.add_argument(
        "--psychiatric-pattern",
        action="append",
        dest="psychiatric_patterns",
        help="Custom regex used to flag psychiatric diseases (can be repeated)",
    )
    parser.add_argument(
        "--metadata-truncate",
        type=int,
        default=PipelineConfig.metadata_truncate,
        help="Maximum characters to retain for long text fields",
    )
    parser.add_argument(
        "--include-reverse",
        action="store_true",
        help="Also add reverse edges to the directed graph",
    )
    parser.add_argument(
        "--ontology-terms",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Optional ontology term tables (CSV/TSV) to ingest (NAME=PATH)",
    )
    parser.add_argument(
        "--ontology-annotations",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Annotation files matching --ontology-terms (NAME=PATH)",
    )
    parser.add_argument(
        "--ontology-term-id-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Override the term ID column for a specific ontology",
    )
    parser.add_argument(
        "--ontology-term-name-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Override the term name column for a specific ontology",
    )
    parser.add_argument(
        "--ontology-parent-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Optional parent column for ontology hierarchies",
    )
    parser.add_argument(
        "--ontology-synonym-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Optional synonym column used for text matching",
    )
    parser.add_argument(
        "--ontology-annotation-entity-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Override the entity ID column in the annotation table",
    )
    parser.add_argument(
        "--ontology-annotation-term-column",
        type=_parse_name_value_arg,
        nargs="*",
        default=[],
        metavar="NAME=COL",
        help="Override the term ID column in the annotation table",
    )
    parser.add_argument(
        "--ontology-node-type",
        default="ontology",
        help="node_type assigned to ingested ontology hubs (default: ontology)",
    )
    parser.add_argument(
        "--ontology-similarity-relation",
        default="shared_ontology_term",
        help="Relation label for entity-entity edges induced by shared ontology terms",
    )
    parser.add_argument(
        "--ontology-entity-id-column",
        default=None,
        help="If provided, entity annotations will attempt exact node ID matches",
    )
    parser.add_argument(
        "--ontology-psych-keyword",
        action="append",
        dest="ontology_psych_keywords",
        help="Substring filter applied to ontology term names; repeat to add multiple keywords",
    )
    parser.add_argument(
        "--ontology-psy-score-threshold",
        type=float,
        default=0.0,
        help="Minimum psy_score for non-flagged entities to receive ontology edges",
    )
    parser.add_argument(
        "--ontology-allow-non-psy-entities",
        action="store_true",
        help="Allow ontology edges to attach even if entities lack psychiatric evidence",
    )
    parser.add_argument(
        "--ontology-output-suffix",
        default=".augmented.graphml",
        help="Suffix for augmented GraphML (set to empty string to skip writing)",
    )
    parser.add_argument(
        "--ontology-weighted-suffix",
        default=".augmented.weighted.graphml",
        help="Suffix for augmented weighted GraphML (set to empty string to skip)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    term_map = _pairs_to_path_dict(args.ontology_terms)
    annotation_map = _pairs_to_path_dict(args.ontology_annotations)
    if term_map and set(term_map) != set(annotation_map):
        missing = set(term_map) ^ set(annotation_map)
        parser.error(
            "Ontology terms/annotations mismatch; provide both for names: "
            + ", ".join(sorted(missing))
        )
    args.ontology_terms = term_map
    args.ontology_annotations = annotation_map
    args.ontology_term_id_column = _pairs_to_str_dict(args.ontology_term_id_column)
    args.ontology_term_name_column = _pairs_to_str_dict(args.ontology_term_name_column)
    args.ontology_parent_column = _pairs_to_str_dict(args.ontology_parent_column)
    args.ontology_synonym_column = _pairs_to_str_dict(args.ontology_synonym_column)
    args.ontology_annotation_entity_column = _pairs_to_str_dict(
        args.ontology_annotation_entity_column
    )
    args.ontology_annotation_term_column = _pairs_to_str_dict(
        args.ontology_annotation_term_column
    )
    if args.ontology_psych_keywords:
        cleaned_keywords = [
            keyword.strip().lower()
            for keyword in args.ontology_psych_keywords
            if keyword and keyword.strip()
        ]
        args.ontology_psych_keywords = cleaned_keywords or None
    else:
        args.ontology_psych_keywords = None
    if args.ontology_output_suffix == "":
        args.ontology_output_suffix = None
    if args.ontology_weighted_suffix == "":
        args.ontology_weighted_suffix = None
    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    pipeline = KnowledgeGraphPipeline()
    config = PipelineConfig(
        kg_path=args.kg_path,
        data_dir=args.data_dir,
        ikraph_dir=args.ikraph_dir,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        allowed_relations=args.allowed_relations,
        psychiatric_patterns=args.psychiatric_patterns,
        include_reverse_edges=args.include_reverse,
        metadata_truncate=args.metadata_truncate,
        ontology_terms=args.ontology_terms,
        ontology_annotations=args.ontology_annotations,
        ontology_term_id_columns=args.ontology_term_id_column,
        ontology_term_name_columns=args.ontology_term_name_column,
        ontology_parent_columns=args.ontology_parent_column,
        ontology_synonym_columns=args.ontology_synonym_column,
        ontology_annotation_entity_columns=args.ontology_annotation_entity_column,
        ontology_annotation_term_columns=args.ontology_annotation_term_column,
        ontology_node_type=args.ontology_node_type,
        ontology_similarity_relation=args.ontology_similarity_relation,
        ontology_entity_id_column=args.ontology_entity_id_column,
        ontology_psych_keywords=args.ontology_psych_keywords,
        ontology_psy_score_threshold=args.ontology_psy_score_threshold,
        ontology_allow_non_psy_entities=args.ontology_allow_non_psy_entities,
        ontology_output_suffix=args.ontology_output_suffix,
        ontology_weighted_suffix=args.ontology_weighted_suffix,
    )
    pipeline.run(config)
