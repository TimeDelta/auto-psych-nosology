#!/usr/bin/env python3
"""Augment an existing knowledge graph with ontology-driven hubs.

This script implements the first six steps of the ontology augmentation plan:

1. Load the base graph (GraphML or JSON node-link) into NetworkX.
2. Import ontology terms (e.g., symptoms, pathways, circuits) as new nodes.
3. Map existing graph nodes to ontology entries via string matching or explicit
   annotation tables.
4. Add edges from entities to ontology terms, and connect ontology terms to
   their hierarchical parents.
5. Preserve ontology hierarchy edges to create multi-level structure.
6. Add similarity edges between entities that share ontology memberships.

The script is intentionally data-format agnostic: ontology metadata and
annotation tables are expected as CSV/TSV files whose columns can be specified
via CLI arguments. Minimal defaults are provided for common layouts.

Example usage (assuming CSVs with appropriate columns):

    python3 augment_graph_with_ontologies.py \
        --graph data/prebuilt-kg.graphml \
        --output data/prebuilt-kg.augmented.graphml \
        --ontology-terms hpo=data/hpo_terms.csv \
        --ontology-annotations hpo=data/hpo_annotations.csv \
        --entity-name-column name \
        --entity-id-column node_id

The script prints a short summary of added nodes/edges for sanity checking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas optional
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_graph_readers = {
    ".graphml": nx.read_graphml,
    ".gexf": nx.read_gexf,
    ".gpickle": nx.read_gpickle,
    ".pickle": nx.read_gpickle,
    ".pkl": nx.read_gpickle,
}


def _load_base_graph(path: pathlib.Path) -> nx.Graph:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    ext = path.suffix.lower()
    if ext in _graph_readers:
        graph = _graph_readers[ext](path)
    elif ext == ".json":
        from networkx.readwrite import json_graph

        data = json.loads(path.read_text(encoding="utf-8"))
        graph = json_graph.node_link_graph(data)
    else:
        raise ValueError(f"Unsupported graph format: {ext}")

    if graph.is_multigraph():
        # collapse multigraph by summing weights
        collapsed = nx.Graph()
        for u, v, data in graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            if collapsed.has_edge(u, v):
                collapsed[u][v]["weight"] += weight
            else:
                collapsed.add_edge(u, v, weight=weight)
        for node, data in graph.nodes(data=True):
            collapsed.add_node(node, **data)
        graph = collapsed
    if graph.is_directed():
        graph = graph.to_undirected(reciprocal=False)
    return graph


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _read_table(path: pathlib.Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError(
            "pandas is required to load ontology tables. Install pandas or provide "
            "already-processed JSON annotations."
        )
    sep = "," if path.suffix.lower() in {".csv", ".txt"} else "\t"
    return pd.read_csv(path, sep=sep, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Ontology ingestion
# ---------------------------------------------------------------------------


class OntologyBundle:
    """Container for one ontology's terms, hierarchy, and annotations."""

    def __init__(
        self,
        name: str,
        term_table: "pd.DataFrame",
        annotations: "pd.DataFrame",
        term_id_col: str,
        term_name_col: str,
        parent_col: Optional[str],
        synonym_col: Optional[str],
        annotation_entity_col: str,
        annotation_term_col: str,
    ) -> None:
        self.name = name
        self.term_table = term_table
        self.annotations = annotations
        self.term_id_col = term_id_col
        self.term_name_col = term_name_col
        self.parent_col = parent_col
        self.synonym_col = synonym_col
        self.annotation_entity_col = annotation_entity_col
        self.annotation_term_col = annotation_term_col

    def iter_terms(self) -> Iterable[Tuple[str, str, List[str]]]:
        for _, row in self.term_table.iterrows():
            term_id = row[self.term_id_col].strip()
            if not term_id:
                continue
            name = row.get(self.term_name_col, term_id).strip()
            synonyms: List[str] = []
            if self.synonym_col and self.synonym_col in row:
                raw = row[self.synonym_col]
                if isinstance(raw, str) and raw:
                    synonyms = [syn.strip() for syn in re.split(r"[;|]", raw) if syn]
            yield term_id, name, synonyms

    def iter_parent_edges(self) -> Iterable[Tuple[str, str]]:
        if not self.parent_col or self.parent_col not in self.term_table.columns:
            return []
        parent_edges = []
        for _, row in self.term_table.iterrows():
            term_id = row[self.term_id_col].strip()
            raw_parents = row[self.parent_col]
            if not term_id or not isinstance(raw_parents, str) or not raw_parents:
                continue
            for parent in re.split(r"[;|]", raw_parents):
                parent = parent.strip()
                if parent:
                    parent_edges.append((term_id, parent))
        return parent_edges

    def iter_annotations(self) -> Iterable[Tuple[str, str]]:
        for _, row in self.annotations.iterrows():
            entity = row[self.annotation_entity_col].strip()
            term = row[self.annotation_term_col].strip()
            if entity and term:
                yield entity, term


def load_ontology_bundle(
    name: str,
    terms_path: pathlib.Path,
    annotations_path: pathlib.Path,
    term_id_col: str,
    term_name_col: str,
    parent_col: Optional[str],
    synonym_col: Optional[str],
    annotation_entity_col: str,
    annotation_term_col: str,
) -> OntologyBundle:
    terms_df = _read_table(terms_path)
    annotations_df = _read_table(annotations_path)
    missing_cols = [
        col for col in [term_id_col, term_name_col] if col not in terms_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Ontology '{name}' terms file missing columns: {', '.join(missing_cols)}"
        )
    if annotation_entity_col not in annotations_df.columns:
        raise ValueError(
            f"Ontology '{name}' annotations file missing column: {annotation_entity_col}"
        )
    if annotation_term_col not in annotations_df.columns:
        raise ValueError(
            f"Ontology '{name}' annotations file missing column: {annotation_term_col}"
        )
    return OntologyBundle(
        name,
        terms_df,
        annotations_df,
        term_id_col,
        term_name_col,
        parent_col,
        synonym_col,
        annotation_entity_col,
        annotation_term_col,
    )


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


def _normalise_keywords(raw: Sequence[str]) -> set[str]:
    return {kw.strip().lower() for kw in raw if kw and kw.strip()}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return False


def _parse_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


DEFAULT_PSYCH_KEYWORDS = _normalise_keywords(
    [
        "psych",
        "psychi",
        "mental",
        "mood",
        "depress",
        "anxiety",
        "schiz",
        "bipolar",
        "autism",
        "adhd",
        "neurodevelopment",
        "posttraumatic",
        "obsessive",
        "compulsive",
        "suicid",
        "psychosis",
        "psychotic",
    ]
)


class GraphAugmenter:
    def __init__(
        self,
        graph: nx.Graph,
        *,
        psych_keywords: Sequence[str],
        psy_score_threshold: float,
        allow_non_psy_entities: bool,
    ) -> None:
        self.graph = graph
        self.node_index: Dict[str, str] = {}
        for node, data in graph.nodes(data=True):
            node_id = str(node)
            self.node_index[_norm(node_id)] = node_id
            for key in ("canonical_name", "name"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    self.node_index.setdefault(_norm(value), node_id)
        self.psych_keywords = _normalise_keywords(psych_keywords)
        self.psy_score_threshold = psy_score_threshold
        self.allow_non_psy_entities = allow_non_psy_entities

    def _term_is_psych(self, text_candidates: Sequence[str]) -> bool:
        if not self.psych_keywords:
            return True
        for text in text_candidates:
            lowered = text.lower()
            if any(keyword in lowered for keyword in self.psych_keywords):
                return True
        return False

    def _entity_is_psych(self, node_id: str) -> bool:
        if self.allow_non_psy_entities:
            return True
        attrs = self.graph.nodes.get(node_id, {})
        if _parse_bool(attrs.get("is_psychiatric")):
            return True
        score = _parse_float(attrs.get("psy_score"))
        if score is not None and score >= self.psy_score_threshold:
            return True
        return False

    def add_ontology_terms(
        self, bundle: OntologyBundle, node_type_label: Optional[str]
    ) -> Dict[str, str]:
        added = {}
        for term_id, name, synonyms in bundle.iter_terms():
            if not self._term_is_psych([name, *synonyms, term_id]):
                continue
            key = _norm(term_id)
            if term_id in self.graph:
                added[term_id] = term_id
                continue
            node_attrs = {
                "name": name,
                "canonical_name": name,
                "ontology": bundle.name,
                "ontology_id": term_id,
            }
            if node_type_label:
                node_attrs["node_type"] = node_type_label
            self.graph.add_node(term_id, **node_attrs)
            added[term_id] = term_id
            for alias in [name, *synonyms]:
                self.node_index[_norm(alias)] = term_id
        return added

    def connect_hierarchy(self, edges: Iterable[Tuple[str, str]], relation: str) -> int:
        count = 0
        for child, parent in edges:
            if child not in self.graph or parent not in self.graph:
                continue
            if self.graph.has_edge(child, parent):
                continue
            self.graph.add_edge(child, parent, relation=relation, weight=1.0)
            count += 1
        return count

    def map_entities(
        self,
        bundle: OntologyBundle,
        added_terms: Dict[str, str],
        entity_id_col: Optional[str],
    ) -> Dict[str, List[str]]:
        entity_to_terms: Dict[str, List[str]] = defaultdict(list)
        if bundle.term_name_col:
            for term_id, name, synonyms in bundle.iter_terms():
                canonical = added_terms.get(term_id, term_id)
                self.node_index.setdefault(_norm(name), canonical)
                for syn in synonyms:
                    self.node_index.setdefault(_norm(syn), canonical)
        for entity, term in bundle.iter_annotations():
            term_node = added_terms.get(term) or self.node_index.get(_norm(term))
            if term_node is None:
                continue
            entity_node = None
            if entity_id_col:
                entity_node = str(entity) if entity in self.graph else None
            if entity_node is None:
                entity_node = self.node_index.get(_norm(entity))
            if entity_node is None or entity_node not in self.graph:
                continue
            if not self._entity_is_psych(entity_node):
                continue
            entity_to_terms[entity_node].append(term_node)
        for entity_node, term_nodes in entity_to_terms.items():
            for term_node in term_nodes:
                if self.graph.has_edge(entity_node, term_node):
                    continue
                self.graph.add_edge(
                    entity_node,
                    term_node,
                    relation=f"maps_to_{bundle.name}",
                    weight=1.0,
                )
        return entity_to_terms

    def add_similarity_edges(
        self, entity_to_terms: Mapping[str, Sequence[str]], relation: str
    ) -> int:
        count = 0
        reverse: Dict[str, List[str]] = defaultdict(list)
        for entity, terms in entity_to_terms.items():
            for term in terms:
                reverse[term].append(entity)
        for term, entities in reverse.items():
            if len(entities) < 2:
                continue
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    u, v = entities[i], entities[j]
                    if self.graph.has_edge(u, v):
                        continue
                    self.graph.add_edge(
                        u,
                        v,
                        relation=relation,
                        weight=1.0,
                        via_term=term,
                    )
                    count += 1
        return count


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def parse_kv_pair(pair: str) -> Tuple[str, str]:
    if "=" not in pair:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH for ontology arguments, got '{pair}'"
        )
    key, value = pair.split("=", 1)
    return key.strip(), value.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment a knowledge graph with ontology-based hubs",
    )
    parser.add_argument(
        "--graph", type=pathlib.Path, required=True, help="Input graph (GraphML/JSON)"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True, help="Output GraphML path"
    )
    parser.add_argument(
        "--ontology-terms",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Ontology term tables (CSV/TSV) keyed by short name",
    )
    parser.add_argument(
        "--ontology-annotations",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Ontology annotation tables (CSV/TSV) keyed by short name",
    )
    parser.add_argument(
        "--term-id-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override term ID column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--term-name-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override term name column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--parent-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override parent column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--synonym-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override synonym column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--annotation-entity-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override entity column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--annotation-term-column",
        type=parse_kv_pair,
        nargs="*",
        default=[],
        help="Override term column per ontology (NAME=column)",
    )
    parser.add_argument(
        "--entity-id-column",
        default=None,
        help="Column in annotations containing explicit entity node IDs (optional)",
    )
    parser.add_argument(
        "--default-node-type",
        default="ontology",
        help="node_type value assigned to ontology nodes when ingesting",
    )
    parser.add_argument(
        "--similarity-relation",
        default="shared_ontology_term",
        help="Relation label for entity-entity similarity edges",
    )
    parser.add_argument(
        "--psych-keywords",
        nargs="*",
        default=sorted(DEFAULT_PSYCH_KEYWORDS),
        help=(
            "Lowercase substrings that must appear in ontology term names/synonyms "
            "to be added. Provide an empty list to disable filtering."
        ),
    )
    parser.add_argument(
        "--psy-score-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum psy_score required for an entity to gain ontology edges when "
            "is_psychiatric is False."
        ),
    )
    parser.add_argument(
        "--allow-non-psy-entities",
        action="store_true",
        help="Allow ontology edges to attach to entities lacking psychiatric flags/scores.",
    )
    return parser


def dict_from_pairs(pairs: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    return {k: v for k, v in pairs}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    term_paths = dict_from_pairs(args.ontology_terms)
    annotation_paths = dict_from_pairs(args.ontology_annotations)
    if not term_paths:
        parser.error("At least one --ontology-terms NAME=PATH must be provided")
    if set(term_paths) != set(annotation_paths):
        missing = set(term_paths) ^ set(annotation_paths)
        parser.error(
            "Ontology terms/annotations mismatch; provide both for names: "
            + ", ".join(sorted(missing))
        )

    term_id_override = dict_from_pairs(args.term_id_column)
    term_name_override = dict_from_pairs(args.term_name_column)
    parent_override = dict_from_pairs(args.parent_column)
    synonym_override = dict_from_pairs(args.synonym_column)
    entity_col_override = dict_from_pairs(args.annotation_entity_column)
    anno_term_override = dict_from_pairs(args.annotation_term_column)

    graph = _load_base_graph(args.graph)
    augmenter = GraphAugmenter(
        graph,
        psych_keywords=args.psych_keywords,
        psy_score_threshold=args.psy_score_threshold,
        allow_non_psy_entities=args.allow_non_psy_entities,
    )

    total_added_nodes = 0
    total_hierarchy_edges = 0
    total_entity_edges = 0
    total_similarity_edges = 0

    for name, terms_path in term_paths.items():
        annotations_path = annotation_paths[name]
        bundle = load_ontology_bundle(
            name,
            pathlib.Path(terms_path),
            pathlib.Path(annotations_path),
            term_id_override.get(name, "id"),
            term_name_override.get(name, "name"),
            parent_override.get(name, None),
            synonym_override.get(name, None),
            entity_col_override.get(name, "entity"),
            anno_term_override.get(name, "term"),
        )
        added_terms = augmenter.add_ontology_terms(bundle, args.default_node_type)
        total_added_nodes += len(added_terms)
        total_hierarchy_edges += augmenter.connect_hierarchy(
            bundle.iter_parent_edges(), relation=f"{name}_is_a"
        )
        entity_to_terms = augmenter.map_entities(
            bundle,
            added_terms,
            entity_id_col=args.entity_id_column,
        )
        total_entity_edges += sum(len(v) for v in entity_to_terms.values())
        total_similarity_edges += augmenter.add_similarity_edges(
            entity_to_terms, relation=args.similarity_relation
        )

    print(
        "Augmentation summary: added {nodes} ontology nodes, {hier} hierarchy edges, "
        "{entity} entity-term edges, {sim} similarity edges.".format(
            nodes=total_added_nodes,
            hier=total_hierarchy_edges,
            entity=total_entity_edges,
            sim=total_similarity_edges,
        )
    )

    nx.write_graphml(graph, args.output)
    print(f"Augmented graph saved to {args.output}")


if __name__ == "__main__":
    main()
