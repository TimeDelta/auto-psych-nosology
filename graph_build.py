"""Dataframe aggregation and NetworkX graph construction utilities."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from models import PaperExtraction
from text_normalization import normalize_name


def accum_extractions(
    extractions: Iterable[PaperExtraction],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []
    for extraction in extractions:
        paper_rows.append(
            {
                "paper_id": extraction.paper_id,
                "doi": extraction.doi,
                "title": extraction.title,
                "year": extraction.year,
                "venue": extraction.venue,
            }
        )
        for node in extraction.nodes:
            node_rows.append(
                {
                    "paper_id": extraction.paper_id,
                    "lemma": node.lemma,
                    "canonical_name": normalize_name(node.canonical_name),
                    "node_type": node.node_type,
                    "synonyms": json.dumps(node.synonyms, ensure_ascii=False),
                    "normalizations": json.dumps(
                        node.normalizations, ensure_ascii=False
                    ),
                }
            )
        for relation in extraction.relations:
            relation_rows.append(
                {
                    "paper_id": extraction.paper_id,
                    "subject": normalize_name(relation.subject),
                    "predicate": relation.predicate,
                    "object": normalize_name(relation.obj),
                    "directionality": relation.directionality,
                    "evidence_span": relation.evidence_span,
                    "confidence": relation.confidence,
                    "qualifiers": json.dumps(relation.qualifiers, ensure_ascii=False),
                }
            )
    nodes_df = pd.DataFrame(node_rows).drop_duplicates()
    relations_df = pd.DataFrame(relation_rows).drop_duplicates()
    papers_df = pd.DataFrame(paper_rows).drop_duplicates()

    if not relations_df.empty:
        support = relations_df.groupby(
            ["subject", "predicate", "object"], as_index=False
        ).agg(
            paper_ids=("paper_id", lambda series: sorted(set(series))),
            n_papers=("paper_id", "nunique"),
        )
        relations_df = relations_df.merge(
            support, on=["subject", "predicate", "object"], how="left"
        )
    return nodes_df, relations_df, papers_df


def build_multilayer_graph(
    nodes_df: pd.DataFrame, relations_df: pd.DataFrame
) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    node_types = (
        nodes_df.groupby(["canonical_name", "node_type"])
        .size()
        .reset_index(name="n")
        .sort_values(["canonical_name", "n"], ascending=[True, False])
    )
    canonical_to_type = (
        node_types.drop_duplicates("canonical_name")
        .set_index("canonical_name")["node_type"]
        .to_dict()
    )
    for name, node_type in canonical_to_type.items():
        graph.add_node(name, node_type=node_type, synonyms=[], normalizations={})
    for _, row in nodes_df.iterrows():
        name = row["canonical_name"]
        graph.nodes[name]["synonyms"] = sorted(
            set(graph.nodes[name]["synonyms"]) | set(json.loads(row["synonyms"]))
        )
        graph.nodes[name]["normalizations"] = {
            **graph.nodes[name]["normalizations"],
            **json.loads(row["normalizations"]),
        }
        if "lemma" in row and isinstance(row["lemma"], str) and row["lemma"]:
            graph.nodes[name]["lemma"] = row["lemma"]
    for _, row in relations_df.iterrows():
        graph.add_edge(
            row["subject"],
            row["object"],
            predicate=row["predicate"],
            paper_id=row["paper_id"],
            directionality=row["directionality"],
            evidence_span=row["evidence_span"],
            confidence=float(row["confidence"]),
            qualifiers=json.loads(row["qualifiers"]),
            n_papers=int(row.get("n_papers", 1) or 1),
            paper_ids=(
                row["paper_ids"]
                if isinstance(row["paper_ids"], list)
                else json.loads(row["paper_ids"])
                if isinstance(row["paper_ids"], str)
                else []
            ),
        )
    return graph


def project_to_weighted_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    weighted = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        weight = float(data.get("confidence", 1.0))
        if weighted.has_edge(u, v):
            weighted[u][v]["weight"] += weight
        else:
            weighted.add_edge(u, v, weight=weight)
    return weighted


__all__ = [
    "accum_extractions",
    "build_multilayer_graph",
    "project_to_weighted_graph",
]
