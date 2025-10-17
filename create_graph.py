"""Pipeline entry point for constructing a psychiatric BioMedKG subgraph."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence
from xml.etree.ElementTree import Element, ElementTree, SubElement

import networkx as nx
import polars as pl

from nlp_extraction import EntityRelationExtractor, ExtractionConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
    output_prefix: str = "psychiatric_biomedkg"
    output_dir: Path = Path("data")
    allowed_relations: Sequence[str] | None = None
    psychiatric_patterns: Sequence[str] | None = None
    metadata_truncate: int = 750
    neighbor_hops: int = 1
    include_reverse_edges: bool = False
    relation_priors: Mapping[str, float] | None = None

    def to_extraction_config(self) -> ExtractionConfig:
        patterns = (
            tuple(self.psychiatric_patterns) if self.psychiatric_patterns else None
        )
        return ExtractionConfig(
            kg_path=self.kg_path,
            data_dir=self.data_dir,
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
        self._write_outputs(config, nodes_df, edges_df, graph, weighted)
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
        write_graphml(weighted, weighted_path)
        logger.info(
            "Wrote %d nodes, %d edges to %s",
            nodes_df.height,
            edges_df.height,
            output_prefix,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a psychiatric BioMedKG subgraph"
    )
    parser.add_argument("--kg-path", type=Path, default=PipelineConfig.kg_path)
    parser.add_argument("--data-dir", type=Path, default=PipelineConfig.data_dir)
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
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    pipeline = KnowledgeGraphPipeline()
    config = PipelineConfig(
        kg_path=args.kg_path,
        data_dir=args.data_dir,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        allowed_relations=args.allowed_relations,
        psychiatric_patterns=args.psychiatric_patterns,
        include_reverse_edges=args.include_reverse,
        metadata_truncate=args.metadata_truncate,
    )
    pipeline.run(config)
