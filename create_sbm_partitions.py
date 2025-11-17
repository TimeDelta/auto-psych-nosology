"""Vanilla SBM partitions via Louvain/greedy modularity heuristics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

LOGGER = logging.getLogger(__name__)

try:
    from networkx.algorithms.community import louvain_communities
except ImportError:  # pragma: no cover - fallback for older NetworkX
    louvain_communities = None

from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

from nosology_filters import should_drop_nosology_node


def _load_graph(graph_path: Path) -> nx.Graph:
    """Load a graph from GraphML and coerce it to a simple undirected graph."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graph = nx.read_graphml(graph_path)
    if graph.is_multigraph():
        LOGGER.info("Collapsing multigraph edges by summing weights.")
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
        LOGGER.info("Symmetrising directed graph via edge union.")
        graph = graph.to_undirected(reciprocal=False)
    to_remove = [
        node for node, data in graph.nodes(data=True) if should_drop_nosology_node(data)
    ]
    if to_remove:
        LOGGER.info("Filtered %d nosology-aligned nodes", len(to_remove))
        graph.remove_nodes_from(to_remove)
    if graph.number_of_nodes() == 0:
        raise ValueError("No nodes remain after filtering; cannot run hSBM.")
    LOGGER.info(
        "Graph after nosology filter: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def _order_communities(communities: Iterable[Iterable[str]]) -> List[List[str]]:
    """Return communities sorted deterministically by size then lexicographically."""

    ordered: List[Tuple[int, List[str]]] = []
    for group in communities:
        members = sorted(str(node) for node in group)
        if not members:
            continue
        ordered.append((-len(members), members))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [members for _, members in ordered]


def _detect_communities(
    graph: nx.Graph,
    weight: str | None,
    resolution: float,
    seed: int,
) -> List[List[str]]:
    """Detect communities using Louvain when available, else greedy modularity."""

    if graph.number_of_nodes() == 0:
        return []

    if louvain_communities is not None:
        communities = louvain_communities(
            graph,
            weight=weight,
            resolution=resolution,
            seed=seed,
        )
    else:  # pragma: no cover - exercised only when Louvain is unavailable
        communities = greedy_modularity_communities(graph, weight=weight)

    return _order_communities(communities)


def _build_node_name_map(graph: nx.Graph) -> Dict[str, str]:
    """Map node identifiers to human-friendly names when available."""

    preferred_attrs = ("name", "label", "title")
    node_name_map: Dict[str, str] = {}

    for node, data in graph.nodes(data=True):
        node_id = str(node)
        label = None
        for attr in preferred_attrs:
            value = data.get(attr)
            if value is not None:
                text = str(value).strip()
                if text:
                    label = text
                    break
        node_name_map[node_id] = label if label is not None else node_id

    return node_name_map


def _flat_partition(
    graph: nx.Graph,
    *,
    weight_attr: str = "weight",
    resolution: float = 1.0,
    seed: int = 13,
) -> Dict[str, Any]:
    """Produce a single-level SBM-style partition via modularity optimisation."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot compute partition on an empty graph.")

    weight_key = (
        weight_attr
        if any(weight_attr in data for _, _, data in graph.edges(data=True))
        else None
    )

    communities = _detect_communities(
        graph,
        weight=weight_key,
        resolution=resolution,
        seed=seed,
    )
    if not communities:
        communities = [sorted(str(node) for node in graph.nodes())]

    node_to_cluster: Dict[str, int] = {}
    cluster_member_ids: Dict[str, List[str]] = {}
    for cluster_idx, members in enumerate(communities):
        cluster_key = str(cluster_idx)
        clean_members = sorted(str(node) for node in members)
        cluster_member_ids[cluster_key] = clean_members
        for node in clean_members:
            node_to_cluster[node] = cluster_idx

    level_partition = [set(members) for members in cluster_member_ids.values()]
    level_modularity = modularity(graph, level_partition, weight=weight_key)

    levels = [
        {
            "level": 0,
            "n_blocks": len(cluster_member_ids),
            "assignments": dict(node_to_cluster),
            "modularity": level_modularity,
        }
    ]

    return {
        "levels": levels,
        "description_length": -level_modularity,
        "deg_corr": True,
        "algorithm": "louvain"
        if louvain_communities is not None
        else "greedy_modularity",
        "node_to_cluster": node_to_cluster,
        "cluster_member_ids": cluster_member_ids,
        "parameters": {
            "resolution": resolution,
            "seed": seed,
            "weight_attr": weight_attr,
        },
    }


def run_sbm(
    graph_path: Path,
    state_path: Path | None = None,
    deg_corr: bool = True,
    ignore_node_types: set[str] | None = None,
    *,
    resolution: float = 1.0,
    seed: int = 13,
) -> Dict[str, Any]:
    """Run a vanilla SBM-style partition without relying on graph-tool."""

    if not deg_corr:
        LOGGER.warning(
            "Degree-correction toggle is not supported in the modularity-based "
            "implementation; proceeding with degree-aware Louvain clustering."
        )

    nx_graph = _load_graph(graph_path)
    if ignore_node_types:
        to_remove = [
            node
            for node, data in nx_graph.nodes(data=True)
            if data.get("node_type") in ignore_node_types
        ]
        if to_remove:
            LOGGER.info(
                "Ignoring %d nodes with types: %s",
                len(to_remove),
                ", ".join(sorted(ignore_node_types)),
            )
            nx_graph.remove_nodes_from(to_remove)
        if not nx_graph:
            raise ValueError(
                "All nodes were filtered out prior to partitioning. "
                "Relax the ignore_node_types setting."
            )

    hierarchy = _flat_partition(
        nx_graph,
        weight_attr="weight",
        resolution=resolution,
        seed=seed,
    )

    node_name_map = _build_node_name_map(nx_graph)
    hierarchy["node_name_map"] = node_name_map
    cluster_member_ids: Dict[str, List[str]] = hierarchy.get("cluster_member_ids", {})
    hierarchy["cluster_members"] = {
        cluster_id: [node_name_map.get(node_id, node_id) for node_id in members]
        for cluster_id, members in cluster_member_ids.items()
    }

    if state_path is not None:
        LOGGER.info(
            "Persisting hierarchy to %s (JSON payload, graph-tool state not available).",
            state_path,
        )
        state_path.write_text(json.dumps(hierarchy, indent=2))

    return hierarchy


def run_hsbm(*args, **kwargs):  # pragma: no cover - backwards compatibility shim
    LOGGER.warning(
        "run_hsbm() is deprecated; falling back to vanilla SBM (flat) partitioning."
    )
    return run_sbm(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vanilla SBM partition generator")
    parser.add_argument(
        "graph", type=Path, help="Input GraphML file produced by create_graph.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("partitions.json"),
        help="Destination for JSON hierarchy (default: partitions.json)",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Optional path to persist the hierarchy JSON (graph-tool state unavailable)",
    )
    parser.add_argument(
        "--no-deg-corr",
        action="store_true",
        help="Disable degree correction in the stochastic block model",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter forwarded to Louvain (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed controlling Louvain initialisation (default: 13)",
    )
    parser.add_argument(
        "--keep-diagnoses",
        action="store_true",
        help="Include nodes labelled as Diagnosis when partitioning (ignored by default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    LOGGER.info(
        "Running vanilla SBM (flat) clustering on %s",
        args.graph,
    )
    ignore_types = set()
    if not args.keep_diagnoses:
        ignore_types.add("Diagnosis")
    result = run_sbm(
        args.graph,
        args.state_path,
        deg_corr=not args.no_deg_corr,
        ignore_node_types=ignore_types or None,
        resolution=args.resolution,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "Saved hierarchy with %d levels to %s", len(result["levels"]), args.output
    )
