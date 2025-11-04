"""
See T. M. Sweet, A. C. Thomas, and B. W. Junker, “Hierarchical mixed membership
stochastic blockmodels for multiple networks and experimental interventions,”
in Handbook of Mixed Membership Models and Their Applications, E. Airoldi,
D. Blei, E. Erosheva, and S. Fienberg, Eds. Boca Raton, FL, USA: Chapman &
Hall/CRC Press, 2014, pp. 463–488.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


class _ClusterNode:
    """Internal representation of a cluster in the hierarchy."""

    __slots__ = ("members", "children", "modularity")

    def __init__(self, members: Sequence[str]) -> None:
        self.members: List[str] = sorted(str(node) for node in members)
        self.children: List["_ClusterNode"] = []
        self.modularity: float | None = None


def _split_cluster(
    graph: nx.Graph,
    node: _ClusterNode,
    *,
    depth: int,
    max_depth: int,
    weight_key: str | None,
    min_cluster_size: int,
    min_modularity_gain: float,
    resolution: float,
    seed: int,
) -> None:
    """Recursively split a cluster using modularity optimisation."""

    if depth >= max_depth:
        return

    if len(node.members) < max(2, min_cluster_size):
        return

    subgraph = graph.subgraph(node.members).copy()
    if subgraph.number_of_edges() == 0:
        return

    communities = _detect_communities(
        subgraph,
        weight=weight_key,
        resolution=resolution,
        seed=seed + depth,
    )

    if len(communities) <= 1:
        return

    community_sets = [set(group) for group in communities]
    sub_modularity = modularity(
        subgraph,
        community_sets,
        weight=weight_key,
    )

    if sub_modularity < min_modularity_gain:
        return

    node.modularity = sub_modularity

    for group in communities:
        child = _ClusterNode(group)
        node.children.append(child)
        if len(child.members) >= min_cluster_size:
            _split_cluster(
                graph,
                child,
                depth=depth + 1,
                max_depth=max_depth,
                weight_key=weight_key,
                min_cluster_size=min_cluster_size,
                min_modularity_gain=min_modularity_gain,
                resolution=resolution,
                seed=seed,
            )


def _collect_levels(
    graph: nx.Graph,
    root: _ClusterNode,
    *,
    weight_key: str | None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Convert the cluster tree into per-level assignments and metrics."""

    levels: List[Dict[str, Any]] = []
    total_negative_modularity = 0.0

    current_level: List[_ClusterNode] = [root]
    level_index = 0

    while current_level:
        next_level: List[_ClusterNode] = []
        clusters: List[_ClusterNode] = []

        for cluster in current_level:
            if cluster.children:
                clusters.extend(cluster.children)
                next_level.extend(cluster.children)

        if not clusters:
            break

        assignments: Dict[str, int] = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster.members:
                assignments[node] = idx

        union_nodes = set(assignments.keys())
        level_graph = graph.subgraph(union_nodes).copy()
        level_partition = [set(cluster.members) for cluster in clusters]
        level_modularity = modularity(level_graph, level_partition, weight=weight_key)
        total_negative_modularity += -level_modularity

        levels.append(
            {
                "level": level_index,
                "n_blocks": len(clusters),
                "assignments": assignments,
                "modularity": level_modularity,
            }
        )

        current_level = next_level
        level_index += 1

    if not levels:
        # No splits occurred; treat entire graph as single block at level 0.
        all_nodes = sorted(str(node) for node in root.members)
        levels.append(
            {
                "level": 0,
                "n_blocks": 1,
                "assignments": {node: 0 for node in all_nodes},
                "modularity": 0.0,
            }
        )

    return levels, total_negative_modularity


def _hierarchical_partition(
    graph: nx.Graph,
    *,
    weight_attr: str = "weight",
    max_levels: int = 5,
    min_cluster_size: int = 5,
    min_modularity_gain: float = 5e-3,
    resolution: float = 1.0,
    seed: int = 13,
) -> Dict[str, Any]:
    """Produce a hierarchical clustering via recursive modularity optimisation."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot compute hierarchy on an empty graph.")

    weight_key = (
        weight_attr
        if any(weight_attr in data for _, _, data in graph.edges(data=True))
        else None
    )

    root = _ClusterNode(list(graph.nodes()))
    _split_cluster(
        graph,
        root,
        depth=0,
        max_depth=max_levels,
        weight_key=weight_key,
        min_cluster_size=min_cluster_size,
        min_modularity_gain=min_modularity_gain,
        resolution=resolution,
        seed=seed,
    )

    levels, total_negative_modularity = _collect_levels(
        graph,
        root,
        weight_key=weight_key,
    )

    return {
        "levels": levels,
        "description_length": total_negative_modularity,
        "deg_corr": True,
        "algorithm": "louvain"
        if louvain_communities is not None
        else "greedy_modularity",
        "parameters": {
            "max_levels": max_levels,
            "min_cluster_size": min_cluster_size,
            "min_modularity_gain": min_modularity_gain,
            "resolution": resolution,
            "seed": seed,
        },
    }


def run_hsbm(
    graph_path: Path,
    state_path: Path | None = None,
    deg_corr: bool = True,
    ignore_node_types: set[str] | None = None,
    *,
    max_levels: int = 5,
    min_cluster_size: int = 5,
    min_modularity_gain: float = 5e-3,
    resolution: float = 1.0,
    seed: int = 13,
) -> Dict[str, Any]:
    """Run hierarchical partitioning without relying on graph-tool."""

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

    hierarchy = _hierarchical_partition(
        nx_graph,
        weight_attr="weight",
        max_levels=max_levels,
        min_cluster_size=min_cluster_size,
        min_modularity_gain=min_modularity_gain,
        resolution=resolution,
        seed=seed,
    )

    if state_path is not None:
        LOGGER.info(
            "Persisting hierarchy to %s (JSON payload, graph-tool state not available).",
            state_path,
        )
        state_path.write_text(json.dumps(hierarchy, indent=2))

    return hierarchy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical SBM partition generator")
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
        "--max-levels",
        type=int,
        default=5,
        help="Maximum number of hierarchical levels to explore (default: 5)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum community size eligible for further splitting (default: 5)",
    )
    parser.add_argument(
        "--min-modularity-gain",
        type=float,
        default=5e-3,
        help=(
            "Minimum modularity improvement required to accept a split "
            "within a community (default: 5e-3)"
        ),
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
        "Running modularity-based hierarchical clustering on %s (max_levels=%d, min_cluster_size=%d)",
        args.graph,
        args.max_levels,
        args.min_cluster_size,
    )
    ignore_types = set()
    if not args.keep_diagnoses:
        ignore_types.add("Diagnosis")
    result = run_hsbm(
        args.graph,
        args.state_path,
        deg_corr=not args.no_deg_corr,
        ignore_node_types=ignore_types or None,
        max_levels=args.max_levels,
        min_cluster_size=args.min_cluster_size,
        min_modularity_gain=args.min_modularity_gain,
        resolution=args.resolution,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "Saved hierarchy with %d levels to %s", len(result["levels"]), args.output
    )
