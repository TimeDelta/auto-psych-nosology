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
from typing import Any, Dict, List

import networkx as nx

try:
    import graph_tool.all as gt
except ImportError as exc:
    raise SystemExit(
        "graph-tool is required for hierarchical SBM inference. "
        "Install it from https://graph-tool.skewed.de/ and re-run the script."
    ) from exc


LOGGER = logging.getLogger(__name__)


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


def _ensure_vertex_id_property(g: gt.Graph, nx_graph: nx.Graph) -> None:
    """Attach a vertex property storing the original NetworkX node identifiers."""
    prop = g.new_vertex_property("string")
    nodes = list(nx_graph.nodes)
    for idx, vertex in enumerate(g.vertices()):
        node_id = nodes[idx]
        prop[vertex] = str(node_id)
    g.vertex_properties["node_id"] = prop


def _networkx_to_graphtool(graph: nx.Graph) -> gt.Graph:
    """Convert a NetworkX graph into a graph-tool graph with weight support."""
    g = gt.Graph(directed=graph.is_directed())
    g.add_vertex(len(graph))
    _ensure_vertex_id_property(g, graph)

    weight_prop = g.new_edge_property("double")
    has_weight = False
    id_to_vertex = {node: g.vertex(i) for i, node in enumerate(graph.nodes)}
    for u, v, data in graph.edges(data=True):
        edge = g.add_edge(id_to_vertex[u], id_to_vertex[v])
        weight = float(data.get("weight", 1.0))
        if weight != 1.0:
            has_weight = True
        weight_prop[edge] = weight
    if has_weight:
        g.edge_properties["weight"] = weight_prop
    return g


def _extract_hierarchy(state: gt.NestedBlockState) -> Dict[str, Any]:
    """Convert block assignments into a JSON-serialisable hierarchy."""
    g = state.g
    node_ids = [
        g.vp["node_id"][v] if "node_id" in g.vp else str(int(v)) for v in g.vertices()
    ]
    hierarchy: List[Dict[str, Any]] = []
    labels = node_ids
    for level, assignments in enumerate(state.get_bs()):
        mapping = {labels[i]: int(assignments[i]) for i in range(len(assignments))}
        n_blocks = int(assignments.max()) + 1 if len(assignments) else 0
        hierarchy.append(
            {
                "level": level,
                "n_blocks": n_blocks,
                "assignments": mapping,
            }
        )
        labels = [f"L{level}_B{i}" for i in range(n_blocks)]
    return {
        "levels": hierarchy,
        "description_length": state.entropy(),
        "deg_corr": state.deg_corr,
    }


def run_hsbm(
    graph_path: Path,
    state_path: Path | None = None,
    deg_corr: bool = True,
    ignore_node_types: set[str] | None = None,
) -> Dict[str, Any]:
    """Run hierarchical SBM inference and optionally persist the full state."""
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
    g = _networkx_to_graphtool(nx_graph)
    state_args: Dict[str, Any] = {}
    if "weight" in g.edge_properties:
        state_args["eweight"] = g.ep.weight
    state = gt.minimize_nested_blockmodel_dl(
        g, deg_corr=deg_corr, state_args=state_args
    )
    if state_path:
        state.save(str(state_path))
    return _extract_hierarchy(state)


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
        help="Optional path to persist the full graph-tool state (extension .gt recommended)",
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
        "--keep-diagnoses",
        action="store_true",
        help="Include nodes labelled as Diagnosis when partitioning (ignored by default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    LOGGER.info("Running hierarchical SBM on %s", args.graph)
    ignore_types = set()
    if not args.keep_diagnoses:
        ignore_types.add("Diagnosis")
    result = run_hsbm(
        args.graph,
        args.state_path,
        deg_corr=not args.no_deg_corr,
        ignore_node_types=ignore_types or None,
    )
    args.output.write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "Saved hierarchy with %d levels to %s", len(result["levels"]), args.output
    )
