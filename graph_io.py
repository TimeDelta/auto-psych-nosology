"""Graph readers shared by pipeline and inspection commands."""

import pickle
from pathlib import Path

import networkx as nx


def read_pickled_graph(graph_path: Path) -> nx.Graph:
    """Read a trusted local graph pickle without NetworkX's removed gpickle API."""
    with Path(graph_path).open("rb") as graph_file:
        graph = pickle.load(graph_file)
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"Pickle does not contain a NetworkX graph: {graph_path}")
    return graph
