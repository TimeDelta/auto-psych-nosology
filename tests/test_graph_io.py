import pickle

import networkx as nx
import pytest

from augment_graph_with_ontologies import _load_base_graph
from check_psych_ratio import _load_nodes_edges
from graph_io import read_pickled_graph
from visualize_graph import load_graph


@pytest.mark.parametrize("extension", [".gpickle", ".pickle", ".pkl"])
def test_pickle_readers_preserve_graph_attributes(tmp_path, extension):
    graph = nx.Graph()
    graph.add_node("n1", name="Phenotype", psy_score=0.8)
    graph.add_node("n2", name="Gene")
    graph.add_edge("n1", "n2", relation="associated_with", weight=2.0)
    graph_path = tmp_path / f"graph{extension}"
    with graph_path.open("wb") as graph_file:
        pickle.dump(graph, graph_file)

    for restored_graph in (
        read_pickled_graph(graph_path),
        _load_base_graph(graph_path),
        load_graph(str(graph_path)),
    ):
        assert restored_graph.nodes["n1"]["psy_score"] == 0.8
        assert restored_graph.edges["n1", "n2"]["weight"] == 2.0
    if extension != ".pickle":
        nodes, edges = _load_nodes_edges(graph_path)
        assert dict(nodes)["n1"]["name"] == "Phenotype"
        assert edges == [("n1", "n2")]


def test_pickle_reader_rejects_non_graph_payload(tmp_path):
    graph_path = tmp_path / "invalid.pkl"
    with graph_path.open("wb") as graph_file:
        pickle.dump({"nodes": []}, graph_file)
    with pytest.raises(ValueError, match="does not contain a NetworkX graph"):
        read_pickled_graph(graph_path)
