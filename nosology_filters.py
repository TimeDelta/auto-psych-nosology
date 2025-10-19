from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def _normalise_keywords(raw: Sequence[str]) -> set[str]:
    return {kw.strip().lower() for kw in raw if kw and kw.strip()}


__all__ = [
    "should_drop_nosology_node",
    "NOSOLOGY_NODE_TYPES",
    "NOSOLOGY_NAME_KEYWORDS",
]

NOSOLOGY_NODE_TYPES = _normalise_keywords(["disease", "disorder", "diagnosis"])
NOSOLOGY_NAME_KEYWORDS = _normalise_keywords(
    ["disorder", "disease", "syndrome", "diagnosis", "illness"]
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return False


def should_drop_nosology_node(attrs: Mapping[str, Any]) -> bool:
    node_type = str(attrs.get("node_type", "")).strip().lower()
    if node_type in NOSOLOGY_NODE_TYPES:
        return True
    for flag in ("ontology_flag", "group_flag", "is_psychiatric"):
        if _parse_bool(attrs.get(flag)):
            return True
    name = str(attrs.get("name", "")).lower()
    if name and any(keyword in name for keyword in NOSOLOGY_NAME_KEYWORDS):
        return True
    return False


def _filter_graph(graph_path: str, output_path: Optional[str]) -> None:
    import networkx as nx

    path = graph_path
    ext = path.split(".")[-1].lower()
    if ext == "graphml":
        graph = nx.read_graphml(path)
    elif ext in {"gexf", "gpickle", "pkl"}:
        graph = getattr(nx, f"read_{ext}")(path)
    else:
        raise ValueError(
            "Unsupported graph format. Use GraphML, GEXF, or networkx pickles."
        )

    to_remove = [
        node for node, data in graph.nodes(data=True) if should_drop_nosology_node(data)
    ]
    graph.remove_nodes_from(to_remove)
    print(
        f"Removed {len(to_remove)} nosology-aligned nodes. Remaining: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
    )
    if output_path:
        nx.write_graphml(graph, output_path)
        print(f"Saved filtered graph to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter nosology-aligned nodes from a graph.",
    )
    parser.add_argument("graph", help="Input graph file (GraphML/GEXF/gpickle)")
    parser.add_argument(
        "--output",
        help="Optional output path for filtered graph (defaults to overwriting input)",
    )
    args = parser.parse_args()
    output = args.output or args.graph
    _filter_graph(args.graph, output)
