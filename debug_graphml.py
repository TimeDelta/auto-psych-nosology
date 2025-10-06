"""Utility script to inspect GraphML graph files for quick debugging."""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable

import networkx as nx


def _summarise_attrs(items) -> set[str]:
    keys: set[str] = set()
    for _, data in items:
        if isinstance(data, dict):
            keys.update(data.keys())
    return keys


def _print_sample(items, limit: int, label: str) -> None:
    print(f"Sample {label} (limit={limit}):")
    for idx, (identifier, data) in enumerate(islice(items, limit)):
        if isinstance(data, dict) and data:
            print(f"  - {identifier}: {data}")
        else:
            print(f"  - {identifier}")
    if limit == 0:
        print("  (skipped)")


def inspect_graphml(
    path: Path, node_limit: int, edge_limit: int, show_graph_attrs: bool
) -> None:
    print(f"=== {path} ===")
    if not path.exists():
        print("! File does not exist", file=sys.stderr)
        return
    try:
        graph = nx.read_graphml(path)
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"! Failed to read GraphML: {exc}", file=sys.stderr)
        return

    graph_type = type(graph).__name__
    print(f"Graph type: {graph_type}")
    print(f"Directed: {graph.is_directed()}")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")

    if show_graph_attrs and graph.graph:
        print("Graph attributes:")
        for key, value in graph.graph.items():
            print(f"  - {key}: {value}")

    node_attr_keys = _summarise_attrs(graph.nodes(data=True))
    edge_attr_keys = _summarise_attrs(graph.edges(data=True))
    print(f"Node attribute keys: {sorted(node_attr_keys)}")
    print(f"Edge attribute keys: {sorted(edge_attr_keys)}")

    _print_sample(graph.nodes(data=True), node_limit, "nodes")
    print()
    _print_sample(graph.edges(data=True), edge_limit, "edges")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GraphML graph structure.")
    parser.add_argument(
        "paths", type=Path, nargs="+", help="One or more GraphML files."
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        default=5,
        help="Number of sample nodes to display (default: 5, set to 0 to skip).",
    )
    parser.add_argument(
        "--edge-limit",
        type=int,
        default=5,
        help="Number of sample edges to display (default: 5, set to 0 to skip).",
    )
    parser.add_argument(
        "--graph-attrs",
        action="store_true",
        help="Display top-level graph attributes if present.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    for path in args.paths:
        inspect_graphml(
            path,
            node_limit=max(args.node_limit, 0),
            edge_limit=max(args.edge_limit, 0),
            show_graph_attrs=args.graph_attrs,
        )
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
