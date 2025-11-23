#!/usr/bin/env python3.10
"""Report degeneracy statistics for a GraphML knowledge graph."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx


def _to_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Return an undirected simple graph used for degeneracy checks."""
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        simple = nx.Graph()
        simple.add_nodes_from(graph.nodes(data=True))
        for u, v in graph.edges():
            if u == v:
                continue
            simple.add_edge(u, v)
        return simple
    if graph.is_directed():
        simple = nx.Graph()
        simple.add_nodes_from(graph.nodes(data=True))
        for u, v in graph.edges():
            if u == v:
                continue
            simple.add_edge(u, v)
        return simple
    return graph.copy()


def _format_distribution(core_numbers: dict[str, int]) -> str:
    counts = Counter(core_numbers.values())
    rows = ["core\tcount"]
    for core, freq in sorted(counts.items()):
        rows.append(f"{core}\t{freq}")
    return "\n".join(rows)


def _top_nodes(core_numbers: dict[str, int], limit: int) -> list[tuple[str, int]]:
    return sorted(core_numbers.items(), key=lambda item: item[1], reverse=True)[
        : max(limit, 0)
    ]


def compute_degeneracy(graphml_path: Path) -> tuple[int, dict[str, int]]:
    graph = nx.read_graphml(graphml_path)
    simple = _to_simple_graph(graph)
    if simple.number_of_nodes() == 0 or simple.number_of_edges() == 0:
        return 0, {}
    core_numbers = nx.core_number(simple)
    degeneracy = max(core_numbers.values(), default=0)
    return degeneracy, core_numbers


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the degeneracy (maximum k-core) of a GraphML graph and print core"
            " number summaries."
        )
    )
    parser.add_argument("graphml", type=Path, help="Path to the GraphML file")
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Show the top-N nodes by core number (default: 15, set to 0 to skip).",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Print a tab-separated histogram of core numbers for quick inspection.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    degeneracy, core_numbers = compute_degeneracy(args.graphml)
    print(f"Degeneracy (max core number): {degeneracy}")
    if not core_numbers:
        print("Graph has no edges or nodes; no core decomposition to report.")
        return 0

    if args.top > 0:
        print("\nTop nodes by core number:")
        for node, score in _top_nodes(core_numbers, args.top):
            print(f"  {node}: k={score}")

    if args.distribution:
        print("\nCore number distribution:")
        print(_format_distribution(core_numbers))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
