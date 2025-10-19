#!/usr/bin/env python3.10
"""Remove stranded (degree-zero) nodes from a GraphML file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import networkx as nx


def remove_stranded_nodes(
    graphml_path: Path,
    output_path: Optional[Path] = None,
    keep_original: bool = False,
) -> tuple[int, int, int]:
    """Load a GraphML graph, drop isolated nodes, and save it.

    Returns a tuple of (original_nodes, removed_nodes, remaining_nodes).
    """

    graph = nx.read_graphml(graphml_path)
    original_nodes = graph.number_of_nodes()

    stranded = [node for node, degree in graph.degree() if degree == 0]
    if stranded:
        graph.remove_nodes_from(stranded)

    remaining_nodes = graph.number_of_nodes()

    if output_path is None:
        if keep_original:
            output_path = graphml_path.with_suffix(".stripped.graphml")
        else:
            output_path = graphml_path

    if not keep_original and output_path == graphml_path:
        # Overwrite original file.
        nx.write_graphml(graph, output_path)
    else:
        nx.write_graphml(graph, output_path)

    return original_nodes, len(stranded), remaining_nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove stranded (degree-zero) nodes from a GraphML file."
    )
    parser.add_argument(
        "graphml",
        type=Path,
        help="Path to the input GraphML file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the cleaned GraphML file. Defaults to overwriting the input unless --keep-original is used.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original file and write the cleaned graph alongside it (adds .stripped suffix unless --output is supplied).",
    )
    args = parser.parse_args()

    original, removed, remaining = remove_stranded_nodes(
        args.graphml,
        output_path=args.output,
        keep_original=args.keep_original,
    )

    print(
        f"Nodes before: {original}\n"
        f"Nodes removed: {removed}\n"
        f"Nodes after: {remaining}\n"
        f"Output written to: {args.output or (args.graphml if not args.keep_original else args.graphml.with_suffix('.stripped.graphml'))}"
    )


if __name__ == "__main__":
    main()
