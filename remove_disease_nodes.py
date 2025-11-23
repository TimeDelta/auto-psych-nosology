#!/usr/bin/env python3.10
"""Utility to strip disease-type nodes from a GraphML knowledge graph."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx

DEFAULT_DISEASE_TYPES = (
    "disease",
    "disorder",
    "condition",
    "diagnosis",
)
DEFAULT_CONTAINS_TERMS = ("disease",)


def _normalise(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalise_many(values: Sequence[str] | None) -> set[str]:
    if not values:
        return set()
    return {v for item in values if (v := _normalise(item))}


def _normalise_list(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    return [v for item in values if (v := _normalise(item))]


def _should_remove(
    attrs: dict,
    match_types: set[str],
    contains: Iterable[str],
) -> bool:
    node_type = _normalise(attrs.get("node_type"))
    if not node_type:
        return False
    if match_types and node_type in match_types:
        return True
    if contains and any(term in node_type for term in contains):
        return True
    return False


def _resolve_output_path(
    graphml_path: Path,
    output_path: Path | None,
    keep_original: bool,
) -> Path:
    if output_path is not None:
        return output_path
    if keep_original:
        return graphml_path.with_name(
            f"{graphml_path.stem}.nodisease{graphml_path.suffix}"
        )
    return graphml_path


def remove_disease_nodes(
    graphml_path: Path,
    *,
    output_path: Path | None = None,
    disease_types: Sequence[str] | None = None,
    contains_terms: Sequence[str] | None = None,
    keep_original: bool = False,
) -> tuple[int, int, int, Path]:
    """Remove nodes whose node_type aligns with disease concepts."""

    match_types = _normalise_many(disease_types) or _normalise_many(
        DEFAULT_DISEASE_TYPES
    )
    contains = _normalise_list(contains_terms) or list(DEFAULT_CONTAINS_TERMS)

    graph = nx.read_graphml(graphml_path)
    original_nodes = graph.number_of_nodes()

    to_remove = [
        node
        for node, attrs in graph.nodes(data=True)
        if _should_remove(attrs, match_types, contains)
    ]
    if to_remove:
        graph.remove_nodes_from(to_remove)

    destination = _resolve_output_path(graphml_path, output_path, keep_original)
    nx.write_graphml(graph, destination)

    remaining_nodes = graph.number_of_nodes()
    return original_nodes, len(to_remove), remaining_nodes, destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop disease-type nodes from a GraphML knowledge graph by matching the "
            "node_type attribute"
        )
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
        help="Optional path for the cleaned graph. Defaults to overwriting the input unless --keep-original is set.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Preserve the source file and write the cleaned graph alongside it (adds .nodisease suffix unless --output is provided).",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help=(
            "Exact node_type values to strip (case-insensitive). Defaults to disease, "
            "disorder, condition, diagnosis."
        ),
    )
    parser.add_argument(
        "--contains",
        nargs="+",
        default=None,
        metavar="SUBSTR",
        help="Substring matches applied to node_type (case-insensitive). Defaults to ['disease'].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original, removed, remaining, output = remove_disease_nodes(
        args.graphml,
        output_path=args.output,
        disease_types=args.types,
        contains_terms=args.contains,
        keep_original=args.keep_original,
    )
    print(
        f"Nodes before: {original}\n"
        f"Nodes removed: {removed}\n"
        f"Nodes after: {remaining}\n"
        f"Output written to: {output}"
    )


if __name__ == "__main__":
    main()
