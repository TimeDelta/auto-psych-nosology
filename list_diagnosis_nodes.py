#!/usr/bin/env python3.10
"""Print diagnosis / nosology nodes contained in a GraphML file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx

from nosology_filters import should_drop_nosology_node

DEFAULT_GRAPH_PATH = Path("data/ikgraph.filtered.graphml")
DIAGNOSIS_KEYWORDS = ("diagnosis", "diagnoses", "diagnostic", "disease")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a GraphML knowledge graph and print nodes that look like diagnostic"
            " / nosology concepts according to the existing filters."
        )
    )
    parser.add_argument(
        "graph_path",
        type=Path,
        help=f"Path to the GraphML file (e.g. {DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one compact JSON object per line for easier downstream processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many diagnosis nodes to print.",
    )
    parser.add_argument(
        "--sort",
        choices=("name", "id"),
        default="name",
        help="Sort diagnosis nodes by their name or identifier (default: name).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _contains_diagnosis_keyword(value: object) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_diagnosis_keyword(item) for item in value)
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in DIAGNOSIS_KEYWORDS)


def _is_diagnosis_node(attrs: dict) -> bool:
    if should_drop_nosology_node(attrs):
        return True

    for candidate in (
        attrs.get("name"),
        attrs.get("node_type"),
        attrs.get("source"),
        attrs.get("node_identifier"),
        attrs.get("metadata"),
        attrs.get("disease_metadata"),
        attrs.get("drug_metadata"),
        attrs.get("protein_metadata"),
        attrs.get("dna_metadata"),
        attrs.get("synonyms"),
    ):
        if _contains_diagnosis_keyword(candidate):
            return True
    return False


def _iter_diagnosis_nodes(graph) -> Iterator[tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if _is_diagnosis_node(attrs):
            yield node_id, attrs


def _sort_nodes(nodes: list[tuple[str, dict]], key: str) -> None:
    if key == "name":
        nodes.sort(key=lambda item: str(item[1].get("name", "")).lower())
    else:
        nodes.sort(key=lambda item: str(item[0]))


def _emit_plain(node_id: str, attrs: dict) -> str:
    name = attrs.get("name", "<unnamed>")
    node_type = attrs.get("node_type", "?")
    source = attrs.get("source", "?")
    identifier = attrs.get("node_identifier", "")
    metadata = attrs.get("metadata")
    if isinstance(metadata, str):
        meta_preview = metadata.strip()
    else:
        meta_preview = json.dumps(metadata) if metadata is not None else ""
    preview = meta_preview[:120] + (
        "…" if meta_preview and len(meta_preview) > 120 else ""
    )
    return (
        f"{node_id}\t{name}\t(node_type={node_type}, source={source}, node_identifier={identifier})"
        + (f"\tmetadata={preview}" if preview else "")
    )


def _emit_json(node_id: str, attrs: dict) -> str:
    payload = {
        "id": node_id,
        "name": attrs.get("name"),
        "node_type": attrs.get("node_type"),
        "source": attrs.get("source"),
        "node_identifier": attrs.get("node_identifier"),
        "metadata": attrs.get("metadata"),
    }
    return json.dumps(payload, ensure_ascii=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    graph_path: Path = args.graph_path
    if not graph_path.exists():
        print(f"Graph file not found: {graph_path}", file=sys.stderr)
        return 1

    try:
        graph = nx.read_graphml(graph_path)
    except Exception as exc:  # pragma: no cover - script entrypoint
        print(f"Failed to read GraphML file: {exc}", file=sys.stderr)
        return 1

    candidates = list(_iter_diagnosis_nodes(graph))
    _sort_nodes(candidates, args.sort)
    total = len(candidates)
    to_show = candidates if args.limit is None else candidates[: max(args.limit, 0)]

    formatter = _emit_json if args.json else _emit_plain
    for node_id, attrs in to_show:
        print(formatter(node_id, attrs))

    summary = f"Found {total} diagnosis-aligned nodes"
    if args.limit is not None and args.limit < total:
        summary += f" (showing first {len(to_show)})"
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
