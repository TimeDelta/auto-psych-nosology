#!/usr/bin/env python3
"""Summarize how much psychiatric signal remains inside a Graph slice."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, List, Tuple

import networkx as nx

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"


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


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_graphml_nodes_edges(
    graph_path: Path,
) -> Tuple[List[Tuple[str, dict[str, str]]], List[Tuple[str, str]]]:
    tree = ET.parse(graph_path)
    root = tree.getroot()

    key_registry: dict[str, Tuple[str, str]] = {}
    for key_elem in root.findall(f"{GRAPHML_NS}key"):
        key_id = key_elem.attrib.get("id")
        if not key_id:
            continue
        key_domain = key_elem.attrib.get("for", "")
        key_name = key_elem.attrib.get("attr.name", key_id)
        key_registry[key_id] = (key_domain, key_name)

    nodes: List[Tuple[str, dict[str, str]]] = []
    for node_elem in root.findall(f".//{GRAPHML_NS}node"):
        node_id = node_elem.attrib["id"]
        attributes: dict[str, str] = {}
        for data_elem in node_elem.findall(f"{GRAPHML_NS}data"):
            key_id = data_elem.attrib.get("key")
            if not key_id:
                continue
            _, key_name = key_registry.get(key_id, ("node", key_id))
            attributes[key_name] = data_elem.text or ""
        nodes.append((node_id, attributes))

    edges: List[Tuple[str, str]] = []
    for edge_elem in root.findall(f".//{GRAPHML_NS}edge"):
        src = edge_elem.attrib.get("source")
        dst = edge_elem.attrib.get("target")
        if src is None or dst is None:
            continue
        edges.append((src, dst))
    return nodes, edges


def _load_nodes_edges(
    graph_path: Path,
) -> Tuple[List[Tuple[str, dict[str, str]]], List[Tuple[str, str]]]:
    ext = graph_path.suffix.lower()
    if ext == ".graphml":
        return _parse_graphml_nodes_edges(graph_path)
    if ext == ".gexf":
        graph = nx.read_gexf(graph_path)
    elif ext in {".gpickle", ".pkl"}:
        graph = nx.read_gpickle(graph_path)
    else:
        raise ValueError(
            f"Unsupported graph format for {graph_path}. Use GraphML, GEXF, or gpickle."
        )
    nodes = [(node_id, dict(attrs)) for node_id, attrs in graph.nodes(data=True)]
    edges = list(graph.edges())
    return nodes, edges


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0 / 0 (0.00%)"
    percent = 100.0 * numerator / max(1, denominator)
    return f"{numerator} / {denominator} ({percent:.2f}%)"


def summarize_graph(graph_path: Path, score_threshold: float, show_types: bool) -> None:
    nodes, edges = _load_nodes_edges(graph_path)
    total_nodes = len(nodes)
    total_edges = len(edges)

    by_type_total: Counter[str] = Counter()
    by_type_flag: Counter[str] = Counter()
    by_type_score: Counter[str] = Counter()

    psych_flag_nodes = set()
    high_score_nodes = set()

    score_sampled = 0
    score_sum = 0.0

    for node_id, attrs in nodes:
        node_type = str(attrs.get("node_type", "unknown")).lower()
        by_type_total[node_type] += 1

        if _parse_bool(attrs.get("is_psychiatric")):
            psych_flag_nodes.add(node_id)
            by_type_flag[node_type] += 1

        score = _parse_float(attrs.get("psy_score"))
        if score is not None:
            score_sampled += 1
            score_sum += score
            if score >= score_threshold:
                high_score_nodes.add(node_id)
                by_type_score[node_type] += 1

    edges_touching_flag = 0
    edges_touching_score = 0
    if total_edges:
        for src, dst in edges:
            if src in psych_flag_nodes or dst in psych_flag_nodes:
                edges_touching_flag += 1
            if src in high_score_nodes or dst in high_score_nodes:
                edges_touching_score += 1

    print(f"Graph: {graph_path}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(
        "Nodes with is_psychiatric=True:",
        _format_ratio(len(psych_flag_nodes), total_nodes),
    )
    print(
        f"Nodes with psy_score >= {score_threshold:g}:",
        _format_ratio(len(high_score_nodes), total_nodes),
    )
    if score_sampled:
        print(
            f"Mean psy_score over {score_sampled} nodes: {score_sum / score_sampled:.4f}"
        )
    else:
        print("Mean psy_score: N/A (no scores present)")
    print(
        "Edges touching is_psychiatric=True nodes:",
        _format_ratio(edges_touching_flag, total_edges),
    )
    print(
        f"Edges touching psy_score >= {score_threshold:g} nodes:",
        _format_ratio(edges_touching_score, total_edges),
    )

    if show_types and total_nodes:
        print("\nPer-node-type breakdown (counts / flag hits / score hits):")
        for node_type, count in sorted(by_type_total.items(), key=lambda kv: kv[0]):
            flag_hits = by_type_flag.get(node_type, 0)
            score_hits = by_type_score.get(node_type, 0)
            print(
                f"  - {node_type or 'unknown'}: {count} total, "
                f"{flag_hits} is_psychiatric, {score_hits} scoring hits"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how many nodes/edges retain psychiatric annotations after filtering."
        )
    )
    parser.add_argument("graph", type=Path, help="GraphML/GEXF/gpickle file to inspect")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="psy_score threshold used to count high-confidence psychiatric nodes",
    )
    parser.add_argument(
        "--per-type",
        action="store_true",
        help="Print per-node-type counts for additional context",
    )
    args = parser.parse_args()
    summarize_graph(args.graph, args.score_threshold, args.per_type)


if __name__ == "__main__":
    main()
