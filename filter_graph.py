"""Utility to prune GraphML knowledge graphs before model training."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"
NODE_TAG = f"{GRAPHML_NS}node"
EDGE_TAG = f"{GRAPHML_NS}edge"
DATA_TAG = f"{GRAPHML_NS}data"
GRAPH_TAG = f"{GRAPHML_NS}graph"
KEY_TAG = f"{GRAPHML_NS}key"


@dataclass
class NodeRecord:
    element: ET.Element
    attrs: Dict[str, str]


@dataclass
class EdgeRecord:
    element: ET.Element
    source: str
    target: str
    attrs: Dict[str, str]
    relation: str
    removed: bool = False


def _normalise(values: Optional[Sequence[str]]) -> Set[str]:
    if not values:
        return set()
    normalised: Set[str] = set()
    for value in values:
        if value is None:
            continue
        trimmed = value.strip()
        if trimmed:
            normalised.add(trimmed)
    return normalised


def _extract_key_registry(root: ET.Element) -> Dict[str, str]:
    key_registry: Dict[str, str] = {}
    for key_elem in root.findall(KEY_TAG):
        key_id = key_elem.attrib.get("id", "")
        key_name = key_elem.attrib.get("attr.name", key_id)
        key_registry[key_id] = key_name
    return key_registry


def _extract_nodes(
    graph_elem: ET.Element, key_registry: Dict[str, str]
) -> Dict[str, NodeRecord]:
    nodes: Dict[str, NodeRecord] = {}
    for node_elem in graph_elem.findall(NODE_TAG):
        node_id = node_elem.attrib.get("id")
        if not node_id:
            continue
        attrs: Dict[str, str] = {}
        for data_elem in node_elem.findall(DATA_TAG):
            key_id = data_elem.attrib.get("key", "")
            key_name = key_registry.get(key_id, key_id)
            attrs[key_name] = (data_elem.text or "").strip()
        nodes[node_id] = NodeRecord(element=node_elem, attrs=attrs)
    return nodes


def _extract_edges(
    graph_elem: ET.Element, key_registry: Dict[str, str]
) -> List[EdgeRecord]:
    edges: List[EdgeRecord] = []
    for edge_elem in graph_elem.findall(EDGE_TAG):
        source = edge_elem.attrib.get("source")
        target = edge_elem.attrib.get("target")
        if not source or not target:
            continue
        attrs: Dict[str, str] = {}
        relation = ""
        for data_elem in edge_elem.findall(DATA_TAG):
            key_id = data_elem.attrib.get("key", "")
            key_name = key_registry.get(key_id, key_id)
            value = (data_elem.text or "").strip()
            attrs[key_name] = value
            if not relation and key_name in {"relation", "predicate", "label"}:
                relation = value
        edges.append(
            EdgeRecord(
                element=edge_elem,
                source=source,
                target=target,
                attrs=attrs,
                relation=relation,
            )
        )
    return edges


def _deactivate_edges(edges: List[EdgeRecord], removed_nodes: Set[str]) -> None:
    if not removed_nodes:
        return
    for edge in edges:
        if edge.removed:
            continue
        if edge.source in removed_nodes or edge.target in removed_nodes:
            edge.removed = True


def _compute_degrees(
    nodes: Dict[str, NodeRecord], edges: List[EdgeRecord]
) -> Dict[str, int]:
    degree = {node_id: 0 for node_id in nodes.keys()}
    for edge in edges:
        if edge.removed:
            continue
        if edge.source not in nodes or edge.target not in nodes:
            continue
        degree[edge.source] += 1
        degree[edge.target] += 1
    return degree


def _prune_isolates(
    nodes: Dict[str, NodeRecord], edges: List[EdgeRecord], protected_types: Set[str]
) -> int:
    isolates_removed = 0
    while True:
        degree = _compute_degrees(nodes, edges)
        isolates = [
            node_id
            for node_id, deg in degree.items()
            if deg == 0
            and nodes[node_id].attrs.get("node_type", "").strip() not in protected_types
        ]
        if not isolates:
            break
        removed_set = set(isolates)
        for node_id in isolates:
            nodes.pop(node_id, None)
        _deactivate_edges(edges, removed_set)
        isolates_removed += len(isolates)
    return isolates_removed


def _prune_low_degree(
    nodes: Dict[str, NodeRecord],
    edges: List[EdgeRecord],
    threshold: int,
    allowed_types: Set[str],
    protected_types: Set[str],
) -> int:
    removed = 0
    if threshold <= 0:
        return 0
    while True:
        degree = _compute_degrees(nodes, edges)
        to_remove: List[str] = []
        for node_id, deg in degree.items():
            node_type = nodes[node_id].attrs.get("node_type", "").strip()
            if protected_types and node_type in protected_types:
                continue
            if allowed_types and node_type not in allowed_types:
                continue
            if deg < threshold:
                to_remove.append(node_id)
        if not to_remove:
            break
        removed_set = set(to_remove)
        for node_id in to_remove:
            nodes.pop(node_id, None)
        _deactivate_edges(edges, removed_set)
        removed += len(to_remove)
    return removed


def _build_adjacency(
    nodes: Dict[str, NodeRecord], edges: List[EdgeRecord]
) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in nodes.keys()}
    for edge in edges:
        if edge.removed:
            continue
        src = edge.source
        dst = edge.target
        if src not in nodes or dst not in nodes:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    return adjacency


def _keep_largest_component(
    nodes: Dict[str, NodeRecord], edges: List[EdgeRecord]
) -> int:
    if not nodes:
        return 0
    adjacency = _build_adjacency(nodes, edges)
    visited: Set[str] = set()
    largest_component: Set[str] = set()
    for node_id in nodes.keys():
        if node_id in visited:
            continue
        component: Set[str] = set()
        stack: deque[str] = deque([node_id])
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbour in adjacency.get(current, ()):  # type: ignore[arg-type]
                if neighbour not in visited:
                    stack.append(neighbour)
        if len(component) > len(largest_component):
            largest_component = component
    if not largest_component:
        return 0
    to_remove = [
        node_id for node_id in list(nodes.keys()) if node_id not in largest_component
    ]
    if not to_remove:
        return 0
    removed_set = set(to_remove)
    for node_id in to_remove:
        nodes.pop(node_id, None)
    _deactivate_edges(edges, removed_set)
    return len(to_remove)


def filter_graph(
    graphml_path: Path,
    output_path: Optional[Path] = None,
    drop_relations: Optional[Sequence[str]] = None,
    keep_relations: Optional[Sequence[str]] = None,
    degree_threshold: int = 0,
    degree_node_types: Optional[Sequence[str]] = None,
    keep_node_types: Optional[Sequence[str]] = None,
    keep_largest_component: bool = False,
    keep_original: bool = False,
) -> Dict[str, int | str]:
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    graph_elem = root.find(GRAPH_TAG)
    if graph_elem is None:
        raise ValueError("GraphML file does not contain a <graph> element.")

    key_registry = _extract_key_registry(root)
    nodes = _extract_nodes(graph_elem, key_registry)
    edges = _extract_edges(graph_elem, key_registry)

    original_nodes = len(nodes)
    original_edges = len(edges)

    drop_relations_set = _normalise(drop_relations)
    keep_relations_set = _normalise(keep_relations)
    if drop_relations_set and keep_relations_set:
        raise ValueError("Specify only drop_relations or keep_relations, not both.")

    protected_types = _normalise(keep_node_types)
    degree_types = _normalise(degree_node_types)

    if keep_relations_set:
        for edge in edges:
            if edge.relation not in keep_relations_set:
                edge.removed = True
    elif drop_relations_set:
        for edge in edges:
            if edge.relation in drop_relations_set:
                edge.removed = True

    isolates_removed = _prune_isolates(nodes, edges, protected_types)
    removed_low_degree = _prune_low_degree(
        nodes,
        edges,
        threshold=degree_threshold,
        allowed_types=degree_types,
        protected_types=protected_types,
    )
    isolates_removed += _prune_isolates(nodes, edges, protected_types)

    removed_components = 0
    if keep_largest_component and nodes:
        removed_components = _keep_largest_component(nodes, edges)
        isolates_removed += _prune_isolates(nodes, edges, protected_types)

    resolved_output: Path
    if output_path is None:
        resolved_output = graphml_path.with_suffix(".filtered.graphml")
        if resolved_output == graphml_path:
            resolved_output = graphml_path.with_name(
                graphml_path.stem + ".filtered.graphml"
            )
    else:
        resolved_output = output_path

    if resolved_output == graphml_path and keep_original:
        resolved_output = graphml_path.with_name(
            graphml_path.stem + ".filtered.graphml"
        )

    new_graph = ET.Element(graph_elem.tag, graph_elem.attrib)
    for node_id in nodes.keys():
        new_graph.append(copy.deepcopy(nodes[node_id].element))
    for edge in edges:
        if edge.removed:
            continue
        if edge.source not in nodes or edge.target not in nodes:
            continue
        new_graph.append(copy.deepcopy(edge.element))

    parent_children = list(root)
    index = parent_children.index(graph_elem)
    root.remove(graph_elem)
    root.insert(index, new_graph)

    tree.write(resolved_output, encoding="utf-8", xml_declaration=True)

    final_edges = sum(
        1
        for edge in edges
        if not edge.removed and edge.source in nodes and edge.target in nodes
    )

    return {
        "original_nodes": original_nodes,
        "original_edges": original_edges,
        "final_nodes": len(nodes),
        "final_edges": final_edges,
        "isolates_removed": isolates_removed,
        "low_degree_removed": removed_low_degree,
        "components_removed": removed_components,
        "output_path": str(resolved_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter edges and nodes from a GraphML knowledge graph to reduce size before training."
    )
    parser.add_argument("graphml", type=Path, help="Path to the input GraphML file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the filtered GraphML (default: append .filtered.graphml)",
    )
    parser.add_argument(
        "--drop-relations",
        nargs="*",
        help="Edge relation labels to drop entirely",
    )
    parser.add_argument(
        "--keep-relations",
        nargs="*",
        help="If provided, only edges with these relation labels are kept",
    )
    parser.add_argument(
        "--degree-threshold",
        type=int,
        default=0,
        help="Remove nodes below this degree (after relation filtering)",
    )
    parser.add_argument(
        "--degree-node-types",
        nargs="*",
        help="Restrict degree pruning to these node types",
    )
    parser.add_argument(
        "--keep-node-types",
        nargs="*",
        help="Node types that are never pruned, even if below the degree threshold",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Retain only the largest connected component after filtering",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Do not overwrite the original file when --output is omitted",
    )
    args = parser.parse_args()

    stats = filter_graph(
        graphml_path=args.graphml,
        output_path=args.output,
        drop_relations=args.drop_relations,
        keep_relations=args.keep_relations,
        degree_threshold=args.degree_threshold,
        degree_node_types=args.degree_node_types,
        keep_node_types=args.keep_node_types,
        keep_largest_component=args.keep_largest_component,
        keep_original=args.keep_original,
    )

    print(
        "Nodes: {original_nodes} -> {final_nodes}\n"
        "Edges: {original_edges} -> {final_edges}\n"
        "Isolates removed: {isolates_removed}\n"
        "Low-degree removed: {low_degree_removed}\n"
        "Components trimmed: {components_removed}\n"
        "Output: {output_path}".format(**stats)
    )


if __name__ == "__main__":
    main()
