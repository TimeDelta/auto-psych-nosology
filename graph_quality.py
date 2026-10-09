"""Descriptive graph integrity and evidence-coverage checks."""

from collections import Counter
from typing import Any, Dict

import networkx as nx

from nosology_filters import should_drop_nosology_node


def summarize_graph_quality(graph: nx.Graph) -> Dict[str, Any]:
    """Audit retained structure without treating coverage as evidence validity."""
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    components = list(nx.connected_components(graph.to_undirected(as_view=True)))
    evidence_fields = (
        "edge_source",
        "source_file",
        "source_record_index",
        "source_record_id",
        "source_record_json",
        "probability",
        "score",
        "direction",
        "correlation",
    )
    coverage_counts = Counter()
    relation_counts = Counter()
    synthetic_reverse_edges = 0
    for _, _, attributes in graph.edges(data=True):
        relation_counts[str(attributes.get("relation") or "unknown")] += 1
        synthetic_reverse_edges += bool(attributes.get("synthetic_reverse", False))
        for field_name in evidence_fields:
            if attributes.get(field_name) not in (None, ""):
                coverage_counts[field_name] += 1
    return {
        "n_nodes": node_count,
        "n_edges": edge_count,
        "connected_components": len(components),
        "largest_component_fraction": max(
            (len(component) for component in components), default=0
        )
        / max(1, node_count),
        "isolated_nodes": sum(1 for _ in nx.isolates(graph)),
        "self_loop_edges": nx.number_of_selfloops(graph),
        "synthetic_reverse_edges": synthetic_reverse_edges,
        "nosology_flagged_nodes": sum(
            should_drop_nosology_node(attributes)
            for _, attributes in graph.nodes(data=True)
        ),
        "node_type_counts": dict(
            sorted(
                Counter(
                    str(attributes.get("node_type") or "unknown")
                    for _, attributes in graph.nodes(data=True)
                ).items()
            )
        ),
        "relation_counts": dict(sorted(relation_counts.items())),
        "edge_attribute_coverage": {
            field_name: {
                "count": coverage_counts[field_name],
                "fraction": coverage_counts[field_name] / max(1, edge_count),
            }
            for field_name in evidence_fields
        },
        "interpretation": "Coverage and connectivity are descriptive checks, not evidence of biological or causal validity.",
    }
