from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


def _normalise_keywords(raw: Sequence[str]) -> set[str]:
    return {kw.strip().lower() for kw in raw if kw and kw.strip()}


__all__ = [
    "should_drop_nosology_node",
    "NOSOLOGY_NODE_TYPES",
    "NOSOLOGY_NAME_KEYWORDS",
    "NOSOLOGY_SOURCE_KEYWORDS",
]

NOSOLOGY_NODE_TYPES = _normalise_keywords(
    [
        "nosology",
        "nosology_category",
        "diagnostic_category",
        "diagnostic_domain",
    ]
)
NOSOLOGY_NAME_KEYWORDS = _normalise_keywords(
    [
        "hitop",
        "hi-top",
        "hierarchical taxonomy of psychopathology",
        "rdoc",
        "research domain criteria",
        "internalizing spectrum",
        "externalizing spectrum",
        "detachment",
        "anankastia",
        "negative affectivity",
        "psychoticism",
        "somatoform spectrum",
        "thought disorder spectrum",
        "dsm",
        "icd",
        "major depressive disorder",
        "bipolar disorder",
        "bipolar i disorder",
        "bipolar ii disorder",
        "schizoaffective disorder",
        "schizophrenia",
        "generalized anxiety disorder",
        "panic disorder",
        "posttraumatic stress disorder",
        "post-traumatic stress disorder",
        "obsessive compulsive disorder",
        "ocd",
        "ocpd",
        "borderline personality disorder",
        "antisocial personality disorder",
        "avoidant personality disorder",
        "anorexia nervosa",
        "bulimia nervosa",
        "binge eating disorder",
        "attention-deficit hyperactivity disorder",
        "adhd",
        "autism spectrum disorder",
        "conduct disorder",
        "oppositional defiant disorder",
        "schizotypal personality disorder",
        "schizoid personality disorder",
        "cyclothymic disorder",
        "persistent depressive disorder",
        "dysthymia",
    ]
)
NOSOLOGY_SOURCE_KEYWORDS = _normalise_keywords(
    ["hitop", "rdoc", "icd", "dsm", "psychiatric_nosology", "nosology"]
)
NOSOLOGY_METADATA_FIELDS = (
    "disease_metadata",
    "drug_metadata",
    "protein_metadata",
    "dna_metadata",
    "metadata",
)
_NOSOLOGY_FLAG_FIELDS = (
    "nosology_flag",
    "taxonomy_flag",
    "alignment_flag",
    "is_nosology",
)
_DSM_CODE_PATTERNS = [
    re.compile(r"^f\d{2}(?:\.\d+)?$", re.IGNORECASE),
    re.compile(r"^\d{3}\.\d+$"),
]


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

    source = str(attrs.get("source", "")).strip().lower()
    if source and any(keyword in source for keyword in NOSOLOGY_SOURCE_KEYWORDS):
        return True

    for flag in _NOSOLOGY_FLAG_FIELDS:
        if _parse_bool(attrs.get(flag)):
            return True

    name = str(attrs.get("name", "")).strip().lower()
    if name and any(keyword in name for keyword in NOSOLOGY_NAME_KEYWORDS):
        return True
    if name and _matches_dsm_code(name):
        return True

    node_id = str(attrs.get("node_id", "")).strip().lower()
    if node_id and any(keyword in node_id for keyword in NOSOLOGY_NAME_KEYWORDS):
        return True
    if node_id and _matches_dsm_code(node_id):
        return True

    node_identifier = str(attrs.get("node_identifier", "")).strip().lower()
    if node_identifier and any(
        keyword in node_identifier for keyword in NOSOLOGY_NAME_KEYWORDS
    ):
        return True
    if node_identifier and _matches_dsm_code(node_identifier):
        return True

    synonyms = attrs.get("synonyms")
    if isinstance(synonyms, str):
        lowered_syn = synonyms.lower()
        if any(keyword in lowered_syn for keyword in NOSOLOGY_NAME_KEYWORDS):
            return True
        if _matches_dsm_code(lowered_syn):
            return True
    elif isinstance(synonyms, (list, tuple)):
        for synonym in synonyms:
            synonym_lower = str(synonym or "").lower()
            if synonym_lower and any(
                keyword in synonym_lower for keyword in NOSOLOGY_NAME_KEYWORDS
            ):
                return True
            if synonym_lower and _matches_dsm_code(synonym_lower):
                return True

    for field in NOSOLOGY_METADATA_FIELDS:
        meta_val = str(attrs.get(field, "")).strip().lower()
        if meta_val and any(keyword in meta_val for keyword in NOSOLOGY_NAME_KEYWORDS):
            return True
        if meta_val and _matches_dsm_code(meta_val):
            return True

    return False


def _matches_dsm_code(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    normalized = normalized.replace(" ", "")
    for pattern in _DSM_CODE_PATTERNS:
        if pattern.match(normalized):
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
