"""Serialization helpers for exporting tables and GraphML files."""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any, Iterable, Optional

import networkx as nx
import pandas as pd

PRIMITIVES = (int, float, bool)
_XML10_BAD = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\uFFFE\uFFFF]")


def _clean_str(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = _XML10_BAD.sub(" ", value)
    return value.encode("utf-8", "ignore").decode("utf-8", "ignore")


def _json_clean(obj: Any) -> str:
    return _clean_str(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def _coerce_for_graphml(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, PRIMITIVES):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _clean_str(value.decode("utf-8", "ignore"))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return _json_clean(list(value))
    if isinstance(value, dict):
        return _json_clean(value)
    if isinstance(value, type):
        return value.__name__
    return _clean_str(str(value))


def sanitize_graph_for_graphml(graph: nx.Graph) -> None:
    for key, value in list(graph.graph.items()):
        new_key = _clean_str(str(key))
        new_value = _coerce_for_graphml(value)
        del graph.graph[key]
        graph.graph[new_key] = new_value

    for _, data in graph.nodes(data=True):
        for key in list(data.keys()):
            data[_clean_str(str(key))] = _coerce_for_graphml(data[key])

    if graph.is_multigraph():
        iterator = graph.edges(keys=True, data=True)
    else:
        iterator = graph.edges(data=True)
    for edge in iterator:
        data = edge[-1]
        for key in list(data.keys()):
            data[_clean_str(str(key))] = _coerce_for_graphml(data[key])


def find_non_primitive_attrs(graph: nx.Graph):
    bad = []

    def has_bad(value: Any) -> bool:
        return isinstance(value, str) and _XML10_BAD.search(value) is not None

    for key, value in graph.graph.items():
        if has_bad(key) or has_bad(value):
            bad.append(("graph", key))
    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
    else:
        edges = graph.edges(data=True)
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if has_bad(key) or has_bad(value):
                bad.append(("node", node, key))
    for edge in edges:
        data = edge[-1]
        for key, value in data.items():
            if has_bad(key) or has_bad(value):
                bad.append(("edge", edge[:-1], key))
    return bad


def save_tables(
    nodes_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    graph: nx.MultiDiGraph,
    output_prefix: pathlib.Path,
    projected: Optional[nx.Graph] = None,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(output_prefix.with_suffix(".nodes.parquet"))
    relations_df.to_parquet(output_prefix.with_suffix(".rels.parquet"))
    papers_df.to_parquet(output_prefix.with_suffix(".papers.parquet"))
    sanitize_graph_for_graphml(graph)
    nx.write_graphml(graph, output_prefix.with_suffix(".graphml"))
    if projected is not None:
        sanitize_graph_for_graphml(projected)
        nx.write_graphml(projected, output_prefix.with_suffix(".weighted.graphml"))


__all__ = [
    "find_non_primitive_attrs",
    "sanitize_graph_for_graphml",
    "save_tables",
]
