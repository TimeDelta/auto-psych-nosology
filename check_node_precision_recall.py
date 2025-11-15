#!/usr/bin/env python3
"""Measure how well a chosen node subset of the graph overlaps HiTOP/RDoC labels."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pandas as pd

from align_partitions import (
    align_partition_to_graph,
    infer_framework_labels_tailored,
    load_graph,
    load_mapping_csv,
    load_partition,
)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "none", "null"}:
            return None
        if stripped in {"true", "t", "1", "yes", "y"}:
            return True
        if stripped in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _series_from_csv(csv_path: Path, id_col: str, label_col: str) -> pd.Series:
    df = load_mapping_csv(csv_path, id_col=id_col, label_col=label_col)
    return df.set_index(id_col)[label_col].astype("string")


def _precision_recall(node_ids: set[str], labels: pd.Series) -> Dict[str, float]:
    labeled = set(labels.index)
    tp = len(node_ids & labeled)
    fp = len(node_ids - labeled)
    fn = len(labeled - node_ids)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "nodes_considered": len(node_ids),
        "labeled_nodes": len(labeled),
        "precision": precision,
        "recall": recall,
    }


def _per_label_metrics(node_ids: set[str], labels: pd.Series) -> list[Dict[str, float]]:
    subset_size = len(node_ids)
    if subset_size == 0:
        return []
    node_ids = set(node_ids)
    rows = []
    grouped = labels.groupby(labels)
    for label, idx in grouped.groups.items():
        members = set(idx)
        support = len(members)
        overlap = len(node_ids & members)
        recall = overlap / support if support else float("nan")
        precision = overlap / subset_size if subset_size else float("nan")
        rows.append(
            {
                "label": label,
                "support": support,
                "overlap": overlap,
                "precision": precision,
                "recall": recall,
            }
        )
    rows.sort(key=lambda r: r["label"])
    return rows


def _graph_nodes(G: nx.Graph) -> set[str]:
    return {str(n) for n in G.nodes}


def _psychiatric_nodes(G: nx.Graph) -> set[str]:
    ids: set[str] = set()
    for n, data in G.nodes(data=True):
        flag = _coerce_bool(data.get("is_psychiatric"))
        if flag:
            ids.add(str(n))
    return ids


def _neighbors_within_subset(
    G: nx.Graph, seeds: set[str], allowed: set[str], hops: int
) -> set[str]:
    if hops <= 0 or not seeds:
        return set(seeds)
    visited = set(seeds)
    frontier = deque(seeds)
    depth = {node: 0 for node in seeds}

    def _iter_neighbors(node: str):
        try:
            for nbr in G.neighbors(node):
                yield str(nbr)
        except Exception:
            return
        if G.is_directed():
            for nbr in G.predecessors(node):
                yield str(nbr)

    while frontier:
        node = frontier.popleft()
        current_depth = depth[node]
        if current_depth >= hops:
            continue
        for nbr in _iter_neighbors(node):
            if nbr not in allowed or nbr in visited:
                continue
            visited.add(nbr)
            depth[nbr] = current_depth + 1
            frontier.append(nbr)
    return visited


def _filter_nodes_by_psy_metrics(
    G: nx.Graph,
    node_ids: set[str],
    *,
    min_psy_score: float,
    neighbor_hops: int,
) -> set[str]:
    if min_psy_score <= 0 and neighbor_hops <= 0:
        return set(node_ids)
    allowed = set(node_ids)
    keep: set[str] = set()
    if min_psy_score > 0:
        for n in allowed:
            score = _coerce_float(G.nodes[n].get("psy_score"))
            if score is not None and score >= min_psy_score:
                keep.add(n)
    else:
        keep = set(allowed)
    if not keep:
        return keep
    if neighbor_hops > 0:
        keep = _neighbors_within_subset(G, keep, allowed, neighbor_hops)
    return keep


def _partition_nodes(G: nx.Graph, partition_path: Path | None) -> set[str]:
    if not partition_path:
        raise ValueError("--partition is required when --subset=partition")
    partition = load_partition(partition_path)
    aligned, missing = align_partition_to_graph(G, partition)
    if missing:
        print(f"Warning: {len(missing)} partition entries were not found in the graph.")
    return set(aligned.keys())


def _resolve_node_ids(
    G: nx.Graph, subset: str, partition_path: Path | None
) -> Tuple[str, set[str]]:
    if subset == "graph":
        return "graph", _graph_nodes(G)
    if subset == "psychiatric":
        return "psychiatric", _psychiatric_nodes(G)
    if subset == "partition":
        return "partition", _partition_nodes(G, partition_path)
    raise ValueError(f"Unknown subset '{subset}'")


def run(
    graph_path: Path,
    subset: str,
    partition_path: Path | None,
    prop_depth: int,
    hpo_terms: Path | None,
    hitop_csv: Path | None,
    rdoc_csv: Path | None,
    id_col: str,
    hitop_col: str,
    rdoc_col: str,
    min_psy_score: float,
    psy_include_neighbors: int,
    save_json: Path | None,
    per_label_csv: Path | None,
) -> None:
    G = load_graph(graph_path)
    subset_name, node_ids = _resolve_node_ids(G, subset, partition_path)
    initial_count = len(node_ids)
    node_ids = _filter_nodes_by_psy_metrics(
        G,
        node_ids,
        min_psy_score=min_psy_score,
        neighbor_hops=psy_include_neighbors,
    )
    if not node_ids:
        raise ValueError(
            "No nodes remain after applying psychiatric filters; relax the thresholds."
        )

    auto_hitop: pd.Series | None = None
    auto_rdoc: pd.Series | None = None
    if not hitop_csv or not rdoc_csv:
        auto_hitop, auto_rdoc = infer_framework_labels_tailored(
            G, prop_depth=prop_depth, hpo_terms_path=hpo_terms
        )

    frameworks: list[Tuple[str, pd.Series]] = []
    if hitop_csv:
        frameworks.append(("HiTOP", _series_from_csv(hitop_csv, id_col, hitop_col)))
    elif auto_hitop is not None:
        frameworks.append(("HiTOP", auto_hitop))

    if rdoc_csv:
        frameworks.append(("RDoC", _series_from_csv(rdoc_csv, id_col, rdoc_col)))
    elif auto_rdoc is not None:
        frameworks.append(("RDoC", auto_rdoc))

    if not frameworks:
        raise ValueError(
            "No framework labels available. Provide CSVs or ensure inference succeeds."
        )

    results = {
        "subset": subset_name,
        "initial_subset_count": initial_count,
        "node_count": len(node_ids),
        "min_psy_score": min_psy_score,
        "psy_include_neighbors": psy_include_neighbors,
        "frameworks": {},
    }
    csv_rows = []
    for name, series in frameworks:
        metrics = _precision_recall(node_ids, series)
        per_label = _per_label_metrics(node_ids, series)
        metrics["per_label"] = per_label
        results["frameworks"][name] = metrics
        print(
            f"{name} ({subset_name} nodes, n={len(node_ids)}): "
            f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
            f"TP={metrics['tp']}, nodes_considered={metrics['nodes_considered']}, "
            f"labeled_nodes={metrics['labeled_nodes']}"
        )
        if per_label:
            print("  Per-label coverage:")
            for row in per_label:
                csv_rows.append({"framework": name, **row})
                print(
                    "    - {label}: overlap={overlap}, support={support}, "
                    "precision={precision:.3f}, recall={recall:.3f}".format(**row)
                )

    if save_json:
        save_json.write_text(json.dumps(results, indent=2))
    if per_label_csv and csv_rows:
        pd.DataFrame(csv_rows).to_csv(per_label_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="GraphML file")
    parser.add_argument(
        "--subset",
        choices=["graph", "psychiatric", "partition"],
        default="graph",
        help="Which node universe to evaluate: whole graph, nodes with is_psychiatric=True, or nodes present in a partition.",
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=None,
        help="Optional partition.json; required only when --subset=partition.",
    )
    parser.add_argument(
        "--prop-depth",
        type=int,
        default=1,
        help="Propagation depth used when auto-inferring RDoC labels.",
    )
    parser.add_argument(
        "--hpo-terms",
        type=Path,
        default=None,
        help="Optional override for the preprocessed HPO CSV used during label inference.",
    )
    parser.add_argument(
        "--hitop-map", type=Path, default=None, help="Optional CSV of HiTOP labels."
    )
    parser.add_argument(
        "--rdoc-map", type=Path, default=None, help="Optional CSV of RDoC labels."
    )
    parser.add_argument(
        "--min-psy-score",
        type=float,
        default=0.0,
        help="Drop nodes whose psy_score falls below this threshold before computing precision/recall.",
    )
    parser.add_argument(
        "--psy-include-neighbors",
        type=int,
        default=0,
        help="Number of hops of neighbors (within the chosen subset) to add after score filtering.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="node_id",
        help="Column shared by mapping CSVs identifying nodes.",
    )
    parser.add_argument(
        "--hitop-col",
        type=str,
        default="hitop_label",
        help="Column inside the HiTOP CSV containing the labels.",
    )
    parser.add_argument(
        "--rdoc-col",
        type=str,
        default="rdoc_label",
        help="Column inside the RDoC CSV containing the labels.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the precision/recall dictionary as JSON.",
    )
    parser.add_argument(
        "--per-label-csv",
        type=Path,
        default=None,
        help="Optional CSV file capturing precision/recall per HiTOP/RDoC label.",
    )
    args = parser.parse_args()
    if args.min_psy_score < 0:
        parser.error("--min-psy-score must be non-negative")
    if args.psy_include_neighbors < 0:
        parser.error("--psy-include-neighbors must be non-negative")
    run(
        graph_path=args.graph,
        subset=args.subset,
        partition_path=args.partition,
        prop_depth=args.prop_depth,
        hpo_terms=args.hpo_terms,
        hitop_csv=args.hitop_map,
        rdoc_csv=args.rdoc_map,
        id_col=args.id_col,
        hitop_col=args.hitop_col,
        rdoc_col=args.rdoc_col,
        min_psy_score=args.min_psy_score,
        psy_include_neighbors=args.psy_include_neighbors,
        save_json=args.save_json,
        per_label_csv=args.per_label_csv,
    )
