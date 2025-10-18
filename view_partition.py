#!/usr/bin/env python3
"""CLI for inspecting rGCN-SCAE partition artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and inspect a trained partition.json artifact."
    )
    parser.add_argument(
        "partition",
        type=Path,
        help="Path to partition.json produced by train_rgcn_scae.py",
    )
    parser.add_argument(
        "--sort",
        choices=("gate", "size", "cluster"),
        default="gate",
        help="Column used to order clusters in the summary (default: gate).",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=5,
        help="Number of node names to show per cluster (default: 5).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="*",
        default=None,
        help="Optional cluster ids to inspect; omit to show every cluster.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Substring filter applied to node names before printing samples.",
    )
    return parser.parse_args(argv)


def load_partition(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Partition file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON in {path}: {exc}") from exc

    for field in ("gate_values", "node_to_cluster"):
        if field not in payload:
            raise SystemExit(f"Missing required field '{field}' in {path}")

    return payload


def default_cluster_members(node_to_cluster: Dict[str, int]) -> Dict[int, List[str]]:
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node_name, raw_cluster in node_to_cluster.items():
        try:
            cluster_idx = int(raw_cluster)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Expected integer cluster id for node '{node_name}', got {raw_cluster}"
            ) from exc
        clusters[cluster_idx].append(node_name)
    return clusters


def merge_cluster_members(
    generated: Dict[int, List[str]],
    provided: Dict[str, List[str]] | None,
) -> Dict[int, List[str]]:
    if not provided:
        return generated

    merged = dict(generated)
    for cluster_id, names in provided.items():
        try:
            idx = int(cluster_id)
        except ValueError as exc:
            raise SystemExit(
                f"Cluster key '{cluster_id}' in cluster_members is not an integer"
            ) from exc
        merged[idx] = list(names)
    return merged


def build_rows(
    gate_values: Sequence[float],
    active_clusters: Iterable[int],
    cluster_members: Dict[int, List[str]],
    top_nodes: int,
    search: str | None,
    only_clusters: Sequence[int] | None,
) -> List[Dict[str, object]]:
    active = set(active_clusters)
    rows: List[Dict[str, object]] = []

    if only_clusters is not None:
        only_set = set(only_clusters)
    else:
        only_set = None

    for cluster_idx, gate_value in enumerate(gate_values):
        if only_set is not None and cluster_idx not in only_set:
            continue

        members = cluster_members.get(cluster_idx, [])
        if search:
            lowered = search.lower()
            filtered_members = [name for name in members if lowered in name.lower()]
            sample_members = filtered_members[:top_nodes]
            match_count = len(filtered_members)
        else:
            sample_members = members[:top_nodes]
            match_count = None

        rows.append(
            {
                "cluster": cluster_idx,
                "active": cluster_idx in active,
                "gate": float(gate_value),
                "size": len(members),
                "sample": sample_members,
                "matches": match_count,
            }
        )

    return rows


def sort_rows(rows: List[Dict[str, object]], strategy: str) -> None:
    if strategy == "gate":
        rows.sort(key=lambda r: (-r["gate"], -r["size"], r["cluster"]))
    elif strategy == "size":
        rows.sort(key=lambda r: (-r["size"], -r["gate"], r["cluster"]))
    else:  # cluster
        rows.sort(key=lambda r: r["cluster"])


def format_table(rows: List[Dict[str, object]], search: str | None) -> str:
    headers = ["cluster", "status", "gate", "size", "sample"]
    if search:
        headers.insert(-1, "sample_matches")

    str_rows: List[List[str]] = []
    for row in rows:
        status = "active" if row["active"] else "inactive"
        sample = ", ".join(row["sample"]) if row["sample"] else ""
        str_row: List[str] = [
            str(row["cluster"]),
            status,
            f"{row['gate']:.3f}",
            str(row["size"]),
        ]
        if search:
            str_row.append(str(row["matches"]))
        str_row.append(sample)
        str_rows.append(str_row)

    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(parts: Sequence[str]) -> str:
        return "  ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    lines = [render_line(headers)]
    lines.append(render_line(["-" * w for w in widths]))
    lines.extend(render_line(row) for row in str_rows)
    return "\n".join(lines)


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    payload = load_partition(args.partition)

    gate_values = payload["gate_values"]
    if not isinstance(gate_values, list):
        raise SystemExit("gate_values must be a list of floats")

    node_to_cluster = payload["node_to_cluster"]
    if not isinstance(node_to_cluster, dict):
        raise SystemExit("node_to_cluster must map node names to cluster ids")

    auto_members = default_cluster_members(node_to_cluster)
    provided_members = payload.get("cluster_members")
    merged_members = merge_cluster_members(auto_members, provided_members)

    active_clusters = payload.get("active_clusters", [])
    rows = build_rows(
        gate_values=gate_values,
        active_clusters=active_clusters,
        cluster_members=merged_members,
        top_nodes=args.top_nodes,
        search=args.search,
        only_clusters=args.clusters,
    )

    if not rows:
        print("No clusters to display.")
        return

    sort_rows(rows, args.sort)

    total_nodes = len(node_to_cluster)
    total_clusters = len(gate_values)
    total_active = len(set(active_clusters))

    header = (
        f"Partition: {args.partition} | nodes={total_nodes} | "
        f"clusters={total_clusters} | active={total_active}"
    )
    print(header)
    print(format_table(rows, args.search))


if __name__ == "__main__":
    main(sys.argv[1:])
