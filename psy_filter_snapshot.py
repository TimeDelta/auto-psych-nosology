#!/usr/bin/env python3.10
"""Utility to preview psychiatric relevance filtering outside the trainer.

This helper loads a multiplex GraphML artifact, applies the same psychiatric
filters that `train_rgcn_scae.py` uses, and summarizes how many nodes/edges
survive. It defaults to the "--min-psy-score 0.33 --psy-include-neighbors 0"
settings but also exposes equivalent CLI flags so you can sweep alternative
thresholds. Optionally, it writes a JSON payload plus a newline-delimited text
file listing every dropped node id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from train_rgcn_scae import (
    MultiplexGraph,
    _filter_graph_by_psy_metrics,
    load_multiplex_graph,
)

DEFAULT_MIN_PSY_SCORE = 0.33
DEFAULT_PSY_INCLUDE_NEIGHBORS = 0


def _graph_stats(graph: MultiplexGraph) -> Dict[str, int]:
    return {
        "num_nodes": int(graph.data.node_types.numel()),
        "num_edges": int(graph.data.edge_index.size(1)),
    }


def _collect_removed_nodes(
    original: MultiplexGraph, filtered: MultiplexGraph
) -> List[str]:
    filtered_ids = set(filtered.node_ids)
    return [node_id for node_id in original.node_ids if node_id not in filtered_ids]


def _default_graphml_out_path(graphml_path: Path) -> Path:
    return graphml_path.with_suffix(".psy-filtered.graphml")


def _write_filtered_graphml(
    source: Path, destination: Path, keep_ids: Set[str]
) -> None:
    graph = nx.read_graphml(source)
    drop_ids = [node for node in graph.nodes if node not in keep_ids]
    if drop_ids:
        graph.remove_nodes_from(drop_ids)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the psychiatric relevance screen used by train_rgcn_scae.py "
            "to a GraphML artifact produced by create_graph.py."
        )
    )
    parser.add_argument(
        "graphml",
        type=Path,
        help="Path to the GraphML file emitted by the graph construction pipeline.",
    )
    parser.add_argument(
        "--min-psy-score",
        type=float,
        default=DEFAULT_MIN_PSY_SCORE,
        help=(
            "Drop nodes with psy_score below this threshold before computing stats "
            "(default: 0.33)."
        ),
    )
    parser.add_argument(
        "--psy-include-neighbors",
        type=int,
        default=DEFAULT_PSY_INCLUDE_NEIGHBORS,
        help=(
            "Number of hops of neighbors to retain around nodes that pass the "
            "psychiatric filters (default: 0)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to store a JSON summary of the filtering results.",
    )
    parser.add_argument(
        "--dropped-node-list",
        type=Path,
        default=None,
        help=(
            "Optional path for a newline-delimited text file containing the node ids "
            "removed by the filter."
        ),
    )
    parser.add_argument(
        "--graphml-out",
        type=Path,
        default=None,
        help="Path to write the filtered GraphML (defaults to <graphml>.psy-filtered.graphml).",
    )
    parser.add_argument(
        "--no-graphml-export",
        action="store_true",
        help="Skip writing a filtered GraphML even when a default path is available.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence stdout logging (still writes files when requested).",
    )
    args = parser.parse_args()

    if args.min_psy_score < 0:
        parser.error("--min-psy-score must be non-negative")
    if args.psy_include_neighbors < 0:
        parser.error("--psy-include-neighbors must be non-negative")

    graphml_out: Optional[Path]
    if args.no_graphml_export:
        graphml_out = None
    elif args.graphml_out is not None:
        graphml_out = args.graphml_out.expanduser()
    else:
        graphml_out = _default_graphml_out_path(args.graphml)

    graph = load_multiplex_graph(
        args.graphml,
        text_encoder_model=None,
        enable_text_embedding_cache=False,
    )
    before_stats = _graph_stats(graph)

    filtered_graph, removed_nodes, removed_edges = _filter_graph_by_psy_metrics(
        graph,
        min_psy_score=args.min_psy_score,
        require_psychiatric_flag=False,
        neighbor_hops=args.psy_include_neighbors,
    )
    after_stats = _graph_stats(filtered_graph)
    dropped_ids = _collect_removed_nodes(graph, filtered_graph)
    keep_ids = set(filtered_graph.node_ids)

    summary: Dict[str, Any] = {
        "graph": str(args.graphml),
        "min_psy_score": args.min_psy_score,
        "psy_include_neighbors": args.psy_include_neighbors,
        "original_nodes": before_stats["num_nodes"],
        "original_edges": before_stats["num_edges"],
        "filtered_nodes": after_stats["num_nodes"],
        "filtered_edges": after_stats["num_edges"],
        "removed_nodes": removed_nodes,
        "removed_edges": removed_edges,
        "graphml_out": str(graphml_out) if graphml_out is not None else None,
    }

    if not args.quiet:
        print(
            "[psy_filter_snapshot] "
            f"Graph stats before filter: {before_stats['num_nodes']} nodes / "
            f"{before_stats['num_edges']} edges"
        )
        print(
            "[psy_filter_snapshot] "
            f"Graph stats after filter:  {after_stats['num_nodes']} nodes / "
            f"{after_stats['num_edges']} edges"
        )
        if dropped_ids:
            print(
                "[psy_filter_snapshot] "
                f"Removed {len(dropped_ids)} nodes (threshold {args.min_psy_score})"
            )
        else:
            print("[psy_filter_snapshot] No nodes removed by the psychiatric screen.")

    if graphml_out is not None:
        _write_filtered_graphml(args.graphml, graphml_out, keep_ids)
        if not args.quiet:
            print(
                "[psy_filter_snapshot] "
                f"Wrote filtered GraphML containing {after_stats['num_nodes']} nodes to {graphml_out}"
            )

    if args.json_out:
        json_payload = dict(summary)
        json_payload["removed_node_ids"] = dropped_ids
        args.json_out.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[psy_filter_snapshot] Wrote JSON summary to {args.json_out}")

    if args.dropped_node_list:
        if dropped_ids:
            args.dropped_node_list.write_text(
                "\n".join(dropped_ids) + "\n",
                encoding="utf-8",
            )
        else:
            args.dropped_node_list.write_text("", encoding="utf-8")
        if not args.quiet:
            print(
                "[psy_filter_snapshot] "
                f"Wrote dropped node list to {args.dropped_node_list}"
            )

    if args.json_out is None and args.dropped_node_list is None:
        # Provide a machine-friendly summary on stdout when no files requested.
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
