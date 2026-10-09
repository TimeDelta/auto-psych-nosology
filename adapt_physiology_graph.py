"""Adapt a frozen mechanistic-pathway-learning graph to the nosology trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import networkx as nx
import pandas as pd

from graph_export import sanitize_graph_for_graphml
from graph_quality import summarize_graph_quality
from nosology_filters import NOSOLOGY_NODE_TYPES
from relation_schema import encode_model_relation, sign_category

ADAPTER_VERSION = 1
REQUIRED_SOURCE_FILES = ("nodes.parquet", "edges.parquet", "relation_types.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(output_prefix: Path, suffix: str) -> Path:
    return Path(f"{output_prefix}{suffix}")


def _require_string_column(table: pd.DataFrame, column: str, table_name: str) -> None:
    if column not in table:
        raise ValueError(f"{table_name} is missing required column {column!r}")
    valid = table[column].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if not valid.all():
        raise ValueError(f"{table_name}.{column} must contain nonempty strings")


def build_adapted_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    declared_relations: list[str],
    *,
    relation_mode: str = "typed-signed",
) -> nx.MultiDiGraph:
    """Keep source identity, edge direction, parallel records and all source columns."""
    if relation_mode not in ("typed", "typed-signed"):
        raise ValueError(f"Unknown relation mode: {relation_mode}")
    for column in ("node_id", "node_type"):
        _require_string_column(nodes, column, "nodes")
    for column in ("source_id", "target_id", "relation_type"):
        _require_string_column(edges, column, "edges")
    if nodes.node_id.duplicated().any():
        raise ValueError("Duplicate node_id values would merge distinct source nodes")
    forbidden_types = NOSOLOGY_NODE_TYPES | {"disease", "diagnosis"}
    if nodes.node_type.str.strip().str.lower().isin(forbidden_types).any():
        raise ValueError("Physiology input contains disease or nosology node types")
    if (
        not isinstance(declared_relations, list)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in declared_relations
        )
        or len(set(declared_relations)) != len(declared_relations)
    ):
        raise ValueError(
            "relation_types.json must be a list of unique nonempty strings"
        )
    undeclared_relations = set(edges.relation_type) - set(declared_relations)
    if undeclared_relations:
        raise ValueError(f"Undeclared relation types: {sorted(undeclared_relations)}")
    missing_endpoints = (set(edges.source_id) | set(edges.target_id)) - set(
        nodes.node_id
    )
    if missing_endpoints:
        raise ValueError(
            f"Edge endpoints absent from nodes: {sorted(missing_endpoints)[:5]}"
        )

    graph = nx.MultiDiGraph(
        adapter_version=ADAPTER_VERSION, relation_mode=relation_mode
    )
    for source_record_index, row in enumerate(nodes.to_dict(orient="records")):
        attributes = dict(row)
        attributes.setdefault("name", row.get("display_name") or row["node_id"])
        attributes.setdefault("node_identifier", row["node_id"])
        attributes["adapter_source_file"] = "nodes.parquet"
        attributes["adapter_source_record_index"] = source_record_index
        graph.add_node(row["node_id"])
        graph.nodes[row["node_id"]].update(attributes)

    for source_record_index, row in enumerate(edges.to_dict(orient="records")):
        attributes = dict(row)
        source_sign = row.get("sign")
        source_sign_category = sign_category(source_sign)
        model_relation = encode_model_relation(
            row["relation_type"], source_sign, relation_mode
        )
        attributes.update(
            relation=row["relation_type"],
            model_relation=model_relation,
            predicate=model_relation,
            sign_category=source_sign_category,
            adapter_source_file="edges.parquet",
            adapter_source_record_index=source_record_index,
        )
        attributes.setdefault("source_file", "edges.parquet")
        attributes.setdefault("source_record_index", source_record_index)
        if row.get("evidence_source") is not None:
            attributes.setdefault("edge_source", row["evidence_source"])
        graph.add_edge(row["source_id"], row["target_id"], key=source_record_index)
        graph.edges[row["source_id"], row["target_id"], source_record_index].update(
            attributes
        )
    return graph


def adapt_physiology_graph(
    graph_directory: Path,
    output_prefix: Path,
    *,
    relation_mode: str = "typed-signed",
    expected_snapshot_sha256: str | None = None,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    """Hash source bytes, validate release pins and write inspectable adapter artifacts."""
    graph_directory = graph_directory.expanduser().resolve()
    output_prefix = output_prefix.expanduser().resolve()
    source_files = {name: graph_directory / name for name in REQUIRED_SOURCE_FILES}
    for source_path in source_files.values():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    if (graph_directory / "graph_summary.json").is_file():
        source_files["graph_summary.json"] = graph_directory / "graph_summary.json"
    source_hashes = {name: _sha256(path) for name, path in sorted(source_files.items())}
    snapshot_sha256 = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        expected_snapshot_sha256 is not None
        and snapshot_sha256 != expected_snapshot_sha256
    ):
        raise ValueError("Source snapshot SHA-256 does not match the requested pin")

    if source_manifest is None:
        candidate = graph_directory.parent / "MANIFEST.json"
        source_manifest = candidate if candidate.is_file() else None
    release_metadata = None
    source_manifest_sha256 = None
    if source_manifest is not None:
        source_manifest = source_manifest.expanduser().resolve()
        source_manifest_sha256 = _sha256(source_manifest)
        release_metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
        for filename, source_path in source_files.items():
            manifest_key = source_path.relative_to(source_manifest.parent).as_posix()
            recorded_hash = (
                release_metadata.get("files", {}).get(manifest_key, {}).get("sha256")
            )
            if recorded_hash != source_hashes[filename]:
                raise ValueError(f"Source release manifest mismatch for {manifest_key}")

    nodes = pd.read_parquet(source_files["nodes.parquet"])
    edges = pd.read_parquet(source_files["edges.parquet"])
    declared_relations = json.loads(
        source_files["relation_types.json"].read_text(encoding="utf-8")
    )
    graph = build_adapted_graph(
        nodes, edges, declared_relations, relation_mode=relation_mode
    )
    graph.graph["source_snapshot_sha256"] = snapshot_sha256
    relation_counts = Counter(
        attributes["model_relation"] for _, _, attributes in graph.edges(data=True)
    )
    relation_encodings = {}
    for _, _, attributes in graph.edges(data=True):
        name = attributes["model_relation"]
        encoding = relation_encodings.setdefault(
            name,
            {"relation_type": attributes["relation_type"], "sign_categories": set()},
        )
        encoding["sign_categories"].add(attributes["sign_category"])
    relations = [
        {
            "index": index,
            "model_relation": name,
            "relation_type": relation_encodings[name]["relation_type"],
            "sign_categories": sorted(relation_encodings[name]["sign_categories"]),
            "edge_count": relation_counts[name],
        }
        for index, name in enumerate(sorted(relation_counts))
    ]
    quality = summarize_graph_quality(graph)
    quality["model_relation_counts"] = dict(sorted(relation_counts.items()))
    quality["sign_category_counts"] = dict(
        sorted(
            Counter(
                attributes["sign_category"]
                for _, _, attributes in graph.edges(data=True)
            ).items()
        )
    )
    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "source_graph_directory": str(graph_directory),
        "source_snapshot_sha256": snapshot_sha256,
        "source_files": {
            name: {"sha256": source_hashes[name], "bytes": path.stat().st_size}
            for name, path in source_files.items()
        },
        "relation_mode": relation_mode,
        "declared_source_relations": declared_relations,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source_release": release_metadata,
        "source_manifest_sha256": source_manifest_sha256,
        "outputs": {},
        "interpretation": "Signs are encoded as categories, not enforced as signed propagation. Source rows are not independent-study counts. Evidence tables are not imported.",
    }

    output_paths = {
        "graphml": _artifact_path(output_prefix, ".graphml"),
        "nodes": _artifact_path(output_prefix, ".nodes.parquet"),
        "edges": _artifact_path(output_prefix, ".rels.parquet"),
        "relations": _artifact_path(output_prefix, ".relations.json"),
        "quality": _artifact_path(output_prefix, ".quality.json"),
    }
    if set(output_paths.values()) & set(source_files.values()):
        raise ValueError("Adapter output would overwrite a source file")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_prefix.parent) as staging_directory:
        staged_paths = {
            name: Path(staging_directory) / path.name
            for name, path in output_paths.items()
        }
        # Copies retain source bytes, including fields GraphML cannot represent exactly.
        shutil.copyfile(source_files["nodes.parquet"], staged_paths["nodes"])
        shutil.copyfile(source_files["edges.parquet"], staged_paths["edges"])
        sanitize_graph_for_graphml(graph)
        nx.write_graphml(graph, staged_paths["graphml"])
        staged_paths["relations"].write_text(
            json.dumps(relations, indent=2) + "\n", encoding="utf-8"
        )
        staged_paths["quality"].write_text(
            json.dumps(quality, indent=2) + "\n", encoding="utf-8"
        )
        for filename, source_path in source_files.items():
            if _sha256(source_path) != source_hashes[filename]:
                raise ValueError(f"Source changed during adaptation: {filename}")
        if (
            source_manifest is not None
            and _sha256(source_manifest) != source_manifest_sha256
        ):
            raise ValueError("Source release manifest changed during adaptation")
        manifest["outputs"] = {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in staged_paths.items()
        }
        staged_manifest = Path(staging_directory) / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for name, path in output_paths.items():
            os.replace(staged_paths[name], path)
        os.replace(staged_manifest, _artifact_path(output_prefix, ".manifest.json"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_directory", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--relation-mode", choices=("typed", "typed-signed"), default="typed-signed"
    )
    parser.add_argument("--expected-snapshot-sha256")
    parser.add_argument("--source-manifest", type=Path)
    arguments = parser.parse_args()
    manifest = adapt_physiology_graph(
        arguments.graph_directory,
        arguments.output_prefix,
        relation_mode=arguments.relation_mode,
        expected_snapshot_sha256=arguments.expected_snapshot_sha256,
        source_manifest=arguments.source_manifest,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "node_count",
                    "edge_count",
                    "relation_mode",
                    "source_snapshot_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
