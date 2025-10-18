"""Utilities to train the rGCN-SCAE model directly from a multiplex GraphML file."""

from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Data

from self_compressing_auto_encoders import (
    NodeAttributeDeepSetEncoder,
    OnlineTrainer,
    PartitionResult,
    SelfCompressingRGCNAutoEncoder,
    SharedAttributeVocab,
)

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_TEXT_TOKEN_LIMIT_PER_FIELD = 32
_TEXT_TOKEN_LIMIT_TOTAL = 128


def _tokenize_text(value: str) -> List[str]:
    if not value:
        return []
    return _TOKEN_RE.findall(value.lower())


def _unique_tokens(tokens: Iterable[str], limit: int) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def _extract_text_tokens(raw: str, per_field_limit: int) -> List[str]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw

    if isinstance(payload, dict):
        source_iter: Iterable[Any] = payload.values()
    elif isinstance(payload, list):
        source_iter = payload
    else:
        source_iter = [payload]

    candidates: List[str] = []
    for value in source_iter:
        if isinstance(value, str):
            candidates.extend(_tokenize_text(value))
    return _unique_tokens(candidates, per_field_limit)


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value in (None, ""):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


@dataclass
class MultiplexGraph:
    """In-memory representation of a multiplex knowledge graph."""

    data: Data
    node_index: Mapping[str, int]
    node_type_index: Mapping[str, int]
    relation_index: Mapping[str, int]
    node_attributes: List[Dict[str, Any]]

    @property
    def node_names(self) -> List[str]:
        reverse = [None] * len(self.node_index)
        for name, idx in self.node_index.items():
            reverse[idx] = name
        return reverse


def _parse_graphml(
    graphml_path: Path,
) -> Tuple[List[Tuple[str, Dict[str, str]]], List[Tuple[str, str, Dict[str, str]]],]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(graphml_path)
    root = tree.getroot()

    key_registry: Dict[str, Tuple[str, str]] = {}
    for key_elem in root.findall(f"{GRAPHML_NS}key"):
        key_id = key_elem.attrib["id"]
        key_domain = key_elem.attrib.get("for", "")
        key_name = key_elem.attrib.get("attr.name", key_id)
        key_registry[key_id] = (key_domain, key_name)

    nodes: List[Tuple[str, Dict[str, str]]] = []
    for node_elem in root.findall(f".//{GRAPHML_NS}node"):
        node_id = node_elem.attrib["id"]
        attributes: Dict[str, str] = {}
        for data_elem in node_elem.findall(f"{GRAPHML_NS}data"):
            key_id = data_elem.attrib.get("key")
            if not key_id:
                continue
            _, key_name = key_registry.get(key_id, ("node", key_id))
            text = data_elem.text or ""
            attributes[key_name] = text.strip()
        nodes.append((node_id, attributes))

    edges: List[Tuple[str, str, Dict[str, str]]] = []
    for edge_elem in root.findall(f".//{GRAPHML_NS}edge"):
        src = edge_elem.attrib["source"]
        dst = edge_elem.attrib["target"]
        attributes: Dict[str, str] = {}
        for data_elem in edge_elem.findall(f"{GRAPHML_NS}data"):
            key_id = data_elem.attrib.get("key")
            if not key_id:
                continue
            _, key_name = key_registry.get(key_id, ("edge", key_id))
            text = data_elem.text or ""
            attributes[key_name] = text.strip()
        edges.append((src, dst, attributes))

    return nodes, edges


def load_multiplex_graph(graphml_path: Path) -> MultiplexGraph:
    nodes, edges = _parse_graphml(graphml_path)

    node_index: Dict[str, int] = {
        node_id: idx for idx, (node_id, _attrs) in enumerate(nodes)
    }

    node_types: List[str] = []
    node_type_index: Dict[str, int] = {}
    node_type_ids: List[int] = []
    node_attribute_dicts: List[Dict[str, Any]] = []

    metadata_fields = (
        "disease_metadata",
        "drug_metadata",
        "protein_metadata",
        "dna_metadata",
    )

    for node_id, attrs in nodes:
        node_type = attrs.get("node_type", "Unknown")
        if node_type not in node_type_index:
            node_type_index[node_type] = len(node_type_index)
            node_types.append(node_type)
        node_type_ids.append(node_type_index[node_type])

        attr_dict: Dict[str, Any] = {}
        token_budget = _TEXT_TOKEN_LIMIT_TOTAL

        if node_type:
            attr_dict[f"node_type::{node_type.lower()}"] = 1.0

        source = attrs.get("source")
        if source:
            attr_dict[f"source::{source.lower()}"] = 1.0

        name_value = attrs.get("name")
        if name_value:
            name_tokens = _unique_tokens(
                _tokenize_text(name_value), min(8, token_budget)
            )
            for token in name_tokens:
                attr_dict[f"name_token::{token}"] = 1.0
            token_budget -= len(name_tokens)

        psy_score = _parse_float(attrs.get("psy_score"))
        if psy_score is not None:
            attr_dict["psy_score"] = psy_score

        text_score = _parse_float(attrs.get("text_score"))
        if text_score is not None:
            attr_dict["text_score"] = text_score

        for flag_field in (
            "is_psychiatric",
            "ontology_flag",
            "group_flag",
            "drug_flag",
        ):
            flag_value = _parse_bool(attrs.get(flag_field))
            if flag_value is not None:
                attr_dict[flag_field] = 1.0 if flag_value else 0.0

        evidence_raw = attrs.get("psy_evidence")
        if evidence_raw:
            try:
                evidence_list = json.loads(evidence_raw)
            except json.JSONDecodeError:
                evidence_list = None
            if isinstance(evidence_list, list):
                for label in evidence_list:
                    if token_budget <= 0:
                        break
                    if isinstance(label, str):
                        token = label.strip().lower()
                        if not token:
                            continue
                        attr_dict[f"evidence::{token}"] = 1.0
                        token_budget -= 1

        for field in metadata_fields:
            if token_budget <= 0:
                break
            tokens = _extract_text_tokens(
                attrs.get(field, ""),
                min(_TEXT_TOKEN_LIMIT_PER_FIELD, token_budget),
            )
            for token in tokens:
                attr_dict[f"{field}::{token}"] = 1.0
            token_budget -= len(tokens)

        node_attribute_dicts.append(attr_dict)

    relation_index: Dict[str, int] = {}
    edge_pairs: List[Tuple[int, int]] = []
    edge_type_ids: List[int] = []
    for src, dst, attrs in edges:
        if src not in node_index or dst not in node_index:
            continue
        predicate = attrs.get("predicate", "rel")
        if predicate not in relation_index:
            relation_index[predicate] = len(relation_index)
        edge_pairs.append((node_index[src], node_index[dst]))
        edge_type_ids.append(relation_index[predicate])

    if not relation_index:
        relation_index["rel"] = 0
        edge_type_ids = [0 for _ in edge_pairs]

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        node_types=torch.tensor(node_type_ids, dtype=torch.long),
        edge_index=edge_index,
        edge_type=torch.tensor(edge_type_ids, dtype=torch.long)
        if edge_type_ids
        else torch.empty((0,), dtype=torch.long),
    )
    data.node_names = [node_id for node_id, _ in nodes]
    data.node_type_names = list(node_type_index.keys())
    data.edge_type_names = list(relation_index.keys())
    data.node_attributes = node_attribute_dicts

    return MultiplexGraph(
        data=data,
        node_index=node_index,
        node_type_index=node_type_index,
        relation_index=relation_index,
        node_attributes=node_attribute_dicts,
    )


def _default_cluster_capacity(graph: MultiplexGraph) -> int:
    """Heuristic for the latent cluster capacity used by the SCAE."""

    num_nodes = max(int(graph.data.node_types.numel()), 1)
    num_relations = max(len(graph.relation_index), 1)
    # Start with a sublinear growth heuristic so capacity scales with graph size
    # without exploding memory on dense graphs.
    sqrt_capacity = max(8, int(round(math.sqrt(num_nodes))))
    # Ensure we have at least a couple of prototypes per relation in multiplex
    # graphs so decoders can differentiate relation patterns.
    relation_floor = num_relations * 4
    return min(num_nodes, max(sqrt_capacity, relation_floor))


def _parse_mlflow_tags(raw_tags: Optional[Sequence[str]]) -> Dict[str, str]:
    if not raw_tags:
        return {}
    tags: Dict[str, str] = {}
    for candidate in raw_tags:
        if candidate is None:
            continue
        if "=" in candidate:
            key, value = candidate.split("=", 1)
        else:
            key, value = candidate, ""
        key = key.strip()
        if not key:
            continue
        tags[key] = value.strip()
    return tags


def _iter_attribute_dicts(
    node_attributes: Optional[List[Any]],
) -> Iterable[Dict[str, Any]]:
    if not node_attributes:
        return []
    if isinstance(node_attributes, list):
        for item in node_attributes:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        yield sub


def _populate_shared_vocab(
    shared_vocab: SharedAttributeVocab, node_attributes: Optional[List[Any]]
) -> None:
    attr_names: set[str] = set()
    string_values: set[str] = set()

    for attrs in _iter_attribute_dicts(node_attributes):
        for name, value in attrs.items():
            attr_names.add(str(name))
            if isinstance(value, str) and value:
                string_values.add(value)

    if attr_names:
        shared_vocab.add_names(sorted(attr_names))
    if string_values:
        shared_vocab.add_names(sorted(string_values))


@contextmanager
def _mlflow_run(
    enabled: bool,
    tracking_uri: Optional[str],
    experiment_name: Optional[str],
    run_name: Optional[str],
    tags: Mapping[str, str],
):
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requested but the 'mlflow' package is not installed."
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run_kwargs = {"run_name": run_name} if run_name else {}
    with mlflow.start_run(**run_kwargs):
        if tags:
            mlflow.set_tags(tags)
        yield mlflow


def train_scae_on_graph(
    graph: MultiplexGraph,
    num_clusters: Optional[int] = None,
    hidden_dims: Sequence[int] = (128, 128),
    type_embedding_dim: int = 64,
    attr_encoder_dims: Tuple[int, int, int] = (32, 64, 32),
    epochs: int = 200,
    lr: float = 1e-3,
    negative_sampling_ratio: float = 1.0,
    gate_threshold: float = 0.5,
    min_cluster_size: int = 1,
    device: Optional[str] = None,
    verbose: bool = True,
    entropy_weight: float = 1e-3,
    dirichlet_alpha: Optional[Sequence[float]] = None,
    dirichlet_weight: float = 1e-3,
    embedding_norm_weight: float = 1e-4,
    kld_weight: float = 1e-3,
    entropy_eps: float = 1e-12,
) -> Tuple[SelfCompressingRGCNAutoEncoder, PartitionResult, List[Dict[str, float]]]:
    if num_clusters is None:
        num_clusters = _default_cluster_capacity(graph)

    shared_vocab = SharedAttributeVocab(
        initial_names=[], embedding_dim=attr_encoder_dims[0]
    )
    _populate_shared_vocab(shared_vocab, getattr(graph.data, "node_attributes", None))
    attr_encoder = NodeAttributeDeepSetEncoder(
        shared_attr_vocab=shared_vocab,
        encoder_hdim=attr_encoder_dims[0],
        aggregator_hdim=attr_encoder_dims[1],
        out_dim=attr_encoder_dims[2],
    )

    if dirichlet_alpha is None:
        dirichlet_param: Union[float, Sequence[float]] = 0.5
    elif len(dirichlet_alpha) == 1:
        dirichlet_param = dirichlet_alpha[0]
    else:
        dirichlet_param = dirichlet_alpha

    model = SelfCompressingRGCNAutoEncoder(
        num_node_types=len(graph.node_type_index),
        attr_encoder=attr_encoder,
        num_clusters=num_clusters,
        num_relations=max(1, len(graph.relation_index)),
        hidden_dims=list(hidden_dims),
        type_embedding_dim=type_embedding_dim,
        negative_sampling_ratio=negative_sampling_ratio,
        entropy_weight=entropy_weight,
        dirichlet_alpha=dirichlet_param,
        dirichlet_weight=dirichlet_weight,
        embedding_norm_weight=embedding_norm_weight,
        kld_weight=kld_weight,
        entropy_eps=entropy_eps,
        active_gate_threshold=gate_threshold,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = OnlineTrainer(model, optimizer, device=device)
    trainer.add_data([graph.data])
    history = trainer.train(
        epochs=epochs,
        batch_size=1,
        negative_sampling_ratio=negative_sampling_ratio,
        verbose=verbose,
    )

    model.eval()
    with torch.no_grad():
        node_types = graph.data.node_types.to(trainer.device)
        edge_index = graph.data.edge_index.to(trainer.device)
        batch_vec = torch.zeros(
            node_types.size(0), dtype=torch.long, device=trainer.device
        )
        edge_type = getattr(graph.data, "edge_type", None)
        if edge_type is not None and edge_type.numel() > 0:
            edge_type = edge_type.to(trainer.device)
        else:
            edge_type = None

        _, assignments, metrics = model(
            node_types=node_types,
            edge_index=edge_index,
            batch=batch_vec,
            node_attributes=getattr(graph.data, "node_attributes", None),
            edge_type=edge_type,
            negative_sampling_ratio=0.0,
        )

    gate_values = metrics["cluster_gate_sample"]
    partition = SelfCompressingRGCNAutoEncoder.hard_partition(
        assignments=assignments.detach().cpu(),
        cluster_gate_values=gate_values,
        gate_threshold=gate_threshold,
        min_cluster_size=min_cluster_size,
    )

    return model, partition, history


def save_partition(
    partition: PartitionResult, output_path: Path, graph: MultiplexGraph
) -> None:
    reverse_node_map = graph.node_names
    cluster_members_named: Dict[str, List[str]] = {}
    for cluster_idx, member_idx in partition.cluster_members.items():
        cluster_members_named[str(cluster_idx)] = [
            reverse_node_map[i] for i in member_idx.tolist()
        ]

    payload = {
        "active_clusters": partition.active_clusters.tolist(),
        "gate_values": partition.gate_values.tolist(),
        "node_to_cluster": {
            reverse_node_map[i]: int(cluster_idx)
            for i, cluster_idx in enumerate(partition.node_to_cluster.tolist())
        },
        "cluster_members": cluster_members_named,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train rGCN-SCAE on a multiplex GraphML file"
    )
    parser.add_argument(
        "graphml", type=Path, help="Path to the GraphML file produced by the pipeline"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("partition.json"),
        help="Where to store the resulting partition JSON",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="*", default=(128, 128))
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--negative-sampling", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=1e-3,
        help="Weight applied to the entropy regularizer",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        nargs="*",
        default=None,
        help="Dirichlet concentration parameter(s); provide one value or K values",
    )
    parser.add_argument(
        "--dirichlet-weight",
        type=float,
        default=1e-3,
        help="Weight for the Dirichlet prior regularizer",
    )
    parser.add_argument(
        "--embedding-norm-weight",
        type=float,
        default=1e-4,
        help="Weight for the embedding norm regularizer",
    )
    parser.add_argument(
        "--kld-weight",
        type=float,
        default=1e-3,
        help="Weight for the KL divergence regularizer",
    )
    parser.add_argument(
        "--entropy-eps",
        type=float,
        default=1e-12,
        help="Numerical floor for entropy/Dirichlet calculations",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for this run",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="Optional MLflow tracking URI; otherwise uses the default profile",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="Name of the MLflow experiment to log under",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional MLflow run name",
    )
    parser.add_argument(
        "--mlflow-tag",
        type=str,
        action="append",
        dest="mlflow_tags",
        default=None,
        help="Additional MLflow tag(s) as KEY=VALUE. May be supplied multiple times.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    graph = load_multiplex_graph(args.graphml)
    effective_num_clusters = (
        args.clusters if args.clusters is not None else _default_cluster_capacity(graph)
    )

    tags = _parse_mlflow_tags(args.mlflow_tags)
    with _mlflow_run(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        tags=tags,
    ) as mlflow_client:
        if mlflow_client is not None:
            hidden_dims_param = ",".join(map(str, args.hidden)) if args.hidden else ""
            dirichlet_alpha_param = (
                ",".join(map(str, args.dirichlet_alpha)) if args.dirichlet_alpha else ""
            )
            mlflow_client.log_params(
                {
                    "graph_path": str(args.graphml),
                    "output_path": str(args.out),
                    "epochs": args.epochs,
                    "device": args.device or "auto",
                    "learning_rate": args.lr,
                    "negative_sampling_ratio": args.negative_sampling,
                    "gate_threshold": args.gate_threshold,
                    "min_cluster_size": args.min_cluster_size,
                    "entropy_weight": args.entropy_weight,
                    "dirichlet_weight": args.dirichlet_weight,
                    "embedding_norm_weight": args.embedding_norm_weight,
                    "kld_weight": args.kld_weight,
                    "entropy_eps": args.entropy_eps,
                    "hidden_dims": hidden_dims_param,
                    "effective_num_clusters": effective_num_clusters,
                    "dirichlet_alpha": dirichlet_alpha_param,
                }
            )
            mlflow_client.log_params(
                {
                    "num_nodes": int(graph.data.node_types.numel()),
                    "num_relations": len(graph.relation_index),
                    "num_node_types": len(graph.node_type_index),
                }
            )

        model, partition, history = train_scae_on_graph(
            graph,
            num_clusters=effective_num_clusters,
            hidden_dims=args.hidden,
            epochs=args.epochs,
            lr=args.lr,
            negative_sampling_ratio=args.negative_sampling,
            gate_threshold=args.gate_threshold,
            min_cluster_size=args.min_cluster_size,
            device=args.device,
            verbose=not args.quiet,
            entropy_weight=args.entropy_weight,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_weight=args.dirichlet_weight,
            embedding_norm_weight=args.embedding_norm_weight,
            kld_weight=args.kld_weight,
            entropy_eps=args.entropy_eps,
        )

        if mlflow_client is not None:
            for epoch_idx, metrics in enumerate(history, start=1):
                mlflow_client.log_metrics(metrics, step=epoch_idx)
            if history:
                final_loss = history[-1].get("loss")
                if final_loss is not None and math.isfinite(final_loss):
                    mlflow_client.log_metric("final_loss", float(final_loss))
            mlflow_client.log_metric(
                "num_active_clusters",
                len(partition.active_clusters),
                step=len(history) + 1,
            )
            type_embedding = getattr(model.encoder, "node_type_embedding", None)
            if type_embedding is not None and hasattr(type_embedding, "embedding_dim"):
                mlflow_client.log_param(
                    "type_embedding_dim", int(type_embedding.embedding_dim)
                )

            attr_encoder = getattr(model.encoder, "attr_encoder", None)
            if attr_encoder is not None and hasattr(attr_encoder, "out_dim"):
                mlflow_client.log_param(
                    "attr_encoder_out_dim", int(attr_encoder.out_dim)
                )

        save_partition(partition, args.out, graph)
        if mlflow_client is not None:
            mlflow_client.log_artifact(str(args.out))
            mlflow_client.log_metric(
                "partition_active_clusters", len(partition.active_clusters)
            )

    print(
        f"Saved partition with {len(partition.active_clusters)} clusters to {args.out}"
    )


if __name__ == "__main__":
    main()
