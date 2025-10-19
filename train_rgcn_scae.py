"""Utilities to train the rGCN-SCAE model directly from a multiplex GraphML file."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from nosology_filters import should_drop_nosology_node
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


_NOSOLOGY_NODE_TYPES = {"disease", "disorder", "diagnosis"}
_NOSOLOGY_NAME_KEYWORDS = {
    "disorder",
    "disease",
    "syndrome",
    "diagnosis",
    "illness",
}


def _should_drop_nosology_node(attrs: Mapping[str, Any]) -> bool:
    node_type = (attrs.get("node_type") or "").strip().lower()
    if node_type in _NOSOLOGY_NODE_TYPES:
        return True
    # Flags indicating the node comes from an external taxonomy
    for flag_attr in ("ontology_flag", "group_flag", "is_psychiatric"):
        flag_val = attrs.get(flag_attr)
        if isinstance(flag_val, bool):
            if flag_val:
                return True
        else:
            parsed = _parse_bool(flag_val)
            if parsed:
                return True

    name = (attrs.get("name") or "").lower()
    if name and any(keyword in name for keyword in _NOSOLOGY_NAME_KEYWORDS):
        return True

    return False


@dataclass
class MultiplexGraph:
    """In-memory representation of a multiplex knowledge graph."""

    data: Data
    node_index: Mapping[str, int]
    node_type_index: Mapping[str, int]
    relation_index: Mapping[str, int]
    node_attributes: List[Dict[str, Any]]
    node_labels: List[str]
    node_ids: List[str]

    @property
    def node_names(self) -> List[str]:
        return self.node_labels


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

    node_types: List[str] = []
    node_type_index: Dict[str, int] = {}
    node_type_ids: List[int] = []
    node_attribute_dicts: List[Dict[str, Any]] = []
    node_labels: List[str] = []
    node_ids_ordered: List[str] = []

    metadata_fields = (
        "disease_metadata",
        "drug_metadata",
        "protein_metadata",
        "dna_metadata",
    )

    for node_id, attrs in nodes:
        if should_drop_nosology_node(attrs):
            continue
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
        node_labels.append(name_value if name_value else node_id)
        node_ids_ordered.append(node_id)

    if not node_ids_ordered:
        raise ValueError("No nodes remain after filtering nosology-aligned entries.")

    relation_index: Dict[str, int] = {}
    edge_pairs: List[Tuple[int, int]] = []
    edge_type_ids: List[int] = []
    node_index: Dict[str, int] = {
        node_id: idx for idx, node_id in enumerate(node_ids_ordered)
    }

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
    data.node_names = list(node_labels)
    data.node_ids = list(node_ids_ordered)
    data.node_type_names = list(node_type_index.keys())
    data.edge_type_names = list(relation_index.keys())
    data.node_attributes = node_attribute_dicts

    return MultiplexGraph(
        data=data,
        node_index=node_index,
        node_type_index=node_type_index,
        relation_index=relation_index,
        node_attributes=node_attribute_dicts,
        node_labels=node_labels,
        node_ids=node_ids_ordered,
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


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def _compute_adaptive_radii(
    data: Data, alpha: float, min_radius: int, max_radius: int
) -> np.ndarray:
    num_nodes = int(data.node_types.size(0))
    if num_nodes == 0:
        return np.zeros(0, dtype=int)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if data.edge_index.numel() > 0:
        edges = data.edge_index.t().cpu().tolist()
        graph.add_edges_from((int(u), int(v)) for u, v in edges)

    degree = np.array([graph.degree(n) for n in range(num_nodes)], dtype=float)
    if graph.number_of_edges() > 0:
        clustering_map = nx.clustering(graph)
    else:
        clustering_map = {}
    clustering = np.array([float(clustering_map.get(n, 0.0)) for n in range(num_nodes)])
    try:
        core_map = nx.core_number(graph)
    except nx.NetworkXError:
        core_map = {n: 0 for n in graph.nodes()}
    core = np.array([float(core_map.get(n, 0.0)) for n in range(num_nodes)])

    z_clustering = _safe_zscore(clustering)
    z_core = _safe_zscore(core)
    z_degree = _safe_zscore(degree)
    composite = z_clustering + z_core - z_degree
    radii = 1.0 + alpha * composite
    radii = np.clip(np.round(radii), min_radius, max_radius)
    return radii.astype(int)


def _build_adaptive_subgraph_dataset(
    graph: MultiplexGraph,
    num_samples: int,
    alpha: float,
    min_radius: int,
    max_radius: int,
    rng_seed: Optional[int],
    verbose: bool,
    min_nodes: int,
    min_edges: int,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> List[Data]:
    data = graph.data
    num_nodes = int(data.node_types.size(0))
    if num_nodes == 0 or num_samples <= 0:
        return [data]

    total_start = time.perf_counter()

    metrics_start = time.perf_counter()
    radii = _compute_adaptive_radii(data, alpha, min_radius, max_radius)
    if radii.size == 0:
        return [data]
    metrics_duration = time.perf_counter() - metrics_start

    rng = random.Random(rng_seed)
    population = list(range(num_nodes))

    node_attrs = getattr(data, "node_attributes", None)
    node_names = getattr(data, "node_names", None)
    node_ids = getattr(data, "node_ids", None)
    edge_type = getattr(data, "edge_type", None)

    dataset: List[Data] = []
    sampling_start = time.perf_counter()
    dropped_no_edges = 0
    dropped_too_small = 0
    sample_attempts = 0
    max_attempts = max(num_samples * 5, num_samples + 10)

    while len(dataset) < num_samples and sample_attempts < max_attempts:
        seed = rng.choice(population)
        sample_start = time.perf_counter()
        sample_attempts += 1
        radius = int(max(min_radius, min(max_radius, radii[seed])))

        def build_subgraph(current_radius: int) -> Tuple[Data, int, int, int]:
            subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                seed,
                current_radius,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            subset = subset.to(torch.long)
            sub = Data(
                node_types=data.node_types[subset].clone(),
                edge_index=edge_index_sub.contiguous(),
            )
            if edge_type is not None and edge_type.numel() > 0:
                sub.edge_type = edge_type[edge_mask].clone()
            else:
                sub.edge_type = torch.empty((0,), dtype=torch.long)
            if node_attrs:
                sub.node_attributes = [
                    copy.deepcopy(node_attrs[int(node_idx)])
                    for node_idx in subset.tolist()
                ]
            if node_names:
                sub.node_names = [
                    node_names[int(node_idx)] for node_idx in subset.tolist()
                ]
            if node_ids:
                sub.node_ids = [node_ids[int(node_idx)] for node_idx in subset.tolist()]
            sub.seed_index = torch.tensor([int(mapping)], dtype=torch.long)
            sub.seed_original_index = torch.tensor([int(seed)], dtype=torch.long)
            return (
                sub,
                current_radius,
                int(sub.node_types.size(0)),
                int(sub.edge_index.size(1)),
            )

        sub_data, radius_used, node_count, edge_count = build_subgraph(radius)
        while edge_count == 0 and radius_used < max_radius:
            new_radius = min(max_radius, radius_used + 1)
            sub_data, radius_used, node_count, edge_count = build_subgraph(new_radius)

        if (max_nodes is not None or max_edges is not None) and edge_count > 0:
            adjusted_radius = radius_used
            while (
                (max_nodes is not None and node_count > max_nodes)
                or (max_edges is not None and edge_count > max_edges)
            ) and adjusted_radius > min_radius:
                adjusted_radius -= 1
                sub_data, radius_used, node_count, edge_count = build_subgraph(
                    adjusted_radius
                )

        if edge_count == 0:
            dropped_no_edges += 1
            if verbose:
                sample_duration = time.perf_counter() - sample_start
                print(
                    "[TIMING] subgraph-skipped",
                    f"attempt={sample_attempts}",
                    f"seed={seed}",
                    f"radius={radius_used}",
                    f"nodes={node_count}",
                    "reason=no_edges",
                    f"duration={sample_duration:.3f}s",
                )
            continue

        if (
            node_count < min_nodes
            or edge_count < min_edges
            or (max_nodes is not None and node_count > max_nodes)
            or (max_edges is not None and edge_count > max_edges)
        ):
            if radius_used < max_radius:
                sub_data, radius_used, node_count, edge_count = build_subgraph(
                    max_radius
                )

        if (
            node_count < min_nodes
            or edge_count < min_edges
            or (max_nodes is not None and node_count > max_nodes)
            or (max_edges is not None and edge_count > max_edges)
        ):
            dropped_too_small += 1
            if verbose:
                sample_duration = time.perf_counter() - sample_start
                print(
                    "[TIMING] subgraph-skipped",
                    f"attempt={sample_attempts}",
                    f"seed={seed}",
                    f"radius={radius_used}",
                    f"nodes={node_count}",
                    f"edges={edge_count}",
                    "reason=too_small",
                    f"duration={sample_duration:.3f}s",
                )
            continue

        dataset.append(sub_data)

        if verbose:
            sample_duration = time.perf_counter() - sample_start
            print(
                "[TIMING] subgraph",
                f"index={len(dataset)-1}",
                f"attempt={sample_attempts}",
                f"seed={seed}",
                f"radius={radius_used}",
                f"nodes={node_count}",
                f"edges={edge_count}",
                f"duration={sample_duration:.3f}s",
            )

    fill_count = 0
    if len(dataset) < num_samples:
        if dataset:
            base_samples = [sample.clone() for sample in dataset]
        else:
            base_samples = [graph.data.clone()]
            dataset.extend(base_samples)
        while len(dataset) < num_samples:
            source = base_samples[
                (len(dataset) - len(base_samples)) % len(base_samples)
            ]
            dataset.append(source.clone())
            fill_count += 1

    sampling_duration = time.perf_counter() - sampling_start
    total_duration = time.perf_counter() - total_start
    if verbose:
        print(
            "[TIMING] dataset",
            f"samples={len(dataset)}",
            f"dropped_no_edges={dropped_no_edges}",
            f"dropped_too_small={dropped_too_small}",
            f"attempts={sample_attempts}",
            f"fills={fill_count}",
            f"metrics={metrics_duration:.3f}s",
            f"sampling={sampling_duration:.3f}s",
            f"total={total_duration:.3f}s",
        )

    return dataset[:num_samples]


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
    batch_size: int = 1,
    min_epochs: int = 0,
    cluster_stability_window: int = 0,
    cluster_stability_tolerance: float = 0.0,
    cluster_stability_relative_tolerance: Optional[float] = None,
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
    ego_samples: int = 0,
    ego_alpha: float = 0.75,
    ego_min_radius: int = 1,
    ego_max_radius: int = 3,
    ego_seed: Optional[int] = None,
    ego_min_nodes: int = 4,
    ego_min_edges: int = 2,
    ego_max_nodes: Optional[int] = None,
    ego_max_edges: Optional[int] = None,
    node_budget: Optional[int] = 60000,
    epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Tuple[SelfCompressingRGCNAutoEncoder, PartitionResult, List[Dict[str, float]]]:
    if num_clusters is None:
        num_clusters = _default_cluster_capacity(graph)

    shared_vocab = SharedAttributeVocab(
        initial_names=[], embedding_dim=attr_encoder_dims[0]
    )
    vocab_start = time.perf_counter()
    _populate_shared_vocab(shared_vocab, getattr(graph.data, "node_attributes", None))
    vocab_duration = time.perf_counter() - vocab_start
    if verbose:
        print(
            "[TIMING] shared vocab populated",
            f"entries={len(shared_vocab.name_to_index)}",
            f"duration={vocab_duration:.3f}s",
        )

    encoder_start = time.perf_counter()
    attr_encoder = NodeAttributeDeepSetEncoder(
        shared_attr_vocab=shared_vocab,
        encoder_hdim=attr_encoder_dims[0],
        aggregator_hdim=attr_encoder_dims[1],
        out_dim=attr_encoder_dims[2],
    )
    encoder_duration = time.perf_counter() - encoder_start
    if verbose:
        print(
            "[TIMING] attr encoder initialized",
            f"duration={encoder_duration:.3f}s",
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

    if ego_samples > 0:
        dataset = _build_adaptive_subgraph_dataset(
            graph,
            num_samples=ego_samples,
            alpha=ego_alpha,
            min_radius=ego_min_radius,
            max_radius=ego_max_radius,
            rng_seed=ego_seed,
            verbose=verbose,
            min_nodes=ego_min_nodes,
            min_edges=ego_min_edges,
            max_nodes=ego_max_nodes,
            max_edges=ego_max_edges,
        )
    else:
        dataset = [graph.data]

    if not dataset:
        dataset = [graph.data]

    if verbose:
        node_counts = np.array(
            [int(sample.node_types.size(0)) for sample in dataset], dtype=float
        )
        edge_counts = np.array(
            [int(sample.edge_index.size(1)) for sample in dataset], dtype=float
        )
        print(
            "[INFO] Training dataset",
            f"samples={len(dataset)}",
            f"nodes_mean={node_counts.mean():.1f}",
            f"nodes_std={node_counts.std():.1f}",
            f"edges_mean={edge_counts.mean():.1f}",
            f"edges_std={edge_counts.std():.1f}",
        )

    trainer = OnlineTrainer(model, optimizer, device=device, log_timing=verbose)
    add_start = time.perf_counter()
    trainer.add_data(dataset)
    add_duration = time.perf_counter() - add_start
    if verbose:
        print(f"[TIMING] trainer.add_data took {add_duration:.3f}s")

    history = trainer.train(
        epochs=epochs,
        batch_size=max(1, int(batch_size)),
        negative_sampling_ratio=negative_sampling_ratio,
        verbose=verbose,
        bucket_by_size=node_budget is None,
        node_budget=node_budget,
        on_epoch_end=epoch_callback,
        stability_metric="num_active_clusters"
        if cluster_stability_window > 0
        else None,
        stability_window=cluster_stability_window,
        stability_tolerance=cluster_stability_tolerance,
        stability_relative_tolerance=cluster_stability_relative_tolerance,
        min_epochs=min_epochs,
    )

    if trainer.early_stop_epoch is not None:
        message = f"Early stop at epoch {trainer.early_stop_epoch}: {trainer.early_stop_reason}"
        if verbose:
            print(f"[INFO] {message}")
        if history:
            history[-1]["early_stop_epoch"] = float(trainer.early_stop_epoch)

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
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=50,
        help="Minimum number of epochs before the stability criterion can stop training (must be <= --epochs).",
    )
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="*", default=(128, 128))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Graphs per batch when node budget sampling is disabled",
    )
    parser.add_argument(
        "--node-budget",
        type=int,
        default=60000,
        help="Total node budget per batch (set to 0 to disable node-budget batching)",
    )
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
        "--cluster-stability-window",
        type=int,
        default=15,
        help="Number of trailing epochs used to test num_active_clusters stability for early stopping (0 disables).",
    )
    parser.add_argument(
        "--cluster-stability-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for num_active_clusters variation across the stability window.",
    )
    parser.add_argument(
        "--cluster-stability-rel-tol",
        type=float,
        default=0.0,
        help="Relative tolerance (fraction of the mean) allowed for num_active_clusters across the window.",
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
        "--ego-samples",
        type=int,
        default=0,
        help="Number of adaptive ego-net subgraphs to sample for training (0 uses full graph)",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Enable ego-net subsampling defaults suited for stability sweeps",
    )
    parser.add_argument(
        "--ego-alpha",
        type=float,
        default=0.75,
        help="Scaling coefficient for the connectivity-based hop radius",
    )
    parser.add_argument(
        "--ego-min-radius",
        type=int,
        default=1,
        help="Minimum hop radius for sampled ego-nets",
    )
    parser.add_argument(
        "--ego-max-radius",
        type=int,
        default=3,
        help="Maximum hop radius for sampled ego-nets",
    )
    parser.add_argument(
        "--ego-min-nodes",
        type=int,
        default=10,
        help="Minimum number of nodes required for an ego-net sample",
    )
    parser.add_argument(
        "--ego-min-edges",
        type=int,
        default=9,
        help="Minimum number of edges required for an ego-net sample",
    )
    parser.add_argument(
        "--ego-max-nodes",
        type=int,
        default=None,
        help="Optional cap on nodes per ego-net sample (None leaves adaptive radius)",
    )
    parser.add_argument(
        "--ego-max-edges",
        type=int,
        default=None,
        help="Optional cap on edges per ego-net sample (None leaves adaptive radius)",
    )
    parser.add_argument(
        "--ego-seed",
        type=int,
        default=None,
        help="Optional RNG seed for ego-net sampling",
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

    if args.min_epochs > args.epochs:
        parser.error("--min-epochs must be less than or equal to --epochs")
    if args.cluster_stability_window < 0:
        parser.error("--cluster-stability-window must be non-negative")
    if (
        args.cluster_stability_window > 0
        and args.cluster_stability_window > args.epochs
    ):
        parser.error("--cluster-stability-window cannot exceed --epochs")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.node_budget is not None and args.node_budget < 0:
        parser.error("--node-budget must be non-negative")
    if args.node_budget == 0:
        args.node_budget = None
    if args.ego_min_radius < 1:
        parser.error("--ego-min-radius must be at least 1")
    if args.ego_max_radius < args.ego_min_radius:
        parser.error("--ego-max-radius must be >= --ego-min-radius")
    if args.ego_min_nodes < 1:
        parser.error("--ego-min-nodes must be positive")
    if args.ego_min_edges < 0:
        parser.error("--ego-min-edges must be non-negative")
    if args.ego_max_nodes is not None and args.ego_max_nodes < 1:
        parser.error("--ego-max-nodes must be positive when provided")
    if args.ego_max_edges is not None and args.ego_max_edges < 1:
        parser.error("--ego-max-edges must be positive when provided")

    if args.stability and args.ego_samples == 0:
        args.ego_samples = 256

    graph = load_multiplex_graph(args.graphml)
    num_nodes_after = int(graph.data.node_types.numel())
    num_edges_after = int(graph.data.edge_index.size(1))
    print(
        f"Graph after nosology filter: {num_nodes_after} nodes, {num_edges_after} edges"
    )
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
                    "min_epochs": args.min_epochs,
                    "device": args.device or "auto",
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "node_budget": args.node_budget
                    if args.node_budget is not None
                    else "",
                    "negative_sampling_ratio": args.negative_sampling,
                    "gate_threshold": args.gate_threshold,
                    "min_cluster_size": args.min_cluster_size,
                    "entropy_weight": args.entropy_weight,
                    "cluster_stability_window": args.cluster_stability_window,
                    "cluster_stability_tol": args.cluster_stability_tol,
                    "cluster_stability_rel_tol": args.cluster_stability_rel_tol,
                    "dirichlet_weight": args.dirichlet_weight,
                    "embedding_norm_weight": args.embedding_norm_weight,
                    "kld_weight": args.kld_weight,
                    "entropy_eps": args.entropy_eps,
                    "ego_samples": args.ego_samples,
                    "ego_alpha": args.ego_alpha,
                    "ego_min_radius": args.ego_min_radius,
                    "ego_max_radius": args.ego_max_radius,
                    "ego_min_nodes": args.ego_min_nodes,
                    "ego_min_edges": args.ego_min_edges,
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

        epoch_logger: Optional[Callable[[int, Dict[str, float]], None]] = None
        if mlflow_client is not None:

            def _epoch_logger(epoch_idx: int, metrics: Dict[str, float]) -> None:
                safe_metrics: Dict[str, float] = {}
                for key, value in metrics.items():
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric):
                        safe_metrics[key] = numeric
                if safe_metrics:
                    mlflow_client.log_metrics(safe_metrics, step=epoch_idx)

            epoch_logger = _epoch_logger

        model, partition, history = train_scae_on_graph(
            graph,
            num_clusters=effective_num_clusters,
            hidden_dims=args.hidden,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            cluster_stability_window=args.cluster_stability_window,
            cluster_stability_tolerance=args.cluster_stability_tol,
            cluster_stability_relative_tolerance=args.cluster_stability_rel_tol,
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
            batch_size=args.batch_size,
            ego_samples=args.ego_samples,
            ego_alpha=args.ego_alpha,
            ego_min_radius=args.ego_min_radius,
            ego_max_radius=args.ego_max_radius,
            ego_seed=args.ego_seed,
            ego_min_nodes=args.ego_min_nodes,
            ego_min_edges=args.ego_min_edges,
            ego_max_nodes=args.ego_max_nodes,
            ego_max_edges=args.ego_max_edges,
            node_budget=args.node_budget,
            epoch_callback=epoch_logger,
        )

        if mlflow_client is not None:
            if history:
                final_loss = history[-1].get("loss")
                if final_loss is not None and math.isfinite(final_loss):
                    mlflow_client.log_metric("final_loss", float(final_loss))
                if "early_stop_epoch" in history[-1]:
                    mlflow_client.log_metric(
                        "early_stop_epoch", history[-1]["early_stop_epoch"]
                    )
            mlflow_client.log_metric("epochs_completed", len(history))
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
