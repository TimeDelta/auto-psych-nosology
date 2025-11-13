"""Utilities to train the rGCN-SCAE model directly from a multiplex GraphML file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
import warnings
from collections import defaultdict
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
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

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
_NAME_TEXT_EMBED_ATTR = "name_text_embedding"
_TEXT_EMBED_CACHE_VERSION = 1


def _default_npz_export_path(base: Path) -> Path:
    base = base.expanduser()
    if base.suffix:
        return base.with_suffix(base.suffix + ".npz")
    return base.with_suffix(".npz")


def _checkpoint_tracker_path(base: Path) -> Path:
    if base.suffix:
        return base.with_suffix(base.suffix + ".mlflow.json")
    return base.with_name(base.name + ".mlflow.json")


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            pass
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        try:
            random.setstate(python_state)
        except Exception:
            pass
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            np.random.set_state(numpy_state)
        except Exception:
            pass
    torch_state = state.get("torch")
    if torch_state is not None:
        try:
            torch.random.set_rng_state(torch_state)
        except Exception:
            pass
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass


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


def _compute_text_fingerprint(node_ids: Sequence[str], texts: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    count = min(len(node_ids), len(texts))
    hasher.update(count.to_bytes(8, "little", signed=False))
    for node_id, text in zip(node_ids, texts):
        node_bytes = (node_id or "").encode("utf-8", "surrogatepass")
        text_bytes = (text or "").encode("utf-8", "surrogatepass")
        hasher.update(len(node_bytes).to_bytes(4, "little", signed=False))
        hasher.update(node_bytes)
        hasher.update(len(text_bytes).to_bytes(4, "little", signed=False))
        hasher.update(text_bytes)
    return hasher.hexdigest()


def _sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "")
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "model"
    return sanitized[:128]


def _default_text_embedding_cache_path(
    graphml_path: Path,
    model_name: str,
    normalize: bool,
    projection_dim: Optional[int],
) -> Path:
    safe_model = _sanitize_for_filename(model_name)
    norm_suffix = "norm" if normalize else "raw"
    proj_suffix = f"proj{int(projection_dim)}" if projection_dim else "proj0"
    suffix = (
        f".{safe_model}.v{_TEXT_EMBED_CACHE_VERSION}.{norm_suffix}.{proj_suffix}.npz"
    )
    return graphml_path.with_suffix(suffix)


def _project_embeddings_random(
    embeddings: np.ndarray, output_dim: Optional[int], fingerprint: str
) -> np.ndarray:
    if output_dim is None or output_dim <= 0:
        return embeddings
    if embeddings.ndim != 2:
        return embeddings
    input_dim = embeddings.shape[1]
    if input_dim == 0 or output_dim >= input_dim:
        return embeddings
    seed_bytes = hashlib.sha256(
        f"{fingerprint}|{input_dim}|{output_dim}".encode("utf-8", "surrogatepass")
    ).digest()
    seed_int = int.from_bytes(seed_bytes[:8], "little", signed=False)
    rng = np.random.default_rng(seed_int)
    projection = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    projection /= math.sqrt(max(output_dim, 1))
    projected = embeddings @ projection
    return projected.astype(np.float32, copy=False)


def _encode_sentence_embeddings(
    texts: Sequence[str],
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 128,
    normalize: bool = False,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "sentence-transformers must be installed to encode node text labels"
        ) from exc

    resolved_device = device
    if resolved_device is None or resolved_device.strip().lower() in {"", "auto"}:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        list(texts),
        batch_size=max(1, batch_size),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings.astype(np.float32, copy=False)


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


def load_multiplex_graph(
    graphml_path: Path,
    *,
    text_encoder_model: Optional[str] = "pritamdeka/BioSimCSE-BioBERT-base-clinical",
    text_encoder_device: Optional[str] = None,
    text_encoder_batch_size: int = 128,
    text_encoder_normalize: bool = False,
    text_embedding_cache: Optional[Path] = None,
    enable_text_embedding_cache: bool = True,
    text_encoder_projection_dim: Optional[int] = 128,
) -> MultiplexGraph:
    raw_text_encoder_model = text_encoder_model
    text_encoder_model = None
    if raw_text_encoder_model:
        normalized_encoder = raw_text_encoder_model.strip().lower()
        if normalized_encoder not in {"none", "", "disable", "disabled", "off"}:
            text_encoder_model = raw_text_encoder_model

    graphml_path = graphml_path.expanduser()
    cache_path: Optional[Path] = None
    if text_encoder_model and enable_text_embedding_cache:
        if text_embedding_cache is not None:
            cache_path = text_embedding_cache.expanduser()
        else:
            cache_path = _default_text_embedding_cache_path(
                graphml_path,
                text_encoder_model,
                text_encoder_normalize,
                text_encoder_projection_dim,
            )
    nodes, edges = _parse_graphml(graphml_path)

    node_types: List[str] = []
    node_type_index: Dict[str, int] = {}
    node_type_ids: List[int] = []
    node_attribute_dicts: List[Dict[str, Any]] = []
    node_labels: List[str] = []
    node_ids_ordered: List[str] = []
    node_texts: List[str] = []

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

        raw_name = attrs.get("name")
        cleaned_name = raw_name.strip() if isinstance(raw_name, str) else ""
        label_for_node = cleaned_name if cleaned_name else node_id

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
        node_labels.append(label_for_node)
        node_ids_ordered.append(node_id)
        node_texts.append(label_for_node)

    if not node_ids_ordered:
        raise ValueError("No nodes remain after filtering nosology-aligned entries.")

    text_embeddings_np: Optional[np.ndarray] = None
    if text_encoder_model and node_attribute_dicts:
        target_projection_dim = (
            text_encoder_projection_dim
            if text_encoder_projection_dim is not None
            and text_encoder_projection_dim > 0
            else None
        )
        text_fingerprint = _compute_text_fingerprint(node_ids_ordered, node_texts)
        cache_loaded = False
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cached_version = int(cached["version"][0])
                    cached_model = str(cached["model_name"][0])
                    cached_normalize = bool(int(cached["normalize"][0]))
                    cached_fingerprint = str(cached["fingerprint"][0])
                    cached_project_dim = int(
                        cached["project_dim"][0] if "project_dim" in cached else 0
                    )
                    cached_embeddings = cached["embeddings"]
                if (
                    cached_version == _TEXT_EMBED_CACHE_VERSION
                    and cached_model == text_encoder_model
                    and cached_normalize == bool(text_encoder_normalize)
                    and cached_embeddings.shape[0] == len(node_attribute_dicts)
                    and cached_fingerprint == text_fingerprint
                    and cached_project_dim == int(target_projection_dim or 0)
                ):
                    text_embeddings_np = cached_embeddings.astype(
                        np.float32, copy=False
                    )
                    cache_loaded = True
                    print(
                        f"[train_rgcn_scae] Loaded {text_embeddings_np.shape[0]} node text embeddings from cache '{cache_path}'."
                    )
                else:
                    print(
                        "[train_rgcn_scae] Text embedding cache metadata mismatch; recomputing."
                    )
            except Exception as cache_exc:
                print(
                    f"[train_rgcn_scae] Failed to load text embedding cache '{cache_path}': {cache_exc}"
                )

        if not cache_loaded:
            try:
                text_embeddings_np = _encode_sentence_embeddings(
                    node_texts,
                    text_encoder_model,
                    device=text_encoder_device,
                    batch_size=text_encoder_batch_size,
                    normalize=text_encoder_normalize,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to encode node text with SentenceTransformer '{text_encoder_model}': {exc}"
                ) from exc

            text_embeddings_np = text_embeddings_np.astype(np.float32, copy=False)

            if text_embeddings_np.shape[0] != len(node_attribute_dicts):
                raise RuntimeError(
                    "Text embedding count does not match node attribute count."
                )

            if target_projection_dim is not None:
                text_embeddings_np = _project_embeddings_random(
                    text_embeddings_np, target_projection_dim, text_fingerprint
                )

            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                try:
                    np.savez_compressed(
                        cache_path,
                        embeddings=text_embeddings_np.astype(np.float32, copy=False),
                        version=np.array([_TEXT_EMBED_CACHE_VERSION], dtype=np.int64),
                        model_name=np.array([text_encoder_model]),
                        normalize=np.array(
                            [1 if text_encoder_normalize else 0], dtype=np.int8
                        ),
                        fingerprint=np.array([text_fingerprint]),
                        project_dim=np.array(
                            [int(target_projection_dim or 0)], dtype=np.int64
                        ),
                    )
                    print(
                        f"[train_rgcn_scae] Cached {text_embeddings_np.shape[0]} node text embeddings to '{cache_path}'."
                    )
                except Exception as cache_exc:
                    print(
                        f"[train_rgcn_scae] Failed to write text embedding cache '{cache_path}': {cache_exc}"
                    )

        if text_embeddings_np is not None:
            for idx, attr_dict in enumerate(node_attribute_dicts):
                embedding_tensor = torch.from_numpy(text_embeddings_np[idx]).clone()
                attr_dict[_NAME_TEXT_EMBED_ATTR] = embedding_tensor

    relation_index: Dict[str, int] = {}
    edge_pairs: List[Tuple[int, int]] = []
    edge_type_ids: List[int] = []
    edge_weights: List[float] = []
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
        weight_value = attrs.get("weight")
        if weight_value in (None, ""):
            edge_weights.append(1.0)
        else:
            try:
                edge_weights.append(float(weight_value))
            except (TypeError, ValueError):
                edge_weights.append(1.0)

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
    if edge_weights:
        data.edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    if text_embeddings_np is not None and text_embeddings_np.size > 0:
        data.name_text_embedding_dim = int(text_embeddings_np.shape[1])
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


@dataclass(frozen=True)
class _AutoRegularizerConfig:
    entropy_weight: float
    dirichlet_weight: float
    embedding_norm_weight: float
    kld_weight: float
    entropy_eps: float
    dirichlet_alpha: float


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


def _auto_regularizer_config(
    num_nodes: int,
    num_edges: int,
    num_clusters: int,
    min_cluster_size: int,
) -> _AutoRegularizerConfig:
    """Derive regularizer defaults from graph statistics.

    The heuristics keep prior strengths roughly invariant for graphs where the
    cluster capacity scales with :math:`\sqrt{N}` while still adapting to
    denser graphs by increasing the regularisation weight to counteract
    over-connection driven collapse.
    """

    node_scale = max(num_nodes, 1)
    cluster_scale = max(1, min(num_clusters, node_scale))
    sqrt_nodes = max(math.sqrt(node_scale), 1.0)
    density = num_edges / float(node_scale) if node_scale else 0.0

    # Clamp the scale so extremely dense graphs do not explode the weights, but
    # sparse graphs still receive a meaningful prior.
    base_scale = cluster_scale / sqrt_nodes
    base_scale = min(3.0, max(0.3, base_scale))
    density_scale = min(2.0, max(0.5, density / 25.0 + 0.5))
    scale = base_scale * density_scale

    entropy_weight = 1e-3 * scale
    dirichlet_weight = 1e-3 * scale
    embedding_norm_weight = 1e-4 * scale
    kld_weight = 1e-3 * max(0.5, min(2.5, base_scale + density_scale / 2.0))

    entropy_eps = max(1e-12, 1e-10 / (1.0 + math.log10(node_scale + 1.0)))

    avg_nodes_per_cluster = node_scale / float(cluster_scale)
    min_cluster = max(min_cluster_size, 1)
    occupancy_target = min_cluster / max(avg_nodes_per_cluster, 1.0)
    dirichlet_alpha = min(2.0, max(0.05, occupancy_target))

    return _AutoRegularizerConfig(
        entropy_weight=entropy_weight,
        dirichlet_weight=dirichlet_weight,
        embedding_norm_weight=embedding_norm_weight,
        kld_weight=kld_weight,
        entropy_eps=entropy_eps,
        dirichlet_alpha=dirichlet_alpha,
    )


def _analyze_calibration_history(
    history: Sequence[Mapping[str, Any]],
    base_lr: float,
    current_gate_threshold: float,
    num_clusters: int,
    min_cluster_size: int,
) -> Dict[str, Any]:
    losses: List[float] = []
    active_clusters: List[float] = []
    for record in history:
        loss_val = record.get("loss")
        if isinstance(loss_val, (int, float)) and math.isfinite(loss_val):
            losses.append(float(loss_val))
        active_val = record.get("num_active_clusters")
        if isinstance(active_val, (int, float)) and math.isfinite(active_val):
            active_clusters.append(float(active_val))

    lr_multiplier = 1.0
    notes: List[str] = []
    loss_slope = 0.0
    loss_curvature = 0.0
    mean_loss = None
    if len(losses) >= 2:
        mean_loss = sum(losses) / len(losses)
        loss_slope = (losses[-1] - losses[0]) / max(1, len(losses) - 1)
        if len(losses) >= 3:
            second_diffs = [
                losses[i + 1] - 2 * losses[i] + losses[i - 1]
                for i in range(1, len(losses) - 1)
            ]
            if second_diffs:
                loss_curvature = sum(second_diffs) / len(second_diffs)

        scale_loss = max(abs(mean_loss or losses[0]), 1e-6)
        if loss_slope > 0.01 * scale_loss:
            lr_multiplier *= 0.5
            notes.append("loss increasing; halving lr")
        elif (
            abs(loss_slope) < 0.002 * scale_loss
            and abs(loss_curvature) < 0.001 * scale_loss
        ):
            lr_multiplier *= 1.1
            notes.append("loss flat; modest lr increase")

        curvature_ratio = abs(loss_curvature) / scale_loss
        if curvature_ratio > 0.05:
            lr_multiplier *= 0.8
            notes.append("high curvature; damping lr")

    lr_multiplier = min(1.5, max(0.1, lr_multiplier))
    adjusted_lr = base_lr * lr_multiplier

    gate_adjustment = 0.0
    mean_active = None
    variance_active = None
    rel_variance_active = None
    if active_clusters:
        mean_active = sum(active_clusters) / len(active_clusters)
        variance_active = sum(
            (value - mean_active) ** 2 for value in active_clusters
        ) / len(active_clusters)
        rel_variance_active = variance_active / max(mean_active**2, 1e-6)

        if variance_active > 1.0 or (
            rel_variance_active is not None and rel_variance_active > 0.05
        ):
            gate_adjustment += 0.05
            notes.append("active cluster variance high; raising gate threshold")
        elif mean_active is not None:
            conservative_capacity = max(
                min_cluster_size * 1.5, 0.3 * max(num_clusters, 1)
            )
            if (
                mean_active < conservative_capacity
                and (rel_variance_active or 0.0) < 0.01
            ):
                gate_adjustment -= 0.05
                notes.append(
                    "active cluster count low and stable; lowering gate threshold"
                )

    new_gate_threshold = min(0.95, max(0.05, current_gate_threshold + gate_adjustment))

    return {
        "lr_multiplier": lr_multiplier,
        "adjusted_lr": adjusted_lr,
        "base_lr": base_lr,
        "loss_slope": loss_slope,
        "loss_curvature": loss_curvature,
        "mean_loss": mean_loss,
        "mean_active_clusters": mean_active,
        "var_active_clusters": variance_active,
        "rel_var_active_clusters": rel_variance_active,
        "gate_threshold": new_gate_threshold,
        "notes": notes,
        "history_length": len(history),
    }


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
    existing_run_id: Optional[str] = None,
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

    run_kwargs: Dict[str, Any] = {}
    if existing_run_id:
        run_kwargs["run_id"] = existing_run_id
    elif run_name:
        run_kwargs["run_name"] = run_name
    with mlflow.start_run(**run_kwargs):
        if tags and not existing_run_id:
            mlflow.set_tags(tags)
        yield mlflow


def train_scae_on_graph(
    graph: MultiplexGraph,
    num_clusters: Optional[int] = None,
    hidden_dims: Sequence[int] = (128, 128),
    type_embedding_dim: int = 64,
    attr_encoder_dims: Tuple[int, int, int] = (32, 64, 32),
    max_epochs: int = 200,
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
    entropy_weight: Optional[float] = None,
    dirichlet_alpha: Optional[Sequence[float]] = None,
    dirichlet_weight: Optional[float] = None,
    embedding_norm_weight: Optional[float] = None,
    kld_weight: Optional[float] = None,
    entropy_eps: Optional[float] = None,
    max_negatives: Optional[int] = None,
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
    cache_node_attributes: bool = True,
    mixed_precision: bool = False,
    grad_accum_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    empty_cache_each_epoch: bool = False,
    gradient_checkpointing: bool = False,
    pos_edge_chunk_size: Optional[int] = 16384,
    neg_edge_chunk_size: Optional[int] = 4096,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 0,
    resume: bool = False,
    reset_optimizer: bool = False,
    checkpoint_blob: Optional[Mapping[str, Any]] = None,
    mlflow_run_id: Optional[str] = None,
    mlflow_last_logged_step: Optional[int] = None,
    mlflow_tracker_path: Optional[Path] = None,
    calibration_epochs: int = 0,
) -> Tuple[
    SelfCompressingRGCNAutoEncoder,
    PartitionResult,
    List[Dict[str, float]],
    Dict[str, Any],
]:
    if num_clusters is None:
        num_clusters = _default_cluster_capacity(graph)

    node_count = int(graph.data.node_types.numel())
    edge_index_tensor = getattr(graph.data, "edge_index", None)
    edge_count = int(edge_index_tensor.size(1)) if edge_index_tensor is not None else 0
    auto_regularizers = _auto_regularizer_config(
        num_nodes=node_count,
        num_edges=edge_count,
        num_clusters=num_clusters,
        min_cluster_size=min_cluster_size,
    )

    resolved_entropy_weight = (
        float(entropy_weight)
        if entropy_weight is not None
        else auto_regularizers.entropy_weight
    )
    resolved_dirichlet_weight = (
        float(dirichlet_weight)
        if dirichlet_weight is not None
        else auto_regularizers.dirichlet_weight
    )
    resolved_embedding_norm_weight = (
        float(embedding_norm_weight)
        if embedding_norm_weight is not None
        else auto_regularizers.embedding_norm_weight
    )
    resolved_kld_weight = (
        float(kld_weight) if kld_weight is not None else auto_regularizers.kld_weight
    )
    resolved_entropy_eps = (
        float(entropy_eps) if entropy_eps is not None else auto_regularizers.entropy_eps
    )

    regularizer_sources = {
        "entropy_weight": "user" if entropy_weight is not None else "auto",
        "dirichlet_weight": "user" if dirichlet_weight is not None else "auto",
        "embedding_norm_weight": "user"
        if embedding_norm_weight is not None
        else "auto",
        "kld_weight": "user" if kld_weight is not None else "auto",
        "entropy_eps": "user" if entropy_eps is not None else "auto",
    }

    provided_dirichlet_alpha_values: Optional[List[float]] = None
    if dirichlet_alpha:
        provided_dirichlet_alpha_values = [float(value) for value in dirichlet_alpha]

    if provided_dirichlet_alpha_values:
        if len(provided_dirichlet_alpha_values) == 1:
            dirichlet_param: Union[
                float, Sequence[float]
            ] = provided_dirichlet_alpha_values[0]
        else:
            dirichlet_param = provided_dirichlet_alpha_values
        dirichlet_alpha_source = "user"
    else:
        dirichlet_param = auto_regularizers.dirichlet_alpha
        dirichlet_alpha_source = "auto"

    regularizer_sources["dirichlet_alpha"] = dirichlet_alpha_source

    base_learning_rate = float(lr)
    effective_learning_rate = base_learning_rate
    base_gate_threshold = float(gate_threshold)

    checkpoint_path = (
        checkpoint_path.expanduser() if checkpoint_path is not None else None
    )
    resume_history: List[Dict[str, float]] = []
    start_epoch = 0
    used_resume = False
    if mlflow_last_logged_step is None:
        mlflow_last_logged_step = -1
    else:
        mlflow_last_logged_step = int(mlflow_last_logged_step)

    vocab_embedding_dim = attr_encoder_dims[0]
    if getattr(graph.data, "node_attributes", None):
        max_tensor_dim = 0
        for attribute_dict in graph.data.node_attributes:
            for value in attribute_dict.values():
                if isinstance(value, torch.Tensor):
                    max_tensor_dim = max(max_tensor_dim, int(value.numel()))
        if max_tensor_dim > 0:
            vocab_embedding_dim = max(vocab_embedding_dim, max_tensor_dim)

    shared_vocab = SharedAttributeVocab(
        initial_names=[], embedding_dim=vocab_embedding_dim
    )
    vocab_start = time.perf_counter()
    _populate_shared_vocab(shared_vocab, getattr(graph.data, "node_attributes", None))
    vocab_duration = time.perf_counter() - vocab_start
    if verbose:
        print(
            "[TIMING] shared vocab populated",
            f"entries={len(shared_vocab.name_to_index)}",
            f"embedding_dim={shared_vocab.embedding.embedding_dim}",
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

    model = SelfCompressingRGCNAutoEncoder(
        num_node_types=len(graph.node_type_index),
        attr_encoder=attr_encoder,
        num_clusters=num_clusters,
        num_relations=max(1, len(graph.relation_index)),
        hidden_dims=list(hidden_dims),
        type_embedding_dim=type_embedding_dim,
        negative_sampling_ratio=negative_sampling_ratio,
        entropy_weight=resolved_entropy_weight,
        dirichlet_alpha=dirichlet_param,
        dirichlet_weight=resolved_dirichlet_weight,
        embedding_norm_weight=resolved_embedding_norm_weight,
        kld_weight=resolved_kld_weight,
        entropy_eps=resolved_entropy_eps,
        max_negatives_per_graph=max_negatives,
        active_gate_threshold=gate_threshold,
        use_gradient_checkpointing=gradient_checkpointing,
        pos_edge_chunk_size=pos_edge_chunk_size,
        neg_edge_chunk_size=neg_edge_chunk_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_signature: Dict[str, Any] = {
        "model_class": type(model).__name__,
        "num_clusters": int(num_clusters),
        "hidden_dims": list(hidden_dims),
        "type_embedding_dim": int(type_embedding_dim),
        "attr_encoder_dims": list(attr_encoder_dims),
        "num_node_types": len(graph.node_type_index),
        "num_relations": max(1, len(graph.relation_index)),
        "negative_sampling_ratio": float(negative_sampling_ratio),
        "entropy_weight": float(resolved_entropy_weight),
        "dirichlet_weight": float(resolved_dirichlet_weight),
        "embedding_norm_weight": float(resolved_embedding_norm_weight),
        "kld_weight": float(resolved_kld_weight),
        "entropy_eps": float(resolved_entropy_eps),
    }

    if isinstance(dirichlet_param, Sequence) and not isinstance(
        dirichlet_param, (str, bytes)
    ):
        checkpoint_signature["dirichlet_alpha"] = [
            float(value) for value in dirichlet_param
        ]
    else:
        checkpoint_signature["dirichlet_alpha"] = float(dirichlet_param)

    def _validate_checkpoint_signature(saved: Mapping[str, Any]) -> None:
        for key, expected in checkpoint_signature.items():
            if key not in saved:
                continue
            if saved[key] != expected:
                raise ValueError(
                    f"Checkpoint signature mismatch for {key}: expected {expected}, found {saved[key]}"
                )

    if resume and checkpoint_path is not None:
        checkpoint_data: Optional[Mapping[str, Any]] = checkpoint_blob
        if checkpoint_data is None and checkpoint_path.exists():
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - surface load errors
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_path}: {exc}"
                ) from exc

        if checkpoint_data is not None:
            saved_signature = checkpoint_data.get("signature")
            if isinstance(saved_signature, Mapping):
                _validate_checkpoint_signature(saved_signature)
            elif saved_signature is not None:
                raise ValueError(
                    "Checkpoint signature is malformed; expected a mapping."
                )

            model_state = checkpoint_data.get("model_state")
            if model_state is None:
                raise ValueError("Checkpoint missing model_state for resume.")
            model.load_state_dict(model_state)

            if not reset_optimizer:
                optimizer_state = checkpoint_data.get("optimizer_state")
                if optimizer_state is not None:
                    optimizer.load_state_dict(optimizer_state)

            _restore_rng_state(checkpoint_data.get("rng_state"))

            history_blob = checkpoint_data.get("history", [])
            if isinstance(history_blob, list):
                resume_history = [
                    dict(entry) for entry in history_blob if isinstance(entry, Mapping)
                ]
            else:
                resume_history = []

            epoch_value = checkpoint_data.get("epoch")
            if epoch_value is None:
                start_epoch = len(resume_history)
            else:
                try:
                    start_epoch = int(epoch_value)
                except (TypeError, ValueError):
                    start_epoch = len(resume_history)
            start_epoch = max(start_epoch, len(resume_history))
            prior_summary = checkpoint_data.get("run_summary")
            if isinstance(prior_summary, Mapping):
                if mlflow_run_id is None:
                    prior_run_id = prior_summary.get("mlflow_run_id")
                    if isinstance(prior_run_id, str) and prior_run_id:
                        mlflow_run_id = prior_run_id
                if mlflow_last_logged_step <= 0:
                    prior_logged_step = prior_summary.get("mlflow_last_logged_step")
                    if isinstance(prior_logged_step, (int, float)):
                        mlflow_last_logged_step = max(
                            mlflow_last_logged_step, int(prior_logged_step)
                        )
            used_resume = True
            if verbose:
                print(
                    f"[INFO] Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}"
                )
        elif checkpoint_path.exists():
            if verbose:
                print(
                    f"[WARN] Resume requested but checkpoint {checkpoint_path} could not be loaded. Starting fresh."
                )
        else:
            if verbose:
                print(
                    f"[WARN] Resume requested but checkpoint {checkpoint_path} was not found. Starting fresh."
                )

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

    trainer = OnlineTrainer(
        model,
        optimizer,
        device=device,
        log_timing=verbose,
        cache_node_attributes=cache_node_attributes,
        mixed_precision=mixed_precision,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        empty_cache_each_epoch=empty_cache_each_epoch,
    )

    if resume_history:
        trainer.history = [dict(entry) for entry in resume_history]
        trainer.total_epochs_trained = start_epoch
        mlflow_last_logged_step = max(mlflow_last_logged_step, start_epoch)

    add_start = time.perf_counter()
    trainer.add_data(dataset)
    add_duration = time.perf_counter() - add_start
    if verbose:
        print(f"[TIMING] trainer.add_data took {add_duration:.3f}s")

    initial_history_len = len(trainer.history)
    checkpoint_interval = max(0, int(checkpoint_every))

    def _update_mlflow_tracker(step: int) -> None:
        if mlflow_tracker_path is None or mlflow_run_id is None:
            return
        tracker_payload = {
            "mlflow_run_id": mlflow_run_id,
            "last_logged_step": int(step),
        }
        try:
            mlflow_tracker_path.parent.mkdir(parents=True, exist_ok=True)
            if mlflow_tracker_path.suffix:
                tmp_tracker = mlflow_tracker_path.with_suffix(
                    mlflow_tracker_path.suffix + ".tmp"
                )
            else:
                tmp_tracker = mlflow_tracker_path.with_name(
                    mlflow_tracker_path.name + ".tmp"
                )
            tmp_tracker.write_text(
                json.dumps(tracker_payload, indent=2), encoding="utf-8"
            )
            tmp_tracker.replace(mlflow_tracker_path)
        except Exception:
            pass

    def _write_checkpoint(epoch_idx: int, reason: str) -> None:
        if checkpoint_path is None:
            return
        history_snapshot = [dict(entry) for entry in trainer.history]
        payload = {
            "version": 1,
            "epoch": int(epoch_idx),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
            "history": history_snapshot,
            "signature": checkpoint_signature,
            "run_summary": {
                "resume_used": used_resume,
                "start_epoch": start_epoch,
                "epochs_trained": trainer.last_run_epochs,
                "total_epochs": trainer.total_epochs_trained,
                "reason": reason,
                "timestamp": time.time(),
                "mlflow_run_id": mlflow_run_id,
                "mlflow_last_logged_step": mlflow_last_logged_step,
            },
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.suffix:
            tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        else:
            tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint_path)
        if verbose:
            print(
                f"[INFO] Saved checkpoint to {checkpoint_path} at epoch {epoch_idx} ({reason})"
            )

    def _combined_epoch_callback(epoch_idx: int, metrics: Dict[str, float]) -> None:
        nonlocal mlflow_last_logged_step
        if epoch_callback is not None and epoch_idx > mlflow_last_logged_step:
            epoch_callback(epoch_idx, metrics)
            mlflow_last_logged_step = epoch_idx
            _update_mlflow_tracker(mlflow_last_logged_step)
        if (
            checkpoint_path is not None
            and checkpoint_interval > 0
            and epoch_idx % checkpoint_interval == 0
        ):
            _write_checkpoint(epoch_idx, "interval")

    if epoch_callback is not None or checkpoint_path is not None:
        epoch_callback_for_trainer = _combined_epoch_callback
    else:
        epoch_callback_for_trainer = None

    total_epochs_available = max(0, int(max_epochs) - int(start_epoch))
    warmup_epochs = min(max(0, int(calibration_epochs)), total_epochs_available)
    calibration_summary: Optional[Dict[str, Any]] = None

    if warmup_epochs > 0 and not used_resume:
        if verbose:
            print(f"[INFO] Starting calibration warm-up for {warmup_epochs} epoch(s)")
        history_start = len(trainer.history)
        trainer.train(
            max_epochs=warmup_epochs,
            batch_size=max(1, int(batch_size)),
            negative_sampling_ratio=negative_sampling_ratio,
            verbose=verbose,
            bucket_by_size=node_budget is None,
            node_budget=node_budget,
            on_epoch_end=epoch_callback_for_trainer,
            stability_metric=None,
            stability_window=0,
            stability_tolerance=0.0,
            stability_relative_tolerance=None,
            min_epochs=0,
            start_epoch=start_epoch,
            realized_cluster_min_size=min_cluster_size,
        )
        warmup_history = trainer.history[history_start:]
        calibration_summary = _analyze_calibration_history(
            warmup_history,
            base_lr=lr,
            current_gate_threshold=gate_threshold,
            num_clusters=num_clusters,
            min_cluster_size=min_cluster_size,
        )

        adjusted_lr = float(calibration_summary["adjusted_lr"])
        if not math.isclose(adjusted_lr, effective_learning_rate, rel_tol=1e-6):
            for param_group in optimizer.param_groups:
                param_group["lr"] = adjusted_lr
            optimizer.defaults["lr"] = adjusted_lr
            if verbose:
                print(
                    f"[INFO] Calibration adjusted learning rate from {effective_learning_rate:.4g} to {adjusted_lr:.4g}"
                )
        effective_learning_rate = adjusted_lr

        new_gate_threshold = float(calibration_summary["gate_threshold"])
        if not math.isclose(new_gate_threshold, gate_threshold, rel_tol=1e-6):
            gate_threshold = new_gate_threshold
            model.active_gate_threshold = new_gate_threshold
            if verbose:
                print(
                    "[INFO] Calibration updated gate threshold to "
                    f"{new_gate_threshold:.3f}"
                )

        calibration_summary["warmup_epochs"] = trainer.last_run_epochs
        calibration_summary["base_gate_threshold"] = base_gate_threshold
        calibration_summary["adjusted_gate_threshold"] = gate_threshold
        calibration_summary["base_lr"] = base_learning_rate
        calibration_summary["effective_lr"] = effective_learning_rate
        calibration_summary["notes"] = calibration_summary.get("notes", [])

        start_epoch = trainer.total_epochs_trained
        total_epochs_available = max(0, int(max_epochs) - int(start_epoch))
    elif warmup_epochs > 0 and used_resume:
        calibration_summary = {
            "lr_multiplier": 1.0,
            "adjusted_lr": effective_learning_rate,
            "base_lr": base_learning_rate,
            "loss_slope": 0.0,
            "loss_curvature": 0.0,
            "mean_loss": None,
            "mean_active_clusters": None,
            "var_active_clusters": None,
            "rel_var_active_clusters": None,
            "gate_threshold": gate_threshold,
            "base_gate_threshold": base_gate_threshold,
            "adjusted_gate_threshold": gate_threshold,
            "notes": ["calibration skipped because training resumed from checkpoint"],
            "history_length": 0,
            "warmup_epochs": 0,
            "effective_lr": effective_learning_rate,
        }

    if total_epochs_available > 0:
        remaining_min_epochs = max(0, int(min_epochs) - int(start_epoch))
        history = trainer.train(
            max_epochs=total_epochs_available,
            batch_size=max(1, int(batch_size)),
            negative_sampling_ratio=negative_sampling_ratio,
            verbose=verbose,
            bucket_by_size=node_budget is None,
            node_budget=node_budget,
            on_epoch_end=epoch_callback_for_trainer,
            stability_metric="realized_active_clusters"
            if cluster_stability_window > 0
            else None,
            stability_window=cluster_stability_window,
            stability_tolerance=cluster_stability_tolerance,
            stability_relative_tolerance=cluster_stability_relative_tolerance,
            min_epochs=remaining_min_epochs,
            start_epoch=start_epoch,
            realized_cluster_min_size=min_cluster_size,
        )
    else:
        if verbose:
            print(
                "[INFO] No epochs remaining to train "
                f"(start_epoch={start_epoch}, max_epochs={max_epochs})."
            )
        trainer.last_run_epochs = 0
        trainer.total_epochs_trained = start_epoch
        history = trainer.history

    if trainer.early_stop_epoch is not None:
        message = f"Early stop at epoch {trainer.early_stop_epoch}: {trainer.early_stop_reason}"
        if verbose:
            print(f"[INFO] {message}")
        if history:
            history[-1]["early_stop_epoch"] = float(trainer.early_stop_epoch)

    epochs_trained_this_run = max(0, trainer.last_run_epochs)
    if history:
        history[-1]["epochs_trained_this_run"] = float(epochs_trained_this_run)
        history[-1]["total_epochs_trained"] = float(trainer.total_epochs_trained)
        history[-1]["resume_start_epoch"] = float(start_epoch)
        history[-1]["resumed"] = float(1 if used_resume else 0)

    if checkpoint_path is not None:
        _write_checkpoint(trainer.total_epochs_trained, "final")

    if mlflow_last_logged_step >= 0:
        _update_mlflow_tracker(mlflow_last_logged_step)

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

    resolved_regularizers: Dict[str, Any] = {
        "entropy_weight": float(resolved_entropy_weight),
        "dirichlet_weight": float(resolved_dirichlet_weight),
        "embedding_norm_weight": float(resolved_embedding_norm_weight),
        "kld_weight": float(resolved_kld_weight),
        "entropy_eps": float(resolved_entropy_eps),
    }
    if isinstance(dirichlet_param, Sequence) and not isinstance(
        dirichlet_param, (str, bytes)
    ):
        resolved_regularizers["dirichlet_alpha"] = [
            float(value) for value in dirichlet_param
        ]
    else:
        resolved_regularizers["dirichlet_alpha"] = float(dirichlet_param)

    auto_regularizer_values: Dict[str, float] = {
        "entropy_weight": float(auto_regularizers.entropy_weight),
        "dirichlet_weight": float(auto_regularizers.dirichlet_weight),
        "embedding_norm_weight": float(auto_regularizers.embedding_norm_weight),
        "kld_weight": float(auto_regularizers.kld_weight),
        "entropy_eps": float(auto_regularizers.entropy_eps),
        "dirichlet_alpha": float(auto_regularizers.dirichlet_alpha),
    }

    training_summary = {
        "epochs_trained": epochs_trained_this_run,
        "total_epochs": trainer.total_epochs_trained,
        "resume_used": used_resume,
        "start_epoch": start_epoch,
        "checkpoint_path": str(checkpoint_path)
        if checkpoint_path is not None
        else None,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_last_logged_step": mlflow_last_logged_step,
        "regularizer_config": {
            "values": resolved_regularizers,
            "auto": auto_regularizer_values,
            "sources": regularizer_sources,
            "graph_stats": {
                "num_nodes": node_count,
                "num_edges": edge_count,
                "num_clusters": num_clusters,
            },
        },
        "calibration_summary": calibration_summary,
        "effective_learning_rate": effective_learning_rate,
        "gate_threshold": gate_threshold,
    }

    return model, partition, history, training_summary


def save_model_artifact(
    model: torch.nn.Module,
    output_path: Path,
    training_summary: Mapping[str, Any],
) -> Path:
    state_dict = model.state_dict()
    state_dict_type = state_dict.__class__
    cpu_state = state_dict_type(
        (
            key,
            value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value),
        )
        for key, value in state_dict.items()
    )

    payload = {
        "version": 1,
        "model_class": type(model).__name__,
        "state_dict": cpu_state,
        "training_summary": copy.deepcopy(training_summary),
        "saved_at": time.time(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    else:
        tmp_path = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    return output_path


def save_partition(
    partition: PartitionResult, output_path: Path, graph: MultiplexGraph
) -> None:
    idx_to_id = graph.node_ids
    idx_to_name = graph.node_names

    cluster_members_named: Dict[str, List[str]] = {}
    cluster_member_ids: Dict[str, List[str]] = {}
    node_to_cluster: Dict[str, int] = {}
    name_collisions: Dict[str, List[str]] = defaultdict(list)

    node_assignments = partition.node_to_cluster.tolist()

    for node_index, cluster_idx in enumerate(node_assignments):
        node_id = idx_to_id[node_index]
        node_name = idx_to_name[node_index]
        int_cluster = int(cluster_idx)

        node_to_cluster[node_id] = int_cluster
        if node_name:
            name_collisions[node_name].append(node_id)

    for cluster_idx, member_idx in partition.cluster_members.items():
        key = str(cluster_idx)
        member_ids = [idx_to_id[i] for i in member_idx.tolist()]
        cluster_member_ids[key] = member_ids
        cluster_members_named[key] = [idx_to_name[i] for i in member_idx.tolist()]

    duplicated_names = {
        name: ids for name, ids in name_collisions.items() if len(ids) > 1
    }
    if duplicated_names:
        preview = ", ".join(
            f"{name}->{len(ids)} ids"
            for name, ids in list(duplicated_names.items())[:5]
        )
        warnings.warn(
            "Detected non-unique node labels while saving the partition; "
            "downstream tools should rely on node IDs. "
            f"Colliding labels: {len(duplicated_names)} (samples: {preview})."
        )

    payload = {
        "active_clusters": partition.active_clusters.tolist(),
        "gate_values": partition.gate_values.tolist(),
        "node_to_cluster": node_to_cluster,
        "node_name_map": {idx_to_id[i]: idx_to_name[i] for i in range(len(idx_to_id))},
        "cluster_members": cluster_members_named,
        "cluster_member_ids": cluster_member_ids,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_cluster_npz(
    model,
    graph,
    npz_path,
    device="cpu",
    metadata: Optional[Mapping[str, Any]] = None,
):
    model.eval()
    with torch.no_grad():
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        data = graph.data
        device_obj = torch.device(device)
        node_types = data.node_types.to(device_obj)
        edge_index = data.edge_index.to(device_obj)
        edge_type = getattr(data, "edge_type", None)
        if isinstance(edge_type, torch.Tensor):
            edge_type = edge_type.to(device_obj)
        positional_encodings = getattr(data, "positional_encodings", None)
        if isinstance(positional_encodings, torch.Tensor):
            positional_encodings = positional_encodings.to(device_obj)
        edge_weight = getattr(data, "edge_weight", None)
        if isinstance(edge_weight, torch.Tensor):
            edge_weight = edge_weight.to(device_obj)
        batch_vec = getattr(data, "batch", None)
        if isinstance(batch_vec, torch.Tensor):
            batch_vec = batch_vec.to(device_obj)
        node_attr_embedding = getattr(data, "node_attr_embedding", None)
        if isinstance(node_attr_embedding, torch.Tensor):
            node_attr_embedding = node_attr_embedding.to(device_obj)

        node_attributes = getattr(data, "node_attributes", None)
        if not node_attributes:
            node_attributes = getattr(graph, "node_attributes", None)

        H, _ = model.encoder(
            node_types=node_types,
            edge_index=edge_index,
            batch=batch_vec,
            node_attributes=node_attributes,
            precomputed_attr=node_attr_embedding,
            edge_type=edge_type,
            positional_encodings=positional_encodings,
            edge_weight=edge_weight,
        )

        prototypes = getattr(model, "prototype_bank", None)
        if prototypes is not None:
            logits = F.normalize(H, dim=-1) @ F.normalize(prototypes, dim=-1).t()
        else:
            logits = H.new_zeros(H.size(0), getattr(model, "num_clusters", 0))

        gates = getattr(model, "cluster_gate", None)
        gate_values = None
        if gates is not None:
            gate_values = gates(training=False).detach().cpu().numpy()

        recon_loss = getattr(model, "last_recon_loss", None)
        node_ids = getattr(graph, "node_ids", None)
        if node_ids is None:
            node_ids = getattr(data, "node_ids", None)
        if isinstance(node_ids, torch.Tensor):
            node_ids_payload = node_ids.detach().cpu().numpy()
        elif isinstance(node_ids, (list, tuple)):
            node_ids_payload = np.asarray(node_ids)
        else:
            node_ids_payload = np.arange(H.shape[0])

        payload: Dict[str, Any] = {
            "Z": logits.detach().cpu().numpy(),
            "gates": gate_values,
            "H": H.detach().cpu().numpy(),
            "recon_loss": float(recon_loss) if recon_loss is not None else None,
            "node_ids": node_ids_payload,
        }
        if metadata:
            payload["metadata_json"] = np.array(
                json.dumps(metadata, default=str), dtype=object
            )
        np.savez_compressed(npz_path, **payload)
        print(f"[NPZ EXPORT] Saved {npz_path}")


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
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path for the trained model weights (defaults to <out> with a .pt suffix)",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=50,
        help="Minimum number of epochs before the stability criterion can stop training (must be <= --max-epochs).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Override the auto-selected latent cluster capacity (default heuristically scales with graph size).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--calibration-epochs",
        type=int,
        default=10,
        help="Warm-up epochs run before main training to auto-adjust learning rate and gate threshold (0 disables)",
    )
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
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable torch.cuda.amp mixed precision (CUDA only) in order to fit larger graphs.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Wrap rGCN encoder layers with torch.utils.checkpoint to shrink activation memory (extra compute).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path for saving and resuming training checkpoints (.pt).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N epochs (0 saves only after training finishes).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume model and optimizer state from --checkpoint-path when it exists.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, discard the saved optimizer state and start fresh.",
    )
    parser.add_argument(
        "--pos-edge-chunk",
        type=int,
        default=16384,
        help="Maximum positive edges decoded per chunk (<=0 disables chunking).",
    )
    parser.add_argument(
        "--neg-edge-chunk",
        type=int,
        default=16384,
        help="Maximum negative edges decoded per chunk (<=0 disables chunking).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Accumulate this many mini-batches before each optimizer step. This lowers peak activation/gradient memory per forward/backward pass. Make sure to also use --max-grad-norm as a stability guard.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Clip gradients to this norm before every optimizer step (disabled by default).",
    )
    parser.add_argument(
        "--no-attr-cache",
        action="store_true",
        help="Skip caching node attribute embeddings to save host memory.",
    )
    parser.add_argument(
        "--cuda-empty-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each epoch (CUDA only).",
    )
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--negative-sampling", type=float, default=1.0)
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=0,
        help="Maximum negatives per graph (0 lets ratio decide)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--text-encoder-model",
        type=str,
        default="pritamdeka/S-Scibert-snli-multinli-stsb",
        help="SentenceTransformers model used to embed node text labels ('none' disables).",
    )
    parser.add_argument(
        "--text-encoder-device",
        type=str,
        default=None,
        help="Device for the text encoder (defaults to CUDA if available).",
    )
    parser.add_argument(
        "--text-encoder-batch-size",
        type=int,
        default=128,
        help="Batch size for node text embedding inference.",
    )
    parser.add_argument(
        "--text-encoder-normalize",
        action="store_true",
        help="L2-normalize text embeddings returned by the SentenceTransformer.",
    )
    parser.add_argument(
        "--text-encoder-projection-dim",
        type=int,
        default=128,
        help="Random projection dimension applied to text embeddings (<=0 keeps original size)",
    )
    parser.add_argument(
        "--text-embedding-cache",
        type=str,
        default="auto",
        help=(
            "Path to cache node text embeddings ("
            "'auto' derives from the GraphML path; 'none' disables caching)."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=None,
        help="Weight applied to the entropy regularizer (auto-derived from graph statistics when omitted)",
    )
    parser.add_argument(
        "--cluster-stability-window",
        type=int,
        default=20,
        help="Number of trailing epochs used to test realized_active_clusters stability (post-argmax) for early stopping (0 disables).",
    )
    parser.add_argument(
        "--cluster-stability-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for realized_active_clusters variation across the stability window.",
    )
    parser.add_argument(
        "--cluster-stability-rel-tol",
        type=float,
        default=0.0,
        help="Relative tolerance (fraction of the mean) allowed for realized_active_clusters across the window.",
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
        default=None,
        help="Weight for the Dirichlet prior regularizer (auto-derived from graph statistics when omitted)",
    )
    parser.add_argument(
        "--embedding-norm-weight",
        type=float,
        default=None,
        help="Weight for the embedding norm regularizer (auto-derived from graph statistics when omitted)",
    )
    parser.add_argument(
        "--kld-weight",
        type=float,
        default=None,
        help="Weight for the KL divergence regularizer (auto-derived from graph statistics when omitted)",
    )
    parser.add_argument(
        "--entropy-eps",
        type=float,
        default=None,
        help="Numerical floor for entropy/Dirichlet calculations (auto-derived from graph statistics when omitted)",
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
    parser.add_argument(
        "--npz-out",
        type=Path,
        default=None,
        help="Path to export cluster logits, gates, embeddings, and recon loss as NPZ (defaults to <out>.npz).",
    )
    parser.add_argument(
        "--no-npz-export",
        action="store_true",
        help="Skip NPZ export entirely (otherwise defaults to <out>.npz when --npz-out is omitted).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    mlflow_auto_reasons: List[str] = []
    mlflow_tracking_uri_default = parser.get_default("mlflow_tracking_uri")
    if args.mlflow_experiment:
        mlflow_auto_reasons.append("--mlflow-experiment")
    if args.mlflow_run_name:
        mlflow_auto_reasons.append("--mlflow-run-name")
    if args.mlflow_tags:
        mlflow_auto_reasons.append("--mlflow-tag")
    if (
        args.mlflow_tracking_uri
        and args.mlflow_tracking_uri != mlflow_tracking_uri_default
    ):
        mlflow_auto_reasons.append("--mlflow-tracking-uri")

    if args.min_epochs > args.max_epochs:
        parser.error("--min-epochs must be less than or equal to --max-epochs")
    if args.cluster_stability_window < 0:
        parser.error("--cluster-stability-window must be non-negative")
    if (
        args.cluster_stability_window > 0
        and args.cluster_stability_window > args.max_epochs
    ):
        parser.error("--cluster-stability-window cannot exceed --max-epochs")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be at least 1")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.node_budget is not None and args.node_budget < 0:
        parser.error("--node-budget must be non-negative")
    if args.node_budget == 0:
        args.node_budget = None
    if args.max_negatives is not None and args.max_negatives < 0:
        parser.error("--max-negatives must be non-negative")
    if args.max_negatives == 0:
        args.max_negatives = None

    if args.model_out is None:
        args.model_out = args.out.with_suffix(".pt")

    if args.no_npz_export:
        args.npz_out = None
    elif args.npz_out is None:
        default_npz = _default_npz_export_path(args.out)
        args.npz_out = default_npz
        if not args.quiet:
            print(f"[INFO] NPZ export path not provided; defaulting to {default_npz}")
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
    if args.pos_edge_chunk is not None and args.pos_edge_chunk < 0:
        parser.error("--pos-edge-chunk must be non-negative")
    if args.neg_edge_chunk is not None and args.neg_edge_chunk < 0:
        parser.error("--neg-edge-chunk must be non-negative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.resume_from_checkpoint and args.checkpoint_path is None:
        parser.error("--resume-from-checkpoint requires --checkpoint-path")

    if args.stability and args.ego_samples == 0:
        args.ego_samples = 256

    pos_edge_chunk = args.pos_edge_chunk if args.pos_edge_chunk > 0 else None
    neg_edge_chunk = args.neg_edge_chunk if args.neg_edge_chunk > 0 else None

    text_encoder_model = args.text_encoder_model
    if text_encoder_model is not None:
        normalized_name = text_encoder_model.strip().lower()
        if normalized_name in {"", "none", "disable", "disabled", "off"}:
            text_encoder_model = None

    text_encoder_projection_dim = args.text_encoder_projection_dim
    if text_encoder_projection_dim is not None and text_encoder_projection_dim <= 0:
        text_encoder_projection_dim = None

    text_embedding_cache_arg = args.text_embedding_cache
    enable_text_embedding_cache = True
    text_embedding_cache_path: Optional[Path] = None
    if text_embedding_cache_arg is not None:
        cache_arg_stripped = text_embedding_cache_arg.strip()
        cache_arg_lower = cache_arg_stripped.lower()
        if cache_arg_lower in {"", "auto"}:
            text_embedding_cache_path = None
        elif cache_arg_lower in {"none", "disable", "disabled", "off"}:
            enable_text_embedding_cache = False
            text_embedding_cache_path = None
        else:
            text_embedding_cache_path = Path(text_embedding_cache_arg).expanduser()

    if not enable_text_embedding_cache:
        text_embedding_cache_param_value = "disabled"
    elif text_embedding_cache_path is not None:
        text_embedding_cache_param_value = str(text_embedding_cache_path)
    else:
        text_embedding_cache_param_value = "auto"

    graph = load_multiplex_graph(
        args.graphml,
        text_encoder_model=text_encoder_model,
        text_encoder_device=args.text_encoder_device,
        text_encoder_batch_size=args.text_encoder_batch_size,
        text_encoder_normalize=args.text_encoder_normalize,
        text_embedding_cache=text_embedding_cache_path,
        enable_text_embedding_cache=enable_text_embedding_cache,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    num_nodes_after = int(graph.data.node_types.numel())
    num_edges_after = int(graph.data.edge_index.size(1))
    print(
        f"Graph after nosology filter: {num_nodes_after} nodes, {num_edges_after} edges"
    )
    effective_num_clusters = (
        args.clusters if args.clusters is not None else _default_cluster_capacity(graph)
    )

    checkpoint_blob: Optional[Mapping[str, Any]] = None
    resume_mlflow_run_id: Optional[str] = None
    mlflow_last_logged_step: Optional[int] = None
    mlflow_tracker_path: Optional[Path] = None
    if args.checkpoint_path is not None:
        mlflow_tracker_path = _checkpoint_tracker_path(args.checkpoint_path)
        if args.resume_from_checkpoint and mlflow_tracker_path.exists():
            try:
                tracker_blob = json.loads(
                    mlflow_tracker_path.read_text(encoding="utf-8")
                )
            except Exception:
                tracker_blob = None
            if isinstance(tracker_blob, Mapping):
                tracker_run = tracker_blob.get("mlflow_run_id")
                if isinstance(tracker_run, str) and tracker_run:
                    resume_mlflow_run_id = tracker_run
                tracker_step = tracker_blob.get("last_logged_step")
                if isinstance(tracker_step, (int, float)):
                    mlflow_last_logged_step = int(tracker_step)

    if (
        args.resume_from_checkpoint
        and args.checkpoint_path is not None
        and args.checkpoint_path.exists()
    ):
        try:
            checkpoint_blob = torch.load(args.checkpoint_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - surface load errors
            print(
                f"[WARN] Failed to preload checkpoint {args.checkpoint_path}: {exc}. "
                "Proceeding without prior state."
            )
            checkpoint_blob = None
        if checkpoint_blob is not None:
            run_summary = checkpoint_blob.get("run_summary")
            if isinstance(run_summary, Mapping):
                candidate_run_id = run_summary.get("mlflow_run_id")
                if (
                    resume_mlflow_run_id is None
                    and isinstance(candidate_run_id, str)
                    and candidate_run_id
                ):
                    resume_mlflow_run_id = candidate_run_id
                candidate_logged_step = run_summary.get("mlflow_last_logged_step")
                if mlflow_last_logged_step is None and isinstance(
                    candidate_logged_step, (int, float)
                ):
                    mlflow_last_logged_step = int(candidate_logged_step)

    if resume_mlflow_run_id is not None:
        mlflow_auto_reasons.append("resume-from-checkpoint")

    if not args.mlflow and mlflow_auto_reasons:
        args.mlflow = True
        if not args.quiet:
            reason_text = ", ".join(dict.fromkeys(mlflow_auto_reasons))
            print(
                "[INFO] Auto-enabled MLflow tracking because "
                f"{reason_text} was provided. Omit those flags to skip MLflow."
            )

    tags = _parse_mlflow_tags(args.mlflow_tags)
    with _mlflow_run(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        tags=tags,
        existing_run_id=resume_mlflow_run_id,
    ) as mlflow_client:
        active_run_id: Optional[str] = None
        if mlflow_client is not None:
            active = mlflow_client.active_run()
            if active is not None:
                active_run_id = active.info.run_id
        if mlflow_client is not None:
            hidden_dims_param = ",".join(map(str, args.hidden)) if args.hidden else ""
            dirichlet_alpha_param = (
                ",".join(map(str, args.dirichlet_alpha)) if args.dirichlet_alpha else ""
            )
            mlflow_client.log_params(
                {
                    "graph_path": str(args.graphml),
                    "output_path": str(args.out),
                    "model_output_path": str(args.model_out),
                    "max_epochs": args.max_epochs,
                    "min_epochs": args.min_epochs,
                    "device": args.device or "auto",
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "node_budget": args.node_budget
                    if args.node_budget is not None
                    else "",
                    "negative_sampling_ratio": args.negative_sampling,
                    "max_negatives": args.max_negatives
                    if args.max_negatives is not None
                    else "",
                    "gate_threshold": args.gate_threshold,
                    "min_cluster_size": args.min_cluster_size,
                    "entropy_weight_arg": (
                        args.entropy_weight
                        if args.entropy_weight is not None
                        else "auto"
                    ),
                    "mixed_precision": args.mixed_precision,
                    "grad_accum_steps": args.grad_accum,
                    "max_grad_norm": args.max_grad_norm
                    if args.max_grad_norm is not None
                    else "",
                    "cache_node_attributes": not args.no_attr_cache,
                    "cuda_empty_cache": args.cuda_empty_cache,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "pos_edge_chunk": pos_edge_chunk
                    if pos_edge_chunk is not None
                    else "",
                    "neg_edge_chunk": neg_edge_chunk
                    if neg_edge_chunk is not None
                    else "",
                    "cluster_stability_window": args.cluster_stability_window,
                    "cluster_stability_tol": args.cluster_stability_tol,
                    "cluster_stability_rel_tol": args.cluster_stability_rel_tol,
                    "dirichlet_weight_arg": (
                        args.dirichlet_weight
                        if args.dirichlet_weight is not None
                        else "auto"
                    ),
                    "embedding_norm_weight_arg": (
                        args.embedding_norm_weight
                        if args.embedding_norm_weight is not None
                        else "auto"
                    ),
                    "kld_weight_arg": (
                        args.kld_weight if args.kld_weight is not None else "auto"
                    ),
                    "entropy_eps_arg": (
                        args.entropy_eps if args.entropy_eps is not None else "auto"
                    ),
                    "ego_samples": args.ego_samples,
                    "ego_alpha": args.ego_alpha,
                    "ego_min_radius": args.ego_min_radius,
                    "ego_max_radius": args.ego_max_radius,
                    "ego_min_nodes": args.ego_min_nodes,
                    "ego_min_edges": args.ego_min_edges,
                    "hidden_dims": hidden_dims_param,
                    "effective_num_clusters": effective_num_clusters,
                    "dirichlet_alpha_arg": dirichlet_alpha_param or "auto",
                    "text_encoder_model": text_encoder_model or "none",
                    "text_encoder_device": args.text_encoder_device
                    if args.text_encoder_device is not None
                    else "auto",
                    "text_encoder_batch_size": args.text_encoder_batch_size,
                    "text_encoder_normalize": args.text_encoder_normalize,
                    "text_encoder_projection_dim": text_encoder_projection_dim
                    if text_encoder_projection_dim is not None
                    else 0,
                    "text_embedding_cache": text_embedding_cache_param_value,
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

        model, partition, history, training_summary = train_scae_on_graph(
            graph,
            num_clusters=effective_num_clusters,
            hidden_dims=args.hidden,
            max_epochs=args.max_epochs,
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
            max_negatives=args.max_negatives,
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
            cache_node_attributes=not args.no_attr_cache,
            mixed_precision=args.mixed_precision,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            empty_cache_each_epoch=args.cuda_empty_cache,
            gradient_checkpointing=args.gradient_checkpointing,
            pos_edge_chunk_size=pos_edge_chunk,
            neg_edge_chunk_size=neg_edge_chunk,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume_from_checkpoint,
            reset_optimizer=args.reset_optimizer,
            checkpoint_blob=checkpoint_blob,
            mlflow_run_id=active_run_id,
            mlflow_last_logged_step=mlflow_last_logged_step,
            mlflow_tracker_path=mlflow_tracker_path,
            calibration_epochs=args.calibration_epochs,
        )

        training_summary["model_path"] = str(args.model_out)
        model_artifact_path = save_model_artifact(
            model, args.model_out, training_summary
        )

        epochs_completed_this_run = int(
            training_summary.get("epochs_trained", len(history))
        )
        total_epochs_completed = int(training_summary.get("total_epochs", len(history)))

        if mlflow_client is not None:
            regularizer_config = training_summary.get("regularizer_config", {})
            regularizer_values = regularizer_config.get("values", {})
            regularizer_sources = regularizer_config.get("sources", {})
            auto_regularizers = regularizer_config.get("auto", {})

            resolved_param_payload: Dict[str, Any] = {}
            for key, value in regularizer_values.items():
                if isinstance(value, list):
                    resolved_param_payload[f"{key}"] = ",".join(map(str, value))
                else:
                    resolved_param_payload[f"{key}"] = float(value)
            for key, source in regularizer_sources.items():
                resolved_param_payload[f"{key}_source"] = source
            for key, value in auto_regularizers.items():
                resolved_param_payload[f"{key}_auto"] = float(value)

            resolved_param_payload["learning_rate_resolved"] = float(
                training_summary.get("effective_learning_rate", args.lr)
            )
            resolved_param_payload["gate_threshold_resolved"] = float(
                training_summary.get("gate_threshold", args.gate_threshold)
            )

            calibration_details = training_summary.get("calibration_summary") or {}
            if calibration_details:
                resolved_param_payload["calibration_lr_multiplier"] = float(
                    calibration_details.get("lr_multiplier", 1.0)
                )
                resolved_param_payload["calibration_warmup_epochs"] = int(
                    calibration_details.get("warmup_epochs", 0)
                )
                resolved_param_payload["calibration_gate_threshold"] = float(
                    calibration_details.get(
                        "adjusted_gate_threshold",
                        training_summary.get("gate_threshold", args.gate_threshold),
                    )
                )
                notes = calibration_details.get("notes", [])
                if notes:
                    resolved_param_payload["calibration_notes"] = " | ".join(notes)

            if resolved_param_payload:
                mlflow_client.log_params(resolved_param_payload)

            if calibration_details:
                calibration_metrics: Dict[str, float] = {}
                for metric_key in (
                    "loss_slope",
                    "loss_curvature",
                    "mean_active_clusters",
                    "var_active_clusters",
                    "rel_var_active_clusters",
                ):
                    metric_value = calibration_details.get(metric_key)
                    if isinstance(metric_value, (int, float)) and math.isfinite(
                        metric_value
                    ):
                        calibration_metrics[f"calibration_{metric_key}"] = float(
                            metric_value
                        )
                if calibration_metrics:
                    mlflow_client.log_metrics(
                        calibration_metrics,
                        step=int(training_summary.get("start_epoch", 0)),
                    )

            if history:
                final_loss = history[-1].get("loss")
                if final_loss is not None and math.isfinite(final_loss):
                    mlflow_client.log_metric("final_loss", float(final_loss))
                if "early_stop_epoch" in history[-1]:
                    mlflow_client.log_metric(
                        "early_stop_epoch", history[-1]["early_stop_epoch"]
                    )
            mlflow_client.log_metric(
                "epochs_completed", float(epochs_completed_this_run)
            )
            mlflow_client.log_metric("epochs_total", float(total_epochs_completed))
            mlflow_client.log_metric(
                "num_active_clusters",
                len(partition.active_clusters),
                step=total_epochs_completed + 1,
            )
            mlflow_client.log_metric(
                "start_epoch_resume",
                float(training_summary.get("start_epoch", 0)),
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

            resume_flag = "true" if training_summary.get("resume_used") else "false"
            mlflow_client.log_param("resume_from_checkpoint", resume_flag)
            if training_summary.get("checkpoint_path"):
                mlflow_client.log_param(  # record resolved path
                    "checkpoint_path", training_summary["checkpoint_path"]
                )

        print(f"[INFO] Saved trained model weights to {args.model_out}")
        save_partition(partition, args.out, graph)
        npz_metadata = {
            "graph_path": str(args.graphml),
            "partition_path": str(args.out),
            "model_path": str(args.model_out),
            "gate_threshold": float(
                training_summary.get("gate_threshold", args.gate_threshold)
            ),
            "num_clusters": int(getattr(model, "num_clusters", 0)),
            "num_nodes": int(graph.data.node_types.numel()),
            "num_relations": len(graph.relation_index),
            "timestamp": time.time(),
        }
        if args.npz_out:
            export_cluster_npz(
                model,
                graph,
                args.npz_out,
                metadata=npz_metadata,
            )
        if mlflow_client is not None:
            mlflow_client.log_artifact(str(model_artifact_path))
            mlflow_client.log_artifact(str(args.out))
            mlflow_client.log_metric(
                "partition_active_clusters", len(partition.active_clusters)
            )

        if args.checkpoint_path is not None:
            checkpoint_display = (
                training_summary.get("checkpoint_path")
                if training_summary.get("checkpoint_path")
                else str(args.checkpoint_path)
            )
            print(
                f"[INFO] Checkpoint stored at {checkpoint_display} after {total_epochs_completed} total epochs"
            )

    print(
        f"Saved partition with {len(partition.active_clusters)} clusters to {args.out}"
    )


if __name__ == "__main__":
    main()
