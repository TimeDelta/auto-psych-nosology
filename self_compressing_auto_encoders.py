import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphNorm, MessagePassing, RGCNConv, global_mean_pool
from torch_geometric.utils import degree, softmax

from utility import generate_random_string


# Ensure every node carries a graph id so per-graph reductions stay size-aware.
def _ensure_batch_vector(
    batch: Optional[torch.LongTensor], num_nodes: int, device: torch.device
) -> torch.LongTensor:
    if batch is not None:
        return batch
    return torch.zeros(num_nodes, dtype=torch.long, device=device)


# Fast scatter add used to average metrics per graph without padding tensors.
def _scatter_add_1d(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    out = torch.zeros(dim_size, device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


# Helper that normalizes node/edge sums by the true graph size to keep losses
# invariant to the number of nodes/edges seen in each mini-batch element.
def _per_graph_mean(
    values: torch.Tensor, batch: torch.Tensor, num_graphs: int, eps: float = 1e-12
) -> torch.Tensor:
    sums = _scatter_add_1d(values, batch, num_graphs)
    counts = _scatter_add_1d(torch.ones_like(values), batch, num_graphs)
    means = torch.zeros_like(sums)
    mask = counts > 0
    means[mask] = sums[mask] / counts[mask].clamp_min(eps)
    return means


# Standardize Laplacian positional encodings per graph to avoid scale drift when
# graphs of very different orders share a batch.
def _standardize_positional_encodings(
    pos_enc: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    if pos_enc.numel() == 0:
        return pos_enc
    device = pos_enc.device
    feat_dim = pos_enc.size(1)
    standardized = torch.zeros_like(pos_enc)
    for graph_id in range(num_graphs):
        mask = batch == graph_id
        if mask.sum() == 0:
            continue
        graph_pe = pos_enc[mask]
        mean = graph_pe.mean(dim=0, keepdim=True)
        std = graph_pe.std(dim=0, unbiased=False, keepdim=True)
        std = std.clamp_min(1e-6)
        standardized[mask] = (graph_pe - mean) / std
    if standardized.shape[1] < feat_dim:
        pad = torch.zeros(
            (standardized.size(0), feat_dim - standardized.shape[1]), device=device
        )
        standardized = torch.cat([standardized, pad], dim=1)
    return standardized


class SharedAttributeVocab(nn.Module):
    """
    Holds one name-to-index mapping plus a single Embedding that
    can grow on‐the‐fly as new names are added.
    """

    def __init__(self, initial_names: List[str], embedding_dim: int):
        super().__init__()
        self.name_to_index = {name: i for i, name in enumerate(initial_names)}
        self.name_to_index["<UNK>"] = len(self.name_to_index)
        self.name_to_index["<EOS>"] = len(self.name_to_index)
        self.index_to_name = {i: name for name, i in self.name_to_index.items()}
        self.embedding = nn.Embedding(len(self.name_to_index), embedding_dim)

    def add_names(self, new_names: List[str]):
        names_to_add: List[str] = []
        for name in new_names:
            if not name:
                continue
            if name not in self.name_to_index:
                names_to_add.append(name)

        if not names_to_add:
            return

        starting_index = len(self.name_to_index)
        for offset, name in enumerate(names_to_add):
            new_idx = starting_index + offset
            self.name_to_index[name] = new_idx
            self.index_to_name[new_idx] = name

        # expand the embedding matrix with He‐style initialization for the new rows
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        old_weight = self.embedding.weight.data.to(device=device, dtype=dtype)
        fan_in = old_weight.size(1)
        new_rows = torch.randn(len(names_to_add), fan_in, device=device, dtype=dtype)
        new_rows = new_rows * math.sqrt(2 / fan_in)
        new_weight = torch.cat([old_weight, new_rows], dim=0)
        self.embedding = nn.Embedding.from_pretrained(new_weight, freeze=False)
        self.embedding.to(device)


class NodeAttributeDeepSetEncoder(nn.Module):
    """
    Permutation‐invariant encoder for a dictionary of arbitrary node attributes.
    See "Deep Sets" by Zaheer et al. at https://arxiv.org/abs/1703.06114
    """

    def __init__(
        self,
        shared_attr_vocab: SharedAttributeVocab,
        encoder_hdim: int,
        aggregator_hdim: int,
        out_dim: int,
    ):
        super().__init__()
        self.shared_attr_vocab = shared_attr_vocab
        self.max_value_dim = shared_attr_vocab.embedding.embedding_dim
        self.attr_encoder = nn.Sequential(  # phi
            nn.Linear(
                shared_attr_vocab.embedding.embedding_dim + self.max_value_dim,
                encoder_hdim,
            ),
            nn.ReLU(),
            nn.Linear(encoder_hdim, encoder_hdim),
            nn.ReLU(),
        )
        self.aggregator = nn.Sequential(  # rho
            nn.Linear(encoder_hdim, aggregator_hdim),
            nn.ReLU(),
            nn.Linear(aggregator_hdim, out_dim),
        )
        self.out_dim = out_dim

    def get_value_tensor(self, value: Any):
        device = self.shared_attr_vocab.embedding.weight.device
        if isinstance(value, (int, float)):
            value_tensor = torch.tensor(
                [float(value)], dtype=torch.float, device=device
            )
        elif isinstance(value, torch.Tensor):
            value_tensor = value.to(device=device)
        elif isinstance(value, str):
            token = value if value else "<UNK>"
            if token not in self.shared_attr_vocab.name_to_index:
                self.shared_attr_vocab.add_names([token])
            index = torch.tensor(
                self.shared_attr_vocab.name_to_index.get(
                    token, self.shared_attr_vocab.name_to_index["<UNK>"]
                ),
                dtype=torch.long,
                device=device,
            )
            value_tensor = self.shared_attr_vocab.embedding(index)
        else:
            raise TypeError(f"Unsupported attribute value type: {type(value)}")
        value_tensor = value_tensor.view(-1)
        if value_tensor.numel() < self.max_value_dim:
            pad_amt = self.max_value_dim - value_tensor.numel()
            return F.pad(value_tensor, (0, pad_amt), "constant", 0.0)
        return value_tensor[: self.max_value_dim]

    def forward(self, attr_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not attr_dict or len(attr_dict) == 0:
            device = self.shared_attr_vocab.embedding.weight.device
            return torch.zeros(self.aggregator[-1].out_features, device=device)

        phis = []
        device = self.shared_attr_vocab.embedding.weight.device
        for attr, value in sorted(
            attr_dict.items(), key=lambda item: getattr(item[0], "name", str(item[0]))
        ):
            attr_name = getattr(attr, "name", str(attr))
            if attr_name not in self.shared_attr_vocab.name_to_index:
                self.shared_attr_vocab.add_names([attr_name])
            name_index = torch.tensor(
                self.shared_attr_vocab.name_to_index[attr_name],
                dtype=torch.long,
                device=device,
            )
            value_tensor = self.get_value_tensor(value)
            name_embedding = self.shared_attr_vocab.embedding(name_index)
            phis.append(
                self.attr_encoder(
                    torch.cat([name_embedding, value_tensor.to(device)], dim=0)
                )
            )
        return self.aggregator(torch.stack(phis, dim=0).sum(dim=0))


class HardConcreteGate(nn.Module):
    """Louizos et al. (2018) hard-concrete L0 gate with stability enhancements."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        temperature: float = 2.0 / 3.0,
        stretch_epsilon: float = 0.1,
        pre_activation_clip: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(shape))
        self.temperature = temperature
        self.stretch_epsilon = stretch_epsilon
        self.pre_activation_clip = pre_activation_clip
        self.eps = eps

    @property
    def stretch_lower_bound(self) -> float:
        # Stretch slightly below zero to keep the relaxed support wide early on.
        return -self.stretch_epsilon

    @property
    def stretch_upper_bound(self) -> float:
        # Allow small overshoot above one for less biased gradients.
        return 1.0 + self.stretch_epsilon

    def _sample(self, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.log(u + self.eps) - torch.log(1 - u + self.eps)
            pre_activation = (s + self.log_alpha) / max(self.temperature, self.eps)
        else:
            pre_activation = self.log_alpha / max(self.temperature, self.eps)
        pre_activation = pre_activation.clamp(
            -self.pre_activation_clip, self.pre_activation_clip
        )
        z = torch.sigmoid(pre_activation)
        z = (
            z * (self.stretch_upper_bound - self.stretch_lower_bound)
            + self.stretch_lower_bound
        )
        return z.clamp(0.0, 1.0)

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        training = self.training if training is None else training
        return self._sample(training)

    def expected_l0(self) -> torch.Tensor:
        limit_ratio = -self.stretch_lower_bound / max(
            self.stretch_upper_bound, self.eps
        )
        limit_ratio = max(limit_ratio, self.eps)
        threshold = max(self.temperature, self.eps) * math.log(limit_ratio)
        return torch.sigmoid(self.log_alpha - threshold)


class ClusterGate(nn.Module):
    """Applies hard-concrete gating across clusters."""

    def __init__(
        self,
        num_clusters: int,
        temperature: float = 2.0 / 3.0,
        stretch_epsilon: float = 0.1,
        pre_activation_clip: float = 2.0,
    ) -> None:
        super().__init__()
        self._gate = HardConcreteGate(
            (num_clusters,),
            temperature=temperature,
            stretch_epsilon=stretch_epsilon,
            pre_activation_clip=pre_activation_clip,
        )

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        return self._gate(training=training)

    def expected_l0(self) -> torch.Tensor:
        return self._gate.expected_l0()


class InterClusterGate(nn.Module):
    """Hard-concrete gates across relation-specific inter-cluster matrices."""

    def __init__(
        self,
        num_relations: int,
        num_clusters: int,
        temperature: float = 2.0 / 3.0,
        stretch_epsilon: float = 0.1,
        pre_activation_clip: float = 2.0,
    ) -> None:
        super().__init__()
        self._gate = HardConcreteGate(
            (num_relations, num_clusters, num_clusters),
            temperature=temperature,
            stretch_epsilon=stretch_epsilon,
            pre_activation_clip=pre_activation_clip,
        )

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        return self._gate(training=training)

    def expected_l0(self) -> torch.Tensor:
        return self._gate.expected_l0()


class RGCNClusterEncoder(nn.Module):
    """Relational GCN feature encoder with size-robust normalization."""

    def __init__(
        self,
        num_node_types: int,
        attr_encoder: NodeAttributeDeepSetEncoder,
        num_clusters: int,
        hidden_dims: Optional[List[int]] = None,
        num_relations: int = 1,
        type_embedding_dim: Optional[int] = None,
        positional_dim: int = 0,
        dropout: float = 0.0,
        use_graphnorm: bool = True,
        dropedge_rate: float = 0.0,
        degree_dropout_rate: float = 0.0,
        use_degree_norm: bool = True,
        enable_relation_reweighting: bool = True,
        relation_reweight_power: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.attr_encoder = attr_encoder
        self.num_relations = max(1, num_relations)
        self.dropout = dropout
        self.positional_dim = positional_dim
        self.use_graphnorm = use_graphnorm
        self.dropedge_rate = max(0.0, min(1.0, dropedge_rate))
        self.degree_dropout_rate = max(0.0, min(1.0, degree_dropout_rate))
        self.use_degree_norm = use_degree_norm
        self.enable_relation_reweighting = enable_relation_reweighting
        self.relation_reweight_power = relation_reweight_power
        embed_dim = type_embedding_dim or attr_encoder.out_dim
        # Keep type embeddings compact so node identity features don't balloon.
        self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)
        in_dim = embed_dim + attr_encoder.out_dim + positional_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.convs.append(
                RGCNConv(
                    in_dim,
                    hidden_dim,
                    self.num_relations,
                    num_bases=min(self.num_relations, hidden_dim),
                    bias=True,
                )
            )
            # Prefer GraphNorm/LayerNorm over BatchNorm so batches mixing tiny and
            # huge graphs produce well-behaved statistics as requested.
            if use_graphnorm:
                self.norms.append(GraphNorm(hidden_dim))
            else:
                self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.output_dim = in_dim

    def _prepare_edge_type(
        self,
        edge_type: Optional[torch.LongTensor],
        edge_count: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if edge_type is None:
            return torch.zeros(edge_count, dtype=torch.long, device=device)
        return edge_type.to(device)

    def forward(
        self,
        node_types: torch.LongTensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        node_attributes: Optional[List[List[Dict[str, Any]]]] = None,
        edge_type: Optional[torch.LongTensor] = None,
        positional_encodings: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = node_types.device
        batch_vec = _ensure_batch_vector(batch, node_types.size(0), device)
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1

        type_embedding = self.node_type_embedding(node_types)

        num_nodes = node_types.size(0)
        edge_index = edge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)
        if positional_encodings is not None:
            positional_encodings = positional_encodings.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        if edge_index.numel() > 0:
            if self.training and self.dropedge_rate > 0.0:
                # DropEdge: randomly thin edges so deep stacks do not oversmooth.
                keep_prob = 1.0 - self.dropedge_rate
                mask = torch.rand(edge_index.size(1), device=device) < keep_prob
                if mask.sum() == 0:
                    idx = torch.randint(0, mask.numel(), (1,), device=device)
                    mask[idx] = True
                edge_index = edge_index[:, mask]
                if edge_type is not None:
                    edge_type = edge_type[mask]
                if edge_weight is not None:
                    edge_weight = edge_weight[mask]

            if (
                self.training
                and self.degree_dropout_rate > 0.0
                and edge_index.numel() > 0
            ):
                row = edge_index[0]
                src_deg = degree(row, num_nodes, dtype=torch.float).to(device)
                max_deg = src_deg.max().clamp_min(1.0)
                # Higher-degree sources receive proportionally stronger dropout.
                drop_prob = self.degree_dropout_rate * (src_deg[row] / max_deg)
                keep_mask = torch.rand_like(drop_prob) > drop_prob
                if keep_mask.sum() == 0:
                    idx = torch.randint(0, keep_mask.numel(), (1,), device=device)
                    keep_mask[idx] = True
                edge_index = edge_index[:, keep_mask]
                if edge_type is not None:
                    edge_type = edge_type[keep_mask]
                if edge_weight is not None:
                    edge_weight = edge_weight[keep_mask]

        etype = self._prepare_edge_type(edge_type, edge_index.size(1), device)

        if edge_index.numel() > 0:
            edge_scale = torch.ones(edge_index.size(1), device=device)
            if (
                self.enable_relation_reweighting
                and etype is not None
                and etype.numel() > 0
            ):
                # Reweight relations by inverse frequency so rare types influence training.
                relation_counts = (
                    torch.bincount(etype.detach().cpu(), minlength=self.num_relations)
                    .to(device=device, dtype=torch.float)
                    .clamp_min(1.0)
                )
                relation_scale = relation_counts.pow(-self.relation_reweight_power)
                edge_scale = relation_scale[etype]
        else:
            edge_scale = torch.ones(0, device=device)

        # Precompute symmetric degree normalization factors when requested.
        if self.use_degree_norm and edge_index.numel() > 0:
            row, col = edge_index
            deg_row = (
                degree(row, num_nodes, dtype=torch.float).to(device).clamp_min(1.0)
            )
            deg_col = (
                degree(col, num_nodes, dtype=torch.float).to(device).clamp_min(1.0)
            )
            src_scale = deg_row.pow(-0.5)
            dst_scale = deg_col.pow(-0.5)
            if edge_scale.numel() > 0:
                src_edge_sum = _scatter_add_1d(edge_scale, row, num_nodes)
                src_scale = src_scale * ((src_edge_sum / deg_row).clamp_min(1e-6))
        else:
            src_scale = dst_scale = torch.ones(num_nodes, device=device)

        if node_attributes and len(node_attributes) > 0:
            # Allow already-collated per-node dicts or per-graph lists from PyG.
            if isinstance(node_attributes[0], dict):
                attr_embedding = torch.stack(
                    [self.attr_encoder(attrs).to(device) for attrs in node_attributes],
                    dim=0,
                )
            else:
                attr_embedding = torch.stack(
                    [
                        self.attr_encoder(attrs).to(device)
                        for graph_attrs in node_attributes
                        for attrs in graph_attrs
                    ],
                    dim=0,
                )
        else:
            attr_embedding = torch.zeros(
                (node_types.size(0), self.attr_encoder.out_dim), device=device
            )

        features = [type_embedding, attr_embedding]

        if self.positional_dim > 0:
            # Clip/pad Laplacian eigenvectors so every graph contributes R modes.
            if positional_encodings is None:
                positional = torch.zeros(
                    (node_types.size(0), self.positional_dim), device=device
                )
            else:
                positional = positional_encodings.to(device)
                if positional.size(1) < self.positional_dim:
                    pad = torch.zeros(
                        (positional.size(0), self.positional_dim - positional.size(1)),
                        device=device,
                    )
                    positional = torch.cat([positional, pad], dim=1)
                elif positional.size(1) > self.positional_dim:
                    positional = positional[:, : self.positional_dim]
                positional = _standardize_positional_encodings(
                    positional, batch_vec, num_graphs
                )
            features.append(positional)

        x = torch.cat(features, dim=-1)
        src_scale_vec = src_scale.unsqueeze(-1)
        dst_scale_vec = dst_scale.unsqueeze(-1)
        for layer_idx, conv in enumerate(self.convs):
            x = x * src_scale_vec
            x = conv(x, edge_index, etype)
            x = x * dst_scale_vec
            norm = self.norms[layer_idx]
            if isinstance(norm, GraphNorm):
                x = norm(x, batch_vec)
            else:
                x = norm(x)
            x = F.relu(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        info = {
            "node_embeddings": x,
            "batch": batch_vec,
        }
        return x, info


class ClusteredGraphReconstructor(nn.Module):
    """Decodes cluster assignments into adjacency logits with L0 gating."""

    def __init__(
        self,
        num_relations: int,
        num_clusters: int,
        negative_sampling_ratio: float = 1.0,
        restrict_negatives_to_types: bool = True,
        gate_temperature: float = 2.0 / 3.0,
        gate_stretch_epsilon: float = 0.1,
        gate_pre_activation_clip: float = 2.0,
        enable_relation_reweighting: bool = True,
        relation_reweight_power: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_relations = max(1, num_relations)
        self.num_clusters = num_clusters
        self.negative_sampling_ratio = max(0.0, negative_sampling_ratio)
        self.restrict_negatives_to_types = restrict_negatives_to_types
        self.enable_relation_reweighting = enable_relation_reweighting
        self.relation_reweight_power = relation_reweight_power
        self.inter_cluster_logits = nn.Parameter(
            torch.zeros(self.num_relations, num_clusters, num_clusters)
        )
        nn.init.xavier_uniform_(self.inter_cluster_logits)
        self.inter_cluster_gate = InterClusterGate(
            self.num_relations,
            num_clusters,
            temperature=gate_temperature,
            stretch_epsilon=gate_stretch_epsilon,
            pre_activation_clip=gate_pre_activation_clip,
        )
        self.absent_bias = nn.Parameter(torch.zeros(self.num_relations))

    def _inter_weights(self, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_sample = self.inter_cluster_gate(training=training)
        weights = torch.sigmoid(self.inter_cluster_logits) * gate_sample
        return weights, gate_sample

    def _edge_logits(
        self,
        assignments: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty(0, device=assignments.device)
        src = assignments[edge_index[0]]
        dst = assignments[edge_index[1]]
        relation_weights = weights[edge_type]
        weighted_dst = torch.bmm(relation_weights, dst.unsqueeze(-1)).squeeze(-1)
        logits = (src * weighted_dst).sum(dim=-1)
        logits = logits + self.absent_bias.to(assignments.device)[edge_type]
        return logits

    def _sample_negative_edges(
        self,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor],
        num_nodes: int,
        node_types: Optional[torch.LongTensor],
        ratio: float,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if ratio <= 0.0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
            )

        if num_nodes == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
            )

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        edge_index_cpu = edge_index.detach().cpu()
        batch_cpu = batch.detach().cpu()
        node_types_cpu = node_types.detach().cpu() if node_types is not None else None
        neg_edges: List[Tuple[int, int]] = []
        neg_types: List[int] = []
        neg_batch: List[int] = []

        for graph_id in range(num_graphs):
            node_ids = (batch_cpu == graph_id).nonzero(as_tuple=False).view(-1).tolist()
            if len(node_ids) <= 1:
                continue
            pos_mask = (batch_cpu[edge_index_cpu[0]] == graph_id) & (
                batch_cpu[edge_index_cpu[1]] == graph_id
            )
            edges = edge_index_cpu[:, pos_mask]
            pos_edges = {tuple(edge) for edge in edges.t().tolist()}
            max_possible = len(node_ids) ** 2
            available = max(0, max_possible - len(pos_edges))
            if available == 0:
                continue
            num_pos = max(1, edges.size(1))
            target = min(available, int(math.ceil(num_pos * ratio)))
            attempts = 0
            seen: set[Tuple[int, int]] = set()
            if self.restrict_negatives_to_types and node_types_cpu is not None:
                # Sample negatives within type buckets so edge difficulty stays
                # comparable even when graphs have uneven type distributions.
                type_to_nodes: Dict[int, List[int]] = {}
                for node_id in node_ids:
                    node_type_id = int(node_types_cpu[node_id])
                    type_to_nodes.setdefault(node_type_id, []).append(node_id)
            while len(seen) < target and attempts < max(100, target * 10):
                if (
                    self.restrict_negatives_to_types
                    and node_types_cpu is not None
                    and len(node_ids) > 1
                ):
                    valid_types = [
                        t for t, members in type_to_nodes.items() if len(members) > 0
                    ]
                    if not valid_types:
                        break
                    type_id = random.choice(valid_types)
                    candidates = type_to_nodes[type_id]
                    u = random.choice(candidates)
                    v = random.choice(candidates)
                else:
                    u = random.choice(node_ids)
                    v = random.choice(node_ids)
                attempts += 1
                pair = (u, v)
                if pair in pos_edges or pair in seen:
                    continue
                seen.add(pair)
            for u, v in seen:
                neg_edges.append((u, v))
                neg_types.append(random.randrange(self.num_relations))
                neg_batch.append(graph_id)

        if not neg_edges:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
            )

        neg_edge_tensor = (
            torch.tensor(neg_edges, dtype=torch.long, device=edge_index.device)
            .t()
            .contiguous()
        )
        neg_type_tensor = torch.tensor(
            neg_types, dtype=torch.long, device=edge_index.device
        )
        neg_batch_tensor = torch.tensor(
            neg_batch, dtype=torch.long, device=edge_index.device
        )
        return neg_edge_tensor, neg_type_tensor, neg_batch_tensor

    def forward(
        self,
        assignments: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        edge_type: Optional[torch.LongTensor] = None,
        negative_sampling_ratio: Optional[float] = None,
        node_types: Optional[torch.LongTensor] = None,
        negative_confidence_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = assignments.device
        ratio = (
            self.negative_sampling_ratio
            if negative_sampling_ratio is None
            else negative_sampling_ratio
        )
        batch_vec = _ensure_batch_vector(
            batch, assignments.size(0), device=assignments.device
        )
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1

        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        weights, gate_sample = self._inter_weights(training=self.training)

        if edge_index.size(1) == 0:
            pos_logits = torch.empty(0, device=device)
            pos_losses = torch.empty(0, device=device)
            pos_batch = torch.empty(0, dtype=torch.long, device=device)
        else:
            pos_logits = self._edge_logits(assignments, edge_index, edge_type, weights)
            pos_losses = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none"
            )
            pos_batch = batch_vec[edge_index[0]]
            if self.enable_relation_reweighting and edge_type.numel() > 0:
                rel_counts = (
                    torch.bincount(
                        edge_type.detach().cpu(), minlength=self.num_relations
                    )
                    .to(device=device, dtype=torch.float)
                    .clamp_min(1.0)
                )
                rel_scale = rel_counts.pow(-self.relation_reweight_power)
                pos_losses = pos_losses * rel_scale[edge_type]

        neg_edges, neg_types, neg_batch = self._sample_negative_edges(
            edge_index, batch_vec, assignments.size(0), node_types, ratio
        )
        if neg_edges.size(1) == 0:
            neg_logits = torch.empty(0, device=device)
            neg_losses = torch.empty(0, device=device)
        else:
            neg_logits = self._edge_logits(assignments, neg_edges, neg_types, weights)
            neg_losses = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )
            if self.enable_relation_reweighting and neg_types.numel() > 0:
                rel_counts = (
                    torch.bincount(
                        neg_types.detach().cpu(), minlength=self.num_relations
                    )
                    .to(device=device, dtype=torch.float)
                    .clamp_min(1.0)
                )
                rel_scale = rel_counts.pow(-self.relation_reweight_power)
                neg_losses = neg_losses * rel_scale[neg_types]
            if negative_confidence_weight is not None:
                neg_losses = neg_losses * negative_confidence_weight

        # Average positive edge loss per graph so sparse and dense subgraphs
        # contribute equally regardless of |E|.
        pos_sum = (
            _scatter_add_1d(pos_losses, pos_batch, num_graphs)
            if pos_losses.numel() > 0
            else torch.zeros(num_graphs, device=device)
        )
        pos_counts = (
            _scatter_add_1d(torch.ones_like(pos_losses), pos_batch, num_graphs)
            if pos_losses.numel() > 0
            else torch.zeros(num_graphs, device=device)
        )
        pos_avg = torch.zeros(num_graphs, device=device)
        mask = pos_counts > 0
        pos_avg[mask] = pos_sum[mask] / pos_counts[mask]

        if neg_losses.numel() > 0:
            # Normalize negatives against requested ratio to keep contrastive
            # pressure constant across size buckets.
            neg_sum = _scatter_add_1d(neg_losses, neg_batch, num_graphs)
            neg_counts = _scatter_add_1d(
                torch.ones_like(neg_losses), neg_batch, num_graphs
            )
        else:
            neg_sum = torch.zeros(num_graphs, device=device)
            neg_counts = torch.zeros(num_graphs, device=device)

        neg_expected = ratio * pos_counts
        neg_term = torch.zeros(num_graphs, device=device)
        valid_neg = neg_expected > 0
        neg_term[valid_neg] = neg_sum[valid_neg] / neg_expected[valid_neg].clamp_min(
            1e-8
        )

        graph_losses = pos_avg + neg_term
        recon_loss = (
            graph_losses.mean()
            if graph_losses.numel() > 0
            else assignments.new_tensor(0.0)
        )

        info: Dict[str, torch.Tensor] = {
            "graph_losses": graph_losses.detach(),
            "graph_pos_loss": pos_avg.detach(),
            "graph_neg_loss": neg_term.detach(),
            "pos_edge_counts": pos_counts.detach(),
            "neg_edge_counts": neg_counts.detach(),
            "num_negatives": neg_counts.sum().detach(),
            "inter_l0": self.inter_cluster_gate.expected_l0(),
            "inter_gate_sample": gate_sample.detach(),
        }
        if negative_confidence_weight is not None:
            if not torch.is_tensor(negative_confidence_weight):
                negative_confidence_weight = torch.tensor(
                    negative_confidence_weight, device=device
                )
            info["negative_confidence_weight"] = negative_confidence_weight.detach()
        if pos_logits.numel() > 0:
            info["pos_logits"] = pos_logits.detach()
        if neg_logits.numel() > 0:
            info["neg_logits"] = neg_logits.detach()
        return recon_loss, info


@dataclass
class PartitionResult:
    """Container holding a hard partition derived from rGCN-SCAE outputs."""

    node_to_cluster: torch.LongTensor
    active_clusters: torch.LongTensor
    cluster_members: Dict[int, torch.LongTensor]
    gate_values: torch.Tensor
    assignments: torch.Tensor


class SelfCompressingRGCNAutoEncoder(nn.Module):
    """Self-compressing AE with rGCN encoder, L0 cluster gates, and relation gates."""

    def __init__(
        self,
        num_node_types: int,
        attr_encoder: NodeAttributeDeepSetEncoder,
        num_clusters: int,
        num_relations: int = 1,
        hidden_dims: Optional[List[int]] = None,
        type_embedding_dim: Optional[int] = None,
        dropout: float = 0.0,
        negative_sampling_ratio: float = 1.0,
        l0_cluster_weight: float = 1e-4,
        l0_inter_weight: float = 1e-4,
        entropy_weight: float = 1e-3,
        dirichlet_alpha: Union[float, Sequence[float]] = 0.5,
        dirichlet_weight: float = 1e-3,
        embedding_norm_weight: float = 1e-4,
        kld_weight: float = 1e-3,
        entropy_eps: float = 1e-12,
        positional_dim: int = 0,
        assignment_temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 1e-3,
        prototype_momentum: float = 0.1,
        restrict_negatives_to_types: bool = True,
        consistency_weight: float = 1e-3,
        memory_bank_momentum: float = 0.5,
        memory_bank_max_size: int = 50000,
        use_virtual_node: bool = False,
        virtual_node_weight: float = 0.1,
        sparsity_warmup_steps: int = 400,
        gate_temperature_start: Optional[float] = None,
        gate_temperature_end: Optional[float] = None,
        gate_temperature_anneal_steps: int = 400,
        gate_stretch_epsilon: float = 0.1,
        gate_pre_activation_clip: float = 2.0,
        assignment_temperature_start: Optional[float] = None,
        assignment_temperature_anneal_steps: int = 400,
        gate_entropy_floor_bits: float = 0.5,
        gate_entropy_weight: float = 1e-3,
        assignment_entropy_floor: Optional[float] = None,
        neg_entropy_scale: float = 1.0,
        neg_entropy_min_bits: float = 0.1,
        neg_entropy_max_weight: float = 5.0,
        ema_gate_decay: float = 0.9,
        revival_gate_threshold: float = 0.05,
        revival_usage_threshold: float = 0.05,
        dropedge_rate: float = 0.15,
        degree_dropout_rate: float = 0.1,
        degree_decorrelation_weight: float = 1e-3,
        relation_reweight_power: float = 0.5,
        enable_relation_reweighting: bool = True,
        active_gate_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            num_node_types: Cardinality of the node-type embedding table.
            attr_encoder: Shared attribute encoder returning per-node feature vectors.
            num_clusters: Number of latent clusters/gates to learn.
            num_relations: Count of observed edge types in the multiplex graph.
            hidden_dims: Hidden dimensions for stacked RGCN layers.
            type_embedding_dim: Optional override for the node-type embedding size.
            dropout: Feature dropout applied after each RGCN layer.
            negative_sampling_ratio: Target ratio of negatives per positive edge.
            l0_cluster_weight: Base weight on the encoder cluster gate L0 penalty.
            l0_inter_weight: Base weight on decoder inter-cluster gate sparsity.
            entropy_weight: Strength of the per-graph assignment entropy floor.
            dirichlet_alpha: Parameters of the Dirichlet prior on cluster usage.
            dirichlet_weight: Scaling on the Dirichlet log-likelihood term.
            embedding_norm_weight: Weight on per-graph latent L2 norms.
            kld_weight: Variational penalty encouraging per-graph latent Gaussianity.
            entropy_eps: Numerical epsilon for entropy- and norm-related clamps.
            positional_dim: Count of positional-encoding features appended to inputs.
            assignment_temperature: Final temperature for Sinkhorn-balanced assignments.
            sinkhorn_iterations: Number of Sinkhorn normalisation steps per batch.
            sinkhorn_epsilon: Stabiliser used inside the Sinkhorn log-space updates.
            prototype_momentum: EMA coefficient for updating assignment prototypes.
            restrict_negatives_to_types: If True, negatives are sampled within type buckets.
            consistency_weight: Weight for memory-bank consistency across batches.
            memory_bank_momentum: EMA factor for stored node embeddings.
            memory_bank_max_size: Maximum number of node embeddings kept in memory bank.
            use_virtual_node: Whether to inject a lightweight virtual node per graph.
            virtual_node_weight: Mix-in weight for virtual-node context features.
            sparsity_warmup_steps: Steps to ramp L0 penalties from zero to target weight.
            gate_temperature_start: Starting temperature for hard-concrete gates.
            gate_temperature_end: Final temperature reached after annealing.
            gate_temperature_anneal_steps: Steps over which gate temperature anneals.
            gate_stretch_epsilon: Stretch magnitude applied to hard-concrete bounds.
            gate_pre_activation_clip: Clamp applied to gate pre-activations for stability.
            assignment_temperature_start: Starting temperature for prototype assignments.
            assignment_temperature_anneal_steps: Steps for assignment temperature anneal.
            gate_entropy_floor_bits: Minimum desired gate entropy (bits) before penalising collapse.
            gate_entropy_weight: Strength of the gate entropy floor penalty.
            assignment_entropy_floor: Per-graph entropy floor for assignments (defaults to log K).
            neg_entropy_scale: Multiplier linking gate entropy to negative-weight scaling.
            neg_entropy_min_bits: Minimum entropy denominator used in the negative reweight term.
            neg_entropy_max_weight: Upper bound on entropy-derived negative weights.
            ema_gate_decay: EMA decay applied when smoothing stochastic gate samples.
            revival_gate_threshold: Threshold below which gates are considered inactive.
            revival_usage_threshold: Usage level needed to resurrect inactive gates.
            dropedge_rate: Probability of dropping an edge during training (DropEdge).
            degree_dropout_rate: Degree-proportional dropout rate for hub mitigation.
            degree_decorrelation_weight: Weight on the penalty forcing latents to decorrelate from log degree.
            relation_reweight_power: Power-law for relation frequency reweighting.
            enable_relation_reweighting: Enables inverse-frequency reweighting when True.
            active_gate_threshold: Probability cutoff used when counting active gates for training diagnostics.
        """
        super().__init__()
        base_gate_temperature = 2.0 / 3.0
        gate_temp_target = (
            float(gate_temperature_end)
            if gate_temperature_end is not None
            else base_gate_temperature
        )
        gate_temp_start = (
            float(gate_temperature_start)
            if gate_temperature_start is not None
            else (
                gate_temp_target * 2.0
                if gate_temperature_anneal_steps > 0
                else gate_temp_target
            )
        )
        gate_temp_end = gate_temp_target
        self.encoder = RGCNClusterEncoder(
            num_node_types=num_node_types,
            attr_encoder=attr_encoder,
            num_clusters=num_clusters,
            hidden_dims=hidden_dims,
            num_relations=num_relations + (1 if use_virtual_node else 0),
            type_embedding_dim=type_embedding_dim,
            positional_dim=positional_dim,
            dropout=dropout,
            dropedge_rate=dropedge_rate,
            degree_dropout_rate=degree_dropout_rate,
            use_degree_norm=True,
            enable_relation_reweighting=enable_relation_reweighting,
            relation_reweight_power=relation_reweight_power,
        )
        self.decoder = ClusteredGraphReconstructor(
            num_relations=num_relations + (1 if use_virtual_node else 0),
            num_clusters=num_clusters,
            negative_sampling_ratio=negative_sampling_ratio,
            restrict_negatives_to_types=restrict_negatives_to_types,
            gate_temperature=gate_temp_start,
            gate_stretch_epsilon=gate_stretch_epsilon,
            gate_pre_activation_clip=gate_pre_activation_clip,
            enable_relation_reweighting=enable_relation_reweighting,
            relation_reweight_power=relation_reweight_power,
        )
        self._base_l0_cluster_weight = float(l0_cluster_weight)
        self._base_l0_inter_weight = float(l0_inter_weight)
        self.l0_cluster_weight = float(l0_cluster_weight)
        self.l0_inter_weight = float(l0_inter_weight)
        self.entropy_weight = float(entropy_weight)
        self.dirichlet_weight = float(dirichlet_weight)
        self._dirichlet_alpha_config = dirichlet_alpha
        self.embedding_norm_weight = float(embedding_norm_weight)
        self.kld_weight = float(kld_weight)
        self.entropy_eps = float(entropy_eps)
        assignment_temp_end = float(assignment_temperature)
        self.assignment_temperature_start = (
            float(assignment_temperature_start)
            if assignment_temperature_start is not None
            else (
                assignment_temp_end * 2.0
                if assignment_temperature_anneal_steps > 0
                else assignment_temp_end
            )
        )
        self.assignment_temperature_end = assignment_temp_end
        self.assignment_temperature = self.assignment_temperature_start
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.prototype_momentum = prototype_momentum
        self.consistency_weight = consistency_weight
        self.use_virtual_node = use_virtual_node
        self.virtual_node_weight = virtual_node_weight
        self.sparsity_warmup_steps = max(0, int(sparsity_warmup_steps))
        self.gate_temperature_start = gate_temp_start
        self.gate_temperature_end = gate_temp_end
        self.gate_temperature_anneal_steps = max(0, int(gate_temperature_anneal_steps))
        self.assignment_temperature_anneal_steps = max(
            0, int(assignment_temperature_anneal_steps)
        )
        self.gate_entropy_floor_bits = float(gate_entropy_floor_bits)
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.assignment_entropy_floor = (
            float(assignment_entropy_floor)
            if assignment_entropy_floor is not None
            else math.log(max(float(num_clusters), 1.0))
        )
        self.neg_entropy_scale = float(neg_entropy_scale)
        self.neg_entropy_min_bits = max(float(neg_entropy_min_bits), 1e-4)
        self.neg_entropy_max_weight = float(neg_entropy_max_weight)
        self.ema_gate_decay = float(ema_gate_decay)
        self.revival_gate_threshold = float(max(revival_gate_threshold, 0.0))
        self.revival_usage_threshold = float(max(revival_usage_threshold, 0.0))
        self.degree_decorrelation_weight = float(degree_decorrelation_weight)
        self.enable_relation_reweighting = enable_relation_reweighting
        self.active_gate_threshold = float(active_gate_threshold)

        self.cluster_gate = ClusterGate(
            num_clusters,
            temperature=gate_temp_start,
            stretch_epsilon=gate_stretch_epsilon,
            pre_activation_clip=gate_pre_activation_clip,
        )
        self.num_clusters = num_clusters
        self.num_relations = num_relations + (1 if use_virtual_node else 0)
        self.restrict_negatives_to_types = restrict_negatives_to_types
        self.memory_bank_momentum = memory_bank_momentum
        self.memory_bank_max_size = memory_bank_max_size
        self._memory_bank: "OrderedDict[Any, torch.Tensor]" = OrderedDict()
        self._memory_counts: "OrderedDict[Any, int]" = OrderedDict()
        self.virtual_relation_id = (
            (num_relations + (1 if use_virtual_node else 0) - 1)
            if use_virtual_node
            else None
        )

        self.register_buffer("_gate_ema", torch.ones(num_clusters, dtype=torch.float))
        self.register_buffer("_global_step", torch.zeros(1, dtype=torch.long))
        self.revival_logit = 0.0

        if use_virtual_node:
            self.virtual_node_type_id = self.encoder.node_type_embedding.num_embeddings
            self._extend_node_type_embedding()
        else:
            self.virtual_node_type_id = None

        # Prototypes live in encoder feature space; orthogonal init helps spread
        # clusters before Sinkhorn balancing kicks in.
        prototype_dim = self.encoder.output_dim
        self.prototype_bank = nn.Parameter(torch.zeros(num_clusters, prototype_dim))
        nn.init.orthogonal_(self.prototype_bank)

        alpha_tensor = torch.as_tensor(dirichlet_alpha, dtype=torch.float)
        if alpha_tensor.dim() == 0:
            alpha_tensor = alpha_tensor.repeat(num_clusters)
        elif alpha_tensor.numel() != num_clusters:
            raise ValueError(
                "dirichlet_alpha must be scalar or have one entry per cluster"
            )
        alpha_tensor = alpha_tensor.clamp_min(self.entropy_eps)
        prior_probs = alpha_tensor / alpha_tensor.sum().clamp_min(self.entropy_eps)
        self.register_buffer("_dirichlet_alpha", alpha_tensor)
        self.register_buffer("_dirichlet_prior", prior_probs)

    def _extend_node_type_embedding(self) -> None:
        old_embedding = self.encoder.node_type_embedding
        device = old_embedding.weight.device
        mean_vec = old_embedding.weight.data.mean(dim=0, keepdim=True)
        new_weight = torch.cat([old_embedding.weight.data, mean_vec], dim=0)
        new_embedding = nn.Embedding.from_pretrained(new_weight, freeze=False)
        self.encoder.node_type_embedding = new_embedding.to(device)

    # Balanced Sinkhorn keeps cluster usage roughly uniform within a batch so
    # tiny graphs cannot collapse all assignments into a single prototype.
    def _sinkhorn_balancing(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits
        Q = logits / max(self.sinkhorn_epsilon, 1e-6)
        Q = Q - Q.max(dim=1, keepdim=True).values
        Q = torch.exp(Q)
        Q = Q + self.entropy_eps
        target_mass = logits.size(0) / max(float(self.num_clusters), 1.0)
        for _ in range(max(self.sinkhorn_iterations, 1)):
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(self.entropy_eps)
            col_sum = Q.sum(dim=0, keepdim=True)
            Q = Q / (col_sum / target_mass).clamp_min(self.entropy_eps)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(self.entropy_eps)
        return Q

    def _balanced_assignments(
        self,
        node_embeddings: torch.Tensor,
        gate_sample: torch.Tensor,
        real_mask: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = F.normalize(self.prototype_bank, dim=-1)
        normalized_nodes = F.normalize(node_embeddings, dim=-1)
        logits = normalized_nodes @ prototypes.t()
        logits = logits / max(self.assignment_temperature, 1e-6)
        logits = logits + torch.log(gate_sample.unsqueeze(0) + self.entropy_eps)
        assignments = torch.zeros_like(logits)
        valid_idx = real_mask.nonzero(as_tuple=False).view(-1)
        if valid_idx.numel() > 0:
            valid_logits = logits[valid_idx]
            balanced = self._sinkhorn_balancing(valid_logits)
            assignments[valid_idx] = balanced
        inactive_idx = (~real_mask).nonzero(as_tuple=False).view(-1)
        if inactive_idx.numel() > 0:
            mean_assign = (
                assignments[valid_idx].mean(dim=0, keepdim=True)
                if valid_idx.numel() > 0
                else torch.full(
                    (1, self.num_clusters),
                    1.0 / self.num_clusters,
                    device=assignments.device,
                )
            )
            assignments[inactive_idx] = mean_assign
        return assignments

    # EMA prototypes avoid exploding gradients when graph sizes vary widely.
    def _update_prototypes(
        self,
        assignments: torch.Tensor,
        node_embeddings: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> None:
        if assignments.numel() == 0 or real_mask.sum() == 0:
            return
        with torch.no_grad():
            valid_assign = assignments[real_mask]
            valid_emb = node_embeddings[real_mask]
            mass = valid_assign.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
            proto_update = (valid_assign.t() @ valid_emb) / mass
            self.prototype_bank.data.mul_(1 - self.prototype_momentum).add_(
                self.prototype_momentum * proto_update
            )
            self.prototype_bank.data = F.normalize(self.prototype_bank.data, dim=-1)

    def _convert_node_ids(
        self, node_ids: Optional[Union[torch.Tensor, Sequence[Any]]]
    ) -> Optional[List[Any]]:
        if node_ids is None:
            return None
        if isinstance(node_ids, torch.Tensor):
            flat = node_ids.detach().cpu().view(-1)
            if flat.dtype.is_floating_point:
                return [float(x) for x in flat.tolist()]
            return flat.tolist()
        if isinstance(node_ids, (list, tuple)):
            normalised: List[Any] = []
            for node_id in node_ids:
                if isinstance(node_id, (list, tuple)):
                    if len(node_id) == 1 and isinstance(
                        node_id[0], (str, bytes, int, float)
                    ):
                        normalised.append(node_id[0])
                    else:
                        normalised.append(tuple(node_id))
                else:
                    normalised.append(node_id)
            return normalised
        return None

    # Memory bank ties repeated node ids together to align embeddings across
    # overlapping subgraphs regardless of batch composition.
    def _memory_consistency_loss(
        self,
        node_ids: Optional[Sequence[Any]],
        node_embeddings: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        # Memory bank reuses a momentum-smoothed target so repeated node ids stay
        # aligned across subgraphs that may never share a batch. Without it,
        # assignments on overlapping graphs can diverge just because of sampling.
        if node_ids is None or self.consistency_weight <= 0.0:
            return node_embeddings.new_tensor(0.0), 0
        overlap_indices: List[int] = []
        targets: List[torch.Tensor] = []
        for idx, node_id in enumerate(node_ids):
            if not real_mask[idx]:
                continue
            if node_id in self._memory_bank:
                overlap_indices.append(idx)
                targets.append(self._memory_bank[node_id].to(node_embeddings.device))
        if not overlap_indices:
            return node_embeddings.new_tensor(0.0), 0
        current = node_embeddings[overlap_indices]
        target = torch.stack(targets, dim=0)
        loss = F.mse_loss(current, target, reduction="mean")
        return loss * self.consistency_weight, len(overlap_indices)

    def _update_memory_bank(
        self,
        node_ids: Optional[Sequence[Any]],
        node_embeddings: torch.Tensor,
        real_mask: torch.Tensor,
    ) -> None:
        if node_ids is None:
            return
        for idx, node_id in enumerate(node_ids):
            if not real_mask[idx]:
                continue
            embedding = node_embeddings[idx].detach().cpu()
            if node_id in self._memory_bank:
                old = self._memory_bank[node_id]
                updated = (
                    1 - self.memory_bank_momentum
                ) * old + self.memory_bank_momentum * embedding
                self._memory_bank[node_id] = updated
                self._memory_counts[node_id] = self._memory_counts.get(node_id, 0) + 1
                self._memory_bank.move_to_end(node_id)
            else:
                if len(self._memory_bank) >= self.memory_bank_max_size:
                    self._memory_bank.popitem(last=False)
                    self._memory_counts.popitem(last=False)
                self._memory_bank[node_id] = embedding
                self._memory_counts[node_id] = 1

    def _sparsity_warmup_factor(self) -> float:
        """Linear ramp for L0 penalties to avoid early gate collapse."""
        if self.sparsity_warmup_steps <= 0:
            return 1.0
        step = int(self._global_step.item())
        return min(1.0, (step + 1) / max(1, self.sparsity_warmup_steps))

    def _apply_schedules(self) -> None:
        """Update gate and assignment temperatures according to anneal settings."""
        step = int(self._global_step.item())
        if self.gate_temperature_anneal_steps <= 0:
            gate_temp = self.gate_temperature_end
        else:
            progress = min(1.0, step / max(1, self.gate_temperature_anneal_steps))
            gate_temp = (
                self.gate_temperature_start
                + (self.gate_temperature_end - self.gate_temperature_start) * progress
            )
        self.cluster_gate._gate.temperature = gate_temp
        self.decoder.inter_cluster_gate._gate.temperature = gate_temp

        if self.assignment_temperature_anneal_steps <= 0:
            self.assignment_temperature = self.assignment_temperature_end
        else:
            progress = min(1.0, step / max(1, self.assignment_temperature_anneal_steps))
            self.assignment_temperature = (
                self.assignment_temperature_start
                + (self.assignment_temperature_end - self.assignment_temperature_start)
                * progress
            )

    def _smooth_gate_sample(self, gate_sample: torch.Tensor) -> torch.Tensor:
        """EMA smooth the stochastic gates to reduce frame-to-frame jitter."""
        if self.ema_gate_decay <= 0.0 or not self.training:
            return gate_sample
        ema = self._gate_ema.to(gate_sample.device)
        smoothed = self.ema_gate_decay * gate_sample + (1.0 - self.ema_gate_decay) * ema
        self._gate_ema.copy_(smoothed.detach())
        return smoothed

    def _revive_dead_clusters(
        self, gate_sample: torch.Tensor, cluster_usage: torch.Tensor
    ) -> None:
        """Push logit mass toward rarely-open gates that still receive traffic."""
        if self.revival_gate_threshold <= 0.0:
            return
        mask = (gate_sample < self.revival_gate_threshold) & (
            cluster_usage > self.revival_usage_threshold
        )
        if mask.any():
            with torch.no_grad():
                self.cluster_gate._gate.log_alpha.data[mask] = self.revival_logit

    def _degree_orthogonal_penalty(
        self,
        edge_index: torch.LongTensor,
        node_embeddings: Optional[torch.Tensor],
        real_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Penalize correlation between latent space and node degree."""
        if (
            self.degree_decorrelation_weight <= 0.0
            or node_embeddings is None
            or node_embeddings.numel() == 0
        ):
            zero = (
                node_embeddings.new_tensor(0.0)
                if node_embeddings is not None
                else edge_index.new_zeros((), dtype=torch.float)
            )
            return zero, zero.detach()
        if edge_index.numel() == 0 or real_mask.sum() == 0:
            zero = node_embeddings.new_tensor(0.0)
            return zero, zero.detach()
        deg = (
            degree(edge_index[0], node_embeddings.size(0), dtype=torch.float)
            .to(node_embeddings.device)
            .clamp_min(1.0)
        )
        selected = deg[real_mask]
        if selected.numel() == 0:
            zero = node_embeddings.new_tensor(0.0)
            return zero, zero.detach()
        log_deg = torch.log(selected)
        latents = node_embeddings[real_mask]
        latents_centered = latents - latents.mean(dim=0, keepdim=True)
        log_deg_centered = log_deg - log_deg.mean()
        cross = (latents_centered * log_deg_centered.unsqueeze(-1)).sum(dim=0)
        denom_lat = latents_centered.pow(2).sum(dim=0).clamp_min(self.entropy_eps)
        denom_deg = log_deg_centered.pow(2).sum().clamp_min(self.entropy_eps)
        corr_sq = (cross.pow(2) / (denom_lat * denom_deg)).sum()
        penalty = self.degree_decorrelation_weight * corr_sq
        return penalty, corr_sq.detach()

    # Lightweight virtual node adds graph-level context without scaling sums by
    # |V|, stabilizing deeper message passing on small graphs.
    def _inject_virtual_context(
        self, node_embeddings: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_virtual_node or node_embeddings.numel() == 0:
            return node_embeddings, torch.ones(
                node_embeddings.size(0), dtype=torch.bool, device=node_embeddings.device
            )
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        counts = _scatter_add_1d(
            torch.ones_like(batch, dtype=torch.float), batch, num_graphs
        ).clamp_min(1.0)
        graph_sum = torch.zeros(
            (num_graphs, node_embeddings.size(1)), device=node_embeddings.device
        )
        graph_sum.index_add_(0, batch, node_embeddings)
        graph_mean = graph_sum / counts.unsqueeze(-1)
        node_embeddings = node_embeddings + self.virtual_node_weight * graph_mean[batch]
        real_mask = torch.ones(
            node_embeddings.size(0), dtype=torch.bool, device=node_embeddings.device
        )
        return node_embeddings, real_mask

    def forward(
        self,
        node_types: torch.LongTensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        node_attributes: Optional[List[List[Dict[str, Any]]]] = None,
        edge_type: Optional[torch.LongTensor] = None,
        negative_sampling_ratio: Optional[float] = None,
        positional_encodings: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        node_ids: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        node_embeddings, enc_info = self.encoder(
            node_types=node_types,
            edge_index=edge_index,
            batch=batch,
            node_attributes=node_attributes,
            edge_type=edge_type,
            positional_encodings=positional_encodings,
            edge_weight=edge_weight,
        )
        batch_vec = enc_info.get("batch")
        if batch_vec is None:
            batch_vec = _ensure_batch_vector(
                batch, node_types.size(0), node_types.device
            )
        if self.training:
            self._apply_schedules()
        # Blend in optional virtual node context before assignment balancing.
        node_embeddings, real_mask = self._inject_virtual_context(
            node_embeddings, batch_vec
        )
        gate_sample = self.cluster_gate(training=self.training)
        gate_sample = self._smooth_gate_sample(gate_sample)
        gate_probs = gate_sample.clamp_min(self.entropy_eps)
        gate_probs = gate_probs / gate_probs.sum().clamp_min(self.entropy_eps)
        # Encourage non-degenerate gate usage measured in information-theoretic bits.
        gate_entropy_nats = -(gate_probs * gate_probs.log()).sum()
        gate_entropy_bits = gate_entropy_nats / math.log(2.0)
        gate_entropy_loss = self.gate_entropy_weight * F.relu(
            self.gate_entropy_floor_bits - gate_entropy_bits
        )
        neg_conf_weight: Optional[torch.Tensor] = None
        if self.neg_entropy_scale > 0.0:
            # Down-weight easy negatives when gates already show high confidence.
            inv_entropy = 1.0 / gate_entropy_bits.clamp_min(self.neg_entropy_min_bits)
            neg_conf_weight = 1.0 + self.neg_entropy_scale * (inv_entropy - 1.0)
            neg_conf_weight = neg_conf_weight.clamp_max(self.neg_entropy_max_weight)

        assignments = self._balanced_assignments(
            node_embeddings, gate_sample, real_mask, batch_vec
        )
        if self.training:
            self._update_prototypes(assignments, node_embeddings, real_mask)

        node_ids_list = self._convert_node_ids(node_ids)
        recon_loss, dec_info = self.decoder(
            assignments=assignments,
            edge_index=edge_index,
            batch=batch_vec,
            edge_type=edge_type,
            negative_sampling_ratio=negative_sampling_ratio,
            node_types=node_types,
            negative_confidence_weight=neg_conf_weight,
        )

        cluster_l0 = self.cluster_gate.expected_l0().sum()
        inter_l0 = dec_info["inter_l0"].sum()
        warmup_factor = self._sparsity_warmup_factor()
        cluster_weight = self._base_l0_cluster_weight * warmup_factor
        inter_weight = self._base_l0_inter_weight * warmup_factor
        self.l0_cluster_weight = cluster_weight
        self.l0_inter_weight = inter_weight
        sparsity_loss = cluster_weight * cluster_l0 + inter_weight * inter_l0

        effective_assignments = assignments[real_mask]
        if effective_assignments.numel() == 0:
            effective_assignments = assignments
        assignment_probs = effective_assignments.clamp_min(self.entropy_eps)
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
        # Compute entropy per graph so small graphs don't dominate regularizer.
        entropy_per_node = -(
            assignment_probs * assignment_probs.log().clamp_min(-1e3)
        ).sum(dim=1)
        entropy_graph = _per_graph_mean(
            entropy_per_node,
            batch_vec[real_mask],
            num_graphs,
            eps=self.entropy_eps,
        )
        # Keep per-graph entropy above the target floor so assignments remain diverse.
        entropy_loss = (
            self.entropy_weight
            * F.relu(self.assignment_entropy_floor - entropy_graph).mean()
        )

        cluster_usage = assignment_probs.mean(dim=0)
        # Encourage cluster usage to match the Dirichlet prior via KL divergence
        usage_probs = cluster_usage.clamp_min(self.entropy_eps)
        usage_probs = usage_probs / usage_probs.sum().clamp_min(self.entropy_eps)
        prior_probs = self._dirichlet_prior.to(
            device=usage_probs.device, dtype=usage_probs.dtype
        )
        dirichlet_kl = (usage_probs * (usage_probs.log() - prior_probs.log())).sum()
        dirichlet_loss = self.dirichlet_weight * dirichlet_kl.clamp_min(0.0)
        self._revive_dead_clusters(gate_sample, cluster_usage)

        if node_embeddings is None or node_embeddings.numel() == 0:
            embedding_norm_loss = assignments.new_tensor(0.0)
            kld_loss = assignments.new_tensor(0.0)
            latent_mean = assignments.new_tensor(0.0)
            latent_var = assignments.new_tensor(0.0)
            kld_per_graph = torch.zeros(num_graphs, device=assignments.device)
            embedding_norm_graph = torch.zeros(num_graphs, device=assignments.device)
        else:
            # Penalize encoder outputs per graph mean to avoid bias toward
            # larger components in the batch.
            embed_norm = node_embeddings.pow(2).sum(dim=1)
            embedding_norm_graph = _per_graph_mean(
                embed_norm[real_mask],
                batch_vec[real_mask],
                num_graphs,
                eps=self.entropy_eps,
            )
            embedding_norm_loss = (
                self.embedding_norm_weight * embedding_norm_graph.mean()
            )

            ones = torch.ones(node_embeddings.size(0), device=node_embeddings.device)
            counts = _scatter_add_1d(
                ones[real_mask],
                batch_vec[real_mask],
                num_graphs,
            ).clamp_min(1.0)
            graph_sum = torch.zeros(
                num_graphs, node_embeddings.size(1), device=node_embeddings.device
            )
            graph_sq_sum = torch.zeros_like(graph_sum)
            graph_sum.index_add_(0, batch_vec[real_mask], node_embeddings[real_mask])
            graph_sq_sum.index_add_(
                0,
                batch_vec[real_mask],
                node_embeddings[real_mask] * node_embeddings[real_mask],
            )
            graph_mean = graph_sum / counts.unsqueeze(-1)
            graph_var = graph_sq_sum / counts.unsqueeze(-1) - graph_mean.pow(2)
            graph_var = graph_var.clamp_min(self.entropy_eps)
            kld_per_graph = (
                0.5
                * self.kld_weight
                * (graph_mean.pow(2) + graph_var - graph_var.log() - 1.0)
            ).sum(dim=1)
            kld_loss = kld_per_graph.mean()
            latent_mean = graph_mean.detach()
            latent_var = graph_var.detach()

        degree_penalty, degree_corr = self._degree_orthogonal_penalty(
            edge_index, node_embeddings, real_mask
        )

        consistency_loss, overlap_count = self._memory_consistency_loss(
            node_ids_list, node_embeddings, real_mask
        )

        total_loss = (
            recon_loss
            + sparsity_loss
            + entropy_loss
            + dirichlet_loss
            + embedding_norm_loss
            + kld_loss
            + consistency_loss
            + gate_entropy_loss
            + degree_penalty
        )

        self._update_memory_bank(node_ids_list, node_embeddings, real_mask)

        gate_expectation = self.cluster_gate.expected_l0()
        if gate_expectation.dim() == 0:
            gate_expectation = gate_expectation.unsqueeze(0)
        active_count = (gate_sample >= self.active_gate_threshold).sum().float()
        expected_active = gate_expectation.sum()

        metrics: Dict[str, torch.Tensor] = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
            "cluster_l0": cluster_l0.detach(),
            "inter_l0": inter_l0.detach(),
            "cluster_gate_sample": gate_sample.detach(),
            "inter_gate_sample": dec_info["inter_gate_sample"],
            "entropy_loss": entropy_loss.detach(),
            "assignment_entropy": entropy_graph.mean().detach(),
            "dirichlet_loss": dirichlet_loss.detach(),
            "embedding_norm_loss": embedding_norm_loss.detach(),
            "kld_loss": kld_loss.detach(),
            "cluster_usage": cluster_usage.detach(),
            "latent_mean": latent_mean,
            "latent_var": latent_var,
            "consistency_loss": consistency_loss.detach(),
            "consistency_overlap": torch.tensor(
                overlap_count, device=assignments.device
            ),
            "graph_entropy": entropy_graph.detach(),
            "graph_embedding_norm": embedding_norm_graph.detach(),
            "graph_kld": kld_per_graph.detach(),
            "gate_entropy_loss": gate_entropy_loss.detach(),
            "gate_entropy_bits": gate_entropy_bits.detach(),
            "sparsity_warmup_factor": torch.tensor(
                warmup_factor, device=assignments.device
            ),
            "degree_penalty": degree_penalty.detach(),
            "degree_correlation_sq": degree_corr.detach(),
            "num_active_clusters": active_count.detach(),
            "expected_active_clusters": expected_active.detach(),
        }
        if neg_conf_weight is not None:
            metrics["negative_confidence_weight"] = neg_conf_weight.detach()
        metrics.update({k: v for k, v in dec_info.items() if "loss" in k})
        metrics["graph_edge_loss"] = dec_info.get("graph_losses")
        if self.training:
            self._global_step += 1
        return total_loss, assignments, metrics

    @staticmethod
    def hard_partition(
        assignments: torch.Tensor,
        cluster_gate_values: torch.Tensor,
        gate_threshold: float = 0.5,
        min_cluster_size: int = 1,
    ) -> PartitionResult:
        """Translate latent assignments + gate activations into a discrete partition.

        Args:
            assignments: Soft assignment matrix of shape [num_nodes, num_clusters].
            cluster_gate_values: Gate activations (sample or expectation) of shape
                [num_clusters] or [1, num_clusters].
            gate_threshold: Minimum gate value required to mark a cluster as active.
            min_cluster_size: Optional minimum number of nodes per cluster
                (clusters falling below are suppressed and their nodes reassigned).

        Returns:
            PartitionResult with per-node hard labels and active cluster metadata.
        """

        if assignments.dim() != 2:
            raise ValueError(
                "assignments must be a 2D tensor [num_nodes, num_clusters]."
            )

        gate_vals = cluster_gate_values.squeeze()
        if gate_vals.dim() != 1 or gate_vals.numel() != assignments.size(1):
            raise ValueError(
                "cluster_gate_values must broadcast to [num_clusters] matching assignments."
            )

        # Determine active clusters based on threshold; ensure at least one cluster survives.
        active_mask = gate_vals >= gate_threshold
        if active_mask.sum() == 0:
            # fallback: keep the cluster with the strongest gate value
            active_mask[gate_vals.argmax()] = True

        active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1)
        num_clusters = assignments.size(1)

        masked_assignments = assignments.clone()
        inactive_mask = ~active_mask
        if inactive_mask.any():
            masked_assignments[:, inactive_mask] = torch.finfo(assignments.dtype).min

        max_vals, node_labels = masked_assignments.max(dim=1)

        # Handle rows where all active clusters were suppressed (e.g., numerical issues).
        unresolved = torch.isinf(max_vals)
        if unresolved.any():
            fallback = assignments[unresolved].argmax(dim=1)
            node_labels[unresolved] = fallback

        if min_cluster_size > 1:
            counts = torch.bincount(node_labels, minlength=num_clusters)
            small_clusters = (counts < min_cluster_size) & active_mask
            if small_clusters.any():
                keepers = active_mask.clone()
                keepers[small_clusters] = False
                if keepers.sum() == 0:
                    # fallback: keep the cluster with max gate even if it is small
                    keepers[gate_vals.argmax()] = True
                active_mask = keepers
                active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1)
                masked_assignments = assignments.clone()
                masked_assignments[:, ~active_mask] = torch.finfo(assignments.dtype).min
                _, node_labels = masked_assignments.max(dim=1)

        cluster_members: Dict[int, torch.LongTensor] = {}
        for idx in active_indices.tolist():
            member_idx = torch.nonzero(node_labels == idx, as_tuple=False).view(-1)
            if member_idx.numel() > 0:
                cluster_members[idx] = member_idx

        return PartitionResult(
            node_to_cluster=node_labels.cpu(),
            active_clusters=active_indices.cpu(),
            cluster_members={k: v.cpu() for k, v in cluster_members.items()},
            gate_values=gate_vals.detach().cpu(),
            assignments=assignments.detach().cpu(),
        )


class GraphEncoder(nn.Module):
    """
    Graph encoder with learnable node type embeddings, deep set encoder for node attributes and DAG attention layers.
    Outputs per-graph mu and logvar for VAE.
    """

    def __init__(
        self,
        num_node_types: int,
        attr_encoder: NodeAttributeDeepSetEncoder,
        latent_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.attr_encoder = attr_encoder
        attr_emb_dim = attr_encoder.out_dim
        self.node_type_embedding = nn.Embedding(num_node_types, attr_emb_dim)
        self.convs = nn.ModuleList()
        prev_dim = attr_emb_dim * 2
        for h in hidden_dims:
            self.convs.append(DAGAttention(prev_dim, h))
            prev_dim = h
        # map to latent parameters
        self.lin_mu = nn.Linear(prev_dim, latent_dim)
        self.lin_logvar = nn.Linear(prev_dim, latent_dim)

    @property
    def max_value_dim(self):
        return self.attr_encoder.max_value_dim

    @property
    def shared_attr_vocab(self):
        return self.attr_encoder.shared_attr_vocab

    def forward(self, node_types, edge_index, node_attributes, batch):
        type_embedding = self.node_type_embedding(node_types)
        if len(node_attributes) > 0:
            attr_embedding = torch.stack(
                [
                    self.attr_encoder(attrs)
                    for graph in node_attributes
                    for attrs in graph
                ],
                dim=0,
            )
        else:
            attr_embedding = torch.zeros((len(node_types), self.attr_encoder.out_dim))
        x = torch.cat([type_embedding, attr_embedding], dim=-1)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        mu = self.lin_mu(x)
        logvar = self.lin_logvar(x)
        return mu, logvar

    def prune_latent_dims(self, kept_idx: torch.LongTensor):
        """
        Keep only the latent dims in `kept_idx` for lin_mu and lin_logvar.
        """

        def prune_lin(old):
            layer = nn.Linear(old.in_features, kept_idx.numel()).to(old.weight.device)
            layer.weight.data = old.weight.data[kept_idx]
            layer.bias.data = old.bias.data[kept_idx]
            return layer

        self.lin_mu = prune_lin(self.lin_mu)
        self.lin_logvar = prune_lin(self.lin_logvar)
        self.latent_dim = len(kept_idx)


class AsyncGraphEncoder(GraphEncoder):
    """Graph encoder using AsyncDAGLayer instead of attention."""

    def __init__(
        self,
        num_node_types: int,
        attr_encoder: NodeAttributeDeepSetEncoder,
        latent_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__(num_node_types, attr_encoder, latent_dim, hidden_dims=[])
        self.convs = nn.ModuleList()
        prev_dim = attr_encoder.out_dim * 2
        for h in hidden_dims:
            self.convs.append(AsyncDAGLayer(prev_dim, h))
            prev_dim = h
        self.lin_mu = nn.Linear(prev_dim, latent_dim)
        self.lin_logvar = nn.Linear(prev_dim, latent_dim)


class GraphDeconvNet(MessagePassing):
    """
    A Graph Deconvolutional Network (GDN) layer that acts as the
    transpose/inverse of a GCNConv.  It takes an input signal X of
    size [N, in_channels] on a graph with edge_index, and produces
    an output signal of size [N, out_channels], without any fixed
    max_nodes or feature‐size assumptions.
    """

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        # weight for the "transpose" convolution
        self.lin = nn.Linear(in_channels, out_channels, bias=True)
        # optional bias after aggregation
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, edge_index: torch.LongTensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index: LongTensor of shape [2, E] with COO edges.
            x:          FloatTensor of shape [N, in_channels] node signals.
        Returns:
            FloatTensor of shape [N, out_channels].
        """
        x = self.lin(x)  # (acts like W^T in a transposed convolution)
        # if there are no edges, skip the propagate/unpack step
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return F.relu(x + self.bias)  # just apply bias + activation
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        out = self.propagate(
            edge_index, x=x, norm=deg.pow(0.5)[col] * deg.pow(0.5)[row]
        )  # each node collects from neighbors
        return out + self.bias

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # x_j: neighbor features [E, out_channels]
        # norm: normalization per edge [E]
        return x_j * norm.view(-1, 1)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # optional nonlinearity
        return F.relu(aggr_out)


class SizeBucketBatchSampler(Sampler[List[int]]):
    """Groups graphs with similar node counts into shared mini-batches."""

    def __init__(
        self, sizes: Sequence[int], batch_size: int, shuffle: bool = True
    ) -> None:
        self.sizes = list(sizes)
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.sizes)))
        indices.sort(key=lambda idx: self.sizes[idx])
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.sizes) / self.batch_size)


class NodeBudgetBatchSampler(Sampler[List[int]]):
    """Creates batches capped by a total node budget."""

    def __init__(
        self, sizes: Sequence[int], node_budget: int, shuffle: bool = True
    ) -> None:
        if node_budget <= 0:
            raise ValueError("node_budget must be positive")
        self.sizes = list(sizes)
        self.node_budget = node_budget
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            random.shuffle(indices)
        current_batch: List[int] = []
        current_total = 0
        for idx in indices:
            size = max(1, int(self.sizes[idx]))
            if current_batch and current_total + size > self.node_budget:
                # Stop before exceeding the budget—pending graph will become the first
                # element in the next mini-batch so nothing ever gets dropped. This
                # keeps memory/negative sampling cost bounded when giant graphs appear.
                yield current_batch
                current_batch = []
                current_total = 0
            current_batch.append(idx)
            current_total += size
        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        total = sum(max(1, int(size)) for size in self.sizes)
        return max(1, math.ceil(total / self.node_budget))


class OnlineTrainer:
    """Minimal trainer for the self-compressing rGCN autoencoder."""

    def __init__(
        self,
        model: SelfCompressingRGCNAutoEncoder,
        optimizer,
        device: Optional[str] = None,
    ):
        resolved_device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(resolved_device)
        self.optimizer = optimizer
        self.device = resolved_device
        self.dataset: List[Data] = []
        self.history: List[Dict[str, float]] = []
        # Track node counts per graph so we can bucket by size or enforce budgets.
        self._node_sizes: List[int] = []
        self.early_stop_epoch: Optional[int] = None
        self.early_stop_reason: Optional[str] = None

    def add_data(self, graphs: List[Data]) -> None:
        for graph in graphs:
            clone = graph.clone()
            self.dataset.append(clone)
            node_count = int(
                getattr(clone, "num_nodes", None) or clone.node_types.size(0)
            )
            self._node_sizes.append(node_count)

    def clear_dataset(self) -> None:
        self.dataset.clear()
        self._node_sizes.clear()

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 16,
        negative_sampling_ratio: Optional[float] = None,
        verbose: bool = True,
        bucket_by_size: bool = True,
        node_budget: Optional[int] = None,
        shuffle: bool = True,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
        stability_metric: Optional[str] = None,
        stability_window: int = 0,
        stability_tolerance: float = 0.0,
        stability_relative_tolerance: Optional[float] = None,
        min_epochs: int = 0,
    ) -> List[Dict[str, float]]:
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not self.dataset:
            raise ValueError("No graphs available to train on. Call add_data first.")

        if node_budget is not None:
            # Node-budget sampler keeps total nodes per batch bounded to limit VRAM.
            batch_sampler = NodeBudgetBatchSampler(
                self._node_sizes, node_budget, shuffle=shuffle
            )
            loader = DataLoader(self.dataset, batch_sampler=batch_sampler)
        elif bucket_by_size:
            # Bucket graphs of similar size together to reduce gradient variance.
            batch_sampler = SizeBucketBatchSampler(
                self._node_sizes, batch_size, shuffle=shuffle
            )
            loader = DataLoader(self.dataset, batch_sampler=batch_sampler)
        else:
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(1, epochs + 1):
            self.model.train()
            batch_count = 0
            epoch_loss = 0.0
            metric_sums: Dict[str, float] = {}

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                positional = None
                for candidate in (
                    "positional_encodings",
                    "laplacian_positional_encoding",
                    "laplacian_eigvecs",
                ):
                    # Support multiple field names people use for Laplacian PEs.
                    if hasattr(batch, candidate):
                        positional = getattr(batch, candidate)
                        break
                node_ids = None
                for candidate in (
                    "global_node_ids",
                    "node_ids",
                    "original_node_ids",
                    "node_names",
                ):
                    # Reuse whichever stable node identifier is present for the
                    # overlap consistency regularizer.
                    if hasattr(batch, candidate):
                        node_ids = getattr(batch, candidate)
                        break
                loss, _assignments, metrics = self.model(
                    node_types=batch.node_types,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    node_attributes=getattr(batch, "node_attributes", None),
                    edge_type=getattr(batch, "edge_type", None),
                    negative_sampling_ratio=negative_sampling_ratio,
                    positional_encodings=positional,
                    edge_weight=getattr(batch, "edge_weight", None),
                    node_ids=node_ids,
                )
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                batch_count += 1

                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        if value.dim() == 0:
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(
                                value.detach().cpu().item()
                            )
                    elif isinstance(value, (int, float)):
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            if batch_count == 0:
                raise RuntimeError("Training loader produced zero batches.")

            averaged_metrics = {k: v / batch_count for k, v in metric_sums.items()}
            self.history.append(averaged_metrics)
            if on_epoch_end is not None:
                try:
                    on_epoch_end(epoch, averaged_metrics)
                except Exception:
                    # Ensure training continues even if logging hook fails.
                    pass

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs} loss={averaged_metrics['total_loss']:.4f} metrics={averaged_metrics}"
                )

            if (
                stability_metric
                and stability_window > 0
                and epoch >= max(min_epochs, stability_window)
            ):
                recent = self.history[-stability_window:]
                if len(recent) == stability_window:
                    values: List[float] = []
                    valid = True
                    for record in recent:
                        metric_value = record.get(stability_metric)
                        if metric_value is None:
                            valid = False
                            break
                        values.append(float(metric_value))
                    if valid:
                        span = max(values) - min(values)
                        within_abs = span <= stability_tolerance
                        within_rel = True
                        if stability_relative_tolerance is not None:
                            mean_abs = max(abs(sum(values) / len(values)), 1e-8)
                            within_rel = (
                                span / mean_abs
                            ) <= stability_relative_tolerance
                        if within_abs and within_rel:
                            self.early_stop_epoch = epoch
                            self.early_stop_reason = f"{stability_metric} stable for last {stability_window} epochs"
                            if verbose:
                                print(
                                    f"[EARLY STOP] {self.early_stop_reason} (span={span:.4f})"
                                )
                            break

        return self.history


if __name__ == "__main__":
    from attributes import FloatAttribute, IntAttribute, StringAttribute

    num_node_types = 3
    num_relations = 2
    num_clusters = num_node_types
    attr_name_vocab = [generate_random_string(5) for _ in range(20)]
    shared_attr_vocab = SharedAttributeVocab(attr_name_vocab, embedding_dim=5)
    attr_encoder = NodeAttributeDeepSetEncoder(
        shared_attr_vocab=shared_attr_vocab,
        encoder_hdim=16,
        aggregator_hdim=32,
        out_dim=32,
    )
    model = SelfCompressingRGCNAutoEncoder(
        num_node_types=num_node_types,
        attr_encoder=attr_encoder,
        num_clusters=num_clusters,
        num_relations=num_relations,
        hidden_dims=[64, 64],
        dropout=0.1,
        negative_sampling_ratio=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def generate_random_graph(num_nodes: int, edge_prob: float = 0.3) -> Data:
        node_types = torch.randint(0, num_node_types, (num_nodes,), dtype=torch.long)
        edges: List[List[int]] = []
        edge_types: List[int] = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if random.random() < edge_prob:
                    edges.append([src, dst])
                    edge_types.append(random.randint(0, num_relations - 1))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)

        attributes_per_node = []
        for _ in range(num_nodes):
            attrs: Dict[Any, Any] = {}
            if random.random() < 0.5:
                attrs[IntAttribute(random.choice(attr_name_vocab))] = random.randint(
                    0, 10
                )
            if random.random() < 0.5:
                attrs[FloatAttribute(random.choice(attr_name_vocab))] = (
                    random.random() * 5.0
                )
            if random.random() < 0.5:
                attrs[StringAttribute(random.choice(attr_name_vocab))] = random.choice(
                    attr_name_vocab
                )
            attributes_per_node.append(attrs)

        data = Data(
            node_types=node_types,
            edge_index=edge_index,
            node_attributes=attributes_per_node,
        )
        data.edge_type = edge_type
        return data

    size_buckets = [
        (6, 0.35, 12),  # small graphs: 6+/-2 nodes
        (18, 0.25, 20),  # medium graphs: around 18 nodes
        (36, 0.15, 48),  # large graphs: around 36 nodes
    ]
    graphs: List[Data] = []
    for target_nodes, edge_prob, count in size_buckets:
        low = max(4, int(0.7 * target_nodes))
        high = int(1.3 * target_nodes)
        for _ in range(count):
            graphs.append(
                generate_random_graph(random.randint(low, high), edge_prob=edge_prob)
            )
    random.shuffle(graphs)

    trainer = OnlineTrainer(model, optimizer)
    trainer.add_data(graphs)
    trainer.train(
        epochs=5,
        batch_size=8,
        negative_sampling_ratio=0.5,
        bucket_by_size=True,
        node_budget=160,
    )
