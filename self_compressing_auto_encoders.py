import math
import random
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, RGCNConv, global_mean_pool
from torch_geometric.utils import degree, softmax

from utility import generate_random_string


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
        names_to_add = [name for name in new_names if name not in self.name_to_index]

        starting_index = len(self.name_to_index)
        num_new_names = len(names_to_add)

        for offset, name in enumerate(names_to_add):
            new_idx = starting_index + offset
            self.name_to_index[name] = new_idx
            self.index_to_name[new_idx] = name

        # expand the embedding matrix with He‐style initialization for the new rows
        old_weight = self.embedding.weight.data
        fan_in = old_weight.size(1)
        new_rows = torch.randn(len(names_to_add), fan_in) * math.sqrt(2 / fan_in)
        new_weight = torch.cat([old_weight, new_rows], dim=0)
        self.embedding = nn.Embedding.from_pretrained(new_weight, freeze=False)


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
        if isinstance(value, (int, float)):
            value = torch.tensor([value], dtype=torch.float)
        elif isinstance(value, str):
            index = self.shared_attr_vocab.name_to_index.get(
                value, self.shared_attr_vocab.name_to_index["<UNK>"]
            )
            index = torch.tensor(index, dtype=torch.long)
            value = self.shared_attr_vocab.embedding(index)
        else:
            raise TypeError(f"Unsupported attribute value type: {type(value)}")
        value = value.view(-1)
        if value.numel() < self.max_value_dim:
            pad_amt = self.max_value_dim - value.numel()
            return F.pad(value, (0, pad_amt), "constant", 0.0)
        else:
            return value[: self.max_value_dim]

    def forward(self, attr_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not attr_dict or len(attr_dict) == 0:
            return torch.zeros(self.aggregator[-1].out_features)

        phis = []
        for attr, value in sorted(
            attr_dict.items(), key=lambda i: i[0].name
        ):  # consistent ordering
            name_index = torch.tensor(
                self.shared_attr_vocab.name_to_index[attr.name], dtype=torch.long
            )
            value = self.get_value_tensor(value)
            phis.append(
                self.attr_encoder(
                    torch.cat(
                        [self.shared_attr_vocab.embedding(name_index), value], dim=0
                    )
                )
            )
        return self.aggregator(torch.stack(phis, dim=0).sum(dim=0))


class HardConcreteGate(nn.Module):
    """Louizos et al. (2018) hard-concrete gate for L0-style sparsity."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        temperature: float = 2.0 / 3.0,
        limit_a: float = -0.1,
        limit_b: float = 1.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(shape))
        self.temperature = temperature
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.eps = eps

    def _sample(self, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.log(u + self.eps) - torch.log(1 - u + self.eps)
            z = torch.sigmoid((s + self.log_alpha) / self.temperature)
        else:
            z = torch.sigmoid(self.log_alpha)
        z = z * (self.limit_b - self.limit_a) + self.limit_a
        return z.clamp(0.0, 1.0)

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        training = self.training if training is None else training
        return self._sample(training)

    def expected_l0(self) -> torch.Tensor:
        limit_ratio = -self.limit_a / self.limit_b
        limit_ratio = max(limit_ratio, self.eps)
        threshold = self.temperature * math.log(limit_ratio)
        return torch.sigmoid(self.log_alpha - threshold)


class ClusterGate(nn.Module):
    """Applies hard-concrete gating across clusters."""

    def __init__(self, num_clusters: int, temperature: float = 2.0 / 3.0) -> None:
        super().__init__()
        self._gate = HardConcreteGate((num_clusters,), temperature=temperature)

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
    ) -> None:
        super().__init__()
        self._gate = HardConcreteGate(
            (num_relations, num_clusters, num_clusters), temperature=temperature
        )

    def forward(self, training: Optional[bool] = None) -> torch.Tensor:
        return self._gate(training=training)

    def expected_l0(self) -> torch.Tensor:
        return self._gate.expected_l0()


class RGCNClusterEncoder(nn.Module):
    """Recurrent relational GCN encoder producing node-to-cluster assignments."""

    def __init__(
        self,
        num_node_types: int,
        attr_encoder: NodeAttributeDeepSetEncoder,
        num_clusters: int,
        hidden_dims: Optional[List[int]] = None,
        num_relations: int = 1,
        type_embedding_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]
        self.attr_encoder = attr_encoder
        self.num_clusters = num_clusters
        self.num_relations = max(1, num_relations)
        self.dropout = dropout
        embed_dim = type_embedding_dim or attr_encoder.out_dim
        self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)
        in_dim = embed_dim + attr_encoder.out_dim
        self.convs = nn.ModuleList()
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
            in_dim = hidden_dim
        self.cluster_assign = nn.Linear(in_dim, num_clusters)
        self.cluster_gate = ClusterGate(num_clusters)

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = node_types.device
        type_embedding = self.node_type_embedding(node_types)

        if node_attributes and len(node_attributes) > 0:
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

        x = torch.cat([type_embedding, attr_embedding], dim=-1)
        etype = self._prepare_edge_type(edge_type, edge_index.size(1), device)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, etype))
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        cluster_logits = self.cluster_assign(x)
        gate_sample = self.cluster_gate(training=self.training).unsqueeze(0)
        cluster_logits = cluster_logits + torch.log(gate_sample + 1e-8)
        assignments = F.softmax(cluster_logits, dim=-1)

        info = {
            "node_embeddings": x,
            "cluster_assignments": assignments,
            "cluster_gate_sample": gate_sample.detach(),
            "cluster_l0": self.cluster_gate.expected_l0(),
        }
        if batch is not None:
            info["batch"] = batch
        return assignments, info


class ClusteredGraphReconstructor(nn.Module):
    """Decodes cluster assignments into adjacency logits with L0 gating."""

    def __init__(
        self,
        num_relations: int,
        num_clusters: int,
        negative_sampling_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_relations = max(1, num_relations)
        self.num_clusters = num_clusters
        self.negative_sampling_ratio = max(0.0, negative_sampling_ratio)
        self.inter_cluster_logits = nn.Parameter(
            torch.zeros(self.num_relations, num_clusters, num_clusters)
        )
        nn.init.xavier_uniform_(self.inter_cluster_logits)
        self.inter_cluster_gate = InterClusterGate(self.num_relations, num_clusters)
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
        ratio: float,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if ratio <= 0.0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
            )

        if num_nodes == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros((0,), dtype=torch.long, device=edge_index.device),
            )

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        edge_index_cpu = edge_index.detach().cpu()
        batch_cpu = batch.detach().cpu()
        neg_edges: List[Tuple[int, int]] = []
        neg_types: List[int] = []

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
            while len(seen) < target and attempts < max(100, target * 10):
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

        if not neg_edges:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
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
        return neg_edge_tensor, neg_type_tensor

    def forward(
        self,
        assignments: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        edge_type: Optional[torch.LongTensor] = None,
        negative_sampling_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = assignments.device
        ratio = (
            self.negative_sampling_ratio
            if negative_sampling_ratio is None
            else negative_sampling_ratio
        )
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        weights, gate_sample = self._inter_weights(training=self.training)

        if edge_index.size(1) == 0:
            pos_loss = assignments.new_tensor(0.0)
            pos_logits = torch.empty(0, device=device)
        else:
            pos_logits = self._edge_logits(assignments, edge_index, edge_type, weights)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )

        neg_edges, neg_types = self._sample_negative_edges(
            edge_index, batch, assignments.size(0), ratio
        )
        if neg_edges.size(1) == 0:
            neg_loss = assignments.new_tensor(0.0)
            neg_logits = torch.empty(0, device=device)
        else:
            neg_logits = self._edge_logits(assignments, neg_edges, neg_types, weights)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )

        recon_loss = pos_loss + neg_loss
        info: Dict[str, torch.Tensor] = {
            "pos_loss": pos_loss.detach(),
            "neg_loss": neg_loss.detach(),
            "num_negatives": torch.tensor(neg_edges.size(1), device=device),
            "inter_l0": self.inter_cluster_gate.expected_l0(),
            "inter_gate_sample": gate_sample.detach(),
        }
        if pos_logits.numel() > 0:
            info["pos_logits"] = pos_logits.detach()
        if neg_edges.size(1) > 0:
            info["neg_logits"] = neg_logits.detach()
        return recon_loss, info


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
        l0_cluster_weight: float = 1e-3,
        l0_inter_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder = RGCNClusterEncoder(
            num_node_types=num_node_types,
            attr_encoder=attr_encoder,
            num_clusters=num_clusters,
            hidden_dims=hidden_dims,
            num_relations=num_relations,
            type_embedding_dim=type_embedding_dim,
            dropout=dropout,
        )
        self.decoder = ClusteredGraphReconstructor(
            num_relations=num_relations,
            num_clusters=num_clusters,
            negative_sampling_ratio=negative_sampling_ratio,
        )
        self.l0_cluster_weight = l0_cluster_weight
        self.l0_inter_weight = l0_inter_weight

    def forward(
        self,
        node_types: torch.LongTensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        node_attributes: Optional[List[List[Dict[str, Any]]]] = None,
        edge_type: Optional[torch.LongTensor] = None,
        negative_sampling_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assignments, enc_info = self.encoder(
            node_types=node_types,
            edge_index=edge_index,
            batch=batch,
            node_attributes=node_attributes,
            edge_type=edge_type,
        )
        recon_loss, dec_info = self.decoder(
            assignments=assignments,
            edge_index=edge_index,
            batch=batch,
            edge_type=edge_type,
            negative_sampling_ratio=negative_sampling_ratio,
        )

        cluster_l0 = enc_info["cluster_l0"].sum()
        inter_l0 = dec_info["inter_l0"].sum()
        sparsity_loss = (
            self.l0_cluster_weight * cluster_l0 + self.l0_inter_weight * inter_l0
        )
        total_loss = recon_loss + sparsity_loss

        metrics: Dict[str, torch.Tensor] = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
            "cluster_l0": cluster_l0.detach(),
            "inter_l0": inter_l0.detach(),
            "cluster_gate_sample": enc_info["cluster_gate_sample"],
            "inter_gate_sample": dec_info["inter_gate_sample"],
        }
        metrics.update({k: v for k, v in dec_info.items() if "loss" in k})
        return total_loss, assignments, metrics


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


class GraphDecoder(nn.Module):
    """
    Recurrent generation + GDN refinement.
    1) Recurrently generate node embeddings & edges (GraphRNN style).
    2) Refine embeddings via Graph Deconvolutional Nets (GDNs).
    """

    def __init__(
        self,
        num_node_types: int,
        latent_dim: int,
        shared_attr_vocab: SharedAttributeVocab,
        hidden_dim: int = 128,
        gdn_layers: int = 2,
    ):
        super().__init__()
        self.shared_attr_vocab = shared_attr_vocab
        self.latent_dim = latent_dim
        # — map latent → initial Node‐RNN state
        self.node_hidden_state_init = nn.Linear(latent_dim, hidden_dim)

        # — Node‐RNN (no input, only hidden evolves)
        self.node_rnn = nn.GRUCell(input_size=0, hidden_size=hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.node_head = nn.Linear(hidden_dim, hidden_dim)
        self.type_head = nn.Linear(hidden_dim, num_node_types)
        self.edge_rnn = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.edge_head = nn.Linear(hidden_dim, 1)

        self.attr_type_rnn = nn.GRU(input_size=1, hidden_size=hidden_dim)
        self.attr_type_head = nn.Linear(hidden_dim, 1)
        self.attr_name_rnn = nn.GRU(input_size=1, hidden_size=hidden_dim)
        self.attr_name_head = nn.Linear(
            hidden_dim, shared_attr_vocab.embedding.embedding_dim
        )
        self.attr_dims_head = nn.Linear(
            hidden_dim, 1
        )  # determine num dimensions per attr
        self.attr_val_rnn = nn.GRU(input_size=1, hidden_size=hidden_dim)
        self.attr_val_head = nn.Linear(
            hidden_dim, 1
        )  # extract value per attr dimension

        # — GraphDeconvNet stack (learned spectral decoders)
        self.gdns = nn.ModuleList(
            [GraphDeconvNet(hidden_dim, hidden_dim) for _ in range(gdn_layers)]
        )
        self.max_nodes = 1000
        self.max_attributes_per_node = 50

    def forward(self, latent):
        """
        (num_graphs, latent_dim)
        returns: list of graphs, each {'node_types': LongTensor[1, N],
                                       'node_attributes': List[{name: attribute_value} x N],
                                       'edge_index': LongTensor[2, E]}
        """
        device = latent.device
        all_graphs = []

        for l in range(latent.size(0)):
            hidden_node = F.relu(self.node_hidden_state_init(latent[l])).unsqueeze(
                0
            )  # (hidden_dim,)
            node_embeddings = []
            edges = []
            node_types = []
            t = 0
            while True:
                if t > self.max_nodes:
                    warn("max nodes reached")
                    break
                hidden_node = self.node_rnn(
                    torch.zeros(hidden_node.shape[0], 0, device=device), hidden_node
                )

                # clamp for precision errors
                p_stop = 1 - torch.sigmoid(self.stop_head(hidden_node))
                p_stop = torch.nan_to_num(p_stop, nan=1.0).clamp(0.0, 1.0)
                if torch.bernoulli(p_stop).item() == 0:
                    break
                new_node = self.node_head(hidden_node).squeeze(0)
                node_embeddings.append(new_node)
                node_types.append(
                    self.type_head(new_node).argmax(dim=-1).cpu().tolist()
                )

                # edge generation to previous nodes
                hidden_edge = hidden_node
                edge_in = torch.zeros(1, 1, device=device)
                for i in range(t):
                    hidden_edge = self.edge_rnn(edge_in, hidden_edge)
                    p_edge = torch.sigmoid(self.edge_head(hidden_edge)).view(-1)
                    p_edge = torch.nan_to_num(p_edge, nan=0.0).clamp(0.0, 1.0)
                    if torch.bernoulli(p_edge).item() == 1:
                        edges.append([i, t])
                    edge_in = p_edge.unsqueeze(0)
                t += 1

            if node_embeddings:
                node_embeddings = torch.stack(node_embeddings, dim=0)
                edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                node_embeddings = torch.zeros(
                    (0, self.node_head.out_features), device=device
                )
                edges = torch.zeros((2, 0), dtype=torch.long, device=device)

            # refine via graph deconvolution
            for gdn in self.gdns:
                node_embeddings = gdn(edges, node_embeddings)

            node_attributes = []
            for embedding in node_embeddings:
                attrs = {}
                name_hidden = embedding.unsqueeze(0)
                val_hidden = None
                t = 0
                while True:
                    if t > self.max_attributes_per_node:
                        warn("max attributes per node reached")
                        break
                    name_input = torch.zeros(1, 1, device=device)
                    name_out, name_hidden = self.attr_name_rnn(name_input, name_hidden)
                    similarity_logits = torch.matmul(
                        self.shared_attr_vocab.embedding.weight,  # [vocab_size, embedding_dim]
                        self.attr_name_head(name_out).squeeze(
                            0
                        ),  # query vector [embedding_dim]
                    )
                    name_index = int(similarity_logits.argmax().item())
                    if name_index == self.shared_attr_vocab.name_to_index["<EOS>"]:
                        break
                    elif name_index == self.shared_attr_vocab.name_to_index["<UNK>"]:
                        name_index = len(self.shared_attr_vocab.name_to_index)
                        name = generate_random_string(
                            8
                        )  # TODO: change this to the true attribute name
                        self.shared_attr_vocab.name_to_index[name] = name_index
                        self.shared_attr_vocab.index_to_name[name_index] = name
                    else:
                        name = self.shared_attr_vocab.index_to_name[name_index]

                    attr_dims = max(
                        1,
                        int(
                            math.ceil(F.softplus(self.attr_dims_head(embedding)).item())
                        ),
                    )
                    values = []
                    value_hidden = embedding.unsqueeze(0).unsqueeze(1)
                    value_input = torch.zeros(1, 1, 1, device=device)
                    for _ in range(attr_dims):
                        value_out, value_hidden = self.attr_val_rnn(
                            value_input, value_hidden
                        )
                        v = self.attr_val_head(value_out).view(-1)
                        values.append(v)
                        value_input = v.unsqueeze(0).unsqueeze(-1)
                    if name in attrs:
                        warn(name + " is already defined for currently decoding node")
                    attrs[name] = torch.stack(values)
                    t += 1
                node_attributes.append(attrs)
            all_graphs.append(
                {
                    "node_types": torch.as_tensor(node_types),
                    "node_attributes": node_attributes,
                    "edge_index": edges,
                }
            )
        return all_graphs

    def prune_latent_dims(self, kept_idx: torch.LongTensor):
        """
        Permanently remove unused latent dimensions from the node_hidden_state_init layer.
        kept_idx: 1D LongTensor of indices to keep from the original latent vector.
        """
        device = self.node_hidden_state_init.weight.device
        new_node_hidden_state_init = nn.Linear(
            kept_idx.numel(), self.node_hidden_state_init.out_features, bias=True
        ).to(device)
        new_node_hidden_state_init.weight.data.copy_(old.weight.data[:, kept_idx])

        new_node_hidden_state_init.bias.data.copy_(old.bias.data)
        self.node_hidden_state_init = new_node_hidden_state_init
        self.latent_dim = kept_idx.numel()


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

    def add_data(self, graphs: List[Data]) -> None:
        for graph in graphs:
            self.dataset.append(graph.clone())

    def clear_dataset(self) -> None:
        self.dataset.clear()

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 16,
        negative_sampling_ratio: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not self.dataset:
            raise ValueError("No graphs available to train on. Call add_data first.")

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            batch_count = 0
            epoch_loss = 0.0
            metric_sums: Dict[str, float] = {}

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                loss, _assignments, metrics = self.model(
                    node_types=batch.node_types,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    node_attributes=batch.node_attributes,
                    edge_type=getattr(batch, "edge_type", None),
                    negative_sampling_ratio=negative_sampling_ratio,
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
            averaged_metrics["loss"] = epoch_loss / batch_count
            self.history.append(averaged_metrics)

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs}  loss={averaged_metrics['loss']:.4f}  metrics={averaged_metrics}"
                )

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

    graphs = [
        generate_random_graph(random.randint(5, 12), edge_prob=0.25) for _ in range(64)
    ]
    trainer = OnlineTrainer(model, optimizer)
    trainer.add_data(graphs)
    trainer.train(epochs=5, batch_size=8)
