import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from partition_stability import PartitionStabilityTracker, summarize_partition
from self_compressing_auto_encoders import OnlineTrainer, SelfCompressingRGCNAutoEncoder
from train_rgcn_scae import _build_argparser


def test_constant_count_does_not_hide_changing_memberships():
    tracker = PartitionStabilityTracker(window=3)
    for node_labels in ([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]):
        metrics = tracker.update(np.array(node_labels))
    assert metrics["realized_active_clusters"] == 2
    assert metrics["partition_window_min_ari"] < 0.99
    assert not tracker.is_stable


def test_membership_comparison_ignores_cluster_label_permutations():
    tracker = PartitionStabilityTracker(window=3)
    for node_labels in ([0, 0, 1, 1], [7, 7, 4, 4], [1, 1, 0, 0]):
        metrics = tracker.update(np.array(node_labels))
    assert metrics["partition_ari"] == 1.0
    assert metrics["partition_window_min_ari"] == 1.0
    assert tracker.is_stable


def test_window_comparison_detects_accumulated_drift():
    tracker = PartitionStabilityTracker(window=3, minimum_ari=0.95)
    node_labels = np.repeat([0, 1], 100)
    tracker.update(node_labels)
    node_labels[[0, 100]] = [1, 0]
    tracker.update(node_labels)
    node_labels[[1, 101]] = [1, 0]
    metrics = tracker.update(node_labels)
    assert metrics["partition_ari"] > 0.95
    assert metrics["partition_window_min_ari"] < 0.95
    assert not tracker.is_stable


@pytest.mark.parametrize("minimum_clusters, expected_stable", [(2, False), (1, True)])
def test_collapsed_partition_requires_explicit_opt_in(
    minimum_clusters, expected_stable
):
    tracker = PartitionStabilityTracker(window=3, minimum_clusters=minimum_clusters)
    for _ in range(3):
        metrics = tracker.update(np.zeros(4, dtype=int))
    assert metrics["partition_collapsed"] == 1.0
    assert metrics["partition_ari"] == 1.0
    assert metrics["largest_cluster_fraction"] == 1.0
    assert metrics["effective_num_clusters"] == 1.0
    assert tracker.is_stable == expected_stable


def test_entire_window_must_meet_minimum_cluster_count():
    tracker = PartitionStabilityTracker(window=3, minimum_ari=-1.0)
    for node_labels in ([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]):
        tracker.update(np.array(node_labels))
    assert not tracker.is_stable
    tracker.update(np.array([0, 0, 1, 1]))
    assert tracker.is_stable


def test_one_epoch_window_requires_an_observed_comparison():
    tracker = PartitionStabilityTracker(window=1)
    tracker.update(np.array([0, 0, 1, 1]))
    assert not tracker.is_stable
    tracker.update(np.array([0, 0, 1, 1]))
    assert tracker.is_stable


def test_disabled_stability_still_reports_imbalance():
    tracker = PartitionStabilityTracker(window=0)
    metrics = tracker.update(np.array([0, 0, 0, 1]))
    assert metrics["largest_cluster_fraction"] == 0.75
    assert 1.0 < metrics["effective_num_clusters"] < 2.0
    assert "partition_ari" not in metrics
    assert not tracker.is_stable


def test_tracker_rejects_changed_node_count():
    tracker = PartitionStabilityTracker(window=2)
    tracker.update(np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="node universe changed"):
        tracker.update(np.array([0, 1]))


@pytest.mark.parametrize("node_labels", [[], [[0, 1]]])
def test_summary_requires_nonempty_vector(node_labels):
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        summarize_partition(np.array(node_labels))


@pytest.mark.parametrize(
    "configuration",
    [
        {"window": -1},
        {"window": 2, "minimum_ari": float("nan")},
        {"window": 2, "minimum_ari": 1.1},
        {"window": 2, "minimum_clusters": 0},
    ],
)
def test_tracker_rejects_invalid_configuration(configuration):
    with pytest.raises(ValueError):
        PartitionStabilityTracker(**configuration)


class ScriptedPartitionModel(torch.nn.Module):
    """Run the real trainer using deterministic epoch-specific soft assignments."""

    hard_partition = staticmethod(SelfCompressingRGCNAutoEncoder.hard_partition)

    def __init__(self, assignment_sequence, gate_values=None):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.1))
        self.assignment_sequence = [
            torch.tensor(values, dtype=torch.float32) for values in assignment_sequence
        ]
        self.num_clusters = self.assignment_sequence[0].shape[1]
        self.active_gate_threshold = 0.5
        self.training_steps = 0
        self.register_buffer(
            "gate_values",
            torch.tensor(
                gate_values if gate_values is not None else [1.0] * self.num_clusters
            ),
        )

    def cluster_gate(self, training=False):
        return self.gate_values

    def forward(self, node_types, **kwargs):
        if self.training:
            self.training_steps += 1
        sequence_index = min(
            max(0, self.training_steps - 1), len(self.assignment_sequence) - 1
        )
        assignments = self.assignment_sequence[sequence_index].to(node_types.device)
        loss = self.parameter.square() + 1.0
        return loss, assignments, {"total_loss": loss.detach()}


def make_trainer(assignment_sequence, gate_values=None):
    model = ScriptedPartitionModel(assignment_sequence, gate_values)
    trainer = OnlineTrainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), device="cpu"
    )
    node_count = len(assignment_sequence[0])
    trainer.add_data(
        [
            Data(
                node_types=torch.zeros(node_count, dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                num_nodes=node_count,
            )
        ]
    )
    return trainer


def soft_assignments(node_labels):
    return [[0.9, 0.1] if label == 0 else [0.1, 0.9] for label in node_labels]


def test_trainer_does_not_stop_on_constant_count_with_changing_memberships():
    trainer = make_trainer(
        [soft_assignments(labels) for labels in ([0, 0, 1, 1], [0, 1, 0, 1]) * 3]
    )
    trainer.train(
        max_epochs=6,
        stability_metric="realized_active_clusters",
        stability_window=3,
        verbose=False,
    )
    assert trainer.early_stop_epoch is None
    assert len(trainer.history) == 6
    assert all(record["realized_active_clusters"] == 2.0 for record in trainer.history)


def test_trainer_marks_collapse_and_continues_to_epoch_budget():
    trainer = make_trainer([soft_assignments([0, 0, 0, 0])])
    trainer.train(
        max_epochs=5,
        stability_metric="realized_active_clusters",
        stability_window=3,
        verbose=False,
    )
    assert trainer.early_stop_epoch is None
    assert len(trainer.history) == 5
    assert trainer.history[-1]["partition_collapsed"] == 1.0


def test_trainer_stops_on_stable_memberships_after_minimum_epoch():
    trainer = make_trainer([soft_assignments([0, 0, 1, 1])])
    trainer.train(
        max_epochs=8,
        stability_metric="realized_active_clusters",
        stability_window=3,
        min_epochs=5,
        verbose=False,
    )
    assert trainer.early_stop_epoch == 5
    assert "partition membership stable" in trainer.early_stop_reason


def test_resumed_trainer_requires_fresh_memberships_and_absolute_minimum_epoch():
    trainer = make_trainer([soft_assignments([0, 0, 1, 1])])
    trainer.history = [{"realized_active_clusters": 2.0}] * 10
    trainer.train(
        max_epochs=8,
        start_epoch=10,
        min_epochs=15,
        stability_metric="realized_active_clusters",
        stability_window=3,
        verbose=False,
    )
    assert trainer.early_stop_epoch == 15
    assert "partition_ari" not in trainer.history[10]


def test_diagnostic_masks_inactive_gates_like_export():
    assignments = [[0.1, 0.9], [0.8, 0.2]]
    trainer = make_trainer([assignments], gate_values=[1.0, 0.1])
    diagnostic_labels = trainer._compute_realized_partition(min_cluster_size=1)
    exported = SelfCompressingRGCNAutoEncoder.hard_partition(
        torch.tensor(assignments), torch.tensor([1.0, 0.1])
    )
    assert torch.equal(diagnostic_labels, exported.node_to_cluster)
    assert diagnostic_labels.tolist() == [0, 0]
    assert trainer.model.training


def test_diagnostic_reassigns_undersized_clusters_like_export():
    assignments = soft_assignments([0, 0, 0, 1])
    trainer = make_trainer([assignments])
    diagnostic_labels = trainer._compute_realized_partition(min_cluster_size=2)
    exported = SelfCompressingRGCNAutoEncoder.hard_partition(
        torch.tensor(assignments), torch.ones(2), min_cluster_size=2
    )
    assert torch.equal(diagnostic_labels, exported.node_to_cluster)
    assert trainer._compute_realized_clusters(min_cluster_size=2) == 1


def test_hard_partition_accepts_single_cluster_capacity():
    partition = SelfCompressingRGCNAutoEncoder.hard_partition(
        torch.ones(3, 1), torch.ones(1)
    )
    assert partition.node_to_cluster.tolist() == [0, 0, 0]


def test_cli_exposes_partition_stability_controls():
    arguments = _build_argparser().parse_args(
        [
            "toy.graphml",
            "--partition-stability-min-ari",
            "0.95",
            "--partition-stability-min-clusters",
            "3",
        ]
    )
    assert arguments.partition_stability_min_ari == 0.95
    assert arguments.partition_stability_min_clusters == 3
