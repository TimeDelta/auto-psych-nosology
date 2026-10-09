"""Track hard-partition membership independently of latent gate counts."""

from collections import deque
from typing import Dict

import numpy as np
from sklearn.metrics import adjusted_rand_score


def summarize_partition(node_labels: np.ndarray) -> Dict[str, float]:
    """Report occupied clusters and imbalance for an ordered node universe."""
    node_labels = np.asarray(node_labels)
    if node_labels.ndim != 1 or node_labels.size == 0:
        raise ValueError("node_labels must be a non-empty one-dimensional array")
    _, cluster_sizes = np.unique(node_labels, return_counts=True)
    cluster_fractions = cluster_sizes / node_labels.size
    partition_entropy = float(-(cluster_fractions * np.log(cluster_fractions)).sum())
    return {
        "realized_active_clusters": float(cluster_sizes.size),
        "largest_cluster_fraction": float(cluster_fractions.max()),
        "partition_entropy": partition_entropy,
        "effective_num_clusters": float(np.exp(partition_entropy)),
        "partition_collapsed": float(cluster_sizes.size < 2),
    }


class PartitionStabilityTracker:
    """Compare fixed-order labels across a trailing window, ignoring label IDs.

    Every partition in the window must meet the minimum occupied-cluster count.
    The latest partition must also agree with every earlier partition in that
    window, so small consecutive changes cannot hide accumulated membership drift.
    Storage is O(window * number of evaluated node occurrences).
    """

    def __init__(
        self, window: int, minimum_ari: float = 0.99, minimum_clusters: int = 2
    ):
        if window < 0:
            raise ValueError("window must be non-negative")
        if not np.isfinite(minimum_ari) or not -1.0 <= minimum_ari <= 1.0:
            raise ValueError("minimum_ari must be finite and between -1 and 1")
        if minimum_clusters < 1:
            raise ValueError("minimum_clusters must be at least 1")
        self.window = window
        self.minimum_ari = minimum_ari
        self.minimum_clusters = minimum_clusters
        # A one-epoch window still needs an observed membership comparison.
        self._partitions = deque(maxlen=max(2, window))
        self._cluster_counts = deque(maxlen=max(2, window))
        self.is_stable = False

    def update(self, node_labels: np.ndarray) -> Dict[str, float]:
        metrics = summarize_partition(node_labels)
        self.is_stable = False
        if self.window == 0:
            return metrics

        node_labels = np.asarray(node_labels)
        if self._partitions and node_labels.shape != self._partitions[-1].shape:
            raise ValueError("the evaluated node universe changed within the window")
        self._partitions.append(node_labels.copy())
        self._cluster_counts.append(metrics["realized_active_clusters"])

        if len(self._partitions) >= 2:
            agreement_scores = [
                float(adjusted_rand_score(previous_labels, node_labels))
                for previous_labels in list(self._partitions)[:-1]
            ]
            metrics["partition_ari"] = agreement_scores[-1]
            metrics["partition_window_min_ari"] = min(agreement_scores)
            self.is_stable = (
                len(self._partitions) >= max(2, self.window)
                and min(self._cluster_counts) >= self.minimum_clusters
                and metrics["partition_window_min_ari"] >= self.minimum_ari
            )
        metrics["partition_stability_ready"] = float(self.is_stable)
        return metrics
