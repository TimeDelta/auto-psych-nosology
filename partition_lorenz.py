import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Lorenz curve and Gini coefficient for a partition.json file."
    )
    parser.add_argument(
        "partition_json",
        type=Path,
        help="Path to partition JSON exported by train_rgcn_scae.py",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=None,
        help="Optional CSV output for the Lorenz curve coordinates.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to save a Lorenz curve plot (PNG).",
    )
    return parser.parse_args()


def load_cluster_sizes(partition_path: Path) -> List[int]:
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    data = json.loads(partition_path.read_text())
    cluster_members = data.get("cluster_member_ids") or {}
    sizes = [len(members) for members in cluster_members.values()]

    if not sizes:
        node_to_cluster = data.get("node_to_cluster") or {}
        if node_to_cluster:
            counts = Counter(node_to_cluster.values())
            sizes = list(counts.values())

    if not sizes:
        raise ValueError(
            "Could not derive cluster sizes from partition; missing 'cluster_member_ids'"
            " or 'node_to_cluster'."
        )
    return sizes


def lorenz_curve(values: List[int]) -> Tuple[List[float], List[float]]:
    if not values:
        raise ValueError("No values provided for Lorenz curve computation.")
    sorted_vals = sorted(max(0, v) for v in values)
    total = sum(sorted_vals)
    if total == 0:
        raise ValueError("All cluster sizes are zero.")

    cumulative = [0.0]
    running = 0
    for val in sorted_vals:
        running += val
        cumulative.append(running / total)

    n = len(sorted_vals)
    fractions = [i / n for i in range(0, n + 1)]
    return fractions, cumulative


def gini_coefficient(fractions: List[float], cumulative: List[float]) -> float:
    area = 0.0
    for i in range(1, len(fractions)):
        dx = fractions[i] - fractions[i - 1]
        avg_height = (cumulative[i] + cumulative[i - 1]) / 2
        area += avg_height * dx
    return 1.0 - 2.0 * area


def maybe_write_csv(
    csv_path: Path, fractions: List[float], cumulative: List[float]
) -> None:
    if csv_path is None:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("fraction_clusters,fraction_nodes\n")
        for x, y in zip(fractions, cumulative):
            handle.write(f"{x:.10f},{y:.10f}\n")


def maybe_plot(
    plot_path: Path, fractions: List[float], cumulative: List[float]
) -> None:
    if plot_path is None:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fractions, cumulative, label="Lorenz curve", color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Equality line")
    ax.set_xlabel("Fraction of clusters")
    ax.set_ylabel("Fraction of nodes")
    ax.set_title("Cluster Size Lorenz Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sizes = load_cluster_sizes(args.partition_json)
    fractions, cumulative = lorenz_curve(sizes)
    gini = gini_coefficient(fractions, cumulative)

    maybe_write_csv(args.curve_csv, fractions, cumulative)
    maybe_plot(args.plot, fractions, cumulative)

    print(
        f"Clusters: {len(sizes)} | Nodes: {sum(sizes)} | "
        f"Gini coefficient: {gini:.6f}"
    )


if __name__ == "__main__":
    main()
