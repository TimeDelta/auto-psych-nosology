import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distribution of node counts per cluster for the SBM partition."
        )
    )
    parser.add_argument(
        "--partition-json",
        type=Path,
        default=Path("sbm_partitions.json"),
        help="Partition JSON exported by train_rgcn_scae.py (defaults to sbm_partitions.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sbm_out/sbm_cluster_size_hist.png"),
        help="Where to save the histogram image.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins (applied in log space).",
    )
    return parser.parse_args()


def load_cluster_sizes(partition_path: Path) -> list[int]:
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


def main() -> None:
    args = parse_args()
    sizes = load_cluster_sizes(args.partition_json)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        sizes, bins=args.bins, edgecolor="black", linewidth=0.6, alpha=0.85, log=True
    )
    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (# nodes)")
    ax.set_ylabel("Number of clusters (log)")
    ax.set_title("SBM Cluster Size Distribution")
    ax.grid(True, which="both", axis="x", linestyle=":", linewidth=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    median_size = statistics.median(sizes)
    max_size = max(sizes)
    print(
        f"Saved histogram for {len(sizes)} clusters to {args.output} "
        f"(median size = {median_size:.1f}, max = {max_size})"
    )


if __name__ == "__main__":
    main()
