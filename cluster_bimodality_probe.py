#!/usr/bin/env python3
"""
cluster_bimodality_probe_v2.py

Like v1, but:
- Works fully WITHOUT external HiTOP labels.
- Adds internal stability diagnostics:
  * Pairwise NMI/ARI between partitions across (checkpoint × τ × θ)
  * Within-K vs cross-K stability summaries
  * Optional consensus matrix on a node sample to quantify co-assignment stability

USAGE (NPZ mode):
  python cluster_bimodality_probe_v2.py \
      --npz_glob "runs/ckpts/*.npz" \
      --outdir analysis/cluster_bimodality_v2

Optional external labels (generic):
  --labels_csv path/to/labels.csv \
  --labels_id_col node_id \
  --labels_cols labelA labelB ...
The script will compute NMI/ARI/VI against each provided column if node_ids exist in NPZ.
(If you previously used HiTOP tiers, you can pass those columns here.)

NPZ expected keys (any subset):
  - Z : float32 [N, K]   -- per-node cluster logits (REQUIRED)
  - H : float32 [N, D]   -- latent embeddings (optional; entropy diagnostic)
  - gates : float32 [K]  -- gate probabilities (optional)
  - recon_loss : float scalar (optional)
  - node_ids : int64 [N] -- required only if you want external label alignment
"""

import argparse
import glob
import math
import os
import sys
from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# --------------------------- Utilities ---------------------------------


def softmax_stable(logits: np.ndarray, tau: float = 1.0, axis: int = -1) -> np.ndarray:
    x = logits / float(tau)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def entropy_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe), axis=1)


def realized_assignments(
    Z: np.ndarray, tau: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    p = softmax_stable(Z, tau=tau, axis=1)
    return np.argmax(p, axis=1), p


def count_active_clusters(
    assignments: np.ndarray, gates: Optional[np.ndarray], gate_threshold: float
) -> int:
    uniq = np.unique(assignments)
    if gates is None:
        return uniq.size
    alive = np.where(gates > gate_threshold)[0]
    return np.intersect1d(uniq, alive).size


def expected_L0_from_gates(gates: Optional[np.ndarray]) -> Optional[float]:
    if gates is None:
        return None
    return float(np.sum(gates))


def realized_L0_from_gates(
    gates: Optional[np.ndarray], gate_threshold: float
) -> Optional[float]:
    if gates is None:
        return None
    return float(np.sum((gates > gate_threshold).astype(np.float32)))


def variation_of_information(
    labels_true: np.ndarray, labels_pred: np.ndarray, eps: float = 1e-12
) -> float:
    true = labels_true.astype(int)
    pred = labels_pred.astype(int)
    n = true.shape[0]
    if n == 0:
        return np.nan
    # contingency
    true_vals, true_inv = np.unique(true, return_inverse=True)
    pred_vals, pred_inv = np.unique(pred, return_inverse=True)
    contingency = np.zeros((true_vals.size, pred_vals.size), dtype=np.float64)
    for i in range(n):
        contingency[true_inv[i], pred_inv[i]] += 1.0
    contingency /= n
    p_true = contingency.sum(axis=1, keepdims=True)
    p_pred = contingency.sum(axis=0, keepdims=True)
    H_true = -np.sum(p_true * np.log(np.clip(p_true, eps, 1.0)))
    H_pred = -np.sum(p_pred * np.log(np.clip(p_pred, eps, 1.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = contingency / (p_true @ p_pred)
        frac = np.clip(frac, eps, None)
        I = np.sum(contingency * np.log(frac))
    return float(H_true + H_pred - 2.0 * I)


def optional_float(value: Optional[np.ndarray]) -> Optional[float]:
    """Best-effort conversion of scalars stored in NPZ (often 0-d arrays)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[-1]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def crude_mdl_proxy(
    K: int,
    recon_loss: Optional[float],
    n_nodes: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Optional[float]:
    recon_loss_val = optional_float(recon_loss)
    if recon_loss_val is None:
        return None
    return alpha * (K * math.log(max(n_nodes, 2))) + beta * recon_loss_val


@dataclass
class SweepRow:
    checkpoint: str
    tau: float
    gate_threshold: float
    n_nodes: int
    K_potential: int
    realized_active_clusters: int
    expected_L0: Optional[float]
    realized_L0: Optional[float]
    recon_loss: Optional[float]
    mdl_proxy: Optional[float]
    mean_assignment_entropy: float
    mean_latent_entropy: Optional[float]
    # store assignments for later stability analysis (saved separately to disk to avoid huge memory)
    assignments_path: str


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_labels_csv(
    path: Optional[str], id_col: str, label_cols: List[str]
) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    df = pd.read_csv(path)
    missing = set([id_col] + label_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")
    return df[[id_col] + label_cols].copy()


def align_external_labels(
    npz_dict: Dict[str, np.ndarray],
    labels_df: Optional[pd.DataFrame],
    id_col: str,
    label_cols: List[str],
) -> Optional[pd.DataFrame]:
    if labels_df is None or "node_ids" not in npz_dict:
        return None
    node_ids = npz_dict["node_ids"].reshape(-1)
    df_nodes = pd.DataFrame({id_col: node_ids, "_row": np.arange(node_ids.shape[0])})
    merged = (
        df_nodes.merge(labels_df, on=id_col, how="left")
        .sort_values("_row")
        .drop(columns=["_row"])
    )
    return merged  # may contain NaNs


def compute_alignment_metrics_df(
    labels_aligned_df: pd.DataFrame, pred_labels: np.ndarray, label_cols: List[str]
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    out = {}
    for col in label_cols:
        lab = labels_aligned_df[col].to_numpy()
        mask = ~pd.isna(lab)
        if mask.sum() == 0:
            out[col] = (None, None, None)
            continue
        y_true = lab[mask].astype(int)
        y_pred = pred_labels[mask].astype(int)
        out[col] = (
            float(normalized_mutual_info_score(y_true, y_pred)),
            float(adjusted_rand_score(y_true, y_pred)),
            float(variation_of_information(y_true, y_pred)),
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Probe bimodality in realized K with internal stability diagnostics."
    )
    grp_in = parser.add_argument_group("Inputs")
    grp_in.add_argument(
        "--npz_glob",
        type=str,
        required=True,
        help="Glob for NPZ files containing Z (N×K); optional: gates (K,), H (N×D), recon_loss, node_ids.",
    )
    grp_in.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Optional CSV with external labels to align to node_ids.",
    )
    grp_in.add_argument(
        "--labels_id_col",
        type=str,
        default="node_id",
        help="ID column name in labels_csv to match NPZ node_ids.",
    )
    grp_in.add_argument(
        "--labels_cols",
        type=str,
        nargs="*",
        default=[],
        help="One or more label column names in labels_csv to evaluate against.",
    )

    grp_sw = parser.add_argument_group("Sweeps")
    grp_sw.add_argument(
        "--taus", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    )
    grp_sw.add_argument(
        "--gate_thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7]
    )

    grp_out = parser.add_argument_group("Outputs")
    grp_out.add_argument("--outdir", type=str, required=True)
    grp_out.add_argument("--alpha", type=float, default=1.0)
    grp_out.add_argument("--beta", type=float, default=1.0)
    grp_out.add_argument(
        "--consensus_sample",
        type=int,
        default=5000,
        help="Max #nodes to sample for consensus matrix (to save memory).",
    )
    grp_out.add_argument(
        "--max_pairs",
        type=int,
        default=5000,
        help="Max #pairs of partitions to compare for stability (randomly sampled).",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    parts_dir = os.path.join(args.outdir, "partitions")
    os.makedirs(parts_dir, exist_ok=True)

    # Load labels if provided
    labels_df = (
        load_labels_csv(args.labels_csv, args.labels_id_col, args.labels_cols)
        if args.labels_csv
        else None
    )

    npz_paths = sorted(glob.glob(args.npz_glob))
    if not npz_paths:
        print(f"No files matched {args.npz_glob}", file=sys.stderr)
        sys.exit(1)

    rows: List[SweepRow] = []
    # For stability analysis later
    assignment_records = []  # (assignments_path, realized_active_clusters, n_nodes)

    for npz_path in npz_paths:
        npz_dict = load_npz(npz_path)
        if "Z" not in npz_dict:
            print(
                f"[WARN] {npz_path} missing 'Z'; skipping.",
                file=sys.stderr,
            )
            continue
        Z = npz_dict["Z"].astype(np.float64)
        N, K = Z.shape
        H = npz_dict.get("H", None)
        gates = npz_dict.get("gates", None)
        recon_loss = optional_float(npz_dict.get("recon_loss", None))

        # External labels aligned to this NPZ (optional)
        labels_aligned_df = None
        if labels_df is not None and "node_ids" in npz_dict:
            labels_aligned_df = align_external_labels(
                npz_dict, labels_df, args.labels_id_col, args.labels_cols
            )

        # Latent entropy proxy
        mean_latent_entropy = None
        if H is not None:
            H_std = (H - H.mean(axis=0, keepdims=True)) / (
                H.std(axis=0, keepdims=True) + 1e-8
            )
            H_p = softmax_stable(H_std, tau=1.0, axis=1)
            latent_ent = entropy_rows(H_p)
            mean_latent_entropy = float(np.mean(latent_ent))

        base = os.path.basename(npz_path)

        for tau in args.taus:
            y, p = realized_assignments(Z, tau=tau)
            assign_ent = entropy_rows(p)
            mean_assign_ent = float(np.mean(assign_ent))

            for thr in args.gate_thresholds:
                K_real = count_active_clusters(y, gates, gate_threshold=thr)
                E_L0 = expected_L0_from_gates(gates)
                R_L0 = realized_L0_from_gates(gates, thr)
                mdl = crude_mdl_proxy(
                    K_real, recon_loss, n_nodes=N, alpha=args.alpha, beta=args.beta
                )

                # Save assignments to disk (npz per partition)
                part_name = f"{base}_tau{tau:.2f}_thr{thr:.2f}.npz"
                part_path = os.path.join(parts_dir, part_name)
                np.savez_compressed(part_path, y=y.astype(np.int32))
                assignment_records.append((part_path, int(K_real), N))

                rows.append(
                    SweepRow(
                        checkpoint=base,
                        tau=float(tau),
                        gate_threshold=float(thr),
                        n_nodes=N,
                        K_potential=K,
                        realized_active_clusters=int(K_real),
                        expected_L0=E_L0,
                        realized_L0=R_L0,
                        recon_loss=recon_loss,
                        mdl_proxy=mdl,
                        mean_assignment_entropy=mean_assign_ent,
                        mean_latent_entropy=mean_latent_entropy,
                        assignments_path=part_path,
                    )
                )

    # Save primary summary
    if rows:
        df = pd.DataFrame([asdict(r) for r in rows])
    else:
        df = pd.DataFrame(columns=[f.name for f in dataclass_fields(SweepRow)])
        summary_csv = os.path.join(args.outdir, "summary.csv")
        df.to_csv(summary_csv, index=False)
        print(
            "[WARN] No NPZ files contained the required 'Z' logits. "
            f"Wrote empty summary to {summary_csv} and exiting.",
            file=sys.stderr,
        )
        return
    summary_csv = os.path.join(args.outdir, "summary.csv")
    df.to_csv(summary_csv, index=False)

    # Histogram of K
    plt.figure()
    plt.hist(
        df["realized_active_clusters"].astype(int),
        bins=range(
            int(df["realized_active_clusters"].min()),
            int(df["realized_active_clusters"].max()) + 2,
        ),
    )
    plt.xlabel("realized_active_clusters")
    plt.ylabel("Count")
    plt.title("Histogram of realized K")
    plt.savefig(
        os.path.join(args.outdir, "hist_realized_active_clusters.png"),
        bbox_inches="tight",
    )
    plt.close()

    # MDL plot if available
    if df["mdl_proxy"].notna().any():
        agg_mdl = (
            df.groupby("realized_active_clusters")["mdl_proxy"].min().reset_index()
        )
        plt.figure()
        plt.plot(agg_mdl["realized_active_clusters"], agg_mdl["mdl_proxy"], marker="o")
        plt.xlabel("realized_active_clusters")
        plt.ylabel("MDL proxy (lower is better)")
        plt.title("Crude MDL proxy vs realized_active_clusters")
        plt.savefig(os.path.join(args.outdir, "mdl_plot.png"), bbox_inches="tight")
        plt.close()

    # -------- Internal stability: pairwise NMI/ARI on random sample of partitions --------
    rng = np.random.default_rng(12345)
    # sample up to max_pairs distinct pairs
    M = len(assignment_records)
    max_pairs = min(args.max_pairs, M * (M - 1) // 2)
    pairs = set()
    while len(pairs) < max_pairs and len(pairs) < M * (M - 1) // 2:
        i, j = rng.integers(0, M, size=2)
        if i >= j:
            continue
        pairs.add((i, j))
    pairs = list(pairs)

    stability_rows = []
    for i, j in pairs:
        path_i, K_i, N_i = assignment_records[i]
        path_j, K_j, N_j = assignment_records[j]
        if N_i != N_j:
            # skip differing node sets (could add alignment by node_ids if present)
            continue
        yi = np.load(path_i)["y"]
        yj = np.load(path_j)["y"]
        nmi = normalized_mutual_info_score(yi, yj)
        ari = adjusted_rand_score(yi, yj)
        vi = variation_of_information(yi, yj)
        stability_rows.append(
            {"K_i": K_i, "K_j": K_j, "NMI": nmi, "ARI": ari, "VI": vi}
        )

    stab_df = pd.DataFrame(stability_rows)
    stab_csv = os.path.join(args.outdir, "stability_pairs.csv")
    stab_df.to_csv(stab_csv, index=False)

    # Summaries: within-K vs cross-K
    def summarize_pairs(metric: str) -> pd.DataFrame:
        same = stab_df[stab_df["K_i"] == stab_df["K_j"]]
        diff = stab_df[stab_df["K_i"] != stab_df["K_j"]]
        rows = []
        if not same.empty:
            rows.append(
                {
                    "type": "within_K",
                    "metric": metric,
                    "mean": same[metric].mean(),
                    "std": same[metric].std(),
                    "n": len(same),
                }
            )
        if not diff.empty:
            rows.append(
                {
                    "type": "cross_K",
                    "metric": metric,
                    "mean": diff[metric].mean(),
                    "std": diff[metric].std(),
                    "n": len(diff),
                }
            )
        return pd.DataFrame(rows)

    sum_nmi = summarize_pairs("NMI")
    sum_ari = summarize_pairs("ARI")
    sum_vi = summarize_pairs("VI")
    pd.concat([sum_nmi, sum_ari, sum_vi]).to_csv(
        os.path.join(args.outdir, "stability_summary.csv"), index=False
    )

    # -------- Consensus matrix on a node sample --------
    # Choose a node sample size S
    # Load first assignment to get N
    if assignment_records:
        first_y = np.load(assignment_records[0][0])["y"]
        N = first_y.shape[0]
        S = min(args.consensus_sample, N)
        idx = rng.choice(N, size=S, replace=False)
        # Build consensus co-assignment matrix: fraction of partitions co-assigning each pair
        # Store only a scalar summary to avoid huge files: average on-diagonal (1) vs off-diagonal mean
        # and an image of the consensus matrix (SxS)
        consensus = np.zeros((S, S), dtype=np.float32)
        for path, _, _ in assignment_records:
            y = np.load(path)["y"][idx]
            # For co-assignment: build equality matrix
            eq = (y[:, None] == y[None, :]).astype(np.float32)
            consensus += eq
        consensus /= len(assignment_records)
        # Save a heatmap
        plt.figure()
        plt.imshow(consensus, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Co-assignment fraction")
        plt.title("Consensus matrix (sampled nodes)")
        plt.xlabel("Sampled nodes")
        plt.ylabel("Sampled nodes")
        plt.savefig(
            os.path.join(args.outdir, "consensus_heatmap.png"), bbox_inches="tight"
        )
        plt.close()

        # Scalar summary: average within inferred clusters per partition vs across clusters
        # We'll compute a partition of the consensus using threshold 0.5 co-assignment as a crude proxy
        # and report the mean difference
        within = consensus[consensus >= 0.5]
        between = consensus[consensus < 0.5]
        with open(os.path.join(args.outdir, "consensus_summary.txt"), "w") as f:
            f.write(
                f"Consensus matrix built from {len(assignment_records)} partitions on {S} sampled nodes.\n"
            )
            f.write(
                f"Mean co-assignment (>=0.5): {float(within.mean()) if within.size else float('nan'):.4f}\n"
            )
            f.write(
                f"Mean co-assignment (< 0.5): {float(between.mean()) if between.size else float('nan'):.4f}\n"
            )

    # ------------- External labels alignment (generic) -------------------
    if labels_df is not None and "node_ids" in load_npz(npz_paths[0]):
        # For each partition compare against each label column
        align_rows = []
        # Build aligned labels DF using the first NPZ (assume same node order across NPZs)
        npz0 = load_npz(npz_paths[0])
        aligned = align_external_labels(
            npz0, labels_df, args.labels_id_col, args.labels_cols
        )
        if aligned is not None:
            for part_path, K_real, N in assignment_records:
                y = np.load(part_path)["y"]
                for col in args.labels_cols:
                    lab = aligned[col].to_numpy()
                    mask = ~pd.isna(lab)
                    if mask.sum() == 0:
                        continue
                    y_true = lab[mask].astype(int)
                    y_pred = y[mask].astype(int)
                    align_rows.append(
                        {
                            "partition": os.path.basename(part_path),
                            "realized_active_clusters": K_real,
                            "label_col": col,
                            "NMI": normalized_mutual_info_score(y_true, y_pred),
                            "ARI": adjusted_rand_score(y_true, y_pred),
                            "VI": variation_of_information(y_true, y_pred),
                        }
                    )
        pd.DataFrame(align_rows).to_csv(
            os.path.join(args.outdir, "external_alignment.csv"), index=False
        )

    # Final console hints
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {stab_csv}, stability_summary.csv")
    print(
        f"Figures in {args.outdir}: hist_realized_active_clusters.png, mdl_plot.png (if recon_loss), consensus_heatmap.png (if built)"
    )


if __name__ == "__main__":
    main()
