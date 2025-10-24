"""
Alignment of learned clusters with HiTOP and RDoC, plus reports.

Features
--------
- Loads GraphML graph and partition.json (node_to_cluster)
- Auto-infers HiTOP/RDoC labels from node attributes & relations (no manual maps)
- Propagation depth knob for RDoC diffusion across edges
- Alignment metrics: NMI/AMI/ARI/H/C/V + confusion + hypergeometric enrichment (BH-FDR)
- Per-cluster alignment table (precision/recall/F1/purity/overlap-rate for top enriched label)
- Semantic coherence via SentenceTransformers (CPU), with caching
- Coherence robustness (bootstrap CI) + size-aware weighting + explain CSV (medoid & neighbors)
- Global correlations between coherence and alignment quality
- Top-enrichment report (Markdown + CSV)
- Scatter-matrix ready CSVs (wide + long)

Usage (example)
---------------
python align_partitions.py \
  --graph /mnt/data/prebuilt-kg.graphml \
  --partition /mnt/data/partition.json \
  --outdir out \
  --prop-depth 1 \
  --coherence sentence \
  --embed-model all-MiniLM-L6-v2 \
  --embed-cache /mnt/data/emb_cache.npz \
  --coh-bootstrap 500 --coh-ci 0.95 \
  --size-weighting log1p --min-cluster-size 3 \
  --neighbors-k 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score as AMI,
    adjusted_rand_score as ARI,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score as NMI,
)
from sklearn.metrics.pairwise import cosine_similarity


def load_partition(partition_path: Path) -> dict[str, int]:
    with open(partition_path, "r") as f:
        data = json.load(f)
    node_to_cluster = data.get("node_to_cluster")
    if node_to_cluster is None:
        raise ValueError("partition.json missing `node_to_cluster`")
    return {str(k): int(v) for k, v in node_to_cluster.items()}


def load_graph(graph_path: Path) -> nx.Graph:
    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes})
    return G


def align_partition_to_graph(
    G: nx.Graph,
    node_to_cluster: dict[str, int],
) -> tuple[dict[str, int], list[str],]:
    """Align partition entries directly to graph node IDs, recording misses."""
    aligned: dict[str, int] = {}
    missing: list[str] = []

    for raw_key, cluster_id in node_to_cluster.items():
        key = str(raw_key)
        if key in G:
            aligned[key] = int(cluster_id)
        else:
            missing.append(key)

    return aligned, missing


def load_mapping_csv(csv_path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in (id_col, label_col):
        if col not in df.columns:
            raise ValueError(f"CSV {csv_path} must contain column `{col}`")
    df = df[[id_col, label_col]].dropna()
    df[id_col] = df[id_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def metrics_against_labels(
    part_series: pd.Series, label_series: pd.Series
) -> tuple[dict, pd.Index, np.ndarray, np.ndarray, dict]:
    common = part_series.index.intersection(label_series.index)
    y_pred = part_series.loc[common].astype(int).to_numpy()
    y_true_labels = label_series.loc[common].astype(str).to_numpy()
    unique_labels = {lab: i for i, lab in enumerate(pd.unique(y_true_labels))}
    y_true = np.array([unique_labels[lab] for lab in y_true_labels], dtype=int)
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    out = {
        "n_common_nodes": int(len(common)),
        "n_unique_true_labels": int(len(unique_labels)),
        "n_unique_pred_clusters": int(pd.unique(y_pred).size),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "nmi": float(NMI(y_true, y_pred, average_method="arithmetic")),
        "ami": float(AMI(y_true, y_pred, average_method="arithmetic")),
        "ari": float(ARI(y_true, y_pred)),
    }
    return out, common, y_true, y_pred, unique_labels


def cluster_label_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, label_map: dict
) -> pd.DataFrame:
    """Return a label (rows) x cluster (columns) contingency table."""
    clusters = np.sort(pd.unique(y_pred))
    columns = pd.Index([f"c{c}" for c in clusters], name="cluster")

    if y_true.size == 0 or y_pred.size == 0 or not label_map:
        return pd.DataFrame(
            index=pd.Index([], name="label"), columns=columns, dtype=int
        )

    id_to_label = {lid: name for name, lid in label_map.items()}
    label_order = [id_to_label[lid] for lid in sorted(id_to_label)]

    true_labels = pd.Series(y_true, name="true").map(id_to_label)
    pred_clusters = pd.Series(y_pred, name="cluster").astype(int)

    contingency = pd.crosstab(true_labels, pred_clusters, dropna=False)
    contingency = contingency.reindex(index=label_order, columns=clusters, fill_value=0)
    contingency.columns = columns
    contingency.index.name = "label"
    return contingency


def hypergeom_enrichment(assignments: pd.Series, labels: pd.Series) -> pd.DataFrame:
    common = assignments.index.intersection(labels.index)
    a = assignments.loc[common]
    l = labels.loc[common]
    N = len(common)
    label_counts = Counter(l)
    clus_counts = Counter(a)
    records = []
    for c_id, c_size in clus_counts.items():
        c_nodes = set(a.index[a == c_id])
        for lab, K in label_counts.items():
            k = sum(1 for n in c_nodes if l[n] == lab)
            p = hypergeom.sf(k - 1, N, K, c_size)
            records.append(
                {
                    "cluster": c_id,
                    "label": lab,
                    "cluster_size": c_size,
                    "label_size": K,
                    "overlap": k,
                    "enrichment_p": float(p),
                }
            )
    cols = [
        "cluster",
        "label",
        "cluster_size",
        "label_size",
        "overlap",
        "enrichment_p",
        "fdr_bh",
    ]
    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("enrichment_p").reset_index(drop=True)
    m = len(df)
    df["fdr_bh"] = (df["enrichment_p"] * m / (np.arange(m) + 1)).clip(upper=1.0)
    return df.sort_values(["cluster", "enrichment_p", "label"]).reset_index(drop=True)


def _norm(s):
    return str(s).lower().strip()


def _contains_any(text: str, subs: list[str]) -> bool:
    t = _norm(text)
    return any(sub in t for sub in subs)


def infer_framework_labels_tailored(
    G: nx.Graph, prop_depth: int = 1
) -> tuple[pd.Series, pd.Series]:
    """Auto-infer HiTOP & RDoC from node attributes + edges; control RDoC propagation depth."""
    HITOP_RULES = [
        (
            ["schizo", "psychosis", "halluc", "delusion", "mania", "manic"],
            "Thought Disorder",
        ),
        (["depress", "anhedon", "dysthym"], "Internalizing"),
        (["anx", "phobi", "panic", "ptsd", "worry", "ruminat"], "Internalizing"),
        (["somat", "functional neurolog", "conversion"], "Somatoform"),
        (
            ["adhd", "impuls", "substance", "alcohol", "nicotin"],
            "Disinhibited Externalizing",
        ),
        (
            ["oppositional defiant", "conduct", "antisocial", "anger", "aggress"],
            "Antagonistic Externalizing",
        ),
        (["withdraw", "detachment", "asocial"], "Detachment"),
        (["obsess", "compuls"], "Obsessive/Compulsive"),
    ]
    RDOC_RULES = [
        (["fear", "threat", "loss", "stress", "avoid"], "Negative Valence"),
        (["reward", "approach", "motivat", "reinforc"], "Positive Valence"),
        (
            [
                "memory",
                "working memory",
                "attention",
                "executive",
                "language",
                "cognitive",
                "cognition",
            ],
            "Cognitive Systems",
        ),
        (
            ["empathy", "attachment", "face", "theory of mind", "social"],
            "Social Processes",
        ),
        (
            ["sleep", "circadian", "arousal", "autonomic", "homeostasis"],
            "Arousal/Regulatory Systems",
        ),
        (["motor", "sensorimotor", "movement"], "Sensorimotor Systems"),
    ]

    def map_by_rules(name: str, rules):
        for keys, lab in rules:
            if _contains_any(name, keys):
                return lab
        return None

    node_type = nx.get_node_attributes(G, "node_type")
    node_name = nx.get_node_attributes(G, "name")

    disease_nodes = [n for n, t in node_type.items() if t == "disease"]
    phenotype_nodes = [n for n, t in node_type.items() if t == "effect/phenotype"]

    hitop, rdoc = {}, {}

    # 1) direct labels from disease/phenotype names
    for n in disease_nodes + phenotype_nodes:
        nm = node_name.get(n, n)
        lab = map_by_rules(nm, HITOP_RULES)
        if lab:
            hitop[n] = lab
        rlab = map_by_rules(nm, RDOC_RULES)
        if rlab:
            rdoc[n] = rlab

    # 2) build adjacency by relation
    disease_to_phen = defaultdict(list)
    disease_to_gene = defaultdict(list)
    disease_to_drug = defaultdict(list)
    for u, v, data in G.edges(data=True):
        r = data.get("relation", "")
        nu, nv = node_type.get(u), node_type.get(v)
        if r == "disease_phenotype_positive":
            d, p = (u, v) if nu == "disease" else (v, u)
            if node_type.get(d) == "disease" and node_type.get(p) == "effect/phenotype":
                disease_to_phen[d].append(p)
        elif r == "disease_protein":
            d, g = (u, v) if nu == "disease" else (v, u)
            if node_type.get(d) == "disease" and node_type.get(g) == "gene/protein":
                disease_to_gene[d].append(g)
        elif r in ("indication", "off-label use"):
            d, dr = (u, v) if nu == "disease" else (v, u)
            if node_type.get(d) == "disease" and node_type.get(dr) == "drug":
                disease_to_drug[d].append(dr)

    # 3) RDoC propagation (depth >=1)
    if prop_depth >= 1:
        # disease from phenotype votes
        for d in disease_nodes:
            labs = [rdoc[p] for p in disease_to_phen.get(d, []) if p in rdoc]
            if labs:
                maj = Counter(labs).most_common(1)[0][0]
                rdoc.setdefault(d, maj)
        # genes/drugs from diseases
        gene_votes, drug_votes = defaultdict(list), defaultdict(list)
        for d in disease_nodes:
            if d in rdoc:
                for g in disease_to_gene.get(d, []):
                    gene_votes[g].append(rdoc[d])
                for dr in disease_to_drug.get(d, []):
                    drug_votes[dr].append(rdoc[d])
        for g, labs in gene_votes.items():
            if labs:
                rdoc[g] = Counter(labs).most_common(1)[0][0]
        for dr, labs in drug_votes.items():
            if labs:
                rdoc[dr] = Counter(labs).most_common(1)[0][0]

    # 4) optional 2-hop
    if prop_depth >= 2:
        next_votes = defaultdict(list)
        for g, lab in list(rdoc.items()):
            if node_type.get(g) == "gene/protein":
                for nbr in G.neighbors(g):
                    if nbr not in rdoc and node_type.get(nbr) in (
                        "drug",
                        "gene/protein",
                    ):
                        next_votes[nbr].append(lab)
        for n, labs in next_votes.items():
            if labs:
                rdoc[n] = Counter(labs).most_common(1)[0][0]

    # 5) defaults
    for n in disease_nodes + phenotype_nodes:
        if n not in hitop:
            hitop[n] = "Unspecified Clinical"

    return (
        pd.Series(hitop, name="hitop_label", dtype="string"),
        pd.Series(rdoc, name="rdoc_label", dtype="string"),
    )


def _node_text(G: nx.Graph, n: str, field: str) -> str:
    val = G.nodes[n].get(field)
    if val is None:
        val = G.nodes[n].get("name", str(n))
    return str(val)


def _hash_key(model_name: str, field: str, texts: Dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(field.encode())
    for k in sorted(texts.keys()):
        h.update(k.encode())
        h.update(texts[k].encode())
    return h.hexdigest()


def _load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _encode_texts_batched(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    E = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )
    return np.array(E, dtype=np.float32)


def _get_or_build_embeddings(
    G: nx.Graph,
    nodes: list[str],
    field: str,
    model_name: str,
    batch_size: int,
    cache_path: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    node_texts = {n: _node_text(G, n, field) for n in nodes}
    content_hash = _hash_key(model_name, field, node_texts)
    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=True)
            if z["content_hash"].item() == content_hash:
                E = z["embeddings"]
                order = list(z["order"])
                idx_map = {nid: i for i, nid in enumerate(order)}
                take = [idx_map[n] for n in nodes]
                return E[take], {n: i for i, n in enumerate(nodes)}
        except Exception:
            pass
    model = _load_sentence_model(model_name)
    texts_in_order = [node_texts[n] for n in nodes]
    E = _encode_texts_batched(model, texts_in_order, batch_size=batch_size)
    if cache_path:
        np.savez_compressed(
            cache_path,
            embeddings=E,
            order=np.array(nodes, dtype=object),
            content_hash=np.array(content_hash, dtype=object),
        )
    return E, {n: i for i, n in enumerate(nodes)}


def _pairwise_cosines_dense(E: np.ndarray) -> np.ndarray:
    S = cosine_similarity(E)
    n = S.shape[0]
    if n <= 1:
        return np.array([])
    iu = np.triu_indices(n, k=1)
    return S[iu]


def _bootstrap_ci(
    vals: np.ndarray,
    iters: int = 500,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
):
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(42)
    n = vals.size
    bs = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n, endpoint=False)
        bs[i] = vals[idx].mean()
    alpha = 1.0 - ci
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def _size_weight(mean_val: float, n_nodes: int, scheme: str = "log1p"):
    if np.isnan(mean_val):
        return float("nan")
    if scheme == "none":
        return mean_val
    if scheme == "log1p":
        return mean_val * np.log1p(n_nodes)
    return mean_val


def _cluster_medoid(E: np.ndarray) -> int:
    if E.shape[0] <= 1:
        return 0
    S = cosine_similarity(E)
    return int(np.argmax(S.mean(axis=1)))


def compute_cluster_coherence_embeddings(
    G: nx.Graph,
    node_to_cluster: Dict[str, int],
    field: str,
    model_name: str,
    batch_size: int,
    cache_path: str,
    min_cluster_size: int = 2,
    bs_iters: int = 500,
    ci_level: float = 0.95,
    size_weighting: str = "log1p",
    neighbors_k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_to_cluster = {n: int(c) for n, c in node_to_cluster.items() if n in G}
    all_nodes = list(node_to_cluster.keys())
    if not all_nodes:
        return pd.DataFrame(), pd.DataFrame()
    E_all, idx_map = _get_or_build_embeddings(
        G, all_nodes, field, model_name, batch_size, cache_path
    )

    clus_to_nodes = defaultdict(list)
    for n, c in node_to_cluster.items():
        clus_to_nodes[c].append(n)

    coh_rows, expl_rows = [], []
    max_n = 0
    for c, nodes in sorted(clus_to_nodes.items()):
        n_nodes = len(nodes)
        max_n = max(max_n, n_nodes)
        idxs = [idx_map[n] for n in nodes]
        E = E_all[idxs]
        if n_nodes < min_cluster_size:
            coh_rows.append(
                {
                    "cluster": c,
                    "n_nodes": n_nodes,
                    "mean": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "count_pairs": 0,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "mean_size_weighted": np.nan,
                    "mean_size_weighted_norm": np.nan,
                }
            )
            continue
        vals = _pairwise_cosines_dense(E)
        mean = float(vals.mean()) if vals.size else np.nan
        median = float(np.median(vals)) if vals.size else np.nan
        p25 = float(np.percentile(vals, 25)) if vals.size else np.nan
        p75 = float(np.percentile(vals, 75)) if vals.size else np.nan
        ci_lo, ci_hi = _bootstrap_ci(vals, iters=bs_iters, ci=ci_level)
        mean_sw = _size_weight(mean, n_nodes, size_weighting)
        coh_rows.append(
            {
                "cluster": c,
                "n_nodes": n_nodes,
                "mean": mean,
                "median": median,
                "p25": p25,
                "p75": p75,
                "count_pairs": int(vals.size),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mean_size_weighted": mean_sw,
                "mean_size_weighted_norm": np.nan,
            }
        )
        # explainability: medoid + neighbors
        m_idx = _cluster_medoid(E)
        m_node = nodes[m_idx]
        sims = cosine_similarity(E[m_idx : m_idx + 1], E).ravel()
        order = np.argsort(-sims)
        top = order[: min(n_nodes, neighbors_k + 1)]
        for rank, j in enumerate(top):
            expl_rows.append(
                {
                    "cluster": c,
                    "rank": rank,
                    "node_id": nodes[j],
                    "name": _node_text(G, nodes[j], field),
                    "cosine_to_medoid": float(sims[j]),
                    "is_medoid": bool(j == m_idx),
                    "medoid_id": m_node,
                    "medoid_name": _node_text(G, m_node, field),
                }
            )

    coh_df = pd.DataFrame(coh_rows).sort_values("cluster").reset_index(drop=True)
    if max_n > 0 and size_weighting == "log1p":
        norm = np.log1p(max_n)
        if norm > 0:
            coh_df["mean_size_weighted_norm"] = coh_df["mean_size_weighted"] / norm
    expl_df = (
        pd.DataFrame(expl_rows).sort_values(["cluster", "rank"]).reset_index(drop=True)
    )
    return coh_df, expl_df


def per_cluster_alignment(
    part_s: pd.Series, labels_s: pd.Series, enrich_df: pd.DataFrame
) -> pd.DataFrame:
    common = part_s.index.intersection(labels_s.index)
    if len(common) == 0:
        return pd.DataFrame()
    if "fdr_bh" in enrich_df.columns:
        top = (
            enrich_df.sort_values(["cluster", "fdr_bh", "enrichment_p"])
            .groupby("cluster", as_index=False)
            .first()
        )
    else:
        top = (
            enrich_df.sort_values(["cluster", "overlap"], ascending=[True, False])
            .groupby("cluster", as_index=False)
            .first()
        )
    rows = []
    for _, r in top.iterrows():
        c = int(r["cluster"])
        L = r["label"]
        overlap = int(r["overlap"])
        c_size = int(r["cluster_size"])
        support = int((labels_s.loc[common] == L).sum())
        purity = overlap / c_size if c_size > 0 else float("nan")
        precision = purity
        recall = overlap / support if support > 0 else float("nan")
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision > 0 and recall > 0)
            else 0.0
        )
        overlap_rate = overlap / max(1, (c_size + support - overlap))
        rows.append(
            {
                "cluster": c,
                "top_label": L,
                "cluster_size": c_size,
                "support": support,
                "overlap": overlap,
                "purity": purity,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "overlap_rate": overlap_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def correlate_coherence_alignment(
    coh_df: pd.DataFrame, align_df: pd.DataFrame, framework_name: str, out_dir: Path
) -> tuple[pd.DataFrame, Path]:
    if coh_df.empty or align_df.empty:
        p = out_dir / f"global_corr_{framework_name.lower()}.csv"
        return pd.DataFrame(), p
    merged = align_df.merge(coh_df, on="cluster", how="left")
    x_cols = ["mean", "ci_lo", "ci_hi", "mean_size_weighted", "mean_size_weighted_norm"]
    y_cols = ["f1", "purity", "precision", "recall", "overlap_rate"]
    rows = []
    for x in x_cols:
        if x not in merged.columns:
            continue
        xv = merged[x].astype(float)
        for y in y_cols:
            yv = merged[y].astype(float)
            mask = ~(np.isnan(xv) | np.isnan(yv))
            xv2, yv2 = xv[mask], yv[mask]
            if len(xv2) >= 3:
                pr, pp = pearsonr(xv2, yv2)
                sr, sp = spearmanr(xv2, yv2)
            else:
                pr = pp = sr = sp = float("nan")
            rows.append(
                {
                    "framework": framework_name.upper(),
                    "x_metric": x,
                    "y_metric": y,
                    "n_clusters": int(mask.sum()),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                }
            )
    out = pd.DataFrame(rows)
    out_path = out_dir / f"global_corr_{framework_name.lower()}.csv"
    out.to_csv(out_path, index=False)
    return out, out_path


def generate_summary_report(out_dir: Path):
    coh_path = out_dir / "cluster_coherence.csv"
    coh_df = pd.read_csv(coh_path) if coh_path.exists() else pd.DataFrame()
    lines = ["# Top Enriched Clusters\n"]
    report_rows = []
    for name in ("hitop", "rdoc"):
        enrich_file = out_dir / f"enrichment_{name}.csv"
        if not enrich_file.exists():
            continue
        df = pd.read_csv(enrich_file)
        sig = df[df["fdr_bh"] < 0.05].copy()
        if sig.empty:
            continue
        tops = (
            sig.sort_values(["fdr_bh", "enrichment_p", "cluster"])
            .groupby("cluster", as_index=False)
            .first()[
                [
                    "cluster",
                    "label",
                    "overlap",
                    "cluster_size",
                    "fdr_bh",
                    "enrichment_p",
                ]
            ]
            .rename(
                columns={
                    "label": f"{name}_label",
                    "fdr_bh": f"{name}_fdr",
                    "enrichment_p": f"{name}_p",
                }
            )
        )
        if not coh_df.empty:
            tops = tops.merge(coh_df, on="cluster", how="left")
        tops["framework"] = name.upper()
        report_rows.append(tops)
        lines.append(f"## {name.upper()}\n")
        for _, r in tops.iterrows():
            coh_bits = []
            if "mean" in r and not pd.isna(r["mean"]):
                coh_bits.append(f"mean={r['mean']:.3f}")
            if "ci_lo" in r and not pd.isna(r["ci_lo"]):
                coh_bits.append(f"CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
            if "mean_size_weighted_norm" in r and not pd.isna(
                r["mean_size_weighted_norm"]
            ):
                coh_bits.append(f"size-w(norm)={r['mean_size_weighted_norm']:.3f}")
            coh_str = (", " + ", ".join(coh_bits)) if coh_bits else ""
            lines.append(
                f"- Cluster {int(r['cluster']):>3}: **{r[f'{name}_label']}** "
                f"(overlap={int(r['overlap'])}, FDR={r[f'{name}_fdr']:.3g}{coh_str})"
            )
        lines.append("")
    if report_rows:
        all_df = pd.concat(report_rows, ignore_index=True)
        all_df.to_csv(out_dir / "top_enriched_clusters.csv", index=False)
        (out_dir / "top_enriched_clusters.md").write_text("\n".join(lines))


def write_global_corr_md(out_dir: Path):
    md_lines = ["# Global Coherence ↔ Alignment Correlations\n"]
    written_any = False
    for name in ("hitop", "rdoc"):
        p = out_dir / f"global_corr_{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        written_any = True
        top = (
            df.dropna(subset=["spearman_r"])
            .assign(abs_sr=lambda d: d["spearman_r"].abs())
            .sort_values("abs_sr", ascending=False)
            .head(8)
        )
        md_lines.append(f"## {name.upper()}\n")
        for _, r in top.iterrows():
            md_lines.append(
                f"- {r['x_metric']} ↔ {r['y_metric']}: "
                f"Spearman r={r['spearman_r']:.3f} (p={r['spearman_p']:.3g}), "
                f"Pearson r={r['pearson_r']:.3f} (p={r['pearson_p']:.3g}), "
                f"n={int(r['n_clusters'])}"
            )
        md_lines.append("")
    if written_any:
        (out_dir / "global_coherence_alignment.md").write_text("\n".join(md_lines))


def export_scatter_data(
    coh_df: pd.DataFrame, align_df: pd.DataFrame, framework_name: str, out_dir: Path
):
    if coh_df.empty or align_df.empty:
        return
    fw = framework_name.upper()
    merged = align_df.merge(coh_df, on="cluster", how="left")
    coh_cols = [
        "mean",
        "ci_lo",
        "ci_hi",
        "mean_size_weighted",
        "mean_size_weighted_norm",
        "n_nodes",
    ]
    align_cols = [
        "f1",
        "purity",
        "precision",
        "recall",
        "overlap_rate",
        "cluster_size",
        "support",
        "overlap",
    ]
    coh_keep = [c for c in coh_cols if c in merged.columns]
    align_keep = [c for c in align_cols if c in merged.columns]
    wide_cols = ["cluster"] + coh_keep + align_keep
    merged[wide_cols].to_csv(
        out_dir / f"scatter_wide_{framework_name.lower()}.csv", index=False
    )
    long_rows = []
    for _, r in merged.iterrows():
        cl = int(r["cluster"])
        for c in coh_keep:
            long_rows.append(
                {
                    "cluster": cl,
                    "metric": c,
                    "value": float(r[c]),
                    "kind": "coherence",
                    "framework": fw,
                }
            )
        for a in align_keep:
            long_rows.append(
                {
                    "cluster": cl,
                    "metric": a,
                    "value": float(r[a]),
                    "kind": "alignment",
                    "framework": fw,
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / f"scatter_long_{framework_name.lower()}.csv", index=False)
    combined_path = out_dir / "scatter_long_all.csv"
    if combined_path.exists():
        prev = pd.read_csv(combined_path)
        pd.concat([prev, long_df], ignore_index=True).to_csv(combined_path, index=False)
    else:
        long_df.to_csv(combined_path, index=False)


# ----------------- Main pipeline -----------------


def run_alignment(
    graph_path: Path,
    partition_path: Path,
    out_dir: Path,
    hitop_csv: Path | None = None,
    rdoc_csv: Path | None = None,
    id_col: str = "node_id",
    hitop_col: str = "hitop_label",
    rdoc_col: str = "rdoc_label",
    prop_depth: int = 1,
    coherence: str = "sentence",
    coherence_field: str = "name",
    embed_model: str = "all-MiniLM-L6-v2",
    embed_batch_size: int = 256,
    embed_cache: str = "emb_cache.npz",
    coh_bootstrap: int = 500,
    coh_ci: float = 0.95,
    size_weighting: str = "log1p",
    min_cluster_size: int = 2,
    neighbors_k: int = 5,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph(graph_path)
    node_to_cluster_raw = load_partition(partition_path)
    node_to_cluster, missing_nodes = align_partition_to_graph(G, node_to_cluster_raw)

    if missing_nodes:
        preview = ", ".join(missing_nodes[:5])
        if len(missing_nodes) > 5:
            preview += ", ..."
        print(
            f"[align_partitions] Warning: {len(missing_nodes)} partition keys missing from graph (sample: {preview})."
        )

    if not node_to_cluster:
        raise ValueError("No partition entries align to graph nodes.")

    part_s = pd.Series(node_to_cluster, name="cluster", dtype="int64")

    # coherence (embeddings)
    coh_df, expl_df = compute_cluster_coherence_embeddings(
        G=G,
        node_to_cluster=part_s.to_dict(),
        field=coherence_field,
        model_name=embed_model,
        batch_size=embed_batch_size,
        cache_path=embed_cache,
        min_cluster_size=min_cluster_size,
        bs_iters=coh_bootstrap,
        ci_level=coh_ci,
        size_weighting=size_weighting,
        neighbors_k=neighbors_k,
    )
    if coh_df.empty:
        raise ValueError(
            "Cluster coherence computation returned no rows; ensure `--coherence` is set "
            "to a mode that produces embeddings and that clusters meet `--min-cluster-size`."
        )

    coh_df.to_csv(out_dir / "cluster_coherence.csv", index=False)
    expl_df.to_csv(out_dir / "cluster_coherence_explain.csv", index=False)

    summary_lines = []

    def do_framework(name: str, csv_path: Path | None, label_col: str):
        nonlocal summary_lines
        # get labels
        if csv_path is None:
            auto_hitop, auto_rdoc = infer_framework_labels_tailored(
                G, prop_depth=prop_depth
            )
            labels_s = auto_hitop if name == "HiTOP" else auto_rdoc
            labels_s.rename("label").to_csv(
                out_dir / f"inferred_{name.lower()}_map.csv", header=True
            )
        else:
            df = load_mapping_csv(csv_path, id_col=id_col, label_col=label_col)
            df = df[df[id_col].isin(G.nodes)]
            labels_s = df.set_index(id_col)[label_col]

        # metrics + artifacts
        M, common, y_true, y_pred, label_map = metrics_against_labels(part_s, labels_s)
        met_path = out_dir / f"metrics_{name.lower()}.csv"
        pd.DataFrame([M]).to_csv(met_path, index=False)

        conf_df = cluster_label_confusion(y_true, y_pred, label_map)
        conf_df.to_csv(out_dir / f"confusion_{name.lower()}.csv")

        enrich_df = hypergeom_enrichment(part_s, labels_s)
        enrich_path = out_dir / f"enrichment_{name.lower()}.csv"
        enrich_df.to_csv(enrich_path, index=False)

        # per-cluster alignment & correlations
        align_df = per_cluster_alignment(part_s, labels_s, enrich_df)
        align_path = out_dir / f"per_cluster_alignment_{name.lower()}.csv"
        align_df.to_csv(align_path, index=False)

        coh_path = out_dir / "cluster_coherence.csv"
        if not coh_path.exists():
            raise FileNotFoundError(
                f"Required coherence table `{coh_path}` not found; rerun with `--coherence` enabled."
            )
        try:
            coh_df_local = pd.read_csv(coh_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"Coherence table `{coh_path}` is empty; ensure coherence computation completed successfully."
            ) from exc
        corr_df, corr_path = correlate_coherence_alignment(
            coh_df_local, align_df, name, out_dir
        )

        # scatter exports
        export_scatter_data(coh_df_local, align_df, name, out_dir)

        summary_lines.append(
            f"- {name}: n={M['n_common_nodes']} nodes; "
            f"NMI={M['nmi']:.3f}, AMI={M['ami']:.3f}, ARI={M['ari']:.3f}, "
            f"H={M['homogeneity']:.3f}, C={M['completeness']:.3f}, V={M['v_measure']:.3f}. "
            f"Tables: `metrics_{name.lower()}.csv`, `confusion_{name.lower()}.csv`, "
            f"`enrichment_{name.lower()}.csv`, `per_cluster_alignment_{name.lower()}.csv`, "
            f"`{corr_path.name}`."
        )

    # frameworks
    do_framework("HiTOP", hitop_csv, hitop_col)
    do_framework("RDoC", rdoc_csv, rdoc_col)

    # coverage
    coverage = {
        "graph_nodes": len(G.nodes),
        "partition_nodes": len(part_s),
        "clusters": int(part_s.nunique()),
        "coverage_ratio": float(len(part_s) / len(G.nodes)) if len(G.nodes) else 0.0,
    }
    pd.DataFrame([coverage]).to_json(
        out_dir / "partition_coverage.json", orient="records", indent=2
    )

    # reports
    generate_summary_report(out_dir)
    write_global_corr_md(out_dir)

    # alignment summary (md)
    md = [
        "# Alignment Summary",
        "",
        f"- Graph: `{graph_path.name}`",
        f"- Partition: `{partition_path.name}`",
        f"- Total graph nodes: {coverage['graph_nodes']}",
        f"- Nodes with partition assignments (∩ graph): {coverage['partition_nodes']} "
        f"({coverage['coverage_ratio']:.2%})",
        f"- #Clusters: {coverage['clusters']}",
        "",
        "## Framework Metrics",
    ] + (
        summary_lines
        or ["- (No frameworks evaluated — provide mapping CSVs to compute alignment.)"]
    )
    (out_dir / "alignment_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to the GraphML file containing the full multiplex knowledge graph "
        "(nodes, edges, and attributes used for label inference and propagation).",
    )
    ap.add_argument(
        "--partition",
        type=Path,
        required=True,
        help="Path to partition.json containing the learned node-to-cluster assignments "
        "produced by your graph model (expects a 'node_to_cluster' dictionary).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="Directory where all outputs, metrics, and reports will be saved "
        "(created if it does not already exist).",
    )
    ap.add_argument(
        "--hitop-map",
        type=Path,
        default=None,
        help="Optional CSV mapping of node IDs to HiTOP labels. "
        "If omitted, HiTOP labels are automatically inferred from node names and types.",
    )
    ap.add_argument(
        "--rdoc-map",
        type=Path,
        default=None,
        help="Optional CSV mapping of node IDs to RDoC domain labels. "
        "If omitted, RDoC labels are automatically inferred and propagated through the graph.",
    )
    ap.add_argument(
        "--id-col",
        type=str,
        default="node_id",
        help="Column name in the mapping CSVs that contains node IDs corresponding to graph nodes.",
    )
    ap.add_argument(
        "--hitop-col",
        type=str,
        default="hitop_label",
        help="Column name in the HiTOP mapping CSV that contains the target label for each node.",
    )
    ap.add_argument(
        "--rdoc-col",
        type=str,
        default="rdoc_label",
        help="Column name in the RDoC mapping CSV that contains the target label for each node.",
    )
    ap.add_argument(
        "--prop-depth",
        type=int,
        default=1,
        help="RDoC propagation depth for auto-labeling (0=direct only, 1=one-hop default, 2=two-hop).",
    )

    # coherence / embeddings
    ap.add_argument(
        "--coherence",
        type=str,
        default="sentence",
        choices=["sentence", "tfidf"],
        help="Backend used to compute semantic coherence within clusters: "
        "'sentence' = SentenceTransformers embeddings (recommended), "
        "'tfidf' = TF-IDF ngrams (fast baseline).",
    )
    ap.add_argument(
        "--coherence-field",
        type=str,
        default="canonical_name",
        help="Node attribute to embed for coherence (e.g., 'name', 'label', or a custom text field).",
    )
    ap.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name or local path for the embedding backend "
        "(e.g., 'all-MiniLM-L6-v2' or a local biomedical SBERT).",
    )
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Batch size for encoding node texts with the embedding model (higher = faster but more RAM).",
    )
    ap.add_argument(
        "--embed-cache",
        type=str,
        default="emb_cache.npz",
        help="Path to on-disk cache for node text embeddings; avoids recomputation when inputs/model are unchanged.",
    )
    ap.add_argument(
        "--coh-bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap resamples for the mean coherence confidence interval per cluster.",
    )
    ap.add_argument(
        "--coh-ci",
        type=float,
        default=0.95,
        help="Confidence level (e.g., 0.95) for bootstrap CI on mean coherence.",
    )
    ap.add_argument(
        "--size-weighting",
        type=str,
        default="log1p",
        choices=["none", "log1p"],
        help="Scheme for size-aware coherence: 'none' = raw mean cosine; "
        "'log1p' = multiply mean coherence by log1p(cluster size), with a normalized variant reported too.",
    )
    ap.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum number of nodes required in a cluster to compute coherence statistics.",
    )
    ap.add_argument(
        "--neighbors-k",
        type=int,
        default=5,
        help="How many nearest neighbors to the cluster medoid to save in the explainability CSV.",
    )

    args = ap.parse_args()
    run_alignment(
        graph_path=args.graph,
        partition_path=args.partition,
        out_dir=args.outdir,
        hitop_csv=args.hitop_map,
        rdoc_csv=args.rdoc_map,
        id_col=args.id_col,
        hitop_col=args.hitop_col,
        rdoc_col=args.rdoc_col,
        prop_depth=args.prop_depth,
        coherence=args.coherence,
        coherence_field=args.coherence_field,
        embed_model=args.embed_model,
        embed_batch_size=args.embed_batch_size,
        embed_cache=args.embed_cache,
        coh_bootstrap=args.coh_bootstrap,
        coh_ci=args.coh_ci,
        size_weighting=args.size_weighting,
        min_cluster_size=args.min_cluster_size,
        neighbors_k=args.neighbors_k,
    )
