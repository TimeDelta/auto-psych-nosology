"""
Interactive cluster inspector + optional token-level saliency + Markdown explainer.

Outputs
-------
- cluster_{id}_medoid_neighbors.csv
- cluster_{id}_top_pairs.csv
- (optional) saliency_* CSVs + meta JSON
- cluster_{id}_explain.md  (with sorted previews + spark-bars)

Usage
-----
python inspect_cluster.py \
  --graph /mnt/data/prebuilt-kg.graphml \
  --partition /mnt/data/partition.json \
  --cluster 12 \
  --embed-model all-MiniLM-L6-v2 \
  --embed-cache /mnt/data/emb_cache.npz \
  --neighbors-k 12 --top-pairs 40 \
  --explain \
  --saliency --saliency-top-pair --saliency-top-tokens 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _node_text(G: nx.Graph, n: str, field: str) -> str:
    val = G.nodes[n].get(field)
    if val is None:
        val = G.nodes[n].get("name", str(n))
    return str(val)


def _load_graph(graph_path: Path) -> nx.Graph:
    G = nx.read_graphml(graph_path)
    return nx.relabel_nodes(G, {n: str(n) for n in G.nodes})


def _load_partition(partition_path: Path) -> Dict[str, int]:
    data = json.loads(Path(partition_path).read_text())
    node_to_cluster = data.get("node_to_cluster")
    if node_to_cluster is None:
        raise ValueError("partition.json missing `node_to_cluster`")
    return {str(k): int(v) for k, v in node_to_cluster.items()}


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


def _encode_texts_batched(model, texts: List[str], batch_size: int = 256) -> np.ndarray:
    return np.array(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def get_or_build_embeddings(
    G: nx.Graph,
    nodes: List[str],
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
                idx_map_all = {nid: i for i, nid in enumerate(order)}
                take = [idx_map_all[n] for n in nodes]
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


def cluster_medoid_index(E: np.ndarray) -> int:
    if E.shape[0] <= 1:
        return 0
    S = cosine_similarity(E)
    return int(np.argmax(S.mean(axis=1)))


def top_pairs(E: np.ndarray, nodes: List[str], K: int = 20) -> pd.DataFrame:
    n = E.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["i", "j", "node_i", "node_j", "cosine"])
    S = cosine_similarity(E)
    tri_i, tri_j = np.triu_indices(n, k=1)
    cos = S[tri_i, tri_j]
    order = np.argsort(-cos)
    keep = order[: min(K, order.size)]
    rows = []
    for idx in keep:
        i, j = int(tri_i[idx]), int(tri_j[idx])
        rows.append(
            {
                "i": i,
                "j": j,
                "node_i": nodes[i],
                "node_j": nodes[j],
                "cosine": float(cos[idx]),
            }
        )
    return pd.DataFrame(rows)


def nearest_to_medoid(E: np.ndarray, nodes: List[str], K: int = 10) -> pd.DataFrame:
    m = cluster_medoid_index(E)
    sims = cosine_similarity(E[m : m + 1], E).ravel()
    order = np.argsort(-sims)
    keep = order[: min(K, len(order))]
    return pd.DataFrame(
        {
            "rank": np.arange(len(keep)),
            "node_id": [nodes[j] for j in keep],
            "cosine_to_medoid": [float(sims[j]) for j in keep],
            "is_medoid": [bool(j == m) for j in keep],
            "medoid_id": nodes[m],
        }
    )


def token_saliency_pair(
    text_a: str, text_b: str, model_name: str, max_tokens: int = 64
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    def encode_tokens(text: str):
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=True,
        )
        with torch.no_grad():
            out = mdl(**{k: v.to(device) for k, v in enc.items()})
            H = out.last_hidden_state.squeeze(0)
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        attn = enc["attention_mask"][0].bool()
        H = H[attn]
        tokens = [t for t, m in zip(tokens, attn.tolist()) if m]
        sent = H.mean(dim=0)
        sent = sent / (sent.norm(p=2) + 1e-12)
        Hn = H / (H.norm(dim=1, keepdim=True) + 1e-12)
        return tokens, Hn.cpu().numpy(), sent.cpu().numpy()

    toks_a, Ta, sa = encode_tokens(text_a)
    toks_b, Tb, sb = encode_tokens(text_b)
    sent_cos = float(np.dot(sa, sb))
    sal_a_vals = Ta @ sb
    sal_b_vals = Tb @ sa
    df_a = pd.DataFrame({"token": toks_a, "saliency_to_B": sal_a_vals})
    df_b = pd.DataFrame({"token": toks_b, "saliency_to_A": sal_b_vals})
    # normalize for readability
    for col in ("saliency_to_B",):
        v = df_a[col].values
        lo, hi = (v.min(), v.max()) if v.size else (0, 1)
        df_a[col + "_norm"] = (v - lo) / (hi - lo + 1e-12)
    for col in ("saliency_to_A",):
        v = df_b[col].values
        lo, hi = (v.min(), v.max()) if v.size else (0, 1)
        df_b[col + "_norm"] = (v - lo) / (hi - lo + 1e-12)
    return df_a, df_b, sent_cos


_BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def _sparkbar(val: float, width: int = 8, lo: float = 0.0, hi: float = 1.0) -> str:
    try:
        x = float(val)
    except Exception:
        return ""
    if not (lo <= x <= hi):
        return ""
    p = 0.0 if hi == lo else (x - lo) / (hi - lo)
    if p <= 0:
        return " " * width
    total = p * width
    full = int(total)
    frac = total - full
    bar = "█" * full
    if full < width:
        idx = min(int(round(frac * (len(_BLOCKS) - 1))), len(_BLOCKS) - 1)
        bar += _BLOCKS[idx]
        bar += " " * (width - full - 1)
    return bar


def _md_table(
    df, cols, head_n=None, sort_by=None, desc=True, bar_cols=None, bar_width=8
):
    if df.empty:
        return "_(no entries)_"
    df = df.copy()
    if sort_by and sort_by in df.columns:
        try:
            df["_sort_key"] = df[sort_by].astype(float)
            df = df.sort_values("_sort_key", ascending=not desc).drop(
                columns=["_sort_key"]
            )
        except Exception:
            df = df.sort_values(sort_by, ascending=not desc)
    if head_n is not None:
        df = df.head(head_n)
    bar_cols = bar_cols or {}
    final_cols = []
    for c in cols:
        final_cols.append(c)
        if c in bar_cols:
            lo, hi = bar_cols[c]
            col_bar = f"{c}_bar"

            def to_bar(v):
                try:
                    return _sparkbar(float(v), width=bar_width, lo=lo, hi=hi)
                except Exception:
                    return ""

            df[col_bar] = df[c].apply(to_bar)
            final_cols.append(col_bar)
    lines = []
    hdr = "| " + " | ".join(final_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(final_cols)) + " |"
    lines.append(hdr)
    lines.append(sep)
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in final_cols) + " |")
    return "\n".join(lines)


def render_cluster_markdown(
    out_path: Path,
    cluster_id: int,
    medoid_neighbors: pd.DataFrame,
    top_pairs: pd.DataFrame,
    G: nx.Graph,
    field: str,
    saliency_a: pd.DataFrame | None = None,
    saliency_b: pd.DataFrame | None = None,
    saliency_meta: dict | None = None,
    saliency_top_tokens: int = 12,
    preview_rows: int = 8,
):
    lines = [f"# Cluster {cluster_id} — Quick Explanation", ""]

    # Medoid + neighbors
    if not medoid_neighbors.empty:
        medoid_id = medoid_neighbors.iloc[0]["medoid_id"]
        medoid_name = G.nodes[medoid_id].get(
            field, G.nodes[medoid_id].get("name", medoid_id)
        )
        lines += [
            "## Medoid",
            f"- **ID:** `{medoid_id}`",
            f"- **Name:** {medoid_name}",
            "",
        ]
        nbr_cols = ["rank", "node_id", "name", "cosine_to_medoid", "is_medoid"]
        nbr_df = medoid_neighbors[nbr_cols].copy()
        # keep numeric for sorting, but render to 3dp string for display
        nbr_df["cosine_to_medoid"] = (
            nbr_df["cosine_to_medoid"].astype(float).map(lambda x: f"{x:.3f}")
        )
        lines += [
            f"## Nearest to Medoid (top {preview_rows} by cosine)",
            _md_table(
                nbr_df,
                nbr_cols,
                head_n=preview_rows,
                sort_by="cosine_to_medoid",
                desc=True,
                bar_cols={"cosine_to_medoid": (0.0, 1.0)},
                bar_width=10,
            ),
            f"_Full CSV: cluster_{cluster_id}_medoid_neighbors.csv_",
            "",
        ]

    # Top pairs
    if not top_pairs.empty:
        pairs_cols = ["node_i", "name_i", "node_j", "name_j", "cosine"]
        pairs_df = top_pairs[pairs_cols].copy()
        pairs_df["cosine"] = pairs_df["cosine"].astype(float).map(lambda x: f"{x:.3f}")
        lines += [
            f"## Top Pairs (top {preview_rows} by cosine)",
            _md_table(
                pairs_df,
                pairs_cols,
                head_n=preview_rows,
                sort_by="cosine",
                desc=True,
                bar_cols={"cosine": (0.0, 1.0)},
                bar_width=10,
            ),
            f"_Full CSV: cluster_{cluster_id}_top_pairs.csv_",
            "",
        ]

    # Saliency (optional)
    if saliency_a is not None and saliency_b is not None and saliency_meta is not None:
        a_id = saliency_meta["node_a"]
        b_id = saliency_meta["node_b"]
        a_name = G.nodes[a_id].get(field, G.nodes[a_id].get("name", a_id))
        b_name = G.nodes[b_id].get(field, G.nodes[b_id].get("name", b_id))
        sent_cos = saliency_meta.get("sentence_cosine", float("nan"))
        lines += [
            "## Token-level Saliency",
            f"**Pair:** `{a_id}` ⇄ `{b_id}`",
            f"- A: *{a_name}*",
            f"- B: *{b_name}*",
            f"- Sentence cosine: **{sent_cos:.3f}**",
            "",
        ]
        sa = saliency_a.sort_values("saliency_to_B", ascending=False).copy()
        sb = saliency_b.sort_values("saliency_to_A", ascending=False).copy()
        sa["saliency_to_B"] = (
            sa["saliency_to_B"].astype(float).map(lambda x: f"{x:.3f}")
        )
        sb["saliency_to_A"] = (
            sb["saliency_to_A"].astype(float).map(lambda x: f"{x:.3f}")
        )
        lines += [
            f"### Top {min(saliency_top_tokens, len(sa))} tokens in A explaining B",
            _md_table(
                sa,
                ["token", "saliency_to_B"],
                head_n=saliency_top_tokens,
                sort_by="saliency_to_B",
                desc=True,
                bar_cols={"saliency_to_B": (0.0, 1.0)},
                bar_width=10,
            ),
            "",
        ]
        lines += [
            f"### Top {min(saliency_top_tokens, len(sb))} tokens in B explaining A",
            _md_table(
                sb,
                ["token", "saliency_to_A"],
                head_n=saliency_top_tokens,
                sort_by="saliency_to_A",
                desc=True,
                bar_cols={"saliency_to_A": (0.0, 1.0)},
                bar_width=10,
            ),
            "",
        ]
        lines.append(
            f"_Full CSVs: saliency_{a_id}_to_{b_id}.csv and saliency_{b_id}_to_{a_id}.csv_"
        )
        lines.append("")
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--field", type=str, default="name")
    ap.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--embed-cache", type=str, default="emb_cache.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--neighbors-k", type=int, default=10)
    ap.add_argument("--top-pairs", type=int, default=20)
    ap.add_argument(
        "--explain",
        action="store_true",
        help="Write Markdown summary for this cluster.",
    )
    ap.add_argument(
        "--saliency",
        action="store_true",
        help="Compute token-level saliency for a pair.",
    )
    ap.add_argument(
        "--saliency-top-pair",
        action="store_true",
        help="Use highest-cosine pair for saliency.",
    )
    ap.add_argument("--pair-i", type=int, default=None)
    ap.add_argument("--pair-j", type=int, default=None)
    ap.add_argument(
        "--saliency-model",
        type=str,
        default=None,
        help="HF model for token saliency (defaults to --embed-model).",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--saliency-top-tokens", type=int, default=12)
    ap.add_argument("--outdir", type=Path, default=Path("out_inspect"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    G = _load_graph(args.graph)
    part = _load_partition(args.partition)
    nodes_in_cluster = [n for n, c in part.items() if c == args.cluster and n in G]
    if not nodes_in_cluster:
        raise SystemExit(f"No nodes found for cluster {args.cluster} in graph.")

    # embeddings
    E, idx_map = get_or_build_embeddings(
        G,
        nodes_in_cluster,
        field=args.field,
        model_name=args.embed_model,
        batch_size=args.batch_size,
        cache_path=args.embed_cache,
    )

    # medoid neighbors
    neigh_df = nearest_to_medoid(E, nodes_in_cluster, K=args.neighbors_k)
    neigh_df["name"] = [_node_text(G, nid, args.field) for nid in neigh_df["node_id"]]
    neigh_df.to_csv(
        args.outdir / f"cluster_{args.cluster}_medoid_neighbors.csv", index=False
    )

    # top pairs
    pairs_df = top_pairs(E, nodes_in_cluster, K=args.top_pairs)
    if not pairs_df.empty:
        pairs_df["name_i"] = [
            _node_text(G, nid, args.field) for nid in pairs_df["node_i"]
        ]
        pairs_df["name_j"] = [
            _node_text(G, nid, args.field) for nid in pairs_df["node_j"]
        ]
    pairs_df.to_csv(args.outdir / f"cluster_{args.cluster}_top_pairs.csv", index=False)

    print(
        f"Saved medoid neighbors → {args.outdir}/cluster_{args.cluster}_medoid_neighbors.csv"
    )
    print(f"Saved top pairs       → {args.outdir}/cluster_{args.cluster}_top_pairs.csv")

    # optional saliency
    sal_a = sal_b = None
    sal_meta = None
    if args.saliency:
        if args.saliency_top_pair:
            if pairs_df.empty:
                raise SystemExit("No pairs available for saliency.")
            row = pairs_df.sort_values("cosine", ascending=False).iloc[0]
            nid_a, nid_b = row["node_i"], row["node_j"]
        else:
            if args.pair_i is None or args.pair_j is None:
                raise SystemExit(
                    "--saliency requires --saliency-top-pair or both --pair-i and --pair-j."
                )
            nid_a = nodes_in_cluster[args.pair_i]
            nid_b = nodes_in_cluster[args.pair_j]
        text_a = _node_text(G, nid_a, args.field)
        text_b = _node_text(G, nid_b, args.field)
        sal_model = args.saliency_model or args.embed_model
        sal_a, sal_b, sent_cos = token_saliency_pair(
            text_a, text_b, model_name=sal_model, max_tokens=args.max_tokens
        )
        sal_a.to_csv(
            args.outdir / f"cluster_{args.cluster}_saliency_{nid_a}_to_{nid_b}.csv",
            index=False,
        )
        sal_b.to_csv(
            args.outdir / f"cluster_{args.cluster}_saliency_{nid_b}_to_{nid_a}.csv",
            index=False,
        )
        sal_meta = {
            "cluster": args.cluster,
            "node_a": nid_a,
            "text_a": text_a,
            "node_b": nid_b,
            "text_b": text_b,
            "sentence_cosine": sent_cos,
            "model": sal_model,
            "max_tokens": args.max_tokens,
        }
        Path(args.outdir / f"cluster_{args.cluster}_saliency_meta.json").write_text(
            json.dumps(sal_meta, indent=2)
        )
        print(f"Saved token saliency (sentence cosine={sent_cos:.3f}) → {args.outdir}")

    # Markdown explainer
    if args.explain:
        md_path = args.outdir / f"cluster_{args.cluster}_explain.md"
        render_cluster_markdown(
            out_path=md_path,
            cluster_id=args.cluster,
            medoid_neighbors=neigh_df,
            top_pairs=pairs_df,
            G=G,
            field=args.field,
            saliency_a=sal_a,
            saliency_b=sal_b,
            saliency_meta=sal_meta,
            saliency_top_tokens=args.saliency_top_tokens,
        )
        print(f"Wrote Markdown explainer → {md_path}")
