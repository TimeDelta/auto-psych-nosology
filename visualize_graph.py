import argparse
import json
import math
import pathlib
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx


def _read_json_node_link(path: pathlib.Path) -> nx.Graph:
    from networkx.readwrite import json_graph

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, multigraph=False)


def _read_csv_edgelist(
    path: pathlib.Path, src_col: str, dst_col: str, directed: bool
) -> nx.Graph:
    import csv

    graph = nx.DiGraph() if directed else nx.Graph()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row[src_col]
            v = row[dst_col]
            attrs = {
                k: try_parse_number(row[k]) for k in row if k not in (src_col, dst_col)
            }
            graph.add_edge(u, v, **attrs)
    return graph


def try_parse_number(x: str) -> Any:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return s
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _ensure_canonical_labels(
    graph: nx.Graph, label_attr: str = "canonical_name"
) -> nx.Graph:
    for node, attrs in graph.nodes(data=True):
        label = attrs.get(label_attr) or attrs.get("name") or str(node)
        attrs[label_attr] = label
    return graph


def load_graph(
    path: str, csv_cols: Tuple[str, str] = None, directed: bool = False
) -> nx.Graph:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext in {".graphml"}:
        graph = nx.read_graphml(p)
        return _ensure_canonical_labels(graph)
    if ext in {".gexf"}:
        graph = nx.read_gexf(p)
        return _ensure_canonical_labels(graph)
    if ext in {".gpickle", ".pickle", ".pkl"}:
        graph = nx.read_gpickle(p)
        return _ensure_canonical_labels(graph)
    if ext in {".json"}:
        graph = _read_json_node_link(p)
        return _ensure_canonical_labels(graph)
    if ext in {".csv", ".tsv"}:
        if csv_cols is None:
            raise ValueError("For CSV/TSV, provide --csv-cols <SOURCE> <TARGET>")
        return _read_csv_edgelist(p, csv_cols[0], csv_cols[1], directed)
    if ext in {".edgelist", ".edges"}:
        return nx.read_edgelist(p)

    raise ValueError(f"Unsupported file extension: {ext}")


DEFAULT_NODE_PALETTE = [
    "#4e79a7",
    "#f28e2c",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]
DEFAULT_EDGE_PALETTE = [
    "#6b7b8c",
    "#c88f00",
    "#c43c39",
    "#4a9c97",
    "#3e7c3b",
    "#bfa300",
    "#8c5a87",
    "#d87a86",
    "#6e5a49",
    "#8f9396",
]


def categorical_color(key: str, seen: Dict[str, str], palette: Iterable[str]) -> str:
    if key in seen:
        return seen[key]
    color = list(palette)[len(seen) % len(list(palette))]
    seen[key] = color
    return color


def stringify_complex_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure attributes are JSON/pyvis friendly."""
    out = {}
    for k, v in attrs.items():
        if isinstance(v, (list, dict, set, tuple)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def compute_layout(
    graph: nx.Graph,
    layout: str,
    seed: int,
    scale: float,
    auto_threshold: int = 3000,
    iterations: Optional[int] = None,
) -> Optional[Dict[Any, Tuple[float, float]]]:
    """Return a 2D layout for the graph or None if layout is disabled."""

    layout_key = (layout or "none").lower()
    if layout_key in {"none", "off"}:
        return None

    n_nodes = graph.number_of_nodes()
    iteration_hint = iterations

    if layout_key == "auto":
        if n_nodes > auto_threshold and iteration_hint is None:
            # For very large graphs, bound the number of iterations so the layout completes promptly.
            iteration_hint = max(20, min(80, int(8000 / max(1.0, math.sqrt(n_nodes)))))
            print(
                f"[INFO] Auto layout: spring_layout with {iteration_hint} iterations for {n_nodes} nodes.",
                flush=True,
            )
        layout_key = "spring"
        if iteration_hint is None:
            iteration_hint = 60

    layout_key = layout_key.replace("-", "_")

    try:
        if layout_key in {"spring", "fr", "fruchterman_reingold"}:
            spring_kwargs: Dict[str, Any] = {"seed": seed}
            if iteration_hint is not None:
                spring_kwargs["iterations"] = iteration_hint
            base_pos = nx.spring_layout(graph, **spring_kwargs)
        elif layout_key in {"kamada", "kamada_kawai"}:
            base_pos = nx.kamada_kawai_layout(graph)
        elif layout_key == "circular":
            base_pos = nx.circular_layout(graph)
        elif layout_key == "spectral":
            base_pos = nx.spectral_layout(graph)
        elif layout_key == "random":
            base_pos = nx.random_layout(graph, seed=seed)
        else:
            raise ValueError(
                "Unsupported layout '"
                f"{layout}'"
                ". Choose from auto, none, spring, fruchterman_reingold, kamada_kawai, circular, spectral, random."
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(
            f"[WARN] Layout computation failed for '{layout_key}' ({exc}). Falling back to browser physics.",
            flush=True,
        )
        return None

    def _scale(value: float) -> float:
        return float(value) * scale

    return {
        node: (_scale(coords[0]), _scale(coords[1]))
        for node, coords in base_pos.items()
    }


def export_visnetwork_html(
    graph: nx.Graph,
    output_html: str,
    node_type_attr: str = "node_type",
    node_label_attr: str = "canonical_name",
    edge_type_attr: str = "relation",
    positions: Optional[Dict[Any, Tuple[float, float]]] = None,
    physics_enabled: bool = True,
    improved_layout: bool = True,
):
    # Build nodes/edges arrays for vis-network
    # Color maps (fixed palette cycling)
    node_colors = {}
    edge_colors = {}

    def pick_color(key, cmap, pal):
        if key not in cmap:
            cmap[key] = pal[len(cmap) % len(pal)]
        return cmap[key]

    nodes = []
    for n, attrs in graph.nodes(data=True):
        attrs_str = stringify_complex_attrs(attrs)
        label = str(attrs.get(node_label_attr, n))
        ntype = str(attrs.get(node_type_attr, "node"))
        color = pick_color(ntype, node_colors, DEFAULT_NODE_PALETTE)
        title_lines = [str(label)]
        title_lines.extend(
            f"{k}: {v}" for k, v in sorted(attrs_str.items()) if k != node_label_attr
        )
        title = "\n".join(title_lines)
        node_payload: Dict[str, Any] = {
            "id": str(n),
            "label": str(label),
            "title": title,
            "color": {"background": color, "border": "#222"},
            "shape": "dot",
            "size": 16,
        }

        if positions and n in positions:
            x, y = positions[n]
            node_payload.update({"x": x, "y": y})
            if not physics_enabled:
                node_payload["fixed"] = {"x": True, "y": True}

        if not physics_enabled:
            node_payload["physics"] = False

        nodes.append(node_payload)

    edges = []
    if graph.is_multigraph():
        for u, v, k, attrs in graph.edges(keys=True, data=True):
            attrs_str = stringify_complex_attrs(attrs)
            etype = str(attrs.get(edge_type_attr, "edge"))
            color = pick_color(etype, edge_colors, DEFAULT_EDGE_PALETTE)
            title_lines = [f"{kk}: {vv}" for kk, vv in sorted(attrs_str.items())]
            title = "\n".join(title_lines) if title_lines else etype
            edges.append(
                {
                    "id": f"{u}->{v}#{k}",
                    "from": str(u),
                    "to": str(v),
                    "color": {"color": color},
                    "arrows": "to" if graph.is_directed() else "",
                    "title": title,
                }
            )
    else:
        for u, v, attrs in graph.edges(data=True):
            attrs_str = stringify_complex_attrs(attrs)
            etype = str(attrs.get(edge_type_attr, "edge"))
            color = pick_color(etype, edge_colors, DEFAULT_EDGE_PALETTE)
            title_lines = [f"{kk}: {vv}" for kk, vv in sorted(attrs_str.items())]
            title = "\n".join(title_lines) if title_lines else etype
            edges.append(
                {
                    "from": str(u),
                    "to": str(v),
                    "color": {"color": color},
                    "arrows": "to" if graph.is_directed() else "",
                    "title": title,
                }
            )

    physics_config: Dict[str, Any] = {"enabled": physics_enabled}
    if physics_enabled:
        physics_config.update(
            {
                "stabilization": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 120,
                    "springConstant": 0.08,
                },
                "solver": "forceAtlas2Based",
            }
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>html,body,#mynet{{height:100%;margin:0;background:#111;color:#fff}}</style>
</head>
<body>
<div id="mynet"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(nodes)});
  const edges = new vis.DataSet({json.dumps(edges)});
  const container = document.getElementById('mynet');
  const data = {{ nodes, edges }};
  const options = {{
    nodes: {{ borderWidth: 1 }},
    interaction: {{ hover: true, multiselect: true }},
    layout: {{ improvedLayout: {str(improved_layout).lower()} }},
    physics: {json.dumps(physics_config)}
  }};
  new vis.Network(container, data, options);
</script>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(output_html)


def _add_legend(
    net, node_color_map: Dict[str, str], edge_color_map: Dict[str, str]
) -> None:
    # Node legend
    x, y = -1000, -1000
    for label, color in node_color_map.items():
        net.add_node(
            f"legend_node_{label}",
            label=f"NODE: {label}",
            color=color,
            shape="box",
            physics=False,
            x=x,
            y=y,
        )
        y += 60
    # Edge legend
    x2, y2 = -700, -1000
    # We use small dummy edges between invisible nodes to show color
    for label, color in edge_color_map.items():
        a = f"legend_edge_a_{label}"
        b = f"legend_edge_b_{label}"
        net.add_node(
            a, label="", color="#333333", shape="dot", size=5, physics=False, x=x2, y=y2
        )
        net.add_node(
            b,
            label=f"EDGE: {label}",
            color="#333333",
            shape="box",
            physics=False,
            x=x2 + 200,
            y=y2,
        )
        net.add_edge(a, b, color=color, physics=False, arrows=False)
        y2 += 60


def visualize_matplotlib(graph: nx.Graph, out_png: str = None) -> None:
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph, seed=42, k=0.3)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=80,
        node_color="#69b3a2",
        edgecolors="#222222",
        linewidths=0.5,
    )
    nx.draw_networkx_edges(
        graph, pos, alpha=0.25, arrows=graph.is_directed(), width=0.6
    )
    # Labels can clutter large graphs; comment out if needed
    nx.draw_networkx_labels(graph, pos, font_size=6)
    plt.axis("off")
    if out_png:
        plt.figure(figsize=(12, 10))
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[OK] Static PNG written to {out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a persisted graph file (GraphML/GEXF/GPickle/JSON/CSV)."
    )
    parser.add_argument("path", help="Path to the graph file.")
    parser.add_argument(
        "--out",
        default="graph.html",
        help="Output HTML for interactive visualization (PyVis).",
    )
    parser.add_argument(
        "--static-png",
        default=None,
        help="Optional PNG path for a Matplotlib static figure.",
    )
    parser.add_argument(
        "--node-type-attr", default="node_type", help="Node attribute to color by."
    )
    parser.add_argument(
        "--node-label-attr",
        default="canonical_name",
        help="Node attribute to label/tooltip.",
    )
    parser.add_argument(
        "--edge-type-attr", default="relation", help="Edge attribute to color by."
    )
    parser.add_argument(
        "--layout",
        default="auto",
        choices=[
            "auto",
            "none",
            "spring",
            "fruchterman_reingold",
            "kamada_kawai",
            "circular",
            "spectral",
            "random",
        ],
        help="Layout algorithm used to precompute node positions (auto bounds spring iterations for large graphs).",
    )
    parser.add_argument(
        "--layout-scale",
        type=float,
        default=500.0,
        help="Scale factor applied to layout coordinates (higher spreads nodes further).",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=42,
        help="Random seed used by layout algorithms that support it.",
    )
    parser.add_argument(
        "--layout-iterations",
        type=int,
        default=None,
        help="Number of iterations for spring/fruchterman_reingold layouts (auto chooses a size-aware default).",
    )
    parser.add_argument(
        "--layout-auto-threshold",
        type=int,
        default=3000,
        help="When --layout auto is set, graphs above this node count trigger reduced iteration counts instead of disabling the layout.",
    )
    parser.add_argument(
        "--physics",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control vis-network physics simulation (auto disables it when a layout is precomputed).",
    )
    parser.add_argument(
        "--csv-cols",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="For CSV edgelist: source and target columns.",
    )
    parser.add_argument(
        "--directed", action="store_true", help="Treat CSV/edgelist as directed."
    )
    args = parser.parse_args()
    graph = load_graph(
        args.path,
        csv_cols=tuple(args.csv_cols) if args.csv_cols else None,
        directed=args.directed,
    )

    for n, attrs in list(graph.nodes(data=True)):
        graph.nodes[n].update(stringify_complex_attrs(attrs))

    if graph.is_multigraph():
        for u, v, k, attrs in list(graph.edges(keys=True, data=True)):
            graph.edges[u, v, k].update(stringify_complex_attrs(attrs))
    else:
        for u, v, attrs in list(graph.edges(data=True)):
            graph.edges[u, v].update(stringify_complex_attrs(attrs))

    layout_positions = compute_layout(
        graph,
        layout=args.layout,
        seed=args.layout_seed,
        scale=args.layout_scale,
        auto_threshold=args.layout_auto_threshold,
        iterations=args.layout_iterations,
    )

    if args.physics == "auto":
        physics_enabled = layout_positions is None
    elif args.physics == "on":
        physics_enabled = True
    else:
        physics_enabled = False

    if layout_positions is not None and args.physics == "auto":
        # Physics fights with fixed layouts; auto-disable when using a precomputed layout.
        physics_enabled = False

    export_visnetwork_html(
        graph,
        output_html=args.out,
        node_type_attr=args.node_type_attr,
        node_label_attr=args.node_label_attr,
        edge_type_attr=args.edge_type_attr,
        positions=layout_positions,
        physics_enabled=physics_enabled,
        improved_layout=layout_positions is None,
    )

    if args.static_png:
        visualize_matplotlib(graph, out_png=args.static_png)
