import argparse
import json
import pathlib
from typing import Any, Dict, Iterable, Tuple

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


def load_graph(
    path: str, csv_cols: Tuple[str, str] = None, directed: bool = False
) -> nx.Graph:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext in {".graphml"}:
        return nx.read_graphml(p)
    if ext in {".gexf"}:
        return nx.read_gexf(p)
    if ext in {".gpickle", ".pickle", ".pkl"}:
        return nx.read_gpickle(p)
    if ext in {".json"}:
        return _read_json_node_link(p)
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


def export_visnetwork_html(
    graph: nx.Graph,
    output_html: str,
    node_type_attr: str = "node_type",
    node_label_attr: str = "canonical_name",
    edge_type_attr: str = "relation",
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
        label = str(attrs.get(node_label_attr, n))
        ntype = str(attrs.get(node_type_attr, "node"))
        color = pick_color(ntype, node_colors, DEFAULT_NODE_PALETTE)
        title_lines = [str(label)]
        title_lines.extend(
            f"{k}: {v}" for k, v in sorted(attrs.items()) if k != node_label_attr
        )
        title = "\n".join(title_lines)
        nodes.append(
            {
                "id": str(n),
                "label": str(label),
                "title": title,
                "color": {"background": color, "border": "#222"},
                "shape": "dot",
                "size": 16,
            }
        )

    edges = []
    if graph.is_multigraph():
        for u, v, k, attrs in graph.edges(keys=True, data=True):
            etype = str(attrs.get(edge_type_attr, "edge"))
            color = pick_color(etype, edge_colors, DEFAULT_EDGE_PALETTE)
            title_lines = [f"{kk}: {vv}" for kk, vv in sorted(attrs.items())]
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
            etype = str(attrs.get(edge_type_attr, "edge"))
            color = pick_color(etype, edge_colors, DEFAULT_EDGE_PALETTE)
            title_lines = [f"{kk}: {vv}" for kk, vv in sorted(attrs.items())]
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
    interaction: {{ hover: true }},
    physics: {{
      stabilization: true,
      forceAtlas2Based: {{
        gravitationalConstant: -50,
        centralGravity: 0.01,
        springLength: 120,
        springConstant: 0.08
      }},
      solver: 'forceAtlas2Based'
    }}
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

    export_visnetwork_html(
        graph,
        output_html=args.out,
        node_type_attr=args.node_type_attr,
        node_label_attr=args.node_label_attr,
        edge_type_attr=args.edge_type_attr,
    )

    if args.static_png:
        visualize_matplotlib(graph, out_png=args.static_png)
