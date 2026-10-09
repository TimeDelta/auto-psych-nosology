"""Render precise, editable architecture diagrams without running any model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "auto-psych-nosology-architecture-v1",
    }
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "figures"
COLORS = {
    "input": ("#eef4ff", "#2358a5"),
    "model": ("#eff9f6", "#137761"),
    "fit": ("#fff6e7", "#a76615"),
    "current": ("#f3f0fa", "#655092"),
    "note": ("#fff0ef", "#a64238"),
}


def canvas(title, subtitle, badge):
    figure, axes = plt.subplots(figsize=(16, 9), facecolor="#ffffff")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis("off")
    axes.text(0.045, 0.94, title, fontsize=25, color="#152b40", weight="bold")
    axes.text(0.045, 0.895, subtitle, fontsize=13, color="#4e6275")
    axes.text(
        0.95, 0.95, badge, ha="right", fontsize=10, color="#4e6275", weight="bold"
    )
    axes.plot([0.045, 0.95], [0.865, 0.865], color="#d7e0e7", lw=1)
    return figure, axes


def box(axes, rectangle, title, body, kind="model", fontsize=13):
    horizontal, vertical, width, height = rectangle
    fill_color, line_color = COLORS[kind]
    axes.add_patch(
        FancyBboxPatch(
            (horizontal, vertical),
            width,
            height,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            linewidth=1.4,
            edgecolor=line_color,
            facecolor=fill_color,
        )
    )
    axes.text(
        horizontal + 0.013,
        vertical + height - 0.038,
        title,
        fontsize=14,
        weight="bold",
        color=line_color,
        va="top",
    )
    axes.text(
        horizontal + 0.013,
        vertical + height - 0.084,
        body,
        fontsize=fontsize,
        color="#253e52",
        va="top",
        linespacing=1.55,
    )


def arrow(axes, start, end, *, kind="data", connection="arc3,rad=0"):
    fitting = kind == "fit"
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.8,
            color="#a76615" if fitting else "#536c81",
            linestyle="--" if fitting else "-",
            connectionstyle=connection,
            shrinkA=2,
            shrinkB=2,
        )
    )


def footer(axes, note):
    axes.text(
        0.045, 0.055, note, fontsize=12, color="#4e6275", va="bottom", linespacing=1.5
    )
    axes.text(
        0.98,
        0.025,
        "auto-psych-nosology | architecture documentation",
        fontsize=9,
        color="#768899",
        ha="right",
    )


def save(figure, stem):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIRECTORY / f"{stem}.svg"
    figure.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    figure.savefig(
        OUTPUT_DIRECTORY / f"{stem}.png",
        dpi=150,
        metadata={"Software": "architecture renderer"},
    )
    plt.close(figure)


def render_sparse_logistic():
    figure, axes = canvas(
        "Sparse reduced-rank logistic baseline",
        "Shared response factors from physiology-derived perturbation features; no learned perturbation-ID embeddings.",
        "IMPLEMENTED IN THIS CHANGE",
    )
    rectangles = [
        (0.05, 0.60, 0.205, 0.19),
        (0.30, 0.60, 0.20, 0.19),
        (0.545, 0.60, 0.20, 0.19),
        (0.79, 0.60, 0.16, 0.19),
    ]
    box(
        axes,
        rectangles[0],
        "Frozen features X",
        "Upstream physiology operator\nPerturbation + graph context\nNamed node/state channels",
        "input",
        fontsize=12,
    )
    box(
        axes,
        rectangles[1],
        "Normalization Z",
        "Means/scales from training\nrows with included evidence\nDense or CSR matrix",
        "model",
        fontsize=12,
    )
    box(
        axes,
        rectangles[2],
        "Shared coefficients B",
        "logits = Z B + intercepts\nOne column per symptom\nSparse rows, reduced rank",
        "model",
        fontsize=12,
    )
    box(
        axes,
        rectangles[3],
        "Sigmoid",
        "Per-symptom\nprobabilities\nfor new perturbations",
        "model",
        fontsize=12,
    )
    for first, second in zip(rectangles, rectangles[1:]):
        arrow(axes, (first[0] + first[2] + 0.01, 0.695), (second[0] - 0.01, 0.695))
    box(
        axes,
        (0.05, 0.23, 0.205, 0.24),
        "Evidence Y, M, w",
        "Binary symptom targets Y\nExplicit inclusion mask M\nNonnegative weights w\nPredefined leakage groups",
        "input",
        fontsize=12,
    )
    box(
        axes,
        (0.30, 0.23, 0.31, 0.24),
        "Fitting objective",
        "Masked binary log loss\n+ row-group penalty\n+ nuclear-norm penalty\n+ optional ridge penalty",
        "fit",
        fontsize=13,
    )
    box(
        axes,
        (0.67, 0.23, 0.28, 0.24),
        "Factor interpretation",
        "Post-fit SVD: B = U diag(s) V^T\nU: feature-factor loadings\nV: symptom-factor loadings\nNo unique biological mechanism implied",
        "model",
        fontsize=11.5,
    )
    arrow(axes, (0.265, 0.35), (0.29, 0.35))
    arrow(axes, (0.455, 0.48), (0.62, 0.59), kind="fit")
    axes.text(
        0.445,
        0.525,
        "proximal gradient\n+ joint Dykstra prox",
        fontsize=10,
        color="#a76615",
        ha="center",
    )
    arrow(axes, (0.85, 0.59), (0.54, 0.48), kind="fit")
    arrow(axes, (0.69, 0.59), (0.81, 0.48))
    footer(
        axes,
        "Solid arrows: prediction/data flow. Dashed arrows: fitting feedback.\nMissing targets are not automatic negatives; test labels do not select the model.",
    )
    save(figure, "sparse_rr_logistic")


def render_mechanistic_modules():
    figure, axes = canvas(
        "Mechanistic pathway module architecture",
        "Implemented in mechanistic-pathway-learning; the graph adapter does not port this model into the nosology trainer.",
        "UPSTREAM MODEL / COMPARATIVE CANDIDATE",
    )
    box(
        axes,
        (0.05, 0.69, 0.285, 0.13),
        "Physiology graph + context",
        "Relations, signs, compartments and context",
        "input",
        fontsize=11.5,
    )
    box(
        axes,
        (0.40, 0.69, 0.275, 0.13),
        "Localized perturbation",
        "Target nodes, direction and magnitude",
        "input",
        fontsize=12,
    )
    box(
        axes,
        (0.05, 0.43, 0.27, 0.18),
        "Physiology encoder",
        "Relational message passing\nor signed linear response\nUpstream biological operators",
        "model",
        fontsize=12,
    )
    box(
        axes,
        (0.37, 0.43, 0.25, 0.18),
        "Node-state field",
        "Perturbation response by node\nKeep location and state channels\nAbsolute or difference reading",
        "model",
        fontsize=12,
    )
    box(
        axes,
        (0.68, 0.43, 0.27, 0.18),
        "Sparse module pooling",
        "Learn support over graph nodes\nSupports may overlap\nWeighted sum or mean",
        "model",
        fontsize=12,
    )
    box(
        axes,
        (0.68, 0.18, 0.27, 0.17),
        "Module activations",
        "Readout + sigmoid per module\nOne activation profile\nfor each perturbation",
        "model",
        fontsize=12,
    )
    box(
        axes,
        (0.37, 0.18, 0.25, 0.17),
        "Noisy-OR head",
        "Module-symptom links\n+ symptom-specific leak\nCombine multiple contributions",
        "model",
        fontsize=12,
    )
    box(
        axes,
        (0.05, 0.18, 0.27, 0.17),
        "Symptom probabilities",
        "Evidence-based fitting\nSupport/link penalties\nSeparate held-out evaluation",
        "model",
        fontsize=12,
    )
    arrow(axes, (0.185, 0.68), (0.185, 0.62))
    arrow(axes, (0.54, 0.68), (0.28, 0.62))
    arrow(axes, (0.33, 0.52), (0.36, 0.52))
    arrow(axes, (0.63, 0.52), (0.67, 0.52))
    arrow(axes, (0.82, 0.42), (0.82, 0.36))
    arrow(axes, (0.67, 0.265), (0.63, 0.265))
    arrow(axes, (0.36, 0.265), (0.33, 0.265))
    footer(
        axes,
        "Several modules can contribute to one symptom; one module can contribute to several symptoms.\nLearned supports and predictions require validation before being interpreted as causal pathways.",
    )
    save(figure, "mechanistic_modules")


def render_current_autoencoder():
    figure, axes = canvas(
        "Current nosology graph autoencoder",
        "This documents the existing implementation. Its reconstruction target is graph edges, not symptom evidence.",
        "CURRENT CODE / REVISIONS PENDING",
    )
    box(
        axes,
        (0.05, 0.62, 0.205, 0.19),
        "Graph inputs",
        "Node types and attributes\nEdges and relation categories\nSampled ego-net batches",
        "input",
        fontsize=12,
    )
    box(
        axes,
        (0.30, 0.62, 0.20, 0.19),
        "R-GCN encoder",
        "Relation-specific messages\nNode embeddings\nLatent regularization",
        "current",
        fontsize=12,
    )
    box(
        axes,
        (0.545, 0.62, 0.20, 0.19),
        "Prototype scores",
        "Embedding/prototype similarity\nCluster-gate adjustments\nTemperature scaling",
        "current",
        fontsize=11.5,
    )
    box(
        axes,
        (0.79, 0.62, 0.16, 0.19),
        "Sinkhorn",
        "Balanced soft\nnode-cluster\nassignments",
        "current",
        fontsize=12,
    )
    box(
        axes,
        (0.68, 0.32, 0.27, 0.20),
        "Relational decoder",
        "Cluster interactions per relation\nLow-rank matrices + gates\nAbsent-edge biases",
        "current",
        fontsize=12,
    )
    box(
        axes,
        (0.365, 0.32, 0.26, 0.20),
        "Edge reconstruction",
        "Observed-edge BCE\n+ sampled-negative BCE\nRelation reweighting",
        "current",
        fontsize=12,
    )
    box(
        axes,
        (0.05, 0.32, 0.26, 0.20),
        "Composite training loss",
        "Reconstruction + L0 penalties\nEntropy/usage controls\nEmbedding/batch consistency",
        "fit",
        fontsize=12,
    )
    for start, end in (
        ((0.265, 0.715), (0.29, 0.715)),
        ((0.51, 0.715), (0.535, 0.715)),
        ((0.755, 0.715), (0.78, 0.715)),
        ((0.87, 0.61), (0.87, 0.53)),
        ((0.67, 0.42), (0.635, 0.42)),
        ((0.355, 0.42), (0.32, 0.42)),
    ):
        arrow(axes, start, end)
    box(
        axes,
        (0.05, 0.12, 0.90, 0.12),
        "Accepted revisions remain separate",
        "Repair gates, balancing, decoder range and negative sampling; establish symptom-evidence compression and a common model code.",
        "note",
        fontsize=11,
    )
    footer(
        axes,
        "A stable node partition or accurate adjacency reconstruction does not by itself establish a psychiatric nosology.",
    )
    save(figure, "current_graph_autoencoder")


if __name__ == "__main__":
    render_sparse_logistic()
    render_mechanistic_modules()
    render_current_autoencoder()
