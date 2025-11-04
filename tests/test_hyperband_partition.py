from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import hyperband_partition as hp
from hyperband_partition import (
    AlignmentEvaluator,
    AlignmentOutcome,
    EvaluationResult,
    HyperbandSearch,
    SearchSpace,
    SearchSpaceError,
    Trial,
    _parse_train_overrides,
    compute_fitness,
)


def _write_graphml(tmp_path: Path) -> Path:
    graph_path = tmp_path / "toy.graphml"
    G = nx.Graph()
    G.add_node("n1", name="Node 1", node_type="phenotype")
    G.add_node("n2", name="Node 2", node_type="phenotype")
    G.add_edge("n1", "n2", relation="linked")
    nx.write_graphml(G, graph_path)
    return graph_path


def _write_label_csv(tmp_path: Path, filename: str) -> Path:
    path = tmp_path / filename
    payload = [
        "id,label",
        "n1,A",
        "n2,B",
    ]
    path.write_text("\n".join(payload), encoding="utf-8")
    return path


def test_alignment_evaluator_reports_perfect_metrics(tmp_path):
    # Stub alignment API to avoid importing SciPy-heavy dependencies during tests.
    def _stub_load_graph(path: Path):
        return nx.read_graphml(path)

    def _stub_align_partition_to_graph(G: nx.Graph, mapping):
        aligned = {k: int(v) for k, v in mapping.items() if k in G}
        missing = [k for k in mapping if k not in G]
        return aligned, missing

    def _stub_infer_framework_labels_tailored(G: nx.Graph, *_, **__):
        labels = {node: G.nodes[node].get("stub_label", node) for node in G.nodes}
        series = pd.Series(labels, name="label")
        return series, series.copy()

    def _stub_load_mapping_csv(path: Path, id_col: str, label_col: str):
        return pd.read_csv(path)

    def _stub_metrics_against_labels(part_series: pd.Series, label_series: pd.Series):
        common = part_series.index.intersection(label_series.index)
        y_pred = part_series.loc[common].astype(int).to_numpy()
        labels = label_series.loc[common].astype(str)
        unique_labels = {lab: idx for idx, lab in enumerate(pd.unique(labels))}
        y_true = np.array([unique_labels[lab] for lab in labels], dtype=int)
        perfect = len(set(zip(y_pred, labels))) == len(common) if len(common) else False
        metrics = {
            "n_common_nodes": int(len(common)),
            "n_unique_true_labels": int(len(unique_labels)),
            "n_unique_pred_clusters": int(np.unique(y_pred).size),
            "nmi": 1.0 if perfect else 0.0,
            "ami": 1.0 if perfect else 0.0,
            "ari": 1.0 if perfect else 0.0,
            "homogeneity": 1.0 if perfect else 0.0,
            "completeness": 1.0 if perfect else 0.0,
            "v_measure": 1.0 if perfect else 0.0,
        }
        return metrics, common, y_true, y_pred, unique_labels

    hp._ALIGN_API = {
        "load_graph": _stub_load_graph,
        "align_partition_to_graph": _stub_align_partition_to_graph,
        "infer_framework_labels_tailored": _stub_infer_framework_labels_tailored,
        "load_mapping_csv": _stub_load_mapping_csv,
        "metrics_against_labels": _stub_metrics_against_labels,
    }

    graph_path = _write_graphml(tmp_path)
    hitop_csv = _write_label_csv(tmp_path, "hitop.csv")
    rdoc_csv = _write_label_csv(tmp_path, "rdoc.csv")

    evaluator = AlignmentEvaluator(
        graph_path,
        prop_depth=1,
        hitop_csv=hitop_csv,
        hitop_id_col="id",
        hitop_label_col="label",
        rdoc_csv=rdoc_csv,
        rdoc_id_col="id",
        rdoc_label_col="label",
    )

    partition = {"n1": 0, "n2": 1}
    outcome = evaluator.evaluate(partition)

    assert outcome.missing_nodes == []
    assert outcome.coverage["coverage_ratio"] == pytest.approx(1.0)
    assert outcome.metrics["hitop"]["nmi"] == pytest.approx(1.0)
    assert outcome.metrics["rdoc"]["nmi"] == pytest.approx(1.0)


def test_compute_fitness_variants():
    outcome = AlignmentOutcome(
        metrics={
            "hitop": {"nmi": 0.6, "v_measure": 0.8},
            "rdoc": {"nmi": 0.9, "v_measure": 0.7},
        },
        coverage={},
        missing_nodes=[],
    )

    assert compute_fitness(outcome, "mean_nmi") == pytest.approx(0.75)
    assert compute_fitness(outcome, "mean_v_measure") == pytest.approx(0.75)
    assert compute_fitness(outcome, "hitop.nmi") == pytest.approx(0.6)
    assert compute_fitness(outcome, "min_nmi") == pytest.approx(0.6)
    normalized_values = [0.6, 0.8, 0.9, 0.7]
    expected_mean_all = sum(normalized_values) / len(normalized_values)
    assert compute_fitness(outcome, "mean_all") == pytest.approx(expected_mean_all)
    assert compute_fitness(outcome, "avg_all_metrics") == pytest.approx(
        expected_mean_all
    )
    with pytest.raises(ValueError):
        compute_fitness(outcome, "unsupported_metric")


def test_hyperband_search_prefers_high_fitness(tmp_path):
    configs = [
        {"score": 0.1},
        {"score": 0.6},
        {"score": 0.9},
        {"score": 0.4},
    ]

    def sample_trial(trial_id: int) -> Trial:
        config = configs[trial_id % len(configs)]
        return Trial(id=trial_id, config=config, seed=trial_id + 1)

    def evaluate(
        trial: Trial, resource: int, rung: int, bracket: int
    ) -> EvaluationResult:
        score = float(trial.config["score"])
        outcome = AlignmentOutcome(
            metrics={"hitop": {"nmi": score}, "rdoc": {"nmi": score}},
            coverage={},
            missing_nodes=[],
        )
        return EvaluationResult(
            trial=trial,
            bracket=bracket,
            rung=rung,
            resource=resource,
            fitness=score,
            outcome=outcome,
            duration=0.0,
            info={"resource": resource},
        )

    search = HyperbandSearch(
        sample_trial=sample_trial,
        evaluate=evaluate,
        max_resource=9,
        min_resource=3,
        eta=3.0,
        max_evaluations=12,
    )

    results = search.run()

    assert results, "Hyperband should return evaluation results"
    max_fitness = max(res.fitness for res in results)
    assert max_fitness == pytest.approx(0.9)

    best_resources = [res.resource for res in results if res.fitness == max_fitness]
    assert max(best_resources) == 9

    assert len(results) <= 12


def test_search_space_sampling(tmp_path):
    space_path = tmp_path / "space.json"
    space_spec = {
        "lr": {"distribution": "loguniform", "min": 1e-4, "max": 1e-2},
        "gate_threshold": {"values": [0.3, 0.4, 0.5]},
    }
    space_path.write_text(json.dumps(space_spec), encoding="utf-8")

    space = SearchSpace(space_spec)
    rng = random.Random(42)
    sample = space.sample(rng)

    assert "lr" in sample and "gate_threshold" in sample
    assert 1e-4 <= sample["lr"] <= 1e-2
    assert sample["gate_threshold"] in {0.3, 0.4, 0.5}


def test_parse_train_overrides_accepts_json(tmp_path):
    overrides_path = tmp_path / "train_defaults.json"
    overrides_path.write_text(
        json.dumps({"pos_edge_chunk": 256, "flag": True}),
        encoding="utf-8",
    )
    overrides = _parse_train_overrides([str(overrides_path), "lr=0.001"])
    assert overrides["pos_edge_chunk"] == 256
    assert overrides["flag"] is True
    assert overrides["lr"] == 0.001


def test_search_space_rejects_nonpositive_loguniform():
    with pytest.raises(SearchSpaceError):
        SearchSpace({"tol": {"distribution": "loguniform", "min": 0.0, "max": 1.0}})
