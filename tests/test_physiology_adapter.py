import hashlib
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from adapt_physiology_graph import adapt_physiology_graph, build_adapted_graph
from relation_schema import resolve_edge_relation
from train_rgcn_scae import load_multiplex_graph, train_scae_on_graph


def test_empty_edge_table_retains_isolated_source_nodes():
    nodes = pd.DataFrame([{"node_id": "GENE:A", "node_type": "gene"}])
    edges = pd.DataFrame(columns=["source_id", "target_id", "relation_type"])
    graph = build_adapted_graph(nodes, edges, [])
    assert list(graph) == ["GENE:A"]
    assert graph.number_of_edges() == 0


def _write_source(tmp_path):
    graph_directory = tmp_path / "release" / "graph_split"
    graph_directory.mkdir(parents=True)
    nodes = pd.DataFrame(
        [
            {
                "node_id": "GENE:A",
                "node_type": "gene",
                "display_name": "A",
                "compartment": "n",
            },
            {
                "node_id": "PROTEIN:A",
                "node_type": "protein",
                "display_name": "A protein",
                "compartment": "c",
            },
            {
                "node_id": "REACTION:R",
                "node_type": "reaction",
                "display_name": "R",
                "compartment": "c",
                "gene_reaction_rule": "A and B",
            },
            {
                "node_id": "METABOLITE:Xc",
                "node_type": "metabolite",
                "display_name": "X",
                "compartment": "c",
            },
            {
                "node_id": "METABOLITE:Xm",
                "node_type": "metabolite",
                "display_name": "X",
                "compartment": "m",
            },
            {
                "node_id": "COMPLEX:C",
                "node_type": "protein_entity",
                "display_name": "C",
                "membership_logic": "all",
                "annotation": {"class": "Complex", "members": ["A", "B"]},
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "source_id": "GENE:A",
                "target_id": "PROTEIN:A",
                "relation_type": "encodes",
                "sign": 1.0,
                "evidence_source": "HGNC",
            },
            {
                "source_id": "PROTEIN:A",
                "target_id": "REACTION:R",
                "relation_type": "regulates",
                "sign": -1.0,
                "evidence_source": "paper1",
                "references": "1234",
            },
            {
                "source_id": "PROTEIN:A",
                "target_id": "REACTION:R",
                "relation_type": "regulates",
                "sign": 1.0,
                "evidence_source": "paper2",
                "references": "5678",
            },
            {
                "source_id": "PROTEIN:A",
                "target_id": "REACTION:R",
                "relation_type": "regulates",
                "sign": 1.0,
                "evidence_source": "paper3",
                "references": "9999",
            },
            {
                "source_id": "REACTION:R",
                "target_id": "METABOLITE:Xc",
                "relation_type": "product_of",
                "sign": 1.0,
                "evidence_source": "Human-GEM",
            },
            {
                "source_id": "METABOLITE:Xm",
                "target_id": "PROTEIN:A",
                "relation_type": "binds",
                "sign": 0.0,
                "evidence_source": "curated",
            },
            {
                "source_id": "PROTEIN:A",
                "target_id": "COMPLEX:C",
                "relation_type": "member_required",
                "sign": None,
                "evidence_source": "Reactome",
                "qualifiers": {"required": True},
            },
        ]
    )
    edges["key"] = "upstream-record-key"
    nodes.to_parquet(graph_directory / "nodes.parquet", index=False)
    edges.to_parquet(graph_directory / "edges.parquet", index=False)
    declared_relations = sorted(edges.relation_type.unique()) + ["unused_relation"]
    (graph_directory / "relation_types.json").write_text(json.dumps(declared_relations))
    return graph_directory, nodes, edges, declared_relations


def test_adapter_preserves_identity_parallel_records_and_future_semantics(tmp_path):
    graph_directory, nodes, edges, _ = _write_source(tmp_path)
    output_prefix = tmp_path / "output" / "physiology.v1"
    manifest = adapt_physiology_graph(graph_directory, output_prefix)
    graph = nx.read_graphml(Path(f"{output_prefix}.graphml"), force_multigraph=True)
    assert set(graph) == set(nodes.node_id)
    assert graph.number_of_edges() == len(edges)
    assert graph.number_of_edges("PROTEIN:A", "REACTION:R") == 3
    assert not graph.has_edge("REACTION:R", "PROTEIN:A")
    assert graph.nodes["METABOLITE:Xc"]["compartment"] == "c"
    assert graph.nodes["METABOLITE:Xm"]["compartment"] == "m"
    assert graph.nodes["REACTION:R"]["gene_reaction_rule"] == "A and B"
    assert graph.nodes["COMPLEX:C"]["membership_logic"] == "all"
    assert json.loads(graph.nodes["COMPLEX:C"]["annotation"])["members"] == ["A", "B"]
    assert sorted(
        attributes["references"]
        for attributes in graph["PROTEIN:A"]["REACTION:R"].values()
    ) == ["1234", "5678", "9999"]
    membership = next(iter(graph["PROTEIN:A"]["COMPLEX:C"].values()))
    assert json.loads(membership["qualifiers"])["required"] is True
    assert membership["predicate"] == "member_required::sign=unknown"
    binding = next(iter(graph["METABOLITE:Xm"]["PROTEIN:A"].values()))
    assert binding["predicate"] == "binds::sign=unsigned"
    assert binding["key"] == "upstream-record-key"
    assert graph.graph["source_snapshot_sha256"] == manifest["source_snapshot_sha256"]
    assert manifest["node_count"] == len(nodes)
    assert "unused_relation" in manifest["declared_source_relations"]
    for source_name, output_suffix in [
        ("nodes.parquet", ".nodes.parquet"),
        ("edges.parquet", ".rels.parquet"),
    ]:
        assert (graph_directory / source_name).read_bytes() == Path(
            f"{output_prefix}{output_suffix}"
        ).read_bytes()
    for artifact in manifest["outputs"].values():
        path = output_prefix.parent / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_trainer_consumes_signed_categories_and_stable_relation_mapping(tmp_path):
    graph_directory, nodes, edges, _ = _write_source(tmp_path)
    output_prefix = tmp_path / "adapted"
    adapt_physiology_graph(graph_directory, output_prefix)
    graph = load_multiplex_graph(
        Path(f"{output_prefix}.graphml"), text_encoder_model=None
    )
    relations = json.loads(Path(f"{output_prefix}.relations.json").read_text())
    assert graph.relation_index == {
        row["model_relation"]: row["index"] for row in relations
    }
    assert "regulates::sign=negative" in graph.relation_index
    assert "regulates::sign=positive" in graph.relation_index
    assert graph.data.edge_index.shape == (2, len(edges))
    assert graph.node_ids == nodes.node_id.tolist()
    assert graph.node_labels[1] == "A protein"
    model, partition, history, training_summary = train_scae_on_graph(
        graph,
        num_clusters=2,
        hidden_dims=(8,),
        type_embedding_dim=4,
        attr_encoder_dims=(4, 8, 4),
        max_epochs=1,
        device="cpu",
        verbose=False,
    )
    assert len(history) == 1
    assert partition.node_to_cluster.numel() == len(nodes)
    assert model.encoder.num_relations == len(relations)
    assert training_summary["epochs_trained"] == 1


def test_typed_ablation_changes_encoding_without_dropping_raw_signs(tmp_path):
    _, nodes, edges, declared_relations = _write_source(tmp_path)
    graph = build_adapted_graph(nodes, edges, declared_relations, relation_mode="typed")
    records = list(graph["PROTEIN:A"]["REACTION:R"].values())
    assert {record["predicate"] for record in records} == {"regulates"}
    assert {record["sign"] for record in records} == {-1.0, 1.0}


@pytest.mark.parametrize(
    "problem, expected_error",
    [
        ("duplicate", "Duplicate node_id"),
        ("endpoint", "Edge endpoints absent"),
        ("relation", "Undeclared relation"),
        ("sign", "Edge sign must"),
        ("diagnosis", "disease or nosology"),
        ("identifier", "nonempty strings"),
    ],
)
def test_adapter_rejects_invalid_source_structure(tmp_path, problem, expected_error):
    _, nodes, edges, declared_relations = _write_source(tmp_path)
    if problem == "duplicate":
        nodes = pd.concat([nodes, nodes.iloc[:1]], ignore_index=True)
    elif problem == "endpoint":
        edges.loc[0, "source_id"] = "missing"
    elif problem == "relation":
        edges.loc[0, "relation_type"] = "undeclared"
    elif problem == "sign":
        edges.loc[0, "sign"] = 0.5
    elif problem == "diagnosis":
        nodes.loc[0, "node_type"] = "disease"
    elif problem == "identifier":
        nodes.loc[0, "node_id"] = None
    with pytest.raises(ValueError, match=expected_error):
        build_adapted_graph(nodes, edges, declared_relations)


def test_release_manifest_and_explicit_pin_prevent_changed_inputs(tmp_path):
    graph_directory, _, _, _ = _write_source(tmp_path)
    source_manifest = {
        "version": "future-release",
        "files": {
            f"graph_split/{path.name}": {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            }
            for path in graph_directory.iterdir()
        },
    }
    (graph_directory.parent / "MANIFEST.json").write_text(json.dumps(source_manifest))
    output_prefix = tmp_path / "adapted"
    first = adapt_physiology_graph(graph_directory, output_prefix)
    assert first["source_release"]["version"] == "future-release"
    second = adapt_physiology_graph(
        graph_directory,
        output_prefix,
        expected_snapshot_sha256=first["source_snapshot_sha256"],
    )
    assert second["outputs"] == first["outputs"]
    previous_graph_bytes = Path(f"{output_prefix}.graphml").read_bytes()
    with pytest.raises(ValueError, match="requested pin"):
        adapt_physiology_graph(
            graph_directory, output_prefix, expected_snapshot_sha256="0" * 64
        )
    (graph_directory / "relation_types.json").write_text('["new_relation"]')
    with pytest.raises(ValueError, match="manifest mismatch"):
        adapt_physiology_graph(graph_directory, output_prefix)
    assert Path(f"{output_prefix}.graphml").read_bytes() == previous_graph_bytes


@pytest.mark.parametrize(
    "attributes, expected",
    [
        ({"predicate": "binds"}, "binds"),
        ({"relation": "activates"}, "activates"),
        ({"relation_type": "inhibits"}, "inhibits"),
        ({"predicate": "", "relation": "binds"}, "binds"),
        ({"predicate": "binds", "relation": "binds"}, "binds"),
        (
            {
                "model_relation": "regulates::sign=negative",
                "predicate": "regulates::sign=negative",
                "relation_type": "regulates",
                "relation": "regulates",
            },
            "regulates::sign=negative",
        ),
        ({}, "rel"),
    ],
)
def test_legacy_relation_aliases_remain_supported(attributes, expected):
    assert resolve_edge_relation(attributes) == expected


@pytest.mark.parametrize(
    "attributes",
    [
        {"relation": "binds", "relation_type": "activates"},
        {"predicate": "binds", "relation": "activates"},
        {"model_relation": "inhibits", "predicate": "activates"},
    ],
)
def test_conflicting_relation_aliases_fail_explicitly(attributes):
    with pytest.raises(ValueError):
        resolve_edge_relation(attributes)


def test_loader_retains_legacy_builder_relations(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        [("a", {"node_type": "gene"}), ("b", {"node_type": "protein"})]
    )
    graph.add_edge("a", "b", relation="activates")
    graph.add_edge("b", "a", relation_type="binds")
    graph.add_edge("a", "b", predicate="inhibits")
    path = tmp_path / "legacy.graphml"
    nx.write_graphml(graph, path)
    loaded = load_multiplex_graph(path, text_encoder_model=None)
    assert loaded.relation_index == {"activates": 0, "binds": 1, "inhibits": 2}
    assert len(set(loaded.data.edge_type.tolist())) == 3
