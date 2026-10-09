import json
from dataclasses import replace

import networkx as nx
import polars as pl
import pytest

from create_graph import EntityRelationExtractor, KnowledgeGraphPipeline, PipelineConfig
from graph_quality import summarize_graph_quality


def make_ikraph_fixture(tmp_path):
    ikraph_directory = tmp_path / "ikraph"
    ikraph_directory.mkdir()
    nodes = [
        {
            "biokdeid": 1,
            "type": "Disease",
            "subtype": "bipolar disorder",
            "id": "MONDO:0005249",
            "common name": "Bipolar disorder",
        },
        {"biokdeid": 2, "type": "Chemical", "common name": "Lithium"},
        {"biokdeid": 3, "type": "Gene", "common name": "GSK3B"},
        {"biokdeid": 4, "type": "Gene", "common name": "Disconnected gene"},
        {"biokdeid": 5, "type": "Chemical", "common name": "Disconnected drug"},
    ]
    relation_types = [
        {"intRep": 1, "relType": "associated_with", "relPrec": 0.9},
        {"intRep": 2, "relType": "inhibits", "relPrec": 0.85},
        {"intRep": 3, "relType": "activates", "relPrec": 0.8},
    ]
    database_edges = [
        {
            "node_one_id": 1,
            "node_two_id": 2,
            "relationship_type": 1,
            "prob": 0.95,
            "score": 0.9,
            "direction": "12",
            "source": "fixture_db",
        },
        {
            "node_one_id": 1,
            "node_two_id": 3,
            "relationship_type": 1,
            "prob": 0.9,
            "score": 0.8,
            "direction": "12",
            "source": "fixture_db",
        },
        {
            "node_one_id": 2,
            "node_two_id": 3,
            "relationship_type": 2,
            "prob": 0.8,
            "score": 0.7,
            "direction": "12",
            "source": "fixture_db",
        },
        {
            "node_one_id": 4,
            "node_two_id": 5,
            "relationship_type": 2,
            "prob": 0.99,
            "score": 0.9,
            "direction": "12",
            "source": "fixture_db",
        },
    ]
    pubmed_edges = [{"id": "2.3.3.2.12.fixture", "list": [[0.7, "paper-A", 0.8, True]]}]
    for filename, records in (
        ("NER_ID_dict_cap_final.json", nodes),
        ("RelTypeInt.json", relation_types),
        ("DBRelations.json", database_edges),
        ("PubMedList.json", pubmed_edges),
    ):
        (ikraph_directory / filename).write_text(json.dumps(records), encoding="utf-8")
    return PipelineConfig(
        ikraph_dir=ikraph_directory,
        output_dir=tmp_path,
        output_prefix="ikraph_test",
        neighbor_hops=2,
    )


def test_two_hop_extraction_retains_biomedical_edges_without_diagnosis_hub(tmp_path):
    configuration = make_ikraph_fixture(tmp_path)
    summary = KnowledgeGraphPipeline().run(configuration)
    assert summary == {"n_nodes": 2, "n_edges": 2}
    graph_path = configuration.resolved_output_prefix.with_suffix(".graphml")
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    assert set(graph) == {"2", "3"}
    assert graph.nodes["2"]["name"] == "Lithium"
    assert graph.nodes["3"]["node_type"] == "Gene"
    assert {attributes["relation"] for _, _, attributes in graph.edges(data=True)} == {
        "inhibits",
        "activates",
    }
    for _, _, attributes in graph.edges(data=True):
        assert attributes["probability"] == 0.8
        assert attributes["score"] == 0.7
        assert attributes["direction"] == "12"
        assert json.loads(attributes["source_record_json"])
        assert attributes["source_file"] in {"DBRelations.json", "PubMedList.json"}
    pubmed_attributes = next(
        attributes
        for _, _, attributes in graph.edges(data=True)
        if attributes["source_file"] == "PubMedList.json"
    )
    assert pubmed_attributes["source_record_id"] == "2.3.3.2.12.fixture"
    assert pubmed_attributes["correlation"] == "2"
    assert pubmed_attributes["evidence_count"] == 1

    node_table = pl.read_parquet(
        configuration.resolved_output_prefix.with_suffix(".nodes.parquet")
    )
    edge_table = pl.read_parquet(
        configuration.resolved_output_prefix.with_suffix(".rels.parquet")
    )
    assert set(node_table["node_index"].to_list()) == {2, 3}
    assert (
        edge_table.height == 2
    )  # repeated expansion passes must not duplicate support
    assert set(edge_table["source_index"].to_list()) <= {2, 3}
    assert set(edge_table["target_index"].to_list()) <= {2, 3}
    quality_report = json.loads(
        configuration.resolved_output_prefix.with_suffix(".quality.json").read_text()
    )
    assert quality_report["n_nodes"] == graph.number_of_nodes()
    assert quality_report["n_edges"] == graph.number_of_edges()
    assert quality_report["nosology_flagged_nodes"] == 0
    assert (
        quality_report["edge_attribute_coverage"]["source_record_json"]["fraction"]
        == 1.0
    )


def test_one_hop_degeneracy_overwrites_previous_graph_artifacts(tmp_path):
    configuration = make_ikraph_fixture(tmp_path)
    KnowledgeGraphPipeline().run(configuration)
    configuration = replace(configuration, neighbor_hops=1)
    summary = KnowledgeGraphPipeline().run(configuration)
    assert summary == {"n_nodes": 0, "n_edges": 0}
    graph = nx.read_graphml(
        configuration.resolved_output_prefix.with_suffix(".graphml")
    )
    assert graph.number_of_nodes() == 0
    assert pl.read_parquet(
        configuration.resolved_output_prefix.with_suffix(".rels.parquet")
    ).is_empty()
    report = json.loads(
        configuration.resolved_output_prefix.with_suffix(".quality.json").read_text()
    )
    assert report["n_edges"] == 0


def test_database_relid_preserves_relation_sign_direction_and_source(tmp_path):
    configuration = make_ikraph_fixture(tmp_path)
    source_path = configuration.ikraph_dir / "DBRelations.json"
    records = json.loads(source_path.read_text())
    records[2] = {
        "node_one_id": 2,
        "node_two_id": 3,
        "relID": "2.3.2.0.21.curated",
        "prob": 0.8,
        "score": 0.7,
    }
    source_path.write_text(json.dumps(records))
    KnowledgeGraphPipeline().run(configuration)
    graph = nx.read_graphml(
        configuration.resolved_output_prefix.with_suffix(".graphml"),
        force_multigraph=True,
    )
    attributes = next(
        attributes
        for _, _, attributes in graph.edges(data=True)
        if attributes["source_file"] == "DBRelations.json"
    )
    assert attributes["relation"] == "inhibits"
    assert attributes["direction"] == "21"
    assert attributes["correlation"] == "0"
    assert attributes["edge_source"] == "curated"


def test_invalid_endpoint_fails_instead_of_creating_placeholder_node(tmp_path):
    configuration = make_ikraph_fixture(tmp_path)
    extractor = EntityRelationExtractor(configuration.to_extraction_config())
    nodes, edges = extractor.build_subgraph()
    nodes = nodes.filter(pl.col("node_index") == 2)
    with pytest.raises(ValueError, match="Edge endpoint missing"):
        extractor.to_networkx(nodes, edges)


def test_synthetic_reverse_edges_are_marked_and_counted(tmp_path):
    configuration = replace(make_ikraph_fixture(tmp_path), include_reverse_edges=True)
    summary = KnowledgeGraphPipeline().run(configuration)
    assert summary == {"n_nodes": 2, "n_edges": 4}
    report = json.loads(
        configuration.resolved_output_prefix.with_suffix(".quality.json").read_text()
    )
    assert report["synthetic_reverse_edges"] == 2
    assert (
        pl.read_parquet(
            configuration.resolved_output_prefix.with_suffix(".rels.parquet")
        ).height
        == 2
    )


def test_quality_report_handles_empty_and_collapsed_graphs():
    empty_report = summarize_graph_quality(nx.MultiDiGraph())
    assert empty_report["largest_component_fraction"] == 0.0
    graph = nx.MultiDiGraph()
    graph.add_node("diagnosis", name="Bipolar disorder", node_type="Disease")
    graph.add_node("isolate", name="Biomarker", node_type="Gene")
    graph.add_edge(
        "diagnosis", "diagnosis", relation="associated_with", synthetic_reverse=True
    )
    report = summarize_graph_quality(graph)
    assert report["isolated_nodes"] == 1
    assert report["self_loop_edges"] == 1
    assert report["synthetic_reverse_edges"] == 1
    assert report["nosology_flagged_nodes"] == 1
    assert report["edge_attribute_coverage"]["probability"]["count"] == 0


def test_hops_must_be_positive(tmp_path):
    configuration = replace(make_ikraph_fixture(tmp_path), neighbor_hops=0)
    extractor = EntityRelationExtractor(configuration.to_extraction_config())
    with pytest.raises(ValueError, match="neighbor_hops"):
        extractor.build_subgraph()
