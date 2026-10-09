from pathlib import Path
from xml.etree import ElementTree as ET

import networkx as nx
import polars as pl

from create_graph import KnowledgeGraphPipeline, PipelineConfig


def _prepare_dataset(
    tmp_path: Path, *, use_diagnosis_names: bool = False
) -> PipelineConfig:
    # Non-diagnostic symptom names survive the nosology filter in the artifact test.
    first_name = (
        "Major depressive disorder"
        if use_diagnosis_names
        else "Persistent depressive symptoms"
    )
    second_name = (
        "Bipolar disorder" if use_diagnosis_names else "Recurrent mood symptoms"
    )
    kg_lines = [
        "relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source",
        f"drug_disease,treats,10,DB0001,drug,Fluoxetine,DrugBank,200,MONDO:0005130,disease,{first_name},MONDO",
        f"drug_disease,treats,11,DB0002,drug,Lithium,DrugBank,201,MONDO:0005249,disease,{second_name},MONDO",
        f"disease_gene,associated_with,200,MONDO:0005130,disease,{first_name},MONDO,2,ENSG000001,gene/protein,BDNF,NCBI",
    ]
    kg_path = tmp_path / "kg.csv"
    kg_path.write_text("\n".join(kg_lines))

    modalities = tmp_path / "modalities"
    modalities.mkdir()

    disease_lines = [
        "node_index,mondo_id,mondo_name,group_name_bert,mondo_definition,umls_description,orphanet_definition,orphanet_prevalence,orphanet_epidemiology,orphanet_clinical_description,orphanet_management_and_treatment,mayo_symptoms,mayo_causes,mayo_risk_factors,mayo_complications,mayo_prevention,mayo_see_doc",
        f"200,MONDO:0005130,{first_name},major depressive disorder,Common mood disorder,,,,,,,,,,,",
        f"201,MONDO:0005249,{second_name},bipolar disorder,Mood disorder with mania,,,,,,,,,,,",
    ]
    (modalities / "disease_feature_base.csv").write_text("\n".join(disease_lines))

    drug_lines = [
        "node_index,description,half_life,indication,mechanism_of_action,protein_binding,pharmacodynamics,state,atc_1,atc_2,atc_3,atc_4,category,group,pathway,molecular_weight,tpsa,clogp,drugbank_ids,generic_name,smiles,sequences",
        "10,SSRI antidepressant,,,Serotonin reuptake inhibition,,,,,,,,Psychiatric,approved,,345.4,,,DB0001,Fluoxetine,,",
        "11,Mood stabilizer,,,Modulates neurotransmitters,,,,,,,,Psychiatric,approved,,25.0,,,DB0002,Lithium,,",
    ]
    (modalities / "drug_feature_base.csv").write_text("\n".join(drug_lines))

    protein_lines = [
        "node_index,protein_id,protein_name,protein_seq,fasta_id,fasta_description,ncbi_summary",
        "2,999,BDNF,MVGGELK,sp|Q9GZ,Brain-derived neurotrophic factor,Supports neurons",
    ]
    (modalities / "protein_aminoacid_sequence.csv").write_text("\n".join(protein_lines))

    return PipelineConfig(
        kg_path=kg_path,
        data_dir=tmp_path,
        output_prefix="unit_test_graph",
        output_dir=tmp_path,
    )


def test_pipeline_produces_artifacts(tmp_path):
    config = _prepare_dataset(tmp_path)
    pipeline = KnowledgeGraphPipeline()
    summary = pipeline.run(config)

    assert summary["n_nodes"] == 5
    assert summary["n_edges"] == 3

    nodes_path = config.resolved_output_prefix.with_suffix(".nodes.parquet")
    edges_path = config.resolved_output_prefix.with_suffix(".rels.parquet")
    graph_path = config.resolved_output_prefix.with_suffix(".graphml")
    weighted_path = config.resolved_output_prefix.with_suffix(".weighted.graphml")

    assert nodes_path.exists()
    assert edges_path.exists()
    assert graph_path.exists()
    assert weighted_path.exists()

    nodes_df = pl.read_parquet(nodes_path)
    node_indices = set(nodes_df.select("node_index").to_series().to_list())
    assert node_indices == {10, 11, 200, 201, 2}

    assert {"psy_score", "psy_evidence", "ontology_flag", "drug_flag"}.issubset(
        set(nodes_df.columns)
    )
    positive_min = (
        nodes_df.filter(pl.col("node_index").is_in([200, 201]))
        .select(pl.col("psy_score").min())
        .item()
    )
    assert positive_min >= 0.6

    inherited_drug_score = (
        nodes_df.filter(pl.col("node_index") == 10)
        .select(pl.col("psy_score").min())
        .item()
    )
    # Neighbor relevance is propagated to drugs; it is not a direct disease score.
    assert (
        inherited_drug_score
        == nodes_df.filter(pl.col("node_index") == 200)["psy_score"].item()
    )

    drug_row = nodes_df.filter(pl.col("node_index") == 10).row(0, named=True)
    assert drug_row["psy_score_direct"] == 0.0
    assert drug_row["psy_score_neighbor_mean"] == inherited_drug_score
    assert drug_row["psy_neighbor_count"] == 1

    tree = ET.parse(weighted_path)
    root = tree.getroot()
    node_ids = {
        node.attrib.get("id")
        for node in root.findall(
            "{http://graphml.graphdrawing.org/xmlns}graph/{http://graphml.graphdrawing.org/xmlns}node"
        )
    }
    assert "200" in node_ids

    ns = "{http://graphml.graphdrawing.org/xmlns}"
    edges = root.findall(f"{ns}graph/{ns}edge")
    assert edges, "GraphML should contain edges"
    weight_values = []
    weight_keys = {
        key.attrib["id"]
        for key in root.findall(f"{ns}key")
        if key.attrib.get("attr.name") == "weight"
    }
    for edge in edges:
        for data in edge.findall(f"{ns}data"):
            if data.attrib.get("key") in weight_keys:
                weight_values.append(float(data.text))
    assert weight_values, "No edge weights found in GraphML"
    assert max(weight_values) >= 0.1
    assert min(weight_values) >= 0.0


def test_pipeline_excludes_named_diagnoses(tmp_path):
    config = _prepare_dataset(tmp_path, use_diagnosis_names=True)
    KnowledgeGraphPipeline().run(config)
    nodes_df = pl.read_parquet(
        config.resolved_output_prefix.with_suffix(".nodes.parquet")
    )
    assert nodes_df.is_empty()
    graph = nx.read_graphml(config.resolved_output_prefix.with_suffix(".graphml"))
    assert "200" not in graph and "201" not in graph
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_primekg_expansion_retains_real_links_between_seed_neighbors(tmp_path):
    config = _prepare_dataset(tmp_path, use_diagnosis_names=True)
    config.neighbor_hops = 2
    with config.kg_path.open("a") as edge_file:
        edge_file.write(
            "\ndrug_protein,inhibits,10,DB0001,drug,Fluoxetine,DrugBank,2,ENSG000001,gene/protein,BDNF,NCBI"
        )
    summary = KnowledgeGraphPipeline().run(config)
    assert summary == {"n_nodes": 2, "n_edges": 1}
    graph = nx.read_graphml(config.resolved_output_prefix.with_suffix(".graphml"))
    assert set(graph) == {"10", "2"}
    assert graph.edges["10", "2"]["source_file"] == "kg.csv"
    assert graph.edges["10", "2"]["source_record_index"] == 3
