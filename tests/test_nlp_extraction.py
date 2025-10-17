import json
from pathlib import Path

import polars as pl

from nlp_extraction import EntityRelationExtractor, ExtractionConfig


def _prepare_dataset(tmp_path: Path) -> tuple[Path, Path]:
    kg_rows = [
        {
            "relation": "drug_disease",
            "display_relation": "treats",
            "x_index": "10",
            "x_id": "DB0001",
            "x_type": "drug",
            "x_name": "Fluoxetine",
            "x_source": "DrugBank",
            "y_index": "200",
            "y_id": "MONDO:0005130",
            "y_type": "disease",
            "y_name": "Major depressive disorder",
            "y_source": "MONDO",
        },
        {
            "relation": "drug_disease",
            "display_relation": "treats",
            "x_index": "11",
            "x_id": "DB0002",
            "x_type": "drug",
            "x_name": "Lithium",
            "x_source": "DrugBank",
            "y_index": "201",
            "y_id": "MONDO:0005249",
            "y_type": "disease",
            "y_name": "Bipolar disorder",
            "y_source": "MONDO",
        },
        {
            "relation": "disease_gene",
            "display_relation": "associated_with",
            "x_index": "200",
            "x_id": "MONDO:0005130",
            "x_type": "disease",
            "x_name": "Major depressive disorder",
            "x_source": "MONDO",
            "y_index": "2",
            "y_id": "ENSG000001",
            "y_type": "gene/protein",
            "y_name": "BDNF",
            "y_source": "NCBI",
        },
        {
            "relation": "drug_gene",
            "display_relation": "targets",
            "x_index": "10",
            "x_id": "DB0001",
            "x_type": "drug",
            "x_name": "Fluoxetine",
            "x_source": "DrugBank",
            "y_index": "2",
            "y_id": "ENSG000001",
            "y_type": "gene/protein",
            "y_name": "BDNF",
            "y_source": "NCBI",
        },
    ]
    kg_headers = kg_rows[0].keys()
    kg_lines = [",".join(kg_headers)]
    for row in kg_rows:
        kg_lines.append(",".join(row[h] for h in kg_headers))
    kg_path = tmp_path / "kg.csv"
    kg_path.write_text("\n".join(kg_lines))

    disease_rows = [
        {
            "node_index": "200",
            "mondo_id": "MONDO:0005130",
            "mondo_name": "Major depressive disorder",
            "group_name_bert": "major depressive disorder",
            "mondo_definition": "A common mood disorder",
            "umls_description": "",
            "orphanet_definition": "",
            "orphanet_prevalence": "",
            "orphanet_epidemiology": "",
            "orphanet_clinical_description": "",
            "orphanet_management_and_treatment": "",
            "mayo_symptoms": "",
            "mayo_causes": "",
            "mayo_risk_factors": "",
            "mayo_complications": "",
            "mayo_prevention": "",
            "mayo_see_doc": "",
        },
        {
            "node_index": "201",
            "mondo_id": "MONDO:0005249",
            "mondo_name": "Bipolar disorder",
            "group_name_bert": "bipolar disorder",
            "mondo_definition": "Mood disorder with mania",
            "umls_description": "",
            "orphanet_definition": "",
            "orphanet_prevalence": "",
            "orphanet_epidemiology": "",
            "orphanet_clinical_description": "",
            "orphanet_management_and_treatment": "",
            "mayo_symptoms": "",
            "mayo_causes": "",
            "mayo_risk_factors": "",
            "mayo_complications": "",
            "mayo_prevention": "",
            "mayo_see_doc": "",
        },
        {
            "node_index": "202",
            "mondo_id": "MONDO:1234",
            "mondo_name": "Hypertension",
            "group_name_bert": "cardiovascular",
            "mondo_definition": "Raised blood pressure",
            "umls_description": "",
            "orphanet_definition": "",
            "orphanet_prevalence": "",
            "orphanet_epidemiology": "",
            "orphanet_clinical_description": "",
            "orphanet_management_and_treatment": "",
            "mayo_symptoms": "",
            "mayo_causes": "",
            "mayo_risk_factors": "",
            "mayo_complications": "",
            "mayo_prevention": "",
            "mayo_see_doc": "",
        },
    ]
    disease_headers = disease_rows[0].keys()
    disease_lines = [",".join(disease_headers)]
    for row in disease_rows:
        disease_lines.append(",".join(row[h] for h in disease_headers))
    (tmp_path / "modalities").mkdir()
    disease_path = tmp_path / "modalities" / "disease_feature_base.csv"
    disease_path.write_text("\n".join(disease_lines))

    drug_headers = [
        "node_index",
        "description",
        "half_life",
        "indication",
        "mechanism_of_action",
        "protein_binding",
        "pharmacodynamics",
        "state",
        "atc_1",
        "atc_2",
        "atc_3",
        "atc_4",
        "category",
        "group",
        "pathway",
        "molecular_weight",
        "tpsa",
        "clogp",
        "drugbank_ids",
        "generic_name",
        "smiles",
        "sequences",
    ]
    drug_rows = [
        [
            "10",
            "SSRI antidepressant",
            "",
            "Major depression",
            "Serotonin reuptake inhibition",
            "",
            "",
            "solid",
            "",
            "",
            "",
            "",
            "Psychiatric",
            "approved",
            "",
            "345.4",
            "",
            "",
            "DB0001",
            "Fluoxetine",
            "",
            "",
        ],
        [
            "11",
            "Mood stabilizer",
            "",
            "Bipolar disorder",
            "Modulates neurotransmitters",
            "",
            "",
            "solid",
            "",
            "",
            "",
            "",
            "Psychiatric",
            "approved",
            "",
            "25.0",
            "",
            "",
            "DB0002",
            "Lithium",
            "",
            "",
        ],
    ]
    drug_lines = [",".join(drug_headers)]
    drug_lines.extend(",".join(row) for row in drug_rows)
    (tmp_path / "modalities" / "drug_feature_base.csv").write_text(
        "\n".join(drug_lines)
    )

    protein_headers = [
        "node_index",
        "protein_id",
        "protein_name",
        "protein_seq",
        "fasta_id",
        "fasta_description",
        "ncbi_summary",
    ]
    protein_rows = [
        [
            "2",
            "999",
            "BDNF",
            "MVGGELK",
            "sp|Q9GZ",
            "Brain-derived neurotrophic factor",
            "Supports neurons",
        ]
    ]
    protein_lines = [",".join(protein_headers)]
    protein_lines.extend(",".join(row) for row in protein_rows)
    (tmp_path / "modalities" / "protein_aminoacid_sequence.csv").write_text(
        "\n".join(protein_lines)
    )

    return kg_path, tmp_path


def test_entity_relation_extractor_filters_psychiatric_nodes(tmp_path):
    kg_path, data_dir = _prepare_dataset(tmp_path)
    config = ExtractionConfig(kg_path=kg_path, data_dir=data_dir)
    extractor = EntityRelationExtractor(config)
    nodes_df, edges_df = extractor.build_subgraph()

    node_indices = set(nodes_df.select("node_index").to_series().to_list())
    assert node_indices == {10, 11, 200, 201, 2}

    psy_nodes = (
        nodes_df.filter(pl.col("is_psychiatric"))
        .select("node_index")
        .to_series()
        .to_list()
    )
    assert set(psy_nodes) == {200, 201}

    assert all(
        score >= 0.6
        for score in nodes_df.filter(pl.col("node_index").is_in([200, 201]))
        .select("psy_score")
        .to_series()
        .to_list()
    )

    evidence_payload = json.loads(
        nodes_df.filter(pl.col("node_index") == 200).row(0, named=True)["psy_evidence"]
    )
    assert "ontology" in evidence_payload

    non_psy_drug = nodes_df.filter(pl.col("node_index") == 10).row(0, named=True)
    assert not non_psy_drug["is_psychiatric"]
    assert non_psy_drug["psy_score"] < 0.6
    assert json.loads(non_psy_drug["drug_metadata"]) != {}

    edges = edges_df.select(["source_index", "target_index", "relation"]).to_dicts()
    assert len(edges) == 3  # the drug-gene edge without disease should be dropped
    assert all(
        {edge["source_index"], edge["target_index"]} & {200, 201} for edge in edges
    )
