import json
from pathlib import Path

import networkx as nx

from align_partitions import infer_framework_labels_tailored


def _write_hpo_fixture(path: Path) -> Path:
    lines = [
        "id,name,parents,synonyms",
        "HP:0000739,Anxiety,,Anxiety|Generalized anxiety disorder|Excessive worry",
        "HP:0100754,Mania,,Mania|Manic episode|Grandiose mood",
        "HP:0002360,Sleep abnormality,,Sleep disturbance|Circadian rhythm disorder|Insomnia",
    ]
    path.write_text("\n".join(lines))
    return path


def test_infer_framework_labels_ontology_mapping(tmp_path):
    hpo_csv = _write_hpo_fixture(tmp_path / "hpo_terms.csv")

    G = nx.Graph()

    # Disease nodes without obvious keyword matches in their names
    G.add_node(
        "D_anx",
        name="Autonomic hyperarousal",
        node_type="disease",
        metadata=json.dumps({"note": "elevated sympathetic tone"}),
    )
    G.add_node(
        "D_mania",
        name="Episode with elevated mood",
        node_type="disease",
        metadata=json.dumps({"summary": "episodic expansive affect"}),
    )
    G.add_node(
        "D_sleep",
        name="Persistent nightly rest disturbance",
        node_type="disease",
    )
    G.add_node("D_none", name="Idiopathic condition", node_type="disease")

    # HPO phenotype nodes
    G.add_node(
        "HP:0000739",
        name="Anxiety",
        node_type="phenotype",
        ontology="hpo",
        ontology_id="HP:0000739",
    )
    G.add_node(
        "HP:0100754",
        name="Mania",
        node_type="phenotype",
        ontology="hpo",
        ontology_id="HP:0100754",
    )
    G.add_node(
        "HP:0002360",
        name="Sleep abnormality",
        node_type="phenotype",
        ontology="hpo",
        ontology_id="HP:0002360",
    )

    # Gene node to test propagation
    G.add_node("G_SLEEP", name="CLOCK", node_type="gene/protein")

    # Ontology edges
    G.add_edge("D_anx", "HP:0000739", relation="maps_to_hpo")
    G.add_edge("D_mania", "HP:0100754", relation="maps_to_hpo")
    G.add_edge("D_sleep", "HP:0002360", relation="maps_to_hpo")

    # Disease to gene relation for propagation
    G.add_edge("D_sleep", "G_SLEEP", relation="disease_protein")

    hitop, rdoc = infer_framework_labels_tailored(
        G, prop_depth=1, hpo_terms_path=hpo_csv
    )

    assert hitop["D_anx"] == "Internalizing"
    assert hitop["D_mania"] == "Thought Disorder"
    assert hitop["D_none"] == "Unspecified Clinical"

    assert rdoc["D_sleep"] == "Arousal/Regulatory Systems"
    assert rdoc["G_SLEEP"] == "Arousal/Regulatory Systems"
