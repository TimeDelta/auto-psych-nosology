#!/usr/bin/env python3
"""Convert the official HPO release into CSVs used by the ontology augmenter."""
from __future__ import annotations

import csv
import pathlib
import sys

try:
    import networkx as nx  # noqa: F401  (obonet requires networkx)
    import obonet  # type: ignore
except ImportError:
    print(
        "This script requires 'obonet' (and networkx). Install via pip install obonet",
        file=sys.stderr,
    )
    sys.exit(1)


def convert(
    hp_obo: pathlib.Path,
    hpoa: pathlib.Path,
    out_dir: pathlib.Path,
    genes_to_pheno: pathlib.Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = obonet.read_obo(hp_obo)
    terms_path = out_dir / "hpo_terms.csv"
    with terms_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "parents", "synonyms"])
        for term_id, data in graph.nodes(data=True):
            name = data.get("name")
            if not name:
                continue  # skip obsolete/header nodes
            parents = [str(p) for p in graph.predecessors(term_id)]
            synonyms = []
            for syn in data.get("synonym", []):
                if isinstance(syn, dict):
                    desc = syn.get("desc")
                    if desc:
                        synonyms.append(desc)
                elif isinstance(syn, str):
                    parts = syn.split('"')
                    if len(parts) >= 3:
                        synonyms.append(parts[1])
            writer.writerow([term_id, name, "|".join(parents), "|".join(synonyms)])

    annotations_path = out_dir / "hpo_annotations.csv"
    with annotations_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["entity", "term"])
        # Primary annotations from phenotype.hpoa (disease-centric)
        with hpoa.open(encoding="utf-8") as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                cols = line.strip().split("\t")
                if len(cols) < 5:
                    continue
                disease_name = cols[1].strip()
                term = cols[3].strip() if cols[3].startswith("HP:") else cols[4].strip()
                if disease_name and term:
                    writer.writerow([disease_name, term])

        if genes_to_pheno and genes_to_pheno.exists():
            with genes_to_pheno.open(encoding="utf-8") as fin:
                for line in fin:
                    if line.startswith("#"):
                        continue
                    cols = line.strip().split("\t")
                    if len(cols) < 3:
                        continue
                    gene_symbol = cols[1].strip()
                    term = cols[2].strip()
                    if gene_symbol and term:
                        writer.writerow([gene_symbol, term])

    print(f"Saved {terms_path} and {annotations_path}")


if __name__ == "__main__":
    parser_args = sys.argv[1:]
    if len(parser_args) not in {3, 4}:
        print(
            "Usage: python prepare_hpo_csv.py hp.obo phenotype.hpoa [genes_to_phenotype.txt] output_dir",
            file=sys.stderr,
        )
        sys.exit(1)
    hp_path = pathlib.Path(parser_args[0])
    hpoa_path = pathlib.Path(parser_args[1])
    if len(parser_args) == 4:
        genes_path = pathlib.Path(parser_args[2])
        out_dir = pathlib.Path(parser_args[3])
    else:
        genes_path = None
        out_dir = pathlib.Path(parser_args[2])
    convert(hp_path, hpoa_path, out_dir, genes_to_pheno=genes_path)
