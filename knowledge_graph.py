"""
knowledge_graph_free.py
=======================

This module provides a simplified alternative to the original `knowledge_graph.py` that
uses only open‑source, zero‑cost NLP models for entity and relation extraction. The
original pipeline relied on OpenAI's proprietary API for information extraction; this
version removes that dependency and instead employs the HuggingFace `transformers`
library along with freely available biomedical models for named‑entity recognition
(NER). Relations are inferred in a naive manner by linking all pairs of entities
that co‑occur within the same document. While the extraction quality is necessarily
lower than a tuned large language model, this approach respects the zero‑budget
constraint and provides a working proof‑of‑concept for downstream graph construction
and partitioning experiments.

To run this script you will need to install the additional dependencies listed in
`requirements.txt`, in particular `transformers` and a model checkpoint such as
`d4data/biomedical-ner-all`. At runtime the model weights will be downloaded
automatically from the HuggingFace Hub (internet connectivity is required when
running for the first time).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import httpx
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from readability import Document
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

NODE_TYPES: Set[str] = {
    "Symptom",
    "Diagnosis",
    "RDoC_Construct",
    "HiTOP_Component",
    "Biomarker",
    "Treatment",
    "Task",
    "Measure",
}

REL_TYPES: Set[str] = {
    "supports",
    "contradicts",
    "predicts",
    "co_occurs",
    "treats",
    "biomarker_for",
    "measure_of",
}

DIAGNOSIS_MASK_LEXICON = {
    "major depressive disorder": [
        "MDD",
        "major depression",
        "unipolar depression",
        "clinical depression",
        "recurrent depression",
    ],
    "bipolar disorder": [
        "BD",
        "bipolar affective disorder",
        "manic depression",
        "bipolar I",
        "bipolar II",
    ],
    "schizophrenia": [
        "SCZ",
        "schizophrenic psychosis",
        "schizoaffective (if not separate)",
        "chronic schizophrenia",
    ],
    "anxiety disorders": [
        "GAD",
        "generalized anxiety disorder",
        "panic disorder",
        "social anxiety disorder",
        "social phobia",
    ],
    "PTSD": [
        "posttraumatic stress disorder",
        "post-traumatic stress",
        "combat stress",
        "traumatic stress disorder",
    ],
    "OCD": [
        "obsessive compulsive disorder",
        "obsessive–compulsive",
        "OCD spectrum",
    ],
    "ADHD": [
        "attention deficit hyperactivity disorder",
        "ADD",
        "attention-deficit disorder",
    ],
}

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("punkt_tab")
except:
    nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


class NodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(
        ..., description="Primary name for the entity as used in the paper."
    )
    node_type: str = Field(..., description=f"One of {sorted(list(NODE_TYPES))}")
    synonyms: List[str] = Field(default_factory=list)
    normalizations: Dict[str, str] = Field(
        default_factory=dict,
        description='UMLS or other dictionary normalizations, e.g. {"UMLS": "C0005586"}',
    )


class RelationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(..., description="Canonical name of subject node")
    predicate: str = Field(..., description=f"One of {sorted(list(REL_TYPES))}")
    obj: str = Field(..., alias="object", description="Canonical name of object node")
    directionality: str = Field(default="directed", description="directed|undirected")
    evidence_span: str = Field(
        default="", description="Short quote from text supporting this"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    qualifiers: Dict[str, Any] = Field(default_factory=dict)


class PaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    doi: Optional[str] = None
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    nodes: List[NodeRecord]
    relations: List[RelationRecord]


def extraction_json_schema() -> Dict[str, Any]:
    # doesn't need input because Pydantic BaseModel class has a classmethod .model_json_schema()
    # that generates a full JSON Schema describing its fields and nested models
    return PaperExtraction.model_json_schema()


OPENALEX_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_FILTER = (
    "from_publication_date:2010-01-01,primary_location.is_oa:true,language:en"
)
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def uninvert_openalex_abstract(inv: Dict[str, List[int]]) -> str:
    """Reconstruct OpenAlex inverted index into a plain abstract string."""
    if not inv:
        return ""
    pos2tok: Dict[int, str] = {}
    for tok, idxs in inv.items():
        for i in idxs:
            pos2tok[i] = tok
    return " ".join(pos2tok[i] for i in range(min(pos2tok), max(pos2tok) + 1))


def fetch_openalex_page(
    query: str,
    filters: str,
    cursor: str = "*",
    per_page: int = 50,
    sort: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a single page of search results from OpenAlex."""
    params = {
        "search": query,
        "filter": filters,
        "per_page": per_page,
        "cursor": cursor,
    }
    if sort:
        params["sort"] = sort
    r = httpx.get(OPENALEX_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_top_n(query: str, filters: str, sort: str, n: int) -> List[Dict[str, Any]]:
    """sort (cited_by_count:desc OR publication_date:desc)"""
    out: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = min(200, max(25, n))
    while len(out) < n:
        page = fetch_openalex_page(query, filters, cursor, per_page=per_page, sort=sort)
        results = page.get("results", [])
        if not results:
            break
        for w in results:
            abstract = uninvert_openalex_abstract(
                w.get("abstract_inverted_index") or {}
            )
            out.append(
                {
                    "id": w.get("id"),
                    "doi": w.get("doi"),
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "venue": (w.get("host_venue") or {}).get("display_name"),
                    "abstract": abstract,
                    "best_oa_location": (w.get("best_oa_location") or {}),
                    "open_access": (w.get("open_access") or {}),
                    "primary_location": (w.get("primary_location") or {}),
                    "cited_by_count": w.get("cited_by_count"),
                    "publication_date": w.get("publication_date"),
                }
            )
            if len(out) >= n:
                break
        cursor = page.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return out


PMC_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def try_pmc_fulltext(pmcid: str) -> Optional[str]:
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    r = httpx.get(PMC_EUTILS, params=params, timeout=60)
    if r.status_code != 200 or not r.text:
        return None
    soup = BeautifulSoup(r.text, "lxml-xml")
    parts: List[str] = []
    for tag in soup.find_all(["abstract", "p", "sec", "title"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts) if parts else None


def extract_text_from_pdf_bytes(content: bytes) -> Optional[str]:
    try:
        text = pdf_extract_text(BytesIO(content))
        text = re.sub(r"\s+\n", "\n", text)
        return text.strip()
    except Exception:
        return None


def extract_text_from_html_bytes(content: bytes, base_url: str = "") -> Optional[str]:
    try:
        html = content.decode("utf-8", errors="ignore")
        doc = Document(html)
        main_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(main_html, "lxml")
        for bad in soup(["script", "style", "noscript"]):
            bad.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return None


def resolve_text_and_download(rec: Dict[str, Any]) -> Optional[str]:
    """Return full text string if available, else None."""
    joined = json.dumps(rec, ensure_ascii=False)
    pmcid = re.search(r"PMCID[:\s]*PMC(\d+)", joined, flags=re.I)
    if pmcid:
        txt = try_pmc_fulltext("PMC" + pmcid.group(1))
        if txt and len(txt) > 1000:
            return txt

    loc = rec.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf")
    html_url = loc.get("url")

    if pdf_url:
        try:
            resp = httpx.get(pdf_url, timeout=90, follow_redirects=True)
            if resp.status_code == 200 and (
                "pdf" in resp.headers.get("content-type", "").lower()
                or pdf_url.lower().endswith(".pdf")
            ):
                txt = extract_text_from_pdf_bytes(resp.content)
                if txt and len(txt) > 1000:
                    return txt
        except Exception:
            pass

    if html_url:
        try:
            resp = httpx.get(html_url, timeout=90, follow_redirects=True)
            if resp.status_code == 200 and (
                "html" in resp.headers.get("content-type", "").lower()
            ):
                txt = extract_text_from_html_bytes(resp.content, html_url)
                if txt and len(txt) > 1000:
                    return txt
        except Exception:
            pass

    prim = rec.get("primary_location") or {}
    for key in ("pdf_url", "landing_page_url", "source_url", "url"):
        url = prim.get(key)
        if not url:
            continue
        try:
            resp = httpx.get(url, timeout=90, follow_redirects=True)
            ctype = resp.headers.get("content-type", "").lower()
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                txt = extract_text_from_pdf_bytes(resp.content)
                if txt and len(txt) > 1000:
                    return txt
            if "html" in ctype:
                txt = extract_text_from_html_bytes(resp.content, url)
                if txt and len(txt) > 1000:
                    return txt
        except Exception:
            continue
    return None


RESULTS_HDR_RE = re.compile(
    r"^\s*(results?|findings?|outcomes?)\s*[:\-]?\s*$", flags=re.I | re.M
)
NEXT_SECTION_RE = re.compile(
    r"^\s*(discussion|conclusions?|limitations?|general discussion|overview|references?|acknowledg(e)?ments?|supplementary)\s*[:\-]?\s$",
    flags=re.I | re.M,
)


def extract_results_only(fulltext: str) -> Optional[str]:
    """Heuristic: take the text from 'Results/Findings/Outcomes' to the next major section."""
    if not fulltext:
        return None
    # normalize newlines and collapse spaces
    doc = re.sub(r"\r", "\n", fulltext)
    doc = re.sub(r"[ \t]+", " ", doc)
    m = RESULTS_HDR_RE.search(doc)
    if not m:
        return None
    start = m.start()
    # find next section header after start
    m2 = NEXT_SECTION_RE.search(doc, pos=start + 1)
    end = m2.start() if m2 else len(doc)
    chunk = doc[start:end].strip()
    # avoid returning tiny strings that are likely noise
    return chunk if len(chunk) > 400 else None


# Aggregation strategy merges sub‑tokens into contiguous entities
_ner_pipeline = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",
)


def categorize_entity(label: str) -> str:
    """Map a coarse entity label into one of our predefined node types.

    The biomedical NER model outputs labels such as 'CHEMICAL', 'DISEASE',
    'PROTEIN', etc.  This function maps those labels onto the simplified set of
    categories used by our knowledge graph. The mapping is heuristic and may
    misclassify edge cases, but it provides a starting point. Unknown labels
    default to 'Symptom'.
    """
    lbl = label.upper()
    if "DISEASE" in lbl or "DISORDER" in lbl or "SYNDROME" in lbl:
        return "Diagnosis"
    if "CHEMICAL" in lbl or "DRUG" in lbl or "MED" in lbl:
        return "Treatment"
    if "GENE" in lbl or "PROTEIN" in lbl or "CELL" in lbl:
        return "Biomarker"
    if "BEHAVIOR" in lbl or "SYMPTOM" in lbl:
        return "Symptom"
    # default fallback
    return "Symptom"


_RELATION_MODE = "nli"
_NLI_MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"
_nli_tok = None
_nli_model = None
_ID2LBL = {0: "contradiction", 1: "neutral", 2: "entailment"}


# hypothesis templates: not used for exact matching. used or replacement by
# detected objects to test for entailment and use that replace relation type
_REL_TEMPLATES = {
    "treats": "{SUBJ} treats {OBJ}.",
    "biomarker_for": "{SUBJ} is a biomarker for {OBJ}.",
    "measure_of": "{SUBJ} is a measure of {OBJ}.",
    "predicts": "{SUBJ} predicts {OBJ}.",
    "supports": "{SUBJ} is positively associated with {OBJ}.",
    "contradicts": "{SUBJ} is negatively associated with {OBJ}.",
}


_REL_TEMPLATES_REV = {
    "treats": "{OBJ} treats {SUBJ}.",
    "biomarker_for": "{OBJ} is a biomarker for {SUBJ}.",
    "measure_of": "{OBJ} is a measure of {SUBJ}.",
    "predicts": "{OBJ} predicts {SUBJ}.",
}


def _init_nli():
    global _nli_tok, _nli_model
    if _nli_tok is None or _nli_model is None:
        _nli_tok = AutoTokenizer.from_pretrained(
            _NLI_MODEL_NAME, token=None, local_files_only=False
        )
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            _NLI_MODEL_NAME, token=None, use_safetensors=True
        )
        _nli_model.eval()


def _nli_score(premise: str, hypothesis: str) -> float:
    """Return entailment probability for (premise -> hypothesis)."""
    with torch.no_grad():
        enc = _nli_tok(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        )
        out = _nli_model(**enc).logits.softmax(-1)[0]
        # entailment index for MNLI heads is 2
        return float(out[2].item())


def _pick_sentence_context(text: str, s1: str, s2: str, window: int = 2) -> str:
    """Grab the sentence(s) where both entities appear, with small window."""
    sents = sent_tokenize(text)
    idxs = [i for i, s in enumerate(sents) if (s1 in s and s2 in s)]
    if not idxs:
        # fallback: nearest pair of sentences containing each entity
        idx1 = next((i for i, s in enumerate(sents) if s1 in s), None)
        idx2 = next((i for i, s in enumerate(sents) if s2 in s), None)
        if idx1 is None or idx2 is None:
            return text[:2000]  # worst-case: cap
        lo, hi = sorted([idx1, idx2])
        lo = max(0, lo - window)
        hi = min(len(sents), hi + window + 1)
        return " ".join(sents[lo:hi])
    lo = max(0, min(idxs) - window)
    hi = min(len(sents), max(idxs) + window + 1)
    return " ".join(sents[lo:hi])


def classify_relation_via_nli(context: str, subj: str, obj: str) -> str:
    """Return one of REL_TYPES using NLI templates; default to co_occurs."""
    _init_nli()
    # Score forward templates
    scores = []
    for rel, tmpl in _REL_TEMPLATES.items():
        hyp = tmpl.format(SUBJ=subj, OBJ=obj)
        scores.append((rel, _nli_score(context, hyp)))
    # Penalize if reversed template is *more* entailed for directional ones
    for rel, tmpl in _REL_TEMPLATES_REV.items():
        hyp_rev = tmpl.format(SUBJ=subj, OBJ=obj)
        rev = _nli_score(context, hyp_rev)
        # subtract small penalty if reverse looks more likely
        for i, (r, sc) in enumerate(scores):
            if r == rel:
                scores[i] = (r, sc - 0.1 * rev)
                break
    scores.sort(key=lambda x: x[1], reverse=True)
    best_rel, best_p = scores[0]
    # small acceptance threshold; otherwise fall back
    return best_rel if best_p >= 0.55 else "co_occurs"


def extract_entities_relations(
    meta: Dict[str, Any], text: str
) -> Optional[PaperExtraction]:
    """Extract nodes and relations from a block of text using NER.

    This function uses a HuggingFace NER pipeline to identify entity spans. It
    collects unique entities, assigns a node type via `categorize_entity`, and
    builds a list of NodeRecord objects. Relations are generated in a naive
    fashion by connecting every distinct pair of entities found in the text with
    a 'co_occurs' edge. Evidence spans are omitted because this pipeline does
    not perform relation classification. Returns None if no entities are
    detected.
    """
    if not text.strip():
        return None
    try:
        ner_results = _ner_pipeline(text)
    except Exception:
        # If model inference fails, skip this document.
        return None
    entities: Dict[str, NodeRecord] = {}
    for ent in ner_results:
        name = ent.get("word", "").strip()
        if not name:
            continue
        label = ent.get("entity_group", "")
        if name not in entities:
            entities[name] = NodeRecord(
                canonical_name=name,
                node_type=categorize_entity(label),
                synonyms=[],
                normalizations={},
            )
    if not entities:
        return None
    entities = {k: v for k, v in entities.items() if v.node_type != "Diagnosis"}
    if not entities:
        return None

    relations: List[RelationRecord] = []
    ent_names = list(entities.keys())
    for i in range(len(ent_names)):
        for j in range(i + 1, len(ent_names)):
            subj, obj = ent_names[i], ent_names[j]
            pred = "co_occurs"
            conf = 0.5
            sent_ctx = _pick_sentence_context(text, subj, obj)
            pred = classify_relation_via_nli(sent_ctx, subj, obj)
            # crude confidence from entailment score of chosen hypothesis
            # (reuse internal scorer to fetch the final probability)
            conf = 0.7 if pred != "co_occurs" else 0.5
            dirn = (
                "directed"
                if pred in {"treats", "predicts", "biomarker_for", "measure_of"}
                else "undirected"
            )
            relations.append(
                RelationRecord(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    directionality=dirn,
                    evidence_span=sent_ctx[:300],
                    confidence=conf,
                    qualifiers={},
                )
            )

    return PaperExtraction(
        paper_id=meta.get("id", ""),
        doi=meta.get("doi"),
        title=meta.get("title", ""),
        year=meta.get("year"),
        venue=meta.get("venue"),
        nodes=list(entities.values()),
        relations=relations,
    )


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def accum_extractions(
    extractions: List[PaperExtraction],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []
    for extraction in extractions:
        paper_rows.append(
            {
                "paper_id": extraction.paper_id,
                "doi": extraction.doi,
                "title": extraction.title,
                "year": extraction.year,
                "venue": extraction.venue,
            }
        )
        for n in extraction.nodes:
            node_rows.append(
                {
                    "paper_id": extraction.paper_id,
                    "canonical_name": normalize_name(n.canonical_name),
                    "node_type": n.node_type,
                    "synonyms": json.dumps(n.synonyms, ensure_ascii=False),
                    "normalizations": json.dumps(n.normalizations, ensure_ascii=False),
                }
            )
        for r in extraction.relations:
            rel_rows.append(
                {
                    "paper_id": extraction.paper_id,
                    "subject": normalize_name(r.subject),
                    "predicate": r.predicate,
                    "object": normalize_name(r.obj),
                    "directionality": r.directionality,
                    "evidence_span": r.evidence_span,
                    "confidence": r.confidence,
                    "qualifiers": json.dumps(r.qualifiers, ensure_ascii=False),
                }
            )
    nodes_df = pd.DataFrame(node_rows).drop_duplicates()
    rels_df = pd.DataFrame(rel_rows).drop_duplicates()
    papers_df = pd.DataFrame(paper_rows).drop_duplicates()
    return nodes_df, rels_df, papers_df


def build_multilayer_graph(
    nodes_df: pd.DataFrame, rels_df: pd.DataFrame
) -> nx.MultiDiGraph:
    """
    Construct a NetworkX MultiDiGraph from dataframes of nodes and relations.

    Nodes are annotated with their most common node_type (by frequency of mentions), as well as merged synonyms and normalizations. Edges are
    directed and store predicate, paper_id, directionality, evidence_span, confidence and qualifiers. Multiple edges between the same pair of
    nodes are preserved by omitting an explicit key when calling ``add_edge``. NetworkX will assign unique keys automatically
    """
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    # determine most frequent type per canonical name
    node_types = (
        nodes_df.groupby(["canonical_name", "node_type"])
        .size()
        .reset_index(name="n")
        .sort_values(["canonical_name", "n"], ascending=[True, False])
    )
    canonical_to_type = (
        node_types.drop_duplicates("canonical_name")
        .set_index("canonical_name")["node_type"]
        .to_dict()
    )
    for name, ntype in canonical_to_type.items():
        graph.add_node(name, node_type=ntype, synonyms=[], normalizations={})
    for _, row in nodes_df.iterrows():
        name = row["canonical_name"]
        graph.nodes[name]["synonyms"] = sorted(
            set(graph.nodes[name]["synonyms"]) | set(json.loads(row["synonyms"]))
        )
        graph.nodes[name]["normalizations"] = {
            **graph.nodes[name]["normalizations"],
            **json.loads(row["normalizations"]),
        }
    for _, row in rels_df.iterrows():
        graph.add_edge(
            row["subject"],
            row["object"],
            predicate=row["predicate"],
            paper_id=row["paper_id"],
            directionality=row["directionality"],
            evidence_span=row["evidence_span"],
            confidence=float(row["confidence"]),
            qualifiers=json.loads(row["qualifiers"]),
        )
    return graph


PRIMITIVES = (str, int, float, bool)


def coerce_for_graphml(v):
    if v is None:
        return ""
    if isinstance(v, PRIMITIVES):
        return v
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple, set)):
        try:
            return json.dumps(list(v), ensure_ascii=False)
        except Exception:
            return ", ".join(map(str, v))
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(v)
    if isinstance(v, type):  # classes / callables
        return v.__name__
    return str(v)


def sanitize_graph_for_graphml(graph: nx.Graph) -> None:
    # nodes
    for _, data in graph.nodes(data=True):
        for k in list(data.keys()):
            data[k] = coerce_for_graphml(data[k])
    # edges (multi or not)
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            for k in list(data.keys()):
                data[k] = coerce_for_graphml(data[k])
    else:
        for _, _, data in graph.edges(data=True):
            for k in list(data.keys()):
                data[k] = coerce_for_graphml(data[k])


def find_non_primitive_attrs(graph: nx.Graph):
    bad = []
    for n, d in graph.nodes(data=True):
        for k, v in d.items():
            if not isinstance(v, PRIMITIVES) and v is not None:
                bad.append(("node", n, k, type(v).__name__))
    if graph.is_multigraph():
        for u, v, key, d in graph.edges(keys=True, data=True):
            for k, val in d.items():
                if not isinstance(val, PRIMITIVES) and val is not None:
                    bad.append(("edge", (u, v, key), k, type(val).__name__))
    else:
        for u, v, d in graph.edges(data=True):
            for k, val in d.items():
                if not isinstance(val, PRIMITIVES) and val is not None:
                    bad.append(("edge", (u, v), k, type(val).__name__))
    return bad


def project_to_weighted_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    """Project a directed multigraph onto an undirected weighted graph."""
    weighted = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        w = float(data.get("confidence", 1.0))
        if weighted.has_edge(u, v):
            weighted[u][v]["weight"] += w
        else:
            weighted.add_edge(u, v, weight=w)
    return weighted


def save_tables(
    nodes_df: pd.DataFrame,
    rels_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    graph: nx.MultiDiGraph,
    out_prefix: str,
    projected: Optional[nx.Graph] = None,
) -> None:
    base = DATA_DIR / out_prefix
    nodes_df.to_parquet(base.with_suffix(".nodes.parquet"))
    rels_df.to_parquet(base.with_suffix(".rels.parquet"))
    papers_df.to_parquet(base.with_suffix(".papers.parquet"))
    sanitize_graph_for_graphml(graph)
    nx.write_graphml(graph, base.with_suffix(".graphml"))
    if projected is not None:
        sanitize_graph_for_graphml(projected)
        nx.write_graphml(projected, base.with_suffix(".weighted.graphml"))


def load_lexicon(path: str) -> Set[str]:
    """Load a lexicon of terms (one per line) into a lowercase set."""
    terms: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                terms.add(t.lower())
    return terms


def evaluate_nodes(
    nodes_df: pd.DataFrame, gold_nodes: Iterable[str]
) -> Dict[str, float]:
    """Compute precision/recall/F1 for node extraction given a gold set."""
    pred: Set[str] = set(nodes_df["canonical_name"].tolist())
    gold: Set[str] = {normalize_name(s) for s in gold_nodes}
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_relations(
    graph: nx.MultiDiGraph, gold_relations: Iterable[Tuple[str, str, str]]
) -> Dict[str, float]:
    """Compute precision/recall/F1 for relation extraction given a gold set."""
    gold_set: Set[Tuple[str, str, str]] = set(
        (
            normalize_name(s),
            p,
            normalize_name(o),
        )
        for (s, p, o) in gold_relations
    )
    pred_set: Set[Tuple[str, str, str]] = set()
    for u, v, data in graph.edges(data=True):
        subj = normalize_name(u)
        obj = normalize_name(v)
        pred = data.get("predicate")
        pred_set.add((subj, pred, obj))
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def load_gold_relations(path: str) -> List[Tuple[str, str, str]]:
    """Load a tab‑separated list of gold relations."""
    triples: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_gold_nodes(path: str) -> List[str]:
    nodes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if ln:
                nodes.append(ln)
    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="OpenAlex search query (title/abstract)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=DEFAULT_FILTER,
        help="Additional OpenAlex filters",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="psych_kg_results_only",
        help="Prefix for output files saved under the data/ directory",
    )
    parser.add_argument(
        "--n-top-cited",
        type=int,
        default=250,
        help="Number of top‑cited OA papers to fetch",
    )
    parser.add_argument(
        "--n-most-recent",
        type=int,
        default=250,
        help="Number of most‑recent OA papers to fetch",
    )
    parser.add_argument(
        "--eval-nodes",
        type=str,
        default=None,
        help="Path to a gold standard node list (one canonical name per line)",
    )
    parser.add_argument(
        "--eval-relations",
        type=str,
        default=None,
        help="Path to a gold standard relation file (tab separated subject, predicate, object)",
    )
    parser.add_argument(
        "--project-to-weighted",
        action="store_true",
        help="If set, project the multigraph to a simple weighted graph and save it",
    )
    args = parser.parse_args()

    # Fetch the two buckets of papers from OpenAlex and deduplicate them
    print("[info] Fetching OpenAlex (top-cited and most-recent)")
    top_cited = fetch_top_n(
        args.query, args.filters, sort="cited_by_count:desc", n=args.n_top_cited
    )
    most_recent = fetch_top_n(
        args.query, args.filters, sort="publication_date:desc", n=args.n_most_recent
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in itertools.chain(top_cited, most_recent):
        if rec.get("id") and rec["id"] not in by_id:
            by_id[rec["id"]] = rec
    records = list(by_id.values())
    print(f"[info] Candidate papers: {len(records)} (unique across both buckets)")

    print("[info] Downloading full texts and extracting (RESULTS-only)")
    paper_level: List[PaperExtraction] = []
    for rec in tqdm(records):
        meta = {
            "id": rec.get("id"),
            "doi": rec.get("doi"),
            "title": rec.get("title"),
            "year": rec.get("year"),
            "venue": rec.get("venue"),
        }

        fulltext = resolve_text_and_download(rec)
        results_only = extract_results_only(fulltext or "") if fulltext else None
        # prefer RESULTS section; fallback to abstract
        text_for_ie = results_only if results_only else (rec.get("abstract", "") or "")
        extraction = extract_entities_relations(meta, text_for_ie)
        if extraction:
            paper_level.append(extraction)

    if not paper_level:
        print("[warn] No extractions; exiting.")
        raise SystemExit(0)

    nodes_df, rels_df, papers_df = accum_extractions(paper_level)
    print(
        f"[info] {len(papers_df)} papers, {len(nodes_df)} node mentions, {len(rels_df)} relations"
    )
    graph = build_multilayer_graph(nodes_df, rels_df)
    weighted_graph: Optional[nx.Graph] = None
    if args.project_to_weighted:
        weighted_graph = project_to_weighted_graph(graph)
    save_tables(
        nodes_df, rels_df, papers_df, graph, args.out_prefix, projected=weighted_graph
    )
    if args.eval_nodes:
        gold_nodes = load_gold_nodes(args.eval_nodes)
        metrics = evaluate_nodes(nodes_df, gold_nodes)
        print(
            f"[eval-nodes] precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )
    if args.eval_relations:
        gold_relations = load_gold_relations(args.eval_relations)
        metrics = evaluate_relations(graph, gold_relations)
        print(
            f"[eval-relations] precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )
    print(f"[done] Saved under data/{args.out_prefix}.*")
