"""
pip install openai httpx pydantic pandas orjson networkx tqdm backoff python-dotenv
export OPENAI_API_KEY=sk-...
python kg_pipeline.py --query '('RDoC' OR 'HiTOP' OR DSM OR psychopathology)' --model gpt-4.1-mini

Notes:
- Now pulls exactly N=first 250 top-cited + first 250 most-recent (configurable) from OpenAlex.
- Extracts ONLY the Results section (falls back to abstract if Results not found).
- “Relation types” are flexible; start with: supports, contradicts, predicts, co_occurs, treats, biomarker_for, measure_of.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import pathlib
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import backoff
import httpx
import networkx as nx
import orjson
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from readability import Document
from tqdm import tqdm

from chunking import chunk

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_FILTER = (
    "from_publication_date:2010-01-01,primary_location.is_oa:true,language:en"
)
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

NODE_TYPES = {
    "Symptom",
    "Diagnosis",
    "RDoC_Construct",
    "HiTOP_Component",
    "Biomarker",
    "Treatment",
    "Task",
    "Measure",
}

REL_TYPES = {
    "supports",
    "contradicts",
    "predicts",
    "co_occurs",
    "treats",
    "biomarker_for",
    "measure_of",
}

try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("openai>=1.0 not installed")


def get_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY")
    return OpenAI()


class NodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(
        ..., description="Primary name for the entity as used in the paper."
    )
    node_type: str = Field(..., description=f"One of {sorted(list(NODE_TYPES))}")
    synonyms: List[str] = Field(default_factory=list)
    # TODO: automate concept normalization
    normalizations: Dict[str, str] = Field(
        default_factory=dict, description='i.e. {"UMLS":"C0005586"}'
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
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
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


EXTRACTION_SYSTEM_PROMPT = f"""You are an expert biomedical information extraction agent.
Extract a compact, non-redundant set of domain entities (nodes) and typed relations (edges) from the provided FULL TEXT CHUNK, which is preferentially from the RESULTS section.
- Use canonical, human-readable names for nodes.
- Node types: {sorted(list(NODE_TYPES))}
- Predicates: {sorted(list(REL_TYPES))}
- Provide brief evidence_span (short quote) for each relation if possible (favor quotes from RESULTS language).
- Do NOT invent facts; prefer 'supports' unless clear causal language exists.
- If RDoC/HiTOP mapping is explicit, include it in 'normalizations'. Return ONLY JSON per schema.
"""

MERGE_SYSTEM_PROMPT = """You are consolidating chunk-level extractions for a single paper.
Merge nodes and relations:
- Deduplicate nodes by canonical_name (merge synonyms/normalizations).
- Deduplicate relations by (subject, predicate, object); keep the highest confidence and concatenate distinct evidence_spans (up to 2 short snippets).
Return JSON ONLY per schema.
"""


def uninvert_openalex_abstract(inv: Dict[str, List[int]]) -> str:
    if not inv:
        return ""
    pos2tok = {}
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
    out = []
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


# for PMCID full‑text retrieval
PMC_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def try_pmc_fulltext(pmcid: str) -> Optional[str]:
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    r = httpx.get(PMC_EUTILS, params=params, timeout=60)
    if r.status_code != 200 or not r.text:
        return None
    soup = BeautifulSoup(r.text, "lxml-xml")
    parts = []
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
    r"^\s*(discussion|conclusions?|limitations?|general discussion|overview|references?|acknowledg(e)?ments?|supplementary)\s*[:\-]?\s*$",
    flags=re.I | re.M,
)


def extract_results_only(fulltext: str) -> Optional[str]:
    """Heuristic: take the text from 'Results/Findings/Outcomes' to the next major section."""
    if not fulltext:
        return None
    # normalize newlines and collapse spaces a bit
    doc = re.sub(r"\r", "\n", fulltext)
    doc = re.sub(r"[ \t]+", " ", doc)
    # try to find RESULTS header
    m = RESULTS_HDR_RE.search(doc)
    if not m:
        return None
    start = m.start()
    # find next section header after start
    m2 = NEXT_SECTION_RE.search(doc, pos=start + 1)
    end = m2.start() if m2 else len(doc)
    chunk = doc[start:end].strip()
    # avoid returning tiny noise
    return chunk if len(chunk) > 400 else None


def build_user_prompt_chunk(meta: Dict[str, Any], chunk_text: str) -> str:
    return f"""Paper metadata:
- openalex_id: {meta.get('id')}
- title: {meta.get('title')}
- doi: {meta.get('doi')}
- year: {meta.get('year')}
- venue: {meta.get('venue')}

Full-text chunk (RESULTS-preferred):
{chunk_text}
"""


@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=None)
def llm_extract_chunk(
    client: OpenAI, model: str, meta: Dict[str, Any], chunk: str
) -> Optional[PaperExtraction]:
    schema = extraction_json_schema()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt_chunk(meta, chunk)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "paper_extraction",
                "schema": schema,
                "strict": True,
            },
        },
        max_output_tokens=2000,
    )
    try:
        text = resp.output_text
    except Exception:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if item.get("type") == "output_text":
                parts.append(item.get("content", ""))
        text = "".join(parts)
    if not text:
        return None
    try:
        payload = json.loads(text)
        return PaperExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def merge_extractions(
    client: OpenAI, model: str, meta: Dict[str, Any], chunks: List[PaperExtraction]
) -> Optional[PaperExtraction]:
    schema = extraction_json_schema()
    chunk_payloads = [json.loads(c.model_dump_json()) for c in chunks]
    merge_user = f"""Combine these chunk-level extractions for ONE paper into a single, deduplicated result (JSON list below).
Paper metadata:
- openalex_id: {meta.get('id')}
- title: {meta.get('title')}
- doi: {meta.get('doi')}
- year: {meta.get('year')}
- venue: {meta.get('venue')}

Chunk-extractions (JSON list):
{json.dumps(chunk_payloads, ensure_ascii=False)[:180000]}
"""
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": merge_user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "paper_extraction",
                "schema": schema,
                "strict": True,
            },
        },
        max_output_tokens=4000,
    )
    try:
        text = resp.output_text  # type: ignore[attr-defined]
    except Exception:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if item.get("type") == "output_text":
                parts.append(item.get("content", ""))
        text = "".join(parts)
    if not text:
        return None
    try:
        payload = json.loads(text)
        return PaperExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


@dataclass
class PaperLite:
    openalex_id: str
    doi: Optional[str]
    title: str
    year: Optional[int]
    venue: Optional[str]


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def accum_extractions(
    extractions: List[PaperExtraction],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows, rel_rows, paper_rows = [], [], []
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
        s, p, o = row["subject"], row["predicate"], row["object"]
        if s not in graph or o not in graph:
            continue
        graph.add_edge(
            s,
            o,
            predicate=p,
            paper_id=row["paper_id"],
            directionality=row["directionality"],
            evidence_span=row["evidence_span"],
            confidence=float(row["confidence"]),
            qualifiers=json.loads(row["qualifiers"]),
        )
    return graph


def project_to_weighted_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    """
    Project a directed MultiDiGraph onto an undirected weighted Graph.

    Each unique pair of nodes (u, v) will result in a single undirected edge
    whose weight is the sum of the confidence scores of all edges (both
    directions) between u and v.  Self‑loops are ignored.  Additional
    attributes on the multigraph edges are not carried over.
    """
    graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        w = float(data.get("confidence", 1.0))
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += w
        else:
            graph.add_edge(u, v, weight=w)
    return graph


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
    nx.write_graphml(graph, base.with_suffix(".graphml"))
    if projected is not None:
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


def mask_nodes(
    nodes_df: pd.DataFrame, rels_df: pd.DataFrame, lexicon: Set[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = ~nodes_df["canonical_name"].str.lower().isin(lexicon)
    nodes_filtered = nodes_df[mask].copy()
    keep = set(nodes_filtered["canonical_name"].unique())
    rels_filtered = rels_df[
        rels_df["subject"].isin(keep) & rels_df["object"].isin(keep)
    ].copy()
    return nodes_filtered, rels_filtered


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
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_relations(
    graph: nx.MultiDiGraph, gold_relations: Iterable[Tuple[str, str, str]]
) -> Dict[str, float]:
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
        # Note: directed/undirected handling – treat as directed here
        pred_set.add((subj, pred, obj))
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def load_gold_relations(path: str) -> List[Tuple[str, str, str]]:
    """tab‑separated"""
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
    """Load a list of node canonical names (one per line)"""
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
    parser.add_argument("--filters", type=str, default=DEFAULT_FILTER)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--out-prefix", type=str, default="psych_kg_results_only")
    parser.add_argument(
        "--chunk-chars", type=int, default=9000, help="approx chars per chunk"
    )
    parser.add_argument("--chunk-overlap", type=int, default=600)
    parser.add_argument(
        "--n-top-cited",
        type=int,
        default=250,
        help="Number of top-cited OA papers to fetch",
    )
    parser.add_argument(
        "--n-most-recent",
        type=int,
        default=250,
        help="Number of most-recent OA papers to fetch",
    )
    parser.add_argument(
        "--mask-lexicon",
        type=str,
        default=None,
        required=True,
        help="Path to a newline‑separated list of nosology terms to mask out",
    )
    parser.add_argument(
        "--project-to-weighted",
        action="store_true",
        help="If set, project the multigraph to a simple weighted graph and save it",
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
    args = parser.parse_args()

    client = get_openai_client()

    # fetch exactly N per sort, then union by OpenAlex id
    print("[info] Fetching OpenAlex (top-cited and most-recent)")
    top_cited = fetch_top_n(
        args.query, args.filters, sort="cited_by_count:desc", n=args.n_top_cited
    )
    most_recent = fetch_top_n(
        args.query, args.filters, sort="publication_date:desc", n=args.n_most_recent
    )

    # dedupe by OpenAlex id
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in itertools.chain(top_cited, most_recent):
        if rec.get("id") and rec["id"] not in by_id:
            by_id[rec["id"]] = rec
    records = list(by_id.values())
    print(f"[info] Candidate papers: {len(records)} (unique across both buckets)")

    # for each paper: resolve full text, slice to RESULTS, chunk, map, reduce
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

        if not text_for_ie.strip():
            continue

        chunks = list(
            chunk(
                text_for_ie,
                max_tokens_chars=args.chunk_chars,
                overlap=args.chunk_overlap,
            )
        )
        chunk_extractions: List[PaperExtraction] = []
        for ch in chunks:
            ex = llm_extract_chunk(client, args.model, meta, ch)
            if ex:
                chunk_extractions.append(ex)

        if not chunk_extractions:
            continue

        merged = merge_extractions(client, args.model, meta, chunk_extractions)
        if merged:
            paper_level.append(merged)

    if not paper_level:
        print("[warn] No extractions; exiting.")
        raise SystemExit(0)

    nodes_df, rels_df, papers_df = accum_extractions(paper_level)
    print(
        f"[info] {len(papers_df)} papers, {len(nodes_df)} node mentions, {len(rels_df)} relations"
    )
    lexicon = load_lexicon(args.mask_lexicon)
    nodes_df, rels_df = mask_nodes(nodes_df, rels_df, lexicon)
    print(
        f"[info] After masking lexicon, {len(nodes_df)} node mentions, {len(rels_df)} relations"
    )
    graph = build_multilayer_graph(nodes_df, rels_df)
    weighted_graph: Optional[nx.Graph] = None
    if args.project_to_weighted:
        weighted_G = project_to_weighted_graph(graph)
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
