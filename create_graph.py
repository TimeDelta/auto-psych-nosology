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
import logging
import math
import os
import pathlib
import re
import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import httpx
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import stanza
import torch
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from readability import Document
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

_LEMMATIZER = WordNetLemmatizer()


class NodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str = Field(
        ..., description="Primary name for the entity as used in the paper."
    )
    lemma: str = Field(
        ..., description="Singular, lowercase canonical key used for deduplication."
    )
    node_type: str = Field(..., description=f"One of {sorted(list(NODE_TYPES))}")
    synonyms: List[str] = Field(default_factory=list)
    normalizations: Dict[str, Any] = Field(
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


@dataclass
class NliScores:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def signal(self) -> float:
        """Heuristic score favouring entailment while penalizing contradiction."""
        return self.entailment - self.contradiction


@dataclass
class RelationInference:
    label: str
    entailment: float
    score: float
    margin: float
    reverse_entailment: float = 0.0


def extraction_json_schema() -> Dict[str, Any]:
    # doesn't need input because Pydantic BaseModel class has a classmethod .model_json_schema()
    # that generates a full JSON Schema describing its fields and nested models
    return PaperExtraction.model_json_schema()


OPENALEX = "https://api.openalex.org"
DEFAULT_FILTER = (
    "from_publication_date:2015-10-03,"
    "open_access.is_oa:true,"
    "language:en,"
    "concepts.id:C61535369"  # biological psychiatry
)
_CLIENT = httpx.Client(
    base_url=OPENALEX,
    headers={
        "User-Agent": f"auto-psych-nosology/0.1",
        "Accept": "application/json",
    },
    timeout=60.0,
)
_OPENALEX_LOCK = threading.Lock()
_LAST_OPENALEX_REQUEST = 0.0
_OPENALEX_MIN_INTERVAL = 0.55  # 2 requests/second per rate limit guidance
_OPENALEX_MAX_RETRIES = 6


def _reserve_openalex_slot() -> None:
    """Block until we are allowed to send the next OpenAlex request."""
    global _LAST_OPENALEX_REQUEST
    while True:
        with _OPENALEX_LOCK:
            now = time.monotonic()
            elapsed = now - _LAST_OPENALEX_REQUEST
            if elapsed >= _OPENALEX_MIN_INTERVAL:
                _LAST_OPENALEX_REQUEST = now
                return
            wait_time = _OPENALEX_MIN_INTERVAL - elapsed
        time.sleep(max(wait_time, 0.0))


def _openalex_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Send an OpenAlex request with basic rate limiting and retry backoff."""
    attempt = 0
    while True:
        attempt += 1
        _reserve_openalex_slot()
        try:
            response = _CLIENT.request(method, path, **kwargs)
        except httpx.RequestError:
            if attempt >= _OPENALEX_MAX_RETRIES:
                raise
            # Exponential backoff capped at ~8 seconds
            backoff = min(2 ** (attempt - 1) * _OPENALEX_MIN_INTERVAL, 8.0)
            time.sleep(backoff)
            continue

        if response.status_code != 429:
            return response

        if attempt >= _OPENALEX_MAX_RETRIES:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        try:
            retry_delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            retry_delay = 0.0
        retry_delay = max(retry_delay, _OPENALEX_MIN_INTERVAL)
        time.sleep(retry_delay)


def _openalex_get(path: str, **kwargs: Any) -> httpx.Response:
    return _openalex_request("GET", path, **kwargs)


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
    r = _openalex_get("/works", params=params)
    if r.status_code != 200:
        print("Status:", r.status_code)
        print("Reason:", r.reason_phrase)
        print("Body:", r.text)
    r.raise_for_status()
    return r.json()


def _normalize_queries(query: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(query, str):
        items = [q.strip() for q in query.split(";")]
    else:
        items = [str(q).strip() for q in query]
    return [q for q in items if q]


def _fetch_top_n_single(
    query: str, filters: str, sort: str, n: int
) -> List[Dict[str, Any]]:
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
                    "search_query": query,
                }
            )
            if len(out) >= n:
                break
        cursor = page.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return out


def fetch_top_n(
    query: Union[str, Sequence[str]], filters: str, sort: str, n: int
) -> List[Dict[str, Any]]:
    """Fetch up to ``n`` works for each search term and concatenate results."""
    queries = _normalize_queries(query)
    if not queries:
        return []
    results: List[Dict[str, Any]] = []
    for term in queries:
        results.extend(_fetch_top_n_single(term, filters, sort, n))
    return results


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
    _quiet_pdfminer_logs()
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
DISCUSSION_HDR_RE = re.compile(
    r"^\s*(discussion|general discussion|discussion and conclusions?)\s*[:\-]?\s*$",
    flags=re.I | re.M,
)
RESULTS_STOP_RE = re.compile(
    r"^\s*(discussion|general discussion|conclusions?|limitations?|implications?|future directions?|overview|references?|acknowledg(e)?ments?|supplementary|appendix|materials?)\s*[:\-]?\s*$",
    flags=re.I | re.M,
)
DISCUSSION_STOP_RE = re.compile(
    r"^\s*(conclusions?|limitations?|implications?|future directions?|overview|references?|acknowledg(e)?ments?|supplementary|appendix|materials?)\s*[:\-]?\s*$",
    flags=re.I | re.M,
)


def _normalise_fulltext(text: str) -> str:
    doc = re.sub(r"\r", "\n", text)
    doc = re.sub(r"[ \t]+", " ", doc)
    return doc


def _extract_section(
    doc: str, header_re: re.Pattern, stop_re: re.Pattern
) -> Optional[str]:
    match = header_re.search(doc)
    if not match:
        return None
    start = match.start()
    stop = stop_re.search(doc, pos=match.end())
    while stop and stop.start() <= start:
        stop = stop_re.search(doc, pos=stop.end())
    end = stop.start() if stop else len(doc)
    chunk = doc[start:end].strip()
    return chunk if len(chunk) > 400 else None


def extract_results_and_discussion(fulltext: str) -> Optional[str]:
    """Heuristic: gather Results and Discussion sections when available."""
    if not fulltext:
        return None
    doc = _normalise_fulltext(fulltext)
    sections: List[str] = []
    results = _extract_section(doc, RESULTS_HDR_RE, RESULTS_STOP_RE)
    if results:
        sections.append(results)
    discussion = _extract_section(doc, DISCUSSION_HDR_RE, DISCUSSION_STOP_RE)
    if discussion:
        sections.append(discussion)
    if sections:
        return "\n\n".join(sections)
    return None


def _parse_stanza_package_spec(spec: str) -> Optional[Any]:
    spec = (spec or "").strip()
    if not spec:
        return None
    if any(ch in spec for ch in "=;,"):
        cfg: Dict[str, str] = {}
        for part in re.split(r"[;,]", spec):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, val = part.split("=", 1)
            cfg[key.strip()] = val.strip()
        return cfg or None
    return spec


def _build_stanza_package_candidates() -> List[Any]:
    forced = os.getenv("STANZA_FORCE_PACKAGE", "").strip()
    if forced:
        parsed = _parse_stanza_package_spec(forced)
        return [parsed] if parsed else []

    tok_pkg = os.getenv("STANZA_TOKENIZE_PACKAGE", "").strip() or "default"
    primary_ner = os.getenv("STANZA_NER_PACKAGE", "").strip() or "bc5cdr"
    candidates: List[Any] = [{"tokenize": tok_pkg, "ner": primary_ner}]

    extra_biomedical = [
        pkg.strip()
        for pkg in os.getenv(
            "STANZA_BIOMED_PACKAGES",
            "anatem|bc4chemd|bionlp13cg|jnlpba|linnaeus|ncbi_disease",
        ).split("|")
        if pkg.strip()
    ]
    for extra_pkg in extra_biomedical:
        candidates.append({"tokenize": tok_pkg, "ner": extra_pkg})

    extras = [
        p.strip()
        for p in os.getenv("STANZA_EXTRA_PACKAGES", "").split("|")
        if p.strip()
    ]
    for extra in extras:
        parsed = _parse_stanza_package_spec(extra)
        if parsed and parsed not in candidates:
            candidates.append(parsed)

    # fallbacks to known biomedical-friendly packages
    for fallback in ("craft", "mimic"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


_STANZA_PIPELINES: List[Tuple[str, "stanza.Pipeline"]] = []
_STANZA_PACKAGE_CANDIDATES = _build_stanza_package_candidates()
_NER_EXCLUDE_TERMS = {
    term.strip().lower()
    for term in os.getenv(
        "NER_EXCLUDE_TERMS",
        "patient,patients,control,controls,participant,participants,subject,subjects,human,humans,donor,donors",
    ).split(",")
    if term.strip()
}


def _quiet_pdfminer_logs() -> None:
    """Mute noisy pdfminer warnings that clutter stderr."""
    for name in ["pdfminer", "pdfminer.pdfcolor", "pdfminer.pdfinterp"]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def _stanza_use_gpu() -> bool:
    """Determine whether to enable GPU for Stanza based on availability and env."""
    if os.getenv("STANZA_USE_GPU", "").strip() == "0":
        return False
    return torch.cuda.is_available()


def _ensure_stanza_pipelines() -> List[Tuple[str, "stanza.Pipeline"]]:
    """Build and cache Stanza pipelines for each biomedical NER package."""
    global _STANZA_PIPELINES
    if _STANZA_PIPELINES:
        return _STANZA_PIPELINES
    lang = os.getenv("STANZA_LANG", "en")
    processors = os.getenv("STANZA_PROCESSORS", "tokenize,mwt,pos,lemma,ner")
    for package in _STANZA_PACKAGE_CANDIDATES:
        pkg_display = (
            package
            if isinstance(package, str)
            else ", ".join(f"{k}={v}" for k, v in package.items())
        )
        download_kwargs = {"processors": processors, "verbose": False}
        try:
            if package:
                download_kwargs["package"] = package
            stanza.download(lang, **download_kwargs)
        except Exception:
            # some of the biomedical packages are bundled with the default
            # Stanza distribution, the download step often raises even when
            # the resources are already present locally
            pass

        pipeline_kwargs = {
            "processors": processors,
            "use_gpu": _stanza_use_gpu(),
            "verbose": False,
        }
        if package:
            pipeline_kwargs["package"] = package
        try:
            pipeline = stanza.Pipeline(lang, **pipeline_kwargs)
            _STANZA_PIPELINES.append((pkg_display, pipeline))
            warnings.warn(f"Using Stanza biomedical NER package {pkg_display}.")
        except Exception as exc:
            warnings.warn(
                f"Stanza package {pkg_display} unavailable ({exc!r}); trying fallback."
            )
            continue

    if not _STANZA_PIPELINES:
        warnings.warn(
            "Unable to initialise any Stanza biomedical NER package; extractions will be empty."
        )
    return _STANZA_PIPELINES


def _stanza_entities(text: str) -> List[Dict[str, Any]]:
    pipelines = _ensure_stanza_pipelines()
    if not pipelines:
        return []
    results: List[Dict[str, Any]] = []
    seen_spans: Set[Tuple[int, int, str]] = set()
    for pkg_display, pipeline in pipelines:
        doc = pipeline(text)
        ents = getattr(doc, "ents", None)
        if ents is None:
            ents = getattr(doc, "entities", [])
        for ent in ents:
            start = getattr(ent, "start_char", None)
            end = getattr(ent, "end_char", None)
            label = getattr(ent, "type", "")
            words = getattr(ent, "words", [])
            lemmas: List[str] = []
            upos: List[str] = []
            xpos: List[str] = []
            tokens: List[str] = []
            for word in words:
                tok = getattr(word, "text", "")
                if tok:
                    tokens.append(tok)
                lemma = getattr(word, "lemma", "")
                if lemma:
                    lemmas.append(lemma)
                elif tok:
                    lemmas.append(tok)
                pos = getattr(word, "upos", "")
                if pos:
                    upos.append(pos)
                xpos_tag = getattr(word, "xpos", "")
                if xpos_tag:
                    xpos.append(xpos_tag)
            if start is not None and end is not None:
                key = (start, end, label)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
            results.append(
                {
                    "start": start,
                    "end": end,
                    "entity_group": label,
                    "word": getattr(ent, "text", ""),
                    "source_package": pkg_display,
                    "lemma": " ".join(lemmas).strip(),
                    "upos": upos,
                    "xpos": xpos,
                    "tokens": tokens,
                }
            )
    return results


def _ner_pipeline(text: str) -> List[Dict[str, Any]]:
    if not text.strip():
        return []
    return _stanza_entities(text)


def categorize_entity(label: str) -> str:
    """Map a coarse entity label into one of our predefined node types.

    The biomedical NER model outputs labels such as 'CHEMICAL', 'DISEASE',
    'PROTEIN', etc.  This function maps those labels onto the simplified set of
    categories used by our knowledge graph. The mapping is heuristic and may
    misclassify edge cases, but it provides a starting point. Unknown labels
    default to 'Symptom'.
    """
    lbl = label.upper()
    direct_map = {
        "PROBLEM": "Symptom",
        "FINDING": "Symptom",
        "SIGN": "Symptom",
        "DIAGNOSIS": "Diagnosis",
        "TEST": "Measure",
        "MEASUREMENT": "Measure",
        "MEASURE": "Measure",
        "ASSESSMENT": "Measure",
        "LAB": "Measure",
        "TREATMENT": "Treatment",
        "PROCEDURE": "Treatment",
        "THERAPY": "Treatment",
        "MEDICATION": "Treatment",
        "DEVICE": "Measure",
        "ANATOMY": "Biomarker",
        "ANATOMICAL": "Biomarker",
    }
    if lbl in direct_map:
        return direct_map[lbl]
    if "DISEASE" in lbl or "DISORDER" in lbl or "SYNDROME" in lbl:
        return "Diagnosis"
    if "CHEMICAL" in lbl or "DRUG" in lbl or "MED" in lbl:
        return "Treatment"
    if "GENE" in lbl or "PROTEIN" in lbl or "CELL" in lbl or "BIOMARKER" in lbl:
        return "Biomarker"
    if "BEHAVIOR" in lbl or "SYMPTOM" in lbl:
        return "Symptom"
    if lbl in {"PERSON", "ORG", "ORGANIZATION", "LOCATION", "EVENT"}:
        return None
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


_REL_DIRECTION_PENALTY = float(os.getenv("RELATION_NLI_REVERSE_PENALTY", "0.5"))
_REL_SCORE_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_SCORE", "0.05"))
_REL_MARGIN_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_MARGIN", "0.04"))
_REL_ENTAILMENT_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_ENT", "0.6"))
_REL_REVERSE_GAP = float(os.getenv("RELATION_NLI_REVERSE_GAP", "0.05"))


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


def _nli_probs(premise: str, hypothesis: str) -> NliScores:
    """Return softmax probabilities over contradiction/neutral/entailment."""
    with torch.no_grad():
        enc = _nli_tok(
            premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
        )
        probs = _nli_model(**enc).logits.softmax(-1)[0]
    return NliScores(
        entailment=float(probs[2].item()),
        neutral=float(probs[1].item()),
        contradiction=float(probs[0].item()),
    )


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


def classify_relation_via_nli(context: str, subj: str, obj: str) -> RelationInference:
    """Return relation prediction with metadata using NLI templates."""
    _init_nli()
    norm_context = context.strip()
    if not norm_context:
        return RelationInference(
            label="co_occurs", entailment=0.0, score=0.0, margin=0.0
        )

    scored: List[Tuple[str, float, NliScores, Optional[NliScores]]] = []
    for rel, tmpl in _REL_TEMPLATES.items():
        forward = _nli_probs(norm_context, tmpl.format(SUBJ=subj, OBJ=obj))
        reverse: Optional[NliScores] = None
        if rel in _REL_TEMPLATES_REV:
            reverse = _nli_probs(
                norm_context, _REL_TEMPLATES_REV[rel].format(SUBJ=subj, OBJ=obj)
            )
        score = forward.signal
        if reverse is not None:
            penalty = max(0.0, reverse.signal)
            score -= _REL_DIRECTION_PENALTY * penalty
        scored.append((rel, score, forward, reverse))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_rel, best_score, best_forward, best_reverse = scored[0]
    runner_score = scored[1][1] if len(scored) > 1 else -1.0
    margin = best_score - runner_score

    reverse_entail = best_reverse.entailment if best_reverse else 0.0
    directional_conflict = (
        best_rel in _REL_TEMPLATES_REV
        and best_reverse is not None
        and (reverse_entail - best_forward.entailment) >= _REL_REVERSE_GAP
    )

    if (
        best_forward.entailment < _REL_ENTAILMENT_THRESHOLD
        or best_score < _REL_SCORE_THRESHOLD
        or margin < _REL_MARGIN_THRESHOLD
        or directional_conflict
    ):
        return RelationInference(
            label="co_occurs",
            entailment=best_forward.entailment,
            score=best_score,
            margin=max(0.0, margin),
            reverse_entailment=reverse_entail,
        )

    return RelationInference(
        label=best_rel,
        entailment=best_forward.entailment,
        score=best_score,
        margin=margin,
        reverse_entailment=reverse_entail,
    )


def extract_entities_relations(
    meta: Dict[str, Any], text: str
) -> Optional[PaperExtraction]:
    """Extract nodes and relations from a block of text using NER.

    This function uses a Stanza NER pipeline to identify entity spans. It
    collects unique entities, assigns a node type via `categorize_entity`, and
    builds a list of NodeRecord objects. Relations are generated in a naive
    fashion by connecting every distinct pair of entities found in the text with
    a 'co_occurs' edge. Evidence spans are omitted because this pipeline does
    not perform relation classification. Returns None if no entities are
    detected.
    """
    paper_label = meta.get("id") or meta.get("doi") or meta.get("title") or "<unknown>"
    if not text.strip():
        print(
            f"[warn] Skipping paper {paper_label}: no text available after preprocessing."
        )
        return None
    try:
        ner_results = _ner_pipeline(text)
    except Exception as exc:
        # If model inference fails, skip this document.
        print(f"[warn] Skipping paper {paper_label}: NER pipeline failed ({exc!r}).")
        return None
    entities: Dict[str, NodeRecord] = {}
    for ent in ner_results:
        if (
            "start" in ent
            and "end" in ent
            and ent["start"] is not None
            and ent["end"] is not None
        ):
            raw_name = text[ent["start"] : ent["end"]]
        else:
            raw_name = ent.get("word", "")
        raw_name = _clean_entity_surface(raw_name)
        if not raw_name:
            continue
        if raw_name.lower() in _NER_EXCLUDE_TERMS:
            continue
        tokens = ent.get("tokens") or []
        pos_tags = [tag.upper() for tag in ent.get("upos") or [] if tag]
        if pos_tags:
            informative = {"NOUN", "PROPN", "VERB", "ADJ"}
            banned = {"PRON", "DET", "PART", "INTJ", "SYM", "PUNCT"}
            if not any(tag in informative for tag in pos_tags):
                continue
            if all(tag in banned for tag in pos_tags):
                continue
            noun_like = {"NOUN", "PROPN"}
            if not any(tag in noun_like for tag in pos_tags):
                token_count = len(tokens) if tokens else len(raw_name.split())
                if token_count <= 1:
                    # Skip adjective-only spans like "psychiatric" that Stanza tags as entities.
                    continue
        lemma_hint = (ent.get("lemma") or "").strip()
        lemma_source = lemma_hint if lemma_hint else raw_name
        canonical_key = canonical_entity_key(lemma_source)
        if not canonical_key:
            continue
        canonical_display = canonical_entity_display(lemma_hint or raw_name) or raw_name
        record = entities.get(canonical_key)
        if record is None:
            label = ent.get("entity_group", "")
            node_type = categorize_entity(label)
            if not node_type:
                continue
            record = NodeRecord(
                canonical_name=canonical_display,
                lemma=canonical_key,
                node_type=node_type,
                synonyms=[],
                normalizations={},
            )
            entities[canonical_key] = record
        if lemma_hint:
            existing_lemmas = record.normalizations.get("lemmas", [])
            record.normalizations["lemmas"] = _dedupe_preserve_order(
                list(existing_lemmas) + [lemma_hint]
            )
        if pos_tags:
            existing_pos = record.normalizations.get("upos", [])
            record.normalizations["upos"] = _dedupe_preserve_order(
                list(existing_pos) + pos_tags
            )
        source_pkg = ent.get("source_package")
        if source_pkg:
            existing_sources = record.normalizations.get("ner_sources", [])
            record.normalizations["ner_sources"] = _dedupe_preserve_order(
                list(existing_sources) + [source_pkg]
            )
        if tokens:
            existing_tokens = record.normalizations.get("tokens", [])
            record.normalizations["tokens"] = _dedupe_preserve_order(
                list(existing_tokens) + tokens
            )
        _update_record_forms(record, [raw_name, canonical_display])
    if not entities:
        print(f"[warn] Skipping paper {paper_label}: no entities detected.")
        return None
    node_records = list(entities.values())
    node_records.sort(key=lambda r: r.canonical_name.lower())

    def _surface_form(record: NodeRecord) -> str:
        forms = record.normalizations.get("surface_forms", [])
        if isinstance(forms, list):
            for form in forms:
                if form and form in text:
                    return form
        return record.canonical_name

    surface_lookup = {rec.canonical_name: _surface_form(rec) for rec in node_records}

    relations: List[RelationRecord] = []
    for i in range(len(node_records)):
        for j in range(i + 1, len(node_records)):
            subj = node_records[i].canonical_name
            obj = node_records[j].canonical_name
            pred = "co_occurs"
            conf = 0.5
            sent_ctx = _pick_sentence_context(
                text, surface_lookup[subj], surface_lookup[obj]
            )
            inference = classify_relation_via_nli(sent_ctx, subj, obj)
            pred = inference.label
            if pred == "co_occurs":
                conf = 0.45
            else:
                conf = max(
                    0.55,
                    min(0.95, inference.entailment + 0.1 * inference.margin),
                )
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
                    qualifiers={
                        "nli_entailment": round(inference.entailment, 4),
                        "nli_margin": round(inference.margin, 4),
                        "nli_reverse_entailment": round(
                            inference.reverse_entailment, 4
                        ),
                        "nli_score": round(inference.score, 4),
                    },
                )
            )

    return PaperExtraction(
        paper_id=meta.get("id", ""),
        doi=meta.get("doi"),
        title=meta.get("title", ""),
        year=meta.get("year"),
        venue=meta.get("venue"),
        nodes=node_records,
        relations=relations,
    )


_LEMMA_OVERRIDES = {
    "antipsychotics": "antipsychotic",
    "bacteria": "bacterium",
    "criteria": "criterion",
    "data": "data",
    "diagnoses": "diagnosis",
    "news": "news",
    "research": "research",
    "series": "series",
    "species": "species",
    "tumours": "tumor",
    "tumors": "tumor",
}

_WORD_RE = re.compile(r"[A-Za-z]+")


def _clean_entity_surface(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def _apply_case_pattern(original: str, base_lower: str) -> str:
    if original.isupper():
        return base_lower.upper()
    if original.islower():
        return base_lower
    if original.istitle():
        return base_lower.title()
    return base_lower


def _lemmatize_word(word: str) -> str:
    if not word:
        return word
    lower = word.lower()
    if word.isupper() and len(word) <= 4:
        return word
    if lower in _LEMMA_OVERRIDES:
        return _apply_case_pattern(word, _LEMMA_OVERRIDES[lower])
    noun = _LEMMATIZER.lemmatize(lower, pos="n")
    verb = _LEMMATIZER.lemmatize(lower, pos="v")
    adj = _LEMMATIZER.lemmatize(lower, pos="a")
    candidates = [c for c in (noun, verb, adj) if c]
    base = min(candidates, key=len, default=lower)
    return _apply_case_pattern(word, base)


def _lemmatize_text(text: str) -> str:
    return _WORD_RE.sub(lambda m: _lemmatize_word(m.group(0)), text)


def canonical_entity_key(name: str) -> str:
    cleaned = _clean_entity_surface(name)
    if not cleaned:
        return ""
    lemma = _lemmatize_text(cleaned.lower())
    return re.sub(r"\s+", " ", lemma).strip()


def canonical_entity_display(name: str) -> str:
    cleaned = _clean_entity_surface(name)
    if not cleaned:
        return ""
    lemma = _lemmatize_text(cleaned)
    lemma = re.sub(r"\s+", " ", lemma).strip()
    if not lemma:
        return None
    if lemma.islower() and re.search(r"[a-z]", lemma):
        lemma = lemma[0].upper() + lemma[1:]
    return lemma


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _update_record_forms(record: NodeRecord, candidates: Iterable[str]) -> None:
    existing_forms = record.normalizations.get("surface_forms")
    if isinstance(existing_forms, list):
        forms = list(existing_forms)
    else:
        forms = [record.canonical_name]
    forms.extend(candidates)
    forms.append(record.canonical_name)
    normalized_forms = _dedupe_preserve_order(forms)
    record.normalizations["surface_forms"] = normalized_forms
    record.synonyms = _dedupe_preserve_order(
        list(record.synonyms)
        + [f for f in normalized_forms if f != record.canonical_name]
    )


def normalize_name(s: str) -> str:
    return _clean_entity_surface(s)


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
                    "lemma": n.lemma,
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

    if not rels_df.empty:
        support = rels_df.groupby(
            ["subject", "predicate", "object"], as_index=False
        ).agg(
            paper_ids=("paper_id", lambda s: sorted(set(s))),
            n_papers=("paper_id", "nunique"),
        )
        rels_df = rels_df.merge(
            support, on=["subject", "predicate", "object"], how="left"
        )
    return nodes_df, rels_df, papers_df


def build_multilayer_graph(
    nodes_df: pd.DataFrame, rels_df: pd.DataFrame
) -> nx.MultiDiGraph:
    """
    Construct a NetworkX MultiDiGraph from dataframes of nodes and relations.

    Nodes are annotated with their most common node_type (by frequency of mentions),
    as well as merged synonyms and normalizations. Edges are directed and store
    predicate, paper_id, directionality, evidence_span, confidence and qualifiers.
    Multiple edges between the same pair of nodes are preserved by omitting an
    explicit key when calling ``add_edge``. NetworkX will assign unique keys
    automatically
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
        if "lemma" in row and isinstance(row["lemma"], str) and row["lemma"]:
            graph.nodes[name]["lemma"] = row["lemma"]
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
            n_papers=int(row.get("n_papers", 1) or 1),
            paper_ids=(
                row["paper_ids"]
                if isinstance(row["paper_ids"], list)
                else json.loads(row["paper_ids"])
                if isinstance(row["paper_ids"], str)
                else []
            ),
        )
    return graph


PRIMITIVES = (int, float, bool)  # NOTE: str handled separately so we always clean it
_XML10_BAD = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\uFFFE\uFFFF]")


def _clean_str(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _XML10_BAD.sub(" ", s)
    # re-encode to drop any stray surrogates
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")


def _json_clean(obj) -> str:
    return _clean_str(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def _coerce_for_graphml(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return _clean_str(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    try:
        import numpy as np

        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, PRIMITIVES):
        return v
    if isinstance(v, (bytes, bytearray)):
        return _clean_str(v.decode("utf-8", "ignore"))
    if isinstance(v, (list, tuple, set, frozenset, Sequence)) and not isinstance(
        v, (str, bytes, bytearray)
    ):
        return _json_clean(list(v))
    if isinstance(v, Mapping):
        return _json_clean(v)
    if isinstance(v, type):
        return v.__name__
    return _clean_str(str(v))


def sanitize_graph_for_graphml(graph):
    # graph-level attributes
    for k, v in list(graph.graph.items()):
        new_k = _clean_str(str(k))
        new_v = _coerce_for_graphml(v)
        if new_v is not None:
            del graph.graph[k]
            graph.graph[new_k] = new_v

    # nodes
    for _, data in graph.nodes(data=True):
        for k in list(data.keys()):
            data[_clean_str(str(k))] = _coerce_for_graphml(data[k])

    # edges (multi or not)
    if graph.is_multigraph():
        for _, _, _, data in graph.edges(keys=True, data=True):
            for k in list(data.keys()):
                data[_clean_str(str(k))] = _coerce_for_graphml(data[k])
    else:
        for _, _, data in graph.edges(data=True):
            for k in list(data.keys()):
                data[_clean_str(str(k))] = _coerce_for_graphml(data[k])


def find_non_primitive_attrs(graph: nx.Graph):
    bad = []

    def has_bad(s):
        return isinstance(s, str) and _XML10_BAD.search(s) is not None

    # graph attrs
    for k, v in graph.graph.items():
        if has_bad(k) or has_bad(v):
            bad.append(("graph", k))
    # node attrs
    for n, d in graph.nodes(data=True):
        for k, v in d.items():
            if has_bad(k) or has_bad(v):
                bad.append(("node", n, k))
    # edge attrs
    es = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else graph.edges(data=True)
    )
    for *ends, d in es:
        for k, v in d.items():
            if has_bad(k) or has_bad(v):
                bad.append(("edge", tuple(ends), k))
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
        help="OpenAlex search query (title/abstract). Use ';' to provide multiple terms",
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
        help="Number of most-recent OA papers to fetch",
    )
    parser.add_argument(
        "--fetch-buffer",
        type=int,
        default=5,
        help="Extra works to over-fetch per bucket to compensate for duplicates",
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
    fetch_buffer = max(0, args.fetch_buffer)
    query_terms = _normalize_queries(args.query)

    def _fetch_bucket(
        sort: str, base_n: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        full = fetch_top_n(
            args.query,
            args.filters,
            sort=sort,
            n=base_n + fetch_buffer,
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in full:
            key = rec.get("search_query", "")
            grouped.setdefault(key, []).append(rec)

        primary: List[Dict[str, Any]] = []
        extra: List[Dict[str, Any]] = []
        for term in query_terms:
            term_results = grouped.get(term, [])
            primary.extend(term_results[:base_n])
            extra.extend(term_results[base_n:])
        return primary, extra

    top_primary, top_extra = _fetch_bucket("cited_by_count:desc", args.n_top_cited)
    recent_primary, recent_extra = _fetch_bucket(
        "publication_date:desc", args.n_most_recent
    )

    terms_multiplier = max(len(query_terms), 1)
    target_total = (args.n_top_cited + args.n_most_recent) * terms_multiplier
    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    def add_candidates(candidates: Iterable[Dict[str, Any]]) -> None:
        for rec in candidates:
            rid = rec.get("id")
            if not rid or rid in seen_ids:
                continue
            records.append(rec)
            seen_ids.add(rid)
            if len(records) >= target_total:
                return

    add_candidates(top_primary)
    add_candidates(recent_primary)
    if len(records) < target_total:
        add_candidates(top_extra)
    if len(records) < target_total:
        add_candidates(recent_extra)

    if len(records) < target_total:
        deficit = target_total - len(records)
        print(
            f"[warn] Only {len(records)} unique works found (short by {deficit}). "
            "Increase --fetch-buffer or relax filters to retrieve more."
        )
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
        study_sections = (
            extract_results_and_discussion(fulltext or "") if fulltext else None
        )
        # prefer Results/Discussion sections; fallback to abstract
        text_for_ie = (
            study_sections if study_sections else (rec.get("abstract", "") or "")
        )
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
