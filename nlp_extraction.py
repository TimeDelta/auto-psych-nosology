"""Entity and relation extraction using Stanza NER and a biomedical NLI model."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import spacy
import stanza
import torch
from nltk.tokenize import sent_tokenize
from scispacy.linking import EntityLinker
from spacy.lang.en.stop_words import STOP_WORDS as _SPACY_STOP_WORDS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models import (
    EVIDENCE_RELATIONS,
    ClaimDescriptor,
    NodeRecord,
    PaperExtraction,
    RelationRecord,
)
from text_normalization import (
    canonical_entity_display,
    canonical_entity_key,
    clean_entity_surface,
    dedupe_preserve_order,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_LOG_LEVEL_NAME = os.getenv("AUTO_PSYCH_LOG_LEVEL", "INFO").strip().upper()
logger.setLevel(getattr(logging, _LOG_LEVEL_NAME, logging.INFO))
logging.getLogger("stanza").setLevel(logging.INFO)


def _log_step_duration(
    step: str,
    start_time: float,
    *,
    paper: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    elapsed = time.perf_counter() - start_time
    detail_suffix = f" ({details})" if details else ""
    if paper:
        logger.debug(
            "[timing] %s completed in %.3f seconds for %s%s",
            step,
            elapsed,
            paper,
            detail_suffix,
        )
    else:
        logger.debug(
            "[timing] %s completed in %.3f seconds%s",
            step,
            elapsed,
            detail_suffix,
        )


RELATION_ROLE_CONSTRAINTS: Dict[str, Dict[str, Set[str]]] = {
    "treats": {"subject": {"Treatment"}, "object": {"Diagnosis", "Symptom"}},
    "biomarker_for": {"subject": {"Biomarker"}, "object": {"Diagnosis", "Symptom"}},
    "measure_of": {"subject": {"Measure"}, "object": {"Diagnosis", "Symptom"}},
    "predicts": {
        "subject": {"Biomarker", "Measure", "Symptom"},
        "object": {"Diagnosis", "Symptom"},
    },
}

_FALLBACK_RELATION_RULES: Tuple[Tuple[Set[str], Set[str], str], ...] = (
    ({"Treatment"}, {"Diagnosis", "Symptom"}, "treats"),
    ({"Biomarker"}, {"Diagnosis", "Symptom"}, "biomarker_for"),
    ({"Measure"}, {"Diagnosis", "Symptom"}, "measure_of"),
    ({"Symptom"}, {"Diagnosis", "Symptom"}, "predicts"),
)

# recognized named entities lemmatized before check
_GENERIC_SPAN_BLOCKLIST_DEFAULT = [
    "adverse effect",
    "side effect",
    "brain organoid",
    "product",
    "treatment",
    "problem",
    "study",
    "this study",
    "the symptom description",
    "symptom description",
    "the brain",
    "brain",
    "the chasm",
    "patient",
    "control",
    "participant",
    "subject",
    "human",
    "donor",
    "disease relevant cell",
    "efficacy of compound",
    "Medium to large effect size",
    "The open access option",
    "Randomization in step",
    "CRISPR",
    "cell",
    "clinical practice set",
    "learn algorithm",
    "collect datum",
    "it customize hardware",
    "very low probability",
    "adaptive trial design",
    "illness",
    "disorder",
    "symptom",
]

_UMLS_SEMANTIC_TYPE_TO_NODE: Dict[str, str] = {
    # Symptom-centric semantic types
    "T184": "Symptom",  # Sign or Symptom
    "T033": "Symptom",  # Finding
    "T041": "Symptom",  # Mental Process
    "T048": "Symptom",  # Mental or Behavioral Dysfunction
    # Broader clinical condition types to prevent dropping key entities
    "T047": "Diagnosis",  # Disease or Syndrome
    "T046": "Diagnosis",  # Pathologic Function
    "T191": "Diagnosis",  # Neoplastic Process
    # Treatment-oriented semantic types
    "T061": "Treatment",  # Therapeutic or Preventive Procedure
    "T200": "Treatment",  # Clinical Drug
    "T121": "Treatment",  # Pharmacologic Substance
    # Biomarker / biological entity types
    "T123": "Biomarker",  # Biologically Active Substance
    "T109": "Biomarker",  # Organic Chemical
    "T126": "Biomarker",  # Enzyme
    "T127": "Biomarker",  # Vitamin
}


def _build_span_blocklist() -> Set[str]:
    base_terms: Set[str] = set(_GENERIC_SPAN_BLOCKLIST_DEFAULT)
    extra_sources = (os.getenv("NER_GENERIC_SPAN_BLOCKLIST"),)
    for source in extra_sources:
        if not source:
            continue
        base_terms.update(part.strip() for part in source.split(",") if part.strip())

    blocklist: Set[str] = set()
    for term in base_terms:
        lower = term.lower()
        if lower:
            blocklist.add(lower)
        canonical = canonical_entity_key(term)
        if canonical:
            blocklist.add(canonical)
    return blocklist


_RELATION_MODE = "nli"
_NLI_MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"
_SUBSTANTIVE_REL_TEMPLATES = {
    "treats": "{SUBJ} treats {OBJ}.",
    "biomarker_for": "{SUBJ} is a biomarker for {OBJ}.",
    "measure_of": "{SUBJ} is a measure of {OBJ}.",
    "predicts": "{SUBJ} predicts {OBJ}.",
}
_SUBSTANTIVE_REL_TEMPLATES_REV = {
    "treats": "{OBJ} treats {SUBJ}.",
    "biomarker_for": "{OBJ} is a biomarker for {SUBJ}.",
    "measure_of": "{OBJ} is a measure of {SUBJ}.",
    "predicts": "{OBJ} predicts {SUBJ}.",
}
_EVIDENCE_REL_TEMPLATES = {
    "supports": "{SUBJ} is positively associated with {OBJ}.",
    "contradicts": "{SUBJ} is negatively associated with {OBJ}.",
    "replicates": "{SUBJ} replicates findings reported for {OBJ}.",
    "null_reported": "{SUBJ} reports null findings with respect to {OBJ}.",
}

_REL_DIRECTION_PENALTY = float(os.getenv("RELATION_NLI_REVERSE_PENALTY", "0.5"))
_REL_SCORE_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_SCORE", "0.05"))
_REL_MARGIN_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_MARGIN", "0.04"))
_REL_ENTAILMENT_THRESHOLD = float(os.getenv("RELATION_NLI_MIN_ENT", "0.6"))
_REL_REVERSE_GAP = float(os.getenv("RELATION_NLI_REVERSE_GAP", "0.05"))


def _stanza_ner_packages() -> List[str]:
    packages: List[str] = []

    configured = os.getenv("STANZA_NER_PACKAGES", "").strip()
    if configured:
        packages.extend(pkg.strip() for pkg in configured.split("|") if pkg.strip())
    else:
        packages.extend(
            [
                "bc5cdr",
                "jnlpba",
            ]
        )
    return dedupe_preserve_order(packages)


@dataclass
class NliScores:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def signal(self) -> float:
        return self.entailment - self.contradiction


@dataclass
class RelationInference:
    label: str
    entailment: float
    score: float
    margin: float
    reverse_entailment: float = 0.0


@dataclass
class ExtractionConfig:
    stanza_lang: str = field(default_factory=lambda: os.getenv("STANZA_LANG", "en"))
    stanza_processors: str = field(
        default_factory=lambda: os.getenv(
            "STANZA_PROCESSORS", "tokenize,mwt,pos,lemma,depparse,ner,coref"
        )
    )
    stanza_tokenize_pkg: str = field(
        default_factory=lambda: os.getenv("STANZA_TOKENIZE_PACKAGE", "").strip()
        or "default"
    )
    stanza_ner_packages: Sequence[str] = field(
        default_factory=lambda: _stanza_ner_packages()
    )
    stanza_extra_packages: Sequence[str] = field(
        default_factory=lambda: [
            pkg.strip()
            for pkg in os.getenv("STANZA_EXTRA_PACKAGES", "").split("|")
            if pkg.strip()
        ]
    )
    nli_model_name: str = _NLI_MODEL_NAME
    relation_role_constraints: Dict[str, Dict[str, Set[str]]] = field(
        default_factory=lambda: dict(RELATION_ROLE_CONSTRAINTS)
    )
    span_blocklist: Set[str] = field(default_factory=_build_span_blocklist)
    spacy_model: Optional[str] = field(
        default_factory=lambda: os.getenv("SPACY_BIOMED_MODEL", "en_core_sci_sm")
    )
    enable_ontology_filter: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ONTOLOGY_FILTER", "0") != "0"
    )
    ontology_min_score: float = field(
        default_factory=lambda: float(os.getenv("ONTOLOGY_MIN_SCORE", "0.5"))
    )
    spacy_linker_name: str = field(
        default_factory=lambda: os.getenv("SPACY_LINKER_NAME", "umls_context")
    )
    prefilter_model_name: Optional[str] = field(
        default_factory=lambda: (  # cross-encoder/nli-deberta-v3-base decimates recall but halves total time
            os.getenv("REL_PREFILTER_MODEL", "").strip()
            or None  # TODO: check other models
        )
    )
    prefilter_min_score: float = field(
        default_factory=lambda: float(os.getenv("REL_PREFILTER_MIN_SCORE", "0.55"))
    )
    prefilter_min_context_chars: int = field(
        default_factory=lambda: int(os.getenv("REL_PREFILTER_MIN_CONTEXT_CHARS", "30"))
    )
    prefilter_hypothesis_template: str = field(
        default_factory=lambda: os.getenv(
            "REL_PREFILTER_TEMPLATE",
            "{SUBJ} is biologically related to {OBJ}.",
        )
    )


class EntityRelationExtractor:
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        stanza_pipelines: Optional[List[Tuple[str, "stanza.Pipeline"]]] = None,
        nli_tokenizer=None,
        nli_model=None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self._pipelines = stanza_pipelines
        self._nli_tokenizer = nli_tokenizer
        self._nli_model = nli_model
        self._nli_lock = threading.Lock()
        self._relation_type_cache: Dict[Tuple[str, str], bool] = {}
        self._spacy_nlp = None
        self._spacy_linker_available = False
        self._latest_link_scores: Dict[str, float] = {}
        self._scispacy_linker = None
        self._umls_type_cache: Dict[str, Optional[str]] = {}
        self._stop_words: Set[str] = set(_SPACY_STOP_WORDS)
        self._span_blocklist = set(self.config.span_blocklist)
        self._prefilter_model = None
        self._prefilter_tokenizer = None
        self._prefilter_entailment_index: Optional[int] = None

    def _stanza_use_gpu(self) -> bool:
        if os.getenv("STANZA_USE_GPU", "").strip() == "0":
            use_gpu = False
        else:
            use_gpu = torch.cuda.is_available()
        logger.debug("Stanza GPU enabled: %s", use_gpu)
        return use_gpu

    def _parse_package_spec(self, spec: str) -> Optional[Any]:
        spec = (spec or "").strip()
        if not spec:
            return None
        if any(ch in spec for ch in "=;,"):
            cfg: Dict[str, str] = {}
            for part in re.split(r"[;,]", spec):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                key, value = part.split("=", 1)
                cfg[key.strip()] = value.strip()
            return cfg or None
        return spec

    def _build_stanza_package_candidates(self) -> List[Any]:
        forced = os.getenv("STANZA_FORCE_PACKAGE", "").strip()
        if forced:
            parsed = self._parse_package_spec(forced)
            candidates = [parsed] if parsed else []
            logger.debug("Forced Stanza package candidates: %s", candidates)
            return candidates

        candidates: List[Any] = []
        base_tokenize_pkg = self.config.stanza_tokenize_pkg

        def normalize_candidate(value: Any) -> Any:
            if isinstance(value, str):
                return {"tokenize": base_tokenize_pkg, "ner": value}
            if isinstance(value, dict):
                candidate = dict(value)
                candidate.setdefault("tokenize", base_tokenize_pkg)
                return candidate
            return value

        for pkg in self.config.stanza_ner_packages:
            parsed = self._parse_package_spec(pkg) if isinstance(pkg, str) else pkg
            if not parsed:
                continue
            candidate = normalize_candidate(parsed)
            if candidate not in candidates:
                candidates.append(candidate)
        for extra in self.config.stanza_extra_packages:
            parsed = self._parse_package_spec(extra)
            if not parsed:
                continue
            candidate = normalize_candidate(parsed)
            if candidate not in candidates:
                candidates.append(candidate)
        for fallback in ("craft", "mimic"):
            candidate = normalize_candidate(fallback)
            if candidate not in candidates:
                candidates.append(candidate)
        logger.debug("Stanza package candidates: %s", candidates)
        return candidates

    def _ensure_stanza_pipelines(self) -> List[Tuple[str, "stanza.Pipeline"]]:
        if self._pipelines is not None:
            return self._pipelines
        lang = self.config.stanza_lang
        raw_processors = self.config.stanza_processors
        if isinstance(raw_processors, str):
            proc_parts = [
                part.strip() for part in raw_processors.split(",") if part.strip()
            ]
        else:
            proc_parts = list(raw_processors or [])
        if "parse" in proc_parts and "depparse" not in proc_parts:
            proc_parts = [
                "depparse" if part == "parse" else part for part in proc_parts
            ]
        processors = ",".join(proc_parts)
        pipelines: List[Tuple[str, "stanza.Pipeline"]] = []
        for package in self._build_stanza_package_candidates():
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
                pass

            pipeline_kwargs = {
                "processors": processors,
                "use_gpu": self._stanza_use_gpu(),
                "verbose": False,
            }
            if package:
                pipeline_kwargs["package"] = package
            try:
                pipeline = stanza.Pipeline(lang, **pipeline_kwargs)
                pipelines.append((str(pkg_display), pipeline))
                logger.info("Using Stanza biomedical NER package %s.", pkg_display)
            except Exception as exc:
                logger.warning(
                    "Stanza package %s unavailable (%s); trying fallback.",
                    pkg_display,
                    exc,
                )
                continue
        if not pipelines:
            logger.error(
                "Unable to initialise any Stanza biomedical NER package; extractions will be empty."
            )
        else:
            pkg_labels = ", ".join(pkg for pkg, _ in pipelines)
            logger.info(
                "Finished loading %d Stanza NER pipeline(s): %s",
                len(pipelines),
                pkg_labels,
            )
        self._pipelines = pipelines
        return self._pipelines

    def _ensure_nli(self):
        if self._nli_tokenizer is not None and self._nli_model is not None:
            return
        with self._nli_lock:
            if self._nli_tokenizer is None or self._nli_model is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.nli_model_name, token=None, local_files_only=False
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.nli_model_name, token=None, use_safetensors=True
                )
                model.eval()
                self._nli_tokenizer = tokenizer
                self._nli_model = model

    def _nli_probs(self, premise: str, hypothesis: str) -> NliScores:
        self._ensure_nli()
        assert self._nli_tokenizer is not None and self._nli_model is not None
        with torch.no_grad():
            encoding = self._nli_tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            logits = self._nli_model(**encoding).logits.softmax(-1)[0]
        return NliScores(
            entailment=float(logits[2].item()),
            neutral=float(logits[1].item()),
            contradiction=float(logits[0].item()),
        )

    def _node_type_from_umls(self, kb_id: Optional[str]) -> Optional[str]:
        if not kb_id or not self._scispacy_linker:
            return None
        cached = self._umls_type_cache.get(kb_id)
        if kb_id in self._umls_type_cache:
            return cached
        node_type: Optional[str] = None
        concept = None
        kb = getattr(self._scispacy_linker, "kb", None)
        if kb is not None:
            concept = getattr(kb, "cui_to_entity", {}).get(kb_id)
            if concept is None and hasattr(kb, "get_concept"):
                concept = kb.get_concept(kb_id)
        concept_name = ""
        if concept is not None:
            concept_name = (
                getattr(concept, "canonical_name", "")
                or getattr(concept, "preferred_name", "")
                or getattr(concept, "name", "")
            )
            semantic_types = getattr(concept, "types", None) or getattr(
                concept, "semantic_types", None
            )
            if semantic_types:
                for tui in semantic_types:
                    mapped = _UMLS_SEMANTIC_TYPE_TO_NODE.get(str(tui).upper())
                    if mapped:
                        node_type = mapped
                        break
        if node_type == "Symptom" and concept_name:
            lowered = concept_name.lower()
            if any(
                keyword in lowered
                for keyword in (
                    "disorder",
                    "disorders",
                    "disease",
                    "diseases",
                    "syndrome",
                    "syndromes",
                )
            ):
                node_type = "Diagnosis"
        self._umls_type_cache[kb_id] = node_type
        return node_type

    def categorize_entity(
        self, label: str, kb_id: Optional[str] = None
    ) -> Optional[str]:
        lbl = label.upper()
        direct_map = {
            "PROBLEM": "Symptom",
            "FINDING": "Symptom",
            "SIGN": "Symptom",
            "BEHAVIOR": "Symptom",
            "SYMPTOM": "Symptom",
            #
            "DIAGNOSIS": "Diagnosis",
            "DISEASE": "Diagnosis",
            "SYNDROME": "Diagnosis",
            #
            "TEST": "Measure",
            "MEASUREMENT": "Measure",
            "MEASURE": "Measure",
            "ASSESSMENT": "Measure",
            "LAB": "Measure",
            #
            "TREATMENT": "Treatment",
            "PROCEDURE": "Treatment",
            "THERAPY": "Treatment",
            "MEDICATION": "Treatment",
            "DEVICE": "Treatment",
            "CHEMICAL": "Treatment",
            "DRUG": "Treatment",
            "MED": "Treatment",
            #
            "ANATOMY": "Biomarker",
            "ANATOMICAL": "Biomarker",
            "CELL_LINE": "Biomarker",
            "CELL_TYPE": "Biomarker",
            "DNA": "Biomarker",
            "PROEIN": "Biomarker",
            "RNA": "Biomarker",
            "GENE": "Biomarker",
            "BIOMARKER": "Biomarker",
        }
        if lbl in direct_map:
            return direct_map[lbl]
        umls_type = self._node_type_from_umls(kb_id)
        if umls_type:
            return umls_type
        return None

    def _relation_allowed_for_types(
        self, predicate: str, subj_type: str, obj_type: str
    ) -> bool:
        rules = self.config.relation_role_constraints.get(predicate)
        if not rules:
            return True
        allowed_subject = rules.get("subject")
        allowed_object = rules.get("object")
        if allowed_subject and subj_type not in allowed_subject:
            return False
        if allowed_object and obj_type not in allowed_object:
            return False
        return True

    def _fallback_relation_for_types(
        self, subj_type: str, obj_type: str
    ) -> Optional[str]:
        for subj_candidates, obj_candidates, predicate in _FALLBACK_RELATION_RULES:
            if subj_type in subj_candidates and obj_type in obj_candidates:
                return predicate
        return None

    def _pair_has_potential_relation(self, type_a: str, type_b: str) -> bool:
        cache_key = (type_a, type_b)
        cached = self._relation_type_cache.get(cache_key)
        if cached is not None:
            return cached

        possible = False
        for predicate in self.config.relation_role_constraints:
            if self._relation_allowed_for_types(
                predicate, type_a, type_b
            ) or self._relation_allowed_for_types(predicate, type_b, type_a):
                possible = True
                break

        if not possible:
            for subj_candidates, obj_candidates, _ in _FALLBACK_RELATION_RULES:
                if (type_a in subj_candidates and type_b in obj_candidates) or (
                    type_b in subj_candidates and type_a in obj_candidates
                ):
                    possible = True
                    break

        self._relation_type_cache[cache_key] = possible
        # store symmetric result to avoid recomputation
        self._relation_type_cache[(type_b, type_a)] = possible
        return possible

    def _ensure_spacy_pipeline(self) -> Optional[Any]:
        if spacy is None:
            self._scispacy_linker = None
            self._umls_type_cache = {}
            return None
        if self.config.spacy_model in (None, ""):
            self._scispacy_linker = None
            self._umls_type_cache = {}
            return None
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            nlp = spacy.load(
                self.config.spacy_model,
                disable=["parser", "textcat"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "spaCy model %s unavailable (%s); disabling spaCy integration.",
                self.config.spacy_model,
                exc,
            )
            self._scispacy_linker = None
            self._umls_type_cache = {}
            self._spacy_nlp = None
            return None

        try:
            if (
                "scispacy_linker" not in nlp.pipe_names
                and "entity_linker" not in nlp.pipe_names
            ):
                nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": self.config.spacy_linker_name,
                    },
                )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Unable to add SciSpaCy linker (%s); continuing without ontology filter.",
                exc,
            )
        self._spacy_linker_available = (
            "scispacy_linker" in nlp.pipe_names or "entity_linker" in nlp.pipe_names
        )
        if self._spacy_linker_available:
            try:
                self._scispacy_linker = nlp.get_pipe("scispacy_linker")
            except Exception:
                try:
                    self._scispacy_linker = nlp.get_pipe("entity_linker")
                except Exception:
                    self._scispacy_linker = None
                    self._spacy_linker_available = False
        else:
            self._scispacy_linker = None
        # Reset cached UMLS lookups whenever the spaCy pipeline is rebuilt.
        self._umls_type_cache = {}
        self._spacy_nlp = nlp
        return self._spacy_nlp

    def _ensure_prefilter_model(self) -> None:
        if self._prefilter_model is not None or not self.config.prefilter_model_name:
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.prefilter_model_name,
                token=None,
                use_fast=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.prefilter_model_name,
                token=None,
                use_safetensors=True,
            )
            model.eval()
            self._prefilter_tokenizer = tokenizer
            self._prefilter_model = model
            entail_idx = None
            for idx, label in model.config.id2label.items():
                if label.lower() == "entailment":
                    entail_idx = idx
                    break
            if entail_idx is None:
                entail_idx = (
                    2 if model.config.num_labels >= 3 else model.config.num_labels - 1
                )
            self._prefilter_entailment_index = entail_idx
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Prefilter model %s unavailable (%s); disabling fast relation filter.",
                self.config.prefilter_model_name,
                exc,
            )
            self._prefilter_model = None
            self._prefilter_tokenizer = None
            self._prefilter_entailment_index = None
            self.config.prefilter_model_name = None

    def _passes_entity_heuristics(
        self,
        surface: str,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
    ) -> bool:
        surface = surface.strip()
        if not surface:
            return False
        if sum(ch.isalpha() for ch in surface) == 0:
            return False
        if tokens:
            stripped_tokens = [tok.lower() for tok in tokens if tok]
            if stripped_tokens and all(
                token in self._stop_words for token in stripped_tokens
            ):
                return False
            if len(stripped_tokens) == 1:
                token = stripped_tokens[0]
                if len(token) < 3 or token in self._stop_words:
                    return False
        informative = {"NOUN", "PROPN", "ADJ", "VERB"}
        if pos_tags and not any(tag in informative for tag in pos_tags):
            return False
        return True

    def _passes_relation_prefilter(self, context: str, subj: str, obj: str) -> bool:
        context = context.strip()
        if len(context) < self.config.prefilter_min_context_chars:
            return False
        lower_context = context.lower()
        subj_lower = subj.lower()
        obj_lower = obj.lower()
        if subj_lower not in lower_context or obj_lower not in lower_context:
            return False
        if not self.config.prefilter_model_name:
            return True
        self._ensure_prefilter_model()
        if not (
            self._prefilter_model
            and self._prefilter_tokenizer
            and self._prefilter_entailment_index is not None
        ):
            return True
        hypothesis = self.config.prefilter_hypothesis_template.format(
            SUBJ=subj,
            OBJ=obj,
        )
        try:
            inputs = self._prefilter_tokenizer(
                context,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = self._prefilter_model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            score = probs[self._prefilter_entailment_index].item()
            return score >= self.config.prefilter_min_score
        except Exception as exc:  # pragma: no cover
            logger.warning("Prefilter scoring failed (%s); allowing pair.", exc)
            return True

    def _coref_mention_info(
        self, doc: "stanza.Document", mention: Any
    ) -> Optional[Dict[str, Any]]:
        try:
            sentence_idx = int(getattr(mention, "sentence", -1))
            start_token = int(getattr(mention, "start_word", -1))
            end_token = int(getattr(mention, "end_word", -1))
        except Exception:
            return None
        if sentence_idx < 0 or start_token < 0 or end_token <= start_token:
            return None
        sentences = getattr(doc, "sentences", [])
        if sentence_idx >= len(sentences):
            return None
        sentence = sentences[sentence_idx]
        tokens = getattr(sentence, "tokens", [])
        if end_token > len(tokens):
            return None
        start = tokens[start_token]
        end = tokens[end_token - 1]
        start_char = getattr(start, "start_char", None)
        end_char = getattr(end, "end_char", None)
        if start_char is None or end_char is None:
            return None
        surface = doc.text[start_char:end_char]
        upos: List[str] = []
        lemmas: List[str] = []
        for token in tokens[start_token:end_token]:
            for word in getattr(token, "words", []) or []:
                if getattr(word, "upos", None):
                    upos.append(word.upos)
                if getattr(word, "lemma", None):
                    lemmas.append(word.lemma)
        return {
            "sentence_idx": sentence_idx,
            "start_char": int(start_char),
            "end_char": int(end_char),
            "surface": clean_entity_surface(surface),
            "upos": upos,
            "lemmas": lemmas,
            "start_token": start_token,
            "end_token": end_token,
        }

    def _select_coref_antecedent(
        self, mentions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not mentions:
            raise ValueError("Cannot select antecedent from empty mention list")
        ranked: List[Tuple[int, int, Dict[str, Any]]] = []
        for idx, info in enumerate(mentions):
            pos = info.get("upos", [])
            score = 0
            if any(tag == "PROPN" for tag in pos):
                score = 3
            elif any(tag == "NOUN" for tag in pos):
                score = 2
            elif any(tag == "ADJ" for tag in pos):
                score = 1
            length = info.get("end_char", 0) - info.get("start_char", 0)
            ranked.append((score, -length, idx))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_index = ranked[0][2]
        return mentions[best_index]

    def _build_coref_reference_map(
        self, doc: "stanza.Document"
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        chains = getattr(doc, "coref", None)
        if not chains:
            return {}
        references: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for chain in chains:
            mention_infos: List[Dict[str, Any]] = []
            for mention in getattr(chain, "mentions", []) or []:
                info = self._coref_mention_info(doc, mention)
                if info is None:
                    continue
                mention_infos.append(info)
            if not mention_infos:
                continue
            mention_infos.sort(key=lambda m: (m["start_char"], m["end_char"]))
            antecedent = self._select_coref_antecedent(mention_infos)
            antecedent_key = (antecedent["start_char"], antecedent["end_char"])
            for info in mention_infos:
                key = (info["start_char"], info["end_char"])
                if key == antecedent_key:
                    continue
                if key in references:
                    continue
                references[key] = {
                    "text": antecedent["surface"],
                    "start": antecedent["start_char"],
                    "end": antecedent["end_char"],
                    "surface": info["surface"],
                }
        return references

    def _pick_sentence_context(
        self, text: str, s1: str, s2: str, window: int = 2
    ) -> str:
        sentences = sent_tokenize(text)
        indexes = [
            i
            for i, sentence in enumerate(sentences)
            if (s1 in sentence and s2 in sentence)
        ]
        if not indexes:
            idx1 = next(
                (i for i, sentence in enumerate(sentences) if s1 in sentence), None
            )
            idx2 = next(
                (i for i, sentence in enumerate(sentences) if s2 in sentence), None
            )
            if idx1 is None or idx2 is None:
                return text[:2000]
            lo, hi = sorted([idx1, idx2])
            lo = max(0, lo - window)
            hi = min(len(sentences), hi + window + 1)
            return " ".join(sentences[lo:hi])
        lo = max(0, min(indexes) - window)
        hi = min(len(sentences), max(indexes) + window + 1)
        return " ".join(sentences[lo:hi])

    def _score_relation_templates(
        self,
        context: str,
        subj: str,
        obj: str,
        templates: Dict[str, str],
        reverse_templates: Optional[Dict[str, str]] = None,
    ) -> Optional[RelationInference]:
        if not context.strip():
            return None
        scored: List[Tuple[str, float, NliScores, Optional[NliScores]]] = []
        for rel, template in templates.items():
            forward = self._nli_probs(context, template.format(SUBJ=subj, OBJ=obj))
            reverse: Optional[NliScores] = None
            if reverse_templates and rel in reverse_templates:
                reverse = self._nli_probs(
                    context, reverse_templates[rel].format(SUBJ=subj, OBJ=obj)
                )
            score = forward.signal
            if reverse is not None:
                penalty = max(0.0, reverse.signal)
                score -= _REL_DIRECTION_PENALTY * penalty
            scored.append((rel, score, forward, reverse))
        if not scored:
            return None
        scored.sort(key=lambda item: item[1], reverse=True)
        best_rel, best_score, best_forward, best_reverse = scored[0]
        runner_score = scored[1][1] if len(scored) > 1 else -1.0
        margin = best_score - runner_score
        reverse_entail = best_reverse.entailment if best_reverse else 0.0
        directional_conflict = (
            bool(reverse_templates)
            and best_rel in (reverse_templates or {})
            and best_reverse is not None
            and (reverse_entail - best_forward.entailment) >= _REL_REVERSE_GAP
        )
        if (
            best_forward.entailment < _REL_ENTAILMENT_THRESHOLD
            or best_score < _REL_SCORE_THRESHOLD
            or margin < _REL_MARGIN_THRESHOLD
            or directional_conflict
        ):
            return None
        return RelationInference(
            label=best_rel,
            entailment=best_forward.entailment,
            score=best_score,
            margin=margin,
            reverse_entailment=reverse_entail,
        )

    def classify_relation_via_nli(
        self, context: str, subj: str, obj: str
    ) -> Tuple[Optional[RelationInference], Optional[RelationInference]]:
        relation = self._score_relation_templates(
            context,
            subj,
            obj,
            _SUBSTANTIVE_REL_TEMPLATES,
            _SUBSTANTIVE_REL_TEMPLATES_REV,
        )
        evidence = self._score_relation_templates(
            context, subj, obj, _EVIDENCE_REL_TEMPLATES
        )
        return relation, evidence

    def _ner_pipeline(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        self._latest_link_scores = {}
        spacy_entities: Dict[str, List[Dict[str, Any]]] = {}
        spacy_allowed_keys: Optional[Set[str]] = None
        spacy_nlp = self._ensure_spacy_pipeline()
        if spacy_nlp is not None:
            try:
                spacy_doc = spacy_nlp(text)
                for ent in spacy_doc.ents:
                    start = ent.start_char
                    end = ent.end_char
                    label = getattr(ent, "label_", "")
                    tokens = [token.text for token in ent]
                    lemmas = [
                        getattr(token, "lemma_", "") or token.text for token in ent
                    ]
                    pos_tags = [getattr(token, "pos_", "").upper() for token in ent]
                    canonical = canonical_entity_key(ent.text)
                    kb_id = None
                    link_score = None
                    if self._spacy_linker_available:
                        kb_ents = getattr(ent._, "kb_ents", None)
                        if kb_ents:
                            kb_id, link_score = kb_ents[0]
                    entry = {
                        "start": start,
                        "end": end,
                        "entity_group": label,
                        "word": ent.text,
                        "source_package": f"spacy:{self.config.spacy_model}",
                        "lemma": " ".join(lemmas).strip(),
                        "upos": pos_tags,
                        "xpos": [getattr(token, "tag_", "") for token in ent],
                        "tokens": tokens,
                        "coref_antecedent": None,
                        "canonical_key": canonical,
                        "ontology_score": link_score,
                        "kb_id": kb_id,
                    }
                    key = canonical or f"{start}:{end}:{label}"
                    spacy_entities.setdefault(key, []).append(entry)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "spaCy extraction failed (%s); disabling spaCy integration.",
                    exc,
                )
                self._spacy_nlp = None
                spacy_entities = {}

        if spacy_entities:
            for key, entries in spacy_entities.items():
                best = max(entries, key=lambda item: item.get("ontology_score") or 0.0)
                canonical = best.get("canonical_key")
                score = best.get("ontology_score")
                if canonical and score is not None:
                    self._latest_link_scores[canonical] = score

        pipelines = self._ensure_stanza_pipelines()
        results: List[Dict[str, Any]] = []
        seen_spans: Set[Tuple[int, int, str, str]] = set()

        if pipelines:
            for pkg_display, pipeline in pipelines:
                doc = pipeline(text)
                coref_map = self._build_coref_reference_map(doc)
                entities = getattr(doc, "ents", None)
                if entities is None:
                    entities = getattr(doc, "entities", [])
                for ent in entities:
                    start = getattr(ent, "start_char", None)
                    end = getattr(ent, "end_char", None)
                    label = getattr(ent, "type", "")
                    words = getattr(ent, "words", [])
                    lemmas: List[str] = []
                    upos: List[str] = []
                    xpos: List[str] = []
                    tokens: List[str] = []
                    for word in words:
                        token = getattr(word, "text", "")
                        if token:
                            tokens.append(token)
                        lemma = getattr(word, "lemma", "")
                        if lemma:
                            lemmas.append(lemma)
                        elif token:
                            lemmas.append(token)
                        pos = getattr(word, "upos", "")
                        if pos:
                            upos.append(pos)
                        xpos_tag = getattr(word, "xpos", "")
                        if xpos_tag:
                            xpos.append(xpos_tag)
                    if start is not None and end is not None:
                        key = (start, end, label, pkg_display)
                        if key in seen_spans:
                            continue
                        seen_spans.add(key)
                    canonical = canonical_entity_key(getattr(ent, "text", ""))
                    if spacy_allowed_keys is not None:
                        if not canonical or canonical not in spacy_allowed_keys:
                            continue
                    if start is not None and end is not None:
                        coref_info = coref_map.get((start, end))
                    else:
                        coref_info = None
                    result = {
                        "start": start,
                        "end": end,
                        "entity_group": label,
                        "word": getattr(ent, "text", ""),
                        "source_package": pkg_display,
                        "lemma": " ".join(lemmas).strip(),
                        "upos": upos,
                        "xpos": xpos,
                        "tokens": tokens,
                        "coref_antecedent": coref_info,
                        "canonical_key": canonical,
                        "ontology_score": self._latest_link_scores.get(canonical),
                    }
                    results.append(result)

        if spacy_entities and not pipelines:
            for entries in spacy_entities.values():
                for entry in entries:
                    key = (
                        entry.get("start"),
                        entry.get("end"),
                        entry.get("entity_group"),
                        entry.get("source_package"),
                    )
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    results.append(entry)

        return results

    def extract(self, meta: Dict[str, Any], text: str) -> Optional[PaperExtraction]:
        overall_start = time.perf_counter()
        paper_label = (
            meta.get("id") or meta.get("doi") or meta.get("title") or "<unknown>"
        )
        if not text.strip():
            logger.info(
                "Skipping paper %s: no text available after preprocessing.",
                paper_label,
            )
            _log_step_duration(
                "extract.total",
                overall_start,
                paper=paper_label,
                details="result=no_text",
            )
            return None
        try:
            ner_start = time.perf_counter()
            ner_results = self._ner_pipeline(text)
            _log_step_duration(
                "extract.ner_pipeline",
                ner_start,
                paper=paper_label,
                details=f"ner_results={len(ner_results)}",
            )
        except Exception:
            _log_step_duration(
                "extract.ner_pipeline",
                ner_start,
                paper=paper_label,
                details="error=true",
            )
            logger.exception("Skipping paper %s: NER pipeline failed.", paper_label)
            _log_step_duration(
                "extract.total",
                overall_start,
                paper=paper_label,
                details="result=ner_error",
            )
            return None

        entity_start = time.perf_counter()
        entities: Dict[str, NodeRecord] = {}
        link_scores = dict(self._latest_link_scores)
        for ent in ner_results:
            coref_info = ent.get("coref_antecedent") or None
            if (
                "start" in ent
                and "end" in ent
                and ent["start"] is not None
                and ent["end"] is not None
            ):
                raw_name = text[ent["start"] : ent["end"]]
            else:
                raw_name = ent.get("word", "")
            raw_name = clean_entity_surface(raw_name)
            if not raw_name:
                continue
            if raw_name.lower() in self._span_blocklist:
                continue
            tokens = ent.get("tokens") or []
            pos_tags = [tag.upper() for tag in ent.get("upos") or [] if tag]
            if not self._passes_entity_heuristics(raw_name, tokens, pos_tags):
                continue
            original_surface = raw_name
            lemma_from_ent = (ent.get("lemma") or "").strip()
            lemma_override: Optional[str] = None
            if coref_info:
                antecedent_text = clean_entity_surface(coref_info.get("text") or "")
                if (
                    antecedent_text
                    and antecedent_text.lower() not in self._span_blocklist
                    and self._passes_entity_heuristics(
                        antecedent_text, tokens, pos_tags
                    )
                ):
                    raw_name = antecedent_text
                    lemma_override = canonical_entity_display(antecedent_text)

            lemma_hint = lemma_override or lemma_from_ent
            lemma_source = lemma_hint if lemma_hint else raw_name
            canonical_key = canonical_entity_key(lemma_source)
            if not canonical_key or canonical_key in self._span_blocklist:
                continue
            canonical_display = (
                canonical_entity_display(lemma_hint or raw_name) or raw_name
            )
            ontology_score = ent.get("ontology_score")
            if ontology_score is None:
                ontology_score = link_scores.get(canonical_key)
            label = ent.get("entity_group", "")
            if label.upper() == "ENTITY":
                if self.config.ontology_min_score > 0.0 and (
                    ontology_score is None
                    or ontology_score < self.config.ontology_min_score
                ):
                    continue
            record = entities.get(canonical_key)
            if record is None:
                node_type = self.categorize_entity(label, ent.get("kb_id"))
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
            lemma_values: List[str] = []
            if lemma_hint:
                lemma_values.append(lemma_hint)
            if lemma_from_ent and lemma_from_ent not in lemma_values:
                lemma_values.append(lemma_from_ent)
            if lemma_values:
                existing_lemmas = record.normalizations.get("lemmas", [])
                record.normalizations["lemmas"] = dedupe_preserve_order(
                    list(existing_lemmas) + lemma_values
                )
            if pos_tags:
                existing_pos = record.normalizations.get("upos", [])
                record.normalizations["upos"] = dedupe_preserve_order(
                    list(existing_pos) + pos_tags
                )
            source_pkg = ent.get("source_package")
            if source_pkg:
                existing_sources = record.normalizations.get("ner_sources", [])
                record.normalizations["ner_sources"] = dedupe_preserve_order(
                    list(existing_sources) + [source_pkg]
                )
            if tokens:
                existing_tokens = record.normalizations.get("tokens", [])
                record.normalizations["tokens"] = dedupe_preserve_order(
                    list(existing_tokens) + tokens
                )
            kb_id = ent.get("kb_id")
            if kb_id:
                umls_ids = record.normalizations.get("umls_ids", [])
                record.normalizations["umls_ids"] = dedupe_preserve_order(
                    list(umls_ids) + [kb_id]
                )
            name_candidates = [raw_name, canonical_display]
            if (
                coref_info
                and original_surface
                and original_surface.lower() != raw_name.lower()
            ):
                name_candidates.append(original_surface)
            self._update_record_forms(record, name_candidates)
        _log_step_duration(
            "extract.entity_postprocess",
            entity_start,
            paper=paper_label,
            details=f"unique_entities={len(entities)}",
        )
        if not entities:
            logger.info("Skipping paper %s: no entities detected.", paper_label)
            _log_step_duration(
                "extract.total",
                overall_start,
                paper=paper_label,
                details="result=no_entities",
            )
            return None

        node_records = list(entities.values())
        node_records.sort(key=lambda record: record.canonical_name.lower())
        surface_lookup = {
            record.canonical_name: self._surface_form(record, text)
            for record in node_records
        }

        relation_start = time.perf_counter()
        relations: List[RelationRecord] = []
        relation_attempts = 0
        candidate_pairs = 0
        skipped_pairs = 0
        prefilter_skipped = 0
        for i in range(len(node_records)):
            for j in range(i + 1, len(node_records)):
                subj = node_records[i].canonical_name
                obj = node_records[j].canonical_name
                subj_type = node_records[i].node_type
                obj_type = node_records[j].node_type
                if not self._pair_has_potential_relation(subj_type, obj_type):
                    skipped_pairs += 1
                    logger.debug(
                        "Skipping relation pair %s (%s) ↔ %s (%s): no compatible predicates",
                        subj,
                        subj_type,
                        obj,
                        obj_type,
                    )
                    continue
                candidate_pairs += 1
                context = self._pick_sentence_context(
                    text, surface_lookup[subj], surface_lookup[obj]
                )
                if not self._passes_relation_prefilter(context, subj, obj):
                    prefilter_skipped += 1
                    continue
                relation_attempts += 1
                relation_inf, evidence_inf = self.classify_relation_via_nli(
                    context, subj, obj
                )
                candidates: List[Dict[str, Any]] = [
                    {
                        "subj": subj,
                        "obj": obj,
                        "subj_type": subj_type,
                        "obj_type": obj_type,
                        "relation": relation_inf,
                        "evidence": evidence_inf,
                    }
                ]

                needs_alternate_orientation = (
                    relation_inf is None
                    or not self._relation_allowed_for_types(
                        relation_inf.label, subj_type, obj_type
                    )
                )
                if needs_alternate_orientation:
                    swapped_relation, swapped_evidence = self.classify_relation_via_nli(
                        context, obj, subj
                    )
                    candidates.append(
                        {
                            "subj": obj,
                            "obj": subj,
                            "subj_type": obj_type,
                            "obj_type": subj_type,
                            "relation": swapped_relation,
                            "evidence": swapped_evidence,
                        }
                    )

                chosen: Optional[Dict[str, Any]] = None
                for candidate in candidates:
                    candidate_rel = candidate["relation"]
                    if candidate_rel and self._relation_allowed_for_types(
                        candidate_rel.label,
                        candidate["subj_type"],
                        candidate["obj_type"],
                    ):
                        chosen = candidate
                        break

                if chosen is None:
                    fallback_candidate: Optional[Dict[str, Any]] = next(
                        (
                            cand
                            for cand in candidates
                            if cand["evidence"] is not None
                            and self._fallback_relation_for_types(
                                cand["subj_type"], cand["obj_type"]
                            )
                        ),
                        None,
                    )
                    if fallback_candidate is None:
                        continue
                    fallback_predicate = self._fallback_relation_for_types(
                        fallback_candidate["subj_type"], fallback_candidate["obj_type"]
                    )
                    assert fallback_predicate is not None
                    evidence_for_fallback = fallback_candidate["evidence"]
                    assert evidence_for_fallback is not None
                    fallback_relation = RelationInference(
                        label=fallback_predicate,
                        entailment=evidence_for_fallback.entailment,
                        score=evidence_for_fallback.score,
                        margin=evidence_for_fallback.margin,
                        reverse_entailment=evidence_for_fallback.reverse_entailment,
                    )
                    chosen = dict(fallback_candidate)
                    chosen["relation"] = fallback_relation

                if chosen is None:
                    continue

                relation_chosen: RelationInference = chosen["relation"]
                evidence_chosen: Optional[RelationInference] = chosen["evidence"]
                subj = chosen["subj"]
                obj = chosen["obj"]
                subj_type = chosen["subj_type"]
                obj_type = chosen["obj_type"]

                predicate = relation_chosen.label
                confidence = min(
                    1.0, relation_chosen.entailment + 0.1 * relation_chosen.margin
                )
                directionality = (
                    "directed"
                    if predicate
                    in {"treats", "predicts", "biomarker_for", "measure_of"}
                    else "undirected"
                )
                claim_descriptor = self._build_claim_descriptor(predicate, subj, obj)
                qualifiers: Dict[str, Any] = {
                    "nli_raw_label": relation_chosen.label,
                    "nli_entailment": round(relation_chosen.entailment, 4),
                    "nli_margin": round(relation_chosen.margin, 4),
                    "nli_reverse_entailment": round(
                        relation_chosen.reverse_entailment, 4
                    ),
                    "nli_score": round(relation_chosen.score, 4),
                }
                if claim_descriptor:
                    qualifiers["claim"] = claim_descriptor.model_dump(exclude_none=True)
                if evidence_chosen is not None:
                    evidence_descriptor = self._build_claim_descriptor(
                        evidence_chosen.label, subj, obj
                    )
                    qualifiers.update(
                        {
                            "evidence_label": evidence_chosen.label,
                            "evidence_entailment": round(evidence_chosen.entailment, 4),
                            "evidence_margin": round(evidence_chosen.margin, 4),
                            "evidence_score": round(evidence_chosen.score, 4),
                            "evidence_reverse_entailment": round(
                                evidence_chosen.reverse_entailment, 4
                            ),
                        }
                    )
                    if evidence_descriptor:
                        qualifiers["evidence_claim"] = evidence_descriptor._model_dump(
                            exclude_none=True
                        )
                relations.append(
                    RelationRecord(
                        subject=subj,
                        predicate=predicate,
                        object=obj,
                        directionality=directionality,
                        evidence_span=context[:300],
                        confidence=confidence,
                        claim=claim_descriptor,
                        qualifiers=qualifiers,
                    )
                )
        _log_step_duration(
            "extract.relation_inference",
            relation_start,
            paper=paper_label,
            details=(
                f"attempts={relation_attempts}, relations={len(relations)}, "
                f"skipped_types={skipped_pairs}, prefilter_skipped={prefilter_skipped}, "
                f"candidates={candidate_pairs}"
            ),
        )
        result = PaperExtraction(
            paper_id=meta.get("id", ""),
            doi=meta.get("doi"),
            title=meta.get("title", ""),
            year=meta.get("year"),
            venue=meta.get("venue"),
            nodes=node_records,
            relations=relations,
        )
        _log_step_duration(
            "extract.total",
            overall_start,
            paper=paper_label,
            details=(f"nodes={len(node_records)}, relations={len(relations)}"),
        )
        return result

    def _surface_form(self, record: NodeRecord, text: str) -> str:
        forms = record.normalizations.get("surface_forms", [])
        if isinstance(forms, list):
            for form in forms:
                if form and form in text:
                    return form
        return record.canonical_name

    def _update_record_forms(
        self, record: NodeRecord, candidates: Iterable[str]
    ) -> None:
        existing_forms = record.normalizations.get("surface_forms")
        if isinstance(existing_forms, list):
            forms = list(existing_forms)
        else:
            forms = [record.canonical_name]
        forms.extend(candidates)
        forms.append(record.canonical_name)
        normalized_forms = dedupe_preserve_order(forms)
        record.normalizations["surface_forms"] = normalized_forms
        record.synonyms = dedupe_preserve_order(
            list(record.synonyms)
            + [form for form in normalized_forms if form != record.canonical_name]
        )

    def _build_claim_descriptor(
        self, predicate: str, subject: str, obj: str
    ) -> ClaimDescriptor:
        """Construct a claim descriptor that explains what an evidence edge supports."""

        predicate = predicate.lower()
        if predicate == "treats":
            return ClaimDescriptor(
                type="causal",
                statement=f"{subject} treats {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "predicts":
            return ClaimDescriptor(
                type="association",
                statement=f"{subject} predicts {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "biomarker_for":
            return ClaimDescriptor(
                type="mechanistic",
                statement=f"{subject} is a biomarker for {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "measure_of":
            return ClaimDescriptor(
                type="measurement",
                statement=f"{subject} is a measure of {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "supports":
            return ClaimDescriptor(
                type="association",
                statement=f"{subject} is positively associated with {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "contradicts":
            return ClaimDescriptor(
                type="association",
                statement=f"{subject} is negatively associated with {obj}",
                direction="negative",
                evidence_type="nli_text_entailment",
            )
        if predicate == "replicates":
            return ClaimDescriptor(
                type="measurement",
                statement=f"{subject} replicates findings reported for {obj}",
                direction="positive",
                evidence_type="nli_text_entailment",
            )
        if predicate == "null_reported":
            return ClaimDescriptor(
                type="association",
                statement=f"{subject} reports null findings with respect to {obj}",
                direction="null",
                evidence_type="nli_text_entailment",
            )

        return ClaimDescriptor(
            statement=f"{subject} relates to {obj}",
            type="unknown",
            direction="unknown",
            evidence_type="nli_text_entailment",
        )


__all__ = [
    "EntityRelationExtractor",
    "ExtractionConfig",
]
