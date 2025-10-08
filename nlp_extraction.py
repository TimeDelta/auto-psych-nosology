"""Entity and relation extraction using Stanza NER and a biomedical NLI model."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import stanza
import torch
from nltk.tokenize import sent_tokenize
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
                "anatem",
                "jnlpba",
                "linnaeus",
                "ncbi_disease",
                "i2b2",
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
    ner_exclude_terms: Set[str] = field(
        default_factory=lambda: {
            term.strip().lower()
            for term in os.getenv(
                "NER_EXCLUDE_TERMS",
                "patient,patients,control,controls,participant,participants,subject,subjects,human,humans,donor,donors",
            ).split(",")
            if term.strip()
        }
    )
    stanza_lang: str = field(default_factory=lambda: os.getenv("STANZA_LANG", "en"))
    stanza_processors: str = field(
        default_factory=lambda: os.getenv(
            "STANZA_PROCESSORS", "tokenize,mwt,pos,lemma,ner"
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
        processors = self.config.stanza_processors
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

    def categorize_entity(self, label: str) -> Optional[str]:
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
            #
            "SPECIES": "Species",
        }
        if lbl in direct_map:
            return direct_map[lbl]
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
        pipelines = self._ensure_stanza_pipelines()
        if not pipelines:
            return []
        results: List[Dict[str, Any]] = []
        seen_spans: Set[Tuple[int, int, str]] = set()
        for pkg_display, pipeline in pipelines:
            doc = pipeline(text)
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

    def extract(self, meta: Dict[str, Any], text: str) -> Optional[PaperExtraction]:
        paper_label = (
            meta.get("id") or meta.get("doi") or meta.get("title") or "<unknown>"
        )
        if not text.strip():
            logger.info(
                "Skipping paper %s: no text available after preprocessing.",
                paper_label,
            )
            return None
        try:
            ner_results = self._ner_pipeline(text)
        except Exception:
            logger.exception("Skipping paper %s: NER pipeline failed.", paper_label)
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
            raw_name = clean_entity_surface(raw_name)
            if not raw_name:
                continue
            if raw_name.lower() in self.config.ner_exclude_terms:
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
                        continue
            lemma_hint = (ent.get("lemma") or "").strip()
            lemma_source = lemma_hint if lemma_hint else raw_name
            canonical_key = canonical_entity_key(lemma_source)
            if not canonical_key:
                continue
            canonical_display = (
                canonical_entity_display(lemma_hint or raw_name) or raw_name
            )
            record = entities.get(canonical_key)
            if record is None:
                label = ent.get("entity_group", "")
                node_type = self.categorize_entity(label)
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
                record.normalizations["lemmas"] = dedupe_preserve_order(
                    list(existing_lemmas) + [lemma_hint]
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
            self._update_record_forms(record, [raw_name, canonical_display])
        if not entities:
            logger.info("Skipping paper %s: no entities detected.", paper_label)
            return None

        node_records = list(entities.values())
        node_records.sort(key=lambda record: record.canonical_name.lower())
        surface_lookup = {
            record.canonical_name: self._surface_form(record, text)
            for record in node_records
        }

        relations: List[RelationRecord] = []
        for i in range(len(node_records)):
            for j in range(i + 1, len(node_records)):
                subj = node_records[i].canonical_name
                obj = node_records[j].canonical_name
                subj_type = node_records[i].node_type
                obj_type = node_records[j].node_type
                context = self._pick_sentence_context(
                    text, surface_lookup[subj], surface_lookup[obj]
                )
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
                        qualifiers["evidence_claim"] = evidence_descriptor.model_dump(
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
        return PaperExtraction(
            paper_id=meta.get("id", ""),
            doi=meta.get("doi"),
            title=meta.get("title", ""),
            year=meta.get("year"),
            venue=meta.get("venue"),
            nodes=node_records,
            relations=relations,
        )

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
