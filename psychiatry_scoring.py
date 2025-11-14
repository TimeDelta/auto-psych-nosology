"""Hybrid psychiatric relevance scoring utilities."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import polars as pl

_DEFAULT_MONDO_IDS = {
    "MONDO:0002025",  # mental disorder
    "MONDO:0016060",  # neurodevelopmental disorder
    "MONDO:0004995",  # mood disorder
    "MONDO:0004975",  # depressive disorder
    "MONDO:0005130",  # major depressive disorder
    "MONDO:0004652",  # anxiety disorder
    "MONDO:0020121",  # obsessive-compulsive disorder
    "MONDO:0002230",  # bipolar disorder
    "MONDO:0005249",  # bipolar disorder type 1/2
    "MONDO:0015159",  # schizophrenia
    "MONDO:0002049",  # attention deficit hyperactivity disorder
    "MONDO:0008605",  # autism spectrum disorder
}

_DEFAULT_GROUP_LABELS = {
    "major depressive disorder",
    "depressive disorder",
    "bipolar disorder",
    "schizophrenia",
    "anxiety disorder",
    "autism spectrum disorder",
    "adhd",
    "posttraumatic stress disorder",
    "obsessive compulsive disorder",
    "mood disorder",
}

_DEFAULT_DRUG_CATEGORY_KEYWORDS = {
    "psychiatric",
    "antidepressant",
    "antipsychotic",
    "anxiolytic",
    "mood stabilizer",
    "psycholeptic",
    "psychotropic",
    "psychostimulant",
}

_DEFAULT_TEXT_PROTOTYPES = (
    "Mental disorders affecting mood, anxiety, cognition, or behaviour",
    "Depressive episodes with symptoms like persistent sadness and anhedonia",
    "Psychotic disorders characterized by delusions or hallucinations",
    "Neurodevelopmental disorders impacting attention, learning, or social communication",
    "Post-traumatic stress responses with intrusive memories and hyperarousal",
)

_DEFAULT_WEIGHTS = {
    "ontology": 0.4,
    "group": 0.25,
    "drug": 0.2,
    "text": 0.1,
    "name": 0.25,
}

_PSYCHIATRIC_NAME_KEYWORDS = (
    "depress",
    "bipolar",
    "schizo",
    "psychosis",
    "psychotic",
    "anxiety",
    "ocd",
    "obsessive",
    "compulsive",
    "adhd",
    "autism",
    "suicid",
    "panic",
    "trauma",
    "ptsd",
    "mania",
    "eating disorder",
    "anorexia",
    "bulimia",
    "personality disorder",
)


@dataclass
class PsychiatricScoringConfig:
    mondo_ids: Iterable[str]
    group_labels: Iterable[str]
    drug_category_keywords: Iterable[str]
    text_prototypes: Sequence[str]
    weights: Mapping[str, float]
    threshold: float


def _normalise_token(token: str) -> str:
    return token.lower()


def _tokenize_text(text: str) -> List[str]:
    cleaned = text.translate({ord(ch): " " for ch in "\t\n\r.,;:!?()[]{}<>\"'`/\\-"})
    return [_normalise_token(part) for part in cleaned.split() if part]


def _build_counter(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not text:
        return counter
    for token in _tokenize_text(text):
        counter[token] += 1
    return counter


def _cosine_similarity(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    if not counter_a or not counter_b:
        return 0.0
    common = set(counter_a.keys()) & set(counter_b.keys())
    if not common:
        return 0.0
    dot = sum(counter_a[token] * counter_b[token] for token in common)
    norm_a = sum(value * value for value in counter_a.values()) ** 0.5
    norm_b = sum(value * value for value in counter_b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class PsychiatricRelevanceScorer:
    """Combine ontology, metadata, and neighbourhood signals into a score."""

    def __init__(self, *, config: PsychiatricScoringConfig) -> None:
        self.config = config
        self._prototype_embeddings = [
            _build_counter(text) for text in self.config.text_prototypes
        ]
        self._psy_mondo_ids = {mid.strip() for mid in self.config.mondo_ids if mid}
        self._group_labels = {
            label.strip().lower() for label in self.config.group_labels if label
        }
        self._drug_keywords = {
            keyword.strip().lower()
            for keyword in self.config.drug_category_keywords
            if keyword
        }
        self._name_keywords = tuple(
            keyword.strip().lower() for keyword in _PSYCHIATRIC_NAME_KEYWORDS if keyword
        )

    def score_diseases(
        self,
        diseases: pl.DataFrame,
        edges: pl.DataFrame,
        drug_features: pl.DataFrame,
    ) -> pl.DataFrame:
        drug_category_lookup = self._build_drug_keyword_lookup(drug_features)
        drug_incident = self._map_diseases_to_psychiatric_drugs(
            edges, drug_category_lookup
        )

        scores: List[float] = []
        evidence_records: List[str] = []
        flags: Dict[str, List[str]] = {}

        for row in diseases.iter_rows(named=True):
            node_index = row["node_index"]
            evidence: List[str] = []
            ontology_flag = bool(row.get("mondo_id") in self._psy_mondo_ids)
            if ontology_flag:
                evidence.append("ontology")

            group_label = (row.get("group_name_bert") or "").lower()
            group_flag = bool(group_label and group_label in self._group_labels)
            if group_flag:
                evidence.append("group_label")

            drug_flag = node_index in drug_incident
            if drug_flag:
                evidence.append("therapeutic_neighbors")

            text_payload = " ".join(
                str(row.get(field, ""))
                for field in (
                    "mondo_definition",
                    "umls_description",
                    "orphanet_definition",
                    "orphanet_clinical_description",
                    "mayo_symptoms",
                    "mayo_causes",
                    "mayo_risk_factors",
                    "mayo_complications",
                    "name",
                    "mondo_name",
                )
            )
            text_embedding = _build_counter(text_payload)
            text_score = max(
                (
                    _cosine_similarity(text_embedding, proto)
                    for proto in self._prototype_embeddings
                ),
                default=0.0,
            )
            if text_score > 0.15:
                evidence.append(f"text:{text_score:.2f}")

            name_value = str(row.get("name") or "")
            keyword_flag = self._name_contains_keyword(name_value)
            if not keyword_flag:
                mondo_name = str(row.get("mondo_name") or "")
                keyword_flag = self._name_contains_keyword(mondo_name)
            if keyword_flag:
                evidence.append("name_keywords")
                text_score = max(text_score, 0.8)

            score = (
                self.config.weights.get("ontology", 0.0) * float(ontology_flag)
                + self.config.weights.get("group", 0.0) * float(group_flag)
                + self.config.weights.get("drug", 0.0) * float(drug_flag)
                + self.config.weights.get("text", 0.0) * float(text_score)
                + self.config.weights.get("name", 0.0) * float(keyword_flag)
            )
            scores.append(score)
            flags.setdefault("ontology", []).append(ontology_flag)
            flags.setdefault("group", []).append(group_flag)
            flags.setdefault("drug", []).append(drug_flag)
            flags.setdefault("text_score", []).append(text_score)
            flags.setdefault("name_keyword", []).append(keyword_flag)
            evidence_records.append(json.dumps(evidence, ensure_ascii=False))

        result = diseases.select("node_index").with_columns(
            pl.Series("psy_score", scores),
            pl.Series("psy_evidence", evidence_records),
            pl.Series("ontology_flag", flags["ontology"]),
            pl.Series("group_flag", flags["group"]),
            pl.Series("drug_flag", flags["drug"]),
            pl.Series("text_score", flags["text_score"]),
            pl.Series("name_keyword_flag", flags["name_keyword"]),
        )

        def _final_decision(row: Mapping[str, object]) -> bool:
            score = float(row["psy_score"])
            bool_flags = sum(
                int(bool(row[column]))
                for column in (
                    "ontology_flag",
                    "group_flag",
                    "drug_flag",
                    "name_keyword_flag",
                )
            )
            text_hit = float(row["text_score"]) > 0.2
            if score >= self.config.threshold:
                return True
            if bool_flags >= 2:
                return True
            if bool_flags >= 1 and text_hit:
                return True
            return False

        decisions = [_final_decision(row) for row in result.iter_rows(named=True)]
        result = result.with_columns(pl.Series("is_psychiatric", decisions))

        if logger.isEnabledFor(logging.DEBUG):
            total = len(decisions)
            positive = sum(1 for d in decisions if d)
            flag_counts = {
                "ontology": sum(flags["ontology"]),
                "group": sum(flags["group"]),
                "drug": sum(flags["drug"]),
                "name": sum(flags["name_keyword"]),
            }
            text_hits = sum(1 for score in flags["text_score"] if score > 0.2)
            logger.debug(
                "Psych scoring summary: total=%d positive=%d ontology=%d group=%d "
                "drug=%d name=%d text_hits=%d threshold=%.3f",
                total,
                positive,
                flag_counts["ontology"],
                flag_counts["group"],
                flag_counts["drug"],
                flag_counts["name"],
                text_hits,
                self.config.threshold,
            )
        return result

    def _build_drug_keyword_lookup(
        self, drug_features: pl.DataFrame
    ) -> Dict[int, bool]:
        if drug_features.is_empty():
            return {}
        keywords = self._drug_keywords
        if not keywords:
            return {}
        relevant: Dict[int, bool] = {}
        for row in drug_features.iter_rows(named=True):
            node_index = row["node_index"]
            text_fields = [
                row.get(field)
                for field in (
                    "category",
                    "group",
                    "pathway",
                    "description",
                    "indication",
                    "mechanism_of_action",
                )
            ]
            tokens = {
                token
                for field in text_fields
                if isinstance(field, str)
                for token in _tokenize_text(field)
            }
            relevant[node_index] = any(token in keywords for token in tokens)
        return relevant

    def _name_contains_keyword(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        for keyword in self._name_keywords:
            if keyword in lowered:
                return True
        return False

    def _map_diseases_to_psychiatric_drugs(
        self,
        edges: pl.DataFrame,
        drug_lookup: Mapping[int, bool],
    ) -> set[int]:
        if edges.is_empty() or not drug_lookup:
            return set()
        diseases: set[int] = set()
        for row in edges.iter_rows(named=True):
            src = int(row["source_index"])
            dst = int(row["target_index"])
            src_type = (row.get("source_type") or "").lower()
            dst_type = (row.get("target_type") or "").lower()
            relation = (row.get("relation") or "").lower()

            if src in drug_lookup and drug_lookup[src] and "disease" in dst_type:
                diseases.add(dst)
            if dst in drug_lookup and drug_lookup[dst] and "disease" in src_type:
                diseases.add(src)
            if (
                "drug" in src_type
                and "disease" in dst_type
                and relation in {"drug_disease", "treats"}
            ):
                if drug_lookup.get(src, False):
                    diseases.add(dst)
            if (
                "drug" in dst_type
                and "disease" in src_type
                and relation in {"drug_disease", "treats"}
            ):
                if drug_lookup.get(dst, False):
                    diseases.add(src)
        return diseases


def build_default_scoring_config(threshold: float) -> PsychiatricScoringConfig:
    return PsychiatricScoringConfig(
        mondo_ids=_DEFAULT_MONDO_IDS,
        group_labels=_DEFAULT_GROUP_LABELS,
        drug_category_keywords=_DEFAULT_DRUG_CATEGORY_KEYWORDS,
        text_prototypes=_DEFAULT_TEXT_PROTOTYPES,
        weights=_DEFAULT_WEIGHTS,
        threshold=threshold,
    )


logger = logging.getLogger(__name__)
