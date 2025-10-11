"""Utilities for normalising entity surface forms and synonyms."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

import nltk
from nltk.stem import WordNetLemmatizer

for resource in ["tokenizers/punkt", "punkt_tab", "corpora/wordnet", "corpora/omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:  # pragma: no cover
        nltk.download(resource.split("/")[-1])

_LEMMATIZER = WordNetLemmatizer()

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


def clean_entity_surface(name: str) -> str:
    cleaned = re.sub(r"\s+|[^a-zA-Z_0-9]", " ", (name or "").strip())
    cleaned = re.sub(r"'s", "", (cleaned or ""))
    return re.sub(r" - ", "-", (cleaned or ""))


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
    return _WORD_RE.sub(lambda match: _lemmatize_word(match.group(0)), text)


def canonical_entity_key(name: str) -> str:
    cleaned = clean_entity_surface(name)
    if not cleaned:
        return ""
    lemma = _lemmatize_text(cleaned.lower())
    return re.sub(r"\s+", " ", lemma).strip()


def canonical_entity_display(name: str) -> str:
    cleaned = clean_entity_surface(name)
    if not cleaned:
        return ""
    lemma = _lemmatize_text(cleaned)
    lemma = re.sub(r"\s+", " ", lemma).strip()
    if not lemma:
        return ""
    if lemma.islower() and re.search(r"[a-z]", lemma):
        lemma = lemma[0].upper() + lemma[1:]
    return lemma


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_name(value: str) -> str:
    return clean_entity_surface(value)


__all__ = [
    "canonical_entity_display",
    "canonical_entity_key",
    "clean_entity_surface",
    "dedupe_preserve_order",
    "normalize_name",
]
