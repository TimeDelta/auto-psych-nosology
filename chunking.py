# chunking.py
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken


@dataclass
class TextChunk:
    section: str
    text: str
    start_char: int
    end_char: int
    est_tokens: int


# Compact default that delivers strong instruction-following quality while
# staying runnable on 16 GB machines.
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

FREE_MODEL_ENCODINGS: Dict[str, str] = {
    DEFAULT_MODEL: "cl100k_base",
    "google/gemma-2-2b-it": "cl100k_base",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "cl100k_base",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "cl100k_base",
    "mistralai/Mistral-7B-Instruct-v0.2": "cl100k_base",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "cl100k_base",
    "Qwen/Qwen2.5-7B-Instruct": "cl100k_base",
    "Qwen/Qwen2.5-72B-Instruct": "cl100k_base",
}

MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "microsoft/Phi-3-mini-4k-instruct": 4096,
    "google/gemma-2-2b-it": 8192,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 8192,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 8192,
    "mistralai/Mistral-7B-Instruct-v0.2": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "Qwen/Qwen2.5-7B-Instruct": 32768,
    "Qwen/Qwen2.5-72B-Instruct": 32768,
}

MODEL_ALIASES: Dict[str, str] = {k.lower(): k for k in FREE_MODEL_ENCODINGS}
MODEL_ALIASES.update(
    {
        "phi-3-mini-4k-instruct": DEFAULT_MODEL,
        "phi-3-mini": DEFAULT_MODEL,
        "microsoft-phi-3-mini": DEFAULT_MODEL,
        "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    }
)


IMRAD_HEADINGS = [
    r"^\s*abstract\b",
    r"^\s*introduction\b",
    r"^\s*background\b",
    r"^\s*methods?\b",
    r"^\s*materials and methods\b",
    r"^\s*results?\b",
    r"^\s*discussion\b",
    r"^\s*conclusion(s)?\b",
    r"^\s*limitations?\b",
    r"^\s*references?\b",
    r"^\s*acknowledg(e)?ments?\b",
    r"^\s*funding\b",
]

SECTION_RE = re.compile("|".join(IMRAD_HEADINGS), flags=re.I | re.M)

SENTENCE_RE = re.compile(r"(?<=\S[.!?])\s+(?=[A-Z(])")


def resolve_model_name(model: Optional[str]) -> str:
    """Return the canonical free model name, enforcing the allowlist."""

    if not model:
        return DEFAULT_MODEL
    key = model.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    supported = ", ".join(sorted(FREE_MODEL_ENCODINGS))
    raise ValueError(f"Unsupported model '{model}'. Choose from: {supported}")


@lru_cache(maxsize=None)
def _load_encoder(canonical_model: str):
    """Load and cache the tokenizer for one of the supported free models."""

    encoding_name = FREE_MODEL_ENCODINGS[canonical_model]
    if hasattr(tiktoken, "encoding_for_model"):
        try:
            return tiktoken.encoding_for_model(canonical_model)
        except (KeyError, ValueError):
            pass
    return tiktoken.get_encoding(encoding_name)


def get_tokenizer(model: Optional[str] = None):
    canonical = resolve_model_name(model or DEFAULT_MODEL)
    return _load_encoder(canonical)


def estimate_tokens(s: str, model: str = DEFAULT_MODEL) -> int:
    enc = get_tokenizer(model)
    return len(enc.encode(s))


def split_into_sections(full_text: str) -> List[Tuple[str, int, int]]:
    """
    Returns list of (section_name, start_idx, end_idx) covering the doc.
    If no headings found, returns a single 'body' section.
    """
    matches = [
        (m.group(0).strip().lower(), m.start()) for m in SECTION_RE.finditer(full_text)
    ]
    if not matches:
        return [("body", 0, len(full_text))]
    spans = []
    for i, (heading, start) in enumerate(matches):
        end = matches[i + 1][1] if i + 1 < len(matches) else len(full_text)
        name = re.sub(r"^\s*", "", heading).strip(": ").lower()
        spans.append((name, start, end))
    return spans


def pack_sentences(
    text: str,
    max_tokens: int,
    model: str = DEFAULT_MODEL,
    min_chunk_tokens: int = 300,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Sentence-bounded packing up to max_tokens, with sentence overlap.
    """
    model = resolve_model_name(model)
    sents = SENTENCE_RE.split(text)
    chunks = []
    i = 0
    while i < len(sents):
        cur = []
        j = i
        while j < len(sents):
            candidate = (" ".join(cur + [sents[j]])).strip()
            if estimate_tokens(candidate, model) > max_tokens and cur:
                break
            cur.append(sents[j])
            j += 1
        chunk = " ".join(cur).strip()
        if estimate_tokens(chunk, model) >= min_chunk_tokens or (j - i) > 1:
            chunks.append(chunk)
        # adaptive sentence overlap: longer sentences → keep 2, else 1
        avg_len = sum(len(s) for s in cur) / max(1, len(cur))
        back = 2 if avg_len > 220 else overlap_sentences
        i = max(j - back, j) if j > i else j + 1
    return chunks


def chunk(
    full_text: str,
    model: str = DEFAULT_MODEL,
    hard_token_limit: int = 7500,
    target_chunk_tokens: int = 2800,
) -> List[TextChunk]:
    """
    Section-aware, sentence-bounded, token-aware chunking.
    Skips references/acknowledgments by default.
    """
    canonical_model = resolve_model_name(model)
    model_limit = MODEL_CONTEXT_LIMITS.get(canonical_model, hard_token_limit)
    hard_limit = min(hard_token_limit, model_limit)
    buffer_tokens = min(1000, max(200, hard_limit // 5))
    if hard_limit <= buffer_tokens:
        buffer_tokens = max(50, hard_limit // 10)
    max_chunk_by_context = min(
        max(200, hard_limit - buffer_tokens), max(50, hard_limit - 50)
    )

    # normalize whitespace
    doc = re.sub(r"[ \t]+", " ", re.sub(r"\s+\n", "\n", full_text)).strip()

    # split into sections
    sections = split_into_sections(doc)

    chunks: List[TextChunk] = []
    for name, start, end in sections:
        sect_text = doc[start:end].strip()
        if not sect_text:
            continue

        # filter boilerplate sections
        if re.match(r"^(references?|acknowledg(e)?ments?|funding)\b", name, flags=re.I):
            continue

        # tighter packing for Methods/Results; looser for Introduction
        if re.match(r"^(methods?|materials and methods|results?)\b", name, flags=re.I):
            tgt = target_chunk_tokens
        else:
            tgt = int(target_chunk_tokens * 0.8)

        # never exceed model context
        tgt = min(tgt, max_chunk_by_context)

        for sub in pack_sentences(
            sect_text,
            max_tokens=tgt,
            model=canonical_model,
            min_chunk_tokens=int(0.4 * tgt),
        ):
            est = estimate_tokens(sub, canonical_model)
            # skip ultra-short noise
            if est < 120:
                continue
            s_idx = doc.find(sub, start, end)
            e_idx = s_idx + len(sub) if s_idx != -1 else end
            chunks.append(
                TextChunk(
                    section=name,
                    text=sub,
                    start_char=s_idx,
                    end_char=e_idx,
                    est_tokens=est,
                )
            )

    # if nothing parsed, do one crude chunk
    if not chunks and doc:
        chunks = [
            TextChunk(
                section="body",
                text=doc[: min(len(doc), 15000)],
                start_char=0,
                end_char=min(len(doc), 15000),
                est_tokens=estimate_tokens(doc[:15000], canonical_model),
            )
        ]
    return chunks
