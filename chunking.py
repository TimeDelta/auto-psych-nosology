# chunking.py
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tiktoken


@dataclass
class TextChunk:
    section: str
    text: str
    start_char: int
    end_char: int
    est_tokens: int


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


def estimate_tokens(s: str, model: str = "gpt-4.1-mini") -> int:
    enc = (
        tiktoken.encoding_for_model(model)
        if hasattr(tiktoken, "encoding_for_model")
        else tiktoken.get_encoding("cl100k_base")
    )
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
    model: str,
    min_chunk_tokens: int = 300,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Sentence-bounded packing up to max_tokens, with sentence overlap.
    """
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
    model: str = "gpt-4.1-mini",
    hard_token_limit: int = 7500,
    target_chunk_tokens: int = 2800,
) -> List[TextChunk]:
    """
    Section-aware, sentence-bounded, token-aware chunking.
    Skips references/acknowledgments by default.
    """
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
        tgt = min(tgt, hard_token_limit - 1000)

        for sub in pack_sentences(
            sect_text, max_tokens=tgt, model=model, min_chunk_tokens=int(0.4 * tgt)
        ):
            est = estimate_tokens(sub, model)
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
                est_tokens=estimate_tokens(doc[:15000], model),
            )
        ]
    return chunks
