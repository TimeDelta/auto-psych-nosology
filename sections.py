"""Heuristics for extracting relevant sections from biomedical manuscripts."""

from __future__ import annotations

import re
from typing import List, Optional

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


__all__ = ["extract_results_and_discussion"]
