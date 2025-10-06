"""Utilities for downloading and normalising open access full texts."""

from __future__ import annotations

import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
from readability import Document

PMC_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _quiet_pdfminer_logs() -> None:
    for name in ["pdfminer", "pdfminer.pdfcolor", "pdfminer.pdfinterp"]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def try_pmc_fulltext(pmcid: str) -> Optional[str]:
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    response = httpx.get(PMC_EUTILS, params=params, timeout=60)
    if response.status_code != 200 or not response.text:
        return None
    soup = BeautifulSoup(response.text, "lxml-xml")
    parts: List[str] = []
    for tag in soup.find_all(["abstract", "p", "sec", "title"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
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
    serialized = json.dumps(rec, ensure_ascii=False)
    pmcid_match = re.search(r"PMCID[:\s]*PMC(\d+)", serialized, flags=re.I)
    if pmcid_match:
        text = try_pmc_fulltext("PMC" + pmcid_match.group(1))
        if text and len(text) > 1000:
            return text

    location = rec.get("best_oa_location") or {}
    pdf_url = location.get("url_for_pdf")
    html_url = location.get("url")

    if pdf_url:
        try:
            response = httpx.get(pdf_url, timeout=90, follow_redirects=True)
            if response.status_code == 200 and (
                "pdf" in response.headers.get("content-type", "").lower()
                or pdf_url.lower().endswith(".pdf")
            ):
                text = extract_text_from_pdf_bytes(response.content)
                if text and len(text) > 1000:
                    return text
        except Exception:
            pass

    if html_url:
        try:
            response = httpx.get(html_url, timeout=90, follow_redirects=True)
            if response.status_code == 200 and (
                "html" in response.headers.get("content-type", "").lower()
            ):
                text = extract_text_from_html_bytes(response.content, html_url)
                if text and len(text) > 1000:
                    return text
        except Exception:
            pass

    primary = rec.get("primary_location") or {}
    for key in ("pdf_url", "landing_page_url", "source_url", "url"):
        url = primary.get(key)
        if not url:
            continue
        try:
            response = httpx.get(url, timeout=90, follow_redirects=True)
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(response.content)
                if text and len(text) > 1000:
                    return text
            if "html" in content_type:
                text = extract_text_from_html_bytes(response.content, url)
                if text and len(text) > 1000:
                    return text
        except Exception:
            continue
    return None


__all__ = [
    "extract_text_from_html_bytes",
    "extract_text_from_pdf_bytes",
    "resolve_text_and_download",
    "try_pmc_fulltext",
]
