"""Minimal OpenAlex client helpers with rate limiting and pagination."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import httpx

OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_FILTER = (
    "from_publication_date:2015-10-03,"
    "open_access.is_oa:true,"
    "language:en,"
    "concepts.id:C61535369",
)

_CLIENT = httpx.Client(
    base_url=OPENALEX_BASE_URL,
    headers={
        "User-Agent": "auto-psych-nosology/0.1",
        "Accept": "application/json",
    },
    timeout=60.0,
)
_OPENALEX_LOCK = threading.Lock()
_LAST_OPENALEX_REQUEST = 0.0
_OPENALEX_MIN_INTERVAL = 0.55
_OPENALEX_MAX_RETRIES = 6


def _reserve_openalex_slot() -> None:
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
    attempt = 0
    while True:
        attempt += 1
        _reserve_openalex_slot()
        try:
            response = _CLIENT.request(method, path, **kwargs)
        except httpx.RequestError:
            if attempt >= _OPENALEX_MAX_RETRIES:
                raise
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


def uninvert_abstract(index: Dict[str, List[int]]) -> str:
    if not index:
        return ""
    positions: Dict[int, str] = {}
    for token, occurrences in index.items():
        for pos in occurrences:
            positions[pos] = token
    return " ".join(positions[i] for i in range(min(positions), max(positions) + 1))


def _normalize_queries(query: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(query, str):
        items = [q.strip() for q in query.split(";")]
    else:
        items = [str(q).strip() for q in query]
    return [q for q in items if q]


def _distribute_evenly(total: int, buckets: int) -> List[int]:
    if buckets <= 0:
        return []
    base = total // buckets
    remainder = total % buckets
    return [base + 1 if i < remainder else base for i in range(buckets)]


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
    response = _openalex_get("/works", params=params)
    response.raise_for_status()
    return response.json()


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
        for work in results:
            abstract = uninvert_abstract(work.get("abstract_inverted_index") or {})
            out.append(
                {
                    "id": work.get("id"),
                    "doi": work.get("doi"),
                    "title": work.get("title"),
                    "year": work.get("publication_year"),
                    "venue": (work.get("host_venue") or {}).get("display_name"),
                    "abstract": abstract,
                    "best_oa_location": (work.get("best_oa_location") or {}),
                    "open_access": (work.get("open_access") or {}),
                    "primary_location": (work.get("primary_location") or {}),
                    "cited_by_count": work.get("cited_by_count"),
                    "publication_date": work.get("publication_date"),
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
    queries = _normalize_queries(query)
    if not queries:
        return []
    results: List[Dict[str, Any]] = []
    for term in queries:
        results.extend(_fetch_top_n_single(term, filters, sort, n))
    return results


def fetch_candidate_records(
    query: Union[str, Sequence[str]],
    filters: str,
    top_n: int,
    recent_n: int,
    fetch_buffer: int = 5,
) -> List[Dict[str, Any]]:
    fetch_buffer = max(0, fetch_buffer)
    queries = _normalize_queries(query)
    terms_multiplier = max(len(queries), 1)
    target_total = (top_n + recent_n) * terms_multiplier

    def _bucket(sort: str, count: int) -> List[Dict[str, Any]]:
        if not queries:
            return fetch_top_n(query, filters, sort=sort, n=count + fetch_buffer)
        extras = _distribute_evenly(fetch_buffer, len(queries))
        results: List[Dict[str, Any]] = []
        for idx, term in enumerate(queries):
            per_term_n = count + extras[idx]
            results.extend(_fetch_top_n_single(term, filters, sort, per_term_n))
        return results

    top_results = _bucket("cited_by_count:desc", top_n)
    recent_results = _bucket("publication_date:desc", recent_n)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in top_results + recent_results:
        key = record.get("search_query", "")
        grouped.setdefault(key, []).append(record)

    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    def add_candidates(candidates: Iterable[Dict[str, Any]], limit: int) -> None:
        for rec in candidates:
            rec_id = rec.get("id")
            if not rec_id or rec_id in seen_ids:
                continue
            records.append(rec)
            seen_ids.add(rec_id)
            if len(records) >= limit:
                return

    for term in queries:
        term_results = grouped.get(term, [])
        add_candidates(term_results[:top_n], target_total)
        add_candidates(term_results[top_n:], target_total)

    extras = [
        rec for rec in top_results + recent_results if rec.get("id") not in seen_ids
    ]
    add_candidates(extras, target_total)
    return records


__all__ = [
    "DEFAULT_FILTER",
    "fetch_candidate_records",
    "fetch_openalex_page",
    "fetch_top_n",
    "uninvert_abstract",
]
