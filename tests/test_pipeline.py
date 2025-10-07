import pathlib
import sys
from typing import Dict, Iterable, List

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from create_graph import KnowledgeGraphPipeline, PipelineConfig
from models import NodeRecord, PaperExtraction

TOP_TARGET = 5
RECENT_TARGET = 10


class DummyExtractor:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def extract(self, meta: Dict[str, str], _text: str) -> PaperExtraction:
        paper_id = meta.get("id") or meta.get("title") or "paper"
        self.calls.append(paper_id)
        node = NodeRecord(
            canonical_name=f"Symptom {paper_id}",
            lemma=f"symptom_{paper_id}",
            node_type="Symptom",
            synonyms=[],
            normalizations={},
        )
        return PaperExtraction(
            paper_id=paper_id,
            doi=meta.get("doi"),
            title=meta.get("title") or paper_id,
            year=meta.get("year"),
            venue=meta.get("venue"),
            nodes=[node],
            relations=[],
        )


def _candidate_record(
    identifier: str,
    term: str,
    bucket: str,
    rank: int,
    base_target: int,
) -> Dict[str, object]:
    return {
        "id": identifier,
        "doi": f"10.1234/{identifier}",
        "title": identifier.replace("_", " "),
        "year": 2024,
        "venue": "Test Venue",
        "abstract": "",
        "_candidate_sources": [
            {
                "term": term,
                "bucket": bucket,
                "is_buffer": rank >= base_target,
                "rank": rank,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _patch_resolve(monkeypatch):
    monkeypatch.setattr(
        "create_graph.resolve_text_and_download", lambda _record: "text"
    )


@pytest.fixture
def _capture_tables(monkeypatch):
    captured: Dict[str, object] = {}

    def _stub_save(nodes, relations, papers, _graph, _prefix, projected=None):
        captured["nodes"] = nodes
        captured["relations"] = relations
        captured["papers"] = papers
        captured["projected"] = projected

    monkeypatch.setattr("create_graph.save_tables", _stub_save)
    return captured


def _make_full_candidate_set(term: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for rank in range(TOP_TARGET):
        identifier = f"{term}_top_{rank}"
        records.append(_candidate_record(identifier, term, "top", rank, TOP_TARGET))
    for rank in range(RECENT_TARGET):
        identifier = f"{term}_recent_{rank}"
        records.append(
            _candidate_record(identifier, term, "recent", rank, RECENT_TARGET)
        )
    return records


def _flatten(
    records_iterables: Iterable[List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for chunk in records_iterables:
        out.extend(chunk)
    return out


def test_pipeline_processes_expected_papers(
    tmp_path: pathlib.Path, monkeypatch, _capture_tables
):
    candidates = _flatten(
        [_make_full_candidate_set("alpha"), _make_full_candidate_set("beta")]
    )

    def _stub_fetch(query, filters, top_n, recent_n, fetch_buffer):
        assert top_n == TOP_TARGET
        assert recent_n == RECENT_TARGET
        return candidates

    monkeypatch.setattr("create_graph.fetch_candidate_records", _stub_fetch)

    extractor = DummyExtractor()
    pipeline = KnowledgeGraphPipeline(extractor=extractor)
    config = PipelineConfig(
        query="alpha;beta",
        n_top_cited=TOP_TARGET,
        n_most_recent=RECENT_TARGET,
        fetch_buffer=3,
        checkpoint_interval=2,
        out_prefix="unit_test",
        data_dir=tmp_path,
    )

    pipeline.run(config)

    assert len(extractor.calls) == 2 * (TOP_TARGET + RECENT_TARGET)
    papers_df = _capture_tables["papers"]
    assert papers_df.shape[0] == 2 * (TOP_TARGET + RECENT_TARGET)


def test_auto_fetch_expands_until_targets_met(
    tmp_path: pathlib.Path, monkeypatch, _capture_tables
):
    partial_candidates = _flatten(
        [
            [
                _candidate_record(f"alpha_top_{i}", "alpha", "top", i, TOP_TARGET)
                for i in range(3)
            ],
            [
                _candidate_record(f"beta_top_{i}", "beta", "top", i, TOP_TARGET)
                for i in range(3)
            ],
            [
                _candidate_record(
                    f"alpha_recent_{i}", "alpha", "recent", i, RECENT_TARGET
                )
                for i in range(6)
            ],
            [
                _candidate_record(
                    f"beta_recent_{i}", "beta", "recent", i, RECENT_TARGET
                )
                for i in range(6)
            ],
        ]
    )

    full_candidates = _flatten(
        [_make_full_candidate_set("alpha"), _make_full_candidate_set("beta")]
    )

    def _fetch_with_buffer(query, filters, top_n, recent_n, fetch_buffer):
        assert query == "alpha;beta"
        assert filters
        assert top_n == TOP_TARGET
        assert recent_n == RECENT_TARGET
        if fetch_buffer <= 3:
            return partial_candidates
        return full_candidates

    buffer_calls: List[int] = []

    def _tracking_fetch(query, filters, top_n, recent_n, fetch_buffer):
        buffer_calls.append(fetch_buffer)
        return _fetch_with_buffer(query, filters, top_n, recent_n, fetch_buffer)

    monkeypatch.setattr("create_graph.fetch_candidate_records", _tracking_fetch)

    extractor = DummyExtractor()
    pipeline = KnowledgeGraphPipeline(extractor=extractor)
    config = PipelineConfig(
        query="alpha;beta",
        n_top_cited=TOP_TARGET,
        n_most_recent=RECENT_TARGET,
        fetch_buffer=3,
        checkpoint_interval=2,
        out_prefix="auto_buffer_test",
        data_dir=tmp_path,
    )

    pipeline.run(config)

    assert buffer_calls[0] == 3
    assert buffer_calls[-1] > buffer_calls[0]
    assert len(extractor.calls) == 2 * (TOP_TARGET + RECENT_TARGET)
    papers_df = _capture_tables["papers"]
    assert papers_df.shape[0] == 2 * (TOP_TARGET + RECENT_TARGET)


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__]))
