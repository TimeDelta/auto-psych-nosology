from __future__ import annotations

import argparse
import logging
import os
import pathlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

from checkpoint_utils import (
    CHECKPOINT_ARG_KEYS,
    CheckpointState,
    checkpoint_extraction_id,
    checkpoint_record_id,
    load_checkpoint_state,
    normalize_for_checkpoint,
    save_checkpoint_state,
)
from fulltext_loader import resolve_text_and_download
from graph_build import (
    accum_extractions,
    build_multilayer_graph,
    project_to_weighted_graph,
)
from graph_export import save_tables
from models import PaperExtraction
from nlp_extraction import EntityRelationExtractor, ExtractionConfig
from openalex_client import DEFAULT_FILTER, fetch_candidate_records
from sections import extract_results_and_discussion

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _log_step_duration(step: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    logger.debug("[timing] %s completed in %.3f seconds", step, elapsed)


def _ensure_spacy_model_available(model_name: Optional[str]) -> None:
    if not model_name:
        return
    try:
        import spacy
    except Exception:  # pragma: no cover
        logger.debug("spaCy not installed; skipping automatic model setup.")
        return

    try:
        spacy.load(model_name)
        return
    except Exception:
        pass

    scispacy_urls = {
        "en_core_sci_sm": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz",
        "en_core_sci_scibert": "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz",
    }
    url = scispacy_urls.get(model_name)
    if not url:
        logger.debug(
            "No known download URL for spaCy model %s; skipping auto-install.",
            model_name,
        )
        return

    cmd = ["python3.10", "-m", "pip", "install", url]
    try:
        logger.info("spaCy model %s not found; attempting pip install.", model_name)
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "pip install for %s failed (code %s): %s",
                model_name,
                result.returncode,
                result.stderr.strip(),
            )
            return
        spacy.load(model_name)
        logger.info("Successfully installed spaCy model %s via pip.", model_name)
    except Exception as exc:
        logger.warning(
            "Automatic install of spaCy model %s failed (%s); proceeding without it.",
            model_name,
            exc,
        )


@dataclass
class PipelineConfig:
    query: Sequence[str] | str
    filters: str = "".join(DEFAULT_FILTER)
    out_prefix: str = "psych_kg_results_only"
    n_top_cited: int = 250
    n_most_recent: int = 250
    fetch_buffer: int = 5
    project_to_weighted: bool = False
    checkpoint_path: Optional[pathlib.Path] = None
    checkpoint_interval: int = 25
    resume: bool = False
    eval_nodes: Optional[pathlib.Path] = None
    eval_relations: Optional[pathlib.Path] = None
    data_dir: pathlib.Path = pathlib.Path("data")


class KnowledgeGraphPipeline:
    def __init__(self, extractor: Optional[EntityRelationExtractor] = None) -> None:
        if extractor is not None:
            self.extractor = extractor
        else:
            extractor_config = ExtractionConfig()
            _ensure_spacy_model_available(extractor_config.spacy_model)
            self.extractor = EntityRelationExtractor(config=extractor_config)

    def _load_gold_nodes(self, path: pathlib.Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _load_gold_relations(self, path: pathlib.Path) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
        return triples

    def _evaluate_nodes(
        self, nodes_df: pd.DataFrame, gold_nodes: Iterable[str]
    ) -> Dict[str, float]:
        predicted = set(nodes_df["canonical_name"].tolist())
        gold = {node.strip() for node in gold_nodes}
        if not predicted and not gold:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def _evaluate_relations(
        self, graph: nx.MultiDiGraph, gold_relations: Iterable[Tuple[str, str, str]]
    ) -> Dict[str, float]:
        gold_set = {
            (subj.strip(), pred, obj.strip()) for subj, pred, obj in gold_relations
        }
        predicted: set[Tuple[str, str, str]] = set()
        for u, v, data in graph.edges(data=True):
            predicted.add((u.strip(), data.get("predicate"), v.strip()))
        if not predicted and not gold_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        tp = len(predicted & gold_set)
        fp = len(predicted - gold_set)
        fn = len(gold_set - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def run(self, config: PipelineConfig) -> None:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            config.checkpoint_path
            if config.checkpoint_path is not None
            else config.data_dir / f"{config.out_prefix}_checkpoint.json"
        )
        resume_state = load_checkpoint_state(checkpoint_path) if config.resume else None
        if config.resume and resume_state is None:
            print(f"[warn] No checkpoint found at {checkpoint_path}; starting fresh.")

        paper_level: List[PaperExtraction] = (
            list(resume_state.extractions) if resume_state else []
        )
        completed_ids = set(resume_state.completed_ids) if resume_state else set()
        records_from_checkpoint = resume_state.records if resume_state else None
        if paper_level:
            completed_ids.update(
                checkpoint_extraction_id(extraction) for extraction in paper_level
            )
        if resume_state:
            print(f"[info] Loaded {len(paper_level)} extracted papers from checkpoint.")

        checkpoint_metadata = (
            dict(resume_state.metadata)
            if resume_state and resume_state.metadata
            else {}
        )
        checkpoint_metadata.setdefault(
            "created_at", datetime.utcnow().isoformat() + "Z"
        )
        checkpoint_metadata["version"] = checkpoint_metadata.get("version", 1)
        args_snapshot = {
            key: normalize_for_checkpoint(getattr(config, key, None))
            for key in CHECKPOINT_ARG_KEYS
        }
        checkpoint_metadata["args_snapshot"] = args_snapshot
        checkpoint_metadata["checkpoint_interval"] = config.checkpoint_interval

        if resume_state and resume_state.metadata:
            previous = resume_state.metadata.get("args_snapshot")
            if previous:
                mismatched = [
                    key
                    for key in CHECKPOINT_ARG_KEYS
                    if normalize_for_checkpoint(previous.get(key))
                    != args_snapshot.get(key)
                ]
                if mismatched:
                    print(
                        "[warn] Checkpoint parameters differ for: "
                        + ", ".join(sorted(mismatched))
                        + ". Proceeding with current arguments."
                    )

        checkpoint_interval = max(0, config.checkpoint_interval)
        periodic_enabled = checkpoint_interval > 0

        LEGACY_BUCKET = "__legacy__"
        default_targets = {
            "top": max(config.n_top_cited, 0),
            "recent": max(config.n_most_recent, 0),
        }
        BUCKET_PRIORITY = {"top": 0, "recent": 1}

        def _bucket_priority(bucket: str) -> int:
            return BUCKET_PRIORITY.get(bucket, 2)

        def _source_sort_key(source: Dict[str, Any]) -> Tuple[int, int, int, str]:
            bucket = str(source.get("bucket") or LEGACY_BUCKET)
            is_buffer = 1 if source.get("is_buffer") else 0
            rank_raw = source.get("rank")
            rank = rank_raw if isinstance(rank_raw, int) else 1_000_000
            term = str(source.get("term", "") or "")
            return (is_buffer, _bucket_priority(bucket), rank, term)

        def _selection_priority(
            item: Tuple[Tuple[str, str], Dict[str, Any], int]
        ) -> Tuple[int, int, int, int]:
            key, source, remaining = item
            bucket_order = _bucket_priority(key[1])
            is_buffer = 1 if source.get("is_buffer") else 0
            rank_raw = source.get("rank")
            rank = rank_raw if isinstance(rank_raw, int) else 1_000_000
            return (-remaining, bucket_order, is_buffer, rank)

        def _select_needed_keys_from_sources(
            sources: Sequence[Dict[str, Any]],
            target_map_local: Dict[Tuple[str, str], int],
            success_counts_ref: Dict[Tuple[str, str], int],
        ) -> set[Tuple[str, str]]:
            candidates: List[Tuple[Tuple[str, str], Dict[str, Any], int]] = []
            for source in sorted(list(sources), key=_source_sort_key):
                bucket = str(source.get("bucket") or LEGACY_BUCKET)
                term = str(source.get("term", "") or "")
                key = (term, bucket)
                target = target_map_local.get(key)
                if target is None or target <= 0:
                    continue
                remaining = target - success_counts_ref.get(key, 0)
                if remaining <= 0:
                    continue
                candidates.append((key, source, remaining))
            if not candidates:
                return set()
            selected_key, _, _ = min(candidates, key=_selection_priority)
            return {selected_key}

        def _preview_target_map(
            records_seq: Sequence[Dict[str, Any]]
        ) -> Dict[Tuple[str, str], int]:
            preview: Dict[Tuple[str, str], int] = {}
            for rec in records_seq:
                sources = rec.get("_candidate_sources") or []
                for source in sources:
                    bucket = source.get("bucket")
                    if bucket not in default_targets:
                        continue
                    target = default_targets[bucket]
                    if target <= 0:
                        continue
                    term = str(source.get("term", "") or "")
                    preview.setdefault((term, bucket), target)
            if not preview:
                legacy_target = default_targets["top"] + default_targets["recent"]
                if legacy_target > 0:
                    preview[("", LEGACY_BUCKET)] = legacy_target
            return preview

        def _simulate_assignable(
            records_seq: Sequence[Dict[str, Any]],
            target_map_local: Dict[Tuple[str, str], int],
        ) -> int:
            if not target_map_local:
                return 0
            simulated_counts: Dict[Tuple[str, str], int] = {
                key: 0 for key in target_map_local
            }
            simulated_total = 0
            for rec in records_seq:
                sources = rec.get("_candidate_sources") or []
                if not sources and ("", LEGACY_BUCKET) in target_map_local:
                    sources = [
                        {"term": "", "bucket": LEGACY_BUCKET, "is_buffer": False}
                    ]
                assignments = _select_needed_keys_from_sources(
                    sources, target_map_local, simulated_counts
                )
                if not assignments:
                    continue
                for key in assignments:
                    simulated_counts[key] = simulated_counts.get(key, 0) + 1
                    simulated_total += 1
            return simulated_total

        effective_fetch_buffer = max(config.fetch_buffer, 0)

        if records_from_checkpoint is not None:
            records = records_from_checkpoint
            print(
                f"[info] Using {len(records)} candidate papers loaded from checkpoint."
            )
        else:
            print("[info] Fetching OpenAlex (top-cited and most-recent)")
            current_fetch_buffer = effective_fetch_buffer
            auto_fetch_limit = max(20, (current_fetch_buffer or 1) * 4)
            previous_possible = -1

            while True:
                records = fetch_candidate_records(
                    config.query,
                    config.filters,
                    config.n_top_cited,
                    config.n_most_recent,
                    fetch_buffer=current_fetch_buffer,
                )
                print(
                    "[info] Candidate papers: "
                    + f"{len(records)} (unique across both buckets; fetch_buffer={current_fetch_buffer})"
                )
                preview_target_map = _preview_target_map(records)
                total_preview_required = sum(preview_target_map.values())
                possible_preview = (
                    _simulate_assignable(records, preview_target_map)
                    if total_preview_required > 0
                    else 0
                )
                if (
                    total_preview_required <= 0
                    or possible_preview >= total_preview_required
                ):
                    effective_fetch_buffer = current_fetch_buffer
                    break
                if possible_preview <= previous_possible:
                    print(
                        "[warn] Additional fetch attempts are not increasing "
                        "unique coverage; proceeding with current candidates."
                    )
                    effective_fetch_buffer = current_fetch_buffer
                    break
                if current_fetch_buffer >= auto_fetch_limit:
                    print(
                        f"[warn] Reached automatic fetch-buffer cap ({auto_fetch_limit}); "
                        f"proceeding with {possible_preview}/{total_preview_required} "
                        "fillable papers."
                    )
                    effective_fetch_buffer = current_fetch_buffer
                    break
                previous_possible = possible_preview
                next_fetch_buffer = max(
                    current_fetch_buffer * 2, current_fetch_buffer + 1, 1
                )
                if next_fetch_buffer > auto_fetch_limit:
                    next_fetch_buffer = auto_fetch_limit
                if next_fetch_buffer == current_fetch_buffer:
                    effective_fetch_buffer = current_fetch_buffer
                    break
                print(
                    f"[info] Only {possible_preview} of {total_preview_required} slots "
                    f"covered; increasing fetch buffer to {next_fetch_buffer}."
                )
                current_fetch_buffer = next_fetch_buffer

        checkpoint_metadata["effective_fetch_buffer"] = effective_fetch_buffer

        record_lookup: Dict[str, Dict[str, Any]] = {}
        target_map: Dict[Tuple[str, str], int] = {}

        for rec in records:
            rec_id = checkpoint_record_id(rec)
            if rec_id:
                record_lookup[rec_id] = rec
            sources = rec.get("_candidate_sources") or []
            for source in sources:
                bucket = source.get("bucket")
                if bucket not in default_targets:
                    continue
                target = default_targets[bucket]
                if target <= 0:
                    continue
                term = str(source.get("term", "") or "")
                target_map.setdefault((term, bucket), target)

        if not target_map:
            legacy_target = default_targets["top"] + default_targets["recent"]
            if legacy_target > 0:
                target_map[("", LEGACY_BUCKET)] = legacy_target

        success_counts: Dict[Tuple[str, str], int] = {key: 0 for key in target_map}
        successful_total = 0
        total_required = sum(target_map.values())

        def _bump_success(key: Tuple[str, str]) -> None:
            nonlocal successful_total
            target = target_map.get(key)
            if target is None or target <= 0:
                return
            current = success_counts.get(key, 0)
            if current >= target:
                return
            success_counts[key] = current + 1
            successful_total += 1

        if paper_level:
            for extraction in paper_level:
                extraction_key = checkpoint_record_id(
                    {
                        "id": extraction.paper_id,
                        "doi": extraction.doi,
                        "title": extraction.title,
                    }
                )
                record_meta = record_lookup.get(extraction_key)
                if record_meta is None:
                    continue
                sources = record_meta.get("_candidate_sources") or []
                if not sources and ("", LEGACY_BUCKET) in target_map:
                    sources = record_meta.setdefault(
                        "_candidate_sources",
                        [{"term": "", "bucket": LEGACY_BUCKET, "is_buffer": False}],
                    )
                assignments = _select_needed_keys_from_sources(
                    sources, target_map, success_counts
                )
                for key in assignments:
                    _bump_success(key)

        def persist_checkpoint(status: Optional[str] = None) -> None:
            metadata = dict(checkpoint_metadata)
            if status is not None:
                metadata["status"] = status
            save_checkpoint_state(
                checkpoint_path,
                CheckpointState(
                    records=list(records),
                    extractions=list(paper_level),
                    completed_ids=set(completed_ids),
                    metadata=metadata,
                ),
            )

        if periodic_enabled or resume_state:
            persist_checkpoint(status="resumed" if resume_state else "initialized")

        print(
            "[info] Downloading full texts and extracting (RESULTS and DISCUSSION only)"
        )
        attempted_since_checkpoint = 0
        processed_new = 0
        skipped_existing = 0

        remaining_required = max(total_required - successful_total, 0)
        progress_bar: Optional[tqdm] = None
        if remaining_required > 0:
            progress_bar = tqdm(total=remaining_required, desc="Processing papers")

        for record in records:
            if total_required > 0 and successful_total >= total_required:
                break
            record_id = checkpoint_record_id(record)
            if record_id in completed_ids:
                skipped_existing += 1
                continue
            success_before = successful_total
            sources = record.get("_candidate_sources") or []
            if not sources and ("", LEGACY_BUCKET) in target_map:
                sources = record.setdefault(
                    "_candidate_sources",
                    [{"term": "", "bucket": LEGACY_BUCKET, "is_buffer": False}],
                )
            needed_keys = _select_needed_keys_from_sources(
                sources, target_map, success_counts
            )
            if not needed_keys:
                continue
            meta = {
                "id": record.get("id"),
                "doi": record.get("doi"),
                "title": record.get("title"),
                "year": record.get("year"),
                "venue": record.get("venue"),
            }
            try:
                fulltext = resolve_text_and_download(record)
                study_sections = (
                    extract_results_and_discussion(fulltext or "") if fulltext else None
                )
                text_for_ie = (
                    study_sections
                    if study_sections
                    else (record.get("abstract", "") or "")
                )
                extraction = self.extractor.extract(meta, text_for_ie)
                if extraction:
                    paper_level.append(extraction)
                    processed_new += 1
                    for key in needed_keys:
                        _bump_success(key)
                    if progress_bar is not None:
                        delta = successful_total - success_before
                        if delta > 0:
                            progress_bar.update(delta)
            except KeyboardInterrupt:
                print("[warn] Interrupted by user; saving checkpoint before exit.")
                persist_checkpoint(status="interrupted")
                raise
            finally:
                completed_ids.add(record_id)
                attempted_since_checkpoint += 1
                if (
                    periodic_enabled
                    and attempted_since_checkpoint >= checkpoint_interval
                ):
                    persist_checkpoint(status="running")
                    attempted_since_checkpoint = 0

        if progress_bar is not None:
            progress_bar.close()

        if periodic_enabled and attempted_since_checkpoint > 0:
            persist_checkpoint(status="running")

        unmet_targets = {
            key: target_map[key] - success_counts.get(key, 0)
            for key in target_map
            if success_counts.get(key, 0) < target_map[key]
        }
        if unmet_targets:
            details = ", ".join(
                f"{term or '[default]'}:{bucket} -> {remaining}"
                for (term, bucket), remaining in sorted(unmet_targets.items())
            )
            print(
                "[warn] Unable to satisfy all candidate slots; consider increasing "
                f"--fetch-buffer. Unfilled targets: {details}"
            )

        if skipped_existing:
            print(
                f"[info] Skipped {skipped_existing} papers already recorded in checkpoint."
            )
        if processed_new:
            print(f"[info] Processed {processed_new} new papers in this run.")

        if not paper_level:
            persist_checkpoint(status="empty")
            print("[warn] No extractions; exiting.")
            return

        start_time = time.perf_counter()
        nodes_df, relations_df, papers_df = accum_extractions(paper_level)
        _log_step_duration("accum_extractions", start_time)
        print(
            f"[info] {len(papers_df)} papers, {len(nodes_df)} node mentions, {len(relations_df)} relations"
        )
        start_time = time.perf_counter()
        graph = build_multilayer_graph(nodes_df, relations_df)
        _log_step_duration("build_multilayer_graph", start_time)
        weighted_graph: Optional[nx.Graph] = None
        if config.project_to_weighted:
            start_time = time.perf_counter()
            weighted_graph = project_to_weighted_graph(graph)
            _log_step_duration("project_to_weighted_graph", start_time)
        else:
            logger.debug("[timing] project_to_weighted_graph skipped (flag disabled)")

        output_prefix = config.data_dir / config.out_prefix
        start_time = time.perf_counter()
        save_tables(
            nodes_df,
            relations_df,
            papers_df,
            graph,
            output_prefix,
            projected=weighted_graph,
        )
        _log_step_duration("save_tables", start_time)
        persist_checkpoint(status="complete")

        if config.eval_nodes:
            gold_nodes = self._load_gold_nodes(config.eval_nodes)
            metrics = self._evaluate_nodes(nodes_df, gold_nodes)
            print(
                f"[eval-nodes] precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
            )
        if config.eval_relations:
            gold_relations = self._load_gold_relations(config.eval_relations)
            metrics = self._evaluate_relations(graph, gold_relations)
            print(
                f"[eval-relations] precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
            )
        print(f"[done] Saved under {output_prefix}.*")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="OpenAlex search query (title/abstract). Use ';' to provide multiple terms",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default="".join(DEFAULT_FILTER),
        help="Additional OpenAlex filters",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="psych_kg_results_only",
        help="Prefix for output files saved under the data/ directory",
    )
    parser.add_argument(
        "--n-top-cited",
        type=int,
        default=250,
        help="Number of top-cited OA papers to fetch",
    )
    parser.add_argument(
        "--n-most-recent",
        type=int,
        default=250,
        help="Number of most-recent OA papers to fetch",
    )
    parser.add_argument(
        "--fetch-buffer",
        type=int,
        default=5,
        help="Extra works to over-fetch per bucket to compensate for duplicates",
    )
    parser.add_argument(
        "--eval-nodes",
        type=str,
        default=None,
        help="Path to a gold standard node list (one canonical name per line)",
    )
    parser.add_argument(
        "--eval-relations",
        type=str,
        default=None,
        help="Path to a gold standard relation file (tab separated subject, predicate, object)",
    )
    parser.add_argument(
        "--project-to-weighted",
        action="store_true",
        help="If set, project the multigraph to a simple weighted graph and save it",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to persist incremental extraction checkpoints. Defaults to data/<out_prefix>_checkpoint.json.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Save a checkpoint after this many new papers are attempted. Set to 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint if available.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where outputs and checkpoints should be stored",
    )
    return parser


def _configure_logging() -> None:
    level_name = os.getenv("AUTO_PSYCH_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root.setLevel(level)
    logger.setLevel(level)
    logger.debug("Logging configured at %s", level_name)


if __name__ == "__main__":
    _configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        query=args.query,
        filters=args.filters,
        out_prefix=args.out_prefix,
        n_top_cited=args.n_top_cited,
        n_most_recent=args.n_most_recent,
        fetch_buffer=args.fetch_buffer,
        project_to_weighted=args.project_to_weighted,
        checkpoint_path=pathlib.Path(args.checkpoint_path)
        if args.checkpoint_path
        else None,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        eval_nodes=pathlib.Path(args.eval_nodes) if args.eval_nodes else None,
        eval_relations=pathlib.Path(args.eval_relations)
        if args.eval_relations
        else None,
        data_dir=pathlib.Path(args.data_dir),
    )

    pipeline = KnowledgeGraphPipeline()
    pipeline.run(config)
