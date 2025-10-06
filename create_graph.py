from __future__ import annotations

import argparse
import pathlib
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
from nlp_extraction import EntityRelationExtractor
from openalex_client import DEFAULT_FILTER, fetch_candidate_records
from sections import extract_results_and_discussion


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
        self.extractor = extractor or EntityRelationExtractor()

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

        if records_from_checkpoint is not None:
            records = records_from_checkpoint
            print(
                f"[info] Using {len(records)} candidate papers loaded from checkpoint."
            )
        else:
            print("[info] Fetching OpenAlex (top-cited and most-recent)")
            records = fetch_candidate_records(
                config.query,
                config.filters,
                config.n_top_cited,
                config.n_most_recent,
                fetch_buffer=config.fetch_buffer,
            )
            print(
                f"[info] Candidate papers: {len(records)} (unique across both buckets)"
            )

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

        for record in tqdm(records):
            record_id = checkpoint_record_id(record)
            if record_id in completed_ids:
                skipped_existing += 1
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
            except KeyboardInterrupt:
                print("[warn] Interrupted by user; saving checkpoint before exit.")
                persist_checkpoint(status="interrupted")
                raise
            except Exception as exc:
                label = meta.get("title") or meta.get("id") or record_id
                print(
                    f"[error] Unexpected failure while processing '{label}': {exc}. Skipping."
                )
            finally:
                completed_ids.add(record_id)
                attempted_since_checkpoint += 1
                if (
                    periodic_enabled
                    and attempted_since_checkpoint >= checkpoint_interval
                ):
                    persist_checkpoint(status="running")
                    attempted_since_checkpoint = 0

        if periodic_enabled and attempted_since_checkpoint > 0:
            persist_checkpoint(status="running")

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

        nodes_df, relations_df, papers_df = accum_extractions(paper_level)
        print(
            f"[info] {len(papers_df)} papers, {len(nodes_df)} node mentions, {len(relations_df)} relations"
        )
        graph = build_multilayer_graph(nodes_df, relations_df)
        weighted_graph: Optional[nx.Graph] = None
        if config.project_to_weighted:
            weighted_graph = project_to_weighted_graph(graph)

        output_prefix = config.data_dir / config.out_prefix
        save_tables(
            nodes_df,
            relations_df,
            papers_df,
            graph,
            output_prefix,
            projected=weighted_graph,
        )
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


if __name__ == "__main__":
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
