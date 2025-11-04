"""Hyperband search over rGCN-SCAE partition training using alignment metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

import align_partitions as align
from train_rgcn_scae import load_multiplex_graph, save_partition, train_scae_on_graph

# ---------------------------------------------------------------------------
# Search-space handling
# ---------------------------------------------------------------------------


class SearchSpaceError(ValueError):
    """Raised when the search space specification is invalid."""


class SearchSpace:
    """Finite or sampling-based hyperparameter space."""

    def __init__(self, specs: Mapping[str, Any]) -> None:
        if not specs:
            raise SearchSpaceError("Search space must contain at least one parameter.")
        self.specs: Dict[str, Mapping[str, Any]] = {}
        for name, raw_spec in specs.items():
            if isinstance(raw_spec, Mapping):
                spec = dict(raw_spec)
            else:
                spec = {"values": raw_spec}
            if "values" in spec:
                values = list(spec["values"])
                if not values:
                    raise SearchSpaceError(
                        f"Parameter '{name}' has no candidate values."
                    )
                spec["values"] = values
            elif spec.get("distribution") in {"uniform", "loguniform"}:
                for bound_key in ("min", "max"):
                    if bound_key not in spec:
                        raise SearchSpaceError(
                            f"Parameter '{name}' missing '{bound_key}' for distribution."
                        )
                min_val = float(spec["min"])
                max_val = float(spec["max"])
                if min_val >= max_val:
                    raise SearchSpaceError(
                        f"Parameter '{name}' requires min < max for distribution sampling."
                    )
                if spec["distribution"] == "loguniform" and (
                    min_val <= 0 or max_val <= 0
                ):
                    raise SearchSpaceError(
                        f"Parameter '{name}' uses loguniform but has non-positive bounds;"
                        " use a positive min/max or switch to 'uniform'."
                    )
            else:
                raise SearchSpaceError(
                    f"Parameter '{name}' must specify either 'values' or a supported distribution."
                )
            self.specs[name] = spec

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        sample: Dict[str, Any] = {}
        for name, spec in self.specs.items():
            if "values" in spec:
                sample[name] = _clone_value(rng.choice(spec["values"]))
            else:
                dist = spec["distribution"]
                if dist == "uniform":
                    sample[name] = rng.uniform(float(spec["min"]), float(spec["max"]))
                elif dist == "loguniform":
                    low = math.log(float(spec["min"]))
                    high = math.log(float(spec["max"]))
                    sample[name] = math.exp(rng.uniform(low, high))
                else:  # pragma: no cover - guarded by __init__
                    raise SearchSpaceError(
                        f"Unsupported distribution '{dist}' for '{name}'."
                    )
        return sample


def _clone_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [_clone_value(v) for v in list(value)]
    return value


# ---------------------------------------------------------------------------
# Alignment metrics aggregation
# ---------------------------------------------------------------------------


@dataclass
class AlignmentOutcome:
    metrics: Dict[str, Dict[str, float]]
    coverage: Dict[str, float]
    missing_nodes: List[str]


class AlignmentEvaluator:
    """Computes alignment metrics for a partition using HiTOP/RDoC mappings."""

    def __init__(
        self,
        graph_path: Path,
        prop_depth: int = 1,
        *,  # force keyword-only use of remaining arguments
        hitop_csv: Optional[Path] = None,
        hitop_id_col: str = "id",
        hitop_label_col: str = "label",
        rdoc_csv: Optional[Path] = None,
        rdoc_id_col: str = "id",
        rdoc_label_col: str = "label",
        hpo_terms_path: Optional[Path] = None,
    ) -> None:
        self.graph_path = graph_path
        self.graph = align.load_graph(graph_path)
        self.prop_depth = int(prop_depth)
        self.hitop_series: Optional[pd.Series] = None
        self.rdoc_series: Optional[pd.Series] = None

        if hitop_csv is None or rdoc_csv is None:
            inferred_hitop, inferred_rdoc = align.infer_framework_labels_tailored(
                self.graph, prop_depth=self.prop_depth, hpo_terms_path=hpo_terms_path
            )
            if hitop_csv is None:
                self.hitop_series = inferred_hitop
            if rdoc_csv is None:
                self.rdoc_series = inferred_rdoc

        if hitop_csv is not None:
            df = align.load_mapping_csv(
                hitop_csv, id_col=hitop_id_col, label_col=hitop_label_col
            )
            df = df[df[hitop_id_col].isin(self.graph.nodes)]
            self.hitop_series = df.set_index(hitop_id_col)[hitop_label_col]

        if rdoc_csv is not None:
            df = align.load_mapping_csv(
                rdoc_csv, id_col=rdoc_id_col, label_col=rdoc_label_col
            )
            df = df[df[rdoc_id_col].isin(self.graph.nodes)]
            self.rdoc_series = df.set_index(rdoc_id_col)[rdoc_label_col]

    def evaluate(self, node_to_cluster: Mapping[str, int]) -> AlignmentOutcome:
        aligned_map, missing_nodes = align.align_partition_to_graph(
            self.graph, node_to_cluster
        )
        if not aligned_map:
            raise ValueError(
                "No partition entries aligned to graph nodes; cannot compute metrics."
            )

        part_series = pd.Series(aligned_map, name="cluster", dtype="int64")

        metrics: Dict[str, Dict[str, float]] = {}

        if self.hitop_series is not None:
            hitop_metrics, *_ = align.metrics_against_labels(
                part_series, self.hitop_series
            )
            metrics["hitop"] = hitop_metrics
        else:
            metrics["hitop"] = {}

        if self.rdoc_series is not None:
            rdoc_metrics, *_ = align.metrics_against_labels(
                part_series, self.rdoc_series
            )
            metrics["rdoc"] = rdoc_metrics
        else:
            metrics["rdoc"] = {}

        coverage = {
            "graph_nodes": float(len(self.graph.nodes)),
            "partition_nodes": float(len(part_series)),
            "coverage_ratio": float(len(part_series) / len(self.graph.nodes))
            if self.graph.nodes
            else 0.0,
        }

        return AlignmentOutcome(
            metrics=metrics, coverage=coverage, missing_nodes=missing_nodes
        )


def compute_fitness(outcome: AlignmentOutcome, target: str) -> float:
    """Aggregate alignment metrics according to the requested fitness expression."""

    target = target.strip().lower()

    def _get(metric_key: str) -> Optional[float]:
        space, _, metric = metric_key.partition(".")
        if not metric:
            raise ValueError(
                "Fitness metric must be of the form 'space.metric' when not using a preset."
            )
        values = outcome.metrics.get(space)
        if not values:
            return None
        value = values.get(metric)
        if value is None or isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    if target in {"mean_nmi", "avg_nmi"}:
        values = [v for v in (_get("hitop.nmi"), _get("rdoc.nmi")) if v is not None]
        return sum(values) / len(values) if values else float("nan")
    if target in {"mean_v_measure", "avg_v"}:
        values = [
            v
            for v in (_get("hitop.v_measure"), _get("rdoc.v_measure"))
            if v is not None
        ]
        return sum(values) / len(values) if values else float("nan")
    if target in {"mean_all", "avg_all_metrics"}:
        metric_bounds = {
            "nmi": (0.0, 1.0),
            "ami": (0.0, 1.0),
            "ari": (-1.0, 1.0),
            "homogeneity": (0.0, 1.0),
            "completeness": (0.0, 1.0),
            "v_measure": (0.0, 1.0),
            "coverage_ratio": (0.0, 1.0),
        }
        normalized_values: List[float] = []
        for space_metrics in outcome.metrics.values():
            for metric_name, value in space_metrics.items():
                bounds = metric_bounds.get(metric_name)
                if bounds is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    continue
                lower, upper = bounds
                if upper <= lower:
                    continue
                normalized = (numeric - lower) / (upper - lower)
                normalized = min(1.0, max(0.0, normalized))
                normalized_values.append(normalized)

        if not normalized_values:
            return float("nan")
        return sum(normalized_values) / len(normalized_values)
    if target == "min_nmi":
        values = [v for v in (_get("hitop.nmi"), _get("rdoc.nmi")) if v is not None]
        return min(values) if values else float("nan")
    if target == "min_v_measure":
        values = [
            v
            for v in (_get("hitop.v_measure"), _get("rdoc.v_measure"))
            if v is not None
        ]
        return min(values) if values else float("nan")
    if "." in target:
        value = _get(target)
        if value is None:
            return float("nan")
        return value
    raise ValueError(f"Unsupported fitness target '{target}'.")


# ---------------------------------------------------------------------------
# Hyperband core
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    id: int
    config: Dict[str, Any]
    seed: int


@dataclass
class EvaluationResult:
    trial: Trial
    bracket: int
    rung: int
    resource: int
    fitness: float
    outcome: Optional[AlignmentOutcome]
    duration: float
    status: str = "ok"
    error: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


EvaluateFn = Callable[[Trial, int, int, int], EvaluationResult]


class HyperbandSearch:
    """Implements the synchronous Hyperband algorithm."""

    def __init__(
        self,
        *,
        sample_trial: Callable[[int], Trial],
        evaluate: EvaluateFn,
        max_resource: int,
        min_resource: int,
        eta: float,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if max_resource <= 0:
            raise ValueError("max_resource must be positive")
        if min_resource <= 0:
            raise ValueError("min_resource must be positive")
        if max_resource < min_resource:
            raise ValueError("max_resource must be >= min_resource")
        if eta <= 1.0:
            raise ValueError("eta must be > 1.0")

        self.sample_trial = sample_trial
        self.evaluate = evaluate
        self.max_resource = int(max_resource)
        self.min_resource = int(min_resource)
        self.eta = float(eta)
        self.max_evaluations = max_evaluations

    def run(self) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        eval_count = 0

        ratio = self.max_resource / self.min_resource
        if ratio <= 1:
            s_max = 0
        else:
            s_max = int(math.floor(math.log(ratio, self.eta)))
        B = (s_max + 1) * self.max_resource

        trial_counter = 0

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / self.max_resource / (s + 1) * (self.eta**s)))
            if n <= 0:
                continue

            remaining_evals = None
            if self.max_evaluations is not None:
                remaining_evals = self.max_evaluations - eval_count
                if remaining_evals <= 0:
                    break
                n = min(n, remaining_evals)
                if n <= 0:
                    break

            r = int(self.max_resource * (self.eta ** (-s)))
            r = max(r, self.min_resource)

            trials: List[Trial] = []
            for _ in range(n):
                trials.append(self.sample_trial(trial_counter))
                trial_counter += 1

            for i in range(s + 1):
                n_i = int(math.floor(n * (self.eta ** (-i))))
                if n_i <= 0:
                    break

                if (
                    self.max_evaluations is not None
                    and eval_count >= self.max_evaluations
                ):
                    break

                current_trials = trials[:n_i]
                if not current_trials:
                    break

                r_i = int(r * (self.eta**i))
                r_i = max(r_i, self.min_resource)

                rung_results: List[EvaluationResult] = []
                for trial in current_trials:
                    if (
                        self.max_evaluations is not None
                        and eval_count >= self.max_evaluations
                    ):
                        break
                    start = time.time()
                    result = self.evaluate(trial, r_i, i, s)
                    duration = time.time() - start
                    result.duration = duration
                    rung_results.append(result)
                    results.append(result)
                    eval_count += 1

                rung_results.sort(key=lambda res: res.fitness, reverse=True)

                if i < s:
                    next_n = int(math.floor(n * (self.eta ** (-(i + 1)))))
                    next_n = max(1, min(next_n, len(rung_results)))
                    trials = [res.trial for res in rung_results[:next_n]]

        return results


# ---------------------------------------------------------------------------
# Partition trainer + evaluator bridge
# ---------------------------------------------------------------------------


class PartitionTrainerEvaluator:
    """Runs partition training and alignment evaluation for a given trial."""

    def __init__(
        self,
        graphml_path: Path,
        alignment_evaluator: AlignmentEvaluator,
        base_train_params: Optional[Mapping[str, Any]],
        fitness_metric: str,
        *,
        device: Optional[str] = None,
        best_partition_path: Optional[Path] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.graphml_path = graphml_path
        self.multiplex_graph = load_multiplex_graph(graphml_path)
        self.alignment_evaluator = alignment_evaluator
        self.device = device
        self.fitness_metric = fitness_metric
        self.base_train_params = dict(base_train_params or {})
        self.base_min_epochs = int(self.base_train_params.pop("min_epochs", 0))
        self.base_calibration_epochs = int(
            self.base_train_params.pop("calibration_epochs", 0)
        )
        self.best_partition_path = best_partition_path
        self.best_result: Optional[EvaluationResult] = None
        self.result_rng = rng or random.Random()

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on environment
            torch.cuda.manual_seed_all(seed)

    def evaluate(
        self, trial: Trial, resource: int, rung: int, bracket: int
    ) -> EvaluationResult:
        seed = trial.seed + rung * 1009 + bracket * 65537
        self._set_seed(seed)

        train_kwargs = dict(self.base_train_params)
        train_kwargs.update(trial.config)

        if self.device is not None:
            train_kwargs.setdefault("device", self.device)

        train_kwargs.setdefault("grad_accum_steps", 1)
        train_kwargs.setdefault("cache_node_attributes", True)
        train_kwargs.setdefault("mixed_precision", False)

        if "cluster_stability_tol" in train_kwargs:
            train_kwargs.setdefault(
                "cluster_stability_tolerance", train_kwargs.pop("cluster_stability_tol")
            )
        if "cluster_stability_rel_tol" in train_kwargs:
            train_kwargs.setdefault(
                "cluster_stability_relative_tolerance",
                train_kwargs.pop("cluster_stability_rel_tol"),
            )

        min_epochs_override = int(train_kwargs.pop("min_epochs", self.base_min_epochs))
        calibration_override = int(
            train_kwargs.pop("calibration_epochs", self.base_calibration_epochs)
        )

        effective_min_epochs = min(resource, max(0, min_epochs_override))
        calibration_epochs = min(resource, max(0, calibration_override))

        status = "ok"
        error: Optional[str] = None
        outcome: Optional[AlignmentOutcome] = None
        fitness = float("-inf")
        info: Dict[str, Any] = {
            "seed": seed,
            "effective_min_epochs": effective_min_epochs,
            "calibration_epochs": calibration_epochs,
        }

        start_time = time.time()
        try:
            model, partition, history, summary = train_scae_on_graph(
                self.multiplex_graph,
                max_epochs=int(resource),
                min_epochs=int(effective_min_epochs),
                calibration_epochs=int(calibration_epochs),
                **train_kwargs,
            )
            del model

            node_ids = self.multiplex_graph.node_ids
            assignments = partition.node_to_cluster.cpu().tolist()
            node_to_cluster = {
                node_ids[i]: int(assignments[i]) for i in range(len(node_ids))
            }

            outcome = self.alignment_evaluator.evaluate(node_to_cluster)
            fitness = compute_fitness(outcome, self.fitness_metric)

            info.update(
                {
                    "training_summary": summary,
                    "history_tail": history[-1] if history else {},
                    "history_length": len(history),
                }
            )

            if (
                isinstance(fitness, float) and math.isnan(fitness)
            ) or not math.isfinite(fitness):
                status = "invalid"
                error = "Fitness resolved to a non-finite value."
                fitness = float("-inf")

            if (
                outcome is not None
                and status == "ok"
                and math.isfinite(fitness)
                and (self.best_result is None or fitness > self.best_result.fitness)
            ):
                if self.best_partition_path is not None:
                    save_partition(
                        partition, self.best_partition_path, self.multiplex_graph
                    )
                self.best_result = EvaluationResult(
                    trial=trial,
                    bracket=bracket,
                    rung=rung,
                    resource=resource,
                    fitness=fitness,
                    outcome=outcome,
                    duration=time.time() - start_time,
                    status=status,
                    error=error,
                    info=info,
                )

        except Exception as exc:  # pragma: no cover - exercised in failure paths
            status = "error"
            error = str(exc)
            fitness = float("-inf")

        info["status"] = status

        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.empty_cache()

        duration = time.time() - start_time
        return EvaluationResult(
            trial=trial,
            bracket=bracket,
            rung=rung,
            resource=resource,
            fitness=fitness,
            outcome=outcome,
            duration=duration,
            status=status,
            error=error,
            info=info,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return _jsonify(vars(obj))
    return str(obj)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_train_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in overrides:
        candidate = item.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Training override file '{candidate}' is not valid JSON: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Training override file '{candidate}' must contain a JSON object."
                )
            params.update(payload)
            continue
        if "=" not in candidate:
            raise ValueError(
                f"Training override '{item}' must be KEY=VALUE or a path to a JSON file"
            )
        key, value = candidate.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        params[key] = parsed
    return params


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperband search for rGCN-SCAE partition training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graphml", type=Path, help="GraphML file used for training")
    parser.add_argument(
        "--search-space",
        type=Path,
        required=True,
        help="JSON file describing the hyperparameter search space",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("hyperband_runs"),
        help="Directory where results will be written",
    )
    parser.add_argument(
        "--fitness",
        type=str,
        default="mean_nmi",
        help="Fitness metric (mean_nmi, mean_v_measure, min_nmi, hitop.nmi, etc.)",
    )
    parser.add_argument("--max-resource", type=int, default=120, help="Max epochs")
    parser.add_argument(
        "--min-resource", type=int, default=20, help="Minimum epochs per evaluation"
    )
    parser.add_argument(
        "--eta", type=float, default=3.0, help="Hyperband reduction factor"
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Optional cap on total training evaluations",
    )
    parser.add_argument(
        "--prop-depth",
        type=int,
        default=1,
        help="Propagation depth used when inferring HiTOP/RDoC labels",
    )
    parser.add_argument(
        "--hpo-terms",
        type=Path,
        default=None,
        help="Optional HPO terms CSV to speed alignment inference",
    )
    parser.add_argument("--hitop-map", type=Path, default=None)
    parser.add_argument("--hitop-id-col", type=str, default="id")
    parser.add_argument("--hitop-label-col", type=str, default="label")
    parser.add_argument("--rdoc-map", type=Path, default=None)
    parser.add_argument("--rdoc-id-col", type=str, default="id")
    parser.add_argument("--rdoc-label-col", type=str, default="label")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Optional JSON file containing base training arguments",
    )
    parser.add_argument(
        "--train-param",
        action="append",
        default=[],
        help="Override training arguments with KEY=VALUE entries or append a JSON file path",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="torch device override"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed controlling hyperparameter sampling",
    )
    parser.add_argument(
        "--best-partition",
        type=Path,
        default=None,
        help="Path to write the best partition JSON (defaults to outdir/best_partition.json)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    search_space = SearchSpace(_load_json(args.search_space))

    base_train_params: Dict[str, Any] = {}
    if args.train_config is not None:
        base_train_params.update(_load_json(args.train_config))
    if args.train_param:
        base_train_params.update(_parse_train_overrides(args.train_param))

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    best_partition_path = args.best_partition
    if best_partition_path is None:
        best_partition_path = outdir / "best_partition.json"

    alignment_evaluator = AlignmentEvaluator(
        args.graphml,
        prop_depth=args.prop_depth,
        hitop_csv=args.hitop_map,
        hitop_id_col=args.hitop_id_col,
        hitop_label_col=args.hitop_label_col,
        rdoc_csv=args.rdoc_map,
        rdoc_id_col=args.rdoc_id_col,
        rdoc_label_col=args.rdoc_label_col,
        hpo_terms_path=args.hpo_terms,
    )

    trainer_evaluator = PartitionTrainerEvaluator(
        args.graphml,
        alignment_evaluator,
        base_train_params,
        args.fitness,
        device=args.device,
        best_partition_path=best_partition_path,
        rng=rng,
    )

    def _sample_trial(trial_id: int) -> Trial:
        config = search_space.sample(rng)
        seed = rng.randint(0, 2**31 - 1)
        return Trial(id=trial_id, config=config, seed=seed)

    def _evaluate(
        trial: Trial, resource: int, rung: int, bracket: int
    ) -> EvaluationResult:
        return trainer_evaluator.evaluate(trial, resource, rung, bracket)

    search = HyperbandSearch(
        sample_trial=_sample_trial,
        evaluate=_evaluate,
        max_resource=args.max_resource,
        min_resource=args.min_resource,
        eta=args.eta,
        max_evaluations=args.max_evals,
    )

    results = search.run()

    results_payload = [_jsonify(res) for res in results]
    (outdir / "hyperband_results.json").write_text(
        json.dumps(results_payload, indent=2), encoding="utf-8"
    )

    best = trainer_evaluator.best_result
    if best is not None:
        best_payload = _jsonify(best)
        (outdir / "best_result.json").write_text(
            json.dumps(best_payload, indent=2), encoding="utf-8"
        )
    else:
        (outdir / "best_result.json").write_text(
            json.dumps({"status": "no_valid_result"}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
